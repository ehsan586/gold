"""Position management: profit protection, break-even, partial close, trailing stop,
and the anti-noise reversal policy. Produces ACTIONS; only the ExecutionEngine acts."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict

from .data import Position, SymbolSpec
from .risk import position_risk
from .settings import Settings


@dataclass
class ManagedState:
    ticket: int
    direction: str
    entry: float
    sl0: float
    initial_risk: float          # money
    peak_profit: float = 0.0     # max unrealized profit (money)
    worst_profit: float = 0.0    # max unrealized loss (negative money) -> MAE
    armed: bool = False          # profit protection armed
    be_done: bool = False
    partial_done: bool = False


@dataclass
class Action:
    kind: str                    # CLOSE / MODIFY_SL / PARTIAL
    reason: str
    price: float | None = None
    volume: float | None = None


class PositionManager:
    def __init__(self, settings: Settings, db):
        self.s, self.db = settings, db
        self.states: dict[int, ManagedState] = {}

    # ---- persistence so peaks survive restarts ------------------------------
    def _save(self, st: ManagedState) -> None:
        self.db.kv_set(f"pos:{st.ticket}", json.dumps(asdict(st)))

    def load(self, ticket: int) -> ManagedState | None:
        if ticket in self.states:
            return self.states[ticket]
        raw = self.db.kv_get(f"pos:{ticket}")
        if raw:
            self.states[ticket] = ManagedState(**json.loads(raw))
            return self.states[ticket]
        return None

    def register(self, p: Position, spec: SymbolSpec) -> ManagedState:
        risk = position_risk(p, spec)
        st = ManagedState(p.ticket, p.direction, p.price_open, p.sl, 0.0 if math.isinf(risk) else risk)
        self.states[p.ticket] = st
        self._save(st)
        return st

    def forget(self, ticket: int) -> ManagedState | None:
        st = self.states.pop(ticket, None) or self.load(ticket)
        self.states.pop(ticket, None)
        self.db.kv_delete(f"pos:{ticket}")
        return st

    # ---- main evaluation ------------------------------------------------------
    def evaluate(self, p: Position, spec: SymbolSpec, bid: float, ask: float, atr_v: float) -> list[Action]:
        s = self.s
        st = self.load(p.ticket) or self.register(p, spec)
        changed = False
        profit = p.profit
        if profit > st.peak_profit:
            st.peak_profit, changed = profit, True
        if profit < st.worst_profit:
            st.worst_profit, changed = profit, True
        acts: list[Action] = []
        r_now = profit / st.initial_risk if st.initial_risk > 0 else 0.0
        # ---- profit protection (close if X% of peak profit is given back) ----
        if s.profit_protection_enabled and st.initial_risk > 0:
            if not st.armed and st.peak_profit >= s.profit_activation_r * st.initial_risk:
                st.armed, changed = True, True
            if st.armed and st.peak_profit > 0 and profit <= st.peak_profit * (1 - s.profit_retrace_percent / 100):
                acts.append(Action("CLOSE", f"PROFIT_PROTECTION (peak {st.peak_profit:.2f}, now {profit:.2f}, retrace>{s.profit_retrace_percent:.0f}%)"))
                self._persist(st, changed)
                return acts
        # ---- partial close ------------------------------------------------
        if s.partial_close_enabled and not st.partial_done and r_now >= s.partial_close_r:
            vol = math.floor(p.volume * s.partial_close_fraction / spec.volume_step + 1e-9) * spec.volume_step
            if vol >= spec.volume_min and p.volume - vol >= spec.volume_min - 1e-9:
                acts.append(Action("PARTIAL", f"partial close at {r_now:.2f}R", volume=round(vol, 8)))
            st.partial_done, changed = True, True        # attempt once
        # ---- break-even ---------------------------------------------------
        sp = ask - bid
        if s.breakeven_enabled and not st.be_done and r_now >= s.breakeven_r:
            be = st.entry + (sp if p.direction == "BUY" else -sp)
            be = round(be, spec.digits)
            better = (p.direction == "BUY" and be > (p.sl or 0)) or (p.direction == "SELL" and (not p.sl or be < p.sl))
            if better:
                acts.append(Action("MODIFY_SL", f"break-even at {r_now:.2f}R", price=be))
            st.be_done, changed = True, True
        # ---- ATR trailing stop --------------------------------------------
        if s.trailing_enabled and r_now >= s.trailing_start_r and atr_v > 0:
            if p.direction == "BUY":
                new = round(bid - s.trailing_atr_mult * atr_v, spec.digits)
                if new > (p.sl or 0) and new < bid - spec.stops_level * spec.point:
                    acts.append(Action("MODIFY_SL", "trailing stop", price=new))
            else:
                new = round(ask + s.trailing_atr_mult * atr_v, spec.digits)
                if (not p.sl or new < p.sl) and new > ask + spec.stops_level * spec.point:
                    acts.append(Action("MODIFY_SL", "trailing stop", price=new))
        self._persist(st, changed)
        return acts

    def _persist(self, st: ManagedState, changed: bool) -> None:
        if changed:
            self._save(st)


class ReversalPolicy:
    """A reversal needs ALL of: strong opposite signal, clear margin over the current side,
    enough strength, persistence over several analysis cycles, minimum holding time and
    daily reversal budget. Tiny price moves can never trigger it."""
    def __init__(self, settings: Settings):
        self.s = settings

    def should_reverse(self, pos_dir: str, decision, opp_streak: int, held_sec: float,
                       reversals_today: int) -> tuple[bool, str]:
        s = self.s
        opp = "SELL" if pos_dir == "BUY" else "BUY"
        if decision.action != opp:
            return False, "no opposite decision"
        same_score = decision.buy_score if pos_dir == "BUY" else decision.sell_score
        opp_score = decision.buy_score if opp == "BUY" else decision.sell_score
        checks = [
            (decision.overall_score >= s.reversal_threshold, f"overall {decision.overall_score:.0f} < REVERSAL_THRESHOLD {s.reversal_threshold:.0f}"),
            (opp_score - same_score >= s.min_signal_change, f"signal change {opp_score - same_score:.0f} < MIN_SIGNAL_CHANGE {s.min_signal_change:.0f}"),
            (decision.strength >= s.min_signal_strength, f"strength {decision.strength:.0f} < MIN_SIGNAL_STRENGTH {s.min_signal_strength:.0f}"),
            (opp_streak >= s.signal_persistence_ticks, f"persistence {opp_streak}/{s.signal_persistence_ticks}"),
            (held_sec >= s.min_hold_before_reversal_sec, f"held {held_sec:.0f}s < {s.min_hold_before_reversal_sec}s"),
            (reversals_today < s.max_reversals_per_day, f"reversals today {reversals_today} >= {s.max_reversals_per_day}"),
        ]
        for ok, why in checks:
            if not ok:
                return False, why
        return True, "reversal conditions met"
