"""Execution Engine: the ONLY component that sends orders to the broker.

Guards on every OPEN:  Risk-Governor token (HMAC, short TTL) -> idempotency claim ->
position/duplicate check -> broker call -> record requested/actual price, retcode, slippage.
Risk-REDUCING actions (close, tighten SL, partial close) need no token but are still logged.
Widening a stop-loss is refused here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .broker import Broker
from .data import OrderResult, iso
from .risk import RiskDecision, RiskGovernor, TradeRequest
from .settings import Settings


@dataclass
class ExecResult:
    ok: bool
    message: str
    trade_id: int | None = None
    ticket: int | None = None
    order: OrderResult | None = None


class ExecutionEngine:
    """Action Interface. Modes: DISABLED (nothing is ever sent), SIMULATION (virtual account), MT5_DEMO (real orders
    on a DEMO account, explicit opt-in). Every attempted action is logged as an event."""
    def __init__(self, broker: Broker, db, settings: Settings, governor: RiskGovernor, clock, events=None, gate=None,
                 mode: str = "SIMULATION"):
        self.broker, self.db, self.s, self.gov, self.clock = broker, db, settings, governor, clock
        self.events, self.gate, self.mode = events, gate, mode
        self.last_success_ts: float | None = None

    def _ev(self, action: str, ok: bool, detail: dict) -> None:
        if self.events:
            kind = "SIMULATION_ACTION" if getattr(self.broker, "name", "") in ("paper", "shadow") else "EXECUTION_ACTION"
            self.events.emit("execution", kind, {"action": action, "ok": ok, "mode": self.mode, **detail}, None, self.clock())

    def _allowed(self, action: str, source: str, ticket: int) -> bool:
        if self.gate is None:
            return True
        return self.gate.check_reducing(action, source, ticket, self.clock())

    # ------------------------------------------------------------------ open
    def open_position(self, req: TradeRequest, approval: RiskDecision, strategy_version: str) -> ExecResult:
        now = self.clock()
        if self.mode == "DISABLED":
            self._ev("OPEN", False, {"status": "SUPPRESSED_DISABLED", "direction": req.direction})
            return ExecResult(False, "execution disabled")
        if not self.gov.verify(req, approval, now):
            self.db.log_event("CRITICAL", "execution", "rejected_no_approval",
                              f"open {req.direction} refused: missing/invalid/expired risk approval", {"cmd": req.command_id})
            return ExecResult(False, "no valid risk approval")
        if not req.sl:
            return ExecResult(False, "stop loss required")
        if not self.db.claim_command("open:" + req.command_id):
            self.db.log_event("WARN", "execution", "duplicate_command", f"duplicate open {req.command_id} ignored")
            return ExecResult(False, "duplicate command")
        existing = [p for p in self.broker.positions(req.symbol) if p.magic == self.s.magic_number]
        if existing:
            return ExecResult(False, "position already exists for this symbol (duplicate protection)")
        res = self.broker.send_market_order(req.symbol, req.direction, approval.lot, req.sl, req.tp,
                                            f"gb {strategy_version}", self.s.magic_number, self.s.max_slippage_points)
        meta = req.meta
        row = {
            "idempotency_key": req.command_id, "ticket": res.ticket or None,
            "status": "OPEN" if res.ok else "REJECTED", "ts_open": iso(now), "symbol": req.symbol,
            "timeframe": meta.get("timeframe"), "market_regime": meta.get("market_regime"),
            "direction": req.direction, "entry_price": res.price if res.ok else None,
            "requested_price": res.requested_price or req.entry, "slippage": res.slippage_points,
            "retcode": res.retcode, "lot": approval.lot, "sl": req.sl, "tp": req.tp,
            "spread": meta.get("spread"), "atr": req.atr, "volatility": meta.get("volatility"),
            "decision_id": req.decision_id, "decision_score": meta.get("decision_score"),
            "agreement_score": meta.get("agreement_score"), "risk_score": approval.risk_pct,
            "agent_outputs_json": json.dumps(meta.get("agents", {}), sort_keys=True),
            "strategy_version": strategy_version,
        }
        tid = self.db.insert("trades", row)
        self._ev("OPEN", res.ok, {"direction": req.direction, "lot": approval.lot, "sl": req.sl, "tp": req.tp, "price": res.price,
                                  "requested": res.requested_price, "retcode": res.retcode, "slippage_pts": round(res.slippage_points, 2)})
        if not res.ok:
            self.db.log_event("ERROR", "execution", "order_rejected", f"{res.retcode}: {res.message}",
                              {"cmd": req.command_id, "retcode": res.retcode})
            return ExecResult(False, f"order rejected: {res.message}", tid, None, res)
        self.last_success_ts = now
        self.db.log_event("INFO", "execution", "opened",
                          f"{req.direction} {approval.lot} @ {res.price} SL {req.sl} TP {req.tp} (req {res.requested_price}, slip {res.slippage_points:.1f}pt)",
                          {"ticket": res.ticket, "retcode": res.retcode, "trade_id": tid})
        return ExecResult(True, "opened", tid, res.ticket, res)

    # ------------------------------------------------------------------ risk-reducing actions
    def close_position(self, ticket: int, reason: str, volume: float | None = None, source: str = "engine") -> ExecResult:
        if not self._allowed("CLOSE" if volume is None else "PARTIAL", source, ticket):
            return ExecResult(False, "safety gate denied")
        self.db.kv_set(f"exit_reason:{ticket}", reason)      # reconciliation reads this
        res = self.broker.close_position(ticket, volume, comment=reason[:20])
        self._ev("CLOSE" if volume is None else "PARTIAL", res.ok, {"ticket": ticket, "reason": reason, "price": res.price, "retcode": res.retcode})
        lvl = "INFO" if res.ok else "ERROR"
        self.db.log_event(lvl, "execution", "close" if volume is None else "partial_close",
                          f"ticket {ticket} {reason}: {'ok' if res.ok else res.message}", {"retcode": res.retcode})
        if res.ok:
            self.last_success_ts = self.clock()
        return ExecResult(res.ok, res.message, None, ticket, res)

    def modify_sl(self, ticket: int, new_sl: float, reason: str, source: str = "engine") -> ExecResult:
        if not self._allowed("MODIFY_SL", source, ticket):
            return ExecResult(False, "safety gate denied")
        pos = next((p for p in self.broker.positions() if p.ticket == ticket), None)
        if pos is None:
            return ExecResult(False, "no such position")
        tighter = (pos.direction == "BUY" and new_sl > (pos.sl or 0)) or \
                  (pos.direction == "SELL" and (not pos.sl or new_sl < pos.sl))
        if not tighter:
            self.db.log_event("WARN", "execution", "sl_widen_refused", f"ticket {ticket}: {pos.sl} -> {new_sl} refused")
            return ExecResult(False, "refused: would not tighten the stop")
        res = self.broker.modify_position(ticket, sl=new_sl)
        self._ev("MODIFY_SL", res.ok, {"ticket": ticket, "from": pos.sl, "to": new_sl, "reason": reason})
        self.db.log_event("INFO" if res.ok else "ERROR", "execution", "modify_sl",
                          f"ticket {ticket} SL {pos.sl} -> {new_sl} ({reason}): {'ok' if res.ok else res.message}")
        if res.ok:
            self.db.update("trades", self._trade_id(ticket), {"sl": new_sl}) if self._trade_id(ticket) else None
        return ExecResult(res.ok, res.message, None, ticket, res)

    def modify_tp(self, ticket: int, new_tp: float, reason: str) -> ExecResult:
        res = self.broker.modify_position(ticket, tp=new_tp)
        self.db.log_event("INFO" if res.ok else "ERROR", "execution", "modify_tp", f"ticket {ticket} TP -> {new_tp} ({reason})")
        return ExecResult(res.ok, res.message, None, ticket, res)

    def partial_close(self, ticket: int, volume: float, reason: str, source: str = "engine") -> ExecResult:
        return self.close_position(ticket, reason, volume=volume, source=source)

    def _trade_id(self, ticket: int):
        r = self.db.query("SELECT id FROM trades WHERE ticket=? AND status='OPEN' ORDER BY id DESC LIMIT 1", (ticket,))
        return r[0]["id"] if r else None
