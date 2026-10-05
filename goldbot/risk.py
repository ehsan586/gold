"""Risk Governor: highest authority before execution. It can only APPROVE (with a signed,
short-lived token that the ExecutionEngine verifies) or BLOCK. Nothing else can create a
valid token, so no agent / decision / learning component can bypass it."""
from __future__ import annotations

import hashlib
import hmac
import math
import secrets
from dataclasses import dataclass, field
from datetime import datetime

from .data import AccountInfo, Position, SymbolSpec, Tick, iso
from .settings import Settings

TOKEN_TTL_SEC = 30


# ------------------------------------------------------------------ sizing / planning
def calc_lot(equity: float, risk_pct: float, entry: float, sl: float, spec: SymbolSpec):
    """Returns (lot, risk_money_at_lot, problem). Rounds DOWN to the broker step and
    never rounds up to the minimum lot (that would exceed the risk budget)."""
    dist = abs(entry - sl)
    if dist <= 0 or spec.tick_size <= 0 or spec.tick_value <= 0:
        return 0.0, 0.0, "invalid SL distance / symbol tick data"
    loss_per_lot = dist / spec.tick_size * spec.tick_value
    raw = equity * risk_pct / 100.0 / loss_per_lot
    lot = math.floor(raw / spec.volume_step + 1e-9) * spec.volume_step
    lot = round(min(lot, spec.volume_max), 8)
    if lot < spec.volume_min - 1e-12:
        return 0.0, 0.0, (f"lot {raw:.4f} is below broker minimum {spec.volume_min}: "
                          f"SL too wide for the risk budget ({risk_pct:.2f}% of equity)")
    return lot, lot * loss_per_lot, ""


def position_risk(p: Position, spec: SymbolSpec) -> float:
    """Money at risk if SL is hit (0 once SL is at/after break-even). Unknown SL => infinite."""
    if not p.sl:
        return math.inf
    d = (p.price_open - p.sl) if p.direction == "BUY" else (p.sl - p.price_open)
    return max(0.0, d) / spec.tick_size * spec.tick_value * p.volume


def plan_sl_tp(direction: str, entry: float, atr_v: float, support, resistance,
               s: Settings, spec: SymbolSpec):
    """SL from market structure when it lies within [min,max] ATR, else ATR-based."""
    base = atr_v * s.sl_atr_mult
    dist = base
    if direction == "BUY" and support:
        d = entry - (support - 0.2 * atr_v)
        if s.min_sl_atr_mult * atr_v <= d <= s.max_sl_atr_mult * atr_v:
            dist = d
    if direction == "SELL" and resistance:
        d = (resistance + 0.2 * atr_v) - entry
        if s.min_sl_atr_mult * atr_v <= d <= s.max_sl_atr_mult * atr_v:
            dist = d
    sign = 1 if direction == "BUY" else -1
    sl = round(entry - sign * dist, spec.digits)
    tp = round(entry + sign * dist * s.tp_rr, spec.digits)
    return sl, tp


# ------------------------------------------------------------------ equity tracking
class EquityTracker:
    def __init__(self, db):
        self.db = db

    def update(self, equity: float, now: float) -> dict:
        day = iso(now)[:10]
        peak = float(self.db.kv_get("eq_peak") or equity)
        if self.db.kv_get("eq_day") != day:
            self.db.kv_set("eq_day", day)
            self.db.kv_set("eq_day_start", repr(equity))
        peak = max(peak, equity)
        self.db.kv_set("eq_peak", repr(peak))
        start = float(self.db.kv_get("eq_day_start") or equity)
        return {"peak": peak, "day_start": start,
                "drawdown_pct": max(0.0, (peak - equity) / peak * 100) if peak > 0 else 0.0,
                "daily_loss_pct": max(0.0, (start - equity) / start * 100) if start > 0 else 0.0,
                "daily_pnl": equity - start}

    def reset_peak(self, equity: float) -> None:
        self.db.kv_set("eq_peak", repr(equity))


def trade_stats(db, now: float) -> dict:
    day = iso(now)[:10]
    q = db.query
    opened = q("SELECT ts_open FROM trades ORDER BY id DESC LIMIT 1")
    closed = q("SELECT ts_close, pnl FROM trades WHERE status='CLOSED' ORDER BY id DESC LIMIT 20")
    loss_rows = q("SELECT ts_close FROM trades WHERE status='CLOSED' AND pnl<0 ORDER BY id DESC LIMIT 1")
    consec = 0
    for r in closed:
        if (r["pnl"] or 0) < 0:
            consec += 1
        else:
            break
    ts = lambda x: datetime.fromisoformat(x).timestamp() if x else None  # noqa: E731
    return {
        "trades_today": q("SELECT COUNT(*) c FROM trades WHERE substr(ts_open,1,10)=?", (day,))[0]["c"],
        "reversals_today": q("SELECT COUNT(*) c FROM trades WHERE exit_reason='REVERSAL' AND substr(ts_close,1,10)=?", (day,))[0]["c"],
        "pnl_today": q("SELECT COALESCE(SUM(pnl),0) s FROM trades WHERE status='CLOSED' AND substr(ts_close,1,10)=?", (day,))[0]["s"],
        "consecutive_losses": consec,
        "last_open": ts(opened[0]["ts_open"]) if opened else None,
        "last_close": ts(closed[0]["ts_close"]) if closed else None,
        "last_loss_close": ts(loss_rows[0]["ts_close"]) if loss_rows else None,
        "last_reversal": float(db.kv_get("last_reversal_ts") or 0) or None,
    }


# ------------------------------------------------------------------ governor
@dataclass
class TradeRequest:
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    atr: float
    command_id: str
    decision_id: int | None = None
    is_reversal: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class RiskDecision:
    approved: bool = False
    status: str = "BLOCKED"            # APPROVED / BLOCKED / CRITICAL
    reasons: list = field(default_factory=list)
    rules: list = field(default_factory=list)
    lot: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    token: str = ""
    expires: float = 0.0


class RiskGovernor:
    def __init__(self, settings: Settings, db):
        self.s, self.db = settings, db
        self._secret = secrets.token_bytes(32)

    def _sig(self, req: TradeRequest, lot: float, expires: float) -> str:
        msg = f"{req.command_id}|{req.symbol}|{req.direction}|{lot}|{req.sl}|{req.tp}|{expires}".encode()
        return hmac.new(self._secret, msg, hashlib.sha256).hexdigest()

    def verify(self, req: TradeRequest, dec: RiskDecision, now: float) -> bool:
        return bool(dec.approved and dec.token and now <= dec.expires and
                    hmac.compare_digest(dec.token, self._sig(req, dec.lot, dec.expires)))

    def evaluate(self, req: TradeRequest, *, account: AccountInfo, spec: SymbolSpec, tick: Tick,
                 positions: list[Position], now: float, new_trades_allowed: bool, health_ok: bool,
                 vol_class: str, eq: dict, stats: dict) -> RiskDecision:
        s = self.s
        dec = RiskDecision()
        rules = dec.rules

        def rule(name: str, ok: bool, detail: str = "") -> bool:
            rules.append({"rule": name, "ok": bool(ok), "detail": detail})
            if not ok:
                dec.reasons.append(f"{name}: {detail}" if detail else name)
            return ok

        rule("system_state", new_trades_allowed, "trading paused / error / emergency stop")
        rule("health", health_ok, "system DEGRADED (a critical component failed)")
        rule("demo_only", account.is_demo or s.allow_real_account, "account is not a demo account")
        rule("equity_positive", account.equity > 0, f"equity {account.equity:.2f}")
        age = now - tick.time
        rule("price_valid", tick.bid > 0 and tick.ask >= tick.bid and age <= s.max_tick_age_sec,
             f"bid={tick.bid} ask={tick.ask} age={age:.0f}s")
        spread_pts = (tick.ask - tick.bid) / spec.point if spec.point else 1e9
        rule("spread", spread_pts <= s.max_spread_points, f"{spread_pts:.0f} pts > {s.max_spread_points}")
        rule("symbol_tradable", spec.trade_allowed, "trading disabled for symbol")
        rule("volatility", vol_class != "EXTREME", "extreme volatility")
        critical = eq["drawdown_pct"] >= s.max_account_drawdown
        rule("account_drawdown", not critical, f"{eq['drawdown_pct']:.2f}% >= {s.max_account_drawdown}%")
        rule("daily_loss", eq["daily_loss_pct"] < s.max_daily_loss, f"{eq['daily_loss_pct']:.2f}% >= {s.max_daily_loss}%")
        rule("consecutive_losses", stats["consecutive_losses"] < s.max_consecutive_losses,
             f"{stats['consecutive_losses']} in a row")
        lim = s.max_trades_per_day
        rule("trades_per_day", stats["trades_today"] < lim, f"{stats['trades_today']} >= {lim}")
        if req.is_reversal:
            rule("reversals_per_day", stats["reversals_today"] <= s.max_reversals_per_day,
                 f"{stats['reversals_today']} > {s.max_reversals_per_day}")
        else:
            def since(k):
                return (now - stats[k]) if stats.get(k) else math.inf
            rule("min_time_between_trades", since("last_open") >= s.min_time_between_trades_sec, "too soon after last entry")
            rule("cooldown_after_loss", since("last_loss_close") >= s.cooldown_after_loss_sec, "cooling down after a loss")
            rule("cooldown_after_exit", since("last_close") >= s.cooldown_after_exit_sec, "cooling down after last exit")
            rule("cooldown_after_reversal", since("last_reversal") >= s.cooldown_after_reversal_sec, "cooling down after a reversal")
        mine = [p for p in positions if p.symbol == req.symbol]
        rule("no_open_position", not mine, "a position is already open (no pyramiding in v1)")
        # ---- stops
        min_dist = spec.stops_level * spec.point
        sl_ok = req.sl > 0 and ((req.direction == "BUY" and req.sl < req.entry - min_dist) or
                                (req.direction == "SELL" and req.sl > req.entry + min_dist))
        rule("stop_loss", sl_ok, f"SL {req.sl} invalid vs entry {req.entry} (min distance {min_dist:.2f})")
        tp_ok = req.tp == 0 or ((req.direction == "BUY" and req.tp > req.entry + min_dist) or
                                (req.direction == "SELL" and req.tp < req.entry - min_dist))
        rule("take_profit", tp_ok, f"TP {req.tp} invalid")
        # ---- sizing
        mult = 0.5 if vol_class == "HIGH" else 1.0
        risk_pct = s.max_risk_per_trade * mult
        lot, risk_amt, why = (0.0, 0.0, "stop loss invalid") if not sl_ok else \
            calc_lot(account.equity, risk_pct, req.entry, req.sl, spec)
        lot = min(lot, s.max_lot_cap) if lot else 0.0
        if lot:
            risk_amt = lot * abs(req.entry - req.sl) / spec.tick_size * spec.tick_value
        rule("position_size", lot > 0, why)
        rule("risk_per_trade", lot > 0 and risk_amt <= account.equity * s.max_risk_per_trade / 100 + 1e-9,
             f"risk {risk_amt:.2f} exceeds {s.max_risk_per_trade}% of equity")
        open_risk = sum(position_risk(p, spec) for p in positions)
        total = (open_risk + risk_amt) / account.equity * 100 if account.equity > 0 else math.inf
        rule("total_exposure", total <= s.max_total_exposure, f"{total:.2f}% > {s.max_total_exposure}%")
        need = lot * spec.contract_size * req.entry / max(account.leverage, 1)
        rule("margin", lot > 0 and account.free_margin >= 2 * need,
             f"free margin {account.free_margin:.2f} < 2x required {need:.2f} (leverage 1:{account.leverage})")
        rule("margin_level", account.margin_level == 0 or account.margin_level >= s.min_margin_level_pct,
             f"{account.margin_level:.0f}% < {s.min_margin_level_pct:.0f}%")
        dec.lot, dec.risk_amount = lot, risk_amt
        dec.risk_pct = risk_amt / account.equity * 100 if account.equity > 0 else 0.0
        dec.approved = not dec.reasons
        dec.status = "APPROVED" if dec.approved else ("CRITICAL" if critical else "BLOCKED")
        if dec.approved:
            dec.expires = now + TOKEN_TTL_SEC
            dec.token = self._sig(req, lot, dec.expires)
        return dec

    def snapshot(self, account: AccountInfo, positions: list[Position], spec: SymbolSpec,
                 eq: dict, stats: dict, spread_pts: float) -> dict:
        s = self.s
        exposure = sum(position_risk(p, spec) for p in positions)
        exposure = 0.0 if math.isinf(exposure) else exposure
        return {
            "limits": {"max_risk_per_trade": s.max_risk_per_trade, "max_total_exposure": s.max_total_exposure,
                       "max_daily_loss": s.max_daily_loss, "max_account_drawdown": s.max_account_drawdown,
                       "max_consecutive_losses": s.max_consecutive_losses, "max_trades_per_day": s.max_trades_per_day,
                       "max_reversals_per_day": s.max_reversals_per_day},
            "exposure_pct": round(exposure / account.equity * 100, 3) if account.equity > 0 else 0,
            "exposure_money": round(exposure, 2),
            "daily_loss_pct": round(eq["daily_loss_pct"], 3), "daily_pnl": round(eq["daily_pnl"], 2),
            "drawdown_pct": round(eq["drawdown_pct"], 3), "peak_equity": round(eq["peak"], 2),
            "margin": round(account.margin, 2), "free_margin": round(account.free_margin, 2),
            "margin_level": round(account.margin_level, 1), "leverage": account.leverage,
            "consecutive_losses": stats["consecutive_losses"], "trades_today": stats["trades_today"],
            "reversals_today": stats["reversals_today"], "spread_points": round(spread_pts, 1),
        }
