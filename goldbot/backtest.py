"""Backtester: replays M1 candles through the REAL engine (same agents, decision engine, risk governor,
execution engine and position manager as live) against the paper broker.

Honest limits: candle-based (no tick data); history is BID; SL-before-TP inside one candle; fixed
spread/slippage assumptions. Results on synthetic data prove the machinery only, never profitability.
"""
from __future__ import annotations

import dataclasses
import math
import time

from .agents import CalendarFile
from .broker_paper import PaperBroker
from .data import Candle
from .db import Database
from .engine import TradingEngine
from .settings import Settings


MIN_MEANINGFUL_TRADES = 30

def metrics(trades: list[dict], curve: list[float], start_balance: float) -> dict:
    pnl = [t["pnl"] for t in trades if t["pnl"] is not None]
    wins, losses = [p for p in pnl if p > 0], [-p for p in pnl if p < 0]
    rs = [t["r_multiple"] for t in trades if t["r_multiple"] is not None]
    peak, mdd = start_balance, 0.0
    for e in curve:
        peak = max(peak, e)
        mdd = max(mdd, (peak - e) / peak * 100 if peak else 0)
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t["exit_reason"] or "?"] = reasons.get(t["exit_reason"] or "?", 0) + 1
    meaningful = len(pnl) >= MIN_MEANINGFUL_TRADES
    return {"trades": len(pnl), "evaluation_status": "VALID" if meaningful else "INSUFFICIENT_DATA",
            "statistically_meaningful": meaningful, "minimum_trades": MIN_MEANINGFUL_TRADES,
            "net_profit": round(sum(pnl), 2),
            "win_rate_pct": round(100 * len(wins) / len(pnl), 1) if pnl else 0.0,
            "profit_factor": round(sum(wins) / sum(losses), 3) if losses else (None if not wins else math.inf),
            "expectancy_money": round(sum(pnl) / len(pnl), 3) if pnl else 0.0,
            "expectancy_R": round(sum(rs) / len(rs), 3) if rs else 0.0,
            "avg_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
            "avg_loss": round(-sum(losses) / len(losses), 2) if losses else 0.0,
            "max_drawdown_pct": round(mdd, 2), "exit_reasons": reasons,
            "final_equity": round(curve[-1], 2) if curve else start_balance}


def run_backtest(m1: list[Candle], settings: Settings, balance: float = 800.0, spread_points: int = 25,
                 slippage_points: int = 2, commission_per_lot: float = 0.0, warmup: int = 3700,
                 weights: dict | None = None, progress=None, disabled_agents=(), use_regime: bool = True,
                 use_forecast: bool = True) -> dict:
    if len(m1) < warmup + 100:
        raise ValueError(f"need at least {warmup + 100} M1 candles (got {len(m1)})")
    db = Database(":memory:")
    db.init_schema()
    broker = PaperBroker(m1, settings.symbol, balance, 100, spread_points, slippage_points, commission_per_lot)
    eng = TradingEngine(settings, db, broker, clock=broker.now, news=CalendarFile("/nonexistent"), sim=True)
    eng.disabled_agents, eng.use_regime, eng.use_forecast = set(disabled_agents), use_regime, use_forecast
    eng.periodic_eval = False
    eng.publish_enabled = False
    if weights:
        eng.weights = {**eng.weights, **weights}
        eng.decider.weights = eng.weights
    for _ in range(warmup):
        broker.advance()
    eng.start()
    curve, t0 = [], time.time()
    n = len(m1) - warmup
    while broker.advance():
        eng.step(broker.now())
        curve.append(broker.account().equity)
        if progress and len(curve) % 2000 == 0:
            progress(len(curve), n)
    trades = db.query("SELECT * FROM trades WHERE status='CLOSED' ORDER BY id")
    res = metrics(trades, curve, balance)
    res["r_values"] = [t["r_multiple"] for t in trades if t["r_multiple"] is not None]
    if not res["statistically_meaningful"]:
        res["evaluation_note"] = (f"Only {res['trades']} closed trades were observed; at least "
                                  f"{res['minimum_trades']} are required before performance metrics are treated as evidence.")
    res.update({"bars": n, "seconds": round(time.time() - t0, 1),
                "blocked_by_risk": db.count("risk_events"), "decisions_logged": db.count("decisions")})
    return res


def walk_forward(m1: list[Candle], settings: Settings, split: float = 0.7, **kw) -> dict:
    """In-sample / out-of-sample: two independent runs on the first and last part of the data."""
    cut = int(len(m1) * split)
    warm = kw.get("warmup", 3700)
    return {"in_sample": run_backtest(m1[:cut], settings, **kw),
            "out_of_sample": run_backtest(m1[cut - warm:], settings, **kw)}   # overlap only for indicator warm-up


def sweep_retrace(m1: list[Candle], settings: Settings, values=(20, 30, 40, 50), **kw) -> dict:
    """Profit-protection retrace test. Do NOT assume 40 is best: compare on out-of-sample data too."""
    out = {}
    for v in values:
        s = dataclasses.replace(settings, profit_retrace_percent=float(v))
        out[v] = run_backtest(m1, s, **kw)
    return out
