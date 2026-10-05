"""The seven analysis agents. Agents ANALYSE ONLY: they have no access to the broker
or the execution engine. Every agent returns a validated AgentResult; on any error or
missing data the wrapper returns NO_DATA (never a guess)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .data import Candle, SymbolSpec, Tick
from .indicators import atr, ema, macd_hist, median, rsi, swings
from .models import AgentResult, AgentOutputError
from .settings import Settings

MIN_BARS = 40


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


@dataclass
class AgentContext:
    symbol: str
    candles: dict[str, list[Candle]]
    tick: Tick
    spec: SymbolSpec
    settings: Settings
    now: float
    news: Optional["CalendarFile"] = None

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.now, timezone.utc)

    def get(self, tf: str, n: int = MIN_BARS) -> list[Candle] | None:
        c = self.candles.get(tf) or []
        return c if len(c) >= n else None

    @property
    def entry_tf(self) -> str:
        l = self.settings.ltf_timeframes
        return l[1] if len(l) > 1 else l[0]

    @property
    def structure_tf(self) -> str:
        return self.settings.ltf_timeframes[0]


class Agent:
    name = "base"

    def analyze(self, ctx: AgentContext) -> AgentResult:
        raise NotImplementedError

    def run(self, ctx: AgentContext) -> AgentResult:
        try:
            res = self.analyze(ctx)
            if not isinstance(res, AgentResult):
                raise AgentOutputError("agent returned wrong type")
            return res
        except Exception as e:  # noqa: BLE001 - any failure => NO_DATA, never a guess
            return AgentResult.no_data(self.name, ctx.symbol, "-", f"agent error {type(e).__name__}: {e}", now=ctx.dt)

    def _res(self, ctx, tf, direction, conf, strength, regime, dq, reason, **extra) -> AgentResult:
        return AgentResult(agent=self.name, symbol=ctx.symbol, timeframe=tf, direction=direction,
                           confidence=_clamp(conf), strength=_clamp(strength), market_regime=regime,
                           data_quality=_clamp(dq), reason=reason, timestamp=ctx.dt, extra=extra)

    def _nodata(self, ctx, why) -> AgentResult:
        return AgentResult.no_data(self.name, ctx.symbol, "-", why, now=ctx.dt)


# ---------------------------------------------------------------- 1. TREND
TF_W = {"H4": 0.3, "H1": 0.4, "M30": 0.3, "M15": 0.3, "M5": 0.2, "M1": 0.1}


class TrendAgent(Agent):
    name = "trend"

    def analyze(self, ctx):
        tfs = list(ctx.settings.htf_timeframes) + list(ctx.settings.ltf_timeframes[:1])
        num = den = strg = 0.0
        signs, used, notes = [], [], []
        hhhl = 0
        for tf in tfs:
            c = ctx.get(tf)
            if not c:
                continue
            cl = [x.close for x in c]
            e20, e50 = ema(cl, 20), ema(cl, 50)
            a = atr(c)[-1] or 1e-9
            slope = e20[-1] - e20[-5]
            s = 1 if (e20[-1] > e50[-1] and cl[-1] > e50[-1] and slope > 0) else \
                -1 if (e20[-1] < e50[-1] and cl[-1] < e50[-1] and slope < 0) else 0
            w = TF_W.get(tf, 0.2)
            num += w * s
            den += w
            strg += w * min(100, abs(e20[-1] - e50[-1]) / a * 60)
            signs.append(s)
            used.append(tf)
            notes.append(f"{tf}:{'UP' if s > 0 else 'DOWN' if s < 0 else 'FLAT'}")
            if not hhhl:                      # swing structure of the highest available TF
                sw = swings(c[-120:])
                hi = [x for x in sw if x.kind == "H"][-2:]
                lo = [x for x in sw if x.kind == "L"][-2:]
                if len(hi) == 2 and len(lo) == 2:
                    if hi[1].price > hi[0].price and lo[1].price > lo[0].price:
                        hhhl = 1
                    elif hi[1].price < hi[0].price and lo[1].price < lo[0].price:
                        hhhl = -1
        if not used:
            return self._nodata(ctx, "not enough candles on any trend timeframe")
        agg, strength = num / den, strg / den
        dq = 100 * len(used) / len(tfs)
        direction = "BUY" if agg >= 0.6 else "SELL" if agg <= -0.6 else "HOLD"
        if direction == "HOLD":
            conf, regime = 40 + (1 - abs(agg)) * 20, "RANGE"
        else:
            aligned = all(s == signs[0] for s in signs) and len(signs) > 1
            conf = min(95, 50 + abs(agg) * 35 + (10 if aligned else 0))
            if hhhl:
                conf = min(95, conf + 5) if (hhhl > 0) == (direction == "BUY") else conf - 10
            regime = "TREND_UP" if direction == "BUY" else "TREND_DOWN"
        return self._res(ctx, "+".join(used), direction, conf, strength, regime, dq,
                         f"{', '.join(notes)}; alignment={agg:+.2f}; swings={'HH/HL' if hhhl > 0 else 'LH/LL' if hhhl < 0 else 'mixed'}")


# ---------------------------------------------------------------- 2. STRUCTURE
class StructureAgent(Agent):
    name = "structure"

    def analyze(self, ctx):
        tf = ctx.structure_tf
        c = ctx.get(tf, 60)
        if not c:
            return self._nodata(ctx, f"not enough {tf} candles")
        c = c[-150:]
        a = atr(c)[-1] or 1e-9
        sw = swings(c)
        hi = [s for s in sw if s.kind == "H"]
        lo = [s for s in sw if s.kind == "L"]
        if len(hi) < 2 or len(lo) < 2:
            return self._res(ctx, tf, "HOLD", 30, 10, "UNKNOWN", 60, "swing structure unclear (fewer than 2 swing highs/lows)")
        hh, hl = hi[-1].price > hi[-2].price, lo[-1].price > lo[-2].price
        bias = 1 if (hh and hl) else -1 if (not hh and not hl) else 0
        last = c[-1].close
        bos = 1 if (bias == 1 and last > hi[-1].price + 0.1 * a) else -1 if (bias == -1 and last < lo[-1].price - 0.1 * a) else 0
        choch = -1 if (bias == 1 and last < lo[-1].price) else 1 if (bias == -1 and last > hi[-1].price) else 0
        fake_up = any(x.high > hi[-1].price and x.close < hi[-1].price for x in c[-3:])
        fake_dn = any(x.low < lo[-1].price and x.close > lo[-1].price for x in c[-3:])
        score = bias * 30 + bos * 20 + choch * 35 + (20 if fake_dn else 0) - (20 if fake_up else 0)
        hc = ctx.get(ctx.settings.htf_timeframes[-1], 60)       # optional higher-TF confirmation
        htf_bias = 0
        if hc:
            hs = swings(hc[-150:])
            h2 = [s for s in hs if s.kind == "H"][-2:]
            l2 = [s for s in hs if s.kind == "L"][-2:]
            if len(h2) == 2 and len(l2) == 2:
                htf_bias = 1 if (h2[1].price > h2[0].price and l2[1].price > l2[0].price) else \
                           -1 if (h2[1].price < h2[0].price and l2[1].price < l2[0].price) else 0
        direction = "BUY" if score >= 25 else "SELL" if score <= -25 else "HOLD"
        conf = 45 + min(40, abs(score) * 0.6)
        if direction != "HOLD":
            conf += 10 if (htf_bias == (1 if direction == "BUY" else -1)) else -5 if htf_bias else 0
        regime = "TREND_UP" if bias == 1 else "TREND_DOWN" if bias == -1 else "RANGE"
        res = max([s.price for s in hi if s.price > last], default=None)
        sup = max([s.price for s in lo if s.price < last], default=None)
        parts = [f"{'HH' if hh else 'LH'}/{'HL' if hl else 'LL'}"]
        if bos: parts.append("BOS " + ("up" if bos > 0 else "down"))
        if choch: parts.append("CHoCH " + ("up" if choch > 0 else "down"))
        if fake_up: parts.append("false breakout above resistance")
        if fake_dn: parts.append("false breakdown below support")
        return self._res(ctx, tf, direction, conf, min(100, abs(score) * 1.3), regime, 90, "; ".join(parts),
                         support=sup, resistance=res)


# ---------------------------------------------------------------- 3. PRICE ACTION
class PriceActionAgent(Agent):
    name = "price_action"

    def analyze(self, ctx):
        tf = ctx.entry_tf
        c, m = ctx.get(tf, 30), ctx.get(ctx.structure_tf, 60)
        if not c or not m:
            return self._nodata(ctx, "not enough candles")
        a = atr(c)[-1] or 1e-9
        last, prev = c[-1], c[-2]
        rng = (last.high - last.low) or 1e-9
        body = abs(last.close - last.open)
        up = last.high - max(last.open, last.close)
        lw = min(last.open, last.close) - last.low
        pats_b, pats_s = [], []
        if lw >= 2 * body and lw >= 0.5 * rng and last.close >= last.low + 0.6 * rng: pats_b.append("bullish rejection")
        if up >= 2 * body and up >= 0.5 * rng and last.close <= last.low + 0.4 * rng: pats_s.append("bearish rejection")
        if prev.close < prev.open and last.close > last.open and last.close >= prev.open and last.open <= prev.close: pats_b.append("bullish engulfing")
        if prev.close > prev.open and last.close < last.open and last.close <= prev.open and last.open >= prev.close: pats_s.append("bearish engulfing")
        if last.close > last.open and body >= 0.8 * a and last.close >= last.low + 0.75 * rng: pats_b.append("bullish momentum candle")
        if last.close < last.open and body >= 0.8 * a and last.close <= last.low + 0.25 * rng: pats_s.append("bearish momentum candle")
        sw = swings(m[-120:])
        px = last.close
        sup = [s.price for s in sw if s.kind == "L" and s.price <= px]
        resi = [s.price for s in sw if s.kind == "H" and s.price >= px]
        near_sup = bool(sup) and px - max(sup) <= 0.5 * a
        near_res = bool(resi) and min(resi) - px <= 0.5 * a
        mc = [x.close for x in m]
        e20, e50 = ema(mc, 20)[-1], ema(mc, 50)[-1]
        tsign = 1 if e20 > e50 else -1
        regime = "TREND_UP" if tsign > 0 else "TREND_DOWN"
        sb, ss = 25 * len(pats_b), 25 * len(pats_s)
        if sb == ss:
            return self._res(ctx, tf, "HOLD", 35, 20, regime, 90, "no clear candle behaviour")
        direction = "BUY" if sb > ss else "SELL"
        pats = pats_b if direction == "BUY" else pats_s
        near = near_sup if direction == "BUY" else near_res
        aligned = tsign == (1 if direction == "BUY" else -1)
        if not (near or aligned):                    # candle patterns alone are NOT a signal
            return self._res(ctx, tf, "HOLD", 30, 15, regime, 90,
                             f"{', '.join(pats)} ignored: no key level and against {ctx.structure_tf} trend")
        conf = 40 + max(sb, ss) * 0.6 + (15 if near else 0) + (10 if aligned else 0)
        return self._res(ctx, tf, direction, min(80, conf), body / a * 70 + max(sb, ss), regime, 90,
                         f"{', '.join(pats)}; {'at key level; ' if near else ''}{'with' if aligned else 'against'} {ctx.structure_tf} trend")


# ---------------------------------------------------------------- 4. MOMENTUM
class MomentumAgent(Agent):
    name = "momentum"

    def analyze(self, ctx):
        tf = ctx.structure_tf
        c = ctx.get(tf, 60)
        if not c:
            return self._nodata(ctx, f"not enough {tf} candles")
        cl = [x.close for x in c]
        a = atr(c)[-1] or 1e-9
        r, h = rsi(cl), macd_hist(cl)
        acc = h[-1] - h[-2]
        pers = sum(1 for x in h[-5:] if (x > 0) == (h[-1] > 0))
        nh = h[-1] / a
        win = cl[-30:-1]
        bear_div = cl[-1] >= max(win) and r[-1] < max(r[-30:-1]) - 3
        bull_div = cl[-1] <= min(win) and r[-1] > min(r[-30:-1]) + 3
        if r[-1] >= 72 and acc < 0:
            return self._res(ctx, tf, "HOLD", 55, 40, "UNKNOWN", 90, f"upside momentum exhausted (RSI {r[-1]:.0f}, decelerating)")
        if r[-1] <= 28 and acc > 0:
            return self._res(ctx, tf, "HOLD", 55, 40, "UNKNOWN", 90, f"downside momentum exhausted (RSI {r[-1]:.0f}, decelerating)")
        if bear_div:
            return self._res(ctx, tf, "SELL", 50, 35, "UNKNOWN", 90, "bearish RSI divergence at new high")
        if bull_div:
            return self._res(ctx, tf, "BUY", 50, 35, "UNKNOWN", 90, "bullish RSI divergence at new low")
        conf = min(75, 45 + pers * 5 + min(15, abs(nh) * 60))
        st = min(100, abs(nh) * 120 + pers * 8)
        if h[-1] > 0 and acc > 0 and 50 <= r[-1] < 72:
            return self._res(ctx, tf, "BUY", conf, st, "TREND_UP", 90, f"accelerating up (RSI {r[-1]:.0f}, persistence {pers}/5)")
        if h[-1] < 0 and acc < 0 and 28 < r[-1] <= 50:
            return self._res(ctx, tf, "SELL", conf, st, "TREND_DOWN", 90, f"accelerating down (RSI {r[-1]:.0f}, persistence {pers}/5)")
        return self._res(ctx, tf, "HOLD", 40, 20, "RANGE", 90, f"no clear momentum (RSI {r[-1]:.0f}, hist {nh:+.2f} ATR)")


# ---------------------------------------------------------------- 5. VOLATILITY (gate)
class VolatilityAgent(Agent):
    name = "volatility"

    def analyze(self, ctx):
        tf = ctx.structure_tf
        c = ctx.get(tf, 60)
        if not c:
            return self._nodata(ctx, f"not enough {tf} candles")
        a = atr(c)
        cur, med = a[-1], median(a[-100:]) or 1e-9
        ratio = cur / med
        spike = (c[-1].high - c[-1].low) >= 3 * a[-2]
        cls = "EXTREME" if (ratio >= 2.5 or spike) else "HIGH" if ratio >= 1.5 else "LOW" if ratio <= 0.6 else "NORMAL"
        regime = "HIGH_VOLATILITY" if cls in ("HIGH", "EXTREME") else "LOW_VOLATILITY" if cls == "LOW" else "UNKNOWN"
        trend = "expanding" if a[-1] > a[-5] * 1.1 else "contracting" if a[-1] < a[-5] * 0.9 else "stable"
        return self._res(ctx, tf, "HOLD", 70, 100 if cls == "EXTREME" else 50, regime, min(100, len(c) / 100 * 100),
                         f"{cls} volatility (ATR {cur:.2f} = {ratio:.2f}x median, {trend}{', SPIKE' if spike else ''})",
                         vol_class=cls, ratio=round(ratio, 3), atr=cur, block=(cls == "EXTREME"))


# ---------------------------------------------------------------- 6. LIQUIDITY
class LiquidityAgent(Agent):
    name = "liquidity"

    def analyze(self, ctx):
        tf = ctx.structure_tf
        c = ctx.get(tf, 60)
        if not c:
            return self._nodata(ctx, f"not enough {tf} candles")
        c = c[-120:]
        a = atr(c)[-1] or 1e-9
        sw = swings(c)
        hi = [s.price for s in sw if s.kind == "H"][-6:]
        lo = [s.price for s in sw if s.kind == "L"][-6:]
        eqh = sorted({round(max(x, y), 2) for i, x in enumerate(hi) for y in hi[i + 1:] if abs(x - y) <= 0.15 * a})
        eql = sorted({round(min(x, y), 2) for i, x in enumerate(lo) for y in lo[i + 1:] if abs(x - y) <= 0.15 * a})
        lv_hi, lv_lo = set(eqh) | set(hi[-2:]), set(eql) | set(lo[-2:])
        found = None
        for k in range(1, 5):                              # most recent candle first
            x = c[-k]
            for lv in lv_hi:
                if x.high > lv and x.close < lv:
                    found = ("SELL", lv, x.high - lv, lv in eqh, k); break
            if not found:
                for lv in lv_lo:
                    if x.low < lv and x.close > lv:
                        found = ("BUY", lv, lv - x.low, lv in eql, k); break
            if found: break
        note = "inferred from OHLC only (no order-flow data in MT5)"
        if not found:
            return self._res(ctx, tf, "HOLD", 40, 20, "UNKNOWN", 70, f"no recent liquidity sweep; {note}",
                             equal_highs=eqh, equal_lows=eql)
        d, lv, wick, eq, k = found
        conf = 55 + (10 if eq else 0) + min(10, wick / a * 20) - (k - 1) * 3
        return self._res(ctx, tf, d, conf, min(100, 40 + wick / a * 40), "UNKNOWN", 70,
                         f"{'equal ' if eq else ''}{'highs' if d == 'SELL' else 'lows'} swept at {lv:.2f} and rejected {k} bar(s) ago; {note}",
                         equal_highs=eqh, equal_lows=eql)


# ---------------------------------------------------------------- 7. NEWS / MACRO
class CalendarFile:
    """Reads a user-maintained JSON calendar: [{"time":"2026-10-02T12:30:00Z","impact":"HIGH","title":"US CPI"}].
    Nothing is invented: no file / stale file => NO_DATA. Plug a real feed in later by
    writing that file from a scheduled job."""
    def __init__(self, path: str):
        self.path, self._mt, self._ev = path, None, None

    def events(self) -> list[dict] | None:
        try:
            mt = os.path.getmtime(self.path)
            if mt != self._mt:
                with open(self.path, encoding="utf-8") as f:
                    raw = json.load(f)
                ev = []
                for e in raw:
                    t = datetime.fromisoformat(e["time"].replace("Z", "+00:00")).timestamp()
                    ev.append({"t": t, "impact": str(e.get("impact", "")).upper(), "title": str(e.get("title", ""))})
                self._ev, self._mt = ev, mt
            return self._ev
        except (OSError, ValueError, KeyError, TypeError):
            return None


class NewsAgent(Agent):
    name = "news"

    def analyze(self, ctx):
        ev = ctx.news.events() if ctx.news else None
        if ev is None:
            return self._nodata(ctx, "no news/calendar source configured")
        if not ev or max(e["t"] for e in ev) < ctx.now:
            return self._nodata(ctx, "calendar has no upcoming events (stale file)")
        w = ctx.settings.news_blackout_minutes * 60
        near = [e for e in ev if e["impact"] == "HIGH" and abs(e["t"] - ctx.now) <= w]
        if near:
            e = min(near, key=lambda x: abs(x["t"] - ctx.now))
            return self._res(ctx, "-", "HOLD", 90, 100, "HIGH_VOLATILITY", 100,
                             f"high-impact event within {ctx.settings.news_blackout_minutes} min: {e['title']}", block=True)
        return self._res(ctx, "-", "HOLD", 50, 0, "UNKNOWN", 100, "no high-impact event nearby (no directional view)", block=False)


def default_agents() -> list[Agent]:
    return [TrendAgent(), StructureAgent(), PriceActionAgent(), MomentumAgent(),
            VolatilityAgent(), LiquidityAgent(), NewsAgent()]
