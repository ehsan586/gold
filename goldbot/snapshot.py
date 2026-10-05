"""Single source of truth for market data + the temporal data contract.

Every analytical component of one decision cycle receives the SAME immutable MarketSnapshot.
Time fields:
  as_of        market-clock time the data is valid for (= close time of the newest candle it contains)
  available_at system-clock time at which the system received/built it
Only CLOSED candles are allowed, and none may close after `as_of` (look-ahead protection)."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from .data import TF_SECONDS, Candle, SymbolSpec, Tick
from .settings import Settings


@dataclass(frozen=True)
class MarketSnapshot:
    snapshot_id: str
    symbol: str
    source: str
    as_of: float
    available_at: float
    feature_version: str
    tick: Tick
    spec: SymbolSpec
    candles: dict                      # tf -> tuple[Candle,...]  (closed candles only)
    data_hash: str

    def closed(self, tf: str) -> list[Candle]:
        return list(self.candles.get(tf, ()))

    def last_close_time(self, tf: str) -> float | None:
        c = self.candles.get(tf)
        return (c[-1].time + TF_SECONDS[tf]) if c else None

    def metadata(self) -> dict:
        return {"snapshot_id": self.snapshot_id, "symbol": self.symbol, "source": self.source, "as_of": self.as_of,
                "available_at": self.available_at, "feature_version": self.feature_version, "data_hash": self.data_hash,
                "timeframes": {tf: len(c) for tf, c in self.candles.items()}}


@dataclass
class DataQuality:
    ok: bool
    score: float
    fatal: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "score": round(self.score, 1), "fatal": self.fatal, "warnings": self.warnings}


def make_snapshot_id(symbol: str, as_of: float, feature_version: str) -> str:
    return hashlib.sha256(f"{symbol}|{as_of:.0f}|{feature_version}".encode()).hexdigest()[:16]


def hash_candles(candles: dict) -> str:
    h = hashlib.sha256()
    for tf in sorted(candles):
        c = candles[tf]
        h.update(f"{tf}:{len(c)}".encode())
        for x in c[-5:]:
            h.update(f"{x.time},{x.open},{x.high},{x.low},{x.close}".encode())
    return h.hexdigest()[:16]


class DataService:
    """The ONLY component that fetches market data from the broker for analysis."""
    def __init__(self, broker, settings: Settings, depth: int = 300):
        self.broker, self.s, self.depth = broker, settings, depth

    def timeframes(self) -> list[str]:
        s = self.s
        return sorted({*s.htf_timeframes, *s.ltf_timeframes, s.sl_timeframe, s.regime_timeframe, s.forecast_timeframe},
                      key=lambda t: TF_SECONDS[t])

    def build(self, symbol: str, spec: SymbolSpec, now: float, as_of: float | None = None, tick: Tick | None = None) -> MarketSnapshot:
        tick = tick or self.broker.tick(symbol)
        candles = {tf: tuple(self.broker.candles(symbol, tf, self.depth)) for tf in self.timeframes()}
        closes = [c[-1].time + TF_SECONDS[tf] for tf, c in candles.items() if c]
        a = as_of if as_of is not None else (max(closes) if closes else 0.0)
        return MarketSnapshot(make_snapshot_id(symbol, a, self.s.feature_version), symbol, getattr(self.broker, "name", "?"),
                              a, now, self.s.feature_version, tick, spec, candles, hash_candles(candles))


class DataValidator:
    def __init__(self, settings: Settings, strict: bool = False, min_bars: int = 40):
        self.s, self.strict, self.min_bars = settings, strict, min_bars

    def validate(self, snap: MarketSnapshot) -> DataQuality:
        fatal, warn = [], []
        t = snap.tick
        if not (all(math.isfinite(v) for v in (t.bid, t.ask)) and t.bid > 0 and t.ask >= t.bid):
            fatal.append(f"INVALID_TICK bid={t.bid} ask={t.ask}")
        age = snap.available_at - t.time
        if age > self.s.max_tick_age_sec:
            fatal.append(f"STALE_TICK {age:.0f}s")
        if not snap.candles or not any(snap.candles.values()):
            fatal.append("NO_CANDLES")
        usable = 0
        for tf, cs in snap.candles.items():
            sec = TF_SECONDS[tf]
            if len(cs) >= self.min_bars:
                usable += 1
            else:
                warn.append(f"{tf}: only {len(cs)} bars")
            prev = None
            for c in cs[-80:]:                      # the newest 80 bars are what every component actually uses
                if not all(math.isfinite(v) and v > 0 for v in (c.open, c.high, c.low, c.close)) or \
                   c.high < max(c.open, c.close) - 1e-9 or c.low > min(c.open, c.close) + 1e-9:
                    fatal.append(f"{tf}: BAD_OHLC at {c.time}")
                    break
                if prev is not None and c.time <= prev:
                    fatal.append(f"{tf}: NON_MONOTONIC_TIME at {c.time}")
                    break
                if c.time % sec:
                    (fatal if self.strict else warn).append(f"{tf}: TIMESTAMP_MISALIGNED {c.time}")
                    break
                prev = c.time
            if cs and cs[-1].time + sec > snap.as_of + 1e-6:
                fatal.append(f"{tf}: FUTURE_DATA candle closes {cs[-1].time + sec - snap.as_of:.0f}s after as_of")
        if usable == 0:
            fatal.append("INSUFFICIENT_HISTORY on every timeframe")
        score = max(0.0, 100.0 - 25 * len(fatal) - 4 * len(warn))
        return DataQuality(not fatal, score, fatal, warn)
