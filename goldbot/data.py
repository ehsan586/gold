"""Market-data and broker-neutral data structures + helpers."""
from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

TF_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}


@dataclass(frozen=True)
class Candle:
    time: int          # open time, epoch seconds (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class Tick:
    time: float
    bid: float
    ask: float


@dataclass(frozen=True)
class SymbolSpec:
    name: str
    digits: int
    point: float
    tick_size: float
    tick_value: float          # account-currency value of one tick for 1 lot
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level: int           # points
    freeze_level: int          # points
    trade_allowed: bool = True
    currency_profit: str = ""          # profit currency reported by the broker (may be empty)


@dataclass(frozen=True)
class AccountInfo:
    login: int
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float        # percent; 0 when no margin used
    leverage: int
    margin_mode: str           # NETTING / HEDGING / EXCHANGE
    is_demo: bool
    currency: str = "USD"


@dataclass
class Position:
    ticket: int
    symbol: str
    direction: str             # BUY / SELL
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float
    time_open: float
    magic: int = 0
    comment: str = ""


@dataclass
class OrderResult:
    ok: bool
    retcode: int = 0
    ticket: int = 0
    price: float = 0.0
    requested_price: float = 0.0
    volume: float = 0.0
    slippage_points: float = 0.0
    message: str = ""


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def resample(m1: Sequence[Candle], tf_sec: int) -> list[Candle]:
    """Aggregate M1 candles into COMPLETE higher-timeframe candles only."""
    if tf_sec == 60:
        return list(m1)
    out: list[Candle] = []
    cur = None
    for c in m1:
        b = c.time - c.time % tf_sec
        if cur is None or cur[0] != b:
            if cur is not None:
                out.append(Candle(cur[0], cur[1], cur[2], cur[3], cur[4], cur[5]))
            cur = [b, c.open, c.high, c.low, c.close, c.volume]
        else:
            cur[2] = max(cur[2], c.high)
            cur[3] = min(cur[3], c.low)
            cur[4] = c.close
            cur[5] += c.volume
    if cur is not None and m1 and cur[0] + tf_sec <= m1[-1].time + 60:
        out.append(Candle(cur[0], cur[1], cur[2], cur[3], cur[4], cur[5]))
    return out


class SeriesIndex:
    """Pre-resampled series with O(log n) 'candles known at time t' slicing (for backtests)."""
    def __init__(self, m1: Sequence[Candle]):
        self.series = {tf: resample(m1, s) for tf, s in TF_SECONDS.items() if s >= 60}
        self.close_times = {tf: [c.time + TF_SECONDS[tf] for c in cs] for tf, cs in self.series.items()}

    def upto(self, tf: str, t_close: float, n: int) -> list[Candle]:
        i = bisect.bisect_right(self.close_times[tf], t_close)
        return self.series[tf][max(0, i - n):i]


def load_csv(path: str) -> list[Candle]:
    """Load M1 candles from CSV. Accepted header names (case-insensitive):
    time (epoch seconds OR 'YYYY.MM.DD HH:MM[:SS]' OR ISO), open, high, low, close, volume/tick_volume.
    Delimiter comma, semicolon or tab is auto-detected."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = max([",", ";", "\t"], key=sample.count)
        rd = csv.DictReader(f, delimiter=delim)
        rows = []
        for r in rd:
            r = {(k or "").strip().lower().strip("<>"): (v or "").strip() for k, v in r.items()}
            t = r.get("time") or r.get("date")
            if "date" in r and "time" in r and r["date"] and r["time"] and not r["time"].isdigit():
                t = r["date"] + " " + r["time"]
            rows.append(Candle(_parse_time(t), float(r["open"]), float(r["high"]),
                               float(r["low"]), float(r["close"]),
                               float(r.get("tick_volume") or r.get("volume") or r.get("vol") or 0)))
    rows.sort(key=lambda c: c.time)
    out, seen = [], set()
    for c in rows:                      # drop duplicate timestamps
        if c.time not in seen:
            seen.add(c.time)
            out.append(c)
    return out


def _parse_time(t: str) -> int:
    t = t.strip()
    if t.isdigit():
        return int(t)
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return int(datetime.strptime(t, fmt).replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            pass
    return int(datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp())
