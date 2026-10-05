"""Plain-Python indicators (no numpy needed). All return lists aligned with the input."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .data import Candle


def ema(values: Sequence[float], period: int) -> list[float]:
    if not values:
        return []
    k = 2.0 / (period + 1)
    out = [float(values[0])]
    for v in values[1:]:
        out.append(out[-1] + k * (v - out[-1]))
    return out


def true_ranges(c: Sequence[Candle]) -> list[float]:
    tr = []
    for i, x in enumerate(c):
        if i == 0:
            tr.append(x.high - x.low)
        else:
            pc = c[i - 1].close
            tr.append(max(x.high - x.low, abs(x.high - pc), abs(x.low - pc)))
    return tr


def atr(c: Sequence[Candle], period: int = 14) -> list[float]:
    tr = true_ranges(c)
    out: list[float] = []
    for i, v in enumerate(tr):
        if i < period:
            out.append(sum(tr[:i + 1]) / (i + 1))
        else:
            out.append((out[-1] * (period - 1) + v) / period)
    return out


def rsi(closes: Sequence[float], period: int = 14) -> list[float]:
    n = len(closes)
    if n < 2:
        return [50.0] * n
    out = [50.0]
    avg_g = avg_l = 0.0
    for i in range(1, n):
        d = closes[i] - closes[i - 1]
        g, l = max(d, 0.0), max(-d, 0.0)
        if i <= period:
            avg_g += g / period
            avg_l += l / period
            if i < period:
                out.append(50.0)
                continue
        else:
            avg_g = (avg_g * (period - 1) + g) / period
            avg_l = (avg_l * (period - 1) + l) / period
        out.append(100.0 if avg_l == 0 else 100.0 - 100.0 / (1 + avg_g / avg_l))
    return out


def macd_hist(closes: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9) -> list[float]:
    ef, es = ema(closes, fast), ema(closes, slow)
    macd = [a - b for a, b in zip(ef, es)]
    sig = ema(macd, signal)
    return [m - s for m, s in zip(macd, sig)]


@dataclass(frozen=True)
class Swing:
    index: int
    price: float
    kind: str  # 'H' or 'L'


def swings(c: Sequence[Candle], left: int = 3, right: int = 3) -> list[Swing]:
    """Confirmed swing points only (need `right` candles after the pivot)."""
    out: list[Swing] = []
    for i in range(left, len(c) - right):
        hi, lo = c[i].high, c[i].low
        if all(hi > c[j].high for j in range(i - left, i)) and all(hi >= c[j].high for j in range(i + 1, i + right + 1)):
            out.append(Swing(i, hi, "H"))
        if all(lo < c[j].low for j in range(i - left, i)) and all(lo <= c[j].low for j in range(i + 1, i + right + 1)):
            out.append(Swing(i, lo, "L"))
    return sorted(out, key=lambda s: s.index)


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
