"""Market Regime detector. It classifies the ENVIRONMENT; it never decides trades.
Features (past-only, regime timeframe): Kaufman efficiency ratio (trendiness), ATR vs its 100-bar median
(volatility regime). `confidence` is a MARGIN score (how far the features are from the class boundaries),
NOT a probability. Classifier quality is measured separately by evaluate_regimes()."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from .data import Candle
from .indicators import atr, median
from .snapshot import MarketSnapshot

REGIME_FEATURE_VERSION = "r1"
REGIMES = ("TRENDING", "RANGING", "HIGH_VOLATILITY", "LOW_VOLATILITY", "TRANSITION", "UNCERTAIN")
ER_TREND, ER_RANGE, ATR_HIGH, ATR_LOW = 0.35, 0.18, 2.0, 0.6


@dataclass
class RegimeResult:
    regime: str
    confidence: float
    timestamp: float
    timeframe: str
    feature_version: str
    features: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"regime": self.regime, "confidence": round(self.confidence, 3), "confidence_kind": "margin_score_not_probability",
                "timestamp": self.timestamp, "timeframe": self.timeframe, "feature_version": self.feature_version,
                "features": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.features.items()},
                "context": self.context}


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def regime_features(c: list[Candle]) -> dict | None:
    if len(c) < 60:
        return None
    cl = [x.close for x in c]
    path = sum(abs(cl[i] - cl[i - 1]) for i in range(len(cl) - 20, len(cl)))
    er = abs(cl[-1] - cl[-21]) / path if path > 0 else 0.0
    a = atr(c)
    med = median(a[-100:]) or 1e-9
    return {"er": er, "atr_ratio": a[-1] / med}


def decide_regime(f: dict) -> tuple[str, float]:
    er, ratio = f["er"], f["atr_ratio"]
    if ratio >= ATR_HIGH:
        return "HIGH_VOLATILITY", _clamp(0.5 + (ratio - ATR_HIGH) / ATR_HIGH)
    if ratio <= ATR_LOW:
        return "LOW_VOLATILITY", _clamp(0.5 + (ATR_LOW - ratio) / ATR_LOW)
    if er >= ER_TREND:
        return "TRENDING", _clamp(0.5 + (er - ER_TREND) / ER_TREND)
    if er <= ER_RANGE:
        return "RANGING", _clamp(0.5 + (ER_RANGE - er) / ER_RANGE)
    mid = (ER_TREND - ER_RANGE) / 2
    return "TRANSITION", _clamp(0.3 + 0.4 * min(er - ER_RANGE, ER_TREND - er) / mid, 0.0, 0.7)


class RegimeDetector:
    def __init__(self, settings, history: int = 300):
        self.s = settings
        self.hist: deque = deque(maxlen=history)      # (bar_time, regime) one entry per regime-timeframe bar
        self._last_bar = None

    def classify(self, snap: MarketSnapshot) -> RegimeResult:
        tf = self.s.regime_timeframe
        c = snap.closed(tf)
        f = regime_features(c)
        if f is None:
            return RegimeResult("UNCERTAIN", 0.0, snap.as_of, tf, REGIME_FEATURE_VERSION, {"reason": "insufficient history"},
                                self._context())
        regime, conf = decide_regime(f)
        bar = c[-1].time
        recent = [r for _, r in list(self.hist)[-5:]] + [regime]
        flips = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
        note = None
        if flips >= 3:
            regime, conf, note = "UNCERTAIN", min(conf, 0.3), f"unstable: {flips} regime changes in the last 6 bars"
        if bar != self._last_bar:
            self.hist.append((bar, regime))
            self._last_bar = bar
        feats = dict(f)
        if note:
            feats["reason"] = note
        return RegimeResult(regime, conf, snap.as_of, tf, REGIME_FEATURE_VERSION, feats, self._context())

    def _context(self) -> dict:
        h = [r for _, r in self.hist]
        run = 0
        for r in reversed(h):
            if r == (h[-1] if h else None):
                run += 1
            else:
                break
        return {"previous": h[-2] if len(h) > 1 else None, "persistence_bars": run,
                "distribution": dict(Counter(h[-100:])), "bars_observed": len(h)}


def evaluate_regimes(c: list[Candle], horizon: int = 8, warmup: int = 120) -> dict:
    """INDEPENDENT measurement. Each bar is classified using ONLY data up to that bar; the forward window
    (next `horizon` bars) is used purely as the evaluation label. If regimes carried no information the
    forward statistics would look the same in every regime."""
    stats: dict[str, dict] = {}
    seq = []
    for i in range(warmup, len(c) - horizon):
        f = regime_features(c[:i + 1])
        if f is None:
            continue
        reg, _ = decide_regime(f)
        seq.append(reg)
        fw = c[i:i + horizon + 1]
        a = atr(c[:i + 1])[-1] or 1e-9
        steps = [abs(fw[k].close - fw[k - 1].close) for k in range(1, len(fw))]
        path = sum(steps)
        s = stats.setdefault(reg, {"n": 0, "fwd_vol_atr": 0.0, "fwd_er": 0.0})
        s["n"] += 1
        s["fwd_vol_atr"] += (path / horizon) / a
        s["fwd_er"] += abs(fw[-1].close - fw[0].close) / path if path > 0 else 0.0
    for s in stats.values():
        s["fwd_vol_atr"], s["fwd_er"] = round(s["fwd_vol_atr"] / s["n"], 4), round(s["fwd_er"] / s["n"], 4)
    flips = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    return {"per_regime": stats, "samples": len(seq), "flip_rate": round(flips / max(1, len(seq) - 1), 4),
            "note": "If forward ER does not differ between TRENDING and RANGING, the classifier carries no information on this data."}
