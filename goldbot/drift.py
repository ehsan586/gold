"""Drift detection: PSI on feature distributions, total-variation distance on categorical behaviour
(regimes, forecast directions, agent outputs), a two-proportion z-test on model error and ECE change on
calibration. A flag triggers EVALUATION; it never triggers automatic retraining or deployment."""
from __future__ import annotations

import math
from collections import Counter, deque

from .forecast import FEATURES


CHI2_P001_DF4 = 18.47        # chi-square critical value, 4 degrees of freedom, p = 0.001


def psi_stat(ref: list[float], cur: list[float], bins: int = 5) -> tuple[float, float]:
    """Returns (noise-corrected PSI, chi-square statistic). PSI*n_eff behaves like chi-square(bins-1) when both samples come
    from the same distribution, so BOTH the effect size (PSI) and the significance (chi-square) are checked: this keeps
    false drift alarms rare for small windows without hiding real shifts."""
    if len(ref) < 20 or len(cur) < 20:
        return 0.0, 0.0
    xs = sorted(ref)
    edges = [xs[int(len(xs) * k / bins)] for k in range(1, bins)]
    def dist(v):
        c = [0] * bins
        for x in v:
            c[sum(1 for e in edges if x > e)] += 1
        return [max(n / len(v), 1e-4) for n in c]
    p, q = dist(ref), dist(cur)
    raw = sum((b - a) * math.log(b / a) for a, b in zip(p, q))
    neff = 1 / (1 / len(ref) + 1 / len(cur))
    return max(0.0, raw - (bins - 1) / neff), raw * neff


def psi(ref: list[float], cur: list[float], bins: int = 5) -> float:
    return psi_stat(ref, cur, bins)[0]


def tv_distance(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()) or 1, sum(b.values()) or 1
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b))


class DriftDetector:
    def __init__(self, settings, events=None):
        w = settings.drift_window
        self.w, self.events = w, events
        self.feat = {k: deque(maxlen=2 * w) for k in FEATURES}
        self.regime = deque(maxlen=2 * w)
        self.fdir = deque(maxlen=2 * w)
        self.err = deque(maxlen=2 * w)
        self.agents: dict[str, deque] = {}
        self.flags: list[dict] = []
        self._announced: dict[tuple, str] = {}

    def observe(self, features: dict | None, regime: str, forecast_dir: str, agent_dirs: dict) -> None:
        if features:
            for k in FEATURES:
                self.feat[k].append(features[k])
        self.regime.append(regime)
        self.fdir.append(forecast_dir)
        for n, d in agent_dirs.items():
            self.agents.setdefault(n, deque(maxlen=2 * self.w)).append(d)

    def observe_error(self, wrong: float) -> None:
        self.err.append(wrong)

    def check(self, calibration: dict | None = None, reference_ece: float | None = None, now: float | None = None) -> list[dict]:
        w, flags = self.w, []
        def split(d):
            d = list(d)
            return (d[-2 * w:-w], d[-w:]) if len(d) >= 2 * w else (None, None)
        for k, d in self.feat.items():
            r, c = split(d)
            if r:
                v, chi2 = psi_stat(r, c)
                if v >= 0.1 and chi2 >= CHI2_P001_DF4:
                    flags.append({"kind": "feature_distribution", "name": k, "severity": "HIGH" if v >= 0.25 else "MEDIUM", "value": round(v, 3), "threshold": "PSI 0.10/0.25"})
        for kind, d in (("regime_distribution", self.regime), ("forecast_behavior", self.fdir)):
            r, c = split(d)
            if r:
                v = tv_distance(Counter(r), Counter(c))
                if v >= 0.3:
                    flags.append({"kind": kind, "name": kind, "severity": "HIGH" if v >= 0.5 else "MEDIUM", "value": round(v, 3), "threshold": "TV 0.30/0.50"})
        for n, d in self.agents.items():
            r, c = split(d)
            if r:
                v = tv_distance(Counter(r), Counter(c))
                if v >= 0.3:
                    flags.append({"kind": "agent_behavior", "name": n, "severity": "HIGH" if v >= 0.5 else "MEDIUM", "value": round(v, 3), "threshold": "TV 0.30/0.50"})
        r, c = split(self.err)
        if r:
            p1, p2 = sum(r) / len(r), sum(c) / len(c)
            p = (sum(r) + sum(c)) / (len(r) + len(c))
            se = math.sqrt(p * (1 - p) * (1 / len(r) + 1 / len(c))) or 1e-9
            z = (p2 - p1) / se
            if z >= 2.0:
                flags.append({"kind": "model_error", "name": "ensemble_error_rate", "severity": "HIGH" if z >= 3 else "MEDIUM", "value": round(z, 2), "threshold": "z 2/3"})
        if calibration and reference_ece is not None and calibration.get("ece") is not None and calibration["ece"] - reference_ece >= 0.08:
            flags.append({"kind": "calibration", "name": "ece", "severity": "MEDIUM", "value": calibration["ece"], "threshold": "ECE +0.08"})
        self.flags = flags
        for f in flags:
            key = (f["kind"], f["name"])
            if self._announced.get(key) != f["severity"] and self.events:
                self.events.emit("drift", "DRIFT_DETECTED", f, None, now)
            self._announced[key] = f["severity"]
        for key in [k for k in self._announced if k not in {(f["kind"], f["name"]) for f in flags}]:
            del self._announced[key]
        return flags
