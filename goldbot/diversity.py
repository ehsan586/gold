"""Agent Diversity Monitor. N agents agreeing is NOT N independent confirmations if they read the same data.
We report AGREEMENT and DIVERSITY separately, from three measurable things:
  * output correlation over recent bars (needs >= MIN_SAMPLES common bars, otherwise unavailable),
  * input-feature overlap (declared feature tags per agent, Jaccard similarity),
  * effective independent opinions  N_eff = n / (1 + (n-1) * mean_redundancy)  (design-effect formula)."""
from __future__ import annotations

import math
from collections import deque
from itertools import combinations

AGENT_FEATURES = {
    "trend": {"price@H4", "price@H1", "price@M15", "ema"},
    "structure": {"price@M15", "price@H1", "swings@M15", "swings@H1"},
    "price_action": {"price@M5", "price@M15", "swings@M15", "ema"},
    "momentum": {"price@M15", "rsi", "macd"},
    "volatility": {"price@M15", "atr"},
    "liquidity": {"price@M15", "swings@M15"},
    "news": {"calendar"},
    "regime": {"price@M15", "atr", "efficiency"},
    "forecast": {"price@M5", "price@M15", "ema", "rsi", "atr", "efficiency"},
}
DIRECTIONAL = ("trend", "structure", "price_action", "momentum", "liquidity", "forecast")
MIN_SAMPLES = 30


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def pearson(x: list, y: list) -> float | None:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < MIN_SAMPLES:
        return None
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


class DiversityMonitor:
    def __init__(self, window: int = 200):
        self.hist: dict[str, deque] = {k: deque(maxlen=window) for k in DIRECTIONAL}
        self._last_key = None

    @staticmethod
    def _sig(direction: str | None) -> float | None:
        return {"BUY": 1.0, "UP": 1.0, "SELL": -1.0, "DOWN": -1.0, "HOLD": 0.0, "FLAT": 0.0}.get(direction)

    def update(self, results: dict, forecast_direction: str | None, bar_key) -> dict:
        cur: dict[str, float | None] = {}
        for n in DIRECTIONAL:
            if n == "forecast":
                cur[n] = self._sig(forecast_direction)
            else:
                r = results.get(n)
                cur[n] = None if (r is None or r.is_no_data) else self._sig(r.direction.value)
        if bar_key != self._last_key:                       # one observation per bar, not per 5-second cycle
            for n, v in cur.items():
                self.hist[n].append(v)
            self._last_key = bar_key
        return self.report(cur)

    def report(self, cur: dict) -> dict:
        voters = [n for n, v in cur.items() if v in (1.0, -1.0)]
        out = {"voters": voters, "agreement": "NONE", "agreement_ratio": None, "diversity": "UNKNOWN",
               "effective_independent_opinions": None, "redundancy_source": None, "per_agent_uniqueness": {}, "redundant_pairs": []}
        if len(voters) < 2:
            return out
        buys = sum(1 for n in voters if cur[n] > 0)
        ratio = max(buys, len(voters) - buys) / len(voters)
        out["agreement_ratio"] = round(ratio, 3)
        out["agreement"] = "HIGH" if (ratio >= 0.8 and len(voters) >= 3) else "MEDIUM" if ratio >= 0.6 else "LOW"
        rho: dict[tuple, float] = {}
        sources = set()
        for a, b in combinations(voters, 2):
            c = pearson(list(self.hist[a]), list(self.hist[b]))
            if c is not None:
                rho[(a, b)] = max(0.0, c); sources.add("correlation")
            else:
                rho[(a, b)] = jaccard(AGENT_FEATURES[a], AGENT_FEATURES[b]); sources.add("feature_overlap")
        mean_rho = sum(rho.values()) / len(rho)
        n = len(voters)
        neff = n / (1 + (n - 1) * mean_rho)
        out["effective_independent_opinions"] = round(neff, 2)
        out["redundancy_source"] = "+".join(sorted(sources))
        share = neff / n
        out["diversity"] = "HIGH" if share >= 0.7 else "MEDIUM" if share >= 0.45 else "LOW"
        out["per_agent_uniqueness"] = {v: round(1 - sum(r for (a, b), r in rho.items() if v in (a, b)) / (n - 1), 3) for v in voters}
        out["redundant_pairs"] = [{"pair": list(k), "redundancy": round(v, 2)} for k, v in sorted(rho.items(), key=lambda kv: -kv[1])[:3]]
        return out
