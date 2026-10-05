"""Consistency Engine: deterministic cross-check of specialist outputs + regime + forecast + data quality.
It reports consensus, disagreement, contradictions and missing information. It never invents information
and does NOT average everything into one confidence number: the score below is a transparent heuristic
(formula in CONSISTENCY_FORMULA) and always travels together with the raw flags."""
from __future__ import annotations

from dataclasses import dataclass, field

from .decision import DEFAULT_WEIGHTS

CONSISTENCY_FORMULA = "100 * (0.50*agreement + 0.25*forecast_alignment + 0.15*regime_fit + 0.10*completeness) * data_quality/100"
SPECIALISTS = ("trend", "structure", "price_action", "momentum", "liquidity")
GATES = ("volatility", "news")
STANCE = {"BUY": "BULLISH", "SELL": "BEARISH", "HOLD": "NEUTRAL"}


@dataclass
class ConsistencyReport:
    consensus: str = "NONE"                 # BULLISH / BEARISH / NEUTRAL / NONE
    disagreement: str = "NONE"              # LOW / MEDIUM / HIGH / NONE
    consistency_score: float = 0.0
    missing_agents: list = field(default_factory=list)
    contradictions: list = field(default_factory=list)
    flags: dict = field(default_factory=dict)
    stances: dict = field(default_factory=dict)
    components: dict = field(default_factory=dict)
    formula: str = CONSISTENCY_FORMULA

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["consistency_score"] = round(self.consistency_score, 1)
        return d


class ConsistencyEngine:
    def __init__(self, settings, weights: dict | None = None):
        self.s = settings
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}

    def evaluate(self, results: dict, regime, forecast, data_quality: float, system_state: str, diversity: dict | None = None,
                 exclude: set | frozenset = frozenset()) -> ConsistencyReport:
        rep = ConsistencyReport()
        bull = bear = 0.0
        present = 0
        expected = [a for a in SPECIALISTS if a not in exclude]
        for a in SPECIALISTS + GATES:
            r = results.get(a)
            if a in exclude:
                continue
            if r is None or r.is_no_data:
                rep.missing_agents.append(a)
                rep.stances[a] = "MISSING"
                continue
            rep.stances[a] = STANCE[r.direction.value]
            if a in SPECIALISTS:
                present += 1
                bull += self.w[a] if r.direction.value == "BUY" else 0
                bear += self.w[a] if r.direction.value == "SELL" else 0
        rep.stances["regime"] = regime.regime if regime else "MISSING"
        f_dir = getattr(forecast, "direction", "UNKNOWN")
        rep.stances["forecast"] = {"UP": "BULLISH", "DOWN": "BEARISH", "FLAT": "NEUTRAL"}.get(f_dir, "MISSING")
        rep.stances["forecast_uncertainty"] = getattr(forecast, "uncertainty_label", "UNKNOWN")
        # ---- consensus / disagreement -----------------------------------------------------------
        if bull + bear == 0:
            rep.consensus, rep.disagreement, split = ("NEUTRAL" if present else "NONE"), ("NONE" if not present else "LOW"), 0.0
        else:
            hi, lo = max(bull, bear), min(bull, bear)
            split = lo / hi
            rep.disagreement = "LOW" if split < 0.2 else "MEDIUM" if split < 0.5 else "HIGH"
            rep.consensus = "NONE" if split >= 0.8 else ("BULLISH" if bull > bear else "BEARISH")
        rep.flags["AGENT_DISAGREEMENT"] = rep.disagreement
        rep.flags["FORECAST_UNCERTAINTY"] = rep.stances["forecast_uncertainty"]
        # ---- contradictions ------------------------------------------------------------------------
        def st(n): return rep.stances.get(n)
        opp = {"BULLISH": "BEARISH", "BEARISH": "BULLISH"}
        for a, b, name in (("trend", "momentum", "TREND_VS_MOMENTUM"), ("trend", "structure", "TREND_VS_STRUCTURE"),
                           ("structure", "momentum", "STRUCTURE_VS_MOMENTUM")):
            if st(a) in opp and st(b) == opp[st(a)]:
                rep.contradictions.append(name)
        if rep.consensus in opp and rep.stances["forecast"] == opp[rep.consensus]:
            rep.contradictions.append("FORECAST_VS_CONSENSUS")
        if rep.consensus in opp and regime and regime.regime == "RANGING":
            rep.contradictions.append("DIRECTIONAL_CONSENSUS_IN_RANGING_REGIME")
        if rep.consensus in opp and rep.stances["forecast_uncertainty"] == "HIGH":
            rep.contradictions.append("DIRECTIONAL_CONSENSUS_WITH_HIGH_FORECAST_UNCERTAINTY")
        if diversity and diversity.get("agreement") == "HIGH" and diversity.get("diversity") == "LOW":
            rep.contradictions.append("HIGH_AGREEMENT_BUT_LOW_DIVERSITY")
        if data_quality < 60:
            rep.contradictions.append("LOW_DATA_QUALITY")
        if system_state in ("DEGRADED", "ERROR", "RECOVERY", "STARTING"):
            rep.contradictions.append(f"SYSTEM_{system_state}")
        # ---- transparent score --------------------------------------------------------------------------
        agreement = 1 - split if (bull + bear) else 0.5
        fa = {"BULLISH": {"BULLISH": 1.0, "BEARISH": 0.0}, "BEARISH": {"BEARISH": 1.0, "BULLISH": 0.0}}
        forecast_al = fa.get(rep.consensus, {}).get(rep.stances["forecast"], 0.5)
        reg_fit = {"TRENDING": 1.0, "RANGING": 0.6, "LOW_VOLATILITY": 0.6, "HIGH_VOLATILITY": 0.6, "TRANSITION": 0.4,
                   "UNCERTAIN": 0.2}.get(regime.regime if regime else "", 0.5)
        completeness = present / len(expected) if expected else 0.0
        rep.components = {"agreement": round(agreement, 3), "forecast_alignment": forecast_al, "regime_fit": reg_fit,
                          "completeness": round(completeness, 3)}
        rep.consistency_score = 100 * (0.5 * agreement + 0.25 * forecast_al + 0.15 * reg_fit + 0.10 * completeness) * (data_quality / 100)
        return rep
