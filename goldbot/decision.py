"""Weighted Decision Engine (NOT majority voting).

Each directional agent contributes weight * quality, where
    quality = (0.6*confidence + 0.4*strength)/100 * data_quality/100
Volatility and News are GATES: they can veto but never vote on direction.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .models import AgentResult
from .settings import Settings

DEFAULT_WEIGHTS = {"trend": 0.25, "structure": 0.20, "price_action": 0.15, "momentum": 0.15,
                   "liquidity": 0.10, "volatility": 0.10, "news": 0.05}
DIRECTIONAL = ("trend", "structure", "price_action", "momentum", "liquidity")
MIN_COVERAGE = 0.6        # share of directional weight that must have usable data
MIN_PARTICIPATION = 0.5   # share of directional weight that must hold an actual opinion
ABSTAIN_BELOW = 50        # a HOLD with confidence below this means 'nothing to say', not 'vote against'


@dataclass
class Decision:
    action: str = "HOLD"
    buy_score: float = 0.0
    sell_score: float = 0.0
    hold_score: float = 0.0
    overall_score: float = 0.0
    agreement_score: float = 0.0
    confidence: float = 0.0
    strength: float = 0.0
    contradiction: float = 0.0
    coverage: float = 0.0
    participation: float = 0.0
    lead: str = "HOLD"                       # which side leads before thresholds
    context_notes: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    gates: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("buy_score", "sell_score", "hold_score", "overall_score", "agreement_score",
                  "confidence", "strength", "contradiction", "coverage", "participation"):
            d[k] = round(d[k], 2)
        return d

    def reason_text(self) -> str:
        return "; ".join(self.gates + self.reasons) or "all conditions met"


class DecisionEngine:
    def __init__(self, settings: Settings, weights: dict | None = None):
        self.s = settings
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    @staticmethod
    def quality(r: AgentResult) -> float:
        return (0.6 * r.confidence + 0.4 * r.strength) / 100.0 * r.data_quality / 100.0

    def decide(self, results: dict[str, AgentResult], exclude=()) -> Decision:
        s = self.s
        w = {k: (0.0 if k in exclude else v) for k, v in self.weights.items()}
        d = Decision()
        # ---- gates ---------------------------------------------------------
        vol, news = results.get("volatility"), results.get("news")
        if "volatility" in exclude:
            pass
        elif vol is None or vol.is_no_data:
            d.gates.append("volatility data unavailable")
        elif vol.extra.get("block"):
            d.gates.append(f"EXTREME volatility ({vol.reason})")
        if news is not None and not news.is_no_data and news.extra.get("block"):
            d.gates.append(news.reason)
        if s.require_news_data and (news is None or news.is_no_data):
            d.gates.append("news data required but unavailable")
        # ---- directional scoring ---------------------------------------------
        w_total = sum(w[n] for n in DIRECTIONAL)
        usable = {n: r for n, r in results.items()
                  if n in DIRECTIONAL and w[n] > 0 and not r.is_no_data and r.data_quality >= s.min_data_quality}
        w_use = sum(w[n] for n in usable)
        d.coverage = w_use / w_total if w_total else 0.0
        if d.coverage < MIN_COVERAGE:
            d.reasons.append(f"insufficient agent data coverage ({d.coverage:.0%})")
            return self._finish(d)
        # event-type agents (price action, liquidity...) often have nothing to say: a weak HOLD abstains
        active = {n: r for n, r in usable.items() if not (r.direction.value == "HOLD" and r.confidence < ABSTAIN_BELOW)}
        w_use = sum(w[n] for n in active)
        d.participation = w_use / w_total
        if d.participation < MIN_PARTICIPATION:
            d.reasons.append(f"too few agents hold a directional opinion ({d.participation:.0%})")
            return self._finish(d)
        usable = active
        mass = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        for n, r in usable.items():
            mass[r.direction.value] += w[n] * self.quality(r)
        d.buy_score, d.sell_score, d.hold_score = (100 * mass[k] / w_use for k in ("BUY", "SELL", "HOLD"))
        if mass["BUY"] == mass["SELL"]:
            d.reasons.append("no directional lead")
            return self._finish(d)
        d.lead = "BUY" if mass["BUY"] > mass["SELL"] else "SELL"
        lead_score = d.buy_score if d.lead == "BUY" else d.sell_score
        aligned = {n: r for n, r in usable.items() if r.direction.value == d.lead}
        w_al = sum(w[n] for n in aligned)
        d.agreement_score = 100 * w_al / w_use
        opp, mine = ("SELL", "BUY") if d.lead == "BUY" else ("BUY", "SELL")
        d.contradiction = 100 * min(mass[opp], mass[mine]) / max(mass[opp], mass[mine], 1e-9)
        d.overall_score = lead_score * (0.5 + 0.5 * d.agreement_score / 100) * (1 - 0.5 * d.contradiction / 100)
        if aligned:
            d.confidence = sum(w[n] * r.confidence for n, r in aligned.items()) / w_al * (0.7 + 0.3 * d.coverage)
            d.strength = sum(w[n] * r.strength for n, r in aligned.items()) / w_al
        # ---- thresholds (HOLD is the default) --------------------------------
        if d.hold_score > lead_score:
            d.reasons.append("HOLD signals outweigh the leading direction")
        if d.overall_score < s.min_decision_score:
            d.reasons.append(f"overall score {d.overall_score:.1f} < {s.min_decision_score:.0f}")
        if d.agreement_score < s.min_agreement_score:
            d.reasons.append(f"agent agreement {d.agreement_score:.0f}% < {s.min_agreement_score:.0f}%")
        if d.contradiction > 50:
            d.reasons.append(f"agents contradict each other ({d.contradiction:.0f}%)")
        return self._finish(d)

    @staticmethod
    def _finish(d: Decision) -> Decision:
        d.action = d.lead if (not d.gates and not d.reasons and d.lead in ("BUY", "SELL")) else "HOLD"
        return d


class ContextualDecision:
    """Second stage: refines the weighted specialist decision with regime, forecast, consistency and diversity.
    It can only make the decision MORE conservative (downgrade to HOLD or demand a higher score), never create a
    trade that the specialists did not propose."""
    def __init__(self, settings):
        self.s = settings

    def refine(self, d: Decision, regime=None, forecast=None, consistency=None, diversity=None,
               use_regime: bool = True, use_forecast: bool = True) -> Decision:
        s = self.s
        if d.action not in ("BUY", "SELL"):
            return d
        need, veto = s.min_decision_score, []
        notes = d.context_notes
        if use_regime and regime is not None:
            if regime.regime == "UNCERTAIN":
                veto.append("market regime UNCERTAIN")
            elif regime.regime == "TRANSITION":
                need += 10; notes.append("regime TRANSITION: score threshold +10")
            elif regime.regime == "HIGH_VOLATILITY":
                need += 5; notes.append("regime HIGH_VOLATILITY: score threshold +5")
        if use_forecast and forecast is not None:
            fdir = getattr(forecast, "direction", "UNKNOWN")
            unc = getattr(forecast, "uncertainty_label", "UNKNOWN")
            if fdir == "UNKNOWN" and s.require_forecast:
                veto.append("forecast required but unavailable")
            opposite = (fdir == "DOWN" and d.action == "BUY") or (fdir == "UP" and d.action == "SELL")
            if opposite and unc in ("LOW", "MEDIUM"):
                veto.append(f"forecast {fdir} contradicts {d.action} (uncertainty {unc})")
            elif opposite:
                notes.append("forecast contradicts but uncertainty is HIGH: not used as a veto")
        if consistency is not None:
            if consistency.disagreement == "HIGH":
                veto.append("specialists strongly disagree")
            elif consistency.disagreement == "MEDIUM":
                need += 5; notes.append("MEDIUM specialist disagreement: score threshold +5")
        if diversity and diversity.get("agreement") == "HIGH" and diversity.get("diversity") == "LOW":
            need += 5; notes.append("high agreement but LOW diversity (correlated confirmations): score threshold +5")
        if diversity and diversity.get("effective_independent_opinions") is not None:
            notes.append(f"effective independent opinions: {diversity['effective_independent_opinions']}")
        if d.overall_score < need and not veto:
            veto.append(f"overall score {d.overall_score:.1f} < context-adjusted threshold {need:.0f}")
        if veto:
            d.reasons.extend(veto)
            d.action = "HOLD"
        return d
