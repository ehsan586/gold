"""Validator: runs AFTER the decision system and may REJECT a proposed decision. Independent from the
Decision Engine (a second pair of eyes). A passed validation carries a signed token that the Safety Gate
requires, so no decision can reach the Action Interface without having been validated."""
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field

from .settings import ConfigError

OK_STATES = ("READY", "ANALYZING", "EVALUATING")


@dataclass
class ValidationResult:
    passed: bool
    action: str
    snapshot_id: str
    reasons: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    token: str = ""

    def to_dict(self) -> dict:
        return {"passed": self.passed, "action": self.action, "snapshot_id": self.snapshot_id, "reasons": self.reasons, "checks": self.checks}


class Validator:
    def __init__(self, settings):
        self.s = settings
        self._secret = secrets.token_bytes(32)

    def _sig(self, snapshot_id, action, score) -> str:
        return hmac.new(self._secret, f"{snapshot_id}|{action}|{score:.3f}".encode(), hashlib.sha256).hexdigest()

    def verify(self, v: ValidationResult | None, snapshot_id: str, action: str, score: float) -> bool:
        return bool(v and v.passed and v.token and v.snapshot_id == snapshot_id and v.action == action
                    and hmac.compare_digest(v.token, self._sig(snapshot_id, action, score)))

    def validate(self, *, decision, snapshot, quality, consistency, forecast, system_state: str, now: float,
                 usable_specialists: int) -> ValidationResult:
        s = self.s
        res = ValidationResult(False, decision.action, snapshot.snapshot_id if snapshot else "")
        trade = decision.action in ("BUY", "SELL")

        def chk(name, ok, detail=""):
            res.checks.append({"check": name, "ok": bool(ok), "detail": "" if ok else detail})
            if not ok:
                res.reasons.append(f"{name}: {detail}")

        chk("supported_action", decision.action in ("BUY", "SELL", "HOLD"), f"unsupported action {decision.action!r}")
        try:
            s.validate()
            cfg_ok = True
        except ConfigError as e:
            cfg_ok = False
            chk("configuration", False, str(e))
        if cfg_ok:
            chk("configuration", True)
        chk("data_present", snapshot is not None and quality is not None and quality.ok,
            "no valid snapshot" if snapshot is None else "; ".join(quality.fatal) if quality else "no data-quality report")
        if snapshot is not None:
            chk("fresh", now - snapshot.available_at <= max(s.max_tick_age_sec, 2 * s.analysis_interval_sec),
                f"snapshot is {now - snapshot.available_at:.0f}s old")
        if trade:
            chk("system_state", system_state in OK_STATES, f"system state is {system_state}")
            chk("enough_agents", usable_specialists >= 3, f"only {usable_specialists} specialists have usable data")
            chk("score_vs_threshold", decision.overall_score >= s.min_decision_score,
                f"overall score {decision.overall_score:.1f} below threshold {s.min_decision_score}")
            chk("confidence_consistency", decision.confidence >= 40 and not (quality and quality.score < 60 and decision.confidence > 80),
                f"confidence {decision.confidence:.0f} inconsistent with score/data quality")
            bull = decision.action == "BUY"
            chk("no_contradiction_with_consensus",
                not (consistency and consistency.disagreement == "HIGH") and
                not (consistency and consistency.consensus == ("BEARISH" if bull else "BULLISH")),
                f"consensus {consistency.consensus if consistency else '?'} / disagreement {consistency.disagreement if consistency else '?'}")
            fdir = getattr(forecast, "direction", "UNKNOWN")
            chk("no_confident_forecast_contradiction",
                not (fdir == ("DOWN" if bull else "UP") and getattr(forecast, "uncertainty_label", "") in ("LOW", "MEDIUM")),
                f"forecast says {fdir} with {getattr(forecast, 'uncertainty_label', '?')} uncertainty")
            chk("real_account_forbidden", not s.allow_real_account, "allow_real_account is enabled")
        res.passed = not res.reasons
        if res.passed:
            res.token = self._sig(res.snapshot_id, res.action, decision.overall_score)
        return res
