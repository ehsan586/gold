"""Two-level memory.
RAW EVENT MEMORY  = the hash-chained `events` table (everything that happened, in order).
EXPERIENCE MEMORY = one structured row per analysed market bar (regime, context, agent outputs, forecast,
uncertainty, state, decision) that is later completed with the observed outcome and prediction error.
Aggregates are computed in SQL, so analysis never needs the whole raw history in RAM."""
from __future__ import annotations

import json

from .data import iso


class ExperienceMemory:
    def __init__(self, db):
        self.db = db

    def record(self, *, now: float, snapshot_id: str, symbol: str, regime: dict, agents: dict, forecast: dict,
               uncertainty: str, system_state: str, decision: str, feature_version: str, model_version: str) -> bool:
        if self.db.query("SELECT 1 FROM experiences WHERE snapshot_id=?", (snapshot_id,)):
            return False
        self.db.insert("experiences", {
            "ts": iso(now), "snapshot_id": snapshot_id, "symbol": symbol, "regime": regime.get("regime"),
            "regime_confidence": regime.get("confidence"), "context_json": json.dumps(regime.get("context", {}), sort_keys=True),
            "agents_json": json.dumps(agents, sort_keys=True), "forecast_json": json.dumps(forecast, sort_keys=True),
            "uncertainty": uncertainty, "system_state": system_state, "decision": decision,
            "feature_version": feature_version, "model_version": model_version})
        return True

    def resolve(self, snapshot_id: str, outcome: dict, prediction_error: float | None, now: float) -> None:
        self.db.execute("UPDATE experiences SET outcome_json=?, prediction_error=?, resolved_ts=? WHERE snapshot_id=?",
                        (json.dumps(outcome, sort_keys=True), prediction_error, iso(now), snapshot_id))

    def summary(self) -> dict:
        q = self.db.query
        tot = q("SELECT COUNT(*) c, SUM(outcome_json IS NOT NULL) r FROM experiences")[0]
        by = q("SELECT regime, COUNT(*) n, SUM(outcome_json IS NOT NULL) resolved, AVG(prediction_error) avg_err "
               "FROM experiences GROUP BY regime ORDER BY n DESC")
        return {"experiences": tot["c"], "resolved": tot["r"] or 0, "by_regime": by,
                "raw_events": q("SELECT COUNT(*) c FROM events")[0]["c"]}

    def recent(self, n: int = 20, regime: str | None = None) -> list[dict]:
        sql = "SELECT ts,snapshot_id,regime,regime_confidence,uncertainty,decision,prediction_error,outcome_json FROM experiences"
        if regime:
            return self.db.query(sql + " WHERE regime=? ORDER BY id DESC LIMIT ?", (regime, n))
        return self.db.query(sql + " ORDER BY id DESC LIMIT ?", (n,))
