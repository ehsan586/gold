"""Event sourcing / audit trail. Append-only, hash-chained (tamper-evident, deterministic):
event_id = hash(prev_hash | ts | component | type | payload). The chain digest lets two replays be
compared byte-for-byte."""
from __future__ import annotations

import hashlib
import json
import threading

from .data import iso

EVENT_VERSION = "1"
TYPES = ("MARKET_SNAPSHOT_RECEIVED", "SNAPSHOT_REJECTED", "AGENT_ANALYSIS_COMPLETED", "REGIME_CLASSIFIED",
         "FORECAST_GENERATED", "FORECAST_RESOLVED", "CONSISTENCY_UPDATED", "DECISION_CREATED",
         "VALIDATION_PASSED", "VALIDATION_REJECTED", "SAFETY_GATE", "SIMULATION_ACTION", "MODEL_EVALUATED",
         "DRIFT_DETECTED", "RECOVERY_STARTED", "RECOVERY_COMPLETED", "SYSTEM_STATE", "MODULE_ISOLATED",
         "EXPERIMENT", "MODEL_STATUS", "STARTUP")


class EventLog:
    def __init__(self, db, clock=None):
        self.db, self.clock = db, clock
        self._lock = threading.RLock()
        r = db.query("SELECT hash FROM events ORDER BY seq DESC LIMIT 1")
        self._prev = r[0]["hash"] if r else "GENESIS"

    def emit(self, component: str, event_type: str, payload: dict | None = None,
             snapshot_id: str | None = None, ts: float | None = None) -> str:
        ts_s = iso(ts if ts is not None else (self.clock() if self.clock else 0.0))
        body = json.dumps(payload or {}, sort_keys=True, default=str)
        with self._lock:
            h = hashlib.sha256(f"{self._prev}|{ts_s}|{component}|{event_type}|{body}".encode()).hexdigest()
            self.db.insert("events", {"event_id": h[:24], "ts": ts_s, "snapshot_id": snapshot_id, "component": component,
                                      "event_type": event_type, "payload_json": body, "version": EVENT_VERSION,
                                      "prev_hash": self._prev, "hash": h})
            self._prev = h
            return h[:24]

    def digest(self) -> str:
        return self._prev

    def tail(self, n: int = 100, event_type: str | None = None) -> list[dict]:
        if event_type:
            return self.db.query("SELECT seq,event_id,ts,snapshot_id,component,event_type,payload_json,version FROM events "
                                 "WHERE event_type=? ORDER BY seq DESC LIMIT ?", (event_type, n))
        return self.db.query("SELECT seq,event_id,ts,snapshot_id,component,event_type,payload_json,version FROM events "
                             "ORDER BY seq DESC LIMIT ?", (n,))

    def verify_chain(self) -> tuple[bool, int]:
        prev, n = "GENESIS", 0
        for r in self.db.query("SELECT ts,component,event_type,payload_json,prev_hash,hash FROM events ORDER BY seq"):
            h = hashlib.sha256(f"{prev}|{r['ts']}|{r['component']}|{r['event_type']}|{r['payload_json']}".encode()).hexdigest()
            if r["prev_hash"] != prev or r["hash"] != h:
                return False, n
            prev, n = h, n + 1
        return True, n
