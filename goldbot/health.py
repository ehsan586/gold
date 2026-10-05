"""Health monitor. Any CRITICAL component failure => DEGRADED => no new trades."""
from __future__ import annotations

import threading

CRITICAL = ("broker", "database", "market_data")


class HealthMonitor:
    def __init__(self, clock):
        self.clock = clock
        self._lock = threading.RLock()
        self.c: dict[str, dict] = {}

    def ok(self, name: str, msg: str = "") -> None:
        with self._lock:
            self.c[name] = {"ok": True, "ts": self.clock(), "msg": msg}

    def fail(self, name: str, msg: str) -> None:
        with self._lock:
            self.c[name] = {"ok": False, "ts": self.clock(), "msg": msg}

    def beat(self, name: str, msg: str = "") -> None:      # informational timestamps
        with self._lock:
            prev = self.c.get(name, {"ok": True})
            self.c[name] = {"ok": prev.get("ok", True), "ts": self.clock(), "msg": msg}

    def degraded(self, max_age_analysis: float = 180.0) -> list[str]:
        with self._lock:
            bad = [f"{n}: {v['msg']}" for n, v in self.c.items() if n in CRITICAL and not v["ok"]]
            for n in CRITICAL:
                if n not in self.c:
                    bad.append(f"{n}: no data yet")
            la = self.c.get("analysis")
            if la and self.clock() - la["ts"] > max_age_analysis:
                bad.append("analysis: stale")
            return bad

    def snapshot(self) -> dict:
        with self._lock:
            bad = self.degraded()
            now = self.clock()
            return {"status": "DEGRADED" if bad else "HEALTHY", "problems": bad,
                    "components": {n: {"ok": v["ok"], "age_sec": round(now - v["ts"], 1), "msg": v["msg"]}
                                   for n, v in self.c.items()}}
