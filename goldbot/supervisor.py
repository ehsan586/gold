"""System-level supervision: State Manager, Module isolation, Recovery Manager, Watchdog, Supervisor.
None of these predict markets. All checks are deterministic (no LLM)."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

STATES = ("STARTING", "READY", "ANALYZING", "EVALUATING", "DEGRADED", "PAUSED", "ERROR", "RECOVERY", "SHUTDOWN")
_ANY = {"DEGRADED", "PAUSED", "ERROR", "SHUTDOWN"}
ALLOWED = {
    "STARTING": {"READY"} | _ANY,
    "READY": {"ANALYZING", "EVALUATING"} | _ANY,
    "ANALYZING": {"READY", "EVALUATING"} | _ANY,
    "EVALUATING": {"READY", "ANALYZING"} | _ANY,
    "DEGRADED": {"READY", "RECOVERY"} | _ANY,
    "PAUSED": {"READY", "RECOVERY"} | _ANY,
    "ERROR": {"RECOVERY", "SHUTDOWN"},
    "RECOVERY": {"READY"} | _ANY,
    "SHUTDOWN": set(),
}
ACTORS = ("supervisor", "engine", "operator", "recovery")
QUIET = {"ANALYZING", "EVALUATING"}        # high-frequency toggles: kept in memory, not written to the event log


class SystemStateManager:
    """Deterministic owner of the system state. Only supervisor/engine/operator/recovery may change it;
    agents never even receive a reference to it."""
    def __init__(self, events, clock):
        self.events, self.clock = events, clock
        self._l = threading.RLock()
        self.state, self.previous = "STARTING", None
        self.last_transition: dict = {}
        self.last_snapshot_id: str | None = None
        self.last_decision: dict | None = None
        self.health: str = "UNKNOWN"
        self.degraded_reasons: list = []
        self.recovering: bool = False

    def transition(self, to: str, reason: str, actor: str) -> bool:
        with self._l:
            if actor not in ACTORS:
                raise PermissionError(f"actor {actor!r} may not change system state")
            if to not in STATES:
                raise ValueError(to)
            if to == self.state:
                return False
            if to not in ALLOWED[self.state]:
                self.events.emit("state_manager", "SYSTEM_STATE", {"rejected": f"{self.state}->{to}", "reason": reason, "actor": actor}, None, self.clock())
                return False
            self.previous, self.state = self.state, to
            self.last_transition = {"from": self.previous, "to": to, "reason": reason, "actor": actor, "ts": self.clock()}
            self.recovering = to == "RECOVERY"
            if to not in QUIET and not (self.previous in QUIET and to == "READY"):
                self.events.emit("state_manager", "SYSTEM_STATE", {"from": self.previous, "to": to, "reason": reason, "actor": actor}, None, self.clock())
            return True

    def snapshot(self) -> dict:
        with self._l:
            return {"state": self.state, "previous": self.previous, "last_transition": dict(self.last_transition),
                    "last_snapshot_id": self.last_snapshot_id, "last_decision": self.last_decision, "health": self.health,
                    "degraded": self.state == "DEGRADED", "degraded_reasons": list(self.degraded_reasons), "recovering": self.recovering}


class ModuleIsolation:
    """A failing module is isolated (skipped, reported unavailable) instead of taking the system down."""
    def __init__(self, settings, events, clock):
        self.s, self.events, self.clock = settings, events, clock
        self.fails: dict[str, int] = {}
        self.until: dict[str, float] = {}
        self.last_error: dict[str, str] = {}

    def call(self, name: str, fn, fallback):
        now = self.clock()
        if self.until.get(name, 0) > now:
            return fallback(f"isolated until +{self.until[name] - now:.0f}s ({self.last_error.get(name, '')})")
        try:
            r = fn()
            self.fails[name] = 0
            return r
        except Exception as e:  # noqa: BLE001
            self.fails[name] = self.fails.get(name, 0) + 1
            self.last_error[name] = f"{type(e).__name__}: {e}"
            if self.fails[name] >= self.s.module_fail_limit:
                self.until[name] = now + self.s.module_cooldown_sec
                self.events.emit("isolation", "MODULE_ISOLATED", {"module": name, "error": self.last_error[name],
                                 "cooldown_sec": self.s.module_cooldown_sec}, None, now)
            return fallback(self.last_error[name])

    def unavailable(self) -> list[str]:
        now = self.clock()
        return [n for n, t in self.until.items() if t > now]

    def status(self) -> dict:
        now = self.clock()
        names = set(self.fails) | set(self.until)
        return {n: {"available": self.until.get(n, 0) <= now, "consecutive_failures": self.fails.get(n, 0),
                    "last_error": self.last_error.get(n, "")} for n in sorted(names)}


class RecoveryManager:
    def __init__(self, events, clock, sleep=time.sleep):
        self.events, self.clock, self.sleep = events, clock, sleep
        self.history: list[dict] = []

    def attempt(self, component: str, fn, retries: int = 3, backoff: float = 0.2):
        """Retry a transient failure. Every attempt is logged; failures are never hidden."""
        err = None
        self.events.emit("recovery", "RECOVERY_STARTED", {"component": component, "retries": retries}, None, self.clock())
        for i in range(retries):
            try:
                r = fn()
                self.events.emit("recovery", "RECOVERY_COMPLETED", {"component": component, "attempt": i + 1, "ok": True}, None, self.clock())
                self.history.append({"component": component, "ok": True, "attempts": i + 1, "ts": self.clock()})
                return True, r
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                self.sleep(backoff * (2 ** i))
        self.events.emit("recovery", "RECOVERY_COMPLETED", {"component": component, "attempts": retries, "ok": False, "error": err}, None, self.clock())
        self.history.append({"component": component, "ok": False, "attempts": retries, "error": err, "ts": self.clock()})
        return False, err


class Watchdog:
    """Deterministic health probes. Each probe returns (status, message) with status OK / WARN / FAIL / NA."""
    def __init__(self, clock):
        self.clock = clock
        self.probes: dict = {}
        self.last: dict = {}

    def register(self, name: str, fn) -> None:
        self.probes[name] = fn

    def run_all(self) -> dict:
        out = {}
        for n, fn in self.probes.items():
            try:
                st, msg = fn()
            except Exception as e:  # noqa: BLE001
                st, msg = "FAIL", f"probe crashed: {type(e).__name__}: {e}"
            out[n] = {"status": st, "message": msg, "ts": self.clock()}
        self.last = out
        return out


@dataclass
class Assessment:
    level: str = "READY"                    # READY / DEGRADED / NOT_READY
    reasons: list = field(default_factory=list)
    unavailable_modules: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"level": self.level, "reasons": self.reasons, "unavailable_modules": self.unavailable_modules}


class SystemSupervisor:
    """Monitors the SYSTEM (not the market). Example from the spec: stale data + >=3 unavailable intelligence
    modules + high forecast uncertainty => NOT_READY, rather than forcing a decision."""
    def assess(self, *, data_ok: bool, data_issues: list, unavailable: list, forecast_uncertainty: str,
               watchdog: dict, config_ok: bool, broker_ok: bool, db_ok: bool) -> Assessment:
        a = Assessment(unavailable_modules=list(unavailable))
        sev = 0
        def bump(level, why):
            nonlocal sev
            sev = max(sev, level)
            a.reasons.append(why)
        if not config_ok: bump(2, "configuration invalid")
        if not db_ok: bump(2, "database unavailable")
        if not broker_ok: bump(2, "broker/data source unavailable")
        if not data_ok: bump(1, "data stale/invalid: " + "; ".join(data_issues[:2]))
        if len(unavailable) >= 3:
            bump(2, f"{len(unavailable)} intelligence modules unavailable: {', '.join(unavailable)}")
        elif unavailable:
            bump(1, f"modules unavailable: {', '.join(unavailable)}")
        if (not data_ok) and len(unavailable) >= 3 and forecast_uncertainty == "HIGH":
            bump(2, "stale data + >=3 modules unavailable + HIGH forecast uncertainty")
        for n, w in watchdog.items():
            if w["status"] == "FAIL" and n in ("database", "data_service"):
                bump(2, f"watchdog {n}: {w['message']}")
            elif w["status"] == "FAIL":
                bump(1, f"watchdog {n}: {w['message']}")
        a.level = ("READY", "DEGRADED", "NOT_READY")[sev]
        return a
