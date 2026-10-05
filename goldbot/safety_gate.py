"""Safety Gate: the final deterministic barrier.  Agent -> Decision -> Validator -> Safety Gate -> Action Interface.
No agent, no dashboard button and no learning component can reach the Action Interface any other way:
 * opening risk needs (1) a Validator-signed token for THIS snapshot/decision, (2) an execution mode that allows it,
   (3) approval from the Risk Governor (which issues the only token the Action Interface accepts);
 * risk-REDUCING actions (close / tighten stop / partial close) are always allowed from trusted sources, but are
   still routed through here and logged."""
from __future__ import annotations

import threading

from .risk import RiskDecision, RiskGovernor, TradeRequest

TRUSTED_SOURCES = ("engine", "position_manager", "operator")


class SafetyGate:
    def __init__(self, settings, governor: RiskGovernor, validator, events, mode: str):
        self.s, self.gov, self.validator, self.events, self.mode = settings, governor, validator, events, mode
        self._l = threading.Lock()
        self.counters = {"approved": 0, "blocked": 0, "suppressed_disabled": 0, "reducing_allowed": 0, "reducing_denied": 0}
        self.last: dict = {}

    def authorize(self, req: TradeRequest, validation, decision, **eval_kwargs) -> RiskDecision:
        now = eval_kwargs["now"]
        blocked = RiskDecision()
        why = None
        if not self.validator.verify(validation, validation.snapshot_id if validation else "", req.direction, decision.overall_score):
            why = "no valid Validator approval for this decision"
        elif self.mode == "DISABLED":
            why = "execution mode DISABLED (research only): action logged but not executed"
            with self._l:
                self.counters["suppressed_disabled"] += 1
            self.events.emit("safety_gate", "SIMULATION_ACTION", {"status": "SUPPRESSED_DISABLED", "direction": req.direction,
                             "entry": req.entry, "sl": req.sl, "tp": req.tp}, validation.snapshot_id, now)
        if why:
            blocked.reasons = [why]
            blocked.status = "BLOCKED"
            self._record(False, blocked.reasons, req, now, validation.snapshot_id if validation else None)
            return blocked
        rd = self.gov.evaluate(req, **eval_kwargs)
        self._record(rd.approved, rd.reasons, req, now, validation.snapshot_id)
        return rd

    def _record(self, ok, reasons, req, now, sid) -> None:
        with self._l:
            self.counters["approved" if ok else "blocked"] += 1
            self.last = {"approved": ok, "direction": req.direction, "reasons": reasons, "ts": now}
        self.events.emit("safety_gate", "SAFETY_GATE", {"approved": ok, "direction": req.direction, "reasons": reasons[:5],
                         "mode": self.mode, "command_id": req.command_id}, sid, now)

    def check_reducing(self, action: str, source: str, ticket: int | None = None, now: float | None = None) -> bool:
        ok = source in TRUSTED_SOURCES and action in ("CLOSE", "MODIFY_SL", "PARTIAL", "MODIFY_TP")
        with self._l:
            self.counters["reducing_allowed" if ok else "reducing_denied"] += 1
        self.events.emit("safety_gate", "SAFETY_GATE", {"reducing_action": action, "source": source, "ticket": ticket, "allowed": ok}, None, now)
        return ok

    def status(self) -> dict:
        with self._l:
            return {"execution_mode": self.mode, "counters": dict(self.counters), "last": dict(self.last),
                    "rule": "Agent > Decision > Validator > Safety Gate > Action Interface"}
