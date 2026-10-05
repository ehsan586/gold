"""System state machine. Every transition is validated and logged.

EMERGENCY_STOP is reachable from every state and can only be left through
reset_emergency(), which requires an explicit human acknowledgement.

PAUSE vs EMERGENCY STOP
  * PAUSE: no NEW trades; open positions keep being managed by risk rules.
  * EMERGENCY_STOP: no new trades, no reversals; learning/logging continue.
"""
from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Optional

log = logging.getLogger("goldbot.state")


class IllegalTransition(RuntimeError):
    pass


class State(str, Enum):
    IDLE = "IDLE"
    ANALYZING = "ANALYZING"
    SIGNAL_READY = "SIGNAL_READY"
    RISK_CHECK = "RISK_CHECK"
    EXECUTING = "EXECUTING"
    IN_POSITION = "IN_POSITION"
    MANAGING = "MANAGING"
    REVERSAL = "REVERSAL"
    EXITING = "EXITING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    EMERGENCY_STOP = "EMERGENCY_STOP"


S = State
ALLOWED: dict[State, set[State]] = {
    S.IDLE:          {S.ANALYZING, S.PAUSED, S.ERROR, S.IN_POSITION},  # IN_POSITION = recovered/adopted position
    S.ANALYZING:     {S.SIGNAL_READY, S.IDLE, S.PAUSED, S.ERROR},
    S.SIGNAL_READY:  {S.RISK_CHECK, S.IDLE, S.ERROR},
    S.RISK_CHECK:    {S.EXECUTING, S.IDLE, S.ERROR},
    S.EXECUTING:     {S.IN_POSITION, S.IDLE, S.ERROR},
    S.IN_POSITION:   {S.MANAGING, S.REVERSAL, S.EXITING, S.ERROR},
    S.MANAGING:      {S.IN_POSITION, S.REVERSAL, S.EXITING, S.ERROR},
    S.REVERSAL:      {S.EXITING, S.IN_POSITION, S.ERROR},
    S.EXITING:       {S.IDLE, S.RISK_CHECK, S.ERROR},   # RISK_CHECK = re-check after a reversal exit
    S.PAUSED:        {S.IDLE, S.ERROR},
    S.ERROR:         {S.IDLE},
    S.EMERGENCY_STOP: set(),                              # only reset_emergency() leaves
}

POSITION_STATES = {S.IN_POSITION, S.MANAGING, S.REVERSAL, S.EXITING}


class StateMachine:
    def __init__(self, db=None, initial: State = State.IDLE):
        self._db = db
        self._state = initial
        self._lock = threading.RLock()
        self._paused = False           # pause flag also applies while a position is open

    @property
    def state(self) -> State:
        return self._state

    # ---- core -------------------------------------------------------------
    def transition(self, to: State, reason: str) -> State:
        to = State(to)
        with self._lock:
            frm = self._state
            if to is State.EMERGENCY_STOP:
                pass  # always allowed
            elif to not in ALLOWED[frm]:
                self._log("WARN", "illegal_transition", frm, to, reason)
                raise IllegalTransition(f"{frm.value} -> {to.value} is not allowed")
            self._state = to
            self._log("CRITICAL" if to in (State.EMERGENCY_STOP, State.ERROR) else "INFO",
                      "transition", frm, to, reason)
            return to

    def _log(self, level: str, event: str, frm: State, to: State, reason: str) -> None:
        msg = f"{frm.value} -> {to.value}: {reason}"
        log.log(logging.INFO if level == "INFO" else logging.WARNING, msg)
        if self._db is not None:
            self._db.log_event(level, "state_machine", event, msg,
                               {"from": frm.value, "to": to.value, "reason": reason})

    # ---- operator controls ------------------------------------------------
    def emergency_stop(self, reason: str) -> None:
        self.transition(State.EMERGENCY_STOP, reason)

    def reset_emergency(self, reviewed_by: str, note: str) -> None:
        if not reviewed_by.strip() or not note.strip():
            raise ValueError("reset requires reviewed_by and a note")
        with self._lock:
            if self._state is not State.EMERGENCY_STOP:
                raise IllegalTransition("not in EMERGENCY_STOP")
            self._state = State.IDLE
            self._paused = False
            self._log("CRITICAL", "emergency_reset", State.EMERGENCY_STOP, State.IDLE,
                      f"reset by {reviewed_by}: {note}")

    def pause(self, reason: str) -> None:
        with self._lock:
            self._paused = True
            if self._state in (State.IDLE, State.ANALYZING):
                self.transition(State.PAUSED, reason)
            elif self._db is not None:
                self._db.log_event("INFO", "state_machine", "pause_flag", reason)

    def resume(self, reason: str) -> None:
        with self._lock:
            self._paused = False
            if self._state is State.PAUSED:
                self.transition(State.IDLE, reason)

    # ---- queries ----------------------------------------------------------
    @property
    def paused(self) -> bool:
        return self._paused

    def new_trades_allowed(self) -> bool:
        """False in PAUSED/ERROR/EMERGENCY_STOP or while the pause flag is set."""
        return (not self._paused) and self._state not in (
            State.PAUSED, State.ERROR, State.EMERGENCY_STOP)

    def reversals_allowed(self) -> bool:
        return self._state is not State.EMERGENCY_STOP and not self._paused

    def has_position_state(self) -> bool:
        return self._state in POSITION_STATES
