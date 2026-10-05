"""ASTRA trading engine - orchestrates the whole pipeline of ONE decision cycle:

 DataService -> DataValidator -> VALIDATED SNAPSHOT -> 7 specialists + Regime + Forecast (same snapshot for all)
   -> Diversity -> Consistency -> Decision (weighted, then context-refined) -> Validator -> Safety Gate -> Action Interface

CHECK != TRADE: step() runs every second (positions, health, protection); analysis is recomputed every
`analysis_interval_sec`; a trade needs a persistent, validated, safety-gated signal. Dashboard commands are queued
and executed inside the engine thread; they never touch the Action Interface directly.
"""
from __future__ import annotations

import copy
import json
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path

from .agents import AgentContext, CalendarFile, default_agents
from .broker import Broker
from .consistency import SPECIALISTS, ConsistencyEngine
from .data import TF_SECONDS, iso
from .decision import DEFAULT_WEIGHTS, ContextualDecision, Decision, DecisionEngine
from .diversity import DiversityMonitor
from .drift import DriftDetector
from .events import EventLog
from .execution import ExecutionEngine
from .experiments import ExperimentLab
from .forecast import ForecastResult, ForecastSystem, ModelRegistry, forecast_features
from .health import HealthMonitor
from .indicators import atr
from .learning import LearningEngine
from .learning_manager import LearningManager
from .llm import LLMAdvisor
from .memory import ExperienceMemory
from .metrics import Metrics
from .models import AgentResult
from .position_manager import PositionManager, ReversalPolicy
from .regime import REGIME_FEATURE_VERSION, RegimeDetector, RegimeResult
from .risk import EquityTracker, RiskDecision, RiskGovernor, TradeRequest, plan_sl_tp, trade_stats
from .safety_gate import SafetyGate
from .settings import Settings
from .snapshot import DataQuality, DataService, DataValidator
from .state_machine import IllegalTransition, State, StateMachine
from .supervisor import ModuleIsolation, RecoveryManager, SystemStateManager, SystemSupervisor, Watchdog
from .units import check_spec, detect_account_money, split_symbol
from .validator import Validator

POS_STATES = (State.IN_POSITION, State.MANAGING, State.REVERSAL, State.EXITING)
CARD_ORDER = ("trend", "structure", "price_action", "momentum", "volatility", "liquidity", "news", "regime", "forecast")
DASH = Path(__file__).with_name("dashboard.html")


class TradingEngine:
    def __init__(self, settings: Settings, db, broker: Broker, clock=None, news: CalendarFile | None = None,
                 llm: LLMAdvisor | None = None, sim: bool = False, execution_mode: str | None = None):
        self.s, self.db, self.broker, self.sim = settings, db, broker, sim
        self.clock = clock or time.time
        self.mode = execution_mode or settings.execution_mode
        # ---- observability / audit -------------------------------------------------
        self.events = EventLog(db, self.clock)
        self.metrics = Metrics()
        self.sysstate = SystemStateManager(self.events, self.clock)
        self.isolation = ModuleIsolation(settings, self.events, self.clock)
        self.recovery = RecoveryManager(self.events, self.clock)
        self.watchdog = Watchdog(self.clock)
        self.supervisor = SystemSupervisor()
        self.health = HealthMonitor(self.clock)
        # ---- deterministic core ------------------------------------------------------
        self.sm = StateMachine(db)
        self.dataservice = DataService(broker, settings)
        self.data_validator = DataValidator(settings, strict=(getattr(broker, "name", "") == "paper"))
        self.learning = LearningEngine(db, settings)
        self.version = self.learning.active_version()
        self.weights = {**DEFAULT_WEIGHTS, **self.learning.load_params(self.version).get("weights", {})}
        self.decider = DecisionEngine(settings, self.weights)
        self.contextual = ContextualDecision(settings)
        self.consistency = ConsistencyEngine(settings, self.weights)
        self.diversity = DiversityMonitor()
        self.validator = Validator(settings)
        self.risk = RiskGovernor(settings, db)
        self.gate = SafetyGate(settings, self.risk, self.validator, self.events, self.mode)
        self.exec = ExecutionEngine(broker, db, settings, self.risk, self.clock, events=self.events, gate=self.gate, mode=self.mode)
        self.pm = PositionManager(settings, db)
        self.reversal = ReversalPolicy(settings)
        self.tracker = EquityTracker(db)
        # ---- intelligence ----------------------------------------------------------------
        self.agents = default_agents()
        self.news = news if news is not None else CalendarFile(settings.news_file)
        self.llm = llm or LLMAdvisor()
        self.regime = RegimeDetector(settings)
        self.memory = ExperienceMemory(db)
        self.drift = DriftDetector(settings, self.events)
        self.registry = ModelRegistry(db, settings, self.events)
        self.forecaster = ForecastSystem(db, settings, self.registry, self.events, on_resolved=self._on_forecast_resolved)
        self.lab = ExperimentLab(db, settings, self.events)
        self.learning_mgr = LearningManager(db, settings, self.events, self.registry, self.forecaster, self.drift,
                                            self.learning, self.lab, self.metrics)
        # ---- experiment switches (ablation) -------------------------------------------------
        self.disabled_agents: set = set()
        self.use_regime = self.use_forecast = True
        self.periodic_eval = True
        self.publish_enabled = True
        # ---- runtime ----------------------------------------------------------------------------
        self.symbol, self.spec = settings.symbol, None
        self._cmds: queue.Queue = queue.Queue()
        self._lock = threading.RLock()
        self._snap: dict = {}
        self._snapshot = None
        self._quality: DataQuality | None = None
        self._results: dict = {}
        self._regime = RegimeResult("UNCERTAIN", 0.0, 0.0, settings.regime_timeframe, REGIME_FEATURE_VERSION, {"reason": "not analysed yet"})
        self._forecast = ForecastResult()
        self._diversity: dict = {}
        self._consistency = None
        self._decision = Decision()
        self._validation = None
        self._assessment = None
        self._decision_id: int | None = None
        self._streak = {"dir": None, "n": 0}
        self._logged_action: str | None = None
        self._last_decision_log = -1e18
        self._last_fbar = None
        self._last_analysis = -1e18
        self._last_wd = -1e18
        self._last_learn_eval = self.clock() if not sim else 0.0
        self._ctx: dict = {}
        self._last_rd: RiskDecision | None = None
        self._risk_log_seen: dict[str, float] = {}
        self._reject_seen: dict[str, float] = {}
        self._suppress_until = -1e18
        self._missing: dict[int, int] = {}
        self._broker_fail = 0
        self._acct = self._tick = None
        self._eq: dict = {"peak": 0, "day_start": 0, "drawdown_pct": 0, "daily_loss_pct": 0, "daily_pnl": 0}
        self._recent: deque = deque(maxlen=8)
        self.last_error = ""
        self.config_ok = False
        self.unit_warnings: list = []
        self.money = None
        self.replay_status: dict | None = None
        self.api_alive = None
        self.last_dashboard_poll: float | None = None

    # ================================================================== startup
    def start(self) -> None:
        self.broker.connect()
        acct = self.broker.account()
        if not acct.is_demo and not self.s.allow_real_account:
            raise RuntimeError("REFUSING TO START: connected account is NOT a demo account (demo/research only)")
        if self.mode not in ("DISABLED", "SIMULATION"):
            raise RuntimeError("final build is non-executing: use DISABLED or SIMULATION")
        self.s.validate()
        self.config_ok = True
        self.symbol = self.broker.find_symbol(self.s.symbol)
        self.spec = self.broker.symbol_spec(self.symbol)
        self.unit_warnings = check_spec(self.spec, acct.currency)
        if any("invalid" in w for w in self.unit_warnings):
            raise RuntimeError(f"bad symbol specification from broker: {self.unit_warnings}")
        self.money = detect_account_money(acct.currency, self.s.account_denomination)
        self.registry.ensure_defaults()
        self.forecaster.reload()
        base, suffix = split_symbol(self.symbol)
        self.db.log_event("INFO", "engine", "startup", f"{self.symbol} mode={self.mode} account={acct.margin_mode} leverage=1:{acct.leverage} version={self.version}",
                          {"spec": self.spec.__dict__, "mode": self.mode})
        self.events.emit("engine", "STARTUP", {"symbol": self.symbol, "base": base, "suffix": suffix, "execution_mode": self.mode,
                         "account_currency": acct.currency, "denomination": self.money.denomination.value, "unit_warnings": self.unit_warnings,
                         "feature_version": self.s.feature_version, "strategy_version": self.version}, None, self.clock())
        if self.db.kv_get("emergency") == "1":
            self.sm.emergency_stop("restored: emergency stop was active before restart")
        self.sysstate.transition("READY", "startup checks passed", "engine")
        if self.db.kv_get("paused") == "1":
            self.sm.pause("restored: pause was active before restart")
            self.sysstate.transition("PAUSED", "restored pause", "engine")
        self._register_probes()

    def _register_probes(self) -> None:
        w = self.watchdog
        def data_service():
            if self._quality is None: return "NA", "no snapshot yet"
            return ("OK", f"quality {self._quality.score:.0f}") if self._quality.ok else ("FAIL", "; ".join(self._quality.fatal[:2]))
        def backend():
            if self.api_alive is None: return "NA", "API not started in this process"
            return ("OK", "api thread alive") if self.api_alive() else ("FAIL", "api thread not alive")
        def database():
            self.db.query("SELECT 1")
            return "OK", f"{self.db.count('events')} events"
        def frontend():
            if not DASH.exists(): return "FAIL", "dashboard file missing"
            if self.last_dashboard_poll is None: return "NA", "no dashboard client yet"
            age = time.time() - self.last_dashboard_poll
            return ("OK", f"last poll {age:.0f}s ago") if age < 30 else ("WARN", f"no client for {age:.0f}s")
        def agents():
            n = len(self.isolation.unavailable())
            return ("OK", "all agents available") if n == 0 else ("WARN" if n < 3 else "FAIL", f"{n} isolated")
        def memory():
            self.db.query("SELECT COUNT(*) FROM experiences"); return "OK", f"{self.db.count('experiences')} experiences"
        def learning():
            if self.learning_mgr.last_error: return "FAIL", self.learning_mgr.last_error
            return "OK", "evaluation ok" if self.learning_mgr.last_eval else "no evaluation yet"
        def replay():
            if self.replay_status is None: return "NA", "selftest not run"
            return ("OK", "deterministic, no look-ahead") if self.replay_status.get("ok") else ("WARN", str(self.replay_status.get("error") or self.replay_status))
        for n, f in (("data_service", data_service), ("backend", backend), ("database", database), ("frontend", frontend),
                     ("agents", agents), ("memory", memory), ("learning", learning), ("replay", replay)):
            w.register(n, f)

    def run_replay_selftest(self) -> dict:
        from .replay import selftest
        self.replay_status = selftest(self.s)
        return self.replay_status

    def _on_forecast_resolved(self, sid: str, outcome: dict, err: float, ts: float) -> None:
        self.memory.resolve(sid, outcome, err, ts)
        self.drift.observe_error(err)

    # ================================================================== commands (thread-safe)
    def submit(self, name: str, params: dict | None = None, timeout: float = 15.0) -> dict:
        fut: Future = Future()
        self._cmds.put((name, params or {}, fut))
        try:
            return fut.result(timeout=timeout)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"engine did not answer: {type(e).__name__}"}

    def _process_commands(self, now: float) -> None:
        while True:
            try:
                name, p, fut = self._cmds.get_nowait()
            except queue.Empty:
                return
            try:
                fut.set_result(self._run_command(name, p, now))
            except Exception as e:  # noqa: BLE001
                fut.set_result({"ok": False, "error": str(e)})

    def _run_command(self, name: str, p: dict, now: float) -> dict:
        self.db.log_event("INFO", "command", name, json.dumps({k: v for k, v in p.items() if k != "token"}))
        if name == "pause":
            self.sm.pause("dashboard pause")
            self.db.kv_set("paused", "1")
            self.sysstate.transition("PAUSED", "operator pause", "operator")
            return {"ok": True, "state": self.sm.state.value}
        if name == "resume":
            if self.sm.state is State.EMERGENCY_STOP:
                return {"ok": False, "error": "emergency stop active: reset it first"}
            self.sm.resume("dashboard resume")
            self.db.kv_delete("paused")
            self.sysstate.transition("READY", "operator resume", "operator")
            return {"ok": True, "state": self.sm.state.value}
        if name == "emergency_stop":
            self._emergency("manual (dashboard)")
            return {"ok": True, "state": self.sm.state.value}
        if name == "emergency_reset":
            self.sm.reset_emergency(str(p.get("reviewed_by", "")), str(p.get("note", "")))
            self.db.kv_delete("emergency")
            if p.get("reset_peak") and self._acct:
                self.tracker.reset_peak(self._acct.equity)
            return {"ok": True, "state": self.sm.state.value}
        if name == "close_position":
            n = 0
            for pos in self._my_positions():
                r = self.exec.close_position(pos.ticket, "MANUAL_DASHBOARD", source="operator")
                n += 1 if r.ok else 0
            self._reconcile(self._my_positions(), now)
            return {"ok": True, "closed": n}
        if name == "reload_strategy":
            if self.sm.has_position_state():
                return {"ok": False, "error": "close the open position before reloading the strategy"}
            self.version = self.learning.active_version()
            self.weights = {**DEFAULT_WEIGHTS, **self.learning.load_params(self.version).get("weights", {})}
            self.decider = DecisionEngine(self.s, self.weights)
            self.consistency = ConsistencyEngine(self.s, self.weights)
            return {"ok": True, "version": self.version}
        if name == "reload_models":
            self.forecaster.reload()
            return {"ok": True}
        if name == "promote_model":
            self.learning_mgr.promote(str(p["model_id"]), str(p["version"]), str(p.get("approver", "")), str(p.get("evidence", "")))
            return {"ok": True}
        if name == "reject_model":
            self.learning_mgr.reject(str(p["model_id"]), str(p["version"]), str(p.get("reason", "manual")))
            return {"ok": True}
        if name == "rollback_model":
            return {"ok": True, "active_version": self.learning_mgr.rollback(str(p["model_id"]), str(p.get("reason", "manual")))}
        if name == "learning_evaluate":
            self.learning_mgr.evaluate(now)
            return {"ok": True}
        return {"ok": False, "error": f"unknown command {name}"}

    def _emergency(self, why: str) -> None:
        self.db.kv_set("emergency", "1")
        if self.sm.state is not State.EMERGENCY_STOP:
            self.sm.emergency_stop(why)

    # ================================================================== helpers
    def _my_positions(self):
        return [p for p in self.broker.positions(self.symbol) if p.magic == self.s.magic_number]

    def _go(self, state: State, why: str) -> bool:
        if self.sm.state is state:
            return True
        try:
            self.sm.transition(state, why)
            return True
        except IllegalTransition:
            return False

    def _holding_since(self, ticket: int, now: float) -> float:
        r = self.db.query("SELECT ts_open FROM trades WHERE ticket=? AND status='OPEN' ORDER BY id DESC LIMIT 1", (ticket,))
        return now - datetime.fromisoformat(r[0]["ts_open"]).timestamp() if r else 1e9

    # ================================================================== main cycle
    def step(self, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        self._process_commands(now)
        if self.sysstate.state == "ERROR":
            self.sysstate.transition("RECOVERY", "next cycle after error", "recovery")
            self.events.emit("recovery", "RECOVERY_STARTED", {"component": "engine", "reason": self.last_error[:120]}, None, now)
        try:
            acct = self.broker.account()
            tick = self.broker.tick(self.symbol)
            self.health.ok("broker", "connected")
            self._broker_fail = 0
            age = now - tick.time
            if age > self.s.max_tick_age_sec:
                self.health.fail("market_data", f"last price update {age:.0f}s ago")
            else:
                self.health.ok("market_data", "fresh")
        except Exception as e:  # noqa: BLE001
            self.health.fail("broker", f"{type(e).__name__}: {e}")
            self.last_error = str(e)
            self._broker_fail += 1
            if not self.sim and self._broker_fail in (3, 30, 300):
                self.recovery.attempt("broker_reconnect", self.broker.connect, retries=2, backoff=0.5)
            self.sysstate.transition("DEGRADED", f"broker/data failure: {e}"[:150], "engine")
            self._enter_error("broker/data failure")
            self._publish(now)
            return
        try:
            self.db.query("SELECT 1")
            self.health.ok("database", "ok")
        except Exception as e:  # noqa: BLE001
            self.health.fail("database", str(e))
            self.sysstate.transition("DEGRADED", "database unavailable", "engine")
            self._publish(now)
            return
        self._acct, self._tick = acct, tick
        mark = getattr(self.broker, "mark", None)
        if mark:
            mark(tick, now)                               # virtual account evaluates SL/TP against the live tick
            acct = self.broker.account()
            self._acct = acct
        self._eq = self.tracker.update(acct.equity, now)
        if self._eq["drawdown_pct"] >= self.s.max_account_drawdown and self.sm.state is not State.EMERGENCY_STOP:
            self._emergency(f"account drawdown {self._eq['drawdown_pct']:.2f}% >= {self.s.max_account_drawdown}%")
        positions = self._my_positions()
        self._reconcile(positions, now)
        positions = self._my_positions()
        self._sync_state(positions)
        if positions:
            self._manage(positions, tick, now)
            positions = self._my_positions()
            self._sync_state(positions)
        if self.sim or now - self._last_analysis >= self.s.analysis_interval_sec:
            self._analyze(now, tick, acct, positions)
            self._last_analysis = now
            if not self.health.degraded() and self._assessment and self._assessment.level == "READY":
                self._act_on_decision(positions, tick, acct, now)
        if now - self._last_wd >= 15:
            self.watchdog.run_all()
            self._last_wd = now
        if self.periodic_eval and not self.sim and now - self._last_learn_eval >= self.s.learning_eval_interval_sec:
            self._evaluate_learning(now)
        elif self.periodic_eval and self.sim and now - self._last_learn_eval >= self.s.learning_eval_interval_sec * 4:
            self._evaluate_learning(now)
        if self.sysstate.state == "RECOVERY" and self._assessment and self._assessment.level == "READY":
            self.sysstate.transition("READY", "recovery complete", "recovery")
            self.events.emit("recovery", "RECOVERY_COMPLETED", {"component": "engine", "ok": True}, None, now)
        self._publish(now)

    def _evaluate_learning(self, now: float) -> None:
        self._last_learn_eval = now
        prev = self.sysstate.state
        self.sysstate.transition("EVALUATING", "periodic learning evaluation", "engine")
        try:
            with self.metrics.timer("learning_eval_ms"):
                self.learning_mgr.evaluate(now)
        except Exception as e:  # noqa: BLE001
            self.db.log_event("ERROR", "learning", "evaluate_failed", f"{type(e).__name__}: {e}")
        finally:
            if self.sysstate.state == "EVALUATING":
                self.sysstate.transition("READY" if prev in ("READY", "ANALYZING") else prev, "evaluation done", "engine")

    # ---------------------------------------------------------------- state sync
    def _enter_error(self, why: str) -> None:
        if self.sm.state not in POS_STATES and self.sm.state not in (State.ERROR, State.EMERGENCY_STOP):
            self._go(State.ERROR, why)

    def _sync_state(self, positions) -> None:
        st = self.sm.state
        bad = self.health.degraded()
        if st is State.ERROR and not bad:
            self._go(State.IDLE, "health restored")
            st = self.sm.state
        if st in (State.EMERGENCY_STOP, State.ERROR):
            return
        if st is State.ANALYZING:
            self._go(State.IDLE, "cycle end")
            st = self.sm.state
        if positions and st is State.IDLE:
            self._go(State.IN_POSITION, "position present (opened or recovered)")
        elif positions and st is State.EXITING:
            self._go(State.IN_POSITION, "exit incomplete, position still open")
        elif not positions and st in (State.IN_POSITION, State.MANAGING, State.REVERSAL):
            self._go(State.EXITING, "position closed")
            self._go(State.IDLE, "flat")
        elif not positions and st is State.EXITING:
            self._go(State.IDLE, "flat")
        if self.health.degraded() and not positions:
            self._enter_error("; ".join(self.health.degraded()))

    # ---------------------------------------------------------------- reconcile broker <-> journal
    def _reconcile(self, positions, now: float) -> None:
        rows = self.db.query("SELECT * FROM trades WHERE status='OPEN' AND symbol=?", (self.symbol,))
        live = {p.ticket for p in positions}
        for row in rows:
            if row["ticket"] in live:
                continue
            summ = self.broker.deal_summary(row["ticket"])
            if summ is None:
                n = self._missing[row["ticket"]] = self._missing.get(row["ticket"], 0) + 1
                if n >= 5:
                    self.db.update("trades", row["id"], {"status": "CLOSED", "ts_close": iso(now), "exit_reason": "UNKNOWN"})
                    self.db.log_event("ERROR", "reconcile", "close_unknown", f"ticket {row['ticket']} vanished without deal history")
                    self.pm.forget(row["ticket"])
                continue
            self._finalize(row, summ, now)
        known = {r["ticket"] for r in self.db.query("SELECT ticket FROM trades WHERE status='OPEN'")}
        for p in positions:
            if p.ticket not in known:
                self._adopt(p, now)

    def _finalize(self, row: dict, summ: dict, now: float) -> None:
        tk = row["ticket"]
        st = self.pm.forget(tk)
        reason = self.db.kv_get(f"exit_reason:{tk}")
        if reason:
            self.db.kv_delete(f"exit_reason:{tk}")
        else:
            reason = {"SL": "STOP_LOSS", "TP": "TAKE_PROFIT"}.get(summ.get("reason"), "BROKER_CLOSE")
        risk0 = st.initial_risk if st and st.initial_risk > 0 else 0.0
        pnl = float(summ["pnl"])
        self.db.update("trades", row["id"], {
            "status": "CLOSED", "ts_close": iso(now), "exit_price": summ["exit_price"], "pnl": pnl,
            "r_multiple": round(pnl / risk0, 3) if risk0 else None,
            "duration_sec": now - datetime.fromisoformat(row["ts_open"]).timestamp(),
            "mfe": st.peak_profit if st else None, "mae": st.worst_profit if st else None, "exit_reason": reason})
        if reason == "REVERSAL":
            self.db.kv_set("last_reversal_ts", repr(now))
        self.db.log_event("INFO", "trade", "closed", f"{row['direction']} ticket {tk} closed @ {summ['exit_price']} pnl {pnl:+.2f} ({reason})",
                          {"trade_id": row["id"], "pnl": pnl, "reason": reason})
        self.learning.on_trade_closed(row["id"])
        self.events.emit("simulation" if getattr(self.broker, "name", "") in ("paper", "shadow") else "execution", "SIMULATION_ACTION",
                         {"action": "TRADE_RESULT", "pnl": round(pnl, 2), "r": round(pnl / risk0, 3) if risk0 else None, "reason": reason,
                          "mode": self.mode}, None, now)

    def _adopt(self, p, now: float) -> None:
        """A position with our magic number that the journal does not know (e.g. after a crash)."""
        self.db.log_event("WARN", "reconcile", "adopt_position", f"adopting unknown position {p.ticket} {p.direction} {p.volume}")
        tid = self.db.insert("trades", {"ticket": p.ticket, "status": "OPEN", "ts_open": iso(now), "symbol": p.symbol,
                                        "direction": p.direction, "entry_price": p.price_open, "lot": p.volume,
                                        "sl": p.sl, "tp": p.tp, "strategy_version": self.version,
                                        "agent_outputs_json": "{}", "exit_reason": None})
        if not p.sl and self._ctx.get("atr"):
            sl, _ = plan_sl_tp(p.direction, p.price_open, self._ctx["atr"], None, None, self.s, self.spec)
            self.exec.modify_sl(p.ticket, sl, "emergency SL for adopted position")
        self.pm.register(next((q for q in self._my_positions() if q.ticket == p.ticket), p), self.spec)
        self.db.update("trades", tid, {"strategy_version": self.version})

    # ---------------------------------------------------------------- position management
    def _manage(self, positions, tick, now: float) -> None:
        atr_v = self._ctx.get("atr", 0.0)
        for p in positions:
            for a in self.pm.evaluate(p, self.spec, tick.bid, tick.ask, atr_v):
                if a.kind == "CLOSE":
                    self._go(State.EXITING, a.reason)
                    self.exec.close_position(p.ticket, a.reason.split(" (")[0].upper().replace(" ", "_"), source="position_manager")
                    self._reconcile(self._my_positions(), now)
                    return
                self._go(State.MANAGING, a.reason)
                if a.kind == "MODIFY_SL":
                    self.exec.modify_sl(p.ticket, a.price, a.reason, source="position_manager")
                elif a.kind == "PARTIAL":
                    self.exec.partial_close(p.ticket, a.volume, a.reason, source="position_manager")
                self._go(State.IN_POSITION, "management done")

    # ================================================================== the analysis stack (one snapshot, one cycle)
    def run_stack(self, snap, quality: DataQuality, now: float, record: bool = True) -> dict:
        s = self.s
        ctx = AgentContext(snap.symbol, {tf: snap.closed(tf) for tf in snap.candles}, snap.tick, snap.spec, s, now, self.news)
        results: dict[str, AgentResult] = {}
        for a in self.agents:
            if a.name in self.disabled_agents:
                results[a.name] = AgentResult.no_data(a.name, snap.symbol, "-", "ablated (experiment)", now=ctx.dt, extra={"ablated": True})
                continue
            def run(a=a):
                with self.metrics.timer("agent." + a.name):
                    r = a.analyze(ctx)
                if not isinstance(r, AgentResult):
                    raise TypeError("agent returned wrong type")
                return r
            results[a.name] = self.isolation.call(a.name, run, lambda why, a=a: AgentResult.no_data(
                a.name, snap.symbol, "-", f"module unavailable: {why}", now=ctx.dt, extra={"unavailable": True}))
        regime = self.isolation.call("regime", lambda: self.regime.classify(snap), lambda why: RegimeResult(
            "UNCERTAIN", 0.0, snap.as_of, s.regime_timeframe, REGIME_FEATURE_VERSION, {"reason": f"module unavailable: {why}"}))
        with self.metrics.timer("forecast_ms"):
            forecast = self.isolation.call("forecast", lambda: self.forecaster.run(snap, regime), lambda why: ForecastResult(
                timestamp=snap.as_of, snapshot_id=snap.snapshot_id, note=f"module unavailable: {why}"))
        fc = snap.closed(s.forecast_timeframe)
        div = self.diversity.update(results, forecast.direction if self.use_forecast else None, fc[-1].time if fc else None)
        fc_used = forecast if self.use_forecast else None
        rg_used = regime if self.use_regime else None
        cons = self.consistency.evaluate(results, rg_used, fc_used, quality.score, self.sysstate.state, div, exclude=self.disabled_agents)
        # ---- supervisor: system-level assessment BEFORE the decision is validated ----------------------------
        unavailable = sorted(set(self.isolation.unavailable()) | {n for n, r in results.items() if r.is_no_data and n != "news"
                                                                   and not r.extra.get("ablated")})
        if regime.features.get("reason", "").startswith(("module", "insufficient")):
            unavailable.append("regime")
        if forecast.direction == "UNKNOWN":
            unavailable.append("forecast")
        assess = self.supervisor.assess(data_ok=quality.ok, data_issues=quality.fatal, unavailable=unavailable,
                                        forecast_uncertainty=forecast.uncertainty_label, watchdog=self.watchdog.last,
                                        config_ok=self.config_ok, broker_ok=self.health.c.get("broker", {}).get("ok", True),
                                        db_ok=self.health.c.get("database", {}).get("ok", True))
        self._apply_assessment(assess)
        d = self.decider.decide(results, exclude=self.disabled_agents)
        if assess.level != "READY" and d.action in ("BUY", "SELL"):
            d.reasons.append(f"system {assess.level}: " + "; ".join(assess.reasons[:2]))
            d.action = "HOLD"
        d = self.contextual.refine(d, regime, forecast, cons, div, self.use_regime, self.use_forecast)
        usable = sum(1 for n in SPECIALISTS if n in results and not results[n].is_no_data)
        validation = self.validator.validate(decision=d, snapshot=snap, quality=quality, consistency=cons, forecast=fc_used,
                                             system_state=self.sysstate.state, now=now, usable_specialists=usable)
        return {"results": results, "regime": regime, "forecast": forecast, "diversity": div, "consistency": cons, "decision": d,
                "validation": validation, "assessment": assess, "unavailable": unavailable}

    def _apply_assessment(self, a) -> None:
        self._assessment = a
        st = self.sysstate
        st.health, st.degraded_reasons = a.level, a.reasons
        if st.state in ("PAUSED", "SHUTDOWN", "ERROR", "STARTING"):
            return
        if a.level == "READY":
            if st.state in ("DEGRADED", "RECOVERY"):
                st.transition("READY", "assessment READY", "supervisor")
        else:
            st.transition("DEGRADED", "; ".join(a.reasons[:2])[:150], "supervisor")

    # ---------------------------------------------------------------- analysis cycle
    def _reject_cycle(self, now: float, reasons: list, sid) -> None:
        self._decision = Decision(reasons=list(reasons))
        self._validation = None
        self._streak = {"dir": None, "n": 0}
        key = "|".join(reasons)[:120]
        if now - self._reject_seen.get(key, -1e18) >= 60:
            self._reject_seen[key] = now
            self.events.emit("data", "SNAPSHOT_REJECTED", {"reasons": reasons[:4]}, sid, now)
        self.metrics.inc("snapshots_rejected")

    def _analyze(self, now: float, tick, acct, positions) -> None:
        s = self.s
        if self.sysstate.state == "READY":
            self.sysstate.transition("ANALYZING", "cycle", "engine")
        if not positions and self.sm.state is State.IDLE:
            self._go(State.ANALYZING, "analysis cycle")
        try:
            with self.metrics.timer("data_service_ms"):
                snap = self.dataservice.build(self.symbol, self.spec, now, as_of=(now if self.sim else None), tick=tick)
        except Exception as e:  # noqa: BLE001
            self.health.fail("market_data", f"snapshot: {e}")
            self._reject_cycle(now, [f"data fetch failed: {e}"], None)
            self._end_cycle()
            return
        q = self.data_validator.validate(snap)
        self._quality = q
        self.metrics.gauge("data_quality_score", q.score)
        self.metrics.gauge("data_age_sec", round(now - tick.time, 2))
        if not q.ok:
            self.health.fail("market_data", "; ".join(q.fatal[:2]))
            self._reject_cycle(now, q.fatal, snap.snapshot_id)
            self._end_cycle()
            return
        self.health.ok("market_data", "fresh")
        self._snapshot = snap
        self.sysstate.last_snapshot_id = snap.snapshot_id
        with self.metrics.timer("analysis_total_ms"):
            st = self.run_stack(snap, q, now)
        self._results, self._regime, self._forecast = st["results"], st["regime"], st["forecast"]
        self._diversity, self._consistency, self._decision, self._validation = st["diversity"], st["consistency"], st["decision"], st["validation"]
        d, res = self._decision, self._results
        sl_c = snap.closed(s.sl_timeframe)
        sr, vol = res.get("structure"), res.get("volatility")
        self._ctx = {"atr": atr(sl_c)[-1] if len(sl_c) >= 20 else 0.0,
                     "support": sr.extra.get("support") if sr else None, "resistance": sr.extra.get("resistance") if sr else None,
                     "vol_class": (vol.extra.get("vol_class") if vol and not vol.is_no_data else "UNKNOWN") or "UNKNOWN"}
        if d.action in ("BUY", "SELL"):
            self._streak = {"dir": d.action, "n": self._streak["n"] + 1 if self._streak["dir"] == d.action else 1}
        else:
            self._streak = {"dir": None, "n": 0}
        self.health.beat("analysis", "ok")
        self.health.beat("decision", d.action)
        self.sysstate.last_decision = {"action": d.action, "overall": round(d.overall_score, 1), "ts": now, "snapshot_id": snap.snapshot_id}
        self.metrics.gauge("agent_disagreement", self._consistency.disagreement)
        self.metrics.gauge("forecast_uncertainty", self._forecast.uncertainty_label)
        self.metrics.gauge("drift_flags", len(self.drift.flags))
        self._record(snap, st, now)
        self._decision_id = None
        if d.action != self._logged_action or now - self._last_decision_log >= (3600 if self.sim else 60):
            self._log_decision(now, "N/A")
            self._logged_action, self._last_decision_log = d.action, now
        self._end_cycle()

    def _end_cycle(self) -> None:
        if self.sysstate.state == "ANALYZING":
            self.sysstate.transition("READY", "cycle done", "engine")

    def _record(self, snap, st, now: float) -> None:
        """Event sourcing + experience memory. Heavy events are written once per forecast-timeframe bar or when the decision changes."""
        fc = snap.closed(self.s.forecast_timeframe)
        fbar = fc[-1].time if fc else None
        new_bar = fbar != self._last_fbar
        d, res, rg, fo, cons = st["decision"], st["results"], st["regime"], st["forecast"], st["consistency"]
        if not (new_bar or d.action != self._logged_action):
            return
        sid = snap.snapshot_id
        r3 = lambda v: round(v, 3) if isinstance(v, float) else v  # noqa: E731
        self.events.emit("data", "MARKET_SNAPSHOT_RECEIVED", {**snap.metadata(), "quality": self._quality.score}, sid, now)
        self.events.emit("agents", "AGENT_ANALYSIS_COMPLETED", {n: ("NO_DATA" if r.is_no_data else f"{r.direction.value}:{r3(r.confidence)}") for n, r in res.items()}, sid, now)
        self.events.emit("regime", "REGIME_CLASSIFIED", {"regime": rg.regime, "confidence": r3(rg.confidence), "timeframe": rg.timeframe,
                         "feature_version": rg.feature_version}, sid, now)
        self.events.emit("consistency", "CONSISTENCY_UPDATED", {"consensus": cons.consensus, "disagreement": cons.disagreement,
                         "score": r3(cons.consistency_score), "contradictions": cons.contradictions, "missing": cons.missing_agents}, sid, now)
        self.events.emit("decision", "DECISION_CREATED", {"action": d.action, "overall": r3(d.overall_score), "agreement": r3(d.agreement_score),
                         "reasons": (d.gates + d.reasons)[:4], "notes": d.context_notes[:3]}, sid, now)
        v = st["validation"]
        if d.action in ("BUY", "SELL") or not v.passed:
            self.events.emit("validator", "VALIDATION_PASSED" if v.passed else "VALIDATION_REJECTED",
                             {"action": v.action, "reasons": v.reasons[:4]}, sid, now)
            self.metrics.inc("validation_passed" if v.passed else "validation_rejected")
        if new_bar:
            self._last_fbar = fbar
            feat = forecast_features(fc)
            self.drift.observe(feat, rg.regime, fo.direction, {n: r.direction.value for n, r in res.items() if not r.is_no_data})
            self.memory.record(now=now, snapshot_id=sid, symbol=self.symbol, regime=rg.to_dict(),
                               agents={n: r.to_dict() for n, r in res.items()}, forecast=fo.to_dict(),
                               uncertainty=fo.uncertainty_label, system_state=self.sysstate.state, decision=d.action,
                               feature_version=self.s.feature_version, model_version=self.version)

    def _log_decision(self, now: float, risk_status: str) -> int:
        d = self._decision
        did = self.db.insert("decisions", {
            "ts": iso(now), "symbol": self.symbol, "action": d.action, "buy_score": d.buy_score,
            "sell_score": d.sell_score, "hold_score": d.hold_score, "overall_score": d.overall_score,
            "agreement_score": d.agreement_score, "confidence": d.confidence, "contradiction": d.contradiction,
            "reason": d.reason_text(), "risk_status": risk_status, "strategy_version": self.version})
        for r in self._results.values():
            self.db.insert_agent_prediction(r, did)
        self._decision_id = did
        self._recent.appendleft({"ts": now, "action": d.action, "status": risk_status, "overall": round(d.overall_score, 1)})
        self.db.log_event("INFO", "journal", "decision", self._journal_text(now, risk_status), {"decision_id": did, "action": d.action})
        return did

    def _journal_text(self, now: float, risk_status: str) -> str:
        t = datetime.fromtimestamp(now, timezone.utc).strftime("%H:%M:%S")
        lines = [f"[{t}] {self.symbol}"]
        for n in ("trend", "structure", "price_action", "momentum", "volatility", "liquidity", "news"):
            r = self._results.get(n)
            if r:
                lines.append(f"  {n}: {'UNKNOWN' if r.is_no_data else f'{r.direction.value} {r.confidence:.0f}'}")
        lines.append(f"  regime: {self._regime.regime}   forecast: {self._forecast.direction} (uncertainty {self._forecast.uncertainty_label})")
        d = self._decision
        lines.append(f"  Decision: {d.action}  overall={d.overall_score:.1f} agreement={d.agreement_score:.0f}")
        lines.append(f"  Risk: {risk_status}")
        lines.append(("  NO TRADE: " if d.action == "HOLD" else "  Reason: ") + d.reason_text())
        return "\n".join(lines)

    # ---------------------------------------------------------------- acting on decisions
    def _act_on_decision(self, positions, tick, acct, now: float) -> None:
        d, v = self._decision, self._validation
        if v is None or not v.passed:
            return
        if not positions:
            if self.sm.state is State.ANALYZING:
                self._go(State.IDLE, "no actionable signal")
            if now < self._suppress_until:
                return
            if d.action in ("BUY", "SELL") and self._streak["n"] >= self.s.signal_persistence_ticks \
                    and self.sm.new_trades_allowed() and self.sm.state is State.IDLE:
                self._go(State.ANALYZING, "persistent signal")
                self._go(State.SIGNAL_READY, f"{d.action} persisted {self._streak['n']} cycles")
                self._attempt_open(d.action, tick, acct, now, positions, is_reversal=False)
            return
        pos = positions[0]
        if self.sm.state not in (State.IN_POSITION, State.MANAGING) or not self.sm.reversals_allowed():
            return
        ok, why = self.reversal.should_reverse(pos.direction, d, self._streak["n"] if self._streak["dir"] != pos.direction else 0,
                                               self._holding_since(pos.ticket, now), trade_stats(self.db, now)["reversals_today"])
        if not ok:
            return
        self.db.log_event("INFO", "reversal", "triggered", f"{pos.direction} -> {d.action}: {why}")
        self._go(State.REVERSAL, why)
        r = self.exec.close_position(pos.ticket, "REVERSAL", source="engine")
        if not r.ok:
            self._go(State.IN_POSITION, "reversal close failed")
            return
        self._go(State.EXITING, "reversal exit")
        self._reconcile(self._my_positions(), now)
        if self.db.query("SELECT 1 FROM trades WHERE ticket=? AND status='OPEN'", (pos.ticket,)):
            self.db.log_event("WARN", "reversal", "aborted", "close not yet confirmed; no re-entry this cycle")
            return
        self._attempt_open(d.action, tick, acct, now, self._my_positions(), is_reversal=True)

    def _attempt_open(self, direction: str, tick, acct, now: float, positions, is_reversal: bool) -> None:
        if not self._go(State.RISK_CHECK, "safety gate"):
            return
        c = self._ctx
        entry = tick.ask if direction == "BUY" else tick.bid
        if not c.get("atr"):
            self._go(State.IDLE, "no ATR available")
            return
        sl, tp = plan_sl_tp(direction, entry, c["atr"], c.get("support"), c.get("resistance"), self.s, self.spec)
        did = self._decision_id or self._log_decision(now, "PENDING")
        d = self._decision
        req = TradeRequest(self.symbol, direction, entry, sl, tp, c["atr"], f"{self.symbol}:{direction}:{did}", did, is_reversal,
                           meta={"timeframe": self.s.ltf_timeframes[0], "spread": (tick.ask - tick.bid) / self.spec.point,
                                 "market_regime": self._regime.regime, "volatility": c.get("vol_class"),
                                 "decision_score": d.overall_score, "agreement_score": d.agreement_score,
                                 "agents": {k: v.to_dict() for k, v in self._results.items()}})
        rd = self.gate.authorize(req, self._validation, d, account=acct, spec=self.spec, tick=tick, positions=positions, now=now,
                                 new_trades_allowed=(self.sm.new_trades_allowed() or (is_reversal and self.sm.reversals_allowed())),
                                 health_ok=not self.health.degraded() and bool(self._assessment and self._assessment.level == "READY"),
                                 vol_class=c.get("vol_class", "UNKNOWN"), eq=self._eq, stats=trade_stats(self.db, now))
        self._last_rd = rd
        self.db.execute("UPDATE decisions SET risk_status=? WHERE id=?", (rd.status, did))
        if self._recent:
            self._recent[0]["status"] = rd.status
        if not rd.approved:
            if self.mode == "DISABLED":
                self._suppress_until = now + 300
            key = "|".join(rd.reasons)
            if now - self._risk_log_seen.get(key, -1e18) >= 60:
                self._risk_log_seen[key] = now
                self.db.insert("risk_events", {"ts": iso(now), "decision_id": did, "level": "CRITICAL" if rd.status == "CRITICAL" else "BLOCK",
                                               "rule": rd.reasons[0].split(":")[0], "message": "; ".join(rd.reasons),
                                               "details_json": json.dumps(rd.rules)})
                self.db.log_event("WARN", "risk", "trade_blocked", "TRADE BLOCKED: " + "; ".join(rd.reasons))
            if rd.status == "CRITICAL":
                self._emergency("risk governor: " + rd.reasons[0])
            else:
                self._go(State.IDLE, "blocked by safety gate")
            return
        review = self.llm.review({"direction": direction, "decision": d.to_dict()}) if self.llm.enabled else None
        if review:
            self.db.log_event("INFO", "llm", "advisory", json.dumps(review))
        self._go(State.EXECUTING, "approved")
        res = self.exec.open_position(req, rd, self.version)
        if res.ok:
            self._go(State.IN_POSITION, "opened")
            pos = next((p for p in self._my_positions() if p.ticket == res.ticket), None)
            if pos:
                self.pm.register(pos, self.spec)
            self._streak = {"dir": None, "n": 0}
        else:
            self._go(State.IDLE, f"execution failed: {res.message}")

    # ================================================================== snapshot for API / dashboard
    def snapshot(self) -> dict:
        self.last_dashboard_poll = time.time()
        with self._lock:
            return copy.deepcopy(self._snap)

    def candles(self, tf: str, n: int = 100) -> list:
        sn = self._snapshot
        if sn is None or tf not in sn.candles:
            return []
        return [[c.time, c.open, c.high, c.low, c.close, c.volume] for c in sn.candles[tf][-n:]]

    @staticmethod
    def _event_line(r: dict) -> str:
        p = json.loads(r["payload_json"] or "{}")
        t = r["event_type"]
        if t == "MARKET_SNAPSHOT_RECEIVED": return f"Market snapshot received ({p.get('symbol')}) #{(r['snapshot_id'] or '')[:6]}"
        if t == "AGENT_ANALYSIS_COMPLETED": return f"Agents completed analysis ({sum(1 for v in p.values() if v != 'NO_DATA')}/{len(p)} with data)"
        if t == "REGIME_CLASSIFIED": return f"Regime: {p.get('regime')} (margin {p.get('confidence')})"
        if t == "FORECAST_GENERATED": return f"Forecast generated: {p.get('direction')} ({p.get('uncertainty')} uncertainty)"
        if t == "FORECAST_RESOLVED": return f"Forecasts resolved against actual outcome: {p.get('resolved_rows')}"
        if t == "CONSISTENCY_UPDATED": return f"Consistency: {p.get('consensus')} / disagreement {p.get('disagreement')}"
        if t == "DECISION_CREATED": return f"Decision engine: {p.get('action')} (score {p.get('overall')})"
        if t.startswith("VALIDATION"): return f"Validator: {t.split('_')[1].lower()} {p.get('action')} {'; '.join(p.get('reasons', []))[:80]}"
        if t == "SAFETY_GATE": return "Safety gate: " + ("approved" if p.get("approved") or p.get("allowed") else "blocked") + f" {p.get('direction', p.get('reducing_action', ''))}"
        if t in ("SIMULATION_ACTION", "EXECUTION_ACTION"): return f"{'Simulated' if t == 'SIMULATION_ACTION' else 'Demo'} action: {p.get('action')} {p.get('direction', '')} {p.get('reason', p.get('status', ''))}"
        if t == "DRIFT_DETECTED": return f"Drift: {p.get('kind')} {p.get('name')} ({p.get('severity')})"
        if t == "MODEL_EVALUATED": return f"Models evaluated (drift flags: {p.get('drift_flags')})"
        return f"{t.replace('_', ' ').title()}"

    def _publish(self, now: float) -> None:
        if not self.publish_enabled:
            return
        a, t, d = self._acct, self._tick, self._decision
        display = self.money
        def disp(amount: float | None):
            return None if amount is None or display is None else round(display.to_major(amount), 2)
        agents = []
        for n in CARD_ORDER:
            if n == "regime":
                r = self._regime
                agents.append({"key": n, "name": "Market Regime", "state": r.regime, "output": r.features.get("reason") or f"ER {r.features.get('er', 0):.2f} · ATR× {r.features.get('atr_ratio', 0):.2f}",
                               "metric": round(r.confidence * 100) if r.regime != "UNCERTAIN" or r.confidence else None, "metric_label": "margin",
                               "data_quality": None, "latency_ms": None, "updated": r.timestamp, "weight": None, "no_data": False, "available": True})
            elif n == "forecast":
                f = self._forecast
                agents.append({"key": n, "name": "Forecast", "state": f.direction, "output": f.note or f"{len(f.models)} models · uncertainty {f.uncertainty_label}",
                               "metric": None if f.model_agreement is None else round(f.model_agreement * 100), "metric_label": "agree",
                               "data_quality": None, "latency_ms": self.metrics.last_ms("forecast_ms"), "updated": f.timestamp, "weight": None,
                               "no_data": f.direction == "UNKNOWN", "available": "forecast" not in self.isolation.unavailable()})
            else:
                r = self._results.get(n)
                if r is None:
                    agents.append({"key": n, "name": n.replace("_", " ").title(), "state": "—", "output": "waiting for first analysis", "metric": None,
                                   "metric_label": "conf", "no_data": True, "available": True, "weight": self.weights.get(n)})
                    continue
                agents.append({"key": n, "name": {"price_action": "Price Action", "news": "News / Macro"}.get(n, n.title()),
                               "state": "NO DATA" if r.is_no_data else r.direction.value, "output": r.reason,
                               "metric": None if r.is_no_data else round(r.confidence), "metric_label": "conf",
                               "strength": None if r.is_no_data else round(r.strength), "data_quality": round(r.data_quality),
                               "latency_ms": self.metrics.last_ms("agent." + n), "updated": r.timestamp.timestamp(),
                               "weight": self.weights.get(n), "no_data": r.is_no_data, "available": n not in self.isolation.unavailable()})
        counts = {"BUY": 0, "SELL": 0, "HOLD": 0, "NO_DATA": 0}
        for x in agents[:7]:
            counts["NO_DATA" if x["no_data"] else x["state"]] += 1
        ps = []
        try:
            ps = self._my_positions() if (self.spec and self.health.c.get("broker", {}).get("ok")) else []
        except Exception:  # noqa: BLE001
            pass
        pos = None
        if ps:
            p, ms = ps[0], self.pm.states.get(ps[0].ticket)
            floor = (ms.peak_profit * (1 - self.s.profit_retrace_percent / 100)) if ms and ms.armed else None
            pos = {"ticket": p.ticket, "direction": p.direction, "volume": p.volume, "entry": p.price_open, "sl": p.sl, "tp": p.tp,
                   "profit": round(p.profit, 2), "display_profit": disp(p.profit),
                   "peak_profit": round(ms.peak_profit, 2) if ms else None,
                   "display_peak_profit": disp(ms.peak_profit) if ms else None,
                   "protected_profit": round(floor, 2) if floor is not None else None,
                   "display_protected_profit": disp(floor) if floor is not None else None,
                   "protection_armed": bool(ms and ms.armed)}
        stats = trade_stats(self.db, now) if self.spec else {}
        suggested = ("MANAGE open position" if ps else f"ENTER {d.action} (persistence {self._streak['n']}/{self.s.signal_persistence_ticks})"
                     if d.action in ("BUY", "SELL") else "WAIT: " + d.reason_text()[:140])
        chg = None
        if self._snapshot is not None:
            h1 = self._snapshot.closed("H1")
            if len(h1) > 25:
                chg = round((h1[-1].close - h1[-25].close) / h1[-25].close * 100, 3)
        wall = time.time()
        if wall - getattr(self, "_heavy_ts", 0) > 3.0:
            ev = self.learning_mgr.last_eval or {}
            self._heavy = {"cal": ev.get("calibration") or self.forecaster.calibration_info(), "exp": self.memory.summary(),
                           "chal": self.registry.rows("CHALLENGER"), "n_exp": self.db.count("experiments"),
                           "n_samples": self.db.count("learning_samples"),
                           "log": [{"ts": r["ts"], "type": r["event_type"], "message": self._event_line(r)} for r in self.events.tail(10)]}
            self._heavy_ts = wall
        if wall - getattr(self, "_board_ts", 0) > 30.0:
            try:
                self._board = [{k: b[k] for k in ("model", "role", "n", "accuracy", "verdict")} for b in self.forecaster.scoreboard()
                               if b["role"] in ("ENSEMBLE", "BASELINE")]
            except Exception:  # noqa: BLE001
                self._board = []
            self._board_ts = wall
        H = self._heavy
        cal, exp, chal = H["cal"], H["exp"], H["chal"]
        div = self._diversity or {}
        rr = [x for x in (self._consistency.contradictions if self._consistency else [])]
        gate = self.gate.status()
        vp = self.metrics.counters.get("validation_passed", 0), self.metrics.counters.get("validation_rejected", 0)
        modules = {
            "data_layer": {"quality": self._quality.score if self._quality else None, "latency_ms": self.metrics.last_ms("data_service_ms"),
                           "status": "Healthy" if self._quality and self._quality.ok else "Degraded"},
            "memory": {"experiences": exp["experiences"], "resolved": exp["resolved"], "raw_events": exp["raw_events"], "status": "Healthy"},
            "learning": {"candidates": len(chal), "experiments": H["n_exp"], "samples": H["n_samples"],
                         "status": "Evaluating" if self.sysstate.state == "EVALUATING" else "Idle"},
            "diversity": {"effective_opinions": div.get("effective_independent_opinions"), "agreement": div.get("agreement"),
                          "diversity": div.get("diversity"), "source": div.get("redundancy_source"),
                          "status": "Healthy" if div.get("diversity") in ("HIGH", "MEDIUM") else ("Low diversity" if div.get("diversity") == "LOW" else "Unknown")},
            "forecast_ensemble": {"models": len(self._forecast.models), "baselines": len(self._forecast.baselines), "calibration": cal.get("status"),
                                  "ensemble_hit_rate": cal.get("ensemble_hit_rate"), "resolved": cal.get("resolved_ensemble_forecasts"),
                                  "status": "Healthy" if self._forecast.direction != "UNKNOWN" else "Unavailable"},
            "safety_gate": {"validation_pass_pct": round(100 * vp[0] / (vp[0] + vp[1]), 1) if (vp[0] + vp[1]) else None,
                            "risk_check": ("Pass" if (self._last_rd and self._last_rd.approved) else "Block" if self._last_rd else "—"),
                            "mode": self.mode, "status": "Safe"},
        }
        snap = {
            "ts": now, "app_version": __import__("goldbot").__version__, "forecast_board": getattr(self, "_board", []),
            "levels": {"support": self._ctx.get("support"), "resistance": self._ctx.get("resistance"), "atr": self._ctx.get("atr")},
            "mode": getattr(self.broker, "name", "?"), "execution_mode": self.mode, "symbol": self.symbol, "version": self.version,
            "state": self.sm.state.value, "system": self.sysstate.snapshot(), "paused": self.sm.paused, "new_trades_allowed": self.sm.new_trades_allowed(),
            "health": self.health.snapshot(), "watchdog": self.watchdog.last, "assessment": self._assessment.to_dict() if self._assessment else None,
            "isolation": self.isolation.status(),
            "account": None if not a else {"balance": round(a.balance, 2), "equity": round(a.equity, 2), "free_margin": round(a.free_margin, 2),
                                           "margin": round(a.margin, 2), "margin_level": round(a.margin_level, 1), "leverage": a.leverage,
                                           "account_type": a.margin_mode, "demo": a.is_demo, "currency": a.currency,
                                           "display_balance": disp(a.balance), "display_equity": disp(a.equity),
                                           "display_free_margin": disp(a.free_margin), "display_margin": disp(a.margin),
                                           "display_currency": display.display_currency if display else a.currency,
                                           "display_denomination": display.denomination.value if display else "STANDARD"},
            "units": None if not self.spec else {"account_currency": a.currency if a else "", "denomination": self.money.denomination.value if self.money else "",
                                                  "minor_per_major": self.money.minor_per_major if self.money else 1, "symbol": self.symbol,
                                                  "base": split_symbol(self.symbol)[0], "suffix": split_symbol(self.symbol)[1],
                                                  "tick_size": self.spec.tick_size, "tick_value": self.spec.tick_value, "contract_size": self.spec.contract_size,
                                                  "digits": self.spec.digits, "point": self.spec.point, "volume_min": self.spec.volume_min,
                                                  "volume_step": self.spec.volume_step, "warnings": self.unit_warnings},
            "market": None if not t else {"bid": t.bid, "ask": t.ask, "spread_points": round((t.ask - t.bid) / self.spec.point, 1) if self.spec else None,
                                          "regime": self._regime.regime, "volatility": self._ctx.get("vol_class", "UNKNOWN"), "change_24h_pct": chg},
            "data_quality": self._quality.to_dict() if self._quality else None,
            "snapshot": self._snapshot.metadata() if self._snapshot else None,
            "position": pos, "agents": agents, "consensus": counts,
            "regime": self._regime.to_dict(), "forecast": self._forecast.to_dict(),
            "diversity": div, "consistency": self._consistency.to_dict() if self._consistency else None,
            "validation": self._validation.to_dict() if self._validation else None,
            "decision": {**d.to_dict(), "risk_status": self._last_rd.status if self._last_rd else "N/A", "suggested_action": suggested,
                         "persistence": self._streak["n"]},
            "risk": (self.risk.snapshot(a, self._my_positions() if self.spec else [], self.spec, self._eq, stats, (t.ask - t.bid) / self.spec.point)
                     if (a and t and self.spec) else None),
            "last_risk_rules": self._last_rd.rules if self._last_rd else [],
            "safety_gate": gate, "modules": modules, "recent_decisions": list(self._recent),
            "mission_log": H["log"],
            "drift_flags": self.drift.flags, "metrics": self.metrics.snapshot(), "replay": self.replay_status,
            "anomalies": len(self.drift.flags) + len(self._consistency.contradictions if self._consistency else []),
            "last_error": self.last_error,
        }
        with self._lock:
            self._snap = snap

    # ================================================================== loop
    def run_forever(self, stop: threading.Event) -> None:
        while not stop.is_set():
            t0 = time.time()
            try:
                self.step()
            except Exception as e:  # noqa: BLE001 - never die silently; fail safe
                self.last_error = f"{type(e).__name__}: {e}"
                self.db.log_event("ERROR", "engine", "step_exception", self.last_error)
                self.sysstate.transition("ERROR", self.last_error[:150], "engine")
                self._enter_error("unexpected exception in engine step")
            stop.wait(max(0.05, self.s.check_interval_sec - (time.time() - t0)))
        self.sysstate.transition("SHUTDOWN", "stop requested", "engine")
