import dataclasses, json, math, random, unittest
from goldbot.agents import AgentContext
from goldbot.consistency import ConsistencyEngine
from goldbot.data import TF_SECONDS, Candle, Tick, resample
from goldbot.db import Database
from goldbot.decision import ContextualDecision, Decision
from goldbot.diversity import DiversityMonitor
from goldbot.drift import DriftDetector, psi, tv_distance
from goldbot.events import EventLog
from goldbot.execution import ExecutionEngine
from goldbot.forecast import (FEATURES, ForecastResult, ForecastSystem, ModelRegistry, OnlineLogistic, forecast_features)
from goldbot.indicators import atr, ema
from goldbot.learning import LearningEngine
from goldbot.learning_manager import LearningManager, forecast_replay, offline_compare
from goldbot.experiments import ExperimentLab, HoldoutExhausted
from goldbot.regime import RegimeDetector, decide_regime, evaluate_regimes, regime_features
from goldbot.risk import RiskGovernor, TradeRequest
from goldbot.safety_gate import SafetyGate
from goldbot.settings import ConfigError, load_settings
from goldbot.snapshot import DataQuality, DataValidator
from goldbot.state_machine import State
from goldbot.supervisor import ModuleIsolation, RecoveryManager, SystemStateManager, SystemSupervisor, Watchdog
from goldbot.units import (Denomination, check_spec, detect_account_money, loss_per_lot, pnl_money, price_to_points, price_to_ticks, split_symbol)
from goldbot.validator import Validator
from goldbot.broker_paper import PaperBroker
from goldbot.risk import calc_lot
from goldbot.synth import generate_m1
from tests.helpers import S, SPEC, acct, flat_candles, m5_series, make_snap, new_db, ranging, res, trending


# ======================================================================= temporal contract / leakage
class TestTemporalContract(unittest.TestCase):
    def snap(self, **kw):
        c = {"M5": m5_series(120), "M15": trending(120)}
        return make_snap(c, **kw)

    def test_valid_snapshot_passes(self):
        sn = self.snap()
        q = DataValidator(S).validate(sn)
        self.assertTrue(q.ok, q.fatal)

    def test_future_candle_is_fatal(self):
        sn = self.snap()
        early = dataclasses.replace(sn, as_of=sn.as_of - 3600)          # pretend "now" is an hour earlier
        q = DataValidator(S).validate(early)
        self.assertFalse(q.ok)
        self.assertTrue(any("FUTURE_DATA" in x for x in q.fatal))

    def test_misaligned_timestamps_detected(self):
        c = m5_series(120)
        bad = c[:-1] + [Candle(c[-1].time + 7, *[getattr(c[-1], k) for k in ("open", "high", "low", "close", "volume")])]
        sn = make_snap({"M5": bad})
        q_warn = DataValidator(S, strict=False).validate(sn)
        q_strict = DataValidator(S, strict=True).validate(sn)
        self.assertTrue(any("MISALIGNED" in x for x in q_warn.warnings))
        self.assertFalse(q_strict.ok)

    def test_non_monotonic_duplicate_and_bad_ohlc(self):
        c = m5_series(120)
        dup = c + [c[-1]]
        self.assertFalse(DataValidator(S).validate(make_snap({"M5": dup}, as_of=c[-1].time + 600)).ok)
        bad = c[:-1] + [Candle(c[-1].time, 100, 90, 95, 99, 1)]            # high < open
        self.assertFalse(DataValidator(S).validate(make_snap({"M5": bad})).ok)

    def test_stale_or_invalid_tick(self):
        c = {"M5": m5_series(120)}
        a = max(x[-1].time + 300 for x in c.values())
        self.assertFalse(DataValidator(S).validate(make_snap(c, tick=Tick(a - 100, 2650, 2650.2), available_at=a)).ok)
        self.assertFalse(DataValidator(S).validate(make_snap(c, tick=Tick(a, 2651, 2650), available_at=a)).ok)       # ask < bid

    def test_indicator_window_does_not_look_ahead(self):
        c = m5_series(300)
        cl = [x.close for x in c]
        for i in (80, 150, 250):
            self.assertAlmostEqual(atr(c)[i], atr(c[:i + 1])[-1], places=9)
            self.assertAlmostEqual(ema(cl, 20)[i], ema(cl[:i + 1], 20)[-1], places=9)

    def test_features_use_only_given_candles(self):
        c = m5_series(300)
        f1 = forecast_features(c[:200])
        c2 = c[:200] + [Candle(x.time, x.open * 1.5, x.high * 1.5, x.low * 1.5, x.close * 1.5, 1) for x in c[200:]]
        self.assertEqual(f1, forecast_features(c2[:200]))

    def test_forecast_replay_predictions_unchanged_by_future_data(self):
        c = m5_series(400)
        K = 300
        c2 = c[:K] + [Candle(x.time, x.open + 40, x.high + 41, x.low + 39, x.close - 35, 1) for x in c[K:]]
        mk = lambda: {"lg": OnlineLogistic(warmup=20)}
        a = forecast_replay(c, 12, 0.25, mk())["lg"]
        b = forecast_replay(c2, 12, 0.25, mk())["lg"]
        ea = [x for x in a if x[0] + 12 < K]
        eb = [x for x in b if x[0] + 12 < K]
        self.assertGreater(len(ea), 50)
        self.assertEqual(ea, eb)                       # identical outcomes for everything fully inside the unchanged past

    def test_engine_cycle_identical_when_only_the_future_differs(self):
        from goldbot.replay import ReplayEngine
        m1 = generate_m1(3900, seed=2)
        K = 3760
        m2 = m1[:K] + [Candle(x.time, x.open + 25, x.high + 26, x.low + 24, x.close - 20, 1) for x in m1[K:]]
        a = ReplayEngine(S).replay(m1, warmup=3700, max_steps=50)
        b = ReplayEngine(S).replay(m2, warmup=3700, max_steps=50)
        self.assertEqual(a.digest, b.digest)           # hash-chained event log is byte-identical
        self.assertEqual(a.leak_violations, 0)


# ======================================================================= units
class TestUnits(unittest.TestCase):
    CENT = dataclasses.replace(SPEC, tick_value=100.0, currency_profit="USC")

    def test_cent_account_never_scales_price(self):
        am = detect_account_money("USC")
        self.assertEqual((am.denomination, am.minor_per_major), (Denomination.CENT, 100))
        self.assertEqual(am.to_major(80000), 800)
        price, sl = 2650.00, 2644.00
        lot, risk, why = calc_lot(80000, 1.0, price, sl, self.CENT)      # equity in CENTS
        std_lot, std_risk, _ = calc_lot(800, 1.0, price, sl, SPEC)       # same money in dollars
        self.assertEqual(lot, std_lot)
        self.assertAlmostEqual(am.to_major(risk), std_risk)
        self.assertEqual(price, 2650.00)                                  # the price object is untouched

    def test_conversions_are_separate_quantities(self):
        self.assertAlmostEqual(price_to_points(1.0, SPEC), 100.0)
        self.assertAlmostEqual(price_to_ticks(1.0, SPEC), 100.0)
        self.assertAlmostEqual(loss_per_lot(6.0, SPEC), 600.0)
        self.assertAlmostEqual(pnl_money("BUY", 2650, 2656, 0.01, SPEC), 6.0)
        self.assertAlmostEqual(pnl_money("SELL", 2650, 2656, 0.01, SPEC), -6.0)

    def test_symbol_suffix_detection(self):
        self.assertEqual(split_symbol("XAUUSDm"), ("XAUUSD", "m"))
        self.assertEqual(split_symbol("GOLD.a"), ("GOLD", ".a"))
        self.assertEqual(split_symbol("XAUUSD"), ("XAUUSD", ""))

    def test_cent_display_currency_is_major_currency(self):
        am = detect_account_money("USC")
        self.assertEqual(am.display_currency, "USD")
        self.assertEqual(am.to_major(80000), 800)

    def test_spec_checks(self):
        self.assertEqual(check_spec(dataclasses.replace(SPEC, currency_profit="USD"), "USD"), [])
        w = check_spec(dataclasses.replace(SPEC, tick_value=0.5, currency_profit="USD"), "USD")
        self.assertTrue(w and "tick_value" in w[0])
        self.assertTrue(check_spec(dataclasses.replace(SPEC, tick_size=0.0), "USD"))
        self.assertEqual(check_spec(self.CENT, "USC"), [])                 # consistent with money in cents
        self.assertEqual(check_spec(dataclasses.replace(self.CENT, currency_profit="USD"), "USC"), [])
        self.assertTrue(check_spec(dataclasses.replace(self.CENT, tick_value=1.0), "USC"))   # cent account but tick_value not in cents


# ======================================================================= regime
class TestRegime(unittest.TestCase):
    def classify(self, c):
        return RegimeDetector(S).classify(make_snap({"M15": c}))

    def test_trending_ranging_volatile_uncertain(self):
        self.assertEqual(self.classify(trending(200)).regime, "TRENDING")
        self.assertEqual(self.classify(ranging(200)).regime, "RANGING")
        c = flat_candles(186, rng=0.5) + [Candle(1767571200 + (186 + i) * 900, 2650, 2658, 2644, 2650, 1) for i in range(14)]
        self.assertEqual(self.classify(c).regime, "HIGH_VOLATILITY")
        r = self.classify(trending(20))
        self.assertEqual((r.regime, r.confidence), ("UNCERTAIN", 0.0))

    def test_fields_stored_and_confidence_is_margin_not_probability(self):
        d = self.classify(trending(200)).to_dict()
        for k in ("regime", "confidence", "timestamp", "timeframe", "feature_version", "features", "context"):
            self.assertIn(k, d)
        self.assertEqual(d["confidence_kind"], "margin_score_not_probability")
        json.dumps(d)

    def test_history_context_and_instability_downgrade(self):
        det = RegimeDetector(S)
        for n in (150, 160, 170, 180):
            det.classify(make_snap({"M15": trending(n)}))
        ctx = det.classify(make_snap({"M15": trending(190)})).context
        self.assertGreaterEqual(ctx["bars_observed"], 4)
        det2 = RegimeDetector(S)
        det2.hist.extend([(i, r) for i, r in enumerate(["TRENDING", "RANGING", "TRENDING", "RANGING", "TRENDING"])])
        self.assertEqual(det2.classify(make_snap({"M15": ranging(200)})).regime, "UNCERTAIN")

    def test_independent_evaluation_reports_per_regime_forward_stats(self):
        r = evaluate_regimes(resample(generate_m1(9000, seed=4), 900), horizon=8, warmup=80)
        self.assertGreater(r["samples"], 50)
        for st in r["per_regime"].values():
            self.assertIn("fwd_er", st)
        self.assertIn("flip_rate", r)


# ======================================================================= forecasting
class TestForecast(unittest.TestCase):
    def setUp(self):
        self.db = new_db()
        self.ev = EventLog(self.db, lambda: 0.0)
        self.reg = ModelRegistry(self.db, S, self.ev)
        self.fs = ForecastSystem(self.db, S, self.reg, self.ev)
        self.c = m5_series(500, seed=6)

    def snap_at(self, n):
        return make_snap({"M5": self.c[:n]})

    def test_baselines_and_members_exist(self):
        self.assertEqual({m.model_id for m in self.fs.members}, {"ema_slope", "mean_rev", "logistic"})
        self.assertEqual({m.model_id for m in self.fs.baselines}, {"naive_majority", "persistence", "stat_drift"})

    def test_no_fake_probability_and_honest_uncertainty(self):
        r = self.fs.run(self.snap_at(300))
        self.assertIsNone(r.probability_up)
        self.assertIn("none", r.probability_source)
        self.assertEqual(r.calibration["status"], "INSUFFICIENT_DATA")
        self.assertGreaterEqual(r.uncertainty_components["calibration"], 0.15)
        lg = next(m for m in r.models if m.model_id == "logistic")
        self.assertIsNone(lg.p_up_raw)
        self.assertIn("warming up", lg.note)
        json.dumps(r.to_dict())

    def test_required_output_fields(self):
        d = self.fs.run(self.snap_at(300), RegimeDetector(S).classify(make_snap({"M15": trending(200)}))).to_dict()
        for k in ("direction", "horizon_bars", "uncertainty_label", "model_agreement", "calibration", "timestamp", "feature_version",
                  "snapshot_id", "regime_context"):
            self.assertIn(k, d)
        self.assertEqual(d["regime_context"], "TRENDING")

    def test_logged_once_per_bar(self):
        sn = self.snap_at(300)
        self.fs.run(sn); self.fs.run(sn); self.fs.run(sn)
        n = self.db.query("SELECT COUNT(*) c FROM forecast_log WHERE role='ENSEMBLE'")[0]["c"]
        self.assertEqual(n, 1)
        roles = {r["role"] for r in self.db.query("SELECT DISTINCT role FROM forecast_log")}
        self.assertEqual(roles, {"ENSEMBLE", "MEMBER", "BASELINE"})

    def test_resolution_waits_for_target_bar_and_uses_only_later_data(self):
        h = S.forecast_horizon_bars
        self.fs.run(self.snap_at(300))
        entry = self.c[299]
        self.assertEqual(self.fs.resolve(self.snap_at(300 + h - 1)), 0)           # target bar (index 299+h) not yet closed
        self.assertEqual(self.db.query("SELECT COUNT(*) c FROM forecast_log WHERE actual IS NOT NULL")[0]["c"], 0)
        n = self.fs.resolve(self.snap_at(300 + h))
        self.assertGreater(n, 0)
        row = self.db.query("SELECT * FROM forecast_log WHERE role='ENSEMBLE'")[0]
        move = self.c[299 + h].close - entry.close
        thr = S.forecast_flat_threshold_atr * row["atr"]
        self.assertEqual(row["actual"], "UP" if move > thr else "DOWN" if move < -thr else "FLAT")
        self.assertEqual(row["correct"], int(row["direction"] == row["actual"]))

    def test_champion_is_immutable_during_live_resolution(self):
        lg = next(m for m in self.fs.members if m.model_id == "logistic")
        for n in range(300, 300 + S.forecast_horizon_bars):
            self.fs.run(self.snap_at(n))
        self.assertEqual(lg.n, 0)
        before = dict(lg.state())
        for n in range(300 + S.forecast_horizon_bars, 380):
            self.fs.run(self.snap_at(n))
        resolved = self.db.query("SELECT COUNT(*) c FROM forecast_log WHERE role='MEMBER' AND model_id='logistic' AND actual IN ('UP','DOWN')")[0]["c"]
        self.assertGreater(resolved, 0)
        self.assertEqual(lg.n, 0)
        self.assertEqual(lg.state(), before)

    def test_scoreboard_verdicts(self):
        def add(i, role, mid, correct, ver="1.0"):
            self.db.insert("forecast_log", dict(ts="t", snapshot_id=f"s{i}", model_id=mid, model_version=ver, role=role, direction="UP",
                                                horizon_bars=12, timeframe="M5", regime="TRENDING", entry_price=1, entry_time=i, atr=1,
                                                actual="UP" if correct else "DOWN", correct=int(correct)))
        for i in range(50):
            add(i, "ENSEMBLE", "ensemble", 1)
            for b in ("naive_majority", "persistence", "stat_drift"):
                add(i, "BASELINE", b, i % 2)
        v = {b["role"]: b["verdict"] for b in self.fs.scoreboard() if b["role"] == "ENSEMBLE"}
        self.assertEqual(v["ENSEMBLE"], "INSUFFICIENT_DATA")                        # < forecast_min_samples
        for i in range(50, 200):
            add(i, "ENSEMBLE", "ensemble", 1)
            for b in ("naive_majority", "persistence", "stat_drift"):
                add(i, "BASELINE", b, i % 2)
        self.assertEqual([b for b in self.fs.scoreboard() if b["role"] == "ENSEMBLE"][0]["verdict"], "BEATS_ALL_BASELINES")

    def test_registry_champion_stable_rollback(self):
        self.reg.register("logistic", "1.1", "CHALLENGER", {"lr": .1})
        with self.assertRaises(ValueError):
            self.reg.promote("logistic", "1.1", "me", "evidence")                   # not VALIDATED
        self.reg.set_status("logistic", "1.1", "VALIDATED", "ok")
        with self.assertRaises(ValueError):
            self.reg.promote("logistic", "1.1", "", "evidence")                     # approver required
        self.reg.promote("logistic", "1.1", "Ehsan", "shadow passed")
        st = {r["version"]: r["status"] for r in self.reg.rows(model_id="logistic")}
        self.assertEqual(st, {"1.0": "STABLE", "1.1": "CHAMPION"})                  # previous version retained
        self.assertEqual(self.reg.rollback("logistic", "degraded"), "1.0")
        st = {r["version"]: r["status"] for r in self.reg.rows(model_id="logistic")}
        self.assertEqual(st, {"1.0": "CHAMPION", "1.1": "REJECTED"})


# ======================================================================= diversity / consistency
class TestDiversityConsistency(unittest.TestCase):
    NAMES = ("trend", "structure", "momentum", "liquidity")

    def feed(self, correlated: bool):
        dm, rnd = DiversityMonitor(), random.Random(1)
        for i in range(60):
            base = rnd.choice(["BUY", "SELL", "HOLD"])
            results = {n: res(n, base if correlated else rnd.choice(["BUY", "SELL", "HOLD"]), 70, 70) for n in self.NAMES}
            dm.update(results, None, i)
        final = {n: res(n, "BUY", 70, 70) for n in self.NAMES}
        return dm.update(final, None, 999)

    def test_high_agreement_low_diversity_vs_high_diversity(self):
        a = self.feed(correlated=True)
        b = self.feed(correlated=False)
        self.assertEqual((a["agreement"], a["diversity"]), ("HIGH", "LOW"))
        self.assertEqual((b["agreement"], b["diversity"]), ("HIGH", "HIGH"))
        self.assertLess(a["effective_independent_opinions"], 1.6)
        self.assertGreater(b["effective_independent_opinions"], 3.0)

    def test_before_history_uses_feature_overlap_and_says_so(self):
        dm = DiversityMonitor()
        r = dm.update({n: res(n, "BUY") for n in self.NAMES}, None, 1)
        self.assertEqual(r["redundancy_source"], "feature_overlap")

    def test_consistency_flags_and_missing_agents(self):
        cons = ConsistencyEngine(S).evaluate(
            {"trend": res("trend", "BUY"), "structure": res("structure", "BUY"), "momentum": res("momentum", "SELL"),
             "price_action": res("price_action", "HOLD", 30, 10), "liquidity": res("liquidity", "HOLD", 0, 0, dq=0),
             "volatility": res("volatility", "HOLD"), "news": res("news", "HOLD", 0, 0, dq=0)},
            RegimeDetector(S).classify(make_snap({"M15": ranging(200)})),
            ForecastResult(direction="DOWN", uncertainty_label="HIGH"), 95.0, "READY",
            {"agreement": "HIGH", "diversity": "LOW"})
        self.assertIn("liquidity", cons.missing_agents)
        self.assertEqual(cons.stances["liquidity"], "MISSING")                      # nothing invented
        for flag in ("TREND_VS_MOMENTUM", "FORECAST_VS_CONSENSUS", "DIRECTIONAL_CONSENSUS_IN_RANGING_REGIME",
                     "DIRECTIONAL_CONSENSUS_WITH_HIGH_FORECAST_UNCERTAINTY", "HIGH_AGREEMENT_BUT_LOW_DIVERSITY"):
            self.assertIn(flag, cons.contradictions)
        self.assertEqual(cons.flags["FORECAST_UNCERTAINTY"], "HIGH")
        self.assertIn(cons.disagreement, ("MEDIUM", "HIGH"))

    def test_consistency_clean_case(self):
        team = {n: res(n, "BUY", 85, 80) for n in ("trend", "structure", "price_action", "momentum", "liquidity")}
        team.update(volatility=res("volatility", "HOLD"), news=res("news", "HOLD", 0, 0, dq=0))
        c = ConsistencyEngine(S).evaluate(team, RegimeDetector(S).classify(make_snap({"M15": trending(200)})),
                                          ForecastResult(direction="UP", uncertainty_label="LOW"), 98.0, "READY", None)
        self.assertEqual((c.consensus, c.disagreement), ("BULLISH", "LOW"))
        self.assertEqual(c.contradictions, [])
        self.assertGreater(c.consistency_score, 80)

    def test_contextual_decision_only_makes_things_more_conservative(self):
        d = Decision(action="BUY", overall_score=72.0, buy_score=72.0, lead="BUY", confidence=70)
        cd = ContextualDecision(S)
        out = cd.refine(dataclasses.replace(d, reasons=[], context_notes=[]), RegimeDetector(S).classify(make_snap({"M15": trending(200)})),
                        ForecastResult(direction="UP", uncertainty_label="LOW"), None, None)
        self.assertEqual(out.action, "BUY")
        out = cd.refine(dataclasses.replace(d, reasons=[], context_notes=[]), None, ForecastResult(direction="DOWN", uncertainty_label="LOW"), None, None)
        self.assertEqual(out.action, "HOLD")
        self.assertIn("contradicts", out.reason_text())
        out = cd.refine(dataclasses.replace(d, reasons=[], context_notes=[]), None, ForecastResult(direction="DOWN", uncertainty_label="HIGH"), None, None)
        self.assertEqual(out.action, "BUY")                                         # HIGH uncertainty: not a veto
        hold = Decision(action="HOLD")
        self.assertEqual(cd.refine(hold, None, ForecastResult(direction="UP", uncertainty_label="LOW"), None, None).action, "HOLD")


# ======================================================================= validator / safety gate / execution modes
class TestValidatorAndGate(unittest.TestCase):
    NOW = 1767600000.0

    def setUp(self):
        self.db = new_db()
        self.ev = EventLog(self.db, lambda: self.NOW)
        self.val = Validator(S)
        self.gov = RiskGovernor(S, self.db)
        c = {"M5": m5_series(120)}
        self.sn = make_snap(c, as_of=self.NOW, available_at=self.NOW, tick=Tick(self.NOW, 2650.0, 2650.25))
        self.q = DataQuality(True, 96.0)
        self.d = Decision(action="BUY", overall_score=75.0, buy_score=75.0, lead="BUY", confidence=72.0)
        team = {n: res(n, "BUY", 85, 80) for n in ("trend", "structure", "price_action", "momentum", "liquidity")}
        team.update(volatility=res("volatility", "HOLD"), news=res("news", "HOLD", 0, 0, dq=0))
        self.cons = ConsistencyEngine(S).evaluate(team, None, ForecastResult(direction="UP", uncertainty_label="LOW"), 96, "READY", None)

    def validate(self, **kw):
        a = dict(decision=self.d, snapshot=self.sn, quality=self.q, consistency=self.cons, forecast=ForecastResult(direction="UP", uncertainty_label="LOW"),
                 system_state="READY", now=self.NOW, usable_specialists=5)
        a.update(kw)
        return self.val.validate(**a)

    def test_good_decision_passes_and_is_signed(self):
        v = self.validate()
        self.assertTrue(v.passed, v.reasons)
        self.assertTrue(self.val.verify(v, self.sn.snapshot_id, "BUY", 75.0))
        self.assertFalse(self.val.verify(v, self.sn.snapshot_id, "SELL", 75.0))     # bound to the action
        self.assertFalse(self.val.verify(v, "other", "BUY", 75.0))                  # ... and to the snapshot
        self.assertFalse(self.val.verify(v, self.sn.snapshot_id, "BUY", 99.0))      # ... and to the score
        self.assertFalse(Validator(S).verify(v, self.sn.snapshot_id, "BUY", 75.0))  # another instance cannot forge it

    def test_rejections(self):
        cases = {
            "unsupported action": dict(decision=dataclasses.replace(self.d, action="YOLO")),
            "degraded system": dict(system_state="DEGRADED"),
            "paused system": dict(system_state="PAUSED"),
            "stale snapshot": dict(now=self.NOW + 500),
            "bad data": dict(quality=DataQuality(False, 20.0, ["STALE_TICK"])),
            "no snapshot": dict(snapshot=None),
            "few agents": dict(usable_specialists=2),
            "score below threshold": dict(decision=dataclasses.replace(self.d, overall_score=50.0)),
            "low confidence": dict(decision=dataclasses.replace(self.d, confidence=20.0)),
            "forecast contradiction": dict(forecast=ForecastResult(direction="DOWN", uncertainty_label="LOW")),
        }
        for name, kw in cases.items():
            self.assertFalse(self.validate(**kw).passed, name)
        bear = dataclasses.replace(self.cons, consensus="BEARISH")
        self.assertFalse(self.validate(consistency=bear).passed)
        hi = dataclasses.replace(self.cons, disagreement="HIGH")
        self.assertFalse(self.validate(consistency=hi).passed)

    def test_hold_needs_no_trade_checks_but_still_needs_valid_data(self):
        self.assertTrue(self.validate(decision=Decision(action="HOLD"), system_state="DEGRADED").passed)      # HOLD is always safe
        self.assertFalse(self.validate(decision=Decision(action="HOLD"), quality=DataQuality(False, 10.0, ["NO_CANDLES"])).passed)

    def eval_kwargs(self):
        return dict(account=acct(), spec=SPEC, tick=self.sn.tick, positions=[], now=self.NOW, new_trades_allowed=True, health_ok=True,
                    vol_class="NORMAL", eq={"peak": 800, "day_start": 800, "drawdown_pct": 0.0, "daily_loss_pct": 0.0, "daily_pnl": 0.0},
                    stats={"trades_today": 0, "reversals_today": 0, "consecutive_losses": 0, "pnl_today": 0.0, "last_open": None,
                           "last_close": None, "last_loss_close": None, "last_reversal": None})

    def req(self):
        return TradeRequest("XAUUSD", "BUY", 2650.25, 2646.25, 2656.25, 2.0, "c-1", 1)

    def test_gate_requires_validator_approval_then_risk(self):
        gate = SafetyGate(S, self.gov, self.val, self.ev, "SIMULATION")
        none = gate.authorize(self.req(), None, self.d, **self.eval_kwargs())
        self.assertFalse(none.approved)
        self.assertEqual(none.token, "")
        failed = self.validate(system_state="DEGRADED")
        self.assertFalse(gate.authorize(self.req(), failed, self.d, **self.eval_kwargs()).approved)
        ok = gate.authorize(self.req(), self.validate(), self.d, **self.eval_kwargs())
        self.assertTrue(ok.approved, ok.reasons)
        self.assertTrue(self.gov.verify(self.req(), ok, self.NOW))
        forged = dataclasses.replace(self.validate(), token="x" * 64)
        self.assertFalse(gate.authorize(self.req(), forged, self.d, **self.eval_kwargs()).approved)

    def test_disabled_mode_suppresses_and_logs(self):
        gate = SafetyGate(S, self.gov, self.val, self.ev, "DISABLED")
        r = gate.authorize(self.req(), self.validate(), self.d, **self.eval_kwargs())
        self.assertFalse(r.approved)
        self.assertEqual(r.token, "")
        evs = [json.loads(x["payload_json"]) for x in self.ev.tail(10, "SIMULATION_ACTION")]
        self.assertTrue(any(e["status"] == "SUPPRESSED_DISABLED" for e in evs))

    def test_reducing_actions_only_from_trusted_sources(self):
        gate = SafetyGate(S, self.gov, self.val, self.ev, "SIMULATION")
        self.assertTrue(gate.check_reducing("CLOSE", "operator", 1, self.NOW))
        self.assertTrue(gate.check_reducing("MODIFY_SL", "position_manager", 1, self.NOW))
        self.assertFalse(gate.check_reducing("CLOSE", "agent", 1, self.NOW))
        self.assertFalse(gate.check_reducing("CLOSE", "learning", 1, self.NOW))
        self.assertFalse(gate.check_reducing("OPEN", "operator", 1, self.NOW))      # opening is never a 'reducing' action

    def test_execution_engine_disabled_mode_never_calls_broker(self):
        b = PaperBroker(generate_m1(300))
        for _ in range(100): b.advance()
        gov = RiskGovernor(S, self.db)
        t = b.tick("XAUUSD")
        req = TradeRequest("XAUUSD", "BUY", t.ask, t.ask - 4, t.ask + 6, 2.0, "dis-1", None, False, {})
        ap = gov.evaluate(req, account=b.account(), spec=SPEC, tick=t, positions=[], now=b.now(), new_trades_allowed=True, health_ok=True,
                          vol_class="NORMAL", eq=self.eval_kwargs()["eq"], stats=self.eval_kwargs()["stats"])
        self.assertTrue(ap.approved, ap.reasons)
        x = ExecutionEngine(b, self.db, S, gov, b.now, events=self.ev, mode="DISABLED")
        self.assertFalse(x.open_position(req, ap, "v1.0").ok)
        self.assertEqual(b.positions(), [])
        self.assertTrue(any(json.loads(e["payload_json"])["status"] == "SUPPRESSED_DISABLED" for e in self.ev.tail(5, "SIMULATION_ACTION")))


# ======================================================================= event log / state manager / supervisor
class TestAuditAndSupervision(unittest.TestCase):
    def test_event_chain_and_tamper_detection(self):
        db = new_db(); ev = EventLog(db, lambda: 1.0)
        for i in range(5):
            ev.emit("c", "DECISION_CREATED", {"i": i}, "snap1")
        ok, n = ev.verify_chain()
        self.assertEqual((ok, n), (True, 5))
        rows = ev.tail(1)[0]
        for k in ("seq", "event_id", "ts", "snapshot_id", "component", "event_type", "payload_json", "version"):
            self.assertIn(k, rows)
        db.execute("UPDATE events SET payload_json='{\"i\":99}' WHERE seq=2")
        self.assertFalse(ev.verify_chain()[0])

    def test_event_digest_is_deterministic(self):
        d = []
        for _ in range(2):
            db = new_db(); ev = EventLog(db, lambda: 5.0)
            ev.emit("a", "STARTUP", {"x": 1}); ev.emit("b", "DECISION_CREATED", {"y": [1, 2]})
            d.append(ev.digest())
        self.assertEqual(d[0], d[1])

    def test_state_manager(self):
        db = new_db(); ev = EventLog(db, lambda: 0.0)
        sm = SystemStateManager(ev, lambda: 0.0)
        self.assertEqual(sm.state, "STARTING")
        self.assertTrue(sm.transition("READY", "ok", "engine"))
        with self.assertRaises(PermissionError):
            sm.transition("PAUSED", "x", "trend_agent")                              # agents cannot mutate state
        sm.transition("ANALYZING", "cycle", "engine"); sm.transition("READY", "done", "engine")
        logged = [json.loads(e["payload_json"]) for e in ev.tail(20, "SYSTEM_STATE")]
        self.assertFalse(any(e.get("to") == "ANALYZING" for e in logged))            # high-frequency toggles stay quiet
        sm.transition("ERROR", "boom", "engine")
        self.assertFalse(sm.transition("READY", "skip recovery", "engine"))         # ERROR -> READY is illegal
        self.assertTrue(sm.transition("RECOVERY", "retry", "recovery"))
        self.assertTrue(sm.transition("READY", "ok", "recovery"))
        snap = sm.snapshot()
        self.assertEqual((snap["state"], snap["previous"]), ("READY", "RECOVERY"))
        sm.transition("SHUTDOWN", "bye", "engine")
        self.assertFalse(sm.transition("READY", "no way", "engine"))

    def test_supervisor_spec_example_and_levels(self):
        sup = SystemSupervisor()
        a = sup.assess(data_ok=False, data_issues=["STALE_TICK"], unavailable=["trend", "structure", "momentum"], forecast_uncertainty="HIGH",
                       watchdog={}, config_ok=True, broker_ok=True, db_ok=True)
        self.assertEqual(a.level, "NOT_READY")
        self.assertTrue(any("stale data" in r for r in a.reasons))
        b = sup.assess(data_ok=False, data_issues=["STALE"], unavailable=[], forecast_uncertainty="LOW", watchdog={}, config_ok=True, broker_ok=True, db_ok=True)
        self.assertEqual(b.level, "DEGRADED")
        c = sup.assess(data_ok=True, data_issues=[], unavailable=[], forecast_uncertainty="LOW", watchdog={}, config_ok=True, broker_ok=True, db_ok=True)
        self.assertEqual(c.level, "READY")
        self.assertEqual(sup.assess(data_ok=True, data_issues=[], unavailable=[], forecast_uncertainty="LOW", watchdog={}, config_ok=True,
                                    broker_ok=False, db_ok=True).level, "NOT_READY")

    def test_watchdog_runs_probes_and_survives_crashes(self):
        w = Watchdog(lambda: 1.0)
        w.register("ok", lambda: ("OK", "fine")); w.register("bad", lambda: 1 / 0)
        r = w.run_all()
        self.assertEqual(r["ok"]["status"], "OK")
        self.assertEqual(r["bad"]["status"], "FAIL")

    def test_module_isolation_and_recovery_after_cooldown(self):
        db = new_db(); ev = EventLog(db, lambda: 0.0); t = [0.0]
        iso = ModuleIsolation(S, ev, lambda: t[0]); calls = []
        def boom(): calls.append(1); raise RuntimeError("down")
        fb = lambda why: ("fallback", why)
        for _ in range(S.module_fail_limit):
            self.assertEqual(iso.call("m", boom, fb)[0], "fallback")
        self.assertEqual(iso.unavailable(), ["m"])
        n = len(calls)
        iso.call("m", boom, fb)
        self.assertEqual(len(calls), n)                                              # isolated: not even called
        t[0] += S.module_cooldown_sec + 1
        self.assertEqual(iso.call("m", lambda: "fine", fb), "fine")
        self.assertEqual(iso.unavailable(), [])
        self.assertTrue(ev.tail(5, "MODULE_ISOLATED"))

    def test_recovery_manager_retries_and_logs(self):
        db = new_db(); ev = EventLog(db, lambda: 0.0); rm = RecoveryManager(ev, lambda: 0.0, sleep=lambda s: None); n = [0]
        def flaky():
            n[0] += 1
            if n[0] < 3: raise ConnectionError("x")
            return "up"
        self.assertEqual(rm.attempt("broker", flaky), (True, "up"))
        ok, err = rm.attempt("db", lambda: 1 / 0, retries=2)
        self.assertFalse(ok)
        self.assertEqual(len(ev.tail(10, "RECOVERY_STARTED")), 2)
        self.assertEqual(len(ev.tail(10, "RECOVERY_COMPLETED")), 2)


# ======================================================================= drift / learning manager / experiments
class TestDriftAndLearning(unittest.TestCase):
    def test_psi_and_tv(self):
        rnd = random.Random(0)
        a = [rnd.gauss(0, 1) for _ in range(500)]
        b = [rnd.gauss(0, 1) for _ in range(500)]
        c = [rnd.gauss(3, 1) for _ in range(500)]
        self.assertLess(psi(a, b), 0.1)
        self.assertGreater(psi(a, c), 0.25)
        from collections import Counter
        self.assertEqual(tv_distance(Counter("AB"), Counter("AB")), 0.0)
        self.assertEqual(tv_distance(Counter("AAAA"), Counter("BBBB")), 1.0)

    def test_drift_detector_flags_shift_not_noise(self):
        s = dataclasses.replace(S, drift_window=60)
        db = new_db(); ev = EventLog(db, lambda: 0.0); rnd = random.Random(2)
        quiet = DriftDetector(s, ev); shifted = DriftDetector(s, ev)
        for i in range(120):
            f0 = {k: rnd.gauss(0, 1) for k in FEATURES}
            f1 = {k: rnd.gauss(0 if i < 60 else 4, 1) for k in FEATURES}
            quiet.observe(f0, "TRENDING", "UP", {"trend": "BUY"})
            shifted.observe(f1, "TRENDING" if i < 60 else "RANGING", "UP" if i < 60 else "DOWN", {"trend": "BUY" if i < 60 else "SELL"})
        self.assertEqual(quiet.check(), [])
        flags = shifted.check()
        kinds = {f["kind"] for f in flags}
        self.assertTrue({"feature_distribution", "regime_distribution", "forecast_behavior", "agent_behavior"} <= kinds)
        self.assertTrue(ev.tail(5, "DRIFT_DETECTED"))
        for i in range(60):
            shifted.observe_error(0.0)
        for i in range(60):
            shifted.observe_error(1.0)
        self.assertTrue(any(f["kind"] == "model_error" for f in shifted.check()))

    def make_lm(self):
        db = new_db(); ev = EventLog(db, lambda: 0.0); reg = ModelRegistry(db, S, ev)
        fs = ForecastSystem(db, S, reg, ev)
        lm = LearningManager(db, S, ev, reg, fs, DriftDetector(S, ev), LearningEngine(db, S), ExperimentLab(db, S, ev))
        return lm, db, reg

    def fill(self, db, n, chal_p, champ_p, ver="1.1", seed=3):
        rnd = random.Random(seed)
        for i in range(n):
            def add(role, mid, v, ok):
                db.insert("forecast_log", dict(ts="t", snapshot_id=f"s{i}", model_id=mid, model_version=v, role=role, direction="UP", horizon_bars=12,
                                               timeframe="M5", regime="TRENDING" if i % 2 else "RANGING", entry_price=1, entry_time=i, atr=1,
                                               actual="UP" if ok else "DOWN", correct=int(ok)))
            add("CHALLENGER", "logistic", ver, rnd.random() < chal_p)
            add("MEMBER", "logistic", "1.0", rnd.random() < champ_p)
            for b in ("naive_majority", "persistence", "stat_drift"):
                add("BASELINE", b, "1.0", rnd.random() < 0.5)

    def test_shadow_evaluation_collects_then_validates(self):
        lm, db, reg = self.make_lm()
        reg.register("logistic", "1.1", "CHALLENGER", {"lr": .1}); lm.forecaster.reload()
        self.fill(db, 20, .9, .4)
        self.assertTrue(lm.evaluate_challengers()[0]["decision"].startswith("COLLECTING"))
        self.fill_more = None
        db.execute("DELETE FROM forecast_log")
        self.fill(db, 260, .8, .45)
        out = lm.evaluate_challengers()[0]
        self.assertEqual(out["decision"], "VALIDATED", out)
        self.assertEqual(reg.rows("VALIDATED", "logistic")[0]["version"], "1.1")
        self.assertEqual({r["version"]: r["status"] for r in reg.rows(model_id="logistic")}["1.0"], "CHAMPION")    # NOT promoted automatically
        lm.promote("logistic", "1.1", "Ehsan", "shadow evaluation passed")
        self.assertEqual({r["version"]: r["status"] for r in reg.rows(model_id="logistic")}, {"1.0": "STABLE", "1.1": "CHAMPION"})
        self.assertEqual(lm.rollback("logistic", "test"), "1.0")

    def test_shadow_evaluation_rejects_worse_challenger(self):
        lm, db, reg = self.make_lm()
        reg.register("logistic", "1.1", "CHALLENGER", {"lr": .1}); lm.forecaster.reload()
        self.fill(db, 300, .35, .65)
        self.assertEqual(lm.evaluate_challengers()[0]["decision"], "REJECTED")
        self.assertEqual({r["version"]: r["status"] for r in reg.rows(model_id="logistic")}["1.0"], "CHAMPION")

    def test_candidate_goes_through_offline_replay_before_shadow(self):
        lm, db, reg = self.make_lm()
        r = lm.propose_challenger(generate_m1(9000, seed=8))
        self.assertIn(r["status"], ("CHALLENGER", "REJECTED"))
        self.assertIn("oos_samples", r["offline"])
        self.assertEqual({x["version"]: x["status"] for x in reg.rows(model_id="logistic")}["1.0"], "CHAMPION")   # champion untouched
        ex = ExperimentLab(db, S).list()[0]
        for k in ("experiment_id", "hypothesis", "code_version", "data_version", "feature_version", "params_json", "eval_range", "status"):
            self.assertTrue(ex[k] is not None and ex[k] != "", k)
        self.assertEqual(ex["status"], "COMPLETED")

    def test_holdout_guard_stops_repeated_tuning_on_same_set(self):
        db = new_db(); lab = ExperimentLab(db, dataclasses.replace(S, max_eval_reuse=2))
        lab.start("h1", {}, "d", "r", eval_ref="oos-A")
        lab.start("h2", {}, "d", "r", eval_ref="oos-A")
        with self.assertRaises(HoldoutExhausted):
            lab.start("h3", {}, "d", "r", eval_ref="oos-A")
        lab.start("h3", {}, "d", "r", eval_ref="oos-B")                              # fresh data is fine
        with self.assertRaises(ValueError):
            lab.start("  ", {}, "d", "r")                                            # a hypothesis is mandatory

    def test_settings_validation_for_astra_fields(self):
        for bad in ({"GOLDBOT_EXECUTION_MODE": "LIVE"}, {"GOLDBOT_FORECAST_TIMEFRAME": "W1"}, {"GOLDBOT_ACCOUNT_DENOMINATION": "MEGA"}):
            with self.assertRaises(ConfigError):
                load_settings(bad)
        self.assertEqual(load_settings({}).execution_mode, "SIMULATION")


if __name__ == "__main__":
    unittest.main()
