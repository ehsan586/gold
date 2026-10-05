import contextlib, dataclasses, http.client, io, json, os, re, socket, subprocess, tempfile, threading, time, unittest
from pathlib import Path
from goldbot.api import ApiServer
from goldbot.broker import BrokerError
from goldbot.broker_mt5 import Mt5Broker
from goldbot.broker_paper import PaperBroker
from goldbot.broker_sim import ShadowBroker
from goldbot.engine import TradingEngine
from goldbot.experiments import ExperimentLab, HoldoutExhausted
from goldbot.replay import ReplayEngine, selftest
from goldbot.settings import load_settings
from goldbot.synth import generate_m1
from tests.helpers import S, make_engine, new_db


class TestEngineWiring(unittest.TestCase):
    def test_one_snapshot_feeds_everything_in_a_cycle(self):
        e, b, db = make_engine(4200)
        ctxs, calls = [], [0]
        for a in e.agents:
            orig = a.analyze
            def spy(ctx, orig=orig): ctxs.append((id(ctx), id(ctx.candles), id(ctx.tick))); return orig(ctx)
            a.analyze = spy
        real = b.candles
        def counting(sym, tf, n): calls[0] += 1; return real(sym, tf, n)
        b.candles = counting
        b.advance(); e.step(b.now())
        self.assertEqual(len(set(ctxs)), 1)                                   # all 7 specialists got the SAME context object
        self.assertEqual(len(ctxs), 7)
        self.assertEqual(calls[0], len(e.dataservice.timeframes()))           # market data fetched exactly once per timeframe
        self.assertEqual(e._snapshot.snapshot_id, e.sysstate.last_snapshot_id)

    def test_system_state_degrades_and_recovers(self):
        e, b, db = make_engine(4300)
        self.assertEqual(e.sysstate.state, "READY")
        b.advance(); e.step(b.now())
        self.assertEqual(e.sysstate.state, "READY")
        good_tick = b.tick
        b.tick = lambda s: (_ for _ in ()).throw(RuntimeError("feed down"))
        b.advance(); e.step(b.now())
        self.assertEqual(e.sysstate.state, "DEGRADED")
        b.tick = good_tick
        for _ in range(3):
            b.advance(); e.step(b.now())
        self.assertEqual(e.sysstate.state, "READY")
        types = [json.loads(r["payload_json"]).get("to") for r in e.events.tail(30, "SYSTEM_STATE")]
        self.assertIn("DEGRADED", types)

    def test_agent_crash_is_isolated_not_fatal(self):
        e, b, db = make_engine(4300)
        e.agents[0].analyze = lambda ctx: 1 / 0
        for _ in range(S.module_fail_limit + 1):
            b.advance(); e.step(b.now())
        self.assertIn("trend", e.isolation.unavailable())
        self.assertTrue(e._results["trend"].is_no_data)
        self.assertTrue(e.events.tail(5, "MODULE_ISOLATED"))
        self.assertEqual(e.sm.state.value in ("IDLE", "ANALYZING"), True)       # the system keeps running
        self.assertEqual(db.query("SELECT COUNT(*) c FROM trades")[0]["c"], 0)

    def test_ablation_switches(self):
        e, b, db = make_engine(4300)
        e.disabled_agents = {"trend"}; e.use_regime = False; e.use_forecast = False
        b.advance(); e.step(b.now())
        self.assertTrue(e._results["trend"].extra.get("ablated"))
        self.assertNotIn("trend", e._consistency.missing_agents)               # ablated != "missing data"

    def test_pause_survives_restart(self):
        e, b, db = make_engine(4300)
        out = []
        t = threading.Thread(target=lambda: out.append(e.submit("pause"))); t.start()
        while t.is_alive(): e.step(b.now()); time.sleep(0.01)
        self.assertEqual(e.sysstate.state, "PAUSED")
        e2 = TradingEngine(S, db, b, clock=b.now, sim=True); e2.start()
        self.assertTrue(e2.sm.paused)
        self.assertEqual(e2.sysstate.state, "PAUSED")

    def test_final_build_rejects_real_execution_mode(self):
        class FakeMt5(PaperBroker):
            name = "mt5"
        b = FakeMt5(generate_m1(300))
        for _ in range(100): b.advance()
        e = TradingEngine(dataclasses.replace(S, execution_mode="MT5_DEMO"), new_db(), b, clock=b.now, sim=True)
        with self.assertRaises(Exception):
            e.start()

    def test_mt5_adapter_fails_cleanly_without_the_package(self):
        with self.assertRaises(BrokerError):
            Mt5Broker().connect() if not _has_mt5() else (_ for _ in ()).throw(BrokerError("skip"))


def _has_mt5():
    try:
        import MetaTrader5  # noqa: F401
        return True
    except ImportError:
        return False


class TestExecutionModes(unittest.TestCase):
    def test_disabled_mode_analyses_but_never_executes(self):
        s = dataclasses.replace(S, execution_mode="DISABLED")
        e, b, db = make_engine(7500, seed=5, settings=s)
        found = False
        while b.advance():
            e.step(b.now())
            if e.events.tail(1, "SIMULATION_ACTION"):
                found = any(json.loads(x["payload_json"]).get("status") == "SUPPRESSED_DISABLED" for x in e.events.tail(5, "SIMULATION_ACTION"))
                if found: break
        self.assertTrue(found, "a persistent signal should have been logged as suppressed")
        self.assertEqual(db.query("SELECT COUNT(*) c FROM trades")[0]["c"], 0)
        self.assertEqual(b.positions(), [])
        self.assertGreater(db.count("decisions"), 0)                            # research output still produced

    def test_simulation_on_live_data_uses_a_virtual_account_only(self):
        db = new_db()
        data = PaperBroker(generate_m1(7500, seed=5))
        for _ in range(3700): data.advance()
        shadow = ShadowBroker(data, db, 800.0, data.now, "XAUUSD")
        e = TradingEngine(S, db, shadow, clock=data.now, sim=True)
        e.start()
        opened = False
        while data.advance():
            e.step(data.now())
            if db.query("SELECT 1 FROM trades WHERE status='OPEN' OR status='CLOSED'"):
                opened = True
                if shadow.positions() or db.query("SELECT 1 FROM trades WHERE status='CLOSED'"): break
        self.assertTrue(opened)
        self.assertEqual(data.positions(), [])                                  # the data broker never received an order
        self.assertEqual(data.balance, 800.0)
        tk = db.query("SELECT ticket FROM trades ORDER BY id")[0]["ticket"]
        self.assertGreaterEqual(tk, 5000)                                       # virtual ticket range
        shadow2 = ShadowBroker(data, db, 800.0, data.now, "XAUUSD")             # virtual state is persisted
        self.assertEqual(len(shadow2.positions()), len(shadow.positions()))
        self.assertEqual(shadow2.balance, shadow.balance)


class TestReplayAndExperiments(unittest.TestCase):
    def test_replay_is_deterministic_without_lookahead(self):
        r = selftest(S)
        self.assertTrue(r["ok"], r)

    def test_reconstruct_what_the_system_knew(self):
        m1 = generate_m1(3900, seed=2)
        out = ReplayEngine(S).reconstruct(m1, 3850)
        self.assertEqual(out["as_of"], m1[3850].time + 60)
        for k in ("decision", "forecast", "regime", "consistency", "validation", "diversity", "data_quality", "snapshot"):
            self.assertIn(k, out)
        self.assertTrue(out["data_quality"]["ok"])

    def test_ablation_records_experiment_and_guards_holdout(self):
        db = new_db()
        lab = ExperimentLab(db, dataclasses.replace(S, max_eval_reuse=1))
        m1 = generate_m1(4000, seed=5)
        out = lab.run_ablation(m1, S, variants=["FULL", "WITHOUT_FORECAST", "WITHOUT_TREND"], split=0.5, warmup=3700)
        self.assertEqual(set(out["results"]), {"FULL", "WITHOUT_FORECAST", "WITHOUT_TREND"})
        for n, r in out["results"].items():
            if n != "FULL":
                self.assertIn("verdict", r["delta_vs_full"])
        ex = lab.list()[0]
        self.assertEqual((ex["status"], ex["experiment_id"]), ("COMPLETED", out["experiment_id"]))
        self.assertNotIn("r_values", ex["validation_json"])
        with self.assertRaises(HoldoutExhausted):
            lab.run_ablation(m1, S, variants=["FULL"], split=0.5, warmup=3700)         # same evaluation set a second time


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class TestAstraApiAndDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e, cls.b, cls.db = make_engine(4400)
        for _ in range(3):
            cls.b.advance(); cls.e.step(cls.b.now())
        cls.port = free_port()
        s = dataclasses.replace(S, api_token="a" * 24, viewer_token="v" * 24, api_port=cls.port)
        cls.api = ApiServer(cls.e, cls.db, s, cls.e.learning); cls.api.serve_background()

    @classmethod
    def tearDownClass(cls):
        cls.api.shutdown()

    def call(self, method, path, token="v" * 24, body=None, step=False):
        out = []
        def go():
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=70)
            c.request(method, path, json.dumps(body) if body is not None else None, {"Authorization": "Bearer " + token} if token else {})
            r = c.getresponse(); data = r.read(); c.close()
            out.append((r.status, data, r))
        t = threading.Thread(target=go); t.start()
        while step and t.is_alive():
            self.e.step(self.b.now()); time.sleep(0.02)
        t.join()
        st, data, r = out[0]
        try: return st, json.loads(data), r
        except ValueError: return st, data, r

    def test_new_read_endpoints(self):
        for p in ("/api/v1/health", "/api/v1/metrics", "/api/v1/events?limit=5", "/api/v1/events/verify", "/api/v1/candles?tf=M5&limit=20",
                  "/api/v1/forecasts", "/api/v1/models", "/api/v1/drift", "/api/v1/experiences", "/api/v1/experiments", "/api/v1/replay", "/api/v1/units"):
            st, j, _ = self.call("GET", p)
            self.assertEqual(st, 200, p)
        st, j, _ = self.call("GET", "/api/v1/health")
        for k in ("status", "system", "assessment", "components", "watchdog", "isolation"):
            self.assertIn(k, j)
        self.assertTrue(self.call("GET", "/api/v1/events/verify")[1]["chain_ok"])
        c = self.call("GET", "/api/v1/candles?tf=M5&limit=20")[1]["candles"]
        self.assertTrue(0 < len(c) <= 20)
        self.assertEqual(len(c[0]), 6)
        self.assertEqual(self.call("GET", "/api/v1/events?limit=500&type=NOT_A_TYPE")[0], 200)

    def test_endpoints_require_auth(self):
        self.assertEqual(self.call("GET", "/api/v1/health", token=None)[0], 401)

    def test_model_promotion_needs_admin_and_a_validated_challenger(self):
        body = {"request_id": "promo-req-0001", "model_id": "logistic", "version": "1.0", "approver": "me", "evidence": "x"}
        self.assertEqual(self.call("POST", "/api/v1/models/promote", token="v" * 24, body=body)[0], 403)   # viewer cannot
        st, j, _ = self.call("POST", "/api/v1/models/promote", token="a" * 24, body={**body, "request_id": "promo-req-0002"}, step=True)
        self.assertEqual(st, 400)                                                                          # 1.0 is a CHAMPION, not VALIDATED
        self.assertIn("VALIDATED", j["error"])

    def test_dashboard_is_served_with_security_headers(self):
        st, data, r = self.call("GET", "/", token=None)
        self.assertEqual(st, 200)
        self.assertIn("default-src 'self'", r.getheader("Content-Security-Policy"))
        self.assertIn(b"ASTRA TRADER", data)

    def test_dashboard_matches_the_spec_and_contains_no_fake_telemetry(self):
        html = (Path(__file__).parent.parent / "goldbot" / "dashboard.html").read_text(encoding="utf-8")
        low = html.lower()
        for needed in ("ASTRA TRADER", "AI • ANALYSIS • RESEARCH", "MISSION CONTROL", "ANALYZE › PREDICT › EVALUATE", "9 Intelligence Agents",
                       "Market Overview", "Market Regime", "Forecast &amp; Uncertainty", "Active Position", "Recent Decisions", "Anomaly &amp; Risk",
                       "System Status", "Mission Log", "Quick Actions", "Safety Gate", "Diversity Monitor", "Forecast Ensemble", "Data Layer",
                       "Memory", "Learning", "decorative", "EXECUTION:", "Chart &amp; Analysis"):
            self.assertTrue(needed.lower() in low, "missing UI element: " + needed)       # (never dump the whole page on failure)
        for item in ("Overview", "Agents", "Market", "Analysis", "Memory", "Replay", "Learning", "Experiments", "Settings"):
            self.assertIn(f'"{item}"', html)
        for fake in ("EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "Order sent to MT4", "12,843", "Confirmed"):
            self.assertNotIn(fake, html, f"hard-coded sample data found: {fake}")
        self.assertIsNone(re.search(r"(api[_-]?key|password)\s*[:=]\s*['\"][^'\"]{8,}", html, re.I))
        if subprocess.run(["which", "node"], capture_output=True).returncode == 0:
            js = re.search(r"<script>(.*)</script>", html, re.S).group(1)
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
                f.write(js)
            r = subprocess.run(["node", "--check", f.name], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)


class TestCli(unittest.TestCase):
    def run_cli(self, *args):
        from goldbot.__main__ import main
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(buf):
            os.environ["GOLDBOT_DB_PATH"] = os.path.join(d, "t.db")
            try:
                try: main(list(args))
                except SystemExit as e: return e.code, buf.getvalue()
            finally: del os.environ["GOLDBOT_DB_PATH"]
        return 0, buf.getvalue()

    def test_models_experiments_and_selftest(self):
        code, out = self.run_cli("models")
        self.assertEqual(code, 0)
        self.assertIn("CHAMPION", out)
        self.assertEqual(self.run_cli("experiments")[0], 0)
        code, out = self.run_cli("selftest")
        self.assertEqual(code, 0)
        self.assertIn('"deterministic": true', out)

    def test_promote_unvalidated_is_refused(self):
        from goldbot.__main__ import main
        with tempfile.TemporaryDirectory() as d:
            os.environ["GOLDBOT_DB_PATH"] = os.path.join(d, "t.db")
            try:
                main(["models"]) if False else None
                with contextlib.redirect_stdout(io.StringIO()):
                    main(["models"])
                with self.assertRaises(ValueError):
                    main(["promote", "logistic", "1.0", "--approver", "me", "--evidence", "x"])
            finally: del os.environ["GOLDBOT_DB_PATH"]


if __name__ == "__main__":
    unittest.main()
