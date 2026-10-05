import dataclasses, http.client, json, os, socket, tempfile, unittest
from goldbot.api import ApiServer
from goldbot.backtest import metrics, run_backtest
from goldbot.synth import generate_m1
from tests.helpers import S, make_engine

ADMIN, VIEW = "a" * 24, "v" * 24


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e, cls.b, cls.db = make_engine(4500)
        cls.e.step(cls.b.now())
        cls.port = free_port()
        s = dataclasses.replace(S, api_token=ADMIN, viewer_token=VIEW, api_port=cls.port, cors_origins=("https://ok.example",))
        cls.api = ApiServer(cls.e, cls.db, s, cls.e.learning); cls.api.serve_background()

    @classmethod
    def tearDownClass(cls):
        cls.api.shutdown()

    def call(self, method, path, token=None, body=None, headers=None, raw=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        h = dict(headers or {})
        if token: h["Authorization"] = "Bearer " + token
        c.request(method, path, raw if raw is not None else (json.dumps(body) if body is not None else None), h)
        r = c.getresponse(); data = r.read(); c.close()
        try: j = json.loads(data)
        except ValueError: j = None
        return r.status, j, r

    def test_auth(self):
        self.assertEqual(self.call("GET", "/api/v1/snapshot")[0], 401)
        self.assertEqual(self.call("GET", "/api/v1/snapshot", "wrong")[0], 401)
        self.assertEqual(self.call("GET", "/api/v1/snapshot", VIEW)[0], 200)
        self.assertEqual(self.call("GET", "/api/v1/snapshot", ADMIN)[0], 200)
        self.assertEqual(self.call("GET", "/healthz")[0], 200)

    def test_viewer_cannot_control(self):
        self.assertEqual(self.call("POST", "/api/v1/control/emergency-stop", VIEW, {"request_id": "viewer-req-1"})[0], 403)
        self.assertNotEqual(self.e.sm.state.value, "EMERGENCY_STOP")

    def test_request_validation_and_idempotency(self):
        self.assertEqual(self.call("POST", "/api/v1/control/pause", ADMIN, {})[0], 400)
        self.assertEqual(self.call("POST", "/api/v1/control/pause", ADMIN, {"request_id": "x"})[0], 400)
        self.assertEqual(self.call("POST", "/api/v1/control/pause", ADMIN, raw="not json")[0], 400)
        self.assertEqual(self.call("POST", "/api/v1/control/pause", ADMIN, raw="[1,2]")[0], 400)
        import threading, time
        st = []
        t = threading.Thread(target=lambda: st.append(self.call("POST", "/api/v1/control/pause", ADMIN, {"request_id": "idem-req-0001"})[:2])); t.start()
        for _ in range(100):
            self.e.step(self.b.now())
            if not t.is_alive(): break
            time.sleep(0.02)
        t.join()
        self.assertEqual(st[0][0], 200)
        self.assertEqual(self.call("POST", "/api/v1/control/pause", ADMIN, {"request_id": "idem-req-0001"})[0], 409)

    def test_body_too_large(self):
        self.assertEqual(self.call("POST", "/api/v1/control/pause", ADMIN, raw="x" * 20000)[0], 413)

    def test_cors_allowlist(self):
        _, _, r = self.call("GET", "/healthz", headers={"Origin": "https://ok.example"})
        self.assertEqual(r.getheader("Access-Control-Allow-Origin"), "https://ok.example")
        _, _, r = self.call("GET", "/healthz", headers={"Origin": "https://evil.example"})
        self.assertIsNone(r.getheader("Access-Control-Allow-Origin"))
        self.assertEqual(self.call("OPTIONS", "/api/v1/snapshot", headers={"Origin": "https://evil.example"})[0], 403)

    def test_config_hides_secrets_and_headers(self):
        st, j, r = self.call("GET", "/api/v1/config", VIEW)
        self.assertNotIn("api_token", j); self.assertNotIn("viewer_token", j)
        self.assertNotIn(ADMIN, json.dumps(j))
        self.assertEqual(r.getheader("X-Content-Type-Options"), "nosniff")

    def test_bad_query_values(self):
        self.assertEqual(self.call("GET", "/api/v1/trades?status=HACK", VIEW)[0], 400)
        self.assertEqual(self.call("GET", "/api/v1/logs?level=NOPE", VIEW)[0], 400)

    def test_zz_rate_limit_on_bad_tokens(self):
        codes = [self.call("GET", "/api/v1/snapshot", "bad%d" % i)[0] for i in range(14)]
        self.assertIn(429, codes)

    def test_short_token_refuses_to_start(self):
        with self.assertRaises(RuntimeError):
            ApiServer(self.e, self.db, dataclasses.replace(S, api_token="short", api_port=free_port()), self.e.learning)


class TestBacktestAndCli(unittest.TestCase):
    def test_metrics(self):
        t = [{"pnl": 10, "r_multiple": 1.5, "exit_reason": "TP"}, {"pnl": -5, "r_multiple": -1.0, "exit_reason": "SL"}]
        m = metrics(t, [800, 810, 805], 800)
        self.assertEqual(m["trades"], 2); self.assertEqual(m["profit_factor"], 2.0); self.assertEqual(m["win_rate_pct"], 50.0)
        self.assertAlmostEqual(m["expectancy_R"], 0.25)

    def test_metrics_marks_small_sample_as_insufficient_data(self):
        t = [{"pnl": 10, "r_multiple": 1.5, "exit_reason": "TP"}]
        m = metrics(t, [800, 810], 800)
        self.assertEqual(m["evaluation_status"], "INSUFFICIENT_DATA")
        self.assertFalse(m["statistically_meaningful"])
        self.assertEqual(m["minimum_trades"], 30)

    def test_short_data_rejected(self):
        with self.assertRaises(ValueError): run_backtest(generate_m1(500), S)

    def test_backtest_runs_through_real_engine(self):
        r = run_backtest(generate_m1(5200, seed=3), S)
        for k in ("trades", "net_profit", "profit_factor", "max_drawdown_pct", "final_equity", "bars"):
            self.assertIn(k, r)

    def test_cli_initdb(self):
        from goldbot.__main__ import main
        with tempfile.TemporaryDirectory() as d:
            os.environ["GOLDBOT_DB_PATH"] = os.path.join(d, "t.db")
            try: self.assertEqual(main(["initdb"]), 0)
            finally: del os.environ["GOLDBOT_DB_PATH"]

    def test_cli_rejects_real_account_config(self):
        from goldbot.__main__ import main
        os.environ["GOLDBOT_ALLOW_REAL_ACCOUNT"] = "true"
        try: self.assertEqual(main(["initdb"]), 2)
        finally: del os.environ["GOLDBOT_ALLOW_REAL_ACCOUNT"]


if __name__ == "__main__":
    unittest.main()
