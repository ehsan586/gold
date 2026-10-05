import json, os, tempfile, unittest
from goldbot.agents import AgentContext, CalendarFile, Agent, default_agents, VolatilityAgent, NewsAgent
from goldbot.data import Candle, Tick
from goldbot.models import AgentResult
from tests.helpers import S, SPEC, flat_candles

NOW = 1767571200 + 120 * 900


def ctx(candles=None, news=None, now=NOW):
    cs = candles if candles is not None else {tf: flat_candles(150) for tf in ("M1", "M5", "M15", "H1", "H4")}
    return AgentContext("XAUUSD", cs, Tick(now, 2650.0, 2650.25), SPEC, S, now, news)


class TestAgents(unittest.TestCase):
    def test_exactly_seven_agents(self):
        self.assertEqual([a.name for a in default_agents()],
                         ["trend", "structure", "price_action", "momentum", "volatility", "liquidity", "news"])

    def test_no_data_when_candles_missing(self):
        for a in default_agents():
            r = a.run(ctx({}))
            self.assertIsInstance(r, AgentResult)
            self.assertTrue(r.is_no_data, a.name)

    def test_agent_exception_becomes_no_data(self):
        class Bad(Agent):
            name = "trend"
            def analyze(self, c): raise RuntimeError("boom")
        r = Bad().run(ctx())
        self.assertTrue(r.is_no_data)
        self.assertIn("boom", r.reason)

    def test_all_results_serializable(self):
        for a in default_agents():
            r = a.run(ctx())
            self.assertEqual(AgentResult.from_json(r.to_json()).to_dict(), r.to_dict())

    def test_volatility_extreme_on_spike_blocks(self):
        c = flat_candles(150, rng=1.0)
        c[-1] = Candle(c[-1].time, 2650, 2665, 2635, 2650, 1)        # 30 range vs ATR~1
        r = VolatilityAgent().run(ctx({"M15": c, "M5": c, "H1": c, "H4": c, "M1": c}))
        self.assertEqual(r.extra["vol_class"], "EXTREME")
        self.assertTrue(r.extra["block"])

    def test_news_never_invented(self):
        self.assertTrue(NewsAgent().run(ctx()).is_no_data)
        self.assertTrue(NewsAgent().run(ctx(news=CalendarFile("/does/not/exist"))).is_no_data)

    def test_news_blackout_from_calendar(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "n.json")
            from datetime import datetime, timezone
            t = datetime.fromtimestamp(NOW + 600, timezone.utc).isoformat()
            json.dump([{"time": t, "impact": "HIGH", "title": "US CPI"}], open(p, "w"))
            r = NewsAgent().run(ctx(news=CalendarFile(p)))
            self.assertFalse(r.is_no_data)
            self.assertTrue(r.extra["block"])
            self.assertIn("CPI", r.reason)
            t2 = datetime.fromtimestamp(NOW + 36000, timezone.utc).isoformat()
            json.dump([{"time": t2, "impact": "HIGH", "title": "later"}], open(p, "w"))
            os.utime(p, (NOW, NOW + 5))
            self.assertFalse(NewsAgent().run(ctx(news=CalendarFile(p))).extra["block"])

    def test_liquidity_states_its_limits(self):
        r = default_agents()[5].run(ctx())
        self.assertIn("OHLC", r.reason)


if __name__ == "__main__":
    unittest.main()
