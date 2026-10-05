from datetime import datetime, timezone
from goldbot.agents import CalendarFile
from goldbot.broker_paper import PaperBroker, default_gold_spec
from goldbot.data import AccountInfo, Candle, Position, Tick
from goldbot.db import Database
from goldbot.engine import TradingEngine
from goldbot.models import AgentResult
from goldbot.settings import load_settings
from goldbot.synth import generate_m1

S = load_settings({"GOLDBOT_SIGNAL_PERSISTENCE_TICKS": "3"})
SPEC = default_gold_spec()


def new_db():
    db = Database(":memory:")
    db.init_schema()
    return db


def make_engine(n=6000, seed=11, settings=S, warm=3700):
    db = new_db()
    b = PaperBroker(generate_m1(n, seed=seed))
    for _ in range(warm):
        b.advance()
    e = TradingEngine(settings, db, b, clock=b.now, news=CalendarFile("/nonexistent"), sim=True)
    e.start()
    return e, b, db


def res(agent, direction, conf=70, strength=70, regime="UNKNOWN", dq=90, **extra):
    return AgentResult(agent, "XAUUSD", "M15", direction, conf, strength, regime, dq, "test",
                       datetime.now(timezone.utc), extra)


def acct(equity=800.0, free=800.0, level=0.0, demo=True, lev=100):
    return AccountInfo(1, equity, equity, 0.0, free, level, lev, "HEDGING", demo)


def flat_candles(n=120, price=2650.0, rng=1.0, t0=1767571200, step=900):
    return [Candle(t0 + i * step, price, price + rng / 2, price - rng / 2, price, 1.0) for i in range(n)]


# ---------------- ASTRA helpers ----------------
from goldbot.data import TF_SECONDS, Tick as _Tick, resample as _resample
from goldbot.snapshot import MarketSnapshot, make_snapshot_id, hash_candles


def make_snap(candles: dict, as_of=None, tick=None, available_at=None, spec=SPEC):
    """Build a MarketSnapshot by hand (candles: tf -> list[Candle], CLOSED candles only)."""
    closes = [c[-1].time + TF_SECONDS[tf] for tf, c in candles.items() if c]
    a = as_of if as_of is not None else (max(closes) if closes else 0.0)
    av = available_at if available_at is not None else a
    t = tick or _Tick(av, 2650.0, 2650.25)
    return MarketSnapshot(make_snapshot_id("XAUUSD", a, "f1"), "XAUUSD", "test", a, av, "f1", t, spec,
                          {k: tuple(v) for k, v in candles.items()}, hash_candles(candles))


def trending(n=200, step=0.6, t0=1767571200, tf=900, rng=1.0, start=2600.0):
    out, p = [], start
    for i in range(n):
        o, c = p, p + step
        out.append(Candle(t0 + i * tf, o, max(o, c) + rng / 4, min(o, c) - rng / 4, c, 10.0))
        p = c
    return out


def ranging(n=200, amp=1.0, t0=1767571200, tf=900, base=2650.0):
    import math
    out = []
    for i in range(n):
        o = base + amp * math.sin(i * 1.3)
        c = base + amp * math.sin((i + 1) * 1.3)
        out.append(Candle(t0 + i * tf, o, max(o, c) + 0.3, min(o, c) - 0.3, c, 10.0))
    return out


def m5_series(n=600, seed=3):
    return _resample(generate_m1(n * 5 + 10, seed=seed), 300)[:n]
