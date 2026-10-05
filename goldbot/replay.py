"""Replay Engine: re-runs history through the REAL engine as if live and reconstructs what the system knew at each
point in time. Because the broker only ever exposes candles whose close time <= 'now', and every snapshot is
checked by the DataValidator (FUTURE_DATA is fatal), the replay cannot see the future. Determinism is verified
by comparing the hash-chain digest of two replays."""
from __future__ import annotations

from dataclasses import dataclass, field

from .agents import CalendarFile
from .broker_paper import PaperBroker
from .data import TF_SECONDS, Candle
from .db import Database
from .engine import TradingEngine
from .settings import Settings
from .snapshot import DataQuality


class ReplayBroker(PaperBroker):
    """PaperBroker with a tripwire: any candle returned whose close lies after 'now' is a look-ahead violation."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.violations = 0

    def candles(self, symbol, tf, n):
        out = super().candles(symbol, tf, n)
        if out and out[-1].time + TF_SECONDS[tf] > self.now() + 1e-6:
            self.violations += 1
        return out


@dataclass
class ReplayResult:
    digest: str
    events: int
    steps: int
    decisions: dict = field(default_factory=dict)
    trades: int = 0
    final_equity: float = 0.0
    leak_violations: int = 0
    rejected_snapshots: int = 0
    chain_ok: bool = True

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class ReplayEngine:
    def __init__(self, settings: Settings):
        self.s = settings

    def _build(self, m1: list[Candle], warmup: int, balance: float = 800.0):
        db = Database(":memory:")
        db.init_schema()
        b = ReplayBroker(m1, self.s.symbol, balance)
        eng = TradingEngine(self.s, db, b, clock=b.now, news=CalendarFile("/nonexistent"), sim=True)
        eng.periodic_eval = False
        eng.publish_enabled = False
        for _ in range(warmup):
            b.advance()
        eng.start()
        return eng, b, db

    def replay(self, m1: list[Candle], warmup: int = 3700, max_steps: int | None = None) -> ReplayResult:
        eng, b, db = self._build(m1, warmup)
        steps, dec = 0, {"BUY": 0, "SELL": 0, "HOLD": 0}
        while b.advance() and (max_steps is None or steps < max_steps):
            eng.step(b.now())
            dec[eng._decision.action] = dec.get(eng._decision.action, 0) + 1
            steps += 1
        ok, n = eng.events.verify_chain()
        return ReplayResult(eng.events.digest(), n, steps, dec, db.query("SELECT COUNT(*) c FROM trades WHERE status!='REJECTED'")[0]["c"],
                            b.account().equity, b.violations, db.query("SELECT COUNT(*) c FROM events WHERE event_type='SNAPSHOT_REJECTED'")[0]["c"], ok)

    def verify_determinism(self, m1: list[Candle], warmup: int = 3700, max_steps: int = 120) -> dict:
        a, b = self.replay(m1, warmup, max_steps), self.replay(m1, warmup, max_steps)
        return {"deterministic": a.digest == b.digest, "digest": a.digest[:16], "events": a.events, "leak_violations": a.leak_violations + b.leak_violations}

    def reconstruct(self, m1: list[Candle], index: int, warmup: int = 3700) -> dict:
        """What did the system know, and what did it conclude, right after M1 bar `index` closed?"""
        eng, b, db = self._build(m1[:index + 1], warmup=min(warmup, index))
        while b.advance():
            pass
        snap = eng.dataservice.build(eng.symbol, eng.spec, b.now(), as_of=b.now())
        q = eng.data_validator.validate(snap)
        return {"as_of": snap.as_of, "snapshot": snap.metadata(), "data_quality": q.to_dict(), **eng.run_stack(snap, q, b.now(), record=False)}


def selftest(settings: Settings) -> dict:
    """Quick deterministic self-test used by the watchdog: tiny synthetic replay twice, compare digests."""
    from .synth import generate_m1
    try:
        r = ReplayEngine(settings).verify_determinism(generate_m1(3900, seed=1), warmup=3700, max_steps=40)
        return {"ok": r["deterministic"] and r["leak_violations"] == 0, **r}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
