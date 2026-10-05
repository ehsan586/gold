"""Structured in-process metrics for observability (latency, freshness, availability, ...)."""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager


class Metrics:
    def __init__(self):
        self._l = threading.RLock()
        self.gauges: dict[str, float | str | None] = {}
        self.counters: dict[str, int] = {}
        self.hist: dict[str, deque] = {}

    def gauge(self, name: str, v) -> None:
        with self._l:
            self.gauges[name] = v

    def inc(self, name: str, n: int = 1) -> None:
        with self._l:
            self.counters[name] = self.counters.get(name, 0) + n

    def observe(self, name: str, v: float) -> None:
        with self._l:
            self.hist.setdefault(name, deque(maxlen=500)).append(v)

    @contextmanager
    def timer(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - t) * 1000.0)

    def last_ms(self, name: str):
        with self._l:
            h = self.hist.get(name)
            return round(h[-1], 2) if h else None

    def snapshot(self) -> dict:
        with self._l:
            out = {}
            for k, d in self.hist.items():
                xs = sorted(d)
                n = len(xs)
                out[k] = {"count": n, "mean_ms": round(sum(xs) / n, 2), "p50_ms": round(xs[n // 2], 2),
                          "p95_ms": round(xs[min(n - 1, int(n * 0.95))], 2), "max_ms": round(xs[-1], 2)}
            return {"gauges": dict(self.gauges), "counters": dict(self.counters), "latency": out}
