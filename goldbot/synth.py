"""Synthetic M1 data for offline demos/tests ONLY. It is a random walk with regime
switches. Results on this data say NOTHING about real gold performance."""
from __future__ import annotations

import random

from .data import Candle


def generate_m1(n: int = 20000, start_price: float = 2650.0, seed: int = 7,
                start_time: int = 1767571200) -> list[Candle]:   # 2026-01-05 00:00 UTC
    rnd = random.Random(seed)
    out: list[Candle] = []
    price, t = start_price, start_time
    left, mu, sigma = 0, 0.0, 0.35
    for _ in range(n):
        if left <= 0:
            left = rnd.randint(150, 600)
            mu = rnd.choice([-0.05, 0.0, 0.0, 0.05, 0.08, -0.08])
            sigma = rnd.choice([0.25, 0.35, 0.5, 0.9])
        left -= 1
        o = price
        c = o + rnd.gauss(mu, sigma)
        h = max(o, c) + abs(rnd.gauss(0, sigma * 0.5))
        l = min(o, c) - abs(rnd.gauss(0, sigma * 0.5))
        out.append(Candle(t, round(o, 2), round(h, 2), round(l, 2), round(c, 2), 100.0))
        price, t = c, t + 60
    return out
