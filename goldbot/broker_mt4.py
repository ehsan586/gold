"""Read-only MetaTrader 4 market-data bridge.

The MT4 EA writes tick/account/spec metadata and closed candles into
MetaTrader's FILE_COMMON directory. Python reads them and never sends orders.
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

from .broker import Broker, BrokerError
from .data import AccountInfo, Candle, OrderResult, Position, SymbolSpec, Tick


class Mt4FeedBroker(Broker):
    name = "mt4_feed"
    TFS = ("M1", "M5", "M15", "M30", "H1", "H4")

    def __init__(self, feed_dir: str | None = None, symbol: str = "XAUUSD", stale_sec: float = 15.0):
        self.symbol = symbol
        self.feed_dir = Path(feed_dir or self._default_common_dir())
        self.stale_sec = stale_sec
        self.connected = False
        self._meta: dict = {}
        self._cache: dict[str, tuple[float, list[Candle]]] = {}

    @staticmethod
    def _default_common_dir() -> str:
        appdata = os.environ.get("APPDATA", "")
        return str(Path(appdata) / "MetaQuotes" / "Terminal" / "Common" / "Files") if appdata else "."

    def _read_meta(self) -> None:
        p = self.feed_dir / "ASTRA_MT4_FEED.json"
        if not p.exists():
            raise BrokerError(f"MT4 feed not found: {p}. Attach ASTRAFeedEA.mq4 to the MT4 chart once.")
        last_err = None
        for delay in (0.0, 0.05, 0.10, 0.20):
            if delay:
                time.sleep(delay)
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(meta, dict):
                    raise ValueError("feed JSON root must be an object")
                break
            except Exception as e:
                last_err = e
        else:
            raise BrokerError(f"cannot read MT4 feed JSON: {last_err}") from last_err
        required = ("updated_at", "symbol", "tick", "account", "symbol_spec")
        missing = [k for k in required if k not in meta]
        if missing:
            raise BrokerError(f"MT4 feed JSON missing fields: {', '.join(missing)}")
        tick = meta.get("tick") or {}
        if float(tick.get("bid", 0) or 0) <= 0 or float(tick.get("ask", 0) or 0) <= 0:
            raise BrokerError("MT4 feed contains an invalid bid/ask")
        ts = float(meta.get("updated_at", 0))
        age = time.time() - ts
        if ts <= 0 or age > self.stale_sec:
            raise BrokerError(f"MT4 feed is stale ({max(0.0, age):.1f}s)")
        self._meta = meta

    def _read_tf(self, tf: str, n: int) -> list[Candle]:
        p = self.feed_dir / f"ASTRA_MT4_{tf}.csv"
        if not p.exists():
            raise BrokerError(f"MT4 {tf} history not found: {p}")
        mt = p.stat().st_mtime
        cached = self._cache.get(tf)
        if cached and cached[0] == mt:
            return cached[1][-n:]
        rows: list[Candle] = []
        last_err = None
        for delay in (0.0, 0.05, 0.10, 0.20):
            if delay:
                time.sleep(delay)
            rows = []
            try:
                with p.open(newline="", encoding="utf-8") as f:
                    rd = csv.DictReader(f)
                    required = {"time", "open", "high", "low", "close"}
                    if not required.issubset(set(rd.fieldnames or [])):
                        raise ValueError("history header is incomplete")
                    for r in rd:
                        rows.append(Candle(int(float(r["time"])), float(r["open"]), float(r["high"]),
                                           float(r["low"]), float(r["close"]), float(r.get("volume") or 0)))
                if rows:
                    break
            except Exception as e:
                last_err = e
        else:
            raise BrokerError(f"cannot read MT4 {tf} history: {last_err}") from last_err
        rows.sort(key=lambda x: x.time)
        self._cache[tf] = (mt, rows)
        return rows[-n:]

    def connect(self) -> bool:
        self.feed_dir.mkdir(parents=True, exist_ok=True)
        self._read_meta()
        for tf in self.TFS:
            self._read_tf(tf, 1)
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        if not self.connected:
            return False
        try:
            self._read_meta()
            return True
        except BrokerError:
            return False

    def _need(self):
        if not self.connected:
            raise BrokerError("MT4 feed not connected")
        self._read_meta()

    def account(self) -> AccountInfo:
        self._need(); a = self._meta.get("account", {})
        return AccountInfo(login=int(a.get("login", 0)), balance=float(a.get("balance", 0)), equity=float(a.get("equity", 0)),
                           margin=float(a.get("margin", 0)), free_margin=float(a.get("free_margin", 0)),
                           margin_level=float(a.get("margin_level", 0)), leverage=int(a.get("leverage", 0) or 0),
                           margin_mode=str(a.get("margin_mode", "HEDGING")), is_demo=bool(a.get("is_demo", True)),
                           currency=str(a.get("currency", "USD")))

    def find_symbol(self, preferred: str) -> str:
        self._need(); self.symbol = str(self._meta.get("symbol") or preferred); return self.symbol

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        self._need(); x = self._meta.get("symbol_spec", {})
        if not x: raise BrokerError("MT4 feed has no symbol specification")
        return SymbolSpec(name=symbol, digits=int(x["digits"]), point=float(x["point"]), tick_size=float(x["tick_size"]),
                          tick_value=float(x["tick_value"]), contract_size=float(x["contract_size"]),
                          volume_min=float(x["volume_min"]), volume_max=float(x["volume_max"]), volume_step=float(x["volume_step"]),
                          stops_level=int(x.get("stops_level", 0)), freeze_level=int(x.get("freeze_level", 0)),
                          trade_allowed=False, currency_profit=str(x.get("currency_profit", "")))

    def tick(self, symbol: str) -> Tick:
        self._need(); t = self._meta.get("tick", {})
        return Tick(float(t.get("time", self._meta.get("updated_at", 0))), float(t["bid"]), float(t["ask"]))

    def candles(self, symbol: str, tf: str, n: int) -> list[Candle]:
        self._need()
        if tf not in self.TFS:
            raise BrokerError(f"unsupported MT4 timeframe {tf}")
        return self._read_tf(tf, n)

    def positions(self, symbol: str | None = None) -> list[Position]:
        return []

    def _no_orders(self, *args, **kwargs) -> OrderResult:
        return OrderResult(False, -1, message="MT4 feed adapter is read-only; use SIMULATION/ShadowBroker")

    send_market_order = _no_orders
    close_position = _no_orders
    modify_position = _no_orders
    def deal_summary(self, ticket): return None
