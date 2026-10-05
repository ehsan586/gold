"""MetaTrader 5 broker adapter (Windows + `pip install MetaTrader5`).

!! This module could NOT be executed in the build environment (no MT5/Windows).
!! It is written against the documented MetaTrader5 Python API. Test it on your DEMO
!! terminal first (see docs/GUIDE_FA.md (section "MT5 connection test")).
"""
from __future__ import annotations

import time

from .broker import Broker, BrokerError
from .data import AccountInfo, Candle, OrderResult, Position, SymbolSpec, Tick


class Mt5Broker(Broker):
    name = "mt5"

    def __init__(self, login: str | None = None, password: str | None = None,
                 server: str | None = None, path: str | None = None, timeout_ms: int = 10000):
        self._creds = dict(login=login, password=password, server=server, path=path)
        self._timeout = timeout_ms
        self._mt5 = None
        self._last_key = None
        self._last_change = 0.0

    # ---- connection -------------------------------------------------------
    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError as e:
            raise BrokerError("MetaTrader5 package missing. Windows only: pip install MetaTrader5") from e
        self._mt5 = mt5
        kw: dict = {"timeout": self._timeout}
        if self._creds["path"]:
            kw["path"] = self._creds["path"]
        if self._creds["login"]:
            kw.update(login=int(self._creds["login"]), password=self._creds["password"], server=self._creds["server"])
        if not mt5.initialize(**kw):
            raise BrokerError(f"mt5.initialize failed: {mt5.last_error()}")
        return True

    def disconnect(self) -> None:
        if self._mt5:
            self._mt5.shutdown()

    def is_connected(self) -> bool:
        if not self._mt5:
            return False
        ti = self._mt5.terminal_info()
        return bool(ti and ti.connected)

    def _need(self):
        if not self._mt5:
            raise BrokerError("not connected")
        return self._mt5

    # ---- account / symbol -------------------------------------------------
    def account(self) -> AccountInfo:
        m = self._need()
        a = m.account_info()
        if a is None:
            raise BrokerError(f"account_info failed: {m.last_error()}")
        mode = {m.ACCOUNT_MARGIN_MODE_RETAIL_NETTING: "NETTING",
                m.ACCOUNT_MARGIN_MODE_EXCHANGE: "EXCHANGE",
                m.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING: "HEDGING"}.get(a.margin_mode, "NETTING")
        return AccountInfo(login=a.login, balance=a.balance, equity=a.equity, margin=a.margin,
                           free_margin=a.margin_free, margin_level=a.margin_level or 0.0,
                           leverage=a.leverage, margin_mode=mode,
                           is_demo=(a.trade_mode == m.ACCOUNT_TRADE_MODE_DEMO), currency=a.currency)

    def find_symbol(self, preferred: str) -> str:
        m = self._need()
        if m.symbol_info(preferred) is not None:
            m.symbol_select(preferred, True)
            return preferred
        names = [s.name for s in (m.symbols_get() or [])]
        cands = [n for n in names if n.upper().startswith("XAUUSD") or n.upper().startswith("GOLD")]
        cands.sort(key=lambda n: (not n.upper().startswith("XAUUSD"), len(n), n))
        for n in cands:
            info = m.symbol_info(n)
            if info and info.trade_mode != m.SYMBOL_TRADE_MODE_DISABLED and 50 <= info.trade_contract_size <= 200:
                m.symbol_select(n, True)
                return n
        raise BrokerError(f"no gold symbol found (preferred={preferred}); set GOLDBOT_SYMBOL explicitly")

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        m = self._need()
        i = m.symbol_info(symbol)
        if i is None:
            raise BrokerError(f"symbol_info({symbol}) failed")
        return SymbolSpec(name=symbol, digits=i.digits, point=i.point, tick_size=i.trade_tick_size,
                          tick_value=i.trade_tick_value, contract_size=i.trade_contract_size,
                          volume_min=i.volume_min, volume_max=i.volume_max, volume_step=i.volume_step,
                          stops_level=i.trade_stops_level, freeze_level=i.trade_freeze_level,
                          trade_allowed=(i.trade_mode == m.SYMBOL_TRADE_MODE_FULL),
                          currency_profit=getattr(i, "currency_profit", ""))

    # ---- market data ------------------------------------------------------
    def tick(self, symbol: str) -> Tick:
        m = self._need()
        t = m.symbol_info_tick(symbol)
        if t is None or t.bid <= 0 or t.ask <= 0:
            raise BrokerError("no valid tick")
        key = (t.time_msc, t.bid, t.ask)
        if key != self._last_key:                 # broker time is server time: track freshness by wall clock
            self._last_key, self._last_change = key, time.time()
        return Tick(self._last_change, t.bid, t.ask)

    def candles(self, symbol: str, tf: str, n: int) -> list[Candle]:
        m = self._need()
        rates = m.copy_rates_from_pos(symbol, getattr(m, "TIMEFRAME_" + tf), 1, n)  # pos 1 = skip forming bar
        if rates is None:
            raise BrokerError(f"copy_rates failed: {m.last_error()}")
        return [Candle(int(r["time"]), float(r["open"]), float(r["high"]), float(r["low"]),
                       float(r["close"]), float(r["tick_volume"])) for r in rates]

    def positions(self, symbol: str | None = None) -> list[Position]:
        m = self._need()
        ps = m.positions_get(symbol=symbol) if symbol else m.positions_get()
        if ps is None:
            raise BrokerError(f"positions_get failed: {m.last_error()}")
        return [Position(ticket=p.ticket, symbol=p.symbol,
                         direction="BUY" if p.type == m.POSITION_TYPE_BUY else "SELL",
                         volume=p.volume, price_open=p.price_open, sl=p.sl, tp=p.tp, profit=p.profit,
                         time_open=float(p.time), magic=p.magic, comment=p.comment) for p in ps]

    # ---- orders (ExecutionEngine only) -------------------------------------
    def _filling(self, symbol: str) -> int:
        m = self._need()
        fm = m.symbol_info(symbol).filling_mode
        if fm & 2:
            return m.ORDER_FILLING_IOC
        if fm & 1:
            return m.ORDER_FILLING_FOK
        return m.ORDER_FILLING_RETURN

    def _send(self, req: dict, requested: float) -> OrderResult:
        m = self._need()
        chk = m.order_check(req)
        if chk is None or chk.retcode != 0:
            return OrderResult(False, getattr(chk, "retcode", -1), requested_price=requested,
                               message=f"order_check failed: {getattr(chk, 'comment', m.last_error())}")
        res = m.order_send(req)
        if res is None:
            return OrderResult(False, -1, requested_price=requested, message=f"order_send None: {m.last_error()}")
        ok = res.retcode == m.TRADE_RETCODE_DONE
        pt = m.symbol_info(req["symbol"]).point
        return OrderResult(ok, res.retcode, int(res.order), float(res.price), requested, float(res.volume),
                           abs(res.price - requested) / pt if ok and pt else 0.0, res.comment)

    def send_market_order(self, symbol, direction, volume, sl, tp, comment, magic, max_slippage_points):
        m = self._need()
        t = m.symbol_info_tick(symbol)
        if t is None:
            return OrderResult(False, -1, message="no tick")
        buy = direction == "BUY"
        price = t.ask if buy else t.bid
        req = {"action": m.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(volume),
               "type": m.ORDER_TYPE_BUY if buy else m.ORDER_TYPE_SELL, "price": price,
               "sl": float(sl), "tp": float(tp or 0.0), "deviation": int(max_slippage_points),
               "magic": int(magic), "comment": comment[:31], "type_time": m.ORDER_TIME_GTC,
               "type_filling": self._filling(symbol)}
        r = self._send(req, price)
        if r.ok:   # resolve the position ticket (hedging: == order ticket; netting: look it up)
            if not m.positions_get(ticket=r.ticket):
                ps = [p for p in (m.positions_get(symbol=symbol) or []) if p.magic == magic]
                if ps:
                    r.ticket = ps[-1].ticket
        return r

    def close_position(self, ticket, volume=None, comment=""):
        m = self._need()
        ps = m.positions_get(ticket=ticket)
        if not ps:
            return OrderResult(False, -1, message="position not found")
        p = ps[0]
        t = m.symbol_info_tick(p.symbol)
        buy = p.type == m.POSITION_TYPE_BUY
        price = t.bid if buy else t.ask
        req = {"action": m.TRADE_ACTION_DEAL, "symbol": p.symbol, "position": ticket,
               "volume": float(volume or p.volume), "type": m.ORDER_TYPE_SELL if buy else m.ORDER_TYPE_BUY,
               "price": price, "deviation": 50, "magic": p.magic, "comment": comment[:31] or "goldbot close",
               "type_time": m.ORDER_TIME_GTC, "type_filling": self._filling(p.symbol)}
        return self._send(req, price)

    def modify_position(self, ticket, sl=None, tp=None):
        m = self._need()
        ps = m.positions_get(ticket=ticket)
        if not ps:
            return OrderResult(False, -1, message="position not found")
        p = ps[0]
        req = {"action": m.TRADE_ACTION_SLTP, "symbol": p.symbol, "position": ticket,
               "sl": float(p.sl if sl is None else sl), "tp": float(p.tp if tp is None else tp)}
        return self._send(req, 0.0)

    def deal_summary(self, ticket):
        m = self._need()
        deals = m.history_deals_get(position=ticket)
        if not deals:
            return None
        outs = [d for d in deals if d.entry in (m.DEAL_ENTRY_OUT, m.DEAL_ENTRY_OUT_BY)]
        if not outs:
            return None
        pnl = sum(d.profit + d.commission + d.swap + getattr(d, "fee", 0.0) for d in deals)
        last = outs[-1]
        reason = {m.DEAL_REASON_SL: "SL", m.DEAL_REASON_TP: "TP"}.get(last.reason, "MANUAL")
        return {"exit_price": last.price, "pnl": pnl, "time": time.time(), "reason": reason,
                "volume": sum(d.volume for d in outs)}
