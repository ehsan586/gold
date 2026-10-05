"""Paper broker: simulates an MT5-like account on M1 candle data.
Used for tests, the backtester and the offline demo. Conservative assumptions:
 * history candles are BID prices; ask = bid + spread
 * if SL and TP are both inside one candle, SL is assumed to hit first
 * gaps beyond SL fill at the open (worse than SL)
"""
from __future__ import annotations

from .broker import Broker, BrokerError
from .data import (AccountInfo, Candle, OrderResult, Position, SeriesIndex, SymbolSpec, Tick,
                   TF_SECONDS)

DONE = 10009


def default_gold_spec(name: str = "XAUUSD") -> SymbolSpec:
    return SymbolSpec(name=name, digits=2, point=0.01, tick_size=0.01, tick_value=1.0,
                      contract_size=100.0, volume_min=0.01, volume_max=100.0, volume_step=0.01,
                      stops_level=10, freeze_level=0)


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, m1: list[Candle], symbol: str = "XAUUSD", balance: float = 800.0,
                 leverage: int = 100, spread_points: int = 25, slippage_points: int = 0,
                 commission_per_lot: float = 0.0, margin_mode: str = "HEDGING",
                 spec: SymbolSpec | None = None):
        self.m1 = m1
        self.symbol = symbol
        self.spec = spec or default_gold_spec(symbol)
        self.balance = balance
        self.leverage = leverage
        self.spread_points = spread_points
        self.slippage_points = slippage_points
        self.commission_per_lot = commission_per_lot
        self.margin_mode = margin_mode
        self.idx = -1
        self.index = SeriesIndex(m1)
        self._pos: dict[int, dict] = {}
        self._deals: dict[int, dict] = {}
        self._next = 1000
        self.connected = True
        self.fail_orders = False        # test hook

    # ---- time / price -----------------------------------------------------
    def advance(self) -> bool:
        if self.idx + 1 >= len(self.m1):
            return False
        self.idx += 1
        self._process_candle(self.m1[self.idx])
        return True

    def now(self) -> float:
        return float(self.m1[self.idx].time + 60) if self.idx >= 0 else 0.0

    def _bid(self) -> float:
        return self.m1[self.idx].close

    def _spread(self) -> float:
        return self.spread_points * self.spec.point

    def _pnl(self, direction: str, vol: float, open_p: float, close_p: float) -> float:
        diff = (close_p - open_p) if direction == "BUY" else (open_p - close_p)
        return diff / self.spec.tick_size * self.spec.tick_value * vol

    def _process_candle(self, c: Candle) -> None:
        sp = self._spread()
        slip = self.slippage_points * self.spec.point
        for t, p in list(self._pos.items()):
            fill = reason = None
            if p["dir"] == "BUY":
                if p["sl"] and c.open <= p["sl"]:
                    fill, reason = c.open - slip, "SL"
                elif p["sl"] and c.low <= p["sl"]:
                    fill, reason = p["sl"] - slip, "SL"
                elif p["tp"] and c.open >= p["tp"]:
                    fill, reason = c.open, "TP"
                elif p["tp"] and c.high >= p["tp"]:
                    fill, reason = p["tp"], "TP"
            else:
                o, h, l = c.open + sp, c.high + sp, c.low + sp
                if p["sl"] and o >= p["sl"]:
                    fill, reason = o + slip, "SL"
                elif p["sl"] and h >= p["sl"]:
                    fill, reason = p["sl"] + slip, "SL"
                elif p["tp"] and o <= p["tp"]:
                    fill, reason = o, "TP"
                elif p["tp"] and l <= p["tp"]:
                    fill, reason = p["tp"], "TP"
            if fill is not None:
                self._finish(t, fill, reason, p["vol"])

    def _finish(self, ticket: int, price: float, reason: str, volume: float) -> None:
        p = self._pos[ticket]
        pnl = self._pnl(p["dir"], volume, p["open"], price) - self.commission_per_lot * volume
        self.balance += pnl
        p["realized"] += pnl
        if volume >= p["vol"] - 1e-9:
            self._deals[ticket] = {"exit_price": price, "pnl": p["realized"], "time": self.now(),
                                   "reason": reason, "volume": p["init_vol"]}
            del self._pos[ticket]
        else:
            p["vol"] = round(p["vol"] - volume, 8)

    # ---- Broker API -------------------------------------------------------
    def connect(self) -> bool:
        self.connected = True
        return True

    def is_connected(self) -> bool:
        return self.connected

    def find_symbol(self, preferred: str) -> str:
        return self.symbol

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        return self.spec

    def tick(self, symbol: str) -> Tick:
        if self.idx < 0:
            raise BrokerError("no data yet")
        b = self._bid()
        return Tick(self.now(), b, b + self._spread())

    def candles(self, symbol: str, tf: str, n: int) -> list[Candle]:
        return self.index.upto(tf, self.now(), n)

    def _profit(self, p: dict) -> float:
        b = self._bid()
        close = b if p["dir"] == "BUY" else b + self._spread()
        return self._pnl(p["dir"], p["vol"], p["open"], close)

    def account(self) -> AccountInfo:
        unreal = sum(self._profit(p) for p in self._pos.values())
        margin = sum(p["vol"] * self.spec.contract_size * p["open"] / self.leverage for p in self._pos.values())
        eq = self.balance + unreal
        return AccountInfo(login=1, balance=self.balance, equity=eq, margin=margin,
                           free_margin=eq - margin, margin_level=(eq / margin * 100 if margin > 0 else 0.0),
                           leverage=self.leverage, margin_mode=self.margin_mode, is_demo=True)

    def positions(self, symbol: str | None = None) -> list[Position]:
        return [Position(ticket=t, symbol=self.symbol, direction=p["dir"], volume=p["vol"],
                         price_open=p["open"], sl=p["sl"], tp=p["tp"], profit=self._profit(p),
                         time_open=p["time"], magic=p["magic"], comment=p["comment"])
                for t, p in self._pos.items()]

    def send_market_order(self, symbol, direction, volume, sl, tp, comment, magic, max_slippage_points):
        if self.fail_orders or not self.connected:
            return OrderResult(False, 10006, message="rejected (test/disconnected)")
        s = self.spec
        if direction not in ("BUY", "SELL"):
            return OrderResult(False, 10030, message="bad direction")
        steps = volume / s.volume_step
        if volume < s.volume_min - 1e-9 or volume > s.volume_max + 1e-9 or abs(steps - round(steps)) > 1e-6:
            return OrderResult(False, 10014, message="invalid volume")
        bid = self._bid()
        price = bid + self._spread() if direction == "BUY" else bid
        min_dist = s.stops_level * s.point
        if not sl:
            return OrderResult(False, 10016, message="SL required")
        if (direction == "BUY" and (sl >= price - min_dist or (tp and tp <= price + min_dist))) or \
           (direction == "SELL" and (sl <= price + min_dist or (tp and tp >= price - min_dist))):
            return OrderResult(False, 10016, message="invalid stops")
        need = volume * s.contract_size * price / self.leverage
        if self.account().free_margin < need:
            return OrderResult(False, 10019, message="not enough money")
        if self.margin_mode in ("NETTING", "EXCHANGE") and self._pos:
            return OrderResult(False, 10030, message="netting paper broker: one position max")
        slip = self.slippage_points * s.point
        fill = price + slip if direction == "BUY" else price - slip
        self._next += 1
        self._pos[self._next] = dict(dir=direction, vol=volume, init_vol=volume, open=fill, sl=sl, tp=tp or 0.0,
                                     time=self.now(), magic=magic, comment=comment, realized=0.0)
        self.balance -= self.commission_per_lot * volume
        self._pos[self._next]["realized"] -= self.commission_per_lot * volume
        return OrderResult(True, DONE, self._next, fill, price, volume, slip / s.point, "done")

    def close_position(self, ticket, volume=None, comment=""):
        if self.fail_orders or not self.connected:
            return OrderResult(False, 10006, message="rejected (test/disconnected)")
        p = self._pos.get(ticket)
        if not p:
            return OrderResult(False, 10013, message="no such position")
        vol = p["vol"] if volume is None else volume
        if vol > p["vol"] + 1e-9:
            return OrderResult(False, 10014, message="volume too large")
        if vol < p["vol"] - 1e-9 and p["vol"] - vol < self.spec.volume_min - 1e-9:
            return OrderResult(False, 10014, message="remaining volume below minimum")
        bid = self._bid()
        price = bid if p["dir"] == "BUY" else bid + self._spread()
        self._finish(ticket, price, "MANUAL", vol)
        return OrderResult(True, DONE, ticket, price, price, vol, 0.0, "closed")

    def modify_position(self, ticket, sl=None, tp=None):
        p = self._pos.get(ticket)
        if not p:
            return OrderResult(False, 10013, message="no such position")
        if sl is not None:
            p["sl"] = sl
        if tp is not None:
            p["tp"] = tp
        return OrderResult(True, DONE, ticket, message="modified")

    def deal_summary(self, ticket):
        return self._deals.get(ticket)
