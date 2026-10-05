"""SIMULATION execution on LIVE market data: market data comes from the real data broker, orders only touch a
VIRTUAL account (no real order is ever sent). This is the default research mode."""
from __future__ import annotations

import json

from .broker import Broker, BrokerError
from .data import AccountInfo, OrderResult, Position
from .units import pnl_money

DONE = 10009


class ShadowBroker(Broker):
    name = "shadow"

    def __init__(self, data: Broker, db, start_balance: float = 800.0, clock=None, symbol: str = "XAUUSD"):
        self.data, self.db, self.clock, self.symbol = data, db, clock, symbol
        self.start_balance_major = float(start_balance)
        self.balance, self.pos, self.deals, self.next = float(start_balance), {}, {}, 5000
        self._money_scale = 1.0
        raw = db.kv_get("shadow_state")
        if raw:
            st = json.loads(raw)
            self.balance, self.next = st["balance"], st["next"]
            self.pos = {int(k): v for k, v in st["pos"].items()}
            self.deals = {int(k): v for k, v in st["deals"].items()}

    def _save(self) -> None:
        self.db.kv_set("shadow_state", json.dumps({"balance": self.balance, "next": self.next, "pos": self.pos,
                                                   "deals": dict(list(self.deals.items())[-200:])}))

    # ---- data side delegates to the real data source -------------------------------------------------
    def connect(self):
        ok = self.data.connect()
        cur = (self.data.account().currency or "").upper()
        self._money_scale = 100.0 if cur in {"USC", "EUC", "GBC", "AUC", "JPC", "CNC"} else 1.0
        if not self.db.kv_get("shadow_state"):
            self.balance = self.start_balance_major * self._money_scale
            self._save()
        return ok
    def disconnect(self): self.data.disconnect()
    def is_connected(self): return self.data.is_connected()
    def find_symbol(self, p):
        self.symbol = self.data.find_symbol(p)
        return self.symbol
    def symbol_spec(self, s): return self.data.symbol_spec(s)
    def tick(self, s): return self.data.tick(s)
    def candles(self, s, tf, n): return self.data.candles(s, tf, n)

    # ---- virtual account ------------------------------------------------------------------------------
    def _spec(self):
        return self.data.symbol_spec(self.symbol)

    def _unreal(self, p, t, spec) -> float:
        return pnl_money(p["dir"], p["open"], t.bid if p["dir"] == "BUY" else t.ask, p["vol"], spec)

    def account(self) -> AccountInfo:
        real = self.data.account()                        # leverage / currency / account type come from the real account
        spec, t = self._spec(), self.data.tick(self.symbol)
        unreal = sum(self._unreal(p, t, spec) for p in self.pos.values())
        margin = sum(p["vol"] * spec.contract_size * p["open"] / max(real.leverage, 1) for p in self.pos.values())
        eq = self.balance + unreal
        return AccountInfo(login=0, balance=self.balance, equity=eq, margin=margin, free_margin=eq - margin,
                           margin_level=(eq / margin * 100 if margin > 0 else 0.0), leverage=real.leverage,
                           margin_mode="HEDGING", is_demo=True, currency=real.currency)

    def positions(self, symbol=None):
        spec, t = self._spec(), self.data.tick(self.symbol)
        return [Position(ticket=k, symbol=self.symbol, direction=p["dir"], volume=p["vol"], price_open=p["open"], sl=p["sl"],
                         tp=p["tp"], profit=self._unreal(p, t, spec), time_open=p["time"], magic=p["magic"], comment="shadow")
                for k, p in self.pos.items()]

    def mark(self, tick, now: float) -> None:
        """Called every engine step with the live tick: stops/targets are evaluated against live bid/ask."""
        spec = self._spec()
        for k, p in list(self.pos.items()):
            px = tick.bid if p["dir"] == "BUY" else tick.ask
            hit = None
            if p["dir"] == "BUY":
                hit = ("SL", px) if (p["sl"] and px <= p["sl"]) else ("TP", p["tp"]) if (p["tp"] and px >= p["tp"]) else None
            else:
                hit = ("SL", px) if (p["sl"] and px >= p["sl"]) else ("TP", p["tp"]) if (p["tp"] and px <= p["tp"]) else None
            if hit:
                self._close(k, hit[1], hit[0], p["vol"], spec, now)

    def _close(self, k, price, reason, vol, spec, now) -> None:
        p = self.pos[k]
        pnl = pnl_money(p["dir"], p["open"], price, vol, spec)
        self.balance += pnl
        p["realized"] += pnl
        if vol >= p["vol"] - 1e-9:
            self.deals[k] = {"exit_price": price, "pnl": p["realized"], "time": now, "reason": reason, "volume": p["init_vol"]}
            del self.pos[k]
        else:
            p["vol"] = round(p["vol"] - vol, 8)
        self._save()

    def send_market_order(self, symbol, direction, volume, sl, tp, comment, magic, max_slippage_points):
        spec, t = self._spec(), self.data.tick(self.symbol)
        steps = volume / spec.volume_step
        if direction not in ("BUY", "SELL") or volume < spec.volume_min - 1e-9 or volume > spec.volume_max + 1e-9 or abs(steps - round(steps)) > 1e-6:
            return OrderResult(False, 10014, message="invalid volume/direction (virtual)")
        if not sl:
            return OrderResult(False, 10016, message="SL required")
        price = t.ask if direction == "BUY" else t.bid
        md = spec.stops_level * spec.point
        if (direction == "BUY" and (sl >= price - md or (tp and tp <= price + md))) or (direction == "SELL" and (sl <= price + md or (tp and tp >= price - md))):
            return OrderResult(False, 10016, message="invalid stops (virtual)")
        acc = self.account()
        if acc.free_margin < volume * spec.contract_size * price / max(acc.leverage, 1):
            return OrderResult(False, 10019, message="not enough virtual margin")
        self.next += 1
        self.pos[self.next] = dict(dir=direction, vol=volume, init_vol=volume, open=price, sl=sl, tp=tp or 0.0,
                                   time=self.clock() if self.clock else 0.0, magic=magic, realized=0.0)
        self._save()
        return OrderResult(True, DONE, self.next, price, price, volume, 0.0, "virtual fill (no slippage modelled)")

    def close_position(self, ticket, volume=None, comment=""):
        p = self.pos.get(ticket)
        if not p:
            return OrderResult(False, 10013, message="no such virtual position")
        spec, t = self._spec(), self.data.tick(self.symbol)
        vol = p["vol"] if volume is None else volume
        if vol > p["vol"] + 1e-9 or (vol < p["vol"] - 1e-9 and p["vol"] - vol < spec.volume_min - 1e-9):
            return OrderResult(False, 10014, message="invalid close volume")
        price = t.bid if p["dir"] == "BUY" else t.ask
        self._close(ticket, price, "MANUAL", vol, spec, self.clock() if self.clock else 0.0)
        return OrderResult(True, DONE, ticket, price, price, vol, 0.0, "virtual close")

    def modify_position(self, ticket, sl=None, tp=None):
        p = self.pos.get(ticket)
        if not p:
            return OrderResult(False, 10013, message="no such virtual position")
        if sl is not None: p["sl"] = sl
        if tp is not None: p["tp"] = tp
        self._save()
        return OrderResult(True, DONE, ticket, message="virtual modify")

    def deal_summary(self, ticket):
        return self.deals.get(ticket)
