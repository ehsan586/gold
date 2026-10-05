"""Unit normalisation. ACCOUNT MONEY, MARKET PRICE, POINTS, TICKS, TICK VALUE, CONTRACT SIZE and
VOLUME are different things and are never mixed:

  * prices are always in the instrument's quote units, exactly as the broker reports them;
  * a CENT account only changes the unit of ACCOUNT MONEY (balance/equity/tick_value are in cents);
    it NEVER divides or scales a market price;
  * P/L = ticks * tick_value(per lot, in ACCOUNT currency as the broker reports it) * volume.
Nothing here is broker specific: everything comes from the broker's symbol specification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .data import SymbolSpec

CENT_CURRENCIES = {"USC", "EUC", "GBC", "AUC", "JPC", "CNC"}
CENT_TO_MAJOR = {"USC": "USD", "EUC": "EUR", "GBC": "GBP", "AUC": "AUD", "JPC": "JPY", "CNC": "CNY"}
KNOWN_BASES = ("XAUUSD", "XAGUSD", "GOLD", "SILVER")


class Denomination(str, Enum):
    STANDARD = "STANDARD"
    CENT = "CENT"


@dataclass(frozen=True)
class AccountMoney:
    currency: str
    denomination: Denomination
    minor_per_major: int            # 100 for cent accounts, 1 otherwise

    @property
    def display_currency(self) -> str:
        """Major/display currency code; account money remains in broker units internally."""
        return CENT_TO_MAJOR.get(self.currency, self.currency or "USD")

    def to_major(self, amount: float) -> float:
        """For DISPLAY only (e.g. 80000 USC -> 800 USD). Never applied to prices."""
        return amount / self.minor_per_major


def detect_account_money(currency: str, setting: str = "AUTO") -> AccountMoney:
    cur = (currency or "").upper()
    cent = setting == "CENT" or (setting == "AUTO" and cur in CENT_CURRENCIES)
    return AccountMoney(cur, Denomination.CENT if cent else Denomination.STANDARD, 100 if cent else 1)


def split_symbol(name: str) -> tuple[str, str]:
    """'XAUUSDm' -> ('XAUUSD','m'); 'GOLD.a' -> ('GOLD','.a'). Suffix is whatever follows the known base."""
    up = name.upper()
    for b in KNOWN_BASES:
        if up.startswith(b):
            return name[:len(b)], name[len(b):]
    return name, ""


# ---- explicit conversions (each documents its units) ------------------------------------
def price_to_points(dprice: float, spec: SymbolSpec) -> float:
    return dprice / spec.point


def points_to_price(points: float, spec: SymbolSpec) -> float:
    return points * spec.point


def price_to_ticks(dprice: float, spec: SymbolSpec) -> float:
    return dprice / spec.tick_size


def pnl_money(direction: str, entry: float, exit_: float, volume: float, spec: SymbolSpec) -> float:
    """Result in ACCOUNT money units (tick_value is reported in the account currency)."""
    d = (exit_ - entry) if direction == "BUY" else (entry - exit_)
    return price_to_ticks(d, spec) * spec.tick_value * volume


def loss_per_lot(sl_distance_price: float, spec: SymbolSpec) -> float:
    return price_to_ticks(abs(sl_distance_price), spec) * spec.tick_value


def check_spec(spec: SymbolSpec, account_currency: str = "") -> list[str]:
    """Returns human-readable WARNINGS (empty = nothing suspicious). Never rescales anything."""
    w: list[str] = []
    for n in ("tick_size", "tick_value", "point", "contract_size", "volume_min", "volume_step", "volume_max"):
        v = getattr(spec, n)
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            w.append(f"{n} invalid ({v})")
    if w:
        return w
    if spec.volume_min > spec.volume_max:
        w.append("volume_min > volume_max")
    if abs(spec.point - 10 ** (-spec.digits)) > 1e-12:
        w.append(f"point {spec.point} != 10^-digits ({spec.digits}); points are used only as the broker defines them")
    if spec.tick_size < spec.point - 1e-12:
        w.append("tick_size smaller than point")
    cur, prof = (account_currency or "").upper(), (spec.currency_profit or "").upper()
    cent = cur in CENT_CURRENCIES
    major = {"USC": "USD", "EUC": "EUR", "GBC": "GBP", "AUC": "AUD", "JPC": "JPY", "CNC": "CNY"}.get(cur, cur)
    if prof and cur and prof in (cur, major):
        expected = spec.contract_size * spec.tick_size * (100 if cent else 1)      # account money units per tick per lot
        if abs(spec.tick_value - expected) / expected > 0.02:
            w.append(f"tick_value {spec.tick_value} != expected {expected:g} (contract_size*tick_size{'*100 for a cent account' if cent else ''}) although profit "
                     f"and account currency match ({cur}); the broker's tick_value is used for P/L")
    return w
