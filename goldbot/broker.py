"""Broker interface. The ExecutionEngine is the only component that may call
the order-sending methods (send_market_order / close_position / modify_position)."""
from __future__ import annotations

from .data import AccountInfo, Candle, OrderResult, Position, SymbolSpec, Tick


class BrokerError(RuntimeError):
    pass


class Broker:
    name = "base"

    def connect(self) -> bool: raise NotImplementedError
    def disconnect(self) -> None: pass
    def is_connected(self) -> bool: raise NotImplementedError
    def account(self) -> AccountInfo: raise NotImplementedError
    def find_symbol(self, preferred: str) -> str: raise NotImplementedError
    def symbol_spec(self, symbol: str) -> SymbolSpec: raise NotImplementedError
    def tick(self, symbol: str) -> Tick: raise NotImplementedError
    def candles(self, symbol: str, tf: str, n: int) -> list[Candle]: raise NotImplementedError
    def positions(self, symbol: str | None = None) -> list[Position]: raise NotImplementedError

    # --- order methods: ExecutionEngine only ---
    def send_market_order(self, symbol: str, direction: str, volume: float, sl: float, tp: float,
                          comment: str, magic: int, max_slippage_points: int) -> OrderResult: raise NotImplementedError
    def close_position(self, ticket: int, volume: float | None = None, comment: str = "") -> OrderResult: raise NotImplementedError
    def modify_position(self, ticket: int, sl: float | None = None, tp: float | None = None) -> OrderResult: raise NotImplementedError
    def deal_summary(self, ticket: int) -> dict | None: raise NotImplementedError
