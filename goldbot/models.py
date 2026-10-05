"""Standard data structures shared by every component.

Every agent MUST return an AgentResult. Validation is strict so that a broken
agent can never feed garbage into the Decision Engine: invalid output raises
AgentOutputError and the caller treats it as NO_DATA (=> no new trade).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum


class AgentOutputError(ValueError):
    pass


class Direction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class MarketRegime(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNKNOWN = "UNKNOWN"


class VolatilityClass(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


AGENT_NAMES = ("trend", "structure", "price_action", "momentum",
               "volatility", "liquidity", "news")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _check_score(name: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AgentOutputError(f"{name} must be a number, got {type(v).__name__}")
    if math.isnan(v) or math.isinf(v):
        raise AgentOutputError(f"{name} is NaN/inf")
    if not (0 <= v <= 100):
        raise AgentOutputError(f"{name} must be within 0-100, got {v}")
    return float(v)


@dataclass
class AgentResult:
    agent: str
    symbol: str
    timeframe: str
    direction: Direction
    confidence: float
    strength: float
    market_regime: MarketRegime
    data_quality: float
    reason: str
    timestamp: datetime
    extra: dict = field(default_factory=dict)   # agent-specific details (levels, ATR, block flags)

    def __post_init__(self):
        if self.agent not in AGENT_NAMES:
            raise AgentOutputError(f"unknown agent {self.agent!r}")
        try:
            self.direction = Direction(self.direction)
            self.market_regime = MarketRegime(self.market_regime)
        except ValueError as e:
            raise AgentOutputError(str(e)) from e
        self.confidence = _check_score("confidence", self.confidence)
        self.strength = _check_score("strength", self.strength)
        self.data_quality = _check_score("data_quality", self.data_quality)
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise AgentOutputError("reason must be a non-empty string")
        if not isinstance(self.timestamp, datetime):
            raise AgentOutputError("timestamp must be a datetime")
        if not isinstance(self.extra, dict):
            raise AgentOutputError("extra must be a dict")
        try:
            json.dumps(self.extra)
        except (TypeError, ValueError) as e:
            raise AgentOutputError(f"extra not serializable: {e}") from e
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    # ---- factories --------------------------------------------------------
    @classmethod
    def no_data(cls, agent: str, symbol: str, timeframe: str, reason: str,
                now: datetime | None = None, extra: dict | None = None) -> "AgentResult":
        """The only legal answer when an agent cannot genuinely analyse
        (missing data, no news feed, ...). Never invent a signal."""
        return cls(agent=agent, symbol=symbol, timeframe=timeframe,
                   direction=Direction.HOLD, confidence=0, strength=0,
                   market_regime=MarketRegime.UNKNOWN, data_quality=0,
                   reason=f"NO_DATA: {reason}", timestamp=now or _utcnow(), extra=extra or {})

    @property
    def is_no_data(self) -> bool:
        return self.data_quality == 0 and self.market_regime is MarketRegime.UNKNOWN

    # ---- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["direction"] = self.direction.value
        d["market_regime"] = self.market_regime.value
        d["timestamp"] = self.timestamp.isoformat()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentResult":
        try:
            d = dict(d)
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
            return cls(**d)
        except (KeyError, TypeError, ValueError) as e:
            if isinstance(e, AgentOutputError):
                raise
            raise AgentOutputError(f"cannot parse agent result: {e}") from e

    @classmethod
    def from_json(cls, s: str) -> "AgentResult":
        try:
            return cls.from_dict(json.loads(s))
        except json.JSONDecodeError as e:
            raise AgentOutputError(f"invalid JSON: {e}") from e
