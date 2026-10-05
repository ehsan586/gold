"""Central, validated configuration. Every value can be overridden by an
environment variable named GOLDBOT_<FIELD_NAME_UPPERCASE>.

Secrets (API token, broker credentials, LLM keys) are NEVER given defaults here
and are never sent to the frontend.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, asdict
from typing import Mapping, Tuple


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    # ---- instrument / mode -------------------------------------------------
    symbol: str = "XAUUSD"                 # broker may use XAUUSDm, GOLD ... (auto-discovery in Phase 3)
    allow_real_account: bool = False       # MUST stay False in v1: demo only
    db_path: str = "data/goldbot.db"
    strategy_version: str = "v1.0"

    # ---- timeframes (adjustable; must be tested, not assumed) ---------------
    htf_timeframes: Tuple[str, ...] = ("H4", "H1")        # higher-timeframe context
    ltf_timeframes: Tuple[str, ...] = ("M15", "M5", "M1")  # lower-timeframe entry context

    # ---- monitoring --------------------------------------------------------
    check_interval_sec: float = 1.0        # CHECK != TRADE: we look every second, trade rarely

    # ---- risk (percent values are in PERCENT of equity) --------------------
    max_risk_per_trade: float = 1.0        # lose at most 1% of equity if SL hit (~$8 on $800). Needs >=1% because min lot 0.01 on gold risks ~$3-8
    max_total_exposure: float = 2.0        # sum of open risk across positions <= 2% equity
    max_daily_loss: float = 3.0            # stop opening trades after -3% on the day
    max_account_drawdown: float = 10.0     # peak-to-now drawdown that triggers EMERGENCY_STOP (demo default)
    max_consecutive_losses: int = 4        # streak of losers => pause, a human should look
    max_trades_per_day: int = 8            # hard cap against overtrading
    max_reversals_per_day: int = 2         # reversals are the costliest behaviour: keep rare
    min_time_between_trades_sec: int = 300 # 5 min spacing between new entries
    cooldown_after_loss_sec: int = 900     # 15 min rest after a losing trade
    cooldown_after_reversal_sec: int = 900 # 15 min rest after any reversal
    max_spread_points: int = 60            # block entries if spread is abnormal (broker-dependent; tune)
    min_margin_level_pct: float = 300.0    # block if margin level below this

    # ---- profit protection --------------------------------------------------
    profit_protection_enabled: bool = True
    profit_activation_r: float = 0.5       # arm protection after +0.5R unrealized
    profit_retrace_percent: float = 40.0   # close if profit gives back 40% of peak (to be tested: 20/30/40/50)

    # ---- reversal / anti-noise ---------------------------------------------
    reversal_threshold: float = 70.0       # opposite overall score must reach this
    min_signal_change: float = 25.0        # opposite score must beat current-side score by this
    min_signal_strength: float = 60.0      # minimum strength to act on any signal
    cooldown_after_exit_sec: int = 120
    signal_persistence_ticks: int = 5      # signal must hold N consecutive checks

    # ---- ASTRA: execution / research mode -----------------------------------
    execution_mode: str = "SIMULATION"     # DISABLED | SIMULATION (final build is non-executing)
    market_data_source: str = "MT4"         # MT4 (read-only feed) | MT5 (optional legacy connector)
    mt4_feed_path: str = ""                  # MT4 FILE_COMMON's Files directory; blank = auto-detect on Windows
    feature_version: str = "f1"
    account_denomination: str = "AUTO"     # AUTO | STANDARD | CENT  (money display only; NEVER scales prices)
    regime_timeframe: str = "M15"
    forecast_timeframe: str = "M5"
    forecast_horizon_bars: int = 12        # forecast target = move over this many forecast-timeframe bars
    forecast_flat_threshold_atr: float = 0.25
    forecast_min_samples: int = 100        # below this: no calibration, no 'beats baseline' claims
    require_forecast: bool = False
    max_eval_reuse: int = 3                # how often one evaluation set may be used for decisions
    module_fail_limit: int = 3
    module_cooldown_sec: int = 300
    drift_window: int = 200
    learning_eval_interval_sec: int = 900
    sim_start_balance: float = 800.0      # starting balance of the VIRTUAL account in SIMULATION mode

    # ---- trade management ---------------------------------------------------
    sl_timeframe: str = "M5"               # ATR timeframe for stop distance (small accounts need tight stops)
    sl_atr_mult: float = 1.5
    min_sl_atr_mult: float = 1.0
    max_sl_atr_mult: float = 3.0
    tp_rr: float = 1.5                     # fixed reward:risk (NOT assumed optimal; must be backtested)
    breakeven_enabled: bool = True
    breakeven_r: float = 1.0
    partial_close_enabled: bool = False
    partial_close_r: float = 1.0
    partial_close_fraction: float = 0.5
    trailing_enabled: bool = False
    trailing_start_r: float = 1.5
    trailing_atr_mult: float = 1.5
    max_lot_cap: float = 0.5               # absolute ceiling regardless of sizing maths
    min_hold_before_reversal_sec: int = 120
    analysis_interval_sec: float = 5.0     # agents are recomputed at most this often
    max_tick_age_sec: float = 15.0         # older price data = market-data failure = no trading
    magic_number: int = 20261002
    max_slippage_points: int = 30
    require_news_data: bool = False        # True => NO_DATA from News agent blocks all entries
    news_blackout_minutes: int = 30
    news_file: str = "data/news_events.json"
    learning_min_samples: int = 50

    # ---- decision thresholds ------------------------------------------------
    min_decision_score: float = 65.0
    min_agreement_score: float = 55.0
    min_data_quality: float = 50.0

    # ---- API / security (token has NO default: must come from env) ----------
    api_token: str = field(default="", repr=False)      # admin (control) token
    viewer_token: str = field(default="", repr=False)   # optional read-only token
    api_host: str = "127.0.0.1"
    api_port: int = 8787
    cors_origins: Tuple[str, ...] = ("http://localhost:3000",)

    def validate(self) -> "Settings":
        errs = []
        if self.allow_real_account:
            errs.append("allow_real_account must be False in v1 (demo only)")
        if not self.symbol.strip():
            errs.append("symbol is empty")
        for name in ("max_risk_per_trade", "max_total_exposure", "max_daily_loss",
                     "max_account_drawdown", "profit_retrace_percent"):
            v = getattr(self, name)
            if not (0 < v <= 100):
                errs.append(f"{name} must be in (0, 100], got {v}")
        if self.max_risk_per_trade > self.max_total_exposure:
            errs.append("max_risk_per_trade cannot exceed max_total_exposure")
        if self.max_risk_per_trade > 2.0:
            errs.append("max_risk_per_trade > 2% is refused as unsafe")
        if self.max_total_exposure > self.max_account_drawdown:
            errs.append("max_total_exposure should not exceed max_account_drawdown")
        for name in ("max_consecutive_losses", "max_trades_per_day", "max_reversals_per_day",
                     "signal_persistence_ticks"):
            if getattr(self, name) < 0:
                errs.append(f"{name} must be >= 0")
        for name in ("min_time_between_trades_sec", "cooldown_after_loss_sec",
                     "cooldown_after_reversal_sec", "cooldown_after_exit_sec"):
            if getattr(self, name) < 0:
                errs.append(f"{name} must be >= 0")
        for name in ("reversal_threshold", "min_signal_change", "min_signal_strength",
                     "min_decision_score", "min_agreement_score", "min_data_quality"):
            if not (0 <= getattr(self, name) <= 100):
                errs.append(f"{name} must be in [0, 100]")
        if self.check_interval_sec < 0.2:
            errs.append("check_interval_sec must be >= 0.2")
        valid_tf = {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}
        for tf in self.htf_timeframes + self.ltf_timeframes:
            if tf not in valid_tf:
                errs.append(f"unknown timeframe {tf}")
        if self.tp_rr <= 0 or self.sl_atr_mult <= 0:
            errs.append("tp_rr and sl_atr_mult must be > 0")
        if not (self.min_sl_atr_mult <= self.sl_atr_mult <= self.max_sl_atr_mult):
            errs.append("need min_sl_atr_mult <= sl_atr_mult <= max_sl_atr_mult")
        if not (0 < self.partial_close_fraction < 1):
            errs.append("partial_close_fraction must be in (0,1)")
        if self.sl_timeframe not in {"M1", "M5", "M15", "M30", "H1", "H4"}:
            errs.append("bad sl_timeframe")
        if not (1 <= self.api_port <= 65535):
            errs.append("api_port out of range")
        if self.execution_mode not in ("DISABLED", "SIMULATION"):
            errs.append("execution_mode must be DISABLED or SIMULATION in the final build")
        if self.market_data_source not in ("MT4", "MT5"):
            errs.append("market_data_source must be MT4 or MT5")
        if self.account_denomination not in ("AUTO", "STANDARD", "CENT"):
            errs.append("account_denomination must be AUTO, STANDARD or CENT")
        for name in ("regime_timeframe", "forecast_timeframe"):
            if getattr(self, name) not in {"M1", "M5", "M15", "M30", "H1", "H4"}:
                errs.append(f"bad {name}")
        if self.forecast_horizon_bars < 1 or self.forecast_min_samples < 10:
            errs.append("forecast_horizon_bars >= 1 and forecast_min_samples >= 10 required")
        if self.max_lot_cap <= 0:
            errs.append("max_lot_cap must be > 0")
        if errs:
            raise ConfigError("; ".join(errs))
        return self

    def public_dict(self) -> dict:
        """Safe to expose to the dashboard: secrets removed."""
        d = asdict(self)
        d.pop("api_token", None)
        d.pop("viewer_token", None)
        return d


def _parse(raw: str, default):
    if isinstance(default, bool):
        low = raw.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ConfigError(f"bad boolean: {raw!r}")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, tuple):
        return tuple(p.strip() for p in raw.split(",") if p.strip())
    return raw


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    defaults = Settings()
    overrides = {}
    for f in fields(Settings):
        key = "GOLDBOT_" + f.name.upper()
        if key in env and env[key] != "":
            try:
                overrides[f.name] = _parse(env[key], getattr(defaults, f.name))
            except ValueError as e:
                raise ConfigError(f"{key}: {e}") from e
    return Settings(**{**asdict(defaults), **overrides}).validate()
