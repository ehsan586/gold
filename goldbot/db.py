"""SQLite data layer with a versioned schema.

All SQL lives in this module so PostgreSQL can replace it later by providing
another class with the same public methods. Schema changes = append a new
entry to MIGRATIONS (never edit an old one).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MIGRATIONS: list[tuple[int, str]] = [
    (1, """
    CREATE TABLE trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        idempotency_key TEXT UNIQUE,
        ticket INTEGER,
        status TEXT NOT NULL DEFAULT 'OPEN',          -- OPEN / CLOSED
        ts_open TEXT NOT NULL,
        ts_close TEXT,
        symbol TEXT NOT NULL,
        timeframe TEXT,
        market_regime TEXT,
        direction TEXT NOT NULL,
        entry_price REAL, exit_price REAL,
        requested_price REAL, slippage REAL, retcode INTEGER,
        lot REAL, sl REAL, tp REAL,
        spread REAL, atr REAL, volatility TEXT,
        decision_id INTEGER,
        decision_score REAL, agreement_score REAL, risk_score REAL,
        agent_outputs_json TEXT,
        duration_sec REAL,
        pnl REAL, r_multiple REAL, mfe REAL, mae REAL,
        exit_reason TEXT,
        strategy_version TEXT NOT NULL
    );
    CREATE INDEX idx_trades_status ON trades(status);
    CREATE INDEX idx_trades_open ON trades(ts_open);

    CREATE TABLE decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        symbol TEXT NOT NULL,
        action TEXT NOT NULL,                          -- BUY / SELL / HOLD
        buy_score REAL, sell_score REAL, hold_score REAL,
        overall_score REAL, agreement_score REAL,
        confidence REAL, contradiction REAL,
        reason TEXT,
        risk_status TEXT,                              -- APPROVED / BLOCKED / N/A
        strategy_version TEXT NOT NULL
    );
    CREATE INDEX idx_decisions_ts ON decisions(ts);

    CREATE TABLE agent_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        decision_id INTEGER REFERENCES decisions(id),
        agent TEXT NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT,
        direction TEXT NOT NULL,
        confidence REAL, strength REAL, market_regime TEXT, data_quality REAL,
        reason TEXT,
        outcome_direction TEXT,                        -- filled later by Learning Engine
        outcome_r REAL,
        resolved_ts TEXT
    );
    CREATE INDEX idx_pred_agent ON agent_predictions(agent, ts);

    CREATE TABLE risk_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        decision_id INTEGER REFERENCES decisions(id),
        level TEXT NOT NULL,                           -- INFO / WARN / BLOCK / CRITICAL
        rule TEXT NOT NULL,
        message TEXT,
        details_json TEXT
    );

    CREATE TABLE market_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        symbol TEXT NOT NULL,
        bid REAL, ask REAL, spread REAL, atr REAL,
        market_regime TEXT, volatility TEXT,
        data_json TEXT
    );

    CREATE TABLE strategy_versions (
        version TEXT PRIMARY KEY,
        created_ts TEXT NOT NULL,
        status TEXT NOT NULL,   -- DRAFT/BACKTESTED/OOS_PASSED/DEMO_FORWARD/APPROVED/ACTIVE/ROLLED_BACK
        parent_version TEXT,
        params_json TEXT,
        notes TEXT
    );

    CREATE TABLE system_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        level TEXT NOT NULL,                           -- DEBUG/INFO/WARN/ERROR/CRITICAL
        component TEXT NOT NULL,
        event TEXT NOT NULL,
        message TEXT,
        details_json TEXT
    );
    CREATE INDEX idx_events_ts ON system_events(ts);

    CREATE TABLE learning_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        trade_id INTEGER REFERENCES trades(id),
        features_json TEXT NOT NULL,
        label_json TEXT,
        strategy_version TEXT NOT NULL
    );
    """),
    (2, """
    CREATE TABLE kv_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        ts TEXT NOT NULL
    )
    """),
    (3, """
    CREATE TABLE events (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT UNIQUE NOT NULL,
        ts TEXT NOT NULL,
        snapshot_id TEXT,
        component TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        version TEXT NOT NULL,
        prev_hash TEXT,
        hash TEXT NOT NULL
    );
    CREATE INDEX idx_ev_type ON events(event_type, seq);
    CREATE INDEX idx_ev_snap ON events(snapshot_id);

    CREATE TABLE experiences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        snapshot_id TEXT UNIQUE,
        symbol TEXT,
        regime TEXT, regime_confidence REAL,
        context_json TEXT, agents_json TEXT, forecast_json TEXT,
        uncertainty TEXT, system_state TEXT, decision TEXT,
        outcome_json TEXT, prediction_error REAL, resolved_ts TEXT,
        feature_version TEXT, model_version TEXT
    );
    CREATE INDEX idx_exp_regime ON experiences(regime);

    CREATE TABLE forecast_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        snapshot_id TEXT NOT NULL,
        model_id TEXT NOT NULL, model_version TEXT NOT NULL,
        role TEXT NOT NULL,                 -- ENSEMBLE / MEMBER / BASELINE / CHALLENGER
        direction TEXT NOT NULL,            -- UP / DOWN / FLAT
        score REAL, p_up_raw REAL,
        horizon_bars INTEGER, timeframe TEXT, regime TEXT,
        entry_price REAL, entry_time INTEGER, atr REAL,
        features_json TEXT,
        actual TEXT, actual_return REAL, correct INTEGER, resolved_ts TEXT
    );
    CREATE INDEX idx_fl_model ON forecast_log(model_id, model_version, role);
    CREATE INDEX idx_fl_open ON forecast_log(actual);
    CREATE INDEX idx_fl_snap ON forecast_log(snapshot_id);

    CREATE TABLE model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id TEXT NOT NULL, version TEXT NOT NULL,
        created_ts TEXT NOT NULL, feature_version TEXT NOT NULL,
        train_dataset_ref TEXT, eval_dataset_ref TEXT,
        status TEXT NOT NULL,               -- CHAMPION / STABLE / CHALLENGER / VALIDATED / REJECTED / RETIRED
        params_json TEXT, state_json TEXT, metrics_json TEXT, notes TEXT,
        UNIQUE(model_id, version)
    );

    CREATE TABLE experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT UNIQUE NOT NULL,
        created_ts TEXT NOT NULL,
        hypothesis TEXT NOT NULL,
        code_version TEXT, data_version TEXT, feature_version TEXT,
        params_json TEXT, eval_range TEXT,
        status TEXT NOT NULL,               -- RUNNING / COMPLETED / FAILED
        validation_json TEXT, outcome TEXT
    );
    """),
]

TABLES = ("trades", "agent_predictions", "decisions", "risk_events",
          "market_snapshots", "strategy_versions", "system_events",
          "learning_samples", "events", "experiences", "forecast_log",
          "model_versions", "experiments")

_INSERTABLE = set(TABLES)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")

    # ---- schema -----------------------------------------------------------
    def init_schema(self) -> int:
        """Apply pending migrations. Safe to call on every start (idempotent)."""
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_ts TEXT NOT NULL)")
            row = self._conn.execute("SELECT COALESCE(MAX(version),0) AS v FROM schema_version").fetchone()
            current = row["v"]
            for version, sql in MIGRATIONS:
                if version <= current:
                    continue
                self._conn.execute("BEGIN")
                try:
                    for stmt in _split(sql):
                        self._conn.execute(stmt)
                    self._conn.execute("INSERT INTO schema_version VALUES (?,?)", (version, _now()))
                    self._conn.execute("COMMIT")
                except Exception:
                    self._conn.execute("ROLLBACK")
                    raise
                current = version
            self._seed_strategy_v1()
            return current

    def _seed_strategy_v1(self) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO strategy_versions(version, created_ts, status, notes) VALUES (?,?,?,?)",
            ("v1.0", _now(), "DRAFT", "Initial rule-based version. Not yet validated."))

    def schema_version(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COALESCE(MAX(version),0) FROM schema_version").fetchone()[0]

    def tables(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
        return sorted(r["name"] for r in rows)

    # ---- generic helpers --------------------------------------------------
    def insert(self, table: str, row: dict[str, Any]) -> int:
        if table not in _INSERTABLE:
            raise ValueError(f"unknown table {table!r}")
        cols = list(row)
        for c in cols:  # column names come from code, but never trust blindly
            if not c.replace("_", "").isalnum():
                raise ValueError(f"bad column name {c!r}")
        sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
        with self._lock:
            cur = self._conn.execute(sql, [row[c] for c in cols])
            return cur.lastrowid

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, tuple(params)).fetchall()]

    def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self._lock:
            self._conn.execute(sql, tuple(params))

    def count(self, table: str) -> int:
        if table not in _INSERTABLE:
            raise ValueError(f"unknown table {table!r}")
        with self._lock:
            return self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def update(self, table: str, row_id: int, row: dict[str, Any]) -> None:
        if table not in _INSERTABLE:
            raise ValueError(f"unknown table {table!r}")
        cols = list(row)
        for c in cols:
            if not c.replace("_", "").isalnum():
                raise ValueError(f"bad column name {c!r}")
        sql = f"UPDATE {table} SET {','.join(c + '=?' for c in cols)} WHERE id=?"
        with self._lock:
            self._conn.execute(sql, [row[c] for c in cols] + [row_id])

    # ---- key/value state + idempotency ------------------------------------
    def kv_get(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            r = self._conn.execute("SELECT value FROM kv_state WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

    def kv_set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO kv_state(key,value,ts) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts=excluded.ts",
                (key, value, _now()))

    def kv_delete(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM kv_state WHERE key=?", (key,))

    def claim_command(self, command_id: str) -> bool:
        """Atomically claim an idempotency key. False => already seen (duplicate)."""
        with self._lock:
            try:
                self._conn.execute("INSERT INTO kv_state(key,value,ts) VALUES(?,?,?)",
                                   ("cmd:" + command_id, "claimed", _now()))
                return True
            except sqlite3.IntegrityError:
                return False

    # ---- convenience writers ----------------------------------------------
    def log_event(self, level: str, component: str, event: str,
                  message: str = "", details: dict | None = None) -> int:
        return self.insert("system_events", {
            "ts": _now(), "level": level, "component": component, "event": event,
            "message": message, "details_json": json.dumps(details or {}, sort_keys=True)})

    def insert_agent_prediction(self, result, decision_id: int | None = None) -> int:
        return self.insert("agent_predictions", {
            "ts": result.timestamp.isoformat(), "decision_id": decision_id,
            "agent": result.agent, "symbol": result.symbol, "timeframe": result.timeframe,
            "direction": result.direction.value, "confidence": result.confidence,
            "strength": result.strength, "market_regime": result.market_regime.value,
            "data_quality": result.data_quality, "reason": result.reason})

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _split(sql: str) -> list[str]:
    # Migrations contain no semicolons inside strings, so a plain split is safe.
    return [s.strip() for s in sql.split(";") if s.strip()]
