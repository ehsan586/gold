"""Learning Engine: records, evaluates and PROPOSES. It can never change live trading.

Pipeline for any change:  DRAFT -> BACKTESTED -> OOS_PASSED -> DEMO_FORWARD -> APPROVED -> ACTIVE
Each step needs written evidence; ACTIVE needs a named human approver; rollback is one call.
The running engine only picks up a new version on an explicit reload command.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone

from .db import Database
from .decision import DEFAULT_WEIGHTS, DIRECTIONAL
from .settings import Settings

PIPELINE = ["DRAFT", "BACKTESTED", "OOS_PASSED", "DEMO_FORWARD", "APPROVED", "ACTIVE"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LearningEngine:
    def __init__(self, db: Database, settings: Settings):
        self.db, self.s = db, settings

    # ---------------------------------------------------------------- recording
    def on_trade_closed(self, trade_id: int) -> None:
        rows = self.db.query("SELECT * FROM trades WHERE id=?", (trade_id,))
        if not rows or rows[0]["pnl"] is None:
            return
        t = rows[0]
        agents = json.loads(t["agent_outputs_json"] or "{}")
        features = {"agents": agents, "regime": t["market_regime"], "atr": t["atr"], "spread": t["spread"],
                    "volatility": t["volatility"], "decision_score": t["decision_score"],
                    "agreement_score": t["agreement_score"], "direction": t["direction"]}
        label = {"pnl": t["pnl"], "r": t["r_multiple"], "win": t["pnl"] > 0, "mfe": t["mfe"], "mae": t["mae"],
                 "exit_reason": t["exit_reason"], "duration_sec": t["duration_sec"]}
        self.db.insert("learning_samples", {"ts": _now(), "trade_id": trade_id, "features_json": json.dumps(features),
                                            "label_json": json.dumps(label), "strategy_version": t["strategy_version"]})
        if t["decision_id"]:   # store each agent's later actual outcome
            correct_dir = t["direction"] if t["pnl"] > 0 else ("SELL" if t["direction"] == "BUY" else "BUY")
            for name in agents:
                self._resolve(t["decision_id"], name, correct_dir, t["r_multiple"])

    def _resolve(self, decision_id, agent, correct_dir, r):
        rows = self.db.query("SELECT id FROM agent_predictions WHERE decision_id=? AND agent=?", (decision_id, agent))
        for row in rows:
            self.db.update("agent_predictions", row["id"],
                           {"outcome_direction": correct_dir, "outcome_r": r, "resolved_ts": _now()})

    # ---------------------------------------------------------------- statistics
    def agent_performance(self, min_samples: int | None = None) -> dict:
        min_samples = min_samples or self.s.learning_min_samples
        trades = self.db.query("SELECT direction,pnl,r_multiple,market_regime,volatility,agent_outputs_json "
                               "FROM trades WHERE status='CLOSED' AND pnl IS NOT NULL")
        acc = defaultdict(lambda: {"supported": [], "n_dir": 0, "correct": 0, "conf": [],
                                   "regime": defaultdict(list), "vol": defaultdict(list), "tf": defaultdict(list)})
        for t in trades:
            try:
                outs = json.loads(t["agent_outputs_json"] or "{}")
            except ValueError:
                continue
            for name, o in outs.items():
                d = o.get("direction")
                if d not in ("BUY", "SELL") or o.get("data_quality", 0) == 0:
                    continue
                a = acc[name]
                a["n_dir"] += 1
                a["conf"].append(o.get("confidence", 0))
                supported = d == t["direction"]
                win = t["pnl"] > 0
                if (supported and win) or (not supported and not win):
                    a["correct"] += 1
                if supported:
                    a["supported"].append((t["pnl"], t["r_multiple"] or 0.0))
                    a["regime"][t["market_regime"] or "UNKNOWN"].append(t["pnl"])
                    a["vol"][t["volatility"] or "UNKNOWN"].append(t["pnl"])
                    a["tf"][o.get("timeframe", "?")].append(t["pnl"])
        out = {}
        for name, a in acc.items():
            sup = a["supported"]
            wins = [p for p, _ in sup if p > 0]
            loss = [-p for p, _ in sup if p < 0]
            grp = lambda g: {k: {"n": len(v), "net": round(sum(v), 2), "win_rate": round(100 * sum(1 for x in v if x > 0) / len(v), 1)}  # noqa: E731
                             for k, v in g.items()}
            out[name] = {
                "n": a["n_dir"], "supported_trades": len(sup),
                "accuracy": round(a["correct"] / a["n_dir"], 4) if a["n_dir"] else 0.0,
                "win_rate": round(len(wins) / len(sup), 4) if sup else 0.0,
                "avg_r": round(sum(r for _, r in sup) / len(sup), 3) if sup else 0.0,
                "profit_factor": round(sum(wins) / sum(loss), 3) if loss else (None if not wins else math.inf),
                "avg_confidence": round(sum(a["conf"]) / len(a["conf"]), 1) if a["conf"] else 0.0,
                "by_regime": grp(a["regime"]), "by_volatility": grp(a["vol"]), "by_timeframe": grp(a["tf"]),
                "sufficient_sample": a["n_dir"] >= min_samples,
            }
        return out

    # ---------------------------------------------------------------- strategy versions
    def ensure_baseline(self) -> None:
        if self.db.kv_get("active_strategy_version") is None:
            self.db.kv_set("active_strategy_version", "v1.0")
            self.db.execute("UPDATE strategy_versions SET status='ACTIVE', notes=? WHERE version='v1.0'",
                                  ("Baseline rule-based version. NOT validated by backtest/forward test yet.",))

    def active_version(self) -> str:
        self.ensure_baseline()
        return self.db.kv_get("active_strategy_version") or "v1.0"

    def load_params(self, version: str) -> dict:
        r = self.db.query("SELECT params_json FROM strategy_versions WHERE version=?", (version,))
        return json.loads(r[0]["params_json"]) if r and r[0]["params_json"] else {}

    def list_versions(self) -> list[dict]:
        return self.db.query("SELECT version,status,parent_version,created_ts,notes,params_json "
                             "FROM strategy_versions ORDER BY created_ts, version")

    def _next_version(self) -> str:
        n = max([int(v["version"].split(".")[1]) for v in self.list_versions()] + [0])
        return f"v1.{n + 1}"

    def create_version(self, parent: str, params: dict, notes: str) -> str:
        v = self._next_version()
        self.db.insert("strategy_versions", {"version": v, "created_ts": _now(), "status": "DRAFT",
                                             "parent_version": parent, "params_json": json.dumps(params, sort_keys=True),
                                             "notes": notes})
        self.db.log_event("INFO", "learning", "version_created", f"{v} proposed from {parent}: {notes}")
        return v

    def advance(self, version: str, new_status: str, evidence: str, approver: str = "") -> None:
        rows = self.db.query("SELECT status FROM strategy_versions WHERE version=?", (version,))
        if not rows:
            raise ValueError("unknown version")
        cur = rows[0]["status"]
        if cur not in PIPELINE or new_status not in PIPELINE or PIPELINE.index(new_status) != PIPELINE.index(cur) + 1:
            raise ValueError(f"illegal status change {cur} -> {new_status} (steps cannot be skipped)")
        if not evidence.strip():
            raise ValueError("evidence is required")
        if new_status == "ACTIVE":
            if not approver.strip():
                raise ValueError("a named human approver is required to activate")
            prev = self.active_version()
            self.db.execute("UPDATE strategy_versions SET status='SUPERSEDED' WHERE version=?", (prev,))
            self.db.kv_set("active_strategy_version", version)
        self.db.execute("UPDATE strategy_versions SET status=?, notes=COALESCE(notes,'')||? WHERE version=?",
                              (new_status, f"\n[{_now()}] {cur}->{new_status} {approver}: {evidence}", version))
        self.db.log_event("INFO", "learning", "version_status", f"{version}: {cur} -> {new_status}", {"approver": approver})

    def rollback(self, reason: str) -> str:
        cur = self.active_version()
        parent = self.db.query("SELECT parent_version FROM strategy_versions WHERE version=?", (cur,))
        parent = parent[0]["parent_version"] if parent else None
        if not parent:
            raise ValueError("active version has no parent to roll back to")
        self.db.execute("UPDATE strategy_versions SET status='ROLLED_BACK', notes=COALESCE(notes,'')||? WHERE version=?",
                              (f"\n[{_now()}] rolled back: {reason}", cur))
        self.db.execute("UPDATE strategy_versions SET status='ACTIVE' WHERE version=?", (parent,))
        self.db.kv_set("active_strategy_version", parent)
        self.db.log_event("WARN", "learning", "rollback", f"{cur} -> {parent}: {reason}")
        return parent

    # ---------------------------------------------------------------- proposals
    def propose_weight_changes(self, current: dict | None = None) -> dict | None:
        """Only with enough samples AND a statistically meaningful edge/deficit (z-test vs 50%).
        Change per weight is capped at +/-10%. The result is a DRAFT; nothing is applied."""
        current = {**DEFAULT_WEIGHTS, **(current or self.load_params(self.active_version()).get("weights", {}))}
        perf, new, evidence = self.agent_performance(), dict(current), []
        for name in DIRECTIONAL:
            p = perf.get(name)
            if not p or not p["sufficient_sample"]:
                continue
            z = (p["accuracy"] - 0.5) / math.sqrt(0.25 / p["n"])
            if z >= 1.645:
                new[name] = round(current[name] * 1.10, 4)
                evidence.append(f"{name}: accuracy {p['accuracy']:.0%} over {p['n']} samples (z={z:.2f}) -> weight +10%")
            elif z <= -1.645:
                new[name] = round(current[name] * 0.90, 4)
                evidence.append(f"{name}: accuracy {p['accuracy']:.0%} over {p['n']} samples (z={z:.2f}) -> weight -10%")
        if not evidence:
            return None
        v = self.create_version(self.active_version(), {"weights": new}, "; ".join(evidence))
        return {"version": v, "weights": new, "evidence": evidence}

    def summary(self) -> dict:
        n = self.db.count("learning_samples")
        return {"samples": n, "min_samples": self.s.learning_min_samples,
                "active_version": self.active_version(), "versions": self.list_versions(),
                "agent_performance": self.agent_performance(),
                "note": ("Not enough samples: weights are NOT adjusted." if n < self.s.learning_min_samples else "")}
