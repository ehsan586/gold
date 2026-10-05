"""Experiment Lab: every experiment records hypothesis, code version, data version, feature version, parameters,
evaluation range, validation result and outcome, so 'what changed / what improved / what did not' is answerable.
A holdout guard stops the same evaluation set from being used again and again for selection (that silently
turns an out-of-sample set into a training set)."""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .data import Candle

ABLATIONS = {
    "FULL": {},
    "WITHOUT_TREND": {"disabled_agents": {"trend"}}, "WITHOUT_STRUCTURE": {"disabled_agents": {"structure"}},
    "WITHOUT_PRICE_ACTION": {"disabled_agents": {"price_action"}}, "WITHOUT_MOMENTUM": {"disabled_agents": {"momentum"}},
    "WITHOUT_VOLATILITY": {"disabled_agents": {"volatility"}}, "WITHOUT_LIQUIDITY": {"disabled_agents": {"liquidity"}},
    "WITHOUT_NEWS": {"disabled_agents": {"news"}},
    "WITHOUT_FORECAST": {"use_forecast": False}, "WITHOUT_REGIME": {"use_regime": False},
}


class HoldoutExhausted(RuntimeError):
    pass


def welch_z(a: list[float], b: list[float]) -> float | None:
    if len(a) < 30 or len(b) < 30:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    return (ma - mb) / se if se > 0 else None


class ExperimentLab:
    def __init__(self, db, settings, events=None):
        self.db, self.s, self.events = db, settings, events

    @staticmethod
    def code_version() -> str:
        h = hashlib.sha256()
        for f in sorted(Path(__file__).parent.glob("*.py")):
            h.update(f.read_bytes())
        return h.hexdigest()[:12]

    @staticmethod
    def data_version(m1: list[Candle]) -> str:
        step = max(1, len(m1) // 50)
        h = hashlib.sha256(f"{len(m1)}|{m1[0].time}|{m1[-1].time}".encode())
        for c in m1[::step]:
            h.update(f"{c.time},{c.close}".encode())
        return h.hexdigest()[:12]

    def holdout_uses(self, ref: str) -> int:
        return int(self.db.kv_get("evaluse:" + ref) or 0)

    def register_evaluation(self, ref: str, force: bool = False) -> int:
        n = self.holdout_uses(ref) + 1
        if n > self.s.max_eval_reuse and not force:
            raise HoldoutExhausted(f"evaluation set {ref} already used {n - 1} times (limit {self.s.max_eval_reuse}). "
                                   "Repeated tuning on one evaluation set overfits it: use new, unseen data.")
        self.db.kv_set("evaluse:" + ref, str(n))
        return n

    def start(self, hypothesis: str, params: dict, data_ref: str, eval_range: str, eval_ref: str | None = None, force: bool = False) -> str:
        if not hypothesis.strip():
            raise ValueError("an experiment needs a hypothesis")
        if eval_ref:
            self.register_evaluation(eval_ref, force)
        eid = "EXP-" + uuid.uuid4().hex[:8].upper()
        self.db.insert("experiments", {"experiment_id": eid, "created_ts": datetime.now(timezone.utc).isoformat(), "hypothesis": hypothesis,
                                       "code_version": self.code_version(), "data_version": data_ref, "feature_version": self.s.feature_version,
                                       "params_json": json.dumps(params, sort_keys=True, default=str), "eval_range": eval_range, "status": "RUNNING"})
        return eid

    def finish(self, eid: str, validation: dict, outcome: str, status: str = "COMPLETED") -> None:
        self.db.execute("UPDATE experiments SET status=?, validation_json=?, outcome=? WHERE experiment_id=?",
                        (status, json.dumps(validation, sort_keys=True, default=str), outcome, eid))
        if self.events:
            self.events.emit("experiments", "EXPERIMENT", {"id": eid, "status": status, "outcome": outcome[:200]}, None, 0.0)

    def list(self, n: int = 50) -> list[dict]:
        return self.db.query("SELECT experiment_id,created_ts,hypothesis,code_version,data_version,feature_version,params_json,"
                             "eval_range,status,validation_json,outcome FROM experiments ORDER BY id DESC LIMIT ?", (n,))

    def run_ablation(self, m1: list[Candle], settings, variants: list[str] | None = None, split: float = 0.7,
                     warmup: int = 3700, force: bool = False, **bt) -> dict:
        """Measures each component's INCREMENTAL contribution on the out-of-sample segment only."""
        from .backtest import run_backtest
        names = variants or list(ABLATIONS)
        if "FULL" not in names:
            names = ["FULL"] + names
        cut = int(len(m1) * split)
        seg = m1[max(0, cut - warmup):]
        dref = self.data_version(m1)
        eid = self.start("Each ablated component contributes nothing measurable on out-of-sample data",
                         {"variants": names, "split": split, **{k: v for k, v in bt.items()}}, dref,
                         f"oos[{cut}:{len(m1)}] of {len(m1)} M1 bars", eval_ref=f"{dref}:oos{split}", force=force)
        out: dict = {}
        try:
            for n in names:
                out[n] = run_backtest(seg, settings, warmup=warmup, **ABLATIONS[n], **bt)
            full = out["FULL"]
            for n, r in out.items():
                if n == "FULL":
                    continue
                z = welch_z(full["r_values"], r["r_values"])
                r["delta_vs_full"] = {"net_profit": round(r["net_profit"] - full["net_profit"], 2),
                                      "expectancy_R": round(r["expectancy_R"] - full["expectancy_R"], 3),
                                      "z_expectancy": None if z is None else round(z, 2),
                                      "verdict": "INSUFFICIENT_DATA (need >=30 trades per variant)" if z is None else
                                                 ("component HELPS" if z >= 1.96 else "component HURTS" if z <= -1.96 else "no measurable difference")}
            self.finish(eid, {k: {kk: vv for kk, vv in v.items() if kk != "r_values"} for k, v in out.items()},
                        "ablation finished; see delta_vs_full per variant")
        except Exception as e:  # noqa: BLE001
            self.finish(eid, {}, f"failed: {e}", "FAILED")
            raise
        return {"experiment_id": eid, "results": out}
