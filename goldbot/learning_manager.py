"""Learning Manager. Lifecycle (nothing skips a stage, nothing deploys itself):

  experience memory -> evaluation -> drift detection -> CANDIDATE
     -> offline REPLAY + OUT-OF-SAMPLE validation (leak-free walk-forward)    [propose_challenger]
     -> SHADOW evaluation next to the champion on live/simulated data          [evaluate_challengers]
     -> VALIDATED (eligible) -> human PROMOTE  |  REJECTED
  the previous champion is kept as STABLE so a rollback is always possible."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

from .data import TF_SECONDS, Candle, resample
from .forecast import (DOWN, FLAT, UP, NaiveMajority, OnlineLogistic, Persistence, StatBaseline, forecast_features)

GRID = [{"lr": 0.02, "l2": 0.01, "warmup": 50, "up_thr": 0.55}, {"lr": 0.1, "l2": 0.001, "warmup": 50, "up_thr": 0.58},
        {"lr": 0.05, "l2": 0.001, "warmup": 100, "up_thr": 0.55}, {"lr": 0.03, "l2": 0.003, "warmup": 80, "up_thr": 0.60}]


def forecast_replay(c: list[Candle], horizon: int, flat_thr_atr: float, models: dict) -> dict:
    """Leak-free walk-forward replay of forecasting on one candle series.
    At step j: (1) resolve forecasts whose target bar is j (their outcome becomes known only now) and let models
    LEARN from them, (2) THEN predict using features of candles[:j+1] only. Returns name -> [(bar, correct, pred, actual)]."""
    pending: list[tuple] = []
    log = {n: [] for n in models}
    for j in range(60, len(c)):
        due = [p for p in pending if p[0] + horizon == j]
        pending = [p for p in pending if p[0] + horizon != j]
        for i, f, preds in due:
            move = c[j].close - c[i].close
            thr = flat_thr_atr * f["_atr"]
            actual = UP if move > thr else DOWN if move < -thr else FLAT
            for n, m in models.items():
                log[n].append((i, int(preds[n] == actual), preds[n], actual))
                m.learn(f, actual)
        f = forecast_features(c[max(0, j - 300):j + 1])
        if f is not None:
            pending.append((j, f, {n: m.predict(f).direction for n, m in models.items()}))
    return log


def _paired(a: list, b: list) -> tuple[int, int, int, float]:
    """McNemar-style: a, b are [(bar, correct,...)] aligned on bar."""
    bb = {x[0]: x[1] for x in b}
    only_a = only_b = n = 0
    for bar, ca, *_ in a:
        if bar in bb:
            n += 1
            only_a += ca == 1 and bb[bar] == 0
            only_b += ca == 0 and bb[bar] == 1
    z = (only_a - only_b) / math.sqrt(only_a + only_b) if only_a + only_b else 0.0
    return n, only_a, only_b, z


def offline_compare(c: list[Candle], settings, champion: dict, candidate: dict, split: float = 0.7) -> dict:
    thr = settings.forecast_flat_threshold_atr
    models = {"champion": OnlineLogistic("c", champion.get("lr", .05), champion.get("l2", 1e-3), champion.get("warmup", 50), champion.get("up_thr", .55)),
              "candidate": OnlineLogistic("x", candidate["lr"], candidate["l2"], candidate["warmup"], candidate["up_thr"]),
              "persistence": Persistence(thr), "stat_drift": StatBaseline(), "naive": NaiveMajority()}
    log = forecast_replay(c, settings.forecast_horizon_bars, thr, models)
    cut = int(len(c) * split)
    oos = {n: [x for x in v if x[0] >= cut] for n, v in log.items()}
    acc = {n: (sum(x[1] for x in v) / len(v) if v else None) for n, v in oos.items()}
    n, a_only, b_only, z = _paired(oos["candidate"], oos["champion"])
    return {"oos_samples": n, "oos_accuracy": {k: (None if v is None else round(v, 4)) for k, v in acc.items()},
            "candidate_vs_champion": {"candidate_only_correct": a_only, "champion_only_correct": b_only, "z": round(z, 2)},
            "split_bar": cut, "total_bars": len(c)}


class LearningManager:
    def __init__(self, db, settings, events, registry, forecaster, drift, learning, lab, metrics=None):
        self.db, self.s, self.events, self.registry = db, settings, events, registry
        self.forecaster, self.drift, self.learning, self.lab, self.metrics = forecaster, drift, learning, lab, metrics
        self.last_eval: dict = {}
        self.last_error: str = ""

    # ---- 1. periodic evaluation (read-only) --------------------------------------------------------------
    def evaluate(self, now: float) -> dict:
        try:
            board = self.forecaster.scoreboard()
            cal = self.forecaster.calibration_info()
            flags = self.drift.check(cal, None, now)
            shadow = self.evaluate_challengers()
            self.last_eval = {"ts": now, "scoreboard": board, "calibration": cal, "drift_flags": flags, "challengers": shadow,
                              "agent_performance": self.learning.agent_performance()}
            self.events.emit("learning", "MODEL_EVALUATED", {"models": len(board), "drift_flags": len(flags), "calibration": cal.get("status"),
                             "challengers": [c["version"] + ":" + c["decision"] for c in shadow]}, None, now)
            self.last_error = ""
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            raise
        return self.last_eval

    # ---- 2. candidate -> offline replay + OOS ----------------------------------------------------------------
    def propose_challenger(self, m1: list[Candle], variant: dict | None = None, force: bool = False) -> dict:
        reg = self.registry
        champ = next((r for r in reg.rows("CHAMPION", "logistic")), None)
        if champ is None:
            raise ValueError("no logistic champion registered")
        tried = [json.loads(r["params_json"]) for r in reg.rows(model_id="logistic")]
        cand = variant or next((g for g in GRID if g not in tried), None)
        if cand is None:
            raise ValueError("all built-in variants were already tried; pass a custom variant")
        c = resample(m1, TF_SECONDS[self.s.forecast_timeframe])
        dref = self.lab.data_version(m1)
        eid = self.lab.start(f"logistic variant {cand} is not worse than the champion out-of-sample", cand, dref,
                             f"oos from bar {int(len(c) * .7)}/{len(c)} ({self.s.forecast_timeframe})", eval_ref=f"{dref}:fc-oos", force=force)
        res = offline_compare(c, self.s, json.loads(champ["params_json"]), cand)
        ok = res["oos_samples"] >= 100 and res["candidate_vs_champion"]["z"] >= -1.0
        version = f"1.{len(reg.rows(model_id='logistic'))}"
        status = "CHALLENGER" if ok else "REJECTED"
        reg.register("logistic", version, status, cand, train_ref=dref, eval_ref=f"{dref}:fc-oos",
                     notes="offline replay+OOS " + ("passed (not worse than champion) -> shadow evaluation" if ok else "failed -> rejected"))
        reg.set_status("logistic", version, status, json.dumps(res), res)
        self.lab.finish(eid, res, f"{version} {status} at offline stage")
        self.forecaster.reload()
        return {"version": version, "status": status, "offline": res, "experiment_id": eid}

    # ---- 3. shadow evaluation (challenger runs next to champion on the SAME snapshots) ------------------------
    def evaluate_challengers(self) -> list[dict]:
        out = []
        champ = next((r for r in self.registry.rows("CHAMPION", "logistic")), None)
        for ch in self.registry.rows("CHALLENGER", "logistic"):
            if champ is None:
                continue
            q = self.db.query(
                "SELECT c.regime reg, COUNT(*) n, SUM(c.correct=1 AND m.correct=0) a_only, SUM(c.correct=0 AND m.correct=1) b_only, "
                "SUM(c.correct) ca, SUM(m.correct) ma FROM forecast_log c JOIN forecast_log m ON c.snapshot_id=m.snapshot_id "
                "AND m.model_id='logistic' AND m.model_version=? AND m.role='MEMBER' "
                "WHERE c.model_id='logistic' AND c.model_version=? AND c.role='CHALLENGER' "
                "AND c.actual IN ('UP','DOWN','FLAT') AND m.actual IN ('UP','DOWN','FLAT') GROUP BY c.regime", (champ["version"], ch["version"]))
            n = sum(r["n"] for r in q)
            a, b = sum(r["a_only"] or 0 for r in q), sum(r["b_only"] or 0 for r in q)
            z = (a - b) / math.sqrt(a + b) if a + b else 0.0
            bad_regimes = [r["reg"] for r in q if r["n"] >= 30 and (r["ca"] / r["n"]) < (r["ma"] / r["n"]) - 0.10]
            base = self.db.query("SELECT AVG(b.correct) acc FROM forecast_log c JOIN forecast_log b ON c.snapshot_id=b.snapshot_id AND b.role='BASELINE' "
                                 "WHERE c.model_id='logistic' AND c.model_version=? AND c.role='CHALLENGER' AND c.actual IN ('UP','DOWN','FLAT') "
                                 "AND b.actual IN ('UP','DOWN','FLAT') GROUP BY b.model_id ORDER BY acc DESC LIMIT 1", (ch["version"],))
            best_base = base[0]["acc"] if base else None
            ch_acc = (sum(r["ca"] or 0 for r in q) / n) if n else None
            m = {"version": ch["version"], "paired_samples": n, "challenger_only_correct": a, "champion_only_correct": b, "z": round(z, 2),
                 "regimes_worse": bad_regimes, "challenger_accuracy": ch_acc, "best_baseline_accuracy": best_base}
            need = 2 * self.s.forecast_min_samples
            if n < need:
                dec = f"COLLECTING ({n}/{need} paired samples)"
            elif z >= 1.96 and not bad_regimes and (best_base is None or (ch_acc or 0) >= best_base):
                dec = "VALIDATED"
                self.registry.set_status("logistic", ch["version"], "VALIDATED", "shadow evaluation passed; awaiting human promotion", m)
            elif z <= -1.96 or (n >= 2 * need and z < 1.0):
                dec = "REJECTED"
                self.registry.set_status("logistic", ch["version"], "REJECTED", "shadow evaluation failed", m)
            else:
                dec = "CONTINUE (no significant difference yet)"
            m["decision"] = dec
            out.append(m)
        if any(x["decision"] in ("VALIDATED", "REJECTED") for x in out):
            self.forecaster.reload()
        return out

    # ---- 4. human-gated promotion / rollback --------------------------------------------------------------------
    def promote(self, model_id: str, version: str, approver: str, evidence: str) -> None:
        self.registry.promote(model_id, version, approver, evidence)
        self.forecaster.reload()

    def reject(self, model_id: str, version: str, reason: str) -> None:
        self.registry.set_status(model_id, version, "REJECTED", reason)
        self.forecaster.reload()

    def rollback(self, model_id: str, reason: str) -> str:
        v = self.registry.rollback(model_id, reason)
        self.forecaster.reload()
        return v

    def status(self) -> dict:
        return {"models": [{k: r[k] for k in ("model_id", "version", "status", "feature_version", "created_ts", "train_dataset_ref", "eval_dataset_ref")}
                           for r in self.registry.rows()],
                "last_evaluation": {k: v for k, v in self.last_eval.items() if k in ("ts", "calibration", "drift_flags", "challengers")},
                "last_error": self.last_error}
