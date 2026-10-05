"""Modular Forecasting System: several independent models + simple baselines, combined with an explicit
uncertainty estimate. NO single 'magic' predictor. Every forecast is logged and later RESOLVED against what
actually happened (only data after the forecast time is used), so every claim is measurable:

  complex models  vs  simple baselines (persistence, statistical)  vs  naive baseline (majority class)

Classes: UP / DOWN / FLAT (move over `forecast_horizon_bars` bars relative to a |ATR| dead-band).
A probability is only reported when it is CALIBRATED on enough resolved forecasts; otherwise it is None."""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .data import TF_SECONDS, Candle, iso
from .indicators import atr, ema, median, rsi
from .snapshot import MarketSnapshot

UP, DOWN, FLAT = "UP", "DOWN", "FLAT"
FORECAST_FEATURE_VERSION = "f1"
FEATURES = ["r1", "r3", "r6", "r12", "er20", "slope", "z20", "rsi", "atr_ratio"]


def forecast_features(c: list[Candle]) -> dict | None:
    """Past-only features of the closed candles (the last element is the newest CLOSED bar)."""
    if len(c) < 60:
        return None
    cl = [x.close for x in c]
    a_all = atr(c)
    a = a_all[-1] or 1e-9
    e20 = ema(cl, 20)
    w = cl[-20:]
    mean = sum(w) / 20
    sd = math.sqrt(sum((x - mean) ** 2 for x in w) / 20) or 1e-9
    path = sum(abs(cl[i] - cl[i - 1]) for i in range(len(cl) - 20, len(cl)))
    rets = [cl[i] - cl[i - 1] for i in range(len(cl) - 48, len(cl))]
    m = sum(rets) / 48
    sdr = math.sqrt(sum((x - m) ** 2 for x in rets) / 47) or 1e-9
    return {"r1": (cl[-1] - cl[-2]) / a, "r3": (cl[-1] - cl[-4]) / a, "r6": (cl[-1] - cl[-7]) / a, "r12": (cl[-1] - cl[-13]) / a,
            "er20": abs(cl[-1] - cl[-21]) / path if path > 0 else 0.0, "slope": (e20[-1] - e20[-6]) / a,
            "z20": (cl[-1] - mean) / sd, "rsi": (rsi(cl)[-1] - 50) / 50, "atr_ratio": a / (median(a_all[-100:]) or 1e-9),
            "tstat48": m / (sdr / math.sqrt(48)), "_atr": a, "_close": cl[-1]}


@dataclass
class ModelForecast:
    model_id: str
    version: str
    role: str
    direction: str
    score: float = 0.0              # signed strength in [-1, 1] (NOT a probability)
    p_up_raw: float | None = None   # raw model output, uncalibrated; None when the model has none
    valid: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return {"model": self.model_id, "version": self.version, "role": self.role, "direction": self.direction,
                "score": round(self.score, 3), "p_up_raw": None if self.p_up_raw is None else round(self.p_up_raw, 3),
                "valid": self.valid, "note": self.note}


class Model:
    model_id, version, role = "base", "1.0", "MEMBER"
    def predict(self, f: dict) -> ModelForecast: raise NotImplementedError
    def learn(self, f: dict, actual: str) -> None: pass
    def state(self) -> dict: return {}
    def load_state(self, s: dict) -> None: pass
    def _mk(self, d, score=0.0, p=None, note="", valid=True):
        return ModelForecast(self.model_id, self.version, self.role, d, score, p, valid, note)


# ---- baselines ----------------------------------------------------------------------------
class NaiveMajority(Model):
    """Naive baseline: always predict the class that has been most frequent so far (FLAT before any data)."""
    model_id, role = "naive_majority", "BASELINE"
    def __init__(self, counts: Counter | None = None):
        self.counts = counts or Counter()
    def predict(self, f):
        d = self.counts.most_common(1)[0][0] if self.counts else FLAT
        return self._mk(d, note="majority class so far")
    def learn(self, f, actual):
        self.counts[actual] += 1
    def state(self): return dict(self.counts)
    def load_state(self, s): self.counts = Counter(s)


class Persistence(Model):
    """Persistence baseline: the next move repeats the last move of the same length."""
    model_id, role = "persistence", "BASELINE"
    def __init__(self, thr: float): self.thr = thr
    def predict(self, f):
        r = f["r12"]
        return self._mk(UP if r > self.thr else DOWN if r < -self.thr else FLAT, max(-1, min(1, r / 2)))


class StatBaseline(Model):
    """Simple statistical baseline: drift of the last 48 one-bar returns, only if its t-statistic is >= 1."""
    model_id, role = "stat_drift", "BASELINE"
    def predict(self, f):
        t = f["tstat48"]
        return self._mk(UP if t >= 1.0 else DOWN if t <= -1.0 else FLAT, max(-1, min(1, t / 3)))


# ---- ensemble members ------------------------------------------------------------------------
class EmaSlope(Model):
    model_id = "ema_slope"
    def predict(self, f):
        s = f["slope"]
        return self._mk(UP if s > 0.15 else DOWN if s < -0.15 else FLAT, max(-1, min(1, s / 0.5)))


class MeanReversion(Model):
    model_id = "mean_rev"
    def predict(self, f):
        z = f["z20"]
        return self._mk(DOWN if z > 1.5 else UP if z < -1.5 else FLAT, -max(-1, min(1, z / 3)))


class OnlineLogistic(Model):
    """Tiny pure-python logistic regression trained ONLINE on RESOLVED forecasts only (strictly past labels).
    Stays FLAT (no probability) until `warmup` training samples exist."""
    model_id = "logistic"
    def __init__(self, version="1.0", lr=0.05, l2=1e-3, warmup=50, up_thr=0.55, role="MEMBER"):
        self.version, self.lr, self.l2, self.warmup, self.up_thr, self.role = version, lr, l2, warmup, up_thr, role
        self.w, self.b = [0.0] * len(FEATURES), 0.0
        self.mu, self.m2, self.n = [0.0] * len(FEATURES), [0.0] * len(FEATURES), 0

    def _x(self, f):
        sd = [math.sqrt(m / self.n) if self.n > 1 and m > 0 else 1.0 for m in self.m2]
        return [(f[k] - self.mu[i]) / (sd[i] or 1.0) for i, k in enumerate(FEATURES)]

    def predict(self, f):
        if self.n < self.warmup:
            return self._mk(FLAT, 0.0, None, f"warming up ({self.n}/{self.warmup} training samples)")
        z = sum(w * x for w, x in zip(self.w, self._x(f))) + self.b
        p = 1 / (1 + math.exp(-max(-30, min(30, z))))
        return self._mk(UP if p > self.up_thr else DOWN if p < 1 - self.up_thr else FLAT, 2 * p - 1, p)

    def learn(self, f, actual):
        if actual not in (UP, DOWN):
            return
        self.n += 1
        for i, k in enumerate(FEATURES):               # Welford running mean/variance (training samples only)
            d = f[k] - self.mu[i]
            self.mu[i] += d / self.n
            self.m2[i] += d * (f[k] - self.mu[i])
        x = self._x(f)
        y = 1.0 if actual == UP else 0.0
        z = sum(w * xi for w, xi in zip(self.w, x)) + self.b
        err = 1 / (1 + math.exp(-max(-30, min(30, z)))) - y
        for i in range(len(self.w)):
            self.w[i] -= self.lr * (err * x[i] + self.l2 * self.w[i])
        self.b -= self.lr * err

    def state(self): return {"w": self.w, "b": self.b, "mu": self.mu, "m2": self.m2, "n": self.n}
    def load_state(self, s):
        if s:
            self.w, self.b, self.mu, self.m2, self.n = s["w"], s["b"], s["mu"], s["m2"], s["n"]


def build_model(model_id: str, version: str, params: dict, state: dict | None, role: str, settings) -> Model:
    if model_id == "ema_slope":
        m = EmaSlope()
    elif model_id == "mean_rev":
        m = MeanReversion()
    elif model_id == "logistic":
        m = OnlineLogistic(version, params.get("lr", 0.05), params.get("l2", 1e-3), params.get("warmup", 50),
                           params.get("up_thr", 0.55), role)
    else:
        raise ValueError(f"unknown model {model_id}")
    m.version, m.role = version, role
    m.load_state(state or {})
    return m


# ---- registry (versioned, never a single mutable file) --------------------------------------------
STATUSES = ("CHAMPION", "STABLE", "CHALLENGER", "VALIDATED", "REJECTED", "RETIRED", "BASELINE")


class ModelRegistry:
    DEFAULT_MEMBERS = {"ema_slope": {}, "mean_rev": {}, "logistic": {"lr": 0.05, "l2": 1e-3, "warmup": 50, "up_thr": 0.55}}

    def __init__(self, db, settings, events=None):
        self.db, self.s, self.events = db, settings, events

    def ensure_defaults(self) -> None:
        for mid, params in self.DEFAULT_MEMBERS.items():
            if not self.db.query("SELECT 1 FROM model_versions WHERE model_id=?", (mid,)):
                self.register(mid, "1.0", "CHAMPION", params, notes="initial champion")
        for mid in ("naive_majority", "persistence", "stat_drift"):
            if not self.db.query("SELECT 1 FROM model_versions WHERE model_id=?", (mid,)):
                self.register(mid, "1.0", "BASELINE", {}, notes="baseline challenger")

    def register(self, model_id, version, status, params, train_ref="", eval_ref="", state=None, notes="") -> None:
        assert status in STATUSES
        self.db.insert("model_versions", {"model_id": model_id, "version": version, "created_ts": datetime.now(timezone.utc).isoformat(),
                                          "feature_version": self.s.feature_version, "train_dataset_ref": train_ref,
                                          "eval_dataset_ref": eval_ref, "status": status,
                                          "params_json": json.dumps(params, sort_keys=True),
                                          "state_json": json.dumps(state or {}), "metrics_json": "{}", "notes": notes})

    def rows(self, status: str | None = None, model_id: str | None = None) -> list[dict]:
        sql, par = "SELECT * FROM model_versions WHERE 1=1", []
        if status:
            sql += " AND status=?"; par.append(status)
        if model_id:
            sql += " AND model_id=?"; par.append(model_id)
        return self.db.query(sql + " ORDER BY id", par)

    def save_state(self, model_id, version, state) -> None:
        self.db.execute("UPDATE model_versions SET state_json=? WHERE model_id=? AND version=?",
                        (json.dumps(state), model_id, version))

    def set_status(self, model_id, version, status, notes="", metrics=None) -> None:
        assert status in STATUSES
        self.db.execute("UPDATE model_versions SET status=?, notes=COALESCE(notes,'')||?, metrics_json=COALESCE(?,metrics_json) "
                        "WHERE model_id=? AND version=?",
                        (status, f"\n[{datetime.now(timezone.utc).isoformat()}] {status}: {notes}",
                         json.dumps(metrics) if metrics is not None else None, model_id, version))
        if self.events:
            self.events.emit("registry", "MODEL_STATUS", {"model": model_id, "version": version, "status": status, "notes": notes})

    def promote(self, model_id: str, version: str, approver: str, evidence: str) -> None:
        r = self.rows(model_id=model_id)
        tgt = next((x for x in r if x["version"] == version), None)
        if tgt is None or tgt["status"] != "VALIDATED":
            raise ValueError("only a VALIDATED challenger can be promoted")
        if not approver.strip() or not evidence.strip():
            raise ValueError("approver and evidence are required")
        for x in r:
            if x["status"] == "STABLE":
                self.set_status(model_id, x["version"], "RETIRED", "superseded")
            if x["status"] == "CHAMPION":
                self.set_status(model_id, x["version"], "STABLE", f"kept as rollback target; replaced by {version}")
        self.set_status(model_id, version, "CHAMPION", f"promoted by {approver}: {evidence}")

    def rollback(self, model_id: str, reason: str) -> str:
        r = self.rows(model_id=model_id)
        st = [x for x in r if x["status"] == "STABLE"]
        if not st:
            raise ValueError("no stable version to roll back to")
        for x in r:
            if x["status"] == "CHAMPION":
                self.set_status(model_id, x["version"], "REJECTED", f"rolled back: {reason}")
        self.set_status(model_id, st[-1]["version"], "CHAMPION", f"rollback: {reason}")
        return st[-1]["version"]


# ---- results ------------------------------------------------------------------------------------
@dataclass
class ForecastResult:
    direction: str = "UNKNOWN"
    horizon_bars: int = 0
    timeframe: str = ""
    uncertainty_label: str = "UNKNOWN"
    uncertainty_score: float | None = None
    uncertainty_components: dict = field(default_factory=dict)
    model_agreement: float | None = None
    probability_up: float | None = None
    probability_source: str = "none (no calibrated probability available)"
    calibration: dict = field(default_factory=dict)
    models: list = field(default_factory=list)
    baselines: list = field(default_factory=list)
    challengers: list = field(default_factory=list)
    timestamp: float = 0.0
    feature_version: str = FORECAST_FEATURE_VERSION
    snapshot_id: str = ""
    regime_context: str = "UNKNOWN"
    note: str = ""

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["uncertainty_score"] = None if self.uncertainty_score is None else round(self.uncertainty_score, 3)
        d["model_agreement"] = None if self.model_agreement is None else round(self.model_agreement, 3)
        d["models"] = [m if isinstance(m, dict) else m.to_dict() for m in self.models]
        d["baselines"] = [m if isinstance(m, dict) else m.to_dict() for m in self.baselines]
        d["challengers"] = [m if isinstance(m, dict) else m.to_dict() for m in self.challengers]
        return d


class ForecastSystem:
    def __init__(self, db, settings, registry: ModelRegistry, events=None, on_resolved=None):
        self.db, self.s, self.registry, self.events, self.on_resolved = db, settings, registry, events, on_resolved
        self._last_bar = None
        self._calib_cache: tuple[int, dict] | None = None
        self.reload()

    # ---- model set -----------------------------------------------------------------------------
    def reload(self) -> None:
        self.registry.ensure_defaults()
        self.members, self.challengers, self.baselines = [], [], []
        thr = self.s.forecast_flat_threshold_atr
        for r in self.registry.rows():
            st = json.loads(r["state_json"] or "{}")
            params = json.loads(r["params_json"] or "{}")
            if r["status"] == "CHAMPION" and r["model_id"] in ("ema_slope", "mean_rev", "logistic"):
                self.members.append(build_model(r["model_id"], r["version"], params, st, "MEMBER", self.s))
            elif r["status"] == "CHALLENGER" and r["model_id"] == "logistic":
                self.challengers.append(build_model(r["model_id"], r["version"], params, st, "CHALLENGER", self.s))
            elif r["status"] == "BASELINE":
                m = {"naive_majority": NaiveMajority(), "persistence": Persistence(thr), "stat_drift": StatBaseline()}[r["model_id"]]
                m.version = r["version"]
                m.load_state(st)
                self.baselines.append(m)
        self._tracked = {(m.model_id, m.version): m for m in self.members + self.challengers + self.baselines}

    # ---- main entry -----------------------------------------------------------------------------
    def run(self, snap: MarketSnapshot, regime=None) -> ForecastResult:
        tf = self.s.forecast_timeframe
        c = snap.closed(tf)
        f = forecast_features(c)
        regime_name = regime.regime if regime else "UNKNOWN"
        if f is None:
            return ForecastResult(timeframe=tf, horizon_bars=self.s.forecast_horizon_bars, timestamp=snap.as_of,
                                  snapshot_id=snap.snapshot_id, regime_context=regime_name, note="insufficient history")
        self.resolve(snap)
        mem = [m.predict(f) for m in self.members]
        base = [m.predict(f) for m in self.baselines]
        chal = [m.predict(f) for m in self.challengers]
        cal = self.calibration_info()
        res = self._combine(mem, base, chal, regime, cal, snap, tf)
        bar = c[-1].time
        if bar != self._last_bar:
            self._log(snap, f, res, mem, base, chal, bar, tf, regime_name)
            self._last_bar = bar
        return res

    def _combine(self, mem, base, chal, regime, cal, snap, tf) -> ForecastResult:
        valid = [m for m in mem if m.valid]
        r = ForecastResult(horizon_bars=self.s.forecast_horizon_bars, timeframe=tf, timestamp=snap.as_of,
                           snapshot_id=snap.snapshot_id, calibration=cal, models=mem, baselines=base, challengers=chal,
                           regime_context=regime.regime if regime else "UNKNOWN")
        if len(valid) < 2:
            r.note = "fewer than 2 valid models"
            return r
        votes = Counter(m.direction for m in valid)
        top = votes.most_common()
        direction = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else FLAT
        r.direction = direction
        r.model_agreement = votes[direction] / len(valid)
        scores = [m.score for m in valid]
        mu = sum(scores) / len(scores)
        disp = math.sqrt(sum((x - mu) ** 2 for x in scores) / len(scores))
        reg_term = 0.15 if (regime and regime.regime in ("TRANSITION", "UNCERTAIN", "HIGH_VOLATILITY")) else 0.0
        cal_term = {"POOR": 0.15, "INSUFFICIENT_DATA": 0.15}.get(cal.get("status"), 0.0)
        comps = {"disagreement": round((1 - r.model_agreement) * 0.5, 3), "score_dispersion": round(min(1.0, disp) * 0.2, 3),
                 "regime": reg_term, "calibration": cal_term}
        r.uncertainty_components = comps
        r.uncertainty_score = min(1.0, sum(comps.values()))
        r.uncertainty_label = "LOW" if r.uncertainty_score < 0.25 else "MEDIUM" if r.uncertainty_score < 0.5 else "HIGH"
        # probability ONLY from a calibrated model output
        lg = next((m for m in valid if m.model_id == "logistic" and m.p_up_raw is not None), None)
        if lg is not None:
            cp = self._calibrated_p(lg.p_up_raw)
            if cp is not None:
                r.probability_up, r.probability_source = cp, "logistic, calibrated on resolved forecasts"
        return r

    # ---- logging & resolution -------------------------------------------------------------------
    def _log(self, snap, f, res, mem, base, chal, bar, tf, regime) -> None:
        feats = json.dumps({k: v for k, v in f.items() if not k.startswith("_")} | {"_atr": f["_atr"]}, sort_keys=True)
        rows = [("ensemble", "1.0", "ENSEMBLE", res.direction, 0.0, None)]
        rows += [(m.model_id, m.version, "MEMBER", m.direction, m.score, m.p_up_raw) for m in mem]
        rows += [(m.model_id, m.version, "BASELINE", m.direction, m.score, None) for m in base]
        rows += [(m.model_id, m.version, "CHALLENGER", m.direction, m.score, m.p_up_raw) for m in chal]
        for mid, ver, role, d, sc, p in rows:
            if d == "UNKNOWN":
                continue
            self.db.insert("forecast_log", {"ts": iso(snap.as_of), "snapshot_id": snap.snapshot_id, "model_id": mid, "model_version": ver,
                                            "role": role, "direction": d, "score": sc, "p_up_raw": p,
                                            "horizon_bars": self.s.forecast_horizon_bars, "timeframe": tf, "regime": regime,
                                            "entry_price": f["_close"], "entry_time": bar, "atr": f["_atr"], "features_json": feats})
        if self.events:
            self.events.emit("forecast", "FORECAST_GENERATED", {"direction": res.direction, "uncertainty": res.uncertainty_label,
                             "agreement": res.model_agreement, "horizon": self.s.forecast_horizon_bars, "timeframe": tf,
                             "models": {m.model_id: m.direction for m in mem}}, snap.snapshot_id, snap.as_of)

    def resolve(self, snap: MarketSnapshot) -> int:
        """Resolve pending forecasts whose target bar has CLOSED (never earlier, never using later data)."""
        tf = self.s.forecast_timeframe
        sec, h = TF_SECONDS[tf], self.s.forecast_horizon_bars
        bars = {c.time: c for c in snap.closed(tf)}
        pend = self.db.query("SELECT * FROM forecast_log WHERE actual IS NULL ORDER BY id LIMIT 800")
        if not pend:
            return 0
        done_snaps: dict[str, dict] = {}
        n = 0
        for row in pend:
            tgt_open = row["entry_time"] + h * sec
            if tgt_open + sec > snap.as_of:
                continue                                         # target not closed yet: unknowable now
            tb = bars.get(tgt_open)
            if tb is None:
                if tgt_open + 3 * h * sec < snap.as_of:          # market gap / out of window: cannot be scored fairly
                    self.db.execute("UPDATE forecast_log SET actual='VOID', resolved_ts=? WHERE id=?", (iso(snap.as_of), row["id"]))
                continue
            move = tb.close - row["entry_price"]
            thr = self.s.forecast_flat_threshold_atr * row["atr"]
            actual = UP if move > thr else DOWN if move < -thr else FLAT
            self.db.execute("UPDATE forecast_log SET actual=?, actual_return=?, correct=?, resolved_ts=? WHERE id=?",
                            (actual, move / row["atr"], int(row["direction"] == actual), iso(snap.as_of), row["id"]))
            n += 1
            m = self._tracked.get((row["model_id"], row["model_version"]))
            # The active CHAMPION is immutable during normal operation.
            # Resolved observations may train a CHALLENGER/BASELINE in memory,
            # but a champion can only change through the explicit
            # candidate -> OOS -> shadow -> validated -> human promotion flow.
            if m is not None and m.role in ("CHALLENGER", "BASELINE") and row["features_json"]:
                m.learn(json.loads(row["features_json"]) | {"_": 0}, actual)
            done_snaps.setdefault(row["snapshot_id"], {})[row["model_id"] + ":" + row["role"]] = (row["direction"], actual, move / row["atr"])
        if n:
            for (mid, ver), m in self._tracked.items():
                if m.role in ("MEMBER", "CHALLENGER", "BASELINE") and m.state():
                    self.registry.save_state(mid, ver, m.state())
            for sid, d in done_snaps.items():
                ens = d.get("ensemble:ENSEMBLE")
                if ens:
                    err = 0.0 if ens[0] == ens[1] else 1.0
                    if self.on_resolved:
                        self.on_resolved(sid, {"forecast": ens[0], "actual": ens[1], "return_atr": round(ens[2], 3)}, err, snap.as_of)
            if self.events:
                self.events.emit("forecast", "FORECAST_RESOLVED", {"resolved_rows": n, "snapshots": len(done_snaps)}, None, snap.as_of)
        return n

    # ---- calibration & scoreboard -------------------------------------------------------------------
    def _bins(self) -> dict:
        rows = self.db.query("SELECT p_up_raw p, actual a FROM forecast_log WHERE role='MEMBER' AND model_id='logistic' "
                             "AND p_up_raw IS NOT NULL AND actual IN ('UP','DOWN')")
        if self._calib_cache and self._calib_cache[0] == len(rows):
            return self._calib_cache[1]
        bins: dict[int, list] = {}
        for r in rows:
            bins.setdefault(min(9, int(r["p"] * 10)), []).append(1 if r["a"] == UP else 0)
        out = {"n": len(rows), "bins": {k: (sum(v) / len(v), len(v)) for k, v in bins.items()}, "rows": rows}
        self._calib_cache = (len(rows), out)
        return out

    def _calibrated_p(self, p: float) -> float | None:
        b = self._bins()
        if b["n"] < self.s.forecast_min_samples:
            return None
        k = b["bins"].get(min(9, int(p * 10)))
        return round(k[0], 3) if k and k[1] >= 10 else None

    def calibration_info(self) -> dict:
        b = self._bins()
        ens = self.db.query("SELECT COUNT(*) n, SUM(correct) c FROM forecast_log WHERE role='ENSEMBLE' AND actual IN ('UP','DOWN','FLAT')")[0]
        n = ens["n"] or 0
        info = {"resolved_ensemble_forecasts": n, "ensemble_hit_rate": round(ens["c"] / n, 4) if n else None,
                "logistic_resolved": b["n"], "min_samples": self.s.forecast_min_samples}
        if b["n"] < self.s.forecast_min_samples:
            info.update(status="INSUFFICIENT_DATA", ece=None, brier=None)
            return info
        rows = b["rows"]
        brier = sum((r["p"] - (1 if r["a"] == UP else 0)) ** 2 for r in rows) / len(rows)
        ece = sum(cnt / len(rows) * abs(sum(r["p"] for r in rows if min(9, int(r["p"] * 10)) == k) / cnt - freq)
                  for k, (freq, cnt) in b["bins"].items())
        info.update(brier=round(brier, 4), ece=round(ece, 4), status="POOR" if ece > 0.12 else "OK")
        return info

    def scoreboard(self) -> list[dict]:
        rows = self.db.query(
            "SELECT model_id, model_version, role, COUNT(*) n, SUM(correct) c FROM forecast_log "
            "WHERE actual IN ('UP','DOWN','FLAT') GROUP BY model_id, model_version, role ORDER BY role, model_id")
        out = []
        for r in rows:
            e = {"model": r["model_id"], "version": r["model_version"], "role": r["role"], "n": r["n"],
                 "accuracy": round(r["c"] / r["n"], 4) if r["n"] else None, "vs_baselines": {}, "verdict": "n/a"}
            if r["role"] in ("ENSEMBLE", "MEMBER", "CHALLENGER"):
                verdicts = []
                for b in self.baselines:
                    q = self.db.query(
                        "SELECT SUM(m.correct=1 AND b.correct=0) b_only, SUM(m.correct=0 AND b.correct=1) m_only, COUNT(*) n "
                        "FROM forecast_log m JOIN forecast_log b ON m.snapshot_id=b.snapshot_id AND b.model_id=? AND b.role='BASELINE' "
                        "WHERE m.model_id=? AND m.model_version=? AND m.role=? AND m.actual IN ('UP','DOWN','FLAT') AND b.actual IN ('UP','DOWN','FLAT')",
                        (b.model_id, r["model_id"], r["model_version"], r["role"]))[0]
                    x, y, nn = q["b_only"] or 0, q["m_only"] or 0, q["n"] or 0
                    z = (x - y) / math.sqrt(x + y) if x + y > 0 else 0.0
                    v = "INSUFFICIENT_DATA" if nn < self.s.forecast_min_samples else ("BETTER" if z >= 1.96 else "WORSE" if z <= -1.96 else "NO_DIFFERENCE")
                    e["vs_baselines"][b.model_id] = {"paired_n": nn, "model_only_correct": x, "baseline_only_correct": y, "z": round(z, 2), "verdict": v}
                    verdicts.append(v)
                e["verdict"] = ("INSUFFICIENT_DATA" if "INSUFFICIENT_DATA" in verdicts or not verdicts else
                                "BEATS_ALL_BASELINES" if all(v == "BETTER" for v in verdicts) else "DOES_NOT_BEAT_ALL_BASELINES")
            out.append(e)
        return out
