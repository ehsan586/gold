"""Command line:  python -m goldbot <command>   (see docs/GUIDE_FA.md)"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import secrets
import sys
import threading
import time
from pathlib import Path


def load_env_file(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.handlers.RotatingFileHandler("logs/goldbot.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    sh = logging.StreamHandler()
    for h in (fh, sh):
        h.setFormatter(fmt)
        root.addHandler(h)


def _db(settings):
    from .db import Database
    db = Database(settings.db_path)
    db.init_schema()
    return db


def cmd_initdb(a, s):
    db = _db(s)
    print("schema version", db.schema_version(), "tables:", ", ".join(db.tables()))


def cmd_mt4check(a, s):
    """Read-only MT4 FILE_COMMON feed self-test. No order API exists in this adapter."""
    from .broker_mt4 import Mt4FeedBroker
    b = Mt4FeedBroker(s.mt4_feed_path or None, s.symbol, s.max_tick_age_sec)
    try:
        b.connect()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        b.disconnect()
        return 1
    acc = b.account()
    sym = b.find_symbol(s.symbol)
    spec = b.symbol_spec(sym)
    t = b.tick(sym)
    print(f"account {acc.login} demo={acc.is_demo} leverage=1:{acc.leverage} balance={acc.balance} {acc.currency}")
    print("symbol:", sym)
    print(json.dumps(spec.__dict__, indent=2))
    if not acc.is_demo:
        print("ERROR: MT4 account is NOT DEMO; final build refuses non-demo accounts.")
        b.disconnect()
        return 1
    print("tick:", t, "spread points:", round((t.ask - t.bid) / spec.point, 1))
    counts = {}
    for tf in ("M1", "M5", "M15", "H1", "H4"):
        counts[tf] = len(b.candles(sym, tf, 300))
        print(tf, "candles:", counts[tf])
    missing = [tf for tf, n in counts.items() if n < 120]
    if missing:
        print("ERROR: insufficient MT4 history for", ", ".join(missing), "(need at least 120 closed bars each).")
        b.disconnect()
        return 1
    print("OK - MT4 read-only feed works; execution remains SIMULATION.")
    b.disconnect()
    return 0


def cmd_mt5check(a, s):
    """Connectivity self-test for YOUR terminal (this part could not be tested by the developer)."""
    from .broker_mt5 import Mt5Broker
    b = Mt5Broker(os.environ.get("GOLDBOT_MT5_LOGIN"), os.environ.get("GOLDBOT_MT5_PASSWORD"),
                  os.environ.get("GOLDBOT_MT5_SERVER"), os.environ.get("GOLDBOT_MT5_PATH"))
    try:
        b.connect()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        b.disconnect()
        return 1
    acc = b.account()
    print(f"account {acc.login}  demo={acc.is_demo}  mode={acc.margin_mode}  leverage=1:{acc.leverage}  equity={acc.equity} {acc.currency}")
    if not acc.is_demo:
        print("!! NOT A DEMO ACCOUNT - the bot will refuse to start on it.")
    sym = b.find_symbol(s.symbol)
    spec = b.symbol_spec(sym)
    print("symbol:", sym)
    print(json.dumps(spec.__dict__, indent=2))
    t = b.tick(sym)
    print("tick:", t, " spread points:", round((t.ask - t.bid) / spec.point, 1))
    for tf in ("M1", "M5", "M15", "H1", "H4"):
        print(tf, "candles:", len(b.candles(sym, tf, 300)))
    print("OK - MT5 integration basics work.")
    b.disconnect()
    return 0


def _start_engine_and_api(engine, db, s, stop, sim_thread=None):
    from .api import ApiServer
    api = ApiServer(engine, db, s, engine.learning, shutdown_callback=(stop.set if stop is not None else None))
    api.serve_background()
    print(f"\nDashboard: http://{s.api_host}:{s.api_port}/   (enter your API token)\nCtrl+C to stop.\n")
    return api


def cmd_run(a, s):
    """Start ASTRA. Default path is read-only MT4 market data + virtual execution."""
    from .broker_sim import ShadowBroker
    from .engine import TradingEngine
    setup_logging()
    db = _db(s)
    if s.market_data_source == "MT4":
        from .broker_mt4 import Mt4FeedBroker
        data_broker = Mt4FeedBroker(s.mt4_feed_path or None, s.symbol, s.max_tick_age_sec)
    else:
        from .broker_mt5 import Mt5Broker
        data_broker = Mt5Broker(os.environ.get("GOLDBOT_MT5_LOGIN"), os.environ.get("GOLDBOT_MT5_PASSWORD"),
                                os.environ.get("GOLDBOT_MT5_SERVER"), os.environ.get("GOLDBOT_MT5_PATH"))
    if s.execution_mode == "MT5_DEMO":
        raise RuntimeError("MT5_DEMO execution is intentionally unavailable in this final safety-first build; use SIMULATION.")
    broker = ShadowBroker(data_broker, db, s.sim_start_balance, time.time, s.symbol)
    print("EXECUTION MODE: SIMULATION - virtual account only; no real orders are sent.")
    print("MARKET DATA SOURCE:", s.market_data_source)
    eng = TradingEngine(s, db, broker)
    stop = threading.Event()
    api = None
    th = None
    try:
        eng.start()
        threading.Thread(target=eng.run_replay_selftest, daemon=True, name="replay-selftest").start()
        th = threading.Thread(target=eng.run_forever, args=(stop,), daemon=True, name="engine")
        th.start()
        api = _start_engine_and_api(eng, db, s, stop)
        while th.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if api is not None:
            try:
                api.shutdown()
            except Exception:
                pass
            try:
                api.httpd.server_close()
            except Exception:
                pass
        try:
            broker.disconnect()
        finally:
            db.close()


def cmd_demo(a, s):
    """OFFLINE demo: paper broker replaying synthetic (or CSV) candles so you can see the whole system + dashboard."""
    from .broker_paper import PaperBroker
    from .data import load_csv
    from .engine import TradingEngine
    from .synth import generate_m1
    import dataclasses
    if not s.api_token:
        tok = secrets.token_urlsafe(24)
        s = dataclasses.replace(s, api_token=tok)
        print("Generated a temporary API token for this demo:", tok)
    m1 = load_csv(a.csv) if a.csv else generate_m1(40000)
    print("DEMO uses", "your CSV" if a.csv else "SYNTHETIC random data (meaningless for performance)")
    from .db import Database
    db = Database(":memory:" if not a.keep else s.db_path)
    db.init_schema()
    b = PaperBroker(m1, s.symbol)
    for _ in range(3700):
        b.advance()
    eng = TradingEngine(s, db, b, clock=b.now, sim=True)
    eng.start()
    threading.Thread(target=eng.run_replay_selftest, daemon=True, name="replay-selftest").start()
    api = _start_engine_and_api(eng, db, s, None)
    try:
        while True:
            if not b.advance():
                print("end of data"); break
            eng.step(b.now())
            time.sleep(a.speed)
    except KeyboardInterrupt:
        pass
    api.shutdown()


def cmd_backtest(a, s):
    from .backtest import run_backtest, sweep_retrace, walk_forward
    from .data import load_csv
    from .synth import generate_m1
    m1 = load_csv(a.csv) if a.csv else generate_m1(a.synthetic or 20000)
    if not a.csv:
        print("!! SYNTHETIC DATA: this only proves the machinery works. It says NOTHING about real gold.\n", file=sys.stderr)
    kw = dict(balance=a.balance, spread_points=a.spread, slippage_points=a.slippage, commission_per_lot=a.commission)
    prog = lambda i, n: print(f"  {i}/{n} bars", file=sys.stderr)  # noqa: E731
    if a.sweep:
        out = sweep_retrace(m1, s, **kw)
    elif a.split:
        out = walk_forward(m1, s, split=a.split, **kw)
    else:
        out = run_backtest(m1, s, progress=prog, **kw)
    print(json.dumps(out, indent=2, default=str))


def _m1(a):
    from .data import load_csv
    if not a.csv:
        sys.exit("--csv FILE is required (M1 candles exported from MT4/MT5)")
    return load_csv(a.csv)


def cmd_replay(a, s):
    from .replay import ReplayEngine
    r = ReplayEngine(s)
    m1 = _m1(a)
    out = r.verify_determinism(m1, max_steps=a.max_steps or 120) if a.verify else r.replay(m1, max_steps=a.max_steps).to_dict()
    print(json.dumps(out, indent=2))


def cmd_reconstruct(a, s):
    from .replay import ReplayEngine
    m1 = _m1(a)
    out = ReplayEngine(s).reconstruct(m1, a.index if a.index is not None else len(m1) - 1)
    def clean(o):
        if hasattr(o, "to_dict"): return o.to_dict()
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        return o
    print(json.dumps(clean(out), indent=2, default=str))


def cmd_ablation(a, s):
    from .experiments import ExperimentLab
    lab = ExperimentLab(_db(s), s)
    out = lab.run_ablation(_m1(a), s, variants=a.variants.split(",") if a.variants else None, split=a.split, force=a.force,
                           balance=a.balance, spread_points=a.spread, slippage_points=a.slippage)
    for name, r in out["results"].items():
        expr = "n/a" if r.get("expectancy_R") is None else f"{r['expectancy_R']:7.3f}"
        print(f"{name:22s} trades={r['trades']:3d} net={r['net_profit']:9.2f} expR={expr:>7s} maxDD={r['max_drawdown_pct']:5.2f}%  "
              f"{r.get('delta_vs_full', {}).get('verdict', '')} [{r.get('evaluation_status', '—')}]")
    print("experiment:", out["experiment_id"])


def cmd_regimes(a, s):
    from .data import TF_SECONDS, resample
    from .regime import evaluate_regimes
    print(json.dumps(evaluate_regimes(resample(_m1(a), TF_SECONDS[s.regime_timeframe])), indent=2))


def _lm(s):
    from .engine import TradingEngine
    from .broker_paper import PaperBroker
    from .synth import generate_m1
    from .learning_manager import LearningManager
    from .events import EventLog
    from .forecast import ForecastSystem, ModelRegistry
    from .drift import DriftDetector
    from .learning import LearningEngine
    from .experiments import ExperimentLab
    db = _db(s)
    ev = EventLog(db, time.time)
    reg = ModelRegistry(db, s, ev)
    fs = ForecastSystem(db, s, reg, ev)
    return LearningManager(db, s, ev, reg, fs, DriftDetector(s, ev), LearningEngine(db, s), ExperimentLab(db, s, ev)), db


def cmd_challenger(a, s):
    lm, _ = _lm(s)
    print(json.dumps(lm.propose_challenger(_m1(a), force=a.force), indent=2, default=str))


def cmd_models(a, s):
    lm, _ = _lm(s)
    print(json.dumps(lm.status(), indent=2, default=str))


def cmd_promote(a, s):
    lm, _ = _lm(s)
    lm.promote(a.model_id, a.version, a.approver, a.evidence)
    print("promoted; previous champion kept as STABLE (rollback possible)")


def cmd_rollback_model(a, s):
    lm, _ = _lm(s)
    print("champion now:", lm.rollback(a.model_id, a.reason))


def cmd_experiments(a, s):
    from .experiments import ExperimentLab
    for e in ExperimentLab(_db(s), s).list():
        print(e["experiment_id"], e["status"], e["created_ts"][:19], "|", e["hypothesis"], "|", (e["outcome"] or "")[:80])


def cmd_selftest(a, s):
    from .replay import selftest
    r = selftest(s)
    print(json.dumps(r, indent=2))
    sys.exit(0 if r.get("ok") else 1)


def cmd_versions(a, s):
    from .learning import LearningEngine
    L = LearningEngine(_db(s), s)
    for v in L.list_versions():
        print(v["version"], v["status"], "parent=", v["parent_version"])
    print("active:", L.active_version())


def cmd_propose(a, s):
    from .learning import LearningEngine
    print(json.dumps(LearningEngine(_db(s), s).propose_weight_changes() or "no change justified (not enough data)", indent=2))


def cmd_advance(a, s):
    from .learning import LearningEngine
    LearningEngine(_db(s), s).advance(a.version, a.to, a.evidence, a.approver or "")
    print("ok")


def cmd_rollback(a, s):
    from .learning import LearningEngine
    print("active now:", LearningEngine(_db(s), s).rollback(a.reason))


def cmd_shutdown(a, s):
    """Request shutdown of a locally running ASTRA instance."""
    import urllib.request, urllib.error
    if not s.api_token:
        print("No API token configured; nothing to stop.")
        return
    body = json.dumps({"request_id": "shutdown-" + secrets.token_hex(8)}).encode()
    req = urllib.request.Request(f"http://{s.api_host}:{s.api_port}/api/v1/control/shutdown", data=body, method="POST",
                                 headers={"Authorization": "Bearer " + s.api_token, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            print(r.read().decode())
    except Exception as e:
        print("ASTRA shutdown request failed:", e)


def main(argv=None) -> int:
    load_env_file()
    from .settings import load_settings
    ap = argparse.ArgumentParser(prog="goldbot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("initdb").set_defaults(f=cmd_initdb)
    sub.add_parser("mt4check").set_defaults(f=cmd_mt4check)
    sub.add_parser("mt5check").set_defaults(f=cmd_mt5check)
    sub.add_parser("run").set_defaults(f=cmd_run)
    sub.add_parser("shutdown").set_defaults(f=cmd_shutdown)
    d = sub.add_parser("demo"); d.add_argument("--csv"); d.add_argument("--speed", type=float, default=0.05); d.add_argument("--keep", action="store_true"); d.set_defaults(f=cmd_demo)
    b = sub.add_parser("backtest")
    b.add_argument("--csv"); b.add_argument("--synthetic", type=int); b.add_argument("--split", type=float)
    b.add_argument("--sweep", action="store_true"); b.add_argument("--balance", type=float, default=800.0)
    b.add_argument("--spread", type=int, default=25); b.add_argument("--slippage", type=int, default=2)
    b.add_argument("--commission", type=float, default=0.0); b.set_defaults(f=cmd_backtest)
    sub.add_parser("selftest").set_defaults(f=cmd_selftest)
    sub.add_parser("models").set_defaults(f=cmd_models)
    sub.add_parser("experiments").set_defaults(f=cmd_experiments)
    rp = sub.add_parser("replay"); rp.add_argument("--csv"); rp.add_argument("--verify", action="store_true"); rp.add_argument("--max-steps", type=int); rp.set_defaults(f=cmd_replay)
    rc = sub.add_parser("reconstruct"); rc.add_argument("--csv"); rc.add_argument("--index", type=int); rc.set_defaults(f=cmd_reconstruct)
    ab = sub.add_parser("ablation"); ab.add_argument("--csv"); ab.add_argument("--variants"); ab.add_argument("--split", type=float, default=0.7)
    ab.add_argument("--force", action="store_true"); ab.add_argument("--balance", type=float, default=800.0); ab.add_argument("--spread", type=int, default=25)
    ab.add_argument("--slippage", type=int, default=2); ab.set_defaults(f=cmd_ablation)
    rg = sub.add_parser("regimes"); rg.add_argument("--csv"); rg.set_defaults(f=cmd_regimes)
    ch = sub.add_parser("challenger"); ch.add_argument("--csv"); ch.add_argument("--force", action="store_true"); ch.set_defaults(f=cmd_challenger)
    pm = sub.add_parser("promote"); pm.add_argument("model_id"); pm.add_argument("version"); pm.add_argument("--approver", required=True); pm.add_argument("--evidence", required=True); pm.set_defaults(f=cmd_promote)
    rb = sub.add_parser("rollback-model"); rb.add_argument("model_id"); rb.add_argument("--reason", required=True); rb.set_defaults(f=cmd_rollback_model)
    sub.add_parser("versions").set_defaults(f=cmd_versions)
    sub.add_parser("propose").set_defaults(f=cmd_propose)
    v = sub.add_parser("advance"); v.add_argument("version"); v.add_argument("to"); v.add_argument("--evidence", required=True); v.add_argument("--approver"); v.set_defaults(f=cmd_advance)
    r = sub.add_parser("rollback"); r.add_argument("--reason", required=True); r.set_defaults(f=cmd_rollback)
    a = ap.parse_args(argv)
    try:
        s = load_settings()
    except Exception as e:  # noqa: BLE001
        print("CONFIG ERROR:", e); return 2
    rc = a.f(a, s)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
