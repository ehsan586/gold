"""Backend API + dashboard host (standard library only).

Security: bearer tokens (admin = control, viewer = read-only), constant-time comparison,
failed-auth rate limit, CORS allow-list, strict JSON validation, body size limit, socket
timeouts, security headers, idempotent control commands (request_id). Binds to 127.0.0.1
by default; expose it only behind HTTPS (see docs).
"""
from __future__ import annotations

import hmac
import json
import math
import re
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

MAX_BODY = 16 * 1024
RID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
LEVELS = {"DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"}
DASH = Path(__file__).with_name("dashboard.html")


def _clean(o):
    if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


class ApiServer:
    def __init__(self, engine, db, settings, learning, shutdown_callback=None):
        if len(settings.api_token) < 16:
            raise RuntimeError("GOLDBOT_API_TOKEN must be set (>=16 chars). Generate: python -c \"import secrets;print(secrets.token_urlsafe(32))\"")
        self.engine, self.db, self.s, self.learning = engine, db, settings, learning
        self.shutdown_callback = shutdown_callback
        self.fails: dict[str, deque] = defaultdict(deque)
        self.httpd = ThreadingHTTPServer((settings.api_host, settings.api_port), self._handler())
        self.httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    def serve_background(self) -> threading.Thread:
        t = threading.Thread(target=self.httpd.serve_forever, daemon=True, name="api")
        t.start()
        self._thread = t
        self.engine.api_alive = lambda: bool(self._thread and self._thread.is_alive())
        return t

    def shutdown(self) -> None:
        try:
            self.httpd.shutdown()
        finally:
            try:
                self.httpd.server_close()
            except Exception:
                pass
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ auth
    def role(self, header: str, ip: str) -> str | None:
        q = self.fails[ip]
        now = time.time()
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= 10:
            return "LIMITED"
        tok = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if tok:
            if hmac.compare_digest(tok.encode(), self.s.api_token.encode()):
                return "admin"
            if self.s.viewer_token and hmac.compare_digest(tok.encode(), self.s.viewer_token.encode()):
                return "viewer"
        q.append(now)
        return None

    # ------------------------------------------------------------------ routes
    def get(self, path: str, qs: dict):
        e, db = self.engine, self.db
        lim = max(1, min(500, int(qs.get("limit", ["100"])[0])))
        if path == "/api/v1/snapshot":
            return e.snapshot()
        if path == "/api/v1/trades":
            st = qs.get("status", [""])[0].upper()
            if st not in ("", "OPEN", "CLOSED"):
                raise ValueError("status must be OPEN or CLOSED")
            cols = "id,ticket,status,ts_open,ts_close,direction,lot,entry_price,exit_price,sl,tp,pnl,r_multiple,duration_sec,exit_reason,decision_score,strategy_version,mfe,mae,slippage"
            if st:
                return db.query(f"SELECT {cols} FROM trades WHERE status=? ORDER BY id DESC LIMIT ?", (st, lim))
            return db.query(f"SELECT {cols} FROM trades WHERE status IN ('OPEN','CLOSED') ORDER BY id DESC LIMIT ?", (lim,))
        if path == "/api/v1/logs":
            lv = qs.get("level", [""])[0].upper()
            comp = qs.get("component", [""])[0]
            sql, par = "SELECT ts,level,component,event,message FROM system_events WHERE 1=1", []
            if lv:
                if lv not in LEVELS:
                    raise ValueError("bad level")
                sql += " AND level=?"; par.append(lv)
            if comp:
                sql += " AND component=?"; par.append(comp[:30])
            else:
                sql += " AND NOT (component='state_machine' AND level='INFO')"
            return db.query(sql + " ORDER BY id DESC LIMIT ?", par + [lim])
        if path == "/api/v1/risk_events":
            return db.query("SELECT ts,level,rule,message FROM risk_events ORDER BY id DESC LIMIT ?", (lim,))
        if path == "/api/v1/learning":
            return self.learning.summary()
        if path == "/api/v1/health":
            sn = e.snapshot()
            return {"status": (sn.get("health") or {}).get("status", "UNKNOWN"), "system": sn.get("system"), "assessment": sn.get("assessment"),
                    "components": (sn.get("health") or {}).get("components"), "watchdog": sn.get("watchdog"), "isolation": sn.get("isolation"),
                    "problems": (sn.get("health") or {}).get("problems")}
        if path == "/api/v1/metrics":
            return e.metrics.snapshot()
        if path == "/api/v1/events":
            et = qs.get("type", [""])[0].upper()
            return e.events.tail(lim, et or None)
        if path == "/api/v1/events/verify":
            ok, n = e.events.verify_chain()
            return {"chain_ok": ok, "events": n, "digest": e.events.digest()[:16]}
        if path == "/api/v1/candles":
            tf = qs.get("tf", ["M5"])[0].upper()
            return {"tf": tf, "candles": e.candles(tf, min(lim, 300))}
        if path == "/api/v1/forecasts":
            return {"scoreboard": e.forecaster.scoreboard(), "calibration": e.forecaster.calibration_info(),
                    "current": e._forecast.to_dict()}
        if path == "/api/v1/models":
            return e.learning_mgr.status()
        if path == "/api/v1/drift":
            return {"flags": e.drift.flags, "window": self.s.drift_window}
        if path == "/api/v1/experiences":
            return {"summary": e.memory.summary(), "recent": e.memory.recent(min(lim, 50), qs.get("regime", [None])[0])}
        if path == "/api/v1/experiments":
            return e.lab.list(min(lim, 100))
        if path == "/api/v1/replay":
            return {"selftest": e.replay_status, "how_to": "python -m goldbot replay --csv XAUUSD_M1.csv --verify"}
        if path == "/api/v1/units":
            return (e.snapshot() or {}).get("units")
        if path == "/api/v1/config":
            return self.s.public_dict()
        return None

    def post(self, path: str, body: dict):
        e = self.engine
        rid = body.get("request_id")
        if not isinstance(rid, str) or not RID.match(rid):
            raise ValueError("request_id (8-64 chars, [A-Za-z0-9_-]) is required")
        if not self.db.claim_command("api:" + rid):
            return 409, {"ok": False, "error": "duplicate request_id: already processed"}
        ctl = {"/api/v1/control/pause": "pause", "/api/v1/control/resume": "resume",
               "/api/v1/control/close-position": "close_position", "/api/v1/control/emergency-stop": "emergency_stop",
               "/api/v1/control/emergency-reset": "emergency_reset", "/api/v1/control/reload-strategy": "reload_strategy"}
        if path == "/api/v1/control/shutdown":
            if self.shutdown_callback:
                threading.Thread(target=self.shutdown_callback, daemon=True, name="shutdown-request").start()
            return 200, {"ok": True, "status": "shutdown_requested"}
        if path in ctl:
            params = {}
            if ctl[path] == "emergency_reset":
                params = {"reviewed_by": str(body.get("reviewed_by", ""))[:60], "note": str(body.get("note", ""))[:300],
                          "reset_peak": bool(body.get("reset_peak", False))}
            r = e.submit(ctl[path], params)
            return (200 if r.get("ok") else 400), r
        ml = {"/api/v1/models/promote": "promote_model", "/api/v1/models/reject": "reject_model", "/api/v1/models/rollback": "rollback_model",
              "/api/v1/models/reload": "reload_models", "/api/v1/learning/evaluate": "learning_evaluate"}
        if path in ml:
            params = {k: str(body.get(k, ""))[:300] for k in ("model_id", "version", "approver", "evidence", "reason")}
            r = e.submit(ml[path], params, timeout=60)
            return (200 if r.get("ok") else 400), r
        if path == "/api/v1/replay/selftest":
            return 200, {"ok": True, "result": e.run_replay_selftest()}
        L = self.learning
        if path == "/api/v1/learning/propose":
            r = L.propose_weight_changes()
            return 200, {"ok": True, "proposal": r, "note": "" if r else "no change justified by the data (or not enough samples)"}
        if path == "/api/v1/learning/advance":
            L.advance(str(body.get("version", "")), str(body.get("to", "")), str(body.get("evidence", "")), str(body.get("approver", "")))
            return 200, {"ok": True}
        if path == "/api/v1/learning/rollback":
            return 200, {"ok": True, "active": L.rollback(str(body.get("reason", "manual rollback")))}
        return 404, {"ok": False, "error": "not found"}

    # ------------------------------------------------------------------ http plumbing
    def _handler(self):
        api = self

        class H(BaseHTTPRequestHandler):
            timeout = 10
            server_version = "goldbot"
            sys_version = ""

            def log_message(self, *a):  # keep stdout clean
                pass

            def _send(self, code, payload, ctype="application/json"):
                data = payload if isinstance(payload, bytes) else json.dumps(_clean(payload)).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype + "; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Referrer-Policy", "no-referrer")
                if ctype == "text/html":
                    self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
                origin = self.headers.get("Origin")
                if origin and origin in api.s.cors_origins:
                    self.send_header("Access-Control-Allow-Origin", origin)
                    self.send_header("Vary", "Origin")
                self.end_headers()
                self.wfile.write(data)

            def do_OPTIONS(self):
                origin = self.headers.get("Origin")
                if origin and origin in api.s.cors_origins:
                    self.send_response(204)
                    self.send_header("Access-Control-Allow-Origin", origin)
                    self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                    self.send_header("Vary", "Origin")
                    self.end_headers()
                else:
                    self._send(403, {"error": "origin not allowed"})

            def _auth(self, need_admin: bool):
                role = api.role(self.headers.get("Authorization", ""), self.client_address[0])
                if role == "LIMITED":
                    self._send(429, {"error": "too many failed attempts"}); return False
                if role is None:
                    self._send(401, {"error": "unauthorized"}); return False
                if need_admin and role != "admin":
                    self._send(403, {"error": "read-only token cannot control the bot"}); return False
                return True

            def do_GET(self):
                u = urlparse(self.path)
                if u.path == "/":
                    return self._send(200, DASH.read_bytes(), "text/html")
                if u.path == "/healthz":
                    return self._send(200, {"status": "ok"})
                if not self._auth(False):
                    return
                try:
                    r = api.get(u.path, parse_qs(u.query))
                except (ValueError, KeyError) as ex:
                    return self._send(400, {"error": str(ex)})
                except Exception:  # noqa: BLE001
                    return self._send(500, {"error": "internal error"})
                self._send(404 if r is None else 200, {"error": "not found"} if r is None else r)

            def do_POST(self):
                u = urlparse(self.path)
                if not self._auth(True):
                    return
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                    if n > MAX_BODY or n < 0:
                        return self._send(413, {"error": "body too large"})
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("JSON object expected")
                    code, r = api.post(u.path, body)
                    self._send(code, r)
                except (ValueError, json.JSONDecodeError) as ex:
                    self._send(400, {"error": str(ex)})
                except Exception:  # noqa: BLE001
                    self._send(500, {"error": "internal error"})

        return H
