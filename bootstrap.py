"""Windows one-click launcher for ASTRA MT4 read-only / simulation build.

Keeps all process/argument handling in Python so CMD/PowerShell quoting cannot
accidentally start an interactive Python REPL.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"
TOKEN_FILE = ROOT / "ASTRA_API_TOKEN.txt"
VENV = ROOT / ".venv"
VENV_PY = VENV / "Scripts" / "python.exe"


def out(msg: str = "") -> None:
    print(msg, flush=True)


def fail(msg: str) -> int:
    out(f"[ERROR] {msg}")
    return 1


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        check=False,
        text=True,
        capture_output=capture,
    )


def choose_system_python() -> str | None:
    """Prefer py.exe, then python.exe. No shell invocation is used."""
    for exe in ("py.exe", "python.exe"):
        found = shutil.which(exe)
        if found:
            return found
    return None


def py_version(exe: str, args: list[str]) -> tuple[int, int, int] | None:
    cp = run([exe, *args, "--version"], capture=True)
    text = (cp.stdout or cp.stderr).strip()
    if cp.returncode != 0:
        return None
    parts = text.replace("Python ", "").split(".")
    try:
        return int(parts[0]), int(parts[1]), int(parts[2].split()[0])
    except (ValueError, IndexError):
        return None


def ensure_python() -> tuple[str, list[str]] | None:
    # py.exe is a launcher: use it with -3 only for the initial bootstrap.
    py = shutil.which("py.exe")
    if py:
        v = py_version(py, ["-3"])
        if v and v >= (3, 10, 0):
            return py, ["-3"]
    py3 = shutil.which("python.exe") or shutil.which("python")
    if py3:
        v = py_version(py3, [])
        if v and v >= (3, 10, 0):
            return py3, []
    return None


def load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def ensure_env(venv_py: Path) -> dict[str, str]:
    if not ENV_FILE.exists():
        shutil.copy2(ENV_EXAMPLE, ENV_FILE)
    values = load_env()
    import secrets
    token = values.get("GOLDBOT_API_TOKEN", "").strip()
    if (not token or token == "change-me-to-a-long-random-string") and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8-sig").strip()
    if not token or token == "change-me-to-a-long-random-string":
        token = secrets.token_urlsafe(32)
    # The local token file is the single source of truth for the browser login.
    TOKEN_FILE.write_text(token + "\n", encoding="utf-8")
    values["GOLDBOT_API_TOKEN"] = token
    values["GOLDBOT_EXECUTION_MODE"] = "SIMULATION"
    values["GOLDBOT_MARKET_DATA_SOURCE"] = "MT4"
    text = "\n".join(f"{k}={v}" for k, v in values.items()) + "\n"
    ENV_FILE.write_text(text, encoding="utf-8")
    return values


def prepare_ea() -> None:
    ea = ROOT / "mql4" / "ASTRAFeedEA.mq4"
    if not ea.exists():
        raise RuntimeError("mql4\\ASTRAFeedEA.mq4 is missing.")
    base = Path(os.environ.get("APPDATA", "")) / "MetaQuotes" / "Terminal"
    copied = False
    if base.is_dir():
        for terminal_dir in base.iterdir():
            if not terminal_dir.is_dir():
                continue
            experts = terminal_dir / "MQL4" / "Experts"
            if experts.is_dir():
                shutil.copy2(ea, experts / "ASTRAFeedEA.mq4")
                out(f"       EA source copied to: {experts}")
                copied = True
    if not copied:
        out("       No standard MT4 data folder was found automatically.")
        out("       This is OK if ASTRAFeedEA is already installed in MT4.")


def wait_for_feed(feed: Path, venv_py: Path, timeout: int = 120) -> bool:
    log = LOGS / "mt4check.log"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if feed.exists():
            with log.open("w", encoding="utf-8") as fh:
                cp = subprocess.run(
                    [str(venv_py), "-m", "goldbot", "mt4check"],
                    cwd=str(ROOT), text=True, stdout=fh, stderr=subprocess.STDOUT,
                    check=False,
                )
            if cp.returncode == 0:
                return True
        time.sleep(1)
    return False


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    out("\n============================================================")
    out("                ASTRA TRADER - STARTUP")
    out("============================================================\n")

    out("[1/6] Checking Python environment...")
    chosen = ensure_python()
    if chosen is None:
        return fail("Python 3.10+ was not found in PATH.")
    system_py, py_prefix = chosen
    v = py_version(system_py, py_prefix)
    out(f"       Python {v[0]}.{v[1]}.{v[2]}")
    if not VENV_PY.exists():
        out("       Creating .venv...")
        cp = run([system_py, *py_prefix, "-m", "venv", str(VENV)])
        if cp.returncode != 0:
            return fail(f"Creating .venv failed (exit {cp.returncode}).")
    if not VENV_PY.exists():
        return fail(".venv was created but Scripts\\python.exe is missing.")
    cp = run([str(VENV_PY), "-m", "pip", "--version"], capture=True)
    if cp.returncode != 0:
        return fail("pip is unavailable inside .venv.")
    marker = VENV / ".deps_ok"
    imp = run([str(VENV_PY), "-c", "import goldbot"], capture=True)
    if not marker.exists() or imp.returncode != 0:
        out("       Installing/repairing Python packages...")
        cp = run([str(VENV_PY), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(ROOT / "requirements.txt")])
        if cp.returncode != 0:
            return fail(f"Dependency installation failed (exit {cp.returncode}).")
        marker.write_text("ok\n", encoding="ascii")
    out("       Python environment OK.")

    out("[2/6] Preparing MT4 read-only bridge...")
    try:
        prepare_ea()
    except Exception as exc:
        return fail(str(exc))

    out("[3/6] Loading configuration...")
    values = ensure_env(VENV_PY)
    out(f"       Personal ASTRA token: {TOKEN_FILE}")
    if values.get("GOLDBOT_EXECUTION_MODE") != "SIMULATION":
        return fail("This build is locked to SIMULATION mode.")
    if values.get("GOLDBOT_MARKET_DATA_SOURCE") != "MT4":
        return fail("This build is locked to the MT4 read-only feed.")

    feed_dir = Path(os.environ.get("APPDATA", "")) / "MetaQuotes" / "Terminal" / "Common" / "Files"
    override = values.get("GOLDBOT_MT4_FEED_PATH", "").strip().strip('"')
    if override:
        feed_dir = Path(override)
    feed = feed_dir / "ASTRA_MT4_FEED.json"

    out("[4/6] Checking MT4 feed...")
    out(f"       Expected feed: {feed}")
    if not wait_for_feed(feed, VENV_PY):
        out("[ERROR] MT4 feed was not detected/validated within 120 seconds.")
        out("       Compile ASTRAFeedEA in MetaEditor, attach it to the XAUUSD DEMO chart, and keep MT4 open.")
        out(f"       Feed path: {feed}")
        return 1
    out("       MT4 read-only feed is healthy.")

    out("[5/6] Checking for an old ASTRA instance...")
    run([str(VENV_PY), "-m", "goldbot", "shutdown"], capture=True)
    time.sleep(0.5)

    out("[6/6] Starting ASTRA in SIMULATION mode...")
    out_log = LOGS / "launcher-console.log"
    err_log = LOGS / "launcher-error.log"
    with out_log.open("a", encoding="utf-8") as stdout, err_log.open("a", encoding="utf-8") as stderr:
        subprocess.Popen(
            [str(VENV_PY), "-m", "goldbot", "run"],
            cwd=str(ROOT), stdout=stdout, stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

    deadline = time.monotonic() + 45
    healthy = False
    url = "http://127.0.0.1:8787/healthz"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                healthy = resp.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            healthy = False
        if healthy:
            break
        time.sleep(1)
    if not healthy:
        return fail(f"ASTRA did not become healthy within 45 seconds. See {out_log} and {err_log}")
    try:
        os.startfile("http://127.0.0.1:8787/")  # type: ignore[attr-defined]
    except OSError:
        pass
    out("\n============================================================")
    out("ASTRA TRADER IS RUNNING AND HEALTH-CHECKED")
    out("Dashboard: http://127.0.0.1:8787/")
    out("Execution: SIMULATION - NO REAL ORDERS")
    out("Logs:      .\\logs\\")
    out("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
