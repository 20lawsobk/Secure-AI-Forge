"""
MaxCore Server — Indestructible Supervisor
==========================================
Runs maxcore_server.py in a subprocess and restarts it automatically
on ANY exit (crash, OOM, segfault, signal).

The ONLY way to stop it permanently is:
  • POST /api/maxcore/shutdown  (writes control/maxcore.stop flag)
  • Delete the file manually:  rm control/maxcore.stop

All other signals (SIGTERM, SIGINT, SIGHUP) are caught and converted
into a subprocess restart rather than a supervisor exit.
"""

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
STOP_FLAG = ROOT / "control" / "maxcore.stop"
LOG_DIR   = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "control").mkdir(exist_ok=True)

SCRIPT     = [sys.executable, str(ROOT / "maxcore_server.py")]
MIN_BACKOFF = 5      # seconds before first restart
MAX_BACKOFF = 120    # cap
FAST_CRASH  = 10     # seconds — if process dies faster than this, it's a crash loop

_restart_requested = False  # set by signal handler


def _ts():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str):
    line = f"[{_ts()}] [Supervisor/MaxCore] {msg}"
    print(line, flush=True)
    try:
        log_path = LOG_DIR / f"supervisor_{datetime.utcnow().strftime('%Y-%m-%d')}.log"
        with open(log_path, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _signal_handler(signum, frame):
    """
    Convert SIGTERM / SIGINT into a restart signal for the child process.
    The supervisor itself keeps running.
    """
    global _restart_requested
    sig_name = signal.Signals(signum).name
    if STOP_FLAG.exists():
        log(f"Received {sig_name} with stop flag present — exiting supervisor cleanly")
        sys.exit(0)
    log(f"Received {sig_name} — forwarding to child and scheduling restart (supervisor stays alive)")
    _restart_requested = True


# Trap every catchable signal
for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    try:
        signal.signal(_sig, _signal_handler)
    except (OSError, ValueError):
        pass


def _stop_child(proc: subprocess.Popen):
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    except OSError:
        pass


def run():
    global _restart_requested
    backoff   = MIN_BACKOFF
    attempt   = 0
    proc      = None

    log("Supervisor started — MaxCore Server will run indefinitely")
    log(f"Stop flag path: {STOP_FLAG}")

    while True:
        # ── Check stop flag ───────────────────────────────────────────────────
        if STOP_FLAG.exists():
            log("Stop flag detected — supervisor exiting (MaxCore Server will not restart)")
            _stop_child(proc)
            sys.exit(0)

        # ── Reset restart request from signal ─────────────────────────────────
        if _restart_requested:
            log("Restart request from signal — killing child before relaunch")
            _stop_child(proc)
            _restart_requested = False

        # ── Start child if not running ────────────────────────────────────────
        if proc is None or proc.poll() is not None:
            if proc is not None:
                exit_code = proc.poll()
                log(f"Child exited with code {exit_code} (attempt {attempt})")

                if STOP_FLAG.exists():
                    log("Stop flag appeared after crash — not restarting")
                    sys.exit(0)

                wait = backoff
                log(f"Restarting in {wait}s … (backoff={backoff}s)")
                # Check stop flag while waiting
                for _ in range(wait):
                    if STOP_FLAG.exists():
                        log("Stop flag detected during backoff — supervisor exiting")
                        sys.exit(0)
                    time.sleep(1)

            attempt  += 1
            t_start   = time.time()
            log(f"Launching MaxCore Server (attempt {attempt}) …")

            try:
                proc = subprocess.Popen(
                    SCRIPT,
                    cwd=str(ROOT),
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                log(f"Child PID: {proc.pid}")
            except Exception as e:
                log(f"Failed to start child: {e} — will retry in {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
                continue

            # Give it a moment to start
            time.sleep(2)
            if proc.poll() is not None:
                elapsed = time.time() - t_start
                log(f"Child died immediately (after {elapsed:.1f}s) — crash loop protection")
                if elapsed < FAST_CRASH:
                    backoff = min(backoff * 2, MAX_BACKOFF)
                else:
                    backoff = MIN_BACKOFF
            else:
                log("Child is running")
                backoff = MIN_BACKOFF  # Reset backoff on healthy start

        # ── Poll loop — check every 5 s ───────────────────────────────────────
        time.sleep(5)


if __name__ == "__main__":
    run()
