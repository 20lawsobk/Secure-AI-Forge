"""
Dataset Downloader — Indestructible Supervisor
===============================================
Runs the Pocket Dimension dataset downloader and automatically restarts
it when it finishes or crashes.  Because the downloader resumes from a
status file, restarting a completed run simply picks up any new/failed
datasets without re-downloading finished ones.

The ONLY way to stop it permanently is:
  • POST /api/maxcore/downloader/stop  (writes control/downloader.stop)
  • DELETE that file manually:  rm control/downloader.stop

All other signals are caught and converted into a subprocess restart.
"""

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
STOP_FLAG = ROOT / "control" / "downloader.stop"
LOG_DIR   = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "control").mkdir(exist_ok=True)

SCRIPT      = [
    sys.executable,
    str(ROOT / "server" / "services" / "diffusion" / "dataset_downloader.py"),
]
DONE_WAIT   = 300    # seconds to wait after a successful complete run before restarting
CRASH_WAIT  = 30     # seconds to wait after a crash before restarting
MAX_BACKOFF = 600    # maximum wait between restarts
FAST_CRASH  = 20     # seconds — classify exit faster than this as a crash

_restart_requested = False


def _ts():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str):
    line = f"[{_ts()}] [Supervisor/Downloader] {msg}"
    print(line, flush=True)
    try:
        log_path = LOG_DIR / f"supervisor_{datetime.utcnow().strftime('%Y-%m-%d')}.log"
        with open(log_path, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _signal_handler(signum, frame):
    global _restart_requested
    sig_name = signal.Signals(signum).name
    if STOP_FLAG.exists():
        log(f"Received {sig_name} with stop flag present — exiting supervisor")
        sys.exit(0)
    log(f"Received {sig_name} — will restart child (supervisor stays alive)")
    _restart_requested = True


for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    try:
        signal.signal(_sig, _signal_handler)
    except (OSError, ValueError):
        pass


def _stop_child(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    except OSError:
        pass


def _interruptible_sleep(seconds: int):
    """Sleep in 1-second increments, bailing early if stop flag appears."""
    for _ in range(seconds):
        if STOP_FLAG.exists():
            return True   # stop requested
        time.sleep(1)
    return False


def run():
    global _restart_requested
    backoff = CRASH_WAIT
    attempt = 0
    proc    = None

    log("Supervisor started — Dataset Downloader will run indefinitely")
    log(f"Stop flag path: {STOP_FLAG}")

    while True:
        if STOP_FLAG.exists():
            log("Stop flag detected — supervisor exiting (downloader will not restart)")
            _stop_child(proc)
            sys.exit(0)

        if _restart_requested:
            log("Restart request from signal — killing child")
            _stop_child(proc)
            _restart_requested = False

        if proc is None or proc.poll() is not None:
            if proc is not None:
                exit_code = proc.poll()
                elapsed   = time.time() - t_start  # noqa: F821

                if exit_code == 0:
                    # Normal completion — datasets finished, wait then re-run to
                    # pick up any new or retried downloads
                    log(f"Downloader completed cleanly (code 0, ran {elapsed:.0f}s) — "
                        f"will restart in {DONE_WAIT}s to retry any failures")
                    if _interruptible_sleep(DONE_WAIT):
                        log("Stop flag during post-completion wait — exiting")
                        sys.exit(0)
                    backoff = CRASH_WAIT
                else:
                    log(f"Downloader exited with code {exit_code} after {elapsed:.0f}s")
                    wait = backoff if elapsed < FAST_CRASH else CRASH_WAIT
                    if elapsed < FAST_CRASH:
                        backoff = min(backoff * 2, MAX_BACKOFF)
                    log(f"Restarting in {wait}s …")
                    if _interruptible_sleep(wait):
                        log("Stop flag during crash backoff — exiting")
                        sys.exit(0)

                if STOP_FLAG.exists():
                    log("Stop flag appeared — supervisor exiting")
                    sys.exit(0)

            attempt  += 1
            t_start   = time.time()
            log(f"Launching Dataset Downloader (attempt {attempt}) …")

            try:
                proc = subprocess.Popen(
                    SCRIPT,
                    cwd=str(ROOT),
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                log(f"Child PID: {proc.pid}")
            except Exception as e:
                log(f"Failed to start downloader: {e} — retrying in {backoff}s")
                if _interruptible_sleep(backoff):
                    sys.exit(0)
                backoff = min(backoff * 2, MAX_BACKOFF)
                continue

        time.sleep(5)


if __name__ == "__main__":
    run()
