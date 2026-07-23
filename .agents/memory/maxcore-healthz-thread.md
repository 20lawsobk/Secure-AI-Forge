---
name: MaxCore healthz liveness thread
description: Why the healthz server exists, how it's wired, and what breaks if it's removed.
---

# MaxCore healthz liveness thread

## The rule
The hung-detection probe in `python-server.ts` must target the healthz port (PYTHON_PORT+1, default 9879), NOT the main uvicorn API port (9878).

**Why:** Python's uvicorn event loop blocks for 10–25 s during GC in production (87–90% memory usage). Probing `/health` on port 9878 during a GC pause times out and looks like a hang — Node kills and restarts a perfectly healthy Python process, causing 90–120 s of downtime for model reload. The healthz thread runs in its own daemon thread (`threading.Thread`), completely independent of uvicorn. GC pauses cannot block it.

**How to apply:**
- `server.py`: `_start_healthz_server()` — called first in `on_startup`, before model init. Binds `_HEALTHZ_PORT = HEALTHZ_PORT env var` (or `MODEL_API_PORT + 1`). Class definition is INSIDE the function (not module-level) so `import http.server` is scoped correctly — putting the class at module scope causes `AttributeError: module 'http' has no attribute 'server'` at import time.
- `python-server.ts` `probeHttpHealth()`: checks main port for `down`, then healthz port with 10 s timeout for `hung`. Falls back to 25 s uvicorn probe if healthz port isn't bound yet (old deployment / startup race).
- `HEALTHZ_PORT` env var is injected by `spawnPython()` so Python knows which port to bind.

## Other uptime fixes in the same pass

**External SIGTERM no longer suppresses restart:**
`python-server.ts` exit handler used to return early on `signal === "SIGTERM"`. Now it waits 500 ms for `shuttingDown` to be set by our own shutdown handler. If we didn't request the shutdown, even a SIGTERM triggers a restart.

**Unowned hung process force-kill:**
When Node restarts and finds Python already running (monitoring-only mode), it can't take ownership. After `MAX_HUNG_PROBES_UNOWNED = 3` consecutive hung probes (~75 s), it runs `pkill -9 -f server.py` so the next "down" poll can spawn a clean replacement. Confirmed working in production logs.

**uvicorn.run() exception guard:**
`server.py __main__` wraps `uvicorn.run()` in try/except. `SystemExit` re-raises (clean shutdown). Any other exception logs traceback + calls `sys.exit(1)` so Node's exit handler always fires and restarts.

**Health poll interval:** reduced from 15 s to 8 s for faster hung detection.
