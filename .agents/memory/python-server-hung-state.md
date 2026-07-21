---
name: Python server hung-state failure modes
description: Two distinct bugs that cause the Node.js api-server to hold ALL requests indefinitely via setPythonRestarting(true) with no recovery path.
---

## Bug 1 — router="application" double-spawn (production)

`router = "application"` in `.replit [deployment]` causes Replit's VM platform to invoke the run command **once per registered artifact**. With two artifacts (api-server + ai-dashboard), two independent cluster primaries start. Both call `ensurePythonServer()`, both try to own port 9878. The two Python instances fight → endless SIGKILL/restart loop → `setPythonRestarting(true)` never cleared → all requests hang.

**Fix:** Remove `router = "application"` from `[deployment]`. The api-server already serves the built dashboard static files in production via `express.static` (NODE_ENV=production block in app.ts). One process, one port (8080).

**Why:** Without the application router, Replit's VM runs the run command exactly once.

## Bug 2 — "monitoring only" mode + hung probe + setPythonRestarting (dev + prod under memory pressure)

When the api-server cluster restarts and Python is already on port 9878, `ensurePythonServer()` goes into "monitoring only" mode: `pythonProcess = null`, starts the health monitor, but doesn't hold a process reference.

At 87–90% memory Python runs GC (30–40k objects freed) which can block its event loop for 10–20s. Old health probe timeout was 10s → GC pause → probe sees "hung" → health monitor fires:

```typescript
// WRONG — old code:
setPythonRestarting(true);   // ← holds all requests
if (pythonProcess) { kill } else { log "cannot force-kill" }
// setPythonRestarting(false) is NEVER called → permanent hang
```

**Fix (python-server.ts):**
1. Increased `probeHttpHealth` AbortController timeout: 10s → 25s (tolerates GC pauses).
2. In the `hung` branch when `pythonProcess` is null: do NOT call `setPythonRestarting(true)`. Log a warning and let requests pass through; circuit breaker surfaces per-request errors. If Python truly dies, the next "down" probe spawns a fresh one.

**Why:** `setPythonRestarting(true)` is only safe to call when there IS a restart cycle that will eventually call `setPythonRestarting(false)`. In monitoring-only mode there is no such cycle.

## How to apply

- Any time you see `[ServerState] Python restarting — holding incoming requests` followed by repeated `Hung process not owned by us` lines with no `Python ready — releasing held requests`, this is Bug 2.
- Any time production logs show two cluster primaries on different ports (8080 and 3000), this is Bug 1.
- Recovery from a live hung state: kill the Python PID directly (`kill -9 <pid>`), then restart `Start application`. The health monitor's "down" probe will respawn Python cleanly.
