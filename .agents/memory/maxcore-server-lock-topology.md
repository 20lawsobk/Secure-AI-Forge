---
name: MaxCore server.py lock topology — which workflow to restart to reload it
description: Why restarting "AI Training Server" often does NOT reload edited server.py, and how to reliably get new server.py code live + verify it
---

# Three workflows can run server.py; one model lock arbitrates
`server.py` (the Python AI engine, port 9878) is launched from MULTIPLE places at once:
- the **"AI Training Server"** workflow runs it directly, AND
- the Express api-server (run by BOTH the **"Start application"** workflow and the standalone **"artifacts/api-server: API Server"** workflow) spawns it as a monitored child with backoff restarts.

Only ONE process actually loads the model and runs `_init_storage` (workers + CoverageWatchdog), guarded by `/tmp/maxcore_model_9878.lock`. Every other instance prints "Another AI server instance already holds … standing down to avoid a duplicate model load" and serves nothing.

**Why this bites:** after editing `server.py`, restarting the **"AI Training Server"** workflow alone frequently just makes it stand down (the lock is already held by an api-server child), so the LIVE process keeps running the OLD code. The restart "succeeds" but your changes are not live.

# How to reliably reload server.py and verify
1. Find the live process and its owner:
   `ps -eo pid,ppid,etimes,cmd | rg 'python3 .*server\.py'` → trace `ppid` up to the cluster-primary PID, match it to the workflow whose log shows that `[Cluster] Primary <pid>`.
2. Restart THAT owning workflow. Killing the current lock-holder frees the lock; whichever monitor respawns first grabs it and loads the new code (all three launch the same file from disk, so the source is always current). Ownership can hop between the two api-server workflows across restarts — that's fine, the code is identical.
3. Verify with an endpoint that only the NEW code exposes: a route added this change should flip from `404` (old code) to `401`/`200`. Pre-existing routes (e.g. `/watchdog/status`) return `401` whether or not your change is live, so they don't prove a reload.

# Auth for endpoint verification
Protected Python endpoints accept an `X-Api-Key` header. Any key in `_ENV_BYPASS_KEYS` (the env admin keys) authenticates with full scopes — convenient for curl checks. NEVER echo the key value into shell output/logs; pass it via a shell variable only.
