---
name: workflow topology & python spawner contention
description: Why only `Start application` should run; how redundant Python spawners cause model-lock contention and how to resolve it.
---

# Workflow topology — single owner of the Python server

`Start application` is the ONLY workflow needed to run the whole app. Its api-server
child (`pnpm --filter @workspace/api-server run dev`) unconditionally **child-spawns and
auto-restarts** the Python AI server (`uv run python3 server.py`, port 9878) via
`ensurePythonServer()`. The dashboard (port 5000) runs in the same workflow. So one
workflow = api-server(8080) + dashboard(5000) + Python(9878). Python logs are forwarded
to the api-server console prefixed `[Python]`.

## The contention
Several things historically spawned `server.py` in parallel:
- the now-removed `AI Training Server` workflow (ran Python directly), and
- the **artifact-managed** per-artifact workflows `artifacts/api-server: API Server`
  and `artifacts/ai-dashboard: web`.

Multiple api-server instances each run a health monitor that tries to own Python. Only one
wins the `fcntl` flock (`/tmp/maxcore_model_9878.lock`); losers log
"standing down to avoid a duplicate model load" / "Port 9878 held by another process —
standing by". The app still works (one Python serves), but ownership becomes murky and
every code reload is a guessing game about which process to restart.

## Constraints learned
- `AI Training Server` was deletable via `removeWorkflow`.
- The two `artifacts/*` workflows are **artifact-managed**: `removeWorkflow` returns
  `PROHIBITED_ACTION` ("managed by an artifact and cannot be deleted"). They can only be
  left **stopped** (state `finished`). Don't try to delete or reconfigure them.
- `configureWorkflow` also returns `PROHIBITED_ACTION` for artifact-managed workflows —
  their command cannot be changed.
- A "finished" artifact workflow can still leave a **lingering process tree** alive
  (its pnpm → tsx → cluster primary + workers) that keeps respawning a competing Python.

## Durable fix: MODEL_API_PORT discriminator (in python-server.ts)
`Start application` is the ONLY workflow that sets `MODEL_API_PORT=9878` explicitly.
Artifact-managed workflows run `pnpm --filter @workspace/api-server run dev` with no env
vars, so `process.env.MODEL_API_PORT` is `undefined` in those processes.

`python-server.ts` now sets:
```
const PYTHON_SPAWN_DISABLED =
  process.env.DISABLE_PYTHON_SPAWN === "1" ||
  process.env.MODEL_API_PORT === undefined;
```

When `PYTHON_SPAWN_DISABLED` is true, `ensurePythonServer()` waits for Python (up to 60s)
but never spawns it and never starts the health monitor. The artifact api-server becomes a
pure proxy that routes to whichever Python is already alive.

**Why this works without needing to reconfigure artifact workflows:** the discriminator
reads an env var that only `Start application` sets — no workflow config changes needed,
and it self-heals on every restart.

## How to recover if ownership hops to the artifact api-server
1. Identify trees by parentage. `Start application`'s pnpm is a child of the bash runner;
   the artifact api-server's pnpm is a direct child of the workflow supervisor (low PID ≈ 26).
2. `kill -TERM <artifact tsx PID> <cluster primary> <workers> <uv PID> <python3 PID>`
3. Start application's health monitor detects Python is down (within 15s) and respawns.
4. Verify: exactly one primary, one Python pair (uv → python3) whose PPID is that primary.

## Normal healthy process tree
```
bash (Start application) → pnpm → tsx → cluster primary (742)
                                              ├── worker 771
                                              ├── worker 772
                                              └── uv (802) → python3 server.py (805)
```
25/25 keepalive endpoints alive = healthy.

## Production Python ownership
Proxy-only mode triggers whenever MODEL_API_PORT is unset. The production `serve` script in artifacts/api-server/package.json MUST set MODEL_API_PORT (it defaults to 9878 there) or the deployed app has NO Python owner: both api-server processes wait forever in proxy-only mode, the circuit breaker keeps tripping, and the entire /api surface 503s in prod while dev looks healthy.
**Why:** July 2026 prod outage — deployment ran `pnpm start` → serve without MODEL_API_PORT; nobody spawned server.py.
**How to apply:** never strip MODEL_API_PORT from the serve script; only the cluster primary calls ensurePythonServer, so a single deployed api-server artifact is safe from double-spawn.

## Prod external port mapping (GCE/VM)
- .replit maps externalPort 80 → localPort 5000, but prod `serve` defaulted PORT=8080 → nothing on 5000 → external health 000 while internal keepalive shows all endpoints alive. Deployment run must set PORT=5000 (now `bash -c "PORT=5000 MODEL_API_PORT=9878 pnpm start"`). Check the [[ports]] map FIRST when prod is externally unreachable but deployment logs look healthy.

## Prod flapping root cause (healthchecks)
- Platform healthchecks GET "/", "/api", "/uploads" and RESTART the VM on 5xx. Any Python boot/restart window used to 500 these → VM kill → minutes-long model reload → flap loop. app.ts now serves always-200 fast-paths for those exact paths (+ /healthz, SPA fallback never 500s, global error middleware). Keep healthcheck paths independent of Python.
- uploads/ janitor (server.py, hourly, 24h TTL + 2GB cap, audio_/video_/scene_/stem_/tmp_ prefixes only) prevents VM disk-fill — disk-full surfaced as mid-render ffmpeg failures.
