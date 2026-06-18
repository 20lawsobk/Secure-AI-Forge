---
name: workflow topology & python spawner contention
description: Why only `Start application` should run; how redundant Python spawners cause model-lock contention and how to resolve it.
---

# Workflow topology — single owner of the Python server

`Start application` is the ONLY workflow needed to run the whole app. Its api-server
child (`pnpm --filter @workspace/api-server run dev`) unconditionally **child-spawns and
auto-restarts** the Python AI server (`uv run python3 server.py`, port 9878) via
`ensurePythonServer()` — there is no env flag to disable that spawn. The dashboard
(port 5000) runs in the same workflow. So one workflow = api-server(8080) + dashboard(5000)
+ Python(9878). Python logs are forwarded to the api-server console prefixed `[Python]`.

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
- A "finished" artifact workflow can still leave a **lingering process tree** alive
  (its pnpm → tsx → cluster primary + workers) that keeps respawning a competing Python.

## How to get a clean single owner
1. Identify trees by parentage, not by guesswork. `Start application`'s api-server pnpm is
   a child of the `Start application` bash; the artifact api-server's pnpm is a direct child
   of the workflow **supervisor** (a low PID like 25). Cluster workers' PPID = their primary.
2. Kill the artifact api-server tree (pnpm + tsx + primary + workers) — NOT the
   `Start application` tree, and not the Python pair.
3. The `Start application` api-server health monitor then spawns and **owns** the single
   Python (`uv` parent → `python3` child = one logical server; two PIDs is normal).
4. Verify: exactly one api-server primary, exactly one Python pair whose PPID is that
   primary, dashboard 5000 → 200, Python 9878 → 200.

**Why:** can't delete the artifact workflows, so the only durable lever is keeping them
stopped and ensuring `Start application` is the sole live api-server. If the platform later
auto-restarts an artifact workflow (e.g. after a package install), the flock self-heals to
one serving Python — contention returns only as a reload annoyance, not a breakage.
