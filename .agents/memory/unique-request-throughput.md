---
name: Unique-request throughput optimisation
description: What broke unique-request throughput and how it was fixed — lessons for any future perf work on server.py / dynamic_batching.py.
---

## The rule
Never let telemetry calls share the batcher queue with user-facing inference.

**Why:** Each `_infer()` call spawned a daemon thread that submitted a distribution agent call to the same `GenerateCoalescer`. Under burst load (120 unique requests) this doubled the batcher's queue depth, cascaded into backpressure on `_ready_q`, and caused latency to blow up on every subsequent wave of the same test run.

**How to apply:** Any model call whose result is unused (fire-and-forget telemetry, analytics, training signal collection) must NOT go through the live batcher. Options: collect offline via the training pipeline, submit to a separate low-priority queue, or skip entirely on the hot path.

---

## Batch config must be hard-assigned, not setdefault

`on_startup()` used `os.environ.setdefault("AI_BATCH_MAX", ...)`. If the server process inherits stale env vars from a previous run (e.g. `AI_BATCH_MAX=8` from an old session), `setdefault` silently leaves them in place. Always use `os.environ["AI_BATCH_MAX"] = ...` for performance-critical config that must reflect the current code.

---

## Redundant GPU lives per request

Before: `_infer()` wrapped `_script_agent.run()` in `with _get_gpu_pool().spawn_sync()`. The batcher already spawns one GPU life per batch. This created 2–3 nested `HyperGPUBackend` allocations (512 lanes, 8 tensor cores, VRAM init) per unique request with zero compute benefit.

Fix: remove `spawn_sync` from `_infer()`. The outer async `spawn()` in the coalesced handler is enough for lifecycle tracking; the batcher owns the real GPU compute lifecycle.

---

## Throughput baseline (post-fix, clean server)

- W3 unique content gen @30 concurrent: **4.9 req/s**
- W4 quality-validated @8 concurrent: **13.1 req/s** (batcher batching visible: +2.3× vs pre-fix 5.7)
- W6 80 fully unique @25 concurrent: **4.8 req/s** (matches W3 — no degradation for unique load)
- Coalescer: 200 identical → 7 GPU ops (99.5% reduction)

---

## Batcher config (post-fix)
- `AI_BATCH_MAX = min(64, max(32, cpu_count * 8))`  — was `min(64, max(16, cpu * 4))`
- `AI_BATCH_WINDOW_MS = 4`  — was 6 (stale: 8)
- Both hard-assigned in `on_startup()` so restarts always get correct values.
