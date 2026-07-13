---
name: Pocket GPU Pool
description: PocketGPUPool replaces INFERENCE_GATE — each unique inference request gets its own HyperGPUBackend life (born→working→dead) from an unbounded pocket-dimension pool.
---

## Rule
Never re-introduce `INFERENCE_GATE.slot()` around inference calls in server.py.
The gate is retired; the pool is the concurrency primitive.

**Why:** The pocket dimension is unbounded — spawning one GPU per unique request
costs nothing in queue latency. INFERENCE_GATE was a bottleneck (single shared GPU,
N requests queued). With the pool, unique requests run in parallel, identical requests
are collapsed by the async coalescer to one GPU life.

## How to apply
- New inference handlers: `with _get_gpu_pool().spawn_sync(digest) as _glife:` inside a worker thread, or `async with _get_gpu_pool().spawn(digest) as _glife:` in the async layer.
- `_digest_str(fields_dict)` → blake2b hex string — use as the spawn key so identical concurrent requests map to one life (handled by the coalescer upstream, but the key helps with telemetry).
- Pool stats (`pool_alive`, `pool_total_born`, `pool_total_dead`, `pool_avg_life_ms`) are merged into `/gpu/status` — keep this wiring when updating that endpoint.
- `PocketGPUInstance.backend` is a fresh `HyperGPUBackend(lanes=512, tensor_cores=8, precision=MIXED)`. VRAM is flushed on death automatically.

## Files
- `ai_model/gpu/pocket_pool.py` — PocketGPUPool + PocketGPUInstance
- `server.py` — `_get_gpu_pool()`, `_digest_str()`, two `spawn_sync` sites, two `async spawn` sites in coalesced coroutines, `/gpu/status` pool stats merge
