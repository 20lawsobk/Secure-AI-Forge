---
name: Pocket accelerator in GPU GEMM paths
description: Adaptive content-hash dedup layer wired into CPUBackend.gemm and HyperGPU gemm paths; gating, aliasing, and test-statefulness lessons
---

# Pocket accelerator wired into the Digital GPU

- All GEMM entry points (maxcore `CPUBackend.gemm`, training `HyperGPU.gemm/mixed_gemm/gemm_batched`) route through one process-wide `PocketAccelerator` singleton: content-hash dedup serves repeated multiplications at O(hash+memcpy), independent of GEMM size.
- **Why:** "infinite speedup" ask is only honest for repeats; the layer must never slow non-repetitive work, so it is adaptively gated.
- **How to apply:** gates = FLOP floor (small GEMMs never hashed) + per-shape-pocket mute when hit-rate <5% after warmup (training activations never repeat → hashing turns itself off) + periodic re-probe; kill switch `MAXCORE_POCKET_ACCEL=0`, budget `MAXCORE_POCKET_ACCEL_MB`.

Lessons:
- `np.ascontiguousarray` ALIASES already-contiguous input — a cache `_put` must `.copy()` or callers mutating their miss result poison the cache (architect caught this; regression test exists).
- Keying must include everything that changes numerics: bias bytes, activation, and mode flags like `MAXCORE_EMULATE_FP16`.
- Tests hitting the fleet dedup store (pdim storage online) are stateful across runs: use per-run unique namespaces, and poll briefly for `source=="cache"` because the leader persists via a background thread.
- Lazy-import the accelerator from backends (`cpu_backend`, `hyper_core`) to avoid maxcore package import cycles.
