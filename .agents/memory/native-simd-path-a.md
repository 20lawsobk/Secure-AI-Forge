---
name: Native SIMD (Path A) compiled kernels
description: ai_model/gpu/native — real gcc-compiled fused CPU SIMD kernels, the honest speedup path; measured 2.5-8x over idiomatic numpy
---

`ai_model/gpu/native/` is the honest **Path A** ("software GPU done right = COMPILE, don't interpret"). It compiles fused C kernels to host SIMD (AVX-512 here) via gcc + ctypes and beats idiomatic numpy — a REAL, measured CPU speedup, still CPU (never GPU hardware). This is additive library code, not in the running model path.

## Why it wins (the honest mechanism)
- The win is **fusion + one memory pass**, not magic. Idiomatic numpy runs an op chain as N ufunc passes with temporaries; a compiled fused loop makes one vectorized pass. So the speedup scales with how memory-bound / how many ops are fused: measured ~2.5x (axpby, 2 reads) → ~5x (rmsnorm, hardswish) → ~8x (affine→relu→sq→scale, 5-op chain). float32, 8M elems, 2-core AVX-512 host.
- Beating numpy on **gemm is NOT the target** — numpy already calls BLAS (SIMD+threaded). Target the fused elementwise/reduction ops numpy is bad at.

## Design rules (don't regress)
- **Never-raise:** `NativeCompiler.compile()` returns None (reason in `last_error`) if no gcc or build fails; every kernel in `NativeKernels` has a numpy fallback with identical math. Correctness NEVER depends on the compiler — native is a speed optimization only. Tests force `compiler.available=False` to cover the fallback path.
- **Honesty:** `describe()["is_hardware_execution"]` is always False; it's labeled compiled-CPU-SIMD (SPMD-on-CPU) with a bounded ceiling, not GPU throughput. Keep consistent with the digital-gpu honesty contract.
- Flags: `-O3 -march=native -funroll-loops -ffast-math -shared -fPIC` (+optional `-fopenmp`, default OFF — libgomp runtime load can fail on NixOS and 2 cores makes threads marginal; SIMD-fusion is the main win). `-ffast-math` was numerically exact vs numpy for these kernels (verify allclose atol~1e-4; rmsnorm/reductions are the loosest).
- Compiled `.so` cached in tempdir keyed by sha1(source+flags+cc_version); atomic publish via tmp+os.replace. Use `get_native_kernels()` singleton so the lib compiles/loads once per process.
- Only float32 + C-contiguous take the native path (`_use_native`); anything else falls back. Output is always float32.

## Environment note
- Toolchain is gcc-only (no clang/ispc/nvcc); AVX-512 host. No pytest → tests run via the repo's inline runner (import module, call `test_*`). `-march=native` = build host == run host, so it's safe here.
