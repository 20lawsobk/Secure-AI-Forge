---
name: Native SIMD (Path A) compiled kernels
description: ai_model/gpu/native — real gcc-compiled fused CPU SIMD kernels, the honest speedup path; measured 2.5-8x over idiomatic numpy
---

`ai_model/gpu/native/` is the honest **Path A** ("software GPU done right = COMPILE, don't interpret"). It compiles fused C kernels to host SIMD (AVX-512 here) via gcc + ctypes and beats idiomatic numpy — a REAL, measured CPU speedup, still CPU (never GPU hardware). This is additive library code, not in the running model path.

## Two kernel classes — the win mechanism differs (this is the key insight)
- **Memory-bound elementwise** (affine_relu_sq, hardswish, axpby): win is **fusion / one memory pass**. Idiomatic numpy runs the chain as N ufunc passes + temporaries; the fused C loop does one pass. SIMD width barely matters here — it's bandwidth-bound, and AVX-512 can even be *slightly slower* than AVX2 via clock throttling. ~2-5x.
- **Compute-bound / transcendental** (softmax, silu, gelu, layernorm, rmsnorm): numpy's `np.exp`/`np.tanh` ufuncs are ALREADY well-vectorized, so a naive scalar-`expf` C kernel **LOSES** (measured 0.32-0.65x). To beat numpy you need (1) a **vectorizable poly exp** (Cephes-style, branchless → auto-vectorizes; ~1e-6 rel err, so results are NOT bit-exact — test with atol~1e-4), (2) **real AVX-512** width, and (3) **OpenMP** across rows. With all three: softmax ~6.5x, silu ~8x, layernorm ~7x, rmsnorm ~8x.
- Beating numpy on **gemm is NOT the target** — numpy already calls BLAS (SIMD+threaded). Target fused elementwise/reduction/transcendental ops.
- Caveat when quoting speedups: numpy's `np.tanh` over large arrays is genuinely slow, so a tanh-approx gelu shows an inflated ~100x+ vs numpy — real ceiling is the ~6-8x band; don't over-claim off the gelu number.

## Design rules (don't regress)
- **Never-raise:** `NativeCompiler.compile()` returns None (reason in `last_error`) if no gcc or build fails; every kernel in `NativeKernels` has a numpy fallback with identical math. Correctness NEVER depends on the compiler — native is a speed optimization only. Tests force `compiler.available=False` to cover the fallback path.
- **Honesty:** `describe()["is_hardware_execution"]` is always False; it's labeled compiled-CPU-SIMD (SPMD-on-CPU) with a bounded ceiling, not GPU throughput. Keep consistent with the digital-gpu honesty contract.
- Flags: `-O3 -march=native -funroll-loops -ffast-math -shared -fPIC` (+optional `-fopenmp`, default OFF — libgomp runtime load can fail on NixOS and 2 cores makes threads marginal; SIMD-fusion is the main win). `-ffast-math` was numerically exact vs numpy for these kernels (verify allclose atol~1e-4; rmsnorm/reductions are the loosest).
- Compiled `.so` cached in tempdir keyed by sha1(source+flags+cc_version); atomic publish via tmp+os.replace. Use `get_native_kernels()` singleton so the lib compiles/loads once per process.
- Only float32 + C-contiguous take the native path (`_use_native`); anything else falls back. Output is always float32.

## Environment note — the flag gotcha (important)
- **`-march=native` is silently STRIPPED on NixOS** (`NIX_ENFORCE_NO_NATIVE is set`), which quietly caps kernels at baseline SSE and kills the transcendental win. gcc prints "Skipping impure flag -march=native" but still returns rc=0, so it's easy to miss. Fix: detect the CPU ISA from `/proc/cpuinfo` and pass **explicit** `-mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma` (or `-mavx2 -mfma`) — those are honored. Always grep the build stderr for "Skipping impure".
- **Why:** without this, softmax/silu/gelu run scalar-ish and are slower than numpy; with it they hit the 6-8x band.
- OpenMP (`-fopenmp`, libgomp) DOES load on this host — enable it (with a non-omp then numpy fallback chain) and gate `parallel for` with `if(rows>64)` / `if(n>100000)` to avoid thread-spawn overhead on small inputs.
- Toolchain is gcc-only (no clang/ispc/nvcc). No pytest → tests run via the repo's inline runner (import module, call `test_*`).
