---
name: CPU honest-perf decisions (quant + thread sizing)
description: Why quantization is a numerics reference (not a CPU speedup) and how BLAS thread sizing must be runtime-derived, not hardcoded.
---

# Quantization is a NUMERICS reference on this CPU box, not a speed path

`ai_model/maxcore/precision.py` (per-tensor + per-channel symmetric INT8/INT4
quantize/dequantize/quantized_matmul) is labeled `reference_numerics` and makes
**no speed claims**.

**Why:** benchmarked on this host — integer matmul via numpy is ~50x SLOWER than
float BLAS (no INT8 SIMD GEMM path in numpy/OpenBLAS); tiled/blocked GEMM is also
4–6x slower than `np.matmul` because scipy-openblas already cache-blocks in C.
INT8 rel-err ~2%, INT4 ~19% per-tensor → per-channel scaling needed for INT4.
**How to apply:** keep quant as a correctness/accuracy tool (e.g. validating a
future real INT8 kernel), never sell it as a CPU throughput win. Calibration
clips beyond the 0.999 percentile by design, so outliers saturate and do NOT
round-trip — assert tight bounds only on in-range values.

**Where the real speed actually lives (2025 research):** the published ~3x INT4/8
CPU speedup requires *hardware* int8 GEMM (AVX-512 VNNI) via oneDNN or
llama.cpp/GGML — NOT numpy. So a real quant win = a NEW oneDNN/GGML-backed
backend, keeping the numpy path unquantized. Separately, `torch.compile`
(Inductor) + oneDNN Graph fusion is the zero-accuracy-loss CPU win that runs on
this host today (op fusion + weight prepack + AVX-512), and it plugs into the
existing torch backend seam. GGUF export + knowledge distillation are the
larger-lift structural gains. None of these change the substrate — all are
same-silicon software wins (consistent with the standing honest line below).

# BLAS/OpenMP thread sizing: runtime-derived, never hardcoded

`ai_model/maxcore/hardware.py` (`plan_blas_threads`, `configure_blas_threads`)
+ a pre-numpy block at the top of `server.py`.

**Why:** dev box = 2 vCPU / 8 GiB; production Reserved VM = 16 vCPU / 64 GiB. A
hardcoded `16` starves dev (oversubscribes 2 cores) or is redundant in prod. So
everything derives from `os.cpu_count()`.
**How to apply:**
- Set `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB_*_NUM_THREADS` via `os.environ.setdefault`
  **before numpy is imported** (BLAS reads them at import). Operator override wins.
- For a **single-process** server this is effectively a no-op — OpenBLAS already
  defaults to all cores. Be honest about that.
- It only earns its keep with **multiple worker PROCESSES**: N procs × all-core
  BLAS pools oversubscribe; cap each to `cpus // N`. The current `PDIMWorker` is
  thread-based (one process, one shared BLAS pool) so there is nothing to cap yet.
- E402 is expected for the pre-numpy block; handled by a `server.py`-scoped
  `[tool.ruff.lint.per-file-ignores]` in `artifacts/ai-training-server/pyproject.toml`.

# Standing honest line (user pushes "Digital GPU == real B200")

Software replicates GPU *semantics/behavior* exactly, but cannot manufacture
*throughput*: simulating a faster machine on a slower host is always slower
(physical limit). Real photographic output path on this box = retrieval +
compositing, not in-house photoreal training.
