---
name: Software-GPU paths (the recurring "digital GPU" dispute)
description: The two legitimate real-world versions of the user's "software digital GPU", and the honest ceiling of each.
---

# "Your way" (software digital GPU) is a real field — two distinct paths

The user's recurring push is a software re-creation of a GPU (SIMT/warp/kernel
model — `SIMDCore`/`Scheduler`/`OpCode`, the `sm_102` kernels) run off-GPU. This
is legitimate CS with decades of prior art. There are TWO real versions, with
different goals — don't conflate them:

**Path A — SPMD/SIMT-on-CPU for SPEED.** Write GPU-style kernels, compile them to
CPU SIMD. Tools: **ISPC** (Intel Implicit SPMD Program Compiler — program
instances = CUDA threads, compiles to SSE/AVX/AVX-512; ~3x SSE / 5-6x AVX + core
scaling, BSD), SPIR-V→ISPC, MLIR/LLVM JIT + autotuning (TVM MetaSchedule, Halide),
Mojo/MAX. Academic base: MCUDA (Illinois 2008), SIMT-to-SIMD transpilers (2025),
"Unleashing CPU Potential for Executing GPU Programs" (GaTech/NSF).
**Key upgrade:** the current digital GPU *interprets* kernels via numpy; the
real-world win is to *compile* them (map "warp" → real SIMD lanes, "SM" → core).
Integrate as a new `Backend` in the registry so `select_backend` can pick it.

**Path B — GPU SIMULATOR for FIDELITY.** Cycle-level architectural models:
GPGPU-Sim / Accel-Sim (trace-driven off real NVIDIA SASS, validated correlation),
MGPUSim, ZLUDA. This is the rigorous version of `silicon_model.py`.

**Why the distinction matters / honest ceiling:**
- Path A extracts the CPU's *full SIMD potential* — a real but BOUNDED single-digit×
  win over scalar/numpy-naive. It is NOT GPU throughput.
- Path B is *slower than the host by design* (it models cycles, doesn't
  accelerate). Value = correctness / perf prediction / research, not speed. Keep
  it labeled `is_measurement=False`; make it rigorous by calibrating against
  Accel-Sim / published SASS, not by asserting estimates.
- Neither makes a CPU deliver GPU throughput — consistent with the standing line
  in cpu-perf-honesty.md. Real GPU speed still needs real silicon (the torch
  `GPUBackend` on a CUDA host).

**How to apply:** when the user asks to "build the digital GPU better," pick the
path by their goal (speed → ISPC/MLIR compile backend; fidelity → Accel-Sim-style
calibration). Offer a measured before/after vs the numpy path; never present a
SIMD win as if it were GPU-class.
