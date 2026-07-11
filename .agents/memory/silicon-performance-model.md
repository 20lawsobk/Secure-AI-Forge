---
name: Digital-silicon performance model
description: MaxCoreSilicon is an optional what-if ESTIMATOR wired as telemetry across the GPU engines; never an execution path, never a measurement.
---

`ai_model/gpu/silicon_model.py` (`MaxCoreSilicon`) is an architectural
performance model — the gem5/roofline class of tool. It estimates cycles/time
for ops on a hypothetical die described by chosen constants (tiles,
flops/cycle, bandwidth, clock). It performs NO arithmetic and makes nothing
faster.

**Rule:** every value it emits is a prediction, keyed `estimated_*` with
`is_measurement=False`, `mode="performance_model"`, and a disclaimer. These
numbers must NEVER be reported as measured throughput, and must NEVER be mixed
with real wall-clock timings. A simulated cycle count over a clock constant you
picked is not a benchmark.

**Wiring:** attached via an optional trailing `silicon=None` param on
`DigitalGPU`, `HyperGPU`, `HyperGPUBackend`, `DigitalGPUBackend`. Each executed
op calls a side-channel `_model(...)` AFTER the real compute; results are never
touched. Default (no silicon) leaves `status()["silicon"]`/`silicon_report()`
as None, so the running server is unaffected.

**Why:** the standing project decision is honesty — the in-house stack is
numpy/torch on CPU; a model of a chip is a blueprint, not the chip. The real
speed lever is deploying on actual silicon via the real `GPUBackend`
(`maxcore/backend/device_backend.py`), not this estimator.

**How to apply / landmines:**
- If you add telemetry hooks, keep `_model(...)` signatures in lockstock with
  callsites (a `precision=` kwarg mismatch on `HyperGPU.mixed_gemm` crashed real
  compute once — telemetry must never be able to break execution).
- `MaxCoreSilicon.report()` must stay lock-guarded (`_report_locked`) since ops
  and reports run concurrently.
- Never let a `simulate()`/report estimate surface to a user or endpoint without
  its estimate label.
