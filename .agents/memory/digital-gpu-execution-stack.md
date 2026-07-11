---
name: Digital GPU execution stack (opcode/precision/scheduler)
description: Honesty contract + correctness rules for ai_model/gpu opcode_spec, precision, execution_graph, telemetry — added on a CPU-only host
---

The `ai_model/gpu/` "digital GPU" is numpy on a CPU host. New execution-stack modules (opcode_spec, precision, telemetry, execution_graph, digital_library) are ADDITIVE library code, not in the running model path — no server reload needed to change them.

## Honesty contract (do not regress)
- Every opcode has `OpcodeSpec.is_hardware_execution`. Anything hardware-implying (fp8 / tensor-core / sm_102) is modeled with `is_hardware_execution=False` plus `target_arch`/`numeric_profile` as *declared modeled metadata* — it captures intent without claiming silicon. `DigitalScheduler` refuses to run any spec with `is_hardware_execution=True` unless a real hardware backend is attached.
- Telemetry keeps measured `wall_ms` (`is_wall_measured=True`) separate from analytic `flops`/`bytes_moved` (`is_flops_derived=True`). Never label a modeled number as measured (e.g. don't hardcode `is_measurement=True` on a toy op).
- **Why:** the standing project rule is numpy/torch-CPU may MODEL GPU semantics but must never masquerade as real hardware. `torch_backend.DigitalGPUBackend` still refuses bare hardware-named kernels; keep the two consistent.

## FP8 must be a real numerics model, not linear quant
- `precision.to_fp8` does OCP e4m3/e5m2 grid rounding (subnormals + round-half-to-even), NOT `round(x*16)/16`. The scale=16 "linear quant" toy is wrong — reject it.
- Format-specific non-finite behavior is required for fidelity: **e4m3** has no inf → finite overflow AND ±inf input both saturate to ±448; **e5m2** is IEEE-like → finite overflow rounds up to inf, ±inf preserved; nan→nan for both. Compute log2 only on finite non-zero magnitudes (mask first) or inf flows to nan.
- `.view(np.int8)` byte-reinterpretation of a float array is a bug, not fp8.

## Scheduler / graph correctness
- `ExecutionGraph.topological_order()` is real Kahn topo sort + cycle detection. `_validate()` rejects duplicate node ids, a tensor id with two producers, and self-dependency (a node consuming a tid it also produces — Kahn edge-building would otherwise silently drop it).
- OOM budget uses a true live-set: track per-tensor-id bytes and REPLACE on overwrite, not a monotonic counter (which false-positives on long graphs). Check the budget on the initial tensors too.
- Scheduler validates each node against its spec's I/O contract (required input arg names present, output names exactly match) and raises typed `ShapeMismatchError`. Symbolic dims like ("B","H","T","D") are not runtime-checked — only the op's I/O name contract is.

## Don't clobber existing infra
- Existing `digital_gpu.py` exports DigitalGPU/VRAM/SIMDCore/Scheduler/OpCode; `maxcore/backend/device_backend.py` exports the real torch `GPUBackend` used by `registry.select_backend`. Replacing either wholesale breaks package import + all tests. Multi-device already exists (`multi_backend.MultiStreamBackend`) — don't duplicate with a toy all_reduce.
- Test runner: no pytest installed → inline runner that imports test modules and calls `test_*`. Full gpu+backend suite was 53/53 after this stack.
