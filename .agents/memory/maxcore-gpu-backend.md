---
name: MaxCore real GPU backend
description: The "gpu" MaxCore backend is a real torch device backend, not a stub; how it stays honest and matches CPUBackend.
---

# MaxCore "gpu" backend is real (torch), gated on actual hardware

The MaxCore backend registry (`ai_model/maxcore/backend/`) exposes named backends
`cpu`/`gpu`/`cluster`/`asic` behind one `Backend` ABC contract. `cpu` runs; the
`gpu` backend is a **genuine torch implementation** (`device_backend.py`,
`GPUBackend`) that dispatches the kernel contract to a real torch device —
**not** a `NotImplementedError` stub. `cluster`/`asic` remain honest stubs
(`future_backends.py`).

**Why:** the codebase follows a "no fake hardware" rule. A GPU backend must
either run on a real GPU or say it can't — it must never silently compute on the
CPU and report itself as a GPU. So:
- `is_available()` returns `torch.cuda.is_available()` for the default
  `device="cuda"` (→ `False` on this CPU-only host); `True` only for an explicit
  `device="cpu"`; `False` for any other device string (truthful reporting).
- Kernels raise a clear `RuntimeError` naming the missing CUDA device when no GPU
  is present — no CPU masquerade.
- `device="cpu"` runs the *identical* code path on torch-CPU, which is how the
  GPU kernels are validated against `CPUBackend` without a GPU present.

**How to apply:**
- To actually run on a GPU, deploy on a CUDA host; the same code executes on the
  device with numerics matching `CPUBackend` within ~1e-3. This is the only lever
  that changes real throughput — everything else in the stack is CPU/numpy.
- There is an honesty guardrail test in `tests/test_maxcore.py`. If you change a
  backend from stub→real, keep the guardrail: assert unavailable backends aren't
  runnable and raise clearly, rather than deleting the check.
- **torch parity gotcha:** `torch.prod`/reductions take a single int `dim`, but
  numpy/`CPUBackend.reduce` accept tuple/list axes. Multi-axis `prod` must be
  folded into sequential reductions (reduce the highest axis first so remaining
  indices stay valid) or it diverges from the CPU backend. Other ops use
  `amax`/`amin`/`sum`/`mean` which accept tuples directly.
