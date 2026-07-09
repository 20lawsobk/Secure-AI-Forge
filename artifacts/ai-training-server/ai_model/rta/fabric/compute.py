"""RTA-1 numeric compute — routed through the self-contained Digital GPU.

Every heavy batched linear-algebra operation in RTA-1 (the path tracer's
ray/shading dot products, the colour-grade matrix, the audio DFT/iDFT) goes
through :meth:`RTACompute.gemm`, which dispatches to the project's own
``DigitalGPU`` SIMD core. This keeps RTA-1 on the same self-contained compute
fabric the rest of the system uses — never raw BLAS, never an external GPU.

A process-wide op counter (``global_op_counts``) records how many GEMMs actually
went through the Digital GPU, so tests/probes can *prove* the fabric is on the
self-contained path rather than silently falling back to ``numpy.matmul``.
"""
from __future__ import annotations

import threading
from typing import Dict

import numpy as np

_ops_lock = threading.Lock()
_GLOBAL_OPS: Dict[str, int] = {"gemm": 0}


def global_op_counts() -> Dict[str, int]:
    """Snapshot of Digital-GPU ops issued by RTA-1 across the process."""
    with _ops_lock:
        return dict(_GLOBAL_OPS)


def _bump(kind: str, n: int = 1) -> None:
    with _ops_lock:
        _GLOBAL_OPS[kind] = _GLOBAL_OPS.get(kind, 0) + n


class RTACompute:
    """Compute context bound to one render.

    Holds its own ``DigitalGPU`` instance (so concurrent renders never share
    VRAM state), and clears VRAM after each op so a long render issuing
    thousands of GEMMs can't grow memory without bound.
    """

    def __init__(self, lanes: int = 32):
        # Import here so the fabric module has no hard import-time dependency on
        # the GPU stack (keeps unit-importing the graph/media cheap).
        from ai_model.gpu.digital_gpu import DigitalGPU
        self._gpu = DigitalGPU(lanes=lanes)
        self.ops = 0

    def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """2-D matrix multiply on the Digital GPU. Returns float32 ``a @ b``."""
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"RTACompute.gemm expects 2-D matrices, got {a.shape} @ {b.shape}")
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"incompatible GEMM shapes: {a.shape} @ {b.shape}")
        out = self._gpu.gemm(a, b)
        # DigitalGPU.gemm allocates persistent VRAM handles per call — release
        # them so a render issuing thousands of GEMMs stays bounded.
        self._gpu.vram._store.clear()
        self._gpu.vram._meta.clear()
        self.ops += 1
        _bump("gemm")
        return np.asarray(out, dtype=np.float32)

    def matvec_rows(self, vecs: np.ndarray, mat3: np.ndarray) -> np.ndarray:
        """Transform each row-vector in ``vecs`` [N,K] by ``mat3`` [K,M] via GEMM."""
        return self.gemm(vecs, mat3)
