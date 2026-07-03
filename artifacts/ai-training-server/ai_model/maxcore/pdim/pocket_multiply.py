"""Pocket-dimension multiplication — deduped GEMM inside one pocket.

A *pocket* is one dedup + single-flight domain on the PDIM orchestrator:
identical multiplications inside the same pocket are computed ONCE (on the
DigitalGPU backend) and every other caller — concurrent or later — receives
the stored result. Different pockets are fully isolated.

Pockets nest without limit. Because every stored payload is compressed
(zlib) before it enters the pocket, a pocket can hold arbitrarily many
sub-pockets and results: ``PocketDimension("root").pocket("a").pocket("b")``
addresses the pocket ``root/a/b``. The nesting is purely namespacing — a
child's results live inside the parent's compressed key-space and never
collide with a sibling's.

Result payloads are dicts (the dedup-cache contract): fp32 bytes are
zlib-compressed then base64-encoded, with shape/dtype/codec recorded so the
matrix can be decoded on the way out.
"""
from __future__ import annotations

import base64
import hashlib
import threading
import zlib
from typing import Any, Optional

import numpy as np

from ..observability import METRICS
from .orchestrator import PDIMOrchestrator

_CODEC = "zlib+b64"


def _digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        c = np.ascontiguousarray(a)
        h.update(str(c.dtype).encode())
        h.update(str(c.shape).encode())
        h.update(c.tobytes())
    return h.hexdigest()


def _encode(result: np.ndarray) -> dict:
    raw = np.ascontiguousarray(result, dtype=np.float32).tobytes()
    packed = zlib.compress(raw, level=1)
    return {
        "op": "matmul",
        "shape": list(result.shape),
        "dtype": "float32",
        "codec": _CODEC,
        "data": base64.b64encode(packed).decode("ascii"),
        "raw_bytes": len(raw),
        "stored_bytes": len(packed),
    }


def _decode(payload: dict) -> np.ndarray:
    if payload.get("codec") != _CODEC:
        raise ValueError(f"Unknown pocket payload codec: {payload.get('codec')!r}")
    raw = zlib.decompress(base64.b64decode(payload["data"]))
    arr = np.frombuffer(raw, dtype=np.float32)
    return arr.reshape(payload["shape"]).copy()


def _pocket_path(path: str) -> str:
    parts = [p.strip() for p in str(path).split("/") if p.strip()]
    if not parts:
        parts = ["default"]
    return "/".join(parts)


class PocketDimension:
    """One pocket: a dedup + single-flight multiplication domain.

    ``pocket(name)`` opens a sub-pocket inside this one — nest as deep as you
    want; compression keeps every level cheap to hold.
    """

    def __init__(self, path: str = "default",
                 orchestrator: Optional[PDIMOrchestrator] = None,
                 gpu: Any = None):
        self.path = _pocket_path(path)
        self._orch = orchestrator or _default_orchestrator()
        self._gpu = gpu

    # ── nesting ────────────────────────────────────────────────────────────
    def pocket(self, name: str) -> "PocketDimension":
        """Open a sub-pocket inside this pocket (unbounded depth)."""
        child = _pocket_path(name)
        return PocketDimension(f"{self.path}/{child}",
                               orchestrator=self._orch, gpu=self._gpu)

    @property
    def namespace(self) -> str:
        return f"pocket:{self.path}"

    def _backend(self):
        if self._gpu is None:
            from ..api import DigitalGPU
            self._gpu = DigitalGPU()
        return self._gpu

    # ── multiplication inside this pocket ──────────────────────────────────
    def matmul(self, A: np.ndarray, B: np.ndarray) -> dict:
        """Multiply A @ B inside this pocket.

        Identical (A, B) pairs in the same pocket are computed once on the
        DigitalGPU backend; repeats and concurrent duplicates share that one
        result. Returns ``{"result": ndarray, "source": ..., "pocket": ...,
        "compression": {...}}`` where source is compute | cache | coalesced.
        """
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        if A.ndim < 2 or B.ndim != 2:
            raise ValueError(
                f"pocket matmul expects A[..., M, K] and B[K, N]; got {A.shape} x {B.shape}")
        if A.shape[-1] != B.shape[0]:
            raise ValueError(f"Incompatible shapes: {A.shape} x {B.shape}")

        request = {
            "op": "matmul",
            "digest": _digest(A, B),
            "a_shape": list(A.shape),
            "b_shape": list(B.shape),
        }

        def _compute(_req: Any) -> dict:
            gpu = self._backend()
            lead = A.shape[:-1]
            a2 = A.reshape(-1, A.shape[-1])
            out = gpu.gemm(a2, B)
            out_np = np.asarray(out.data if hasattr(out, "data") else out,
                                dtype=np.float32)
            out_np = out_np.reshape(*lead, B.shape[1])
            METRICS.incr("pdim.pocket_matmul")
            return _encode(out_np)

        envelope = self._orch.compute(request, _compute, namespace=self.namespace)
        payload = envelope["result"]
        return {
            "result": _decode(payload),
            "source": envelope["source"],
            "pocket": self.path,
            "compression": {
                "codec": payload["codec"],
                "raw_bytes": payload["raw_bytes"],
                "stored_bytes": payload["stored_bytes"],
                "ratio": round(payload["raw_bytes"] / payload["stored_bytes"], 3)
                if payload["stored_bytes"] else 0.0,
            },
        }

    def stats(self) -> dict:
        return {"pocket": self.path, **self._orch.stats()}


# ── module-level convenience ────────────────────────────────────────────────
_shared_orch: Optional[PDIMOrchestrator] = None
_shared_lock = threading.Lock()


def _default_orchestrator() -> PDIMOrchestrator:
    global _shared_orch
    if _shared_orch is None:
        with _shared_lock:
            if _shared_orch is None:
                _shared_orch = PDIMOrchestrator()
    return _shared_orch


def pocket_matmul(A: np.ndarray, B: np.ndarray, pocket: str = "default",
                  orchestrator: Optional[PDIMOrchestrator] = None) -> dict:
    """Multiply A @ B inside the named pocket (``"a/b/c"`` nests pockets)."""
    return PocketDimension(pocket, orchestrator=orchestrator).matmul(A, B)
