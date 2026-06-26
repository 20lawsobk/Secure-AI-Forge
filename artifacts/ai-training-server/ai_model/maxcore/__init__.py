"""MaxCore — the in-house software-defined GPU stack.

A backend-agnostic compute layer wrapping the existing NumPy/SIMD Digital GPU
engine: a public ``DigitalGPU`` API, a hardware-agnostic IR + graph builder, a
caching compiler with fusion, a runtime executor, sessions/KV stores, a unified
backend registry (CPU real; GPU/cluster/ASIC honest plug-points), and a PDIM
orchestrator (dedup + single-flight + durable micro-batch queue).

All additive — importing this package has no effect on the running server.
"""
from __future__ import annotations

from .api import DigitalGPU
from .backend import (
    ASICBackend,
    Backend,
    ClusterBackend,
    CPUBackend,
    GPUBackend,
    available,
    available_runtime,
    get_backend,
)
from .compiler import CompiledGraph, Compiler
from .ir import GraphBuilder, MaxCoreGraph, MaxCoreNode, OpType, TensorSpec
from .observability import METRICS, Metrics
from .pdim import PDIMConfig, PDIMOrchestrator, PDIMStorage, PDIMWorker
from .runtime import Runtime
from .session import KVStore, Session, Stream
from .tensor import Tensor, as_tensor, to_numpy

__all__ = [
    "DigitalGPU",
    "Tensor", "as_tensor", "to_numpy",
    "GraphBuilder", "MaxCoreGraph", "MaxCoreNode", "OpType", "TensorSpec",
    "Compiler", "CompiledGraph",
    "Runtime",
    "Session", "KVStore", "Stream",
    "Backend", "CPUBackend", "GPUBackend", "ClusterBackend", "ASICBackend",
    "get_backend", "register", "available", "available_runtime",
    "PDIMOrchestrator", "PDIMStorage", "PDIMWorker", "PDIMConfig",
    "METRICS", "Metrics",
]

from .backend import register  # noqa: E402  (re-export after __all__ for clarity)
