"""MaxCore — the in-house software-defined GPU stack.

A backend-agnostic compute layer wrapping the NumPy/SIMD Digital GPU engine:
a public ``DigitalGPU`` API, a hardware-agnostic IR + graph builder, a caching
compiler with fusion, a runtime executor, sessions/KV stores, a unified backend
registry (DigitalGPU primary; GPU/cluster/ASIC honest plug-points), and a PDIM
orchestrator (dedup + single-flight + durable micro-batch queue).

All additive — importing this package has no effect on the running server.
"""
from __future__ import annotations

from .api import DigitalGPU
from .backend import (
    ASICBackend,
    Backend,
    ClusterBackend,
    CPUBackend,           # backwards-compatible alias for DigitalGPUBackend
    DigitalGPUBackend,
    GPUBackend,
    available,
    available_runtime,
    get_backend,
)
from .compiler import CompiledGraph, Compiler
from .hardware import configure_blas_threads, cpu_count, plan_blas_threads
from .ir import GraphBuilder, MaxCoreGraph, MaxCoreNode, OpType, TensorSpec
from .observability import METRICS, Metrics
from .pdim import PDIMConfig, PDIMOrchestrator, PDIMStorage, PDIMWorker
from .precision import (
    calibrate_scale,
    dequantize,
    dequantize_per_channel,
    quantize,
    quantize_per_channel,
    quantized_matmul,
    quantized_matmul_per_channel,
    relative_error,
)
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
    "Backend", "DigitalGPUBackend", "CPUBackend", "GPUBackend", "ClusterBackend", "ASICBackend",
    "get_backend", "register", "available", "available_runtime",
    "PDIMOrchestrator", "PDIMStorage", "PDIMWorker", "PDIMConfig",
    "METRICS", "Metrics",
    "calibrate_scale", "quantize", "dequantize",
    "quantize_per_channel", "dequantize_per_channel",
    "quantized_matmul", "quantized_matmul_per_channel", "relative_error",
    "cpu_count", "plan_blas_threads", "configure_blas_threads",
]

from .backend import register  # noqa: E402  (re-export after __all__ for clarity)
