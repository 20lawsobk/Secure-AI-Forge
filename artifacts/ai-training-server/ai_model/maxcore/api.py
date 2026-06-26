"""DigitalGPU — the public API façade.

This is the single entry point model/training code calls. It exposes eager
primitives (gemm/attention/conv2d/mlp/reduce/softmax), graph construction +
compilation + execution, sessions, and a metrics snapshot — all backed by a
pluggable backend (default: the in-house CPU engine).
"""
from __future__ import annotations

from typing import Any

from .backend.base import Backend
from .backend.registry import available, available_runtime, get_backend
from .compiler.pipeline import CompiledGraph, Compiler
from .ir.builder import GraphBuilder
from .ir.nodes import MaxCoreGraph
from .observability import METRICS
from .runtime.engine import Runtime
from .session import Session
from .tensor import Tensor


class DigitalGPU:
    def __init__(self, backend: str | Backend = "cpu", deterministic: bool = False,
                 compiler: Compiler | None = None, **backend_kwargs):
        self.backend: Backend = (
            backend if isinstance(backend, Backend) else get_backend(backend, **backend_kwargs)
        )
        self.compiler = compiler or Compiler()
        self.runtime = Runtime(self.backend)
        self.deterministic = deterministic

    # ── tensors / graph construction ─────────────────────────────────────────
    def tensor(self, data: Any, dtype: str = "float32", device: str = "cpu") -> Tensor:
        return self.backend.create_tensor(data, dtype=dtype)

    def graph_builder(self) -> GraphBuilder:
        return GraphBuilder()

    # ── eager primitives ─────────────────────────────────────────────────────
    def gemm(self, a, b, bias=None, activation=None):
        return self.backend.gemm(a, b, bias=bias, activation=activation)

    def attention(self, q, k, v, mask=None, causal=False):
        return self.backend.attention(q, k, v, mask=mask, causal=causal)

    def conv2d(self, x, w, bias=None, stride=1, padding=0):
        return self.backend.conv2d(x, w, bias=bias, stride=stride, padding=padding)

    def mlp(self, x, w1, b1, w2, b2, activation="relu"):
        return self.backend.mlp(x, w1, b1, w2, b2, activation=activation)

    def reduce(self, x, op, axis, keepdims=False):
        return self.backend.reduce(x, op, axis, keepdims=keepdims)

    def softmax(self, x, axis=-1):
        return self.backend.softmax(x, axis=axis)

    # ── graph compile / execute ──────────────────────────────────────────────
    def compile(self, graph: MaxCoreGraph) -> CompiledGraph:
        return self.compiler.compile(graph)

    def run_graph(self, graph, inputs: dict, run_config: dict | None = None) -> dict:
        rc = run_config or {}
        compiled = graph if isinstance(graph, CompiledGraph) else self.compiler.compile(graph)
        return self.runtime.run(
            compiled, inputs,
            deterministic=rc.get("deterministic", self.deterministic),
            seed=rc.get("seed", 0),
        )

    # ── sessions / introspection ─────────────────────────────────────────────
    def create_session(self, session_id: str, policy: dict | None = None) -> Session:
        return Session(session_id, self, policy)

    def metrics(self) -> dict:
        return METRICS.snapshot()

    def backends(self) -> dict:
        return {"registered": available(), "runnable": available_runtime(),
                "active": self.backend.name}
