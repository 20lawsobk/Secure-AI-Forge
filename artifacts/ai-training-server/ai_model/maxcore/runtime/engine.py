"""Runtime — execute a CompiledGraph on a backend.

Walks the compiled topological order, binds consts and inputs, dispatches each
IR op to the backend, and returns the requested outputs. Timing for the whole
run and per-op counts go to the metrics registry. Deterministic mode seeds NumPy
so any stochastic op is reproducible.
"""
from __future__ import annotations

import numpy as np

from ..backend.registry import get_backend
from ..compiler.pipeline import CompiledGraph
from ..ir.nodes import OpType
from ..observability import METRICS
from ..tensor import Tensor


class Runtime:
    def __init__(self, backend=None):
        self.backend = backend if backend is not None else get_backend("digital_gpu")

    def run(self, compiled: CompiledGraph, inputs: dict, deterministic: bool = False,
            seed: int = 0) -> dict:
        if deterministic:
            np.random.seed(seed)
        graph = compiled.graph
        env: dict = {}
        for name, value in graph.consts.items():
            env[name] = self.backend.create_tensor(value)
        for name in graph.inputs:
            if name not in inputs:
                raise ValueError(f"missing required input '{name}'")
            val = inputs[name]
            env[name] = val if isinstance(val, Tensor) else self.backend.create_tensor(np.asarray(val))
        with METRICS.timer("runtime.run"):
            for node in compiled.order:
                env[node.output] = self._exec(node, env)
        METRICS.incr("runtime.graphs")
        return {o: env[o] for o in graph.outputs}

    def _exec(self, node, env):
        b = self.backend
        op = node.op_type
        a = node.attrs
        ins = [env[i] for i in node.inputs]
        if op == OpType.GEMM:
            bias = ins[2] if a.get("has_bias") and len(ins) > 2 else None
            return b.gemm(ins[0], ins[1], bias=bias, activation=a.get("activation"))
        if op == OpType.ADD:
            return b.add(ins[0], ins[1])
        if op == OpType.RELU:
            return b.relu(ins[0])
        if op == OpType.SOFTMAX:
            return b.softmax(ins[0], axis=a.get("axis", -1))
        if op == OpType.ATTENTION:
            mask = ins[3] if a.get("has_mask") and len(ins) > 3 else None
            return b.attention(ins[0], ins[1], ins[2], mask=mask, causal=a.get("causal", False))
        if op == OpType.CONV2D:
            bias = ins[2] if a.get("has_bias") and len(ins) > 2 else None
            return b.conv2d(ins[0], ins[1], bias=bias,
                            stride=a.get("stride", 1), padding=a.get("padding", 0))
        if op == OpType.MLP:
            return b.mlp(ins[0], ins[1], ins[2], ins[3], ins[4],
                         activation=a.get("activation", "relu"))
        if op == OpType.REDUCE:
            return b.reduce(ins[0], a["op"], a["axis"], a.get("keepdims", False))
        if op == OpType.COPY:
            return ins[0]
        if op == OpType.BARRIER:
            return ins[0] if ins else None
        raise ValueError(f"runtime: unknown op '{op}'")
