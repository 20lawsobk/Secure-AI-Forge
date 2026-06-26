"""GraphBuilder — fluent construction of a validated MaxCore IR graph.

Each op method appends a node and returns the name of its output tensor, so ops
can be chained. Inputs are checked for define-before-use at construction time,
which guarantees insertion order is a valid topological order.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .nodes import MaxCoreGraph, MaxCoreNode, OpType, TensorSpec


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: list[MaxCoreNode] = []
        self.inputs: list[str] = []
        self.consts: dict[str, np.ndarray] = {}
        self.specs: dict[str, TensorSpec] = {}
        self._produced: set[str] = set()
        self._counter = 0

    def _name(self, prefix: str = "t") -> str:
        name = f"{prefix}_{self._counter}"
        self._counter += 1
        return name

    def add_input(self, name: str, shape: tuple | None = None, dtype: str = "float32") -> str:
        self.inputs.append(name)
        self.specs[name] = TensorSpec(name, shape, dtype)
        self._produced.add(name)
        return name

    def const(self, value: Any, name: str | None = None) -> str:
        name = name or self._name("const")
        self.consts[name] = np.asarray(value)
        self._produced.add(name)
        return name

    def _op(self, op: OpType, inputs, attrs=None, out=None, name="") -> str:
        out = out or self._name()
        for i in inputs:
            if i not in self._produced:
                raise ValueError(f"op {op.value}: input '{i}' used before defined")
        self.nodes.append(MaxCoreNode(op, list(inputs), out, attrs or {}, name))
        self._produced.add(out)
        return out

    # ── ops ─────────────────────────────────────────────────────────────────
    def gemm(self, a, b, bias=None, activation=None, out=None) -> str:
        inputs = [a, b] + ([bias] if bias is not None else [])
        return self._op(OpType.GEMM, inputs,
                        {"has_bias": bias is not None, "activation": activation}, out)

    def add(self, a, b, out=None) -> str:
        return self._op(OpType.ADD, [a, b], {}, out)

    def relu(self, x, out=None) -> str:
        return self._op(OpType.RELU, [x], {}, out)

    def softmax(self, x, axis=-1, out=None) -> str:
        return self._op(OpType.SOFTMAX, [x], {"axis": axis}, out)

    def attention(self, q, k, v, mask=None, causal=False, out=None) -> str:
        inputs = [q, k, v] + ([mask] if mask is not None else [])
        return self._op(OpType.ATTENTION, inputs,
                        {"causal": causal, "has_mask": mask is not None}, out)

    def conv2d(self, x, w, bias=None, stride=1, padding=0, out=None) -> str:
        inputs = [x, w] + ([bias] if bias is not None else [])
        return self._op(OpType.CONV2D, inputs,
                        {"stride": stride, "padding": padding, "has_bias": bias is not None}, out)

    def mlp(self, x, w1, b1, w2, b2, activation="relu", out=None) -> str:
        return self._op(OpType.MLP, [x, w1, b1, w2, b2], {"activation": activation}, out)

    def reduce(self, x, op, axis, keepdims=False, out=None) -> str:
        return self._op(OpType.REDUCE, [x], {"op": op, "axis": axis, "keepdims": keepdims}, out)

    def copy(self, x, out=None) -> str:
        return self._op(OpType.COPY, [x], {}, out)

    def barrier(self, *deps) -> str:
        return self._op(OpType.BARRIER, list(deps), {})

    def build(self, outputs) -> MaxCoreGraph:
        if isinstance(outputs, str):
            outputs = [outputs]
        g = MaxCoreGraph(self.nodes, self.inputs, outputs, self.consts, self.specs)
        g.validate()
        return g
