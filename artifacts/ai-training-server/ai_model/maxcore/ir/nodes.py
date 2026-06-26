"""MaxCore IR — the hardware-agnostic intermediate representation (HLG layer).

A ``MaxCoreGraph`` is a small, validated DAG of ``MaxCoreNode`` ops over named
tensors. It is the contract between the public DigitalGPU API (which builds it),
the compiler (which optimizes it), and the runtime (which executes it on a
backend). The IR carries shapes/dtypes (``TensorSpec``), op attributes, and a
``structural_hash`` used to cache compilation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class OpType(str, Enum):
    INPUT = "input"
    CONST = "const"
    GEMM = "gemm"
    ADD = "add"
    RELU = "relu"
    SOFTMAX = "softmax"
    ATTENTION = "attention"
    CONV2D = "conv2d"
    MLP = "mlp"
    REDUCE = "reduce"
    COPY = "copy"
    BARRIER = "barrier"


@dataclass
class TensorSpec:
    """Static metadata for a named tensor (shape may contain ``None`` dims)."""

    name: str
    shape: tuple | None = None
    dtype: str = "float32"
    layout: str = "row_major"


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of attr values to a stable JSON-able form."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    return str(obj)


@dataclass
class MaxCoreNode:
    op_type: OpType
    inputs: list           # names of input tensors
    output: str            # name of the (single) output tensor
    attrs: dict = field(default_factory=dict)
    name: str = ""

    def clone(self) -> "MaxCoreNode":
        return MaxCoreNode(self.op_type, list(self.inputs), self.output, dict(self.attrs), self.name)

    def to_dict(self) -> dict:
        op = self.op_type.value if isinstance(self.op_type, OpType) else str(self.op_type)
        return {
            "op": op,
            "inputs": list(self.inputs),
            "output": self.output,
            "attrs": _jsonable(self.attrs),
            "name": self.name,
        }


class MaxCoreGraph:
    """A validated DAG of MaxCore ops over named tensors."""

    def __init__(self, nodes, inputs, outputs, consts=None, specs=None):
        self.nodes: list[MaxCoreNode] = list(nodes)
        self.inputs: list[str] = list(inputs)
        self.outputs: list[str] = list(outputs)
        self.consts: dict[str, np.ndarray] = {k: np.asarray(v) for k, v in (consts or {}).items()}
        self.specs: dict[str, TensorSpec] = dict(specs or {})

    def validate(self) -> bool:
        """Ensure every input is defined before use, outputs are unique, and
        all declared graph outputs are produced. Raises ``ValueError`` on any
        structural violation."""
        produced = set(self.inputs) | set(self.consts.keys())
        seen: set[str] = set()
        for n in self.nodes:
            for i in n.inputs:
                if i not in produced:
                    raise ValueError(
                        f"node {n.op_type}({n.name or n.output}) uses '{i}' before it is defined"
                    )
            if not n.output:
                raise ValueError(f"node {n.op_type} has no output name")
            if n.output in seen:
                raise ValueError(f"duplicate output name '{n.output}'")
            produced.add(n.output)
            seen.add(n.output)
        if not self.outputs:
            raise ValueError("graph has no outputs")
        for o in self.outputs:
            if o not in produced:
                raise ValueError(f"graph output '{o}' is never produced")
        return True

    def topo(self) -> list[MaxCoreNode]:
        """Return nodes in a valid execution order (insertion order is already
        topological because the builder enforces define-before-use)."""
        self.validate()
        return list(self.nodes)

    def structural_hash(self) -> str:
        """Stable hash of graph *structure* (ops + wiring + const shapes), used
        as the compilation cache key. Const *values* are excluded: compilation
        produces an execution plan that is independent of weight values."""
        payload = {
            "nodes": [n.to_dict() for n in self.nodes],
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "consts": {
                k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)}
                for k, v in sorted(self.consts.items())
            },
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "consts": sorted(self.consts.keys()),
        }
