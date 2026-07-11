"""Execution graph (DAG) + graph scheduler for the digital GPU.

``ExecutionGraph`` is a dataflow DAG of :class:`Node`s connected by named tensor
ids. ``DigitalScheduler`` runs it: it topologically orders the nodes (real
dependency ordering with cycle detection — not just insertion order), validates
each opcode against the :mod:`ai_model.gpu.opcode_spec` contract, executes it on a
:class:`~ai_model.gpu.digital_gpu.DigitalGPU`, and records honest telemetry.

Honesty enforcement: a node whose spec sets ``is_hardware_execution=True`` is
refused unless a real hardware backend is wired in — on this CPU host nothing
claims hardware, so nothing is silently run as if it were.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ai_model.gpu.digital_gpu import (
    DigitalGPU, InvalidOpcodeError, OOMError, GPUError, ShapeMismatchError,
)
from ai_model.gpu.opcode_spec import get_spec
from ai_model.gpu import precision
from ai_model.gpu.telemetry import Telemetry


@dataclass
class Node:
    id: str
    opcode: str                                  # "gemm" or "gemm:v1"
    inputs: Dict[str, str]                        # arg name -> tensor id
    outputs: Dict[str, str]                       # result name -> tensor id
    params: Dict = field(default_factory=dict)    # e.g. {"causal": True}
    stream: int = 0


@dataclass
class ExecutionGraph:
    nodes: List[Node] = field(default_factory=list)

    def add_node(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def _validate(self) -> Dict[str, Node]:
        """Structural checks before scheduling. Returns the producer map.

        Rejects: duplicate node ids, a tensor id produced by more than one node,
        and a node that consumes a tensor id it also produces (self-dependency,
        which Kahn's edge-building would otherwise silently drop)."""
        seen_ids = set()
        producer: Dict[str, Node] = {}
        for n in self.nodes:
            if n.id in seen_ids:
                raise GPUError(f"duplicate node id {n.id!r} in execution graph")
            seen_ids.add(n.id)
            shared = set(n.inputs.values()) & set(n.outputs.values())
            if shared:
                raise GPUError(
                    f"node {n.id!r} consumes and produces the same tensor(s) "
                    f"{sorted(shared)} (self-dependency)")
            for tid in n.outputs.values():
                if tid in producer:
                    raise GPUError(
                        f"tensor {tid!r} is produced by multiple nodes "
                        f"({producer[tid].id!r} and {n.id!r})")
                producer[tid] = n
        return producer

    def topological_order(self) -> List[Node]:
        """Kahn's algorithm over producer→consumer edges, with cycle detection."""
        producer = self._validate()

        indeg: Dict[str, int] = {n.id: 0 for n in self.nodes}
        adj: Dict[str, List[str]] = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            deps = set()
            for tid in n.inputs.values():
                p = producer.get(tid)
                if p is not None and p.id != n.id:
                    deps.add(p.id)
            for d in deps:
                adj[d].append(n.id)
                indeg[n.id] += 1

        by_id = {n.id: n for n in self.nodes}
        ready = [nid for nid, d in indeg.items() if d == 0]
        order: List[Node] = []
        while ready:
            nid = ready.pop(0)
            order.append(by_id[nid])
            for m in adj[nid]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
        if len(order) != len(self.nodes):
            raise GPUError("execution graph has a cycle; cannot schedule")
        return order


# ── opcode dispatch: name -> fn(gpu, inputs, params) -> {out_name: array} ──────
def _default_dispatch() -> Dict[str, Callable]:
    def gemm(gpu, i, p):
        return {"C": gpu.gemm(i["A"], i["B"])}

    def add(gpu, i, p):
        return {"C": gpu.add(i["A"], i["B"])}

    def softmax(gpu, i, p):
        return {"Y": gpu.softmax(i["X"], axis=p.get("axis", -1))}

    def attention(gpu, i, p):
        return {"O": gpu.attention(i["Q"], i["K"], i["V"],
                                   causal=p.get("causal", False))}

    def conv2d(gpu, i, p):
        from ai_model.gpu.digital_library import DigitalDNN
        y = DigitalDNN(gpu).conv2d(i["X"], i["W"], bias=p.get("bias"),
                                   stride=p.get("stride", 1),
                                   padding=p.get("padding", 0))
        return {"Y": y}

    def flash_attention_fp8_sm102(gpu, i, p):
        # numerics MODEL — see precision.flash_attention_fp8_model docstring.
        return {"O": precision.flash_attention_fp8_model(
            i["Q"], i["K"], i["V"], causal=p.get("causal", False))}

    return {
        "gemm": gemm, "add": add, "softmax": softmax, "attention": attention,
        "conv2d": conv2d,
        "flash_attention_fp8_sm102": flash_attention_fp8_sm102,
    }


def _derived_flops(name: str, ins: Dict[str, np.ndarray]) -> float:
    """Analytic FLOP estimate from shapes (NOT measured)."""
    try:
        if name == "gemm":
            M, K = ins["A"].shape
            N = ins["B"].shape[1]
            return 2.0 * M * K * N
        if name in ("attention", "flash_attention_fp8_sm102"):
            Q = ins["Q"]
            *b, T, D = Q.shape
            batch = float(np.prod(b)) if b else 1.0
            return 4.0 * batch * T * T * D
        if name == "add":
            return float(ins["A"].size)
        if name == "softmax":
            return 5.0 * float(ins["X"].size)
        if name == "conv2d":
            X, W = ins["X"], ins["W"]
            F = W.shape[0]
            return 2.0 * X.shape[0] * F * W.shape[1] * W.shape[2] * W.shape[3] * \
                X.shape[2] * X.shape[3]
    except Exception:
        return 0.0
    return 0.0


class DigitalScheduler:
    """Runs an :class:`ExecutionGraph` on a DigitalGPU with contract checks."""

    def __init__(self, gpu: Optional[DigitalGPU] = None,
                 telemetry: Optional[Telemetry] = None,
                 max_bytes: Optional[int] = None):
        self.gpu = gpu or DigitalGPU()
        self.telemetry = telemetry
        self.max_bytes = max_bytes
        self._dispatch = _default_dispatch()

    def _check_contract(self, spec, node: Node) -> None:
        """Validate the node against the opcode spec: required input arg names are
        present and its output names are exactly those the spec declares. (Symbolic
        dims like ("B","H","T","D") are not runtime-checked — only the op's I/O
        contract is.) Raises the typed ShapeMismatchError on violation."""
        missing = [k for k in spec.inputs if k not in node.inputs]
        if missing:
            raise ShapeMismatchError(
                f"{spec.key}: node {node.id!r} missing required inputs {missing}; "
                f"got {sorted(node.inputs)}")
        expected_out = set(spec.output_shapes)
        got_out = set(node.outputs)
        if got_out != expected_out:
            raise ShapeMismatchError(
                f"{spec.key}: node {node.id!r} outputs {sorted(got_out)} but spec "
                f"declares {sorted(expected_out)}")

    def run(self, graph: ExecutionGraph, tensors: Dict[str, np.ndarray]
            ) -> Dict[str, np.ndarray]:
        tensors = dict(tensors)
        # Track per-tensor bytes so overwrites replace (not add) — a true live-set,
        # not a monotonic counter that would false-positive on long graphs.
        sizes: Dict[str, int] = {tid: int(np.asarray(v).nbytes)
                                 for tid, v in tensors.items()}

        def _check_budget(where: str) -> None:
            if self.max_bytes is not None:
                live = sum(sizes.values())
                if live > self.max_bytes:
                    raise OOMError(
                        f"digital VRAM budget exceeded: {live} > "
                        f"{self.max_bytes} bytes {where}")

        _check_budget("for the initial tensors")

        for node in graph.topological_order():
            spec = get_spec(node.opcode)          # -> InvalidOpcodeError if unknown
            if spec.is_hardware_execution:
                raise GPUError(
                    f"{spec.key} is marked is_hardware_execution=True but no real "
                    f"hardware backend is attached; refusing to run it on CPU.")
            fn = self._dispatch.get(spec.name)
            if fn is None:
                raise InvalidOpcodeError(
                    f"no executor registered for opcode {spec.name!r}")
            self._check_contract(spec, node)

            ins = {k: tensors[v] for k, v in node.inputs.items()}
            t0 = time.perf_counter()
            outs = fn(self.gpu, ins, node.params)
            t1 = time.perf_counter()

            for name, tid in node.outputs.items():
                arr = outs[name]
                tensors[tid] = arr
                sizes[tid] = int(np.asarray(arr).nbytes)   # replace-on-write
            _check_budget(f"after {spec.key}")

            if self.telemetry is not None:
                self.telemetry.record(
                    opcode=spec.key, numeric_profile=spec.numeric_profile,
                    wall_ms=(t1 - t0) * 1000.0,
                    flops=_derived_flops(spec.name, ins),
                    bytes_moved=sum(np.asarray(v).nbytes for v in ins.values()),
                )
        return tensors
