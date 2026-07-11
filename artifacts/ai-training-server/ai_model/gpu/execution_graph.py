"""Execution graph (DAG) + graph scheduler for the digital GPU.

``ExecutionGraph`` is a dataflow DAG of :class:`Node`s connected by named tensor
ids. ``DigitalScheduler`` runs it: it topologically orders the nodes (real
dependency ordering with cycle detection — not just insertion order), validates
each opcode against the :mod:`ai_model.gpu.opcode_spec` contract, executes it on a
:class:`~ai_model.gpu.digital_gpu.DigitalGPU`, and records honest telemetry.

Honesty enforcement: a node whose spec sets ``is_hardware_execution=True`` is
refused unless a real hardware backend is wired in — on this CPU host nothing
claims hardware, so nothing is silently run as if it were.

FLOP derivation
---------------
Every ``OpcodeSpec`` optionally carries a ``flop_formula`` callable. ``_derived_flops``
delegates to it (never raises) and returns a ``(flops, flops_unknown)`` pair.
Unknown FLOPs are surfaced as ``flops_unknown=True`` in ``OpRecord`` and as an
incrementing ``_flop_derive_errors`` counter on the scheduler — observable via
``last_profile()`` — rather than being silently swallowed. Only the specific
exceptions that a formula can legitimately raise are caught; everything else
propagates so real bugs aren't hidden.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ai_model.gpu.digital_gpu import (
    DigitalGPU, InvalidOpcodeError, OOMError, GPUError, ShapeMismatchError,
)
from ai_model.gpu.opcode_spec import OpcodeSpec, get_spec
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
    """Build the opcode-name → handler map.

    Each handler follows the convention ``fn(gpu, ins, params) -> dict[str, ndarray]``
    and uses duck-typing: it calls the GPU's native method when available, then
    falls back to a pure-numpy implementation. This means the scheduler works
    correctly with any DigitalGPU subclass (including HyperGPU) without hard
    dependencies in either direction.
    """
    # ── base ops (DigitalGPU level) ──────────────────────────────────────────

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
        # Numerics MODEL — see precision.flash_attention_fp8_model docstring.
        return {"O": precision.flash_attention_fp8_model(
            i["Q"], i["K"], i["V"], causal=p.get("causal", False))}

    # ── HyperGPU / HyperSIMDCore ops ─────────────────────────────────────────
    # Each handler tries the GPU's native method first (duck-typed), then falls
    # back to a standalone numpy implementation with the same numeric contract.
    # Adding a new op = add one function here + one entry in the return dict.

    def layer_norm(gpu, i, p):
        X, gamma, beta = i["X"], i["gamma"], i["beta"]
        eps = p.get("eps", 1e-5)
        if hasattr(gpu, "layer_norm"):
            return {"Y": gpu.layer_norm(X, gamma, beta, eps=eps)}
        # Standalone numpy fallback (normalise over last axis)
        mean = X.mean(axis=-1, keepdims=True)
        var = X.var(axis=-1, keepdims=True)
        return {"Y": gamma * ((X - mean) / np.sqrt(var + eps)) + beta}

    def batch_norm(gpu, i, p):
        X, gamma, beta = i["X"], i["gamma"], i["beta"]
        eps = p.get("eps", 1e-5)
        training = p.get("training", True)
        if hasattr(gpu, "batch_norm"):
            out, _, _ = gpu.batch_norm(X, gamma, beta, training=training, eps=eps)
            return {"Y": out}
        # Numpy training-mode fallback (reduce over all axes except channel=1)
        reduce_axes = tuple(j for j in range(X.ndim) if j != 1) if X.ndim > 1 else (0,)
        mean = X.mean(axis=reduce_axes, keepdims=True)
        var = X.var(axis=reduce_axes, keepdims=True)
        g_shape = [1] * X.ndim
        g_shape[1 if X.ndim > 1 else 0] = -1
        Y = (gamma.reshape(g_shape) * ((X - mean) / np.sqrt(var + eps))
             + beta.reshape(g_shape))
        return {"Y": Y}

    def gelu(gpu, i, p):
        X = i["X"]
        if hasattr(gpu, "gelu"):
            return {"Y": gpu.gelu(X)}
        # tanh approximation (Hendrycks & Gimpel 2016)
        return {"Y": 0.5 * X * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (X + 0.044715 * X ** 3)))}

    def silu(gpu, i, p):
        X = i["X"]
        if hasattr(gpu, "silu"):
            return {"Y": gpu.silu(X)}
        return {"Y": X / (1.0 + np.exp(-X))}

    def relu(gpu, i, p):
        X = i["X"]
        if hasattr(gpu, "relu"):
            return {"Y": gpu.relu(X)}
        return {"Y": np.maximum(X, 0.0)}

    def conv3d(gpu, i, p):
        X, W = i["X"], i["W"]
        stride = p.get("stride", (1, 1, 1))
        padding = p.get("padding", (0, 0, 0))
        if hasattr(gpu, "conv3d"):
            return {"Y": gpu.conv3d(X, W, stride=stride, padding=padding)}
        # Lazy import so DigitalScheduler doesn't hard-depend on HyperSIMDCore
        from ai_model.gpu.hyper_core import HyperSIMDCore
        return {"Y": HyperSIMDCore(lanes=1).conv3d(X, W, stride=stride,
                                                    padding=padding)}

    def gemm_batched(gpu, i, p):
        A, B = i["A"], i["B"]
        if hasattr(gpu, "gemm_batched"):
            return {"C": gpu.gemm_batched(A, B)}
        return {"C": np.matmul(A.astype(np.float64, copy=False),
                               B.astype(np.float64, copy=False))}

    def grouped_gemm(gpu, i, p):
        # Inputs are stacked [G, M, K] / [G, K, N]; split on leading dim.
        A, B = i["A"], i["B"]
        G = A.shape[0]
        if hasattr(gpu, "grouped_gemm"):
            results = gpu.grouped_gemm(
                [A[g] for g in range(G)], [B[g] for g in range(G)])
            return {"C": np.stack(results, axis=0)}
        return {"C": np.matmul(A.astype(np.float64, copy=False),
                               B.astype(np.float64, copy=False))}

    def mixed_precision_gemm(gpu, i, p):
        A, B = i["A"], i["B"]
        if hasattr(gpu, "mixed_gemm"):
            return {"C": gpu.mixed_gemm(A, B)}
        return {"C": np.matmul(A.astype(np.float64, copy=False),
                               B.astype(np.float64, copy=False))}

    def fused_attention_norm(gpu, i, p):
        Q, K, V = i["Q"], i["K"], i["V"]
        gamma, beta = i["gamma"], i["beta"]
        causal = p.get("causal", False)
        eps = p.get("eps", 1e-5)
        if hasattr(gpu, "fused_attention_norm"):
            return {"O": gpu.fused_attention_norm(
                Q, K, V, gamma, beta, causal=causal, eps=eps)}
        from ai_model.gpu.hyper_core import HyperSIMDCore
        return {"O": HyperSIMDCore(lanes=1).fused_attention_norm(
            Q, K, V, gamma, beta, causal=causal, eps=eps)}

    return {
        # ── base set ──────────────────────────────────────────────────────────
        "gemm": gemm, "add": add, "softmax": softmax,
        "attention": attention, "conv2d": conv2d,
        "flash_attention_fp8_sm102": flash_attention_fp8_sm102,
        # ── HyperGPU / HyperSIMDCore set ─────────────────────────────────────
        "layer_norm": layer_norm, "batch_norm": batch_norm,
        "gelu": gelu, "silu": silu, "relu": relu,
        "conv3d": conv3d,
        "gemm_batched": gemm_batched, "grouped_gemm": grouped_gemm,
        "mixed_precision_gemm": mixed_precision_gemm,
        "fused_attention_norm": fused_attention_norm,
    }


def _derived_flops(spec: OpcodeSpec,
                   ins: Dict[str, np.ndarray]) -> tuple:
    """Derive analytic FLOP count from the opcode spec's ``flop_formula``.

    Returns ``(flops: float, flops_unknown: bool)``.

    * ``flops_unknown=False`` — formula returned a valid count.
    * ``flops_unknown=True``  — formula was missing or raised a known exception;
      ``flops`` is 0.0. The caller increments an observable counter rather than
      silently discarding the event.

    Only ``KeyError``, ``TypeError``, ``AttributeError``, and ``IndexError`` are
    caught — the specific exceptions a shape-driven formula can legitimately raise
    from bad or incomplete input dicts. Anything else (programming error, OOM,
    etc.) propagates so real bugs are visible. This follows the OpenTelemetry SDK
    guidance: "minimize the scope of error handlers; add special processing for
    expected exceptions."
    """
    formula = spec.flop_formula
    if formula is None:
        return 0.0, True
    try:
        return float(formula(ins)), False
    except (KeyError, TypeError, AttributeError, IndexError):
        return 0.0, True


class DigitalScheduler:
    """Runs an :class:`ExecutionGraph` on a DigitalGPU with contract checks."""

    def __init__(self, gpu: Optional[DigitalGPU] = None,
                 telemetry: Optional[Telemetry] = None,
                 max_bytes: Optional[int] = None):
        self.gpu = gpu or DigitalGPU()
        self.telemetry = telemetry
        self.max_bytes = max_bytes
        self._dispatch = _default_dispatch()
        # Observable counter for ops whose flop_formula was absent or raised.
        # Surfaced in last_profile() so coverage gaps are detectable without
        # the telemetry layer ever crashing (OpenTelemetry never-raise pattern).
        self._flop_derive_errors: int = 0

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

    def last_profile(self) -> dict:
        """Return a summary of the scheduler's last run, including FLOP derivation
        error count so callers can detect missing formula coverage."""
        return {
            "flop_derive_errors": self._flop_derive_errors,
            "telemetry": self.telemetry.summary() if self.telemetry else None,
        }

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
                _flops, _flops_unknown = _derived_flops(spec, ins)
                if _flops_unknown:
                    self._flop_derive_errors += 1
                # bytes_moved = inputs + outputs = total DRAM traffic (roofline
                # convention: read + write, not just read).
                bytes_in = sum(np.asarray(v).nbytes for v in ins.values())
                bytes_out = sum(np.asarray(v).nbytes for v in outs.values())
                self.telemetry.record(
                    opcode=spec.key,
                    numeric_profile=spec.numeric_profile,
                    wall_ms=(t1 - t0) * 1000.0,
                    flops=_flops,
                    flops_unknown=_flops_unknown,
                    bytes_moved=bytes_in + bytes_out,
                )
        return tensors
