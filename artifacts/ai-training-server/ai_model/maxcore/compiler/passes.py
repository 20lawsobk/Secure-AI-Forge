"""Compiler passes over the MaxCore IR.

Each pass is a pure function ``graph -> (new_graph, log)`` so the pipeline can
chain them and report what each did. Implemented passes:

- ``normalize_pass`` — canonicalize attrs / fill defaults.
- ``simplify_pass``  — dead-node elimination (drop nodes not feeding outputs).
- ``fuse_pass``      — fuse ``gemm -> add(bias) -> relu`` into a single fused
                       GEMM that maps onto the engine's ``gemm_bias_relu`` kernel.
- ``validate_pass``  — structural validation.
"""
from __future__ import annotations

from ..ir.nodes import MaxCoreGraph, MaxCoreNode, OpType


def _consumers(nodes: list[MaxCoreNode]) -> dict[str, list[MaxCoreNode]]:
    cons: dict[str, list[MaxCoreNode]] = {}
    for n in nodes:
        for i in n.inputs:
            cons.setdefault(i, []).append(n)
    return cons


def normalize_pass(graph: MaxCoreGraph):
    nodes = [n.clone() for n in graph.topo()]
    for n in nodes:
        a = n.attrs
        if a.get("activation") in ("none", "linear"):
            a["activation"] = None
        if n.op_type == OpType.GEMM:
            a.setdefault("activation", None)
            a.setdefault("has_bias", len(n.inputs) > 2)
        elif n.op_type == OpType.SOFTMAX:
            a.setdefault("axis", -1)
    g = MaxCoreGraph(nodes, graph.inputs, graph.outputs, graph.consts, graph.specs)
    return g, ["normalize"]


def simplify_pass(graph: MaxCoreGraph):
    """Dead-node elimination: keep only nodes whose outputs (transitively) feed
    a graph output. BARRIER nodes are always preserved (ordering intent)."""
    nodes = graph.topo()
    needed = set(graph.outputs)
    keep_rev: list[MaxCoreNode] = []
    for n in reversed(nodes):
        if n.output in needed or n.op_type == OpType.BARRIER:
            keep_rev.append(n)
            needed.update(n.inputs)
    keep = list(reversed(keep_rev))
    removed = len(nodes) - len(keep)
    g = MaxCoreGraph(keep, graph.inputs, graph.outputs, graph.consts, graph.specs)
    return g, [f"dce:removed={removed}"]


def fuse_pass(graph: MaxCoreGraph):
    """Fuse a GEMM with a following bias-ADD and/or RELU.

    Only fires when the GEMM's output is consumed by exactly one node and is not
    itself a graph output, so the rewrite is always semantics-preserving. Bias is
    only folded when the bias operand is a graph input or a constant, guaranteeing
    it is defined before the (earlier) GEMM node — preserving define-before-use.
    """
    nodes = [n.clone() for n in graph.topo()]
    out_set = set(graph.outputs)
    safe_operands = set(graph.inputs) | set(graph.consts.keys())
    log: list[str] = []

    changed = True
    while changed:
        changed = False
        cons = _consumers(nodes)
        for g in nodes:
            if g.op_type != OpType.GEMM or g.output in out_set:
                continue
            gc = cons.get(g.output, [])
            if len(gc) != 1:
                continue
            c = gc[0]
            if (not g.attrs.get("has_bias")) and c.op_type == OpType.ADD:
                others = [i for i in c.inputs if i != g.output]
                if len(others) == 1 and others[0] in safe_operands:
                    g.inputs = list(g.inputs) + [others[0]]
                    g.attrs["has_bias"] = True
                    g.output = c.output
                    nodes.remove(c)
                    log.append("fuse:gemm+add->gemm(bias)")
                    changed = True
                    break
            if g.attrs.get("activation") in (None, "none", "linear") and c.op_type == OpType.RELU:
                g.attrs["activation"] = "relu"
                g.output = c.output
                nodes.remove(c)
                log.append("fuse:gemm+relu->gemm(relu)")
                changed = True
                break

    out = MaxCoreGraph(nodes, graph.inputs, graph.outputs, graph.consts, graph.specs)
    return out, log


def validate_pass(graph: MaxCoreGraph):
    graph.validate()
    return graph, ["validate:ok"]


DEFAULT_PASSES = [normalize_pass, simplify_pass, fuse_pass, validate_pass]
