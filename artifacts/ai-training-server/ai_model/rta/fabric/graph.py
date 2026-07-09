"""UMRF node graph — a real DAG with topological scheduling + cycle detection.

Mirrors the RTA-1 spec's ``Node`` / ``NodeGraph`` shape. Edges express data
dependencies (``from`` must run before ``to``). ``topological_sort`` returns a
concrete execution order (Kahn's algorithm) and raises on a cycle — no
"omitted for brevity" placeholder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


class GraphCycleError(ValueError):
    """Raised when the node graph contains a cycle and cannot be scheduled."""


@dataclass
class Node:
    id: str
    type: str
    params: dict = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)     # upstream node ids
    outputs: List[str] = field(default_factory=list)    # downstream node ids


@dataclass
class NodeGraph:
    id: str
    nodes: List[Node] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)

    def node_map(self) -> Dict[str, Node]:
        return {n.id: n for n in self.nodes}


def _all_edges(graph: NodeGraph) -> List[Tuple[str, str]]:
    """Union of explicit ``edges`` and edges implied by each node's ``inputs``."""
    edges = list(graph.edges)
    ids = {n.id for n in graph.nodes}
    for n in graph.nodes:
        for src in n.inputs:
            if (src, n.id) not in edges:
                edges.append((src, n.id))
    # Validate references
    for src, dst in edges:
        if src not in ids or dst not in ids:
            raise ValueError(f"edge references unknown node: ({src} -> {dst})")
    return edges


def topological_sort(graph: NodeGraph) -> List[Node]:
    """Return nodes in a valid execution order. Raises ``GraphCycleError`` on a cycle."""
    nmap = graph.node_map()
    if len(nmap) != len(graph.nodes):
        raise ValueError("duplicate node ids in graph")

    edges = _all_edges(graph)
    indeg: Dict[str, int] = {nid: 0 for nid in nmap}
    adj: Dict[str, List[str]] = {nid: [] for nid in nmap}
    for src, dst in edges:
        adj[src].append(dst)
        indeg[dst] += 1

    # Deterministic ordering: process ready nodes in their declared order.
    order_index = {n.id: i for i, n in enumerate(graph.nodes)}
    ready = sorted([nid for nid, d in indeg.items() if d == 0], key=lambda x: order_index[x])
    out: List[Node] = []
    while ready:
        nid = ready.pop(0)
        out.append(nmap[nid])
        newly: List[str] = []
        for nxt in adj[nid]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                newly.append(nxt)
        if newly:
            ready.extend(sorted(newly, key=lambda x: order_index[x]))
            ready.sort(key=lambda x: order_index[x])

    if len(out) != len(graph.nodes):
        remaining = [nid for nid in nmap if nmap[nid] not in out]
        raise GraphCycleError(f"node graph has a cycle; unscheduled: {remaining}")
    return out
