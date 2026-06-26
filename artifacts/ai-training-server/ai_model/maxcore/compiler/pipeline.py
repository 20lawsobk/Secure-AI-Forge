"""Compiler pipeline — run IR passes and cache the compiled result.

Compilation is pure structure transformation, so results are cached by the
graph's ``structural_hash``. A ``CompiledGraph`` bundles the optimized graph, a
ready-to-execute topological order, the pass log, and the source hash.
"""
from __future__ import annotations

from ..ir.nodes import MaxCoreGraph
from ..observability import METRICS
from .passes import DEFAULT_PASSES


class CompiledGraph:
    def __init__(self, graph: MaxCoreGraph, order, pass_log, source_hash: str):
        self.graph = graph
        self.order = order
        self.pass_log = list(pass_log)
        self.source_hash = source_hash

    def __repr__(self) -> str:
        return (f"CompiledGraph(nodes={len(self.order)}, "
                f"passes={self.pass_log}, hash={self.source_hash[:12]})")


class Compiler:
    def __init__(self, passes=None, cache: bool = True):
        self.passes = list(passes) if passes is not None else list(DEFAULT_PASSES)
        self.cache = cache
        self._cache: dict[str, CompiledGraph] = {}

    def compile(self, graph: MaxCoreGraph) -> CompiledGraph:
        src = graph.structural_hash()
        if self.cache and src in self._cache:
            METRICS.incr("compiler.cache_hit")
            return self._cache[src]
        METRICS.incr("compiler.cache_miss")
        log: list[str] = []
        g = graph
        with METRICS.timer("compiler.compile"):
            for p in self.passes:
                g, plog = p(g)
                log.extend(plog)
            order = g.topo()
        compiled = CompiledGraph(g, order, log, src)
        if self.cache:
            self._cache[src] = compiled
        return compiled

    def clear_cache(self) -> None:
        self._cache.clear()
