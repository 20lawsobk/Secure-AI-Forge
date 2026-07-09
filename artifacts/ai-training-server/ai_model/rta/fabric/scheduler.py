"""UMRF scheduler — topologically orders a node graph and executes it.

Node executors are registered by the domain modules (IRC/VRC/ARC) via
:func:`register_node`. Each executor has the signature::

    fn(node: Node, state: MediaState, ctx: UMRFScheduler) -> MediaState

The scheduler exposes its :class:`RTACompute` as ``ctx.compute`` so every node
runs its heavy math on the self-contained Digital GPU.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from .cache import NodeCache
from .compute import RTACompute
from .graph import Node, topological_sort
from .media import MediaState

# type -> executor
_NODE_REGISTRY: Dict[str, Callable] = {}


def register_node(node_type: str, fn: Callable) -> None:
    _NODE_REGISTRY[node_type] = fn


def node_registry() -> Dict[str, Callable]:
    return dict(_NODE_REGISTRY)


class UMRFScheduler:
    def __init__(self, compute: Optional[RTACompute] = None,
                 cache: Optional[NodeCache] = None):
        self.compute = compute or RTACompute()
        self.cache = cache
        self.registry = dict(_NODE_REGISTRY)
        self.profile: list = []

    def execute(self, media: MediaState) -> MediaState:
        if media.timeline is None:
            raise ValueError("MediaState.timeline (NodeGraph) is required")
        order = topological_sort(media.timeline)
        state = media
        self.profile = []
        for node in order:
            state = self._exec_node(node, state)
        return state

    def _exec_node(self, node: Node, state: MediaState) -> MediaState:
        fn = self.registry.get(node.type)
        if fn is None:
            raise KeyError(f"unknown RTA node type: {node.type!r}")

        cache_key = None
        if self.cache is not None and node.params.get("cacheable") and node.params.get("cache_key"):
            cache_key = NodeCache.make_key(node.type, node.params["cache_key"])
            hit = self.cache.get(cache_key)
            if hit is not None:
                self.profile.append({"node": node.id, "type": node.type, "cached": True})
                return hit

        result = fn(node, state, self)
        if not isinstance(result, MediaState):
            raise TypeError(f"node {node.id} ({node.type}) must return a MediaState")

        if cache_key is not None:
            self.cache.put(cache_key, result)
        self.profile.append({"node": node.id, "type": node.type, "cached": False})
        return result
