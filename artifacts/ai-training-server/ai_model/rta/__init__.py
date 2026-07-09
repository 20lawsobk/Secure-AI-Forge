"""RTA-1 — Unified Rendering Trinity Architecture.

A node-graph render fabric that unifies the three generation mediums (image,
video, audio) behind one shared graph + deterministic scheduler, with domain
node sets that lift fidelity:

  * IRC (image)  — a from-scratch Monte-Carlo path tracer (V-Ray behaviour).
  * VRC (video)  — a node-based colour-grade pipeline (Resolve behaviour).
  * ARC (audio)  — STFT spectral processing / restoration (iZotope behaviour).

All heavy numeric work is routed through the project's self-contained Digital
GPU (``ai_model.gpu.digital_gpu.DigitalGPU``) — the same compute fabric the rest
of the system uses — never raw BLAS or an external accelerator.

Design rule (matches the rest of the project): every node either really renders
or fails explicitly. No node ever emits silent placeholder output.
"""
from __future__ import annotations

from .fabric.media import MediaType, FrameState, MediaState
from .fabric.graph import Node, NodeGraph, topological_sort, GraphCycleError
from .fabric.compute import RTACompute, global_op_counts
from .fabric.scheduler import UMRFScheduler, register_node, node_registry

# Importing ``api`` registers all domain node executors (IRC/VRC/ARC) as a side
# effect, so ``import ai_model.rta`` yields a fully wired fabric.
from . import api  # noqa: E402,F401

__all__ = [
    "MediaType", "FrameState", "MediaState",
    "Node", "NodeGraph", "topological_sort", "GraphCycleError",
    "RTACompute", "global_op_counts",
    "UMRFScheduler", "register_node", "node_registry",
    "api",
]
