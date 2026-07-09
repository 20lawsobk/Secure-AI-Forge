"""IRC node executors — scene build → path trace → tonemap.

Registered into the UMRF scheduler so an image render is a real graph:

    IRC_SCENE_BUILD -> IRC_PATH_TRACE -> IRC_TONEMAP
"""
from __future__ import annotations

import numpy as np

from ..fabric.graph import Node
from ..fabric.media import FrameState, MediaState
from ..fabric.scheduler import register_node
from .path_tracer import PathTracer, tonemap_filmic
from .scene_builder import build_scene


def _scene_build(node: Node, state: MediaState, ctx) -> MediaState:
    p = node.params
    scene = build_scene(
        color_scheme=p.get("color_scheme", "dark_neon"),
        mood=p.get("mood", "cinematic"),
        seed=int(p.get("seed", 0)),
        aspect=float(p.get("aspect", 1.0)),
    )
    state.metadata["scene"] = scene
    return state


def _path_trace(node: Node, state: MediaState, ctx) -> MediaState:
    scene = state.metadata.get("scene")
    if scene is None:
        raise ValueError("IRC_PATH_TRACE requires a scene from IRC_SCENE_BUILD")
    p = node.params
    tracer = PathTracer(compute=ctx.compute)
    hdr = tracer.render(
        scene,
        width=int(p.get("width", 256)),
        height=int(p.get("height", 256)),
        samples=int(p.get("samples", 4)),
        max_bounces=int(p.get("max_bounces", 2)),
        seed=int(p.get("seed", 0)),
    )
    state.frames = [FrameState(id="beauty", index=0, payload=hdr,
                              metadata={"space": "linear-hdr"})]
    return state


def _tonemap(node: Node, state: MediaState, ctx) -> MediaState:
    frame = state.primary()
    if frame is None or frame.payload is None:
        raise ValueError("IRC_TONEMAP requires a rendered HDR frame")
    ldr = tonemap_filmic(np.asarray(frame.payload, dtype=np.float64))
    state.frames = [FrameState(id="ldr", index=0, payload=ldr,
                              metadata={"space": "srgb-uint8"})]
    return state


register_node("IRC_SCENE_BUILD", _scene_build)
register_node("IRC_PATH_TRACE", _path_trace)
register_node("IRC_TONEMAP", _tonemap)
