"""VRC node executor — colour grade applied to every frame in the state."""
from __future__ import annotations

import numpy as np

from ..fabric.graph import Node
from ..fabric.media import FrameState, MediaState
from ..fabric.scheduler import register_node
from .color_engine import ColorEngine, resolve_grade


def _color_grade(node: Node, state: MediaState, ctx) -> MediaState:
    engine = ColorEngine(compute=ctx.compute)
    grade = resolve_grade(node.params.get("grade", "cinematic"))
    graded = []
    for fr in state.frames:
        if fr.payload is None:
            graded.append(fr)
            continue
        out = engine.grade_frame(np.asarray(fr.payload), grade)
        graded.append(FrameState(id=fr.id, index=fr.index, timestamp=fr.timestamp,
                                 payload=out, metadata=dict(fr.metadata)))
    state.frames = graded
    return state


register_node("VRC_COLOR_GRADE", _color_grade)
