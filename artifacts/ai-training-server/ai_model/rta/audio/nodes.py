"""ARC node executor — spectral denoise applied to the audio buffer frame."""
from __future__ import annotations

import numpy as np

from ..fabric.graph import Node
from ..fabric.media import FrameState, MediaState
from ..fabric.scheduler import register_node
from .spectral import SpectralConfig, SpectralEngine


def _spectral_clean(node: Node, state: MediaState, ctx) -> MediaState:
    frame = state.primary()
    if frame is None or frame.payload is None:
        raise ValueError("ARC_SPECTRAL_CLEAN requires an audio buffer frame")
    p = node.params
    sr = int(frame.metadata.get("sample_rate", p.get("sample_rate", 44100)))
    cfg = SpectralConfig(
        reduction_db=float(p.get("reduction_db", 12.0)),
        noise_percentile=float(p.get("noise_percentile", 15.0)),
        over_subtraction=float(p.get("over_subtraction", 1.5)),
    )
    engine = SpectralEngine(compute=ctx.compute)
    cleaned = engine.denoise(np.asarray(frame.payload, dtype=np.float64), sr, cfg)
    state.frames = [FrameState(id="clean", index=0, payload=cleaned,
                              metadata={**frame.metadata, "sample_rate": sr})]
    return state


register_node("ARC_SPECTRAL_CLEAN", _spectral_clean)
