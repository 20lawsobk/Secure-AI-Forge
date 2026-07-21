"""RTA-1 high-level entrypoints — build a UMRF graph and run it.

Each function assembles a real node graph and executes it through the
:class:`UMRFScheduler`, so the fabric (graph + scheduler + Digital-GPU compute)
is genuinely exercised rather than bypassed. These are what the server endpoints
call.

Importing this module registers every domain node executor (IRC/VRC/ARC).
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from .fabric.cache import NodeCache
from .fabric.compute import RTACompute, global_op_counts
from .fabric.graph import Node, NodeGraph
from .fabric.media import FrameState, MediaState, MediaType
from .fabric.scheduler import UMRFScheduler

# Register domain nodes as an import side effect.
from .image import nodes as _image_nodes   # noqa: F401
from .video import nodes as _video_nodes   # noqa: F401
from .audio import nodes as _audio_nodes   # noqa: F401

__all__ = [
    "render_image", "grade_video_frame", "spectral_clean_audio",
    "global_op_counts", "self_test",
]

# ── Persistent shared RTACompute ───────────────────────────────────────────
# RTACompute initialises the Digital GPU connection (pocket GEMM engine,
# VRAM allocator) once. Creating a fresh instance per scene call wastes
# ~50–200 ms of cold-start per render thread. The compute instance is
# stateless between calls; UMRFScheduler is still created per-execution
# (cheap — it holds only per-run state). Thread-safe double-checked init.
_SHARED_COMPUTE: Optional[RTACompute] = None
_COMPUTE_LOCK = threading.Lock()


def _get_compute() -> RTACompute:
    """Return the singleton RTACompute, initialising it on first call."""
    global _SHARED_COMPUTE
    if _SHARED_COMPUTE is None:
        with _COMPUTE_LOCK:
            if _SHARED_COMPUTE is None:
                _SHARED_COMPUTE = RTACompute()
    return _SHARED_COMPUTE


# ── IMAGE (IRC) ────────────────────────────────────────────────────────────
def render_image(color_scheme: str = "dark_neon", mood: str = "cinematic",
                 width: int = 256, height: int = 256, samples: int = 4,
                 max_bounces: int = 2, seed: int = 0) -> np.ndarray:
    """Path-trace a hero image. Returns an ``HxWx3`` uint8 array (sRGB)."""
    aspect = width / float(height) if height else 1.0
    graph = NodeGraph(id="irc", nodes=[
        Node(id="scene", type="IRC_SCENE_BUILD",
             params={"color_scheme": color_scheme, "mood": mood, "seed": seed, "aspect": aspect}),
        Node(id="trace", type="IRC_PATH_TRACE", inputs=["scene"],
             params={"width": width, "height": height, "samples": samples,
                     "max_bounces": max_bounces, "seed": seed}),
        Node(id="tone", type="IRC_TONEMAP", inputs=["trace"]),
    ])
    media = MediaState(id="img", type=MediaType.IMAGE, timeline=graph)
    sched = UMRFScheduler(compute=_get_compute())
    out = sched.execute(media)
    frame = out.primary()
    if frame is None or frame.payload is None:
        raise RuntimeError("RTA image render produced no frame")
    return np.asarray(frame.payload, dtype=np.uint8)


# ── VIDEO (VRC) ────────────────────────────────────────────────────────────
def grade_video_frame(frame: np.ndarray, grade: str = "cinematic") -> np.ndarray:
    """Colour-grade a single ``HxWx3`` frame through the VRC node. Returns uint8."""
    fs = FrameState(id="f0", index=0, payload=np.asarray(frame))
    graph = NodeGraph(id="vrc", nodes=[
        Node(id="grade", type="VRC_COLOR_GRADE", params={"grade": grade}),
    ])
    media = MediaState(id="vid", type=MediaType.VIDEO, frames=[fs], timeline=graph)
    sched = UMRFScheduler(compute=_get_compute())
    out = sched.execute(media)
    return np.asarray(out.primary().payload, dtype=np.uint8)


# ── AUDIO (ARC) ────────────────────────────────────────────────────────────
def spectral_clean_audio(samples: np.ndarray, sample_rate: int,
                         reduction_db: float = 12.0) -> np.ndarray:
    """Spectral-denoise a mono float buffer through the ARC node. Returns float32."""
    fs = FrameState(id="a0", index=0, payload=np.asarray(samples),
                    metadata={"sample_rate": int(sample_rate)})
    graph = NodeGraph(id="arc", nodes=[
        Node(id="clean", type="ARC_SPECTRAL_CLEAN",
             params={"sample_rate": sample_rate, "reduction_db": reduction_db}),
    ])
    media = MediaState(id="aud", type=MediaType.AUDIO, frames=[fs], timeline=graph)
    sched = UMRFScheduler(compute=_get_compute())
    out = sched.execute(media)
    return np.asarray(out.primary().payload, dtype=np.float32)


def self_test() -> dict:
    """Fast smoke test proving every path really runs on the Digital GPU."""
    before = global_op_counts().get("gemm", 0)
    img = render_image(width=48, height=48, samples=1, max_bounces=1, seed=1)
    frame = np.full((16, 16, 3), 120, dtype=np.uint8)
    graded = grade_video_frame(frame, grade="neon")
    tone = 0.05 * np.sin(2 * np.pi * 220 * np.arange(4096) / 8000.0)
    noise = 0.01 * np.random.default_rng(0).standard_normal(4096)
    clean = spectral_clean_audio((tone + noise).astype(np.float32), 8000)
    after = global_op_counts().get("gemm", 0)
    return {
        "image_shape": list(img.shape),
        "video_shape": list(graded.shape),
        "audio_len": int(clean.shape[0]),
        "digital_gpu_gemms": after - before,
    }
