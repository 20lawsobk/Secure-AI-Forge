"""Unified media data model shared by every RTA-1 medium.

A ``MediaState`` flows through the node graph; each node reads it and returns an
updated state. ``FrameState`` is one unit of media payload (a rasterised image,
a video frame, or a block of audio samples) carried as a NumPy array so the
Digital-GPU compute layer can operate on it directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class MediaType(str, Enum):
    VIDEO = "VIDEO"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"
    HYBRID = "HYBRID"


@dataclass
class FrameState:
    """One media payload unit.

    ``payload`` is a NumPy array: an ``HxWx3`` uint8/float image or video frame,
    or a 1-D float32 audio buffer. ``metadata`` carries per-frame descriptors
    (sample rate, colourspace, timecode, …).
    """
    id: str
    index: int
    timestamp: float = 0.0            # ms
    payload: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MediaState:
    """The object that flows through the UMRF node graph."""
    id: str
    type: MediaType
    frames: List[FrameState] = field(default_factory=list)
    timeline: Any = None                        # NodeGraph (avoid import cycle)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def primary(self) -> Optional[FrameState]:
        """The first frame, or ``None`` when the state carries no payload yet."""
        return self.frames[0] if self.frames else None

    def set_frames(self, frames: List[FrameState]) -> "MediaState":
        self.frames = frames
        return self
