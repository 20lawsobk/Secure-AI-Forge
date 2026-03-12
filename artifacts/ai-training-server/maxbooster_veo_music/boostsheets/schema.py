from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PlatformTarget:
    platform: str
    goal: str
    duration_sec: Optional[float] = None
    aspect_ratio: Optional[str] = None


@dataclass
class BoostSheet:
    track_id: str
    title: str
    artist: str
    album: Optional[str]

    story: str
    mood: str
    era: str
    references: List[str]

    label: Optional[str]
    brand_notes: str

    lyrics: Optional[str]

    primary_platforms: List[str]
    campaign_notes: str

    targets: List[PlatformTarget] = field(default_factory=list)
