from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import uuid
import time


@dataclass
class BoostSheet:
    sheet_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "script_sheet"
    platform: str = "tiktok"
    goal: str = "growth"
    tone: str = "default"
    source_url: str | None = None
    layout: Dict[str, Any] = field(default_factory=dict)
    blocks: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: {
        "text": [],
        "image": [],
        "audio": [],
        "video": []
    })
    agent_notes: List[str] = field(default_factory=list)
    performance_links: List[Dict[str, Any]] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_history(self, entry: str):
        self.history.append(entry)

    def add_agent_note(self, note: str):
        self.agent_notes.append(note)
