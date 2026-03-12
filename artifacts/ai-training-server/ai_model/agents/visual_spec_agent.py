from __future__ import annotations
from dataclasses import dataclass
from ..model.creative_model import CreativeModel


@dataclass
class VisualSpecRequest:
    idea: str
    platform: str
    tone: str


@dataclass
class VisualSpecResponse:
    thumbnail_prompt: str
    color_scheme: str
    layout: str


PLATFORM_LAYOUTS = {
    "tiktok": "vertical_9_16",
    "instagram": "square_1_1",
    "youtube": "landscape_16_9",
    "facebook": "landscape_16_9",
    "twitter": "landscape_2_1",
    "linkedin": "landscape_1200x627",
    "google_business": "landscape_16_9",
    "threads": "square_1_1",
}

TONE_COLORS = {
    "edgy": "dark_neon",
    "playful": "vibrant_pastel",
    "serious": "monochrome",
    "energetic": "high_contrast",
    "professional": "corporate_blue",
    "casual": "warm_earth",
    "promotional": "bold_red_gold",
}


class VisualSpecAgent:
    def __init__(self, model: CreativeModel):
        self.model = model

    def run(self, req: VisualSpecRequest) -> VisualSpecResponse:
        platform_token = f"<PLATFORM_{req.platform.upper()}>"
        tone_token = f"<TONE_{req.tone.upper()}>"
        prompt = (
            f"{platform_token} {tone_token} "
            f"Generate thumbnail spec for: {req.idea}\n"
        )

        thumbnail_prompt = None
        try:
            output = self.model.generate(prompt)
            if self._is_meaningful(output):
                thumbnail_prompt = output
        except Exception:
            pass

        if not thumbnail_prompt:
            thumbnail_prompt = f"Eye-catching {req.tone} thumbnail for: {req.idea}"

        platform_key = req.platform.lower().replace(" ", "_")
        tone_key = req.tone.lower()

        return VisualSpecResponse(
            thumbnail_prompt=thumbnail_prompt,
            color_scheme=TONE_COLORS.get(tone_key, "high_contrast"),
            layout=PLATFORM_LAYOUTS.get(platform_key, "square_1_1"),
        )

    def _is_meaningful(self, text: str) -> bool:
        if not text or len(text) < 10:
            return False
        control_count = sum(1 for w in text.split() if w.startswith("<") and w.endswith(">"))
        total = len(text.split())
        if total == 0:
            return False
        return control_count / total < 0.5
