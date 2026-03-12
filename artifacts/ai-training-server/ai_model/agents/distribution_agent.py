from __future__ import annotations
from dataclasses import dataclass
from typing import List
from ..model.creative_model import CreativeModel


@dataclass
class DistributionRequest:
    script: str
    platform: str
    goal: str


@dataclass
class DistributionResponse:
    caption: str
    hashtags: List[str]
    posting_time: str


DEFAULT_HASHTAGS = {
    "tiktok": ["#maxbooster", "#tiktokgrowth", "#fyp", "#viral"],
    "instagram": ["#maxbooster", "#reels", "#newmusic", "#artist"],
    "youtube": ["#maxbooster", "#shorts", "#musicvideo", "#newrelease"],
    "facebook": ["#maxbooster", "#facebookreels", "#newmusic"],
    "twitter": ["#maxbooster", "#growth", "#NowPlaying"],
    "linkedin": ["#maxbooster", "#contentstrategy", "#musicindustry"],
    "google_business": ["#maxbooster", "#localbusiness", "#livemusic"],
    "threads": ["#maxbooster", "#threads", "#newmusic"],
}

BEST_POSTING_TIMES = {
    "tiktok": "T18:00:00Z",
    "instagram": "T11:00:00Z",
    "youtube": "T14:00:00Z",
    "facebook": "T09:00:00Z",
    "twitter": "T12:00:00Z",
    "linkedin": "T08:00:00Z",
    "google_business": "T10:00:00Z",
    "threads": "T13:00:00Z",
}


class DistributionAgent:
    def __init__(self, model: CreativeModel):
        self.model = model

    def run(self, req: DistributionRequest) -> DistributionResponse:
        platform_token = f"<PLATFORM_{req.platform.upper()}>"
        goal_token = f"<GOAL_{req.goal.upper()}>"

        prompt = (
            f"{platform_token} {goal_token} <STAGE_CTA>\n"
            f"Script: {req.script}\n"
            f"Generate caption + hashtags + best posting time.\n"
        )

        caption = None
        try:
            output = self.model.generate(prompt)
            if self._is_meaningful(output):
                caption = output
        except Exception:
            pass

        if not caption:
            caption = req.script

        platform_key = req.platform.lower().replace(" ", "_")
        hashtags = DEFAULT_HASHTAGS.get(platform_key, ["#maxbooster"])
        posting_time = f"2026-02-19{BEST_POSTING_TIMES.get(platform_key, 'T12:00:00Z')}"

        return DistributionResponse(
            caption=caption,
            hashtags=hashtags,
            posting_time=posting_time,
        )

    def _is_meaningful(self, text: str) -> bool:
        if not text or len(text) < 10:
            return False
        control_count = sum(1 for w in text.split() if w.startswith("<") and w.endswith(">"))
        total = len(text.split())
        if total == 0:
            return False
        return control_count / total < 0.5
