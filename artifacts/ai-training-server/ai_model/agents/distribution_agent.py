from __future__ import annotations
import datetime
import re
from dataclasses import dataclass
from typing import List, Optional
from ..model.creative_model import CreativeModel


@dataclass
class DistributionRequest:
    script: str
    platform: str
    goal: str
    awareness: str = ""


@dataclass
class DistributionResponse:
    caption: str
    hashtags: List[str]
    posting_time: str


DEFAULT_HASHTAGS = {
    "tiktok":          ["#maxbooster", "#tiktokgrowth", "#fyp", "#viral"],
    "instagram":       ["#maxbooster", "#reels", "#newmusic", "#artist"],
    "youtube":         ["#maxbooster", "#shorts", "#musicvideo", "#newrelease"],
    "facebook":        ["#maxbooster", "#facebookreels", "#newmusic"],
    "twitter":         ["#maxbooster", "#growth", "#NowPlaying"],
    "linkedin":        ["#maxbooster", "#contentstrategy", "#musicindustry"],
    "google_business": ["#maxbooster", "#localbusiness", "#livemusic"],
    "threads":         ["#maxbooster", "#threads", "#newmusic"],
}

BEST_POSTING_TIMES = {
    "tiktok":          "T18:00:00Z",
    "instagram":       "T11:00:00Z",
    "youtube":         "T14:00:00Z",
    "facebook":        "T09:00:00Z",
    "twitter":         "T12:00:00Z",
    "linkedin":        "T08:00:00Z",
    "google_business": "T10:00:00Z",
    "threads":         "T13:00:00Z",
}


# ── Awareness parsing ─────────────────────────────────────────────────────────

def _parse_hashtags_from_awareness(awareness: str, platform: str) -> List[str]:
    """
    Extract hashtags from the awareness context string.
    Priority order: platform-specific signal tags → trending topics section → global tags.
    Returns up to 8 unique hashtags.
    """
    if not awareness:
        return []

    plat_lower = platform.lower()
    priority: List[str] = []
    general: List[str] = []
    in_trending = False

    for line in awareness.splitlines():
        stripped = line.strip()

        # Trending Topics section
        if "TRENDING TOPICS" in stripped:
            in_trending = True
            continue
        if in_trending:
            if stripped.startswith("==="):
                in_trending = False
            else:
                tags = re.findall(r"#\w+", stripped)
                general.extend(tags)
            continue

        # Platform-relevant signal lines → higher priority
        tags_on_line = re.findall(r"#\w+", stripped)
        if not tags_on_line:
            continue

        if plat_lower in stripped.lower() or "Trending:" in stripped or "Tags:" in stripped:
            priority.extend(tags_on_line)
        else:
            general.extend(tags_on_line)

    # Deduplicate preserving order, priority first
    seen: set = set()
    result: List[str] = []
    for tag in priority + general:
        if tag not in seen:
            seen.add(tag)
            result.append(tag)

    return result[:8]


def _pdim_hashtags(platform: str) -> List[str]:
    """
    Fetch platform-specific hashtags stored in pdim by prior awareness cycles.
    Returns [] if pdim is offline or no data exists yet.
    """
    try:
        from storage_client import get_storage
        store = get_storage()
        key = f"dist:hashtags:{platform.lower()}"
        data = store.lrange(key, 0, 7)
        return [h for h in data if isinstance(h, str) and h.startswith("#")]
    except Exception:
        return []


def _store_hashtags_to_pdim(platform: str, hashtags: List[str]) -> None:
    """Persist discovered hashtags back to pdim for future requests."""
    if not hashtags:
        return
    try:
        from storage_client import get_storage
        store = get_storage()
        key = f"dist:hashtags:{platform.lower()}"
        existing = set(store.lrange(key, 0, -1) or [])
        new_tags = [h for h in hashtags if h not in existing]
        if new_tags:
            store.lpush(key, *new_tags)
            # Cap at 32 entries
            store.ltrim(key, 0, 31)
    except Exception:
        pass


def _parse_timing_from_awareness(awareness: str, platform: str) -> Optional[str]:
    """
    Extract platform-specific posting hour from the awareness context.
    Looks for time patterns (e.g. "6pm", "18:00") near platform mentions.
    Returns an ISO-like posting time string, or None if not found.
    """
    if not awareness or not platform:
        return None

    plat_lower = platform.lower()
    in_timing = False

    for line in awareness.splitlines():
        stripped = line.strip()
        if "PLATFORM TIMING" in stripped or "TIMING" in stripped:
            in_timing = True
            continue
        if in_timing:
            if stripped.startswith("==="):
                break
            if plat_lower in stripped.lower():
                m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", stripped, re.IGNORECASE)
                if m:
                    hour = int(m.group(1))
                    minute = int(m.group(2) or 0)
                    meridiem = m.group(3).lower()
                    if meridiem == "pm" and hour != 12:
                        hour += 12
                    elif meridiem == "am" and hour == 12:
                        hour = 0
                    today = datetime.date.today()
                    return f"{today}T{hour:02d}:{minute:02d}:00Z"

    return None


# ── Agent ─────────────────────────────────────────────────────────────────────

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
        awareness = req.awareness or ""

        # ── Hashtag resolution (awareness → pdim → static) ──────────────────
        hashtags: List[str] = []

        # 1. Live awareness signals (highest priority — most current)
        awareness_tags = _parse_hashtags_from_awareness(awareness, platform_key)
        hashtags.extend(awareness_tags)

        # 2. pdim-stored hashtags from previous awareness cycles
        if len(hashtags) < 4:
            pdim_tags = _pdim_hashtags(platform_key)
            for t in pdim_tags:
                if t not in hashtags:
                    hashtags.append(t)

        # 3. Static defaults as fallback
        if len(hashtags) < 2:
            hashtags.extend(DEFAULT_HASHTAGS.get(platform_key, ["#maxbooster"]))

        # Deduplicate, cap at 8
        seen: set = set()
        unique: List[str] = []
        for h in hashtags:
            if h not in seen:
                seen.add(h)
                unique.append(h)
        hashtags = unique[:8]

        # Persist awareness-discovered hashtags to pdim for next time
        if awareness_tags:
            _store_hashtags_to_pdim(platform_key, awareness_tags)

        # ── Posting time resolution (awareness → static) ─────────────────────
        posting_time = _parse_timing_from_awareness(awareness, platform_key)
        if not posting_time:
            today = datetime.date.today()
            posting_time = f"{today}{BEST_POSTING_TIMES.get(platform_key, 'T12:00:00Z')}"

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
