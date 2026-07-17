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


# ── Static dicts — dead code in normal operation (awareness is always active) ──
# Kept as a true last resort for when awareness is completely absent.

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

# Low-friction CTA closes per platform — appended to the fallback caption when the
# script doesn't already contain a CTA keyword.  One-tap language ("drop a 🔥",
# "save", "tag") scores higher than generic "check it out" closers per the
# content_playbook research.
_FALLBACK_CTAS = {
    "tiktok":          "Drop a 🔥 if this goes hard — link in bio to stream now!",
    "instagram":       "Save this and tag someone who needs to hear it 🎧",
    "youtube":         "Subscribe and hit the bell so you never miss a drop 🔔",
    "facebook":        "Share this with someone who needs to hear it today!",
    "twitter":         "RT if this goes hard — stream link in bio 🎵",
    "linkedin":        "Follow for more music industry insights — share if this resonates.",
    "google_business": "Follow for more live music updates and exclusive drops!",
    "threads":         "Repost if this hits different 🔥 — stream now.",
}

_CTA_KEYWORDS = {
    "click", "follow", "link", "save", "share", "buy", "get", "stream",
    "listen", "subscribe", "comment", "tap", "join", "shop", "watch", "bio",
    "drop a", "tag ", "repost",
}


# ── Awareness parsing ──────────────────────────────────────────────────────────
# Both functions guarantee a non-empty result when `awareness` is non-empty.

def _parse_hashtags_from_awareness(awareness: str, platform: str) -> List[str]:
    """
    Extract hashtags from the awareness context string.
    When no hashtags are present in the text, synthesises them from signal
    keywords so the result is always non-empty when awareness is non-empty.
    """
    if not awareness:
        return []

    plat_lower = platform.lower()
    priority: List[str] = []
    general: List[str] = []
    in_trending = False

    for line in awareness.splitlines():
        stripped = line.strip()

        if "TRENDING TOPICS" in stripped:
            in_trending = True
            continue
        if in_trending:
            if stripped.startswith("==="):
                in_trending = False
            else:
                general.extend(re.findall(r"#\w+", stripped))
            continue

        tags_on_line = re.findall(r"#\w+", stripped)
        if not tags_on_line:
            continue

        if plat_lower in stripped.lower() or "Trending:" in stripped or "Tags:" in stripped:
            priority.extend(tags_on_line)
        else:
            general.extend(tags_on_line)

    seen: set = set()
    result: List[str] = []
    for tag in priority + general:
        if tag not in seen:
            seen.add(tag)
            result.append(tag)

    # If no hashtags were embedded in the awareness text, synthesise from
    # signal keywords — guarantees non-empty output.
    if not result:
        for line in awareness.splitlines():
            m = re.match(r"\[(HIGH|MEDIUM|LOW)\]\s+(.+)", line.strip())
            if m:
                words = re.findall(r"[A-Za-z]{5,}", m.group(2))
                for w in words[:3]:
                    tag = f"#{w.lower()}"
                    if tag not in seen:
                        seen.add(tag)
                        result.append(tag)
            if len(result) >= 4:
                break

        # Always include a platform tag if we had to synthesise
        plat_tag = f"#{plat_lower}"
        if plat_tag not in seen:
            result.insert(0, plat_tag)

    return result[:8]


def _parse_timing_from_awareness(awareness: str, platform: str) -> str:
    """
    Extract posting time from the awareness context.
    Falls back to broader time-pattern search, then to engagement-keyword
    inference, so the result is always non-empty when awareness is non-empty.
    """
    if not awareness:
        return ""

    plat_lower = platform.lower()
    today = datetime.datetime.now(datetime.timezone.utc).date()

    # 1. Dedicated PLATFORM TIMING section
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
                    return f"{today}T{hour:02d}:{minute:02d}:00Z"

    # 2. Any time pattern on a line mentioning the platform
    for line in awareness.splitlines():
        if plat_lower in line.lower():
            m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", line, re.IGNORECASE)
            if m:
                hour = int(m.group(1))
                minute = int(m.group(2) or 0)
                meridiem = m.group(3).lower()
                if meridiem == "pm" and hour != 12:
                    hour += 12
                elif meridiem == "am" and hour == 12:
                    hour = 0
                return f"{today}T{hour:02d}:{minute:02d}:00Z"

    # 3. Infer from engagement-time keywords anywhere in the awareness text
    text_lower = awareness.lower()
    if "peak evening" in text_lower or "evening engagement" in text_lower or "6pm" in text_lower or "7pm" in text_lower:
        hour = 19
    elif "morning" in text_lower or "9am" in text_lower:
        hour = 9
    elif "afternoon" in text_lower or "2pm" in text_lower or "3pm" in text_lower:
        hour = 14
    elif "lunch" in text_lower or "midday" in text_lower or "noon" in text_lower:
        hour = 12
    elif "night" in text_lower or "late" in text_lower:
        hour = 21
    else:
        # Awareness is present but has no time signal — use 18:00 as the
        # highest-engagement default across all major platforms.
        hour = 18

    return f"{today}T{hour:02d}:00:00Z"


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
            store.ltrim(key, 0, 31)
    except Exception:
        pass


# ── Agent ──────────────────────────────────────────────────────────────────────

class DistributionAgent:
    def __init__(self, model: CreativeModel):
        self.model = model

    def run(self, req: DistributionRequest) -> DistributionResponse:
        # Awareness is always primary.  The script already carries live industry
        # signals (it came from ScriptAgent._awareness_compose), so it IS the
        # awareness-driven caption.  We use it directly rather than feeding it
        # as a hint to model.generate() — which may ignore or dilute those
        # signals.  Hashtags and posting time come from awareness parsers, which
        # guarantee non-empty results whenever awareness is present.
        platform_key = req.platform.lower().replace(" ", "_")
        awareness = req.awareness or ""

        # ── Caption: script is already awareness-composed ─────────────────────
        # Append a platform CTA if the script doesn't already carry one.
        script_lower = req.script.lower()
        if any(kw in script_lower for kw in _CTA_KEYWORDS):
            caption = req.script
        else:
            plat_cta = _FALLBACK_CTAS.get(platform_key, "Follow for more content!")
            caption = f"{req.script}\n\n{plat_cta}"

        # ── Hashtags & timing from awareness ─────────────────────────────────
        if awareness:
            hashtags = _parse_hashtags_from_awareness(awareness, platform_key)

            # Supplement with pdim-stored tags from prior awareness cycles.
            pdim_tags = _pdim_hashtags(platform_key)
            seen: set = set(hashtags)
            for t in pdim_tags:
                if t not in seen and len(hashtags) < 8:
                    hashtags.append(t)
                    seen.add(t)

            posting_time = _parse_timing_from_awareness(awareness, platform_key)

            # Persist newly discovered tags back to pdim for future requests.
            _store_hashtags_to_pdim(platform_key, hashtags)
        else:
            # True last resort: awareness absent (should not occur in production).
            hashtags = _pdim_hashtags(platform_key) or DEFAULT_HASHTAGS.get(platform_key, ["#maxbooster"])
            today = datetime.date.today()
            posting_time = f"{today}{BEST_POSTING_TIMES.get(platform_key, 'T12:00:00Z')}"

        return DistributionResponse(
            caption=caption,
            hashtags=hashtags[:8],
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
