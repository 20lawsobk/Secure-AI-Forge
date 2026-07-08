"""
Content Playbook — research-distilled world knowledge about top social content.

A curated, deterministic pattern library distilled from an extensive July-2026
web research pass over published engagement studies and creator-economy data
(Buffer 45M+ posts, Socialinsider/Later/Sprout multi-million-post caption
studies, Paddy Galloway's 3.3B-Shorts retention analysis, Later.com's 10k viral
TikTok hook study, Meta for Business caption-structure research, SSRN/PMC
emotional-arousal sharing research, and 2025-26 music-marketing playbooks).

What the research converged on:

  * Structure   — Hook -> Value -> CTA captions earn ~23% more engagement;
                  only the first ~125 characters are visible pre-fold, so the
                  hook must land there. 150-220 words is the IG comment-rate
                  sweet spot for long captions; Reels captions stay short.
  * Hooks       — the highest-retention opening archetypes are identity calls
                  ("if you're X..."), curiosity gaps, result previews, reveals,
                  and emotion-first storytelling. Hooks that resolve in under
                  2s beat 4-5s wind-ups by ~23% completion.
  * Emotion     — arousal level beats emotion type: awe, excitement and joy
                  are the most shareable non-divisive triggers.
  * CTAs        — low-friction beats generic: one-tap reactions ("drop a 🔥"),
                  tag-a-friend share triggers, and save prompts out-engage
                  plain "check it out" closers.
  * Hashtags    — 2025-26 platforms moved to quality-over-quantity; Instagram
                  hard-caps at 5, and 3-5 relevant tags is the working norm.
  * Music       — emotion-first context ("made this for the nights that...")
                  outperforms bare song clips; behind-the-scenes content is
                  the highest trust-builder for artists.

Like the quality buffer, this is borrowed world knowledge: the blending layer
(ai_model/quality_awareness.py) scales it by the same self-sufficiency weight,
so it retires as MaxBooster's own corpus grows. Everything here is pure data +
pure functions — no I/O, no model calls, never-raise.
"""
from __future__ import annotations

import re
from typing import Dict, List

PLAYBOOK_VERSION = "2026-07"

# ── Hook archetypes (ranked roughly by 2026 retention-test performance) ─────
# All templates are finished, user-facing copy with {idea}/{artist} slots.

HOOK_ARCHETYPES: Dict[str, List[str]] = {
    "identity_call": [
        "If {idea} is exactly what you've been waiting for — this one's yours",
        "This is for everyone who plays a song 40 times in a row: {idea}",
    ],
    "curiosity_gap": [
        "Here's what nobody tells you about {idea}",
        "There's one moment in {idea} you'll rewind twice",
    ],
    "result_preview": [
        "This is how {idea} sounds when it's finished — now watch how it started",
        "The final version of {idea} first. The story behind it next",
    ],
    "reveal": [
        "{artist} kept {idea} quiet for months. Not anymore",
        "What {artist} was really making this whole time: {idea}",
    ],
    "emotion_story": [
        "{artist} made {idea} for the nights that don't make the highlight reel",
        "Some songs are written. {idea} was survived",
    ],
    "pattern_interrupt": [
        "Don't scroll — {idea} earns the next 15 seconds",
        "Wait. Play {idea} out loud",
    ],
}

# ── CTA bank by intent (low-friction, research-backed phrasing) ─────────────

CTA_BANK: Dict[str, List[str]] = {
    "drive_streams": [
        "Save this so you're early when {idea} takes off 🎧",
        "Add {idea} to the playlist — link in bio",
        "Play {idea} once. That's the whole ask",
    ],
    "drive_engagement": [
        "Drop a 🔥 if {idea} hits",
        "Tag someone who needs {idea} on their playlist",
        "One word for {idea} — comments open 👇",
    ],
    "grow_followers": [
        "Follow — the {idea} story is just getting started 🔔",
        "Stay close: everything behind {idea} drops here first",
    ],
    "drive_conversion": [
        "Link in bio before it's gone 🛒",
        "First listeners get first access — link in bio",
    ],
    "build_awareness": [
        "Share this with someone who hears music differently",
        "Remember where you heard {idea} first 👀",
    ],
}

# ── Video scene phrase templates (hook / body / cta pools) ──────────────────

SCENE_TEMPLATES: Dict[str, List[str]] = {
    "hook": [
        "Here's what nobody tells you about {idea}",
        "{artist} kept {idea} quiet for months. Not anymore",
        "Some songs are written. {idea} was survived",
        "Don't scroll — {idea} earns the next 15 seconds",
    ],
    "body": [
        "{artist} made {idea} for the nights that don't make the highlight reel",
        "Every second of {idea} was a choice — here's the one that mattered",
        "The room went quiet when {idea} played back the first time",
    ],
    "cta": [
        "Save this so you're early when {idea} takes off",
        "Tag someone who needs {idea} on their playlist",
        "Play {idea} once. That's the whole ask",
    ],
}

# ── High-arousal emotion lexicon (awe / excitement / joy — most shareable) ──

HIGH_AROUSAL_WORDS = frozenset({
    "unbelievable", "insane", "chills", "goosebumps", "wild", "unreal",
    "wait", "finally", "obsessed", "loud", "alive", "electric", "explosive",
    "heart", "cry", "survived", "quiet", "rewind", "loop", "repeat",
    "first", "never", "everything", "nothing", "everyone", "nobody",
})

# ── Platform norms (2025-26 research corrections) ───────────────────────────

HOOK_VISIBLE_CHARS = 125          # caption fold: hook must land inside this
HASHTAG_CAPS: Dict[str, int] = {  # quality-over-quantity era, IG hard cap 5
    "tiktok": 4, "instagram": 5, "instagram_reels": 5,
    "youtube": 3, "youtube_shorts": 3, "twitter": 2,
    "facebook": 3, "linkedin": 3, "general": 4,
}

# Research-backed creative directives blended into every GenerationBrief.
_DIRECTIVES_CORE = [
    "Land the full hook inside the first 125 characters — that is all "
    "viewers see before the fold",
    "Aim for high-arousal emotion (awe, excitement, joy) — arousal level "
    "drives shares more than emotion type",
]
_DIRECTIVES_BY_INTENT: Dict[str, str] = {
    "drive_streams": "Give the song context, not just a clip — emotion-first "
                     "framing converts viewers into listeners",
    "drive_engagement": "Close with a one-tap ask (emoji reaction or "
                        "tag-a-friend) — low friction wins comments",
    "grow_followers": "Tease a continuing story so following feels like "
                      "subscribing to the next chapter",
    "drive_conversion": "Pair the ask with urgency or early-access framing",
    "build_awareness": "Optimise for shareability — make the viewer look "
                       "good for passing it on",
}


# ── pure functions (all deterministic, never-raise) ─────────────────────────

def hook_candidates(topic: str, artist: str) -> List[str]:
    """Formatted hook candidates from every researched archetype."""
    idea = (topic or "").strip() or "this drop"
    art = (artist or "").strip() or "the artist"
    out: List[str] = []
    for bank in HOOK_ARCHETYPES.values():
        for tpl in bank:
            try:
                out.append(tpl.format(idea=idea, artist=art))
            except (KeyError, IndexError, ValueError):
                continue
    return out


def cta_candidates(intent: str, topic: str) -> List[str]:
    """Formatted CTA candidates for an intent."""
    idea = (topic or "").strip() or "this"
    out: List[str] = []
    for tpl in CTA_BANK.get(intent, CTA_BANK["build_awareness"]):
        try:
            out.append(tpl.format(idea=idea))
        except (KeyError, IndexError, ValueError):
            continue
    return out


def scene_phrase_templates(scene_type: str) -> List[str]:
    """Raw {idea}/{artist} phrase templates for a video scene pool."""
    key = {"hook": "hook", "drop": "hook", "build": "hook", "chorus": "hook",
           "cta": "cta", "outro": "cta"}.get(scene_type, "body")
    return list(SCENE_TEMPLATES.get(key, []))


def hashtag_cap(platform: str) -> int:
    return HASHTAG_CAPS.get(platform, HASHTAG_CAPS["general"])


def brief_directives(intent: str) -> List[str]:
    """Research-backed directives for the GenerationBrief (internal only)."""
    out = list(_DIRECTIVES_CORE)
    extra = _DIRECTIVES_BY_INTENT.get(intent)
    if extra:
        out.append(extra)
    return out


_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def structure_score(text: str) -> float:
    """0-1 score for Hook->Value->CTA shape + arousal, per the research.

    Rewards: a front-loaded first line that fits inside the visible fold,
    a multi-part (hook/value/CTA) layout, and high-arousal wording.
    Deterministic and never-raise; scoring only, never mutates text.
    """
    try:
        t = (text or "").strip()
        if not t:
            return 0.0
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        first = lines[0] if lines else t

        score = 0.0
        # Hook fits inside the pre-fold window (first 125 chars).
        if len(first) <= HOOK_VISIBLE_CHARS:
            score += 0.35
        # Distinct hook / value / CTA sections.
        if len(lines) >= 3:
            score += 0.30
        elif len(lines) == 2:
            score += 0.15
        # High-arousal wording anywhere in the copy.
        low = t.lower()
        hits = sum(1 for w in HIGH_AROUSAL_WORDS if w in low)
        score += min(0.20, 0.07 * hits)
        # A low-friction interactive close (emoji ask / tag / save language).
        last = lines[-1].lower() if lines else low
        if (_EMOJI_RE.search(lines[-1]) if lines else False) or any(
            k in last for k in ("tag ", "save ", "drop a", "comment", "share")
        ):
            score += 0.15
        return min(1.0, round(score, 4))
    except Exception:  # noqa: BLE001 - scoring must never break generation
        return 0.0
