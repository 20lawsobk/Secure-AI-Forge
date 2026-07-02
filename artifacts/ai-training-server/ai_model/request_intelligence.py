"""
Request Intelligence Layer
==========================

A fast, deterministic, dependency-free pre-generation analysis module shared by
every generation service endpoint (content, text, image, audio, video).

Given a raw generation request it:
  1. Classifies the *intent* of the request (drive streams, awareness,
     engagement, conversion, follower growth).
  2. Infers the likely *audience* and best-practice *tone* for the platform.
  3. Extracts the salient *keywords* from the topic / prompt.
  4. Selects a per-platform *generation strategy* (aspect ratio, hook style,
     CTA style, target word-count window, hashtag count, candidate count,
     sampling temperature).
  5. Produces human-readable *creative directives* plus an *augmented idea*
     string that is fed to the underlying agents to steer better output.
  6. Scores and ranks candidate texts so endpoints can return the best one.

Everything here is pure-python and side-effect free so it can run before the
(slow, gated) transformer inference without adding latency. It is purely
additive — endpoints keep their existing behaviour and merely gain an
`intelligence` block plus higher-quality, better-targeted inputs/outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Platform best-practice profiles
# ---------------------------------------------------------------------------
# word_count is a (low, high) target window for a single post/caption.

PLATFORM_PROFILES: Dict[str, Dict[str, Any]] = {
    "tiktok": {
        "aspect_ratio": "9:16", "layout": "portrait_9_16", "word_count": (12, 40),
        "hook_style": "pattern_interrupt", "cta_style": "comment_engage",
        "hashtags": 5, "tempo": "fast", "temperature": 0.95,
    },
    "instagram": {
        "aspect_ratio": "1:1", "layout": "square_1_1", "word_count": (20, 60),
        "hook_style": "aesthetic_curiosity", "cta_style": "save_share",
        "hashtags": 8, "tempo": "medium", "temperature": 0.9,
    },
    "instagram_reels": {
        "aspect_ratio": "9:16", "layout": "portrait_9_16", "word_count": (12, 40),
        "hook_style": "pattern_interrupt", "cta_style": "save_share",
        "hashtags": 6, "tempo": "fast", "temperature": 0.95,
    },
    "youtube": {
        "aspect_ratio": "16:9", "layout": "landscape_16_9", "word_count": (40, 90),
        "hook_style": "value_promise", "cta_style": "subscribe",
        "hashtags": 4, "tempo": "steady", "temperature": 0.85,
    },
    "youtube_shorts": {
        "aspect_ratio": "9:16", "layout": "portrait_9_16", "word_count": (12, 40),
        "hook_style": "pattern_interrupt", "cta_style": "subscribe",
        "hashtags": 4, "tempo": "fast", "temperature": 0.95,
    },
    "twitter": {
        "aspect_ratio": "16:9", "layout": "landscape_16_9", "word_count": (10, 35),
        "hook_style": "bold_claim", "cta_style": "reply_retweet",
        "hashtags": 2, "tempo": "punchy", "temperature": 0.92,
    },
    "facebook": {
        "aspect_ratio": "1:1", "layout": "square_1_1", "word_count": (20, 60),
        "hook_style": "relatable_story", "cta_style": "learn_more",
        "hashtags": 3, "tempo": "medium", "temperature": 0.85,
    },
    "linkedin": {
        "aspect_ratio": "16:9", "layout": "landscape_16_9", "word_count": (40, 90),
        "hook_style": "value_promise", "cta_style": "learn_more",
        "hashtags": 3, "tempo": "steady", "temperature": 0.8,
    },
    "general": {
        "aspect_ratio": "1:1", "layout": "square_1_1", "word_count": (20, 55),
        "hook_style": "curiosity", "cta_style": "act_now",
        "hashtags": 5, "tempo": "medium", "temperature": 0.9,
    },
}

# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

GOAL_SIGNALS: Dict[str, List[str]] = {
    "drive_streams": [
        "stream", "listen", "spotify", "apple music", "play", "song", "single",
        "track", "ep", "album", "release", "drop", "out now", "new music",
    ],
    "drive_conversion": [
        "buy", "ticket", "merch", "sale", "presale", "shop", "store",
        "preorder", "pre-order", "tour", "vinyl", "purchase", "checkout",
    ],
    "grow_followers": [
        "follow", "subscribe", "grow", "audience", "follower", "join",
    ],
    "drive_engagement": [
        "engage", "comment", "share", "community", "fans", "interact",
        "poll", "question", "duet", "stitch", "reply", "tag",
    ],
    "build_awareness": [
        "awareness", "discover", "introduce", "brand", "launch", "announce",
        "reveal", "behind the scenes", "story", "journey", "new",
    ],
}

# Map free-form goal strings the callers send to canonical intents.
GOAL_ALIASES: Dict[str, str] = {
    "growth": "grow_followers",
    "followers": "grow_followers",
    "conversion": "drive_conversion",
    "sales": "drive_conversion",
    "streams": "drive_streams",
    "engagement": "drive_engagement",
    "awareness": "build_awareness",
    "reach": "build_awareness",
}

INTENT_LABELS: Dict[str, str] = {
    "drive_streams": "Drive streams / plays",
    "drive_conversion": "Drive conversions / sales",
    "grow_followers": "Grow followers",
    "drive_engagement": "Drive engagement",
    "build_awareness": "Build awareness",
}

# CTA phrasing per intent, tuned per platform CTA style.
CTA_LIBRARY: Dict[str, str] = {
    "drive_streams": "Stream it now — link in bio 🎧",
    "drive_conversion": "Grab yours before they're gone — link in bio 🛒",
    "grow_followers": "Follow for the full journey 🔔",
    "drive_engagement": "Drop a 🔥 if this hits — tell me in the comments",
    "build_awareness": "This is just the beginning — stay close 👀",
}

# Hook scaffolds per hook style (used to generate ranked candidate variants).
HOOK_TEMPLATES: Dict[str, List[str]] = {
    "pattern_interrupt": [
        "Stop scrolling — {artist} just changed the game with {topic}",
        "You weren't ready for this: {topic} 🔥",
    ],
    "aesthetic_curiosity": [
        "{artist} — {topic}. The vibe is everything ✨",
        "This is what {topic} feels like 🌙",
    ],
    "value_promise": [
        "Here's why {topic} is about to be everywhere",
        "{artist} breaks down {topic} — watch till the end",
    ],
    "bold_claim": [
        "{topic} is the best thing {artist} has ever made. Period.",
        "Nobody is talking about {topic} yet. They will be.",
    ],
    "relatable_story": [
        "{artist} made {topic} for nights like these.",
        "We've all needed something like {topic} 🤍",
    ],
    "curiosity": [
        "🎵 {artist} just dropped something you need to hear — {topic}",
        "There's a reason everyone's on {topic} right now",
    ],
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "for", "of", "to", "in", "on", "at",
    "with", "is", "are", "was", "were", "be", "been", "this", "that", "it",
    "as", "by", "from", "your", "you", "my", "our", "we", "i", "me", "about",
    "just", "new", "now", "out", "get", "make", "made", "into", "up", "via",
    "feat", "ft", "song", "track", "music", "video", "content", "post",
}

_POWER_WORDS = {
    "secret", "proven", "instantly", "exclusive", "free", "now", "never",
    "stop", "first", "best", "viral", "insane", "real", "raw", "unreleased",
    "finally", "limited", "drop", "fire", "everyone", "nobody",
}


# ---------------------------------------------------------------------------
# Generation brief
# ---------------------------------------------------------------------------

@dataclass
class GenerationBrief:
    """The structured, enriched output of the request intelligence layer."""

    modality: str
    platform: str
    intent: str
    intent_label: str
    intent_confidence: float
    audience: str
    tone: str
    keywords: List[str]
    aspect_ratio: str
    layout: str
    hook_style: str
    cta_style: str
    suggested_cta: str
    word_count_target: Tuple[int, int]
    hashtags_target: int
    tempo: str
    candidate_count: int
    temperature: float
    directives: List[str]
    augmented_idea: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Compact, transparent representation returned to API callers."""
        return {
            "modality": self.modality,
            "platform": self.platform,
            "intent": self.intent,
            "intent_label": self.intent_label,
            "intent_confidence": round(self.intent_confidence, 3),
            "audience": self.audience,
            "tone": self.tone,
            "keywords": self.keywords,
            "strategy": {
                "aspect_ratio": self.aspect_ratio,
                "layout": self.layout,
                "hook_style": self.hook_style,
                "cta_style": self.cta_style,
                "word_count_target": list(self.word_count_target),
                "hashtags_target": self.hashtags_target,
                "tempo": self.tempo,
                "candidate_count": self.candidate_count,
                "temperature": self.temperature,
            },
            "directives": self.directives,
            "suggested_cta": self.suggested_cta,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Analysis primitives
# ---------------------------------------------------------------------------

def _norm(text: Optional[str]) -> str:
    return (text or "").strip()


def classify_intent(goal: str, topic: str, extra: str = "") -> Tuple[str, float]:
    """Return (canonical_intent, confidence 0-1).

    Combines an explicit goal alias (strong signal) with keyword scanning over
    the topic/extra text (weaker, corroborating signal).
    """
    goal_l = _norm(goal).lower()
    blob = " ".join([goal_l, _norm(topic).lower(), _norm(extra).lower()])

    scores: Dict[str, float] = {k: 0.0 for k in GOAL_SIGNALS}

    # Explicit goal alias is the strongest signal.
    for alias, intent in GOAL_ALIASES.items():
        if alias in goal_l:
            scores[intent] += 3.0

    # Keyword corroboration from the whole request blob.
    for intent, signals in GOAL_SIGNALS.items():
        for sig in signals:
            if sig in blob:
                scores[intent] += 1.0

    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]
    if best_score <= 0:
        return "drive_engagement", 0.4  # sensible neutral default

    total = sum(scores.values()) or 1.0
    confidence = min(1.0, 0.45 + best_score / (total + 2.0))
    return best_intent, confidence


def extract_keywords(text: str, k: int = 6) -> List[str]:
    """Frequency-ranked salient keywords, stopwords removed, order-stable."""
    # `[^\W_]` matches Unicode word characters (incl. non-Latin scripts) but
    # excludes underscore, so multilingual topics still yield keyword signal.
    tokens = re.findall(r"[^\W_]+", _norm(text).lower(), flags=re.UNICODE)
    freq: Dict[str, int] = {}
    first_pos: Dict[str, int] = {}
    for idx, tok in enumerate(tokens):
        if len(tok) < 3 or tok in _STOPWORDS:
            continue
        if tok not in first_pos:
            first_pos[tok] = idx
        freq[tok] = freq.get(tok, 0) + 1
    ranked = sorted(freq.keys(), key=lambda t: (-freq[t], first_pos[t]))
    return ranked[:k]


def infer_audience(genre: Optional[str], platform: str, intent: str) -> str:
    """Heuristic audience descriptor from genre + platform + intent."""
    genre_l = _norm(genre).lower()
    age = {
        "tiktok": "Gen-Z (16-24)",
        "instagram": "millennials & Gen-Z (18-34)",
        "instagram_reels": "Gen-Z (16-24)",
        "youtube": "broad music fans (18-44)",
        "youtube_shorts": "Gen-Z (16-24)",
        "twitter": "music-savvy early adopters (20-35)",
        "linkedin": "industry & professionals (25-45)",
    }.get(platform, "engaged music fans (18-34)")
    if genre_l:
        return f"{genre_l} listeners — {age}"
    if intent == "drive_conversion":
        return f"high-intent buyers & superfans — {age}"
    return age


def _profile_for(platform: str) -> Dict[str, Any]:
    return PLATFORM_PROFILES.get(platform, PLATFORM_PROFILES["general"])


def _resolve_tone(tone: Optional[str], platform: str, intent: str) -> str:
    if _norm(tone):
        return _norm(tone)
    by_intent = {
        "drive_streams": "energetic",
        "drive_conversion": "confident",
        "grow_followers": "authentic",
        "drive_engagement": "playful",
        "build_awareness": "cinematic",
    }
    return by_intent.get(intent, "authentic")


def _build_directives(brief_bits: Dict[str, Any]) -> List[str]:
    intent = brief_bits["intent"]
    profile_tempo = brief_bits["tempo"]
    hook_style = brief_bits["hook_style"].replace("_", " ")
    cta_style = brief_bits["cta_style"].replace("_", " ")
    lo, hi = brief_bits["word_count"]
    directives = [
        f"Optimise for: {INTENT_LABELS.get(intent, intent)}",
        f"Open with a {hook_style} hook within the first line",
        f"Keep copy {profile_tempo} and roughly {lo}-{hi} words",
        f"Close with a {cta_style} call-to-action",
    ]
    if brief_bits["keywords"]:
        directives.append("Work in keywords: " + ", ".join(brief_bits["keywords"][:4]))
    return directives


def build_brief(
    modality: str,
    platform: str,
    topic: str,
    goal: Optional[str] = None,
    tone: Optional[str] = None,
    genre: Optional[str] = None,
    artist: Optional[str] = None,
    extra: Optional[str] = None,
) -> GenerationBrief:
    """Analyse a request and produce a structured GenerationBrief.

    `platform` should already be normalised by the caller (normalize_platform).
    """
    platform = _norm(platform) or "general"
    topic = _norm(topic)
    extra_text = _norm(extra)

    profile = _profile_for(platform)
    intent, confidence = classify_intent(goal or "", topic, extra_text)
    keywords = extract_keywords(" ".join([topic, extra_text]))
    audience = infer_audience(genre, platform, intent)
    resolved_tone = _resolve_tone(tone, platform, intent)

    directive_bits = {
        "intent": intent,
        "tempo": profile["tempo"],
        "hook_style": profile["hook_style"],
        "cta_style": profile["cta_style"],
        "word_count": profile["word_count"],
        "keywords": keywords,
    }
    directives = _build_directives(directive_bits)

    # Augmented idea fed to the underlying agents to steer better output.
    aug_parts = [topic or "music content"]
    if resolved_tone:
        aug_parts.append(f"tone: {resolved_tone}")
    aug_parts.append(f"goal: {INTENT_LABELS.get(intent, intent)}")
    aug_parts.append(f"audience: {audience}")
    if keywords:
        aug_parts.append("themes: " + ", ".join(keywords[:4]))
    augmented_idea = " | ".join(aug_parts)

    # Candidate count: >1 whenever ranking is cheap relative to render cost.
    # Text/content ranking is essentially free. Image spec generation is a
    # short prompt string (not a pixel render) so it can also afford ranked
    # candidates. Audio/video candidate work happens at the pixel/audio
    # render layer itself (too expensive to multiply), so those stay at 1
    # and instead get scored phrase/spec selection upstream of the render.
    candidate_count = 3 if modality in ("content", "text", "image") else 1

    return GenerationBrief(
        modality=modality,
        platform=platform,
        intent=intent,
        intent_label=INTENT_LABELS.get(intent, intent),
        intent_confidence=confidence,
        audience=audience,
        tone=resolved_tone,
        keywords=keywords,
        aspect_ratio=profile["aspect_ratio"],
        layout=profile["layout"],
        hook_style=profile["hook_style"],
        cta_style=profile["cta_style"],
        suggested_cta=CTA_LIBRARY.get(intent, CTA_LIBRARY["build_awareness"]),
        word_count_target=tuple(profile["word_count"]),
        hashtags_target=int(profile["hashtags"]),
        tempo=profile["tempo"],
        candidate_count=candidate_count,
        temperature=float(profile["temperature"]),
        directives=directives,
        augmented_idea=augmented_idea,
    )


# ---------------------------------------------------------------------------
# Candidate scoring & ranking
# ---------------------------------------------------------------------------

_CTA_KEYWORDS = [
    "click", "follow", "link", "save", "share", "buy", "get", "stream",
    "listen", "subscribe", "comment", "tap", "join", "shop", "watch", "bio",
]


def score_candidate(text: str, brief: GenerationBrief) -> float:
    """Brief-aware quality score (0-100).

    Blends: length fit to the platform window, CTA presence, keyword coverage,
    and first-line hook strength. Deterministic so ranking is stable.
    """
    text = _norm(text)
    if not text:
        return 0.0
    words = text.split()
    n = len(words)

    lo, hi = brief.word_count_target
    half = max(1.0, (hi - lo) / 2.0)
    if lo <= n <= hi:
        length_score = 1.0
    else:
        dist = (lo - n) if n < lo else (n - hi)
        length_score = max(0.0, 1.0 - dist / (half * 2.0))

    low = text.lower()
    cta_score = 1.0 if any(w in low for w in _CTA_KEYWORDS) else 0.0

    if brief.keywords:
        covered = sum(1 for kw in brief.keywords if kw in low)
        keyword_score = covered / len(brief.keywords)
    else:
        keyword_score = 0.5

    first_line = (text.splitlines() or [text])[0].lower()
    hook_score = 0.0
    if any(p in first_line for p in _POWER_WORDS):
        hook_score += 0.5
    if "?" in first_line or "!" in first_line:
        hook_score += 0.25
    if re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", first_line):
        hook_score += 0.15
    if len(first_line.split()) <= 12:
        hook_score += 0.1
    hook_score = min(1.0, hook_score)

    blended = (
        length_score * 0.35
        + cta_score * 0.20
        + keyword_score * 0.20
        + hook_score * 0.25
    )
    return round(min(100.0, blended * 100), 1)


def rank_candidates(
    candidates: List[str], brief: GenerationBrief
) -> List[Tuple[str, float]]:
    """Return candidates sorted best-first with their scores (dedup, stable)."""
    seen = set()
    scored: List[Tuple[str, float]] = []
    for cand in candidates:
        key = _norm(cand)
        if not key or key in seen:
            continue
        seen.add(key)
        scored.append((cand, score_candidate(cand, brief)))
    scored.sort(key=lambda cs: cs[1], reverse=True)
    return scored


def hook_variants(topic: str, artist: str, brief: GenerationBrief) -> List[str]:
    """Deterministic stylistic hook candidates for the brief's hook style."""
    topic = _norm(topic) or "this"
    artist = _norm(artist) or "the artist"
    templates = HOOK_TEMPLATES.get(brief.hook_style, HOOK_TEMPLATES["curiosity"])
    out: List[str] = []
    for tpl in templates:
        try:
            out.append(tpl.format(artist=artist, topic=topic))
        except (KeyError, IndexError):
            continue
    return out


def best_hook(topic: str, artist: str, agent_hook: str, brief: GenerationBrief) -> Tuple[str, float, int]:
    """Pick the best hook among the agent's hook + deterministic variants.

    Returns (hook_text, score, num_candidates_considered).
    """
    candidates = []
    if _norm(agent_hook):
        candidates.append(agent_hook)
    candidates.extend(hook_variants(topic, artist, brief))
    ranked = rank_candidates(candidates, brief)
    if not ranked:
        return agent_hook, 0.0, 0
    return ranked[0][0], ranked[0][1], len(ranked)


def score_scene_phrase(
    phrase: str,
    scene_type: str,
    keywords: Optional[List[str]] = None,
) -> float:
    """Score a raw (pre-personalisation) video scene phrase candidate.

    Applies the same hook/CTA/keyword heuristics used for text ranking,
    tuned per scene type, so video phrase selection is no longer purely
    random — the highest-scoring available candidate wins. Scored on the
    raw template (with `{idea}`/`{artist}`/`{genre}` placeholders intact)
    since personalisation happens after selection.
    """
    text = _norm(phrase)
    if not text:
        return 0.0
    low = text.lower()
    keywords = keywords or []

    # Templates that will bind to this specific idea/artist/genre outrank
    # generic filler that carries no personalisation.
    placeholder_score = 1.0 if re.search(r"\{(idea|artist|genre)\}", phrase) else 0.4

    length = len(text.split())
    length_score = 1.0 if 4 <= length <= 14 else max(0.0, 1.0 - abs(length - 9) / 12.0)

    if scene_type in ("hook", "drop", "build", "chorus"):
        punch_score = 0.0
        if any(p in low for p in _POWER_WORDS):
            punch_score += 0.5
        if "?" in low or "!" in low or "—" in text or "-" in text:
            punch_score += 0.25
        if length <= 10:
            punch_score += 0.25
        punch_score = min(1.0, punch_score)
        blended = placeholder_score * 0.3 + length_score * 0.3 + punch_score * 0.4
    elif scene_type in ("cta", "outro"):
        cta_score = 1.0 if any(w in low for w in _CTA_KEYWORDS) else 0.0
        blended = placeholder_score * 0.3 + length_score * 0.2 + cta_score * 0.5
    else:  # body / verse / bridge / transition
        kw_score = (
            sum(1 for kw in keywords if kw in low) / len(keywords)
            if keywords else 0.5
        )
        blended = placeholder_score * 0.3 + length_score * 0.3 + kw_score * 0.4

    return round(min(100.0, blended * 100), 1)


def rank_scene_phrases(
    pool: List[str],
    scene_type: str,
    keywords: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """Score+sort a pool of raw scene phrase candidates, best first."""
    scored = [(p, score_scene_phrase(p, scene_type, keywords)) for p in pool]
    scored.sort(key=lambda ps: ps[1], reverse=True)
    return scored


def deterministic_candidate(topic: str, artist: str, brief: GenerationBrief) -> str:
    """A fully deterministic raw-topic caption (no model call).

    Used as a quality guardrail: it is ranked alongside model output so that if
    the (undertrained) model is degraded by prompt steering, a clean raw-topic
    candidate can still win.
    """
    variants = hook_variants(topic, artist, brief)
    hook = variants[0] if variants else (_norm(topic) or "New drop")
    body = f"{(_norm(topic) or 'New music').capitalize()} — made for {brief.audience}."
    return f"{hook}\n{body}\n{brief.suggested_cta}"
