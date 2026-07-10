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
    # Hashtag counts follow the 2025-26 quality-over-quantity norms curated
    # in ai_model/content_playbook.py (Instagram hard-caps at 5).
    "tiktok": {
        "aspect_ratio": "9:16", "layout": "portrait_9_16", "word_count": (12, 40),
        "hook_style": "pattern_interrupt", "cta_style": "comment_engage",
        "hashtags": 4, "tempo": "fast", "temperature": 0.95,
    },
    "instagram": {
        "aspect_ratio": "1:1", "layout": "square_1_1", "word_count": (20, 60),
        "hook_style": "aesthetic_curiosity", "cta_style": "save_share",
        "hashtags": 5, "tempo": "medium", "temperature": 0.9,
    },
    "instagram_reels": {
        "aspect_ratio": "9:16", "layout": "portrait_9_16", "word_count": (12, 40),
        "hook_style": "pattern_interrupt", "cta_style": "save_share",
        "hashtags": 5, "tempo": "fast", "temperature": 0.95,
    },
    "youtube": {
        "aspect_ratio": "16:9", "layout": "landscape_16_9", "word_count": (40, 90),
        "hook_style": "value_promise", "cta_style": "subscribe",
        "hashtags": 3, "tempo": "steady", "temperature": 0.85,
    },
    "youtube_shorts": {
        "aspect_ratio": "9:16", "layout": "portrait_9_16", "word_count": (12, 40),
        "hook_style": "pattern_interrupt", "cta_style": "subscribe",
        "hashtags": 3, "tempo": "fast", "temperature": 0.95,
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
        "hashtags": 4, "tempo": "medium", "temperature": 0.9,
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
    # Distinctive-copy inputs: the freeform narrative directive plus the
    # artist-specific facts used to make copy unmistakably about THIS release.
    narrative: str = ""
    artist: str = ""
    track: str = ""
    themes: List[str] = field(default_factory=list)
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
    # Research-backed directives from the content playbook (world knowledge
    # distilled from published engagement studies — internal steering only).
    # Borrowed knowledge: gated by the same self-retirement contract as the
    # quality buffer, so ALL outside influence fades out together as the
    # platform's own corpus grows.
    try:
        from ai_model.quality_awareness import self_sufficiency
        if not self_sufficiency()["retired"]:
            from ai_model.content_playbook import brief_directives as _pb
            directives.extend(_pb(intent))
    except Exception:  # noqa: BLE001 - playbook must never break a brief
        pass
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
    narrative: Optional[str] = None,
    track: Optional[str] = None,
    themes: Optional[List[str]] = None,
) -> GenerationBrief:
    """Analyse a request and produce a structured GenerationBrief.

    `platform` should already be normalised by the caller (normalize_platform).
    """
    platform = _norm(platform) or "general"
    topic = _norm(topic)
    extra_text = _norm(extra)
    narrative_text = _norm(narrative)
    # Strip the imperative lead-in ("write a caption about …") BEFORE mining
    # keywords, or instruction verbs ("write", "caption") pollute the themes.
    narrative_clean = _narrative_clause(narrative_text)
    explicit_themes = [_norm(t) for t in (themes or []) if _norm(t)]

    profile = _profile_for(platform)
    intent, confidence = classify_intent(goal or "", topic, extra_text)
    keywords = extract_keywords(
        " ".join([topic, extra_text, narrative_clean, " ".join(explicit_themes)])
    )
    # Caller-supplied themes are authoritative; extracted keywords fill the rest.
    merged_themes = list(dict.fromkeys([*explicit_themes, *keywords]))
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

    # ── Quality awareness buffer (temporary, self-retiring) ────────────────
    # World-studied chart/content patterns blended in while the own pdim
    # corpus is small. Never-raise + TTL-cached: cannot break or slow a brief.
    notes: List[str] = []
    try:
        from ai_model.quality_awareness import brief_enrichment
        _enr = brief_enrichment()
        if _enr:
            directives.append(_enr["directive"])
            notes.append(_enr["note"])
    except Exception:
        pass

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
        narrative=narrative_text,
        artist=_norm(artist),
        track=_norm(track) or topic,
        themes=merged_themes,
        notes=notes,
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

    # Research-backed Hook->Value->CTA structure + high-arousal bonus
    # (content_playbook: HVC captions earn ~23% more engagement; hook must
    # land inside the first 125 visible characters).
    try:
        from ai_model.content_playbook import structure_score as _pb_structure
        struct_score = _pb_structure(text)
    except Exception:  # noqa: BLE001 - scoring must never break ranking
        struct_score = 0.0

    blended = (
        length_score * 0.30
        + cta_score * 0.15
        + keyword_score * 0.20
        + hook_score * 0.20
        + struct_score * 0.15
    )
    score = blended * 100

    # Garbled model output (glued tokens, letter-digit fusions) must never
    # outrank a clean composed candidate — apply a decisive penalty. The
    # whitelist carries the brief's full raw request context (augmented_idea
    # includes the raw topic) so legitimate alphanumeric names (e.g. an
    # artist called "Frequency82") are never penalised.
    if looks_garbled(text, whitelist=f"{' '.join(brief.keywords)} {brief.augmented_idea}"):
        score -= 40.0

    return round(min(100.0, max(0.0, score)), 1)


def rank_candidates(
    candidates: List[str], brief: GenerationBrief
) -> List[Tuple[str, float]]:
    """Return candidates sorted best-first with their scores (dedup, stable)."""
    # Stage 8 constraint enforcement (post-generation): each candidate is run
    # through the content-safety policy so the stored text is already redacted /
    # refused, and unsafe candidates are penalised so a clean variant wins.
    try:
        from ai_model.safety import get_safety
        _safety = get_safety()
    except Exception:
        _safety = None

    seen = set()
    scored: List[Tuple[str, float]] = []
    for cand in candidates:
        safe_cand, pen = cand, 0.0
        if _safety is not None:
            # Screen once (counters increment once), reuse for text + penalty.
            res = _safety.enforce(cand)
            safe_cand = res.text
            pen = _safety.penalty_of(res)
        key = _norm(safe_cand)
        if not key or key in seen:
            continue
        seen.add(key)
        scored.append((safe_cand, score_candidate(safe_cand, brief) + pen))
    scored.sort(key=lambda cs: cs[1], reverse=True)
    return scored


def hook_variants(
    topic: str,
    artist: str,
    brief: GenerationBrief,
    weave_active: Optional[bool] = None,
) -> List[str]:
    """Deterministic stylistic hook candidates for the brief's hook style."""
    if weave_active is None:
        weave_active = _weave_active()
    topic = _norm(topic) or "this"
    artist = _norm(artist) or "the artist"
    templates = HOOK_TEMPLATES.get(brief.hook_style, HOOK_TEMPLATES["curiosity"])
    out: List[str] = []
    for tpl in templates:
        try:
            out.append(tpl.format(artist=artist, topic=topic))
        except (KeyError, IndexError):
            continue

    # Request-specific hooks: reference the actual story/track/themes so the
    # hook can only belong to THIS release. This is hand-authored borrowed
    # weave, so it retires with the corpus (see _weave_active) — once the model
    # can render the direction itself, these stop competing.
    if weave_active:
        track = _norm(brief.track) or topic
        themes = [t for t in (brief.themes or []) if _norm(t)][:2]
        clause = _narrative_clause(brief.narrative, limit=70)
        if clause:
            out.append(f"{clause}?")
            if track:
                out.append(f"{clause} — and then came {track}.")
        if themes and artist.lower() != "the artist":
            out.append(f"Nobody turns {themes[0]} into a record like {artist}.")
    return out


def best_hook(
    topic: str,
    artist: str,
    agent_hook: str,
    brief: GenerationBrief,
    weave_active: Optional[bool] = None,
) -> Tuple[str, float, int]:
    """Pick the best hook among the agent's hook + deterministic variants.

    Returns (hook_text, score, num_candidates_considered).
    """
    candidates = []
    if _norm(agent_hook):
        candidates.append(agent_hook)
    candidates.extend(hook_variants(topic, artist, brief, weave_active=weave_active))
    # Quality-buffer hooks compete in the same ranking (empty once retired).
    # This pool already blends the harvester's live-chart templates with the
    # research playbook's archetypes (see quality_awareness.scene_phrases).
    try:
        from ai_model.quality_awareness import hook_candidates as _qa_hooks
        candidates.extend(_qa_hooks(topic, artist))
    except Exception:
        pass
    ranked = rank_candidates(candidates, brief)
    if not ranked:
        return agent_hook, 0.0, 0
    # If a quality-buffer hook wins, its template graduates into the own
    # corpus so text generation also progresses buffer retirement.
    try:
        from ai_model.quality_awareness import graduate_hook
        graduate_hook(ranked[0][0])
    except Exception:
        pass
    return ranked[0][0], ranked[0][1], len(ranked)


def best_image_headline(
    topic: str,
    artist: str,
    agent_headline: str,
    brief: GenerationBrief,
) -> Tuple[str, float, int]:
    """Pick the best on-image headline among the agent's headline + the
    quality-buffer's candidates — mirrors best_hook (text) and the video
    scene sampler's tier-1 buffer blend, so image generation runs the same
    "borrowed knowledge competes, winner graduates" pattern as the other
    two modalities.

    Returns (headline_text, score, num_candidates_considered).
    """
    candidates = []
    if _norm(agent_headline):
        candidates.append(agent_headline)
    # Quality-buffer headlines compete in the same ranking (empty once
    # retired) — same borrowed pool as text hooks, tracked separately so
    # graduation counts toward image generation's own corpus.
    try:
        from ai_model.quality_awareness import image_headline_candidates as _qa_headlines
        candidates.extend(_qa_headlines(topic, artist))
    except Exception:
        pass
    ranked = rank_candidates(candidates, brief)
    if not ranked:
        return agent_headline, 0.0, 0
    # If a quality-buffer headline wins, its template graduates into image
    # generation's own corpus so it also progresses buffer retirement.
    try:
        from ai_model.quality_awareness import graduate_image_headline
        graduate_image_headline(ranked[0][0])
    except Exception:
        pass
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


# ---------------------------------------------------------------------------
# Garble detection — undertrained-model output guard
# ---------------------------------------------------------------------------
# The in-house transformer sometimes emits glued-together tokens ("beingpre-save",
# "beingstarting", "Frequency82") that pass length/repetition checks but read as
# gibberish in user-facing overlays. This detector is deterministic, dependency-
# free and whitelist-aware (words from the request itself are never flagged).

# Function words that essentially never form the PREFIX of a longer real English
# word with 4+ extra chars (e.g. "being"+"pre-save"). Chosen so real words like
# "everything", "startling" or "forever" are NOT flagged.
_GLUE_PREFIXES = ("being", "because", "would", "their", "going", "about")


def looks_garbled(text: str, whitelist: str = "") -> bool:
    """True when text shows glued-token / letter-digit-fusion artefacts.

    ``whitelist`` should carry the raw request context (topic, artist,
    awareness) — any word appearing there is trusted verbatim.
    """
    if not text:
        return False
    wl = set(re.findall(r"[a-z0-9]+", whitelist.lower()))
    words = re.findall(r"[A-Za-z0-9'’\-]+", text)
    if not words:
        return False
    bad = 0
    for w in words:
        base = w.strip("'’-").lower()
        core = re.sub(r"[^a-z0-9]", "", base)
        if not core or core in wl:
            continue
        # Implausibly long single token (mashed words)
        if len(core) > 14:
            bad += 1
            continue
        # Letters fused with a trailing multi-digit run ("frequency82")
        if re.search(r"[a-z]{4,}\d{2,}$", core):
            bad += 1
            continue
        # Function word glued onto a content word ("beingpre-save")
        for p in _GLUE_PREFIXES:
            if core.startswith(p) and len(core) - len(p) >= 4:
                bad += 1
                break
    return bad >= 2 or (bad / len(words)) > 0.2


def deterministic_candidate(
    topic: str,
    artist: str,
    brief: GenerationBrief,
    weave_active: Optional[bool] = None,
) -> str:
    """A fully deterministic raw-topic caption (no model call).

    Used as a quality guardrail: it is ranked alongside model output so that if
    the (undertrained) model is degraded by prompt steering, a clean raw-topic
    candidate can still win. ``weave_active`` threads the per-request retirement
    decision so callers get one consistent gate; when omitted it self-computes.
    """
    variants = hook_variants(topic, artist, brief, weave_active=weave_active)
    hook = variants[0] if variants else (_norm(topic) or "New drop")
    body = f"{(_norm(topic) or 'New music').capitalize()} — made for {brief.audience}."
    return f"{hook}\n{body}\n{brief.suggested_cta}"


# ---------------------------------------------------------------------------
# Intelligence-driven caption composer
# ---------------------------------------------------------------------------

# Vocabulary for stripping imperative lead-ins from a creative directive. A
# procedural stripper (below) is used instead of one big regex because real
# directives phrase the lead-in many ways — with a connector ("write a caption
# ABOUT x"), with a gerund ("write a caption HYPING x"), or with neither
# ("hype the new single x") — and a single pattern reliably handles only the
# first form. Leaking the imperative verbatim into a caption is the failure
# this exists to prevent.
_LEADIN_PREAMBLE = {
    "please", "kindly", "can", "could", "would", "you", "i", "to", "help",
    "me", "us", "lets", "let", "let's", "'d", "d", "just", "now",
}
_IMPERATIVE_VERBS = {
    "write", "create", "make", "generate", "draft", "compose", "craft",
    "produce", "tell", "share", "describe", "hype", "announce", "promote",
    "introduce", "tease", "post", "caption", "give", "need", "want", "cook",
    "whip", "put",
}
_LEADIN_DETERMINERS = {"a", "an", "the", "some", "me", "us", "my", "your", "up"}
_ARTIFACT_NOUNS = {
    "caption", "captions", "post", "posts", "tweet", "tweets", "copy",
    "content", "blurb", "hook", "hooks", "line", "lines", "bio", "description",
    "message", "thread", "piece", "something", "one", "text", "words",
}
_LEADIN_CONNECTORS = {
    "about", "for", "on", "that", "which", "to", "of", "around", "regarding",
    "hyping", "celebrating", "announcing", "promoting", "introducing",
    "teasing", "describing", "explaining", "highlighting", "showcasing",
    "spotlighting", "covering", "detailing", "capturing",
}


def _leadin_norm(token: str) -> str:
    return re.sub(r"[^a-z']", "", token.lower())


def _strip_instruction_leadin(s: str) -> str:
    """Remove a leading imperative directive clause, keeping the actual content.

    Handles connector lead-ins ("write a caption about X"), gerund lead-ins
    ("write an IG caption hyping X"), and connector-less ones ("hype the new
    single X"). Returns the input unchanged when it does not begin with an
    imperative directive (so genuine narrative prose is never mangled).
    """
    tokens = s.split()
    n = len(tokens)
    i = 0
    while i < n and _leadin_norm(tokens[i]) in _LEADIN_PREAMBLE:
        i += 1
    if i >= n or _leadin_norm(tokens[i]) not in _IMPERATIVE_VERBS:
        return s  # not an imperative directive — leave narrative prose intact
    j = i + 1
    while j < n and _leadin_norm(tokens[j]) in _LEADIN_DETERMINERS:
        j += 1
    # Scan a short window for an artifact noun or connector; cut just past it.
    cut = None
    for k in range(j, min(n, j + 5)):
        w = _leadin_norm(tokens[k])
        if w in _LEADIN_CONNECTORS:
            cut = k + 1
            break
        if w in _ARTIFACT_NOUNS:
            cut = k + 1
            if cut < n and _leadin_norm(tokens[cut]) in _LEADIN_CONNECTORS:
                cut += 1
            break
    if cut is None:
        cut = j  # no artifact/connector — drop just the verb + determiners
    remainder = " ".join(tokens[cut:]).strip(" ,:;-–—")
    return remainder or s


def _narrative_clause(narrative: str, limit: int = 140) -> str:
    """Turn a freeform creative directive into a usable subject clause.

    Strips imperative lead-ins ("write a caption about ...", "make a post
    hyping ...", "hype the new single ...") so the directive's *content* — not
    the instruction addressed to the model — is what gets woven into copy, then
    trims to a clean clause boundary. Returns "" when nothing usable remains.
    """
    s = _norm(narrative)
    if not s:
        return ""
    # Strip repeatedly (max 2) in case of a stacked directive ("please write a
    # caption. make it about ...").
    for _ in range(2):
        stripped = _strip_instruction_leadin(s)
        if stripped == s:
            break
        s = stripped
    s = s or _norm(narrative)
    if len(s) > limit:
        cut = s[:limit]
        for sep in (". ", "; ", ", ", " — ", " - "):
            idx = cut.rfind(sep)
            if idx > limit * 0.5:
                cut = cut[:idx]
                break
        s = cut
    s = s.strip(" ,;.—-")
    if not s:
        return ""
    return s[:1].upper() + s[1:]


def _weave_active() -> bool:
    """Whether the hand-authored narrative/theme weave should still compete.

    The weave is borrowed knowledge — a crutch for rendering the caller's
    direction while the in-house model is undertrained. It retires on the SAME
    self-sufficiency contract as every other borrowed source (quality-buffer
    hooks, playbook CTAs): once the own-corpus has graduated, the
    model/awareness path carries composition unassisted and the weave stops
    competing. Without this gate the weave would out-score the model forever
    and silently pin the system to templates, defeating the retirement design.
    """
    try:
        from ai_model.quality_awareness import self_sufficiency
        return not self_sufficiency()["retired"]
    except Exception:  # noqa: BLE001 - never let the buffer break composition
        return True


def awareness_from_direction(
    narrative: str,
    themes: Optional[List[str]] = None,
) -> str:
    """Serialise the caller's creative direction into awareness signal lines the
    ScriptAgent's parsers recognise, so the awareness *bridge* (model
    conditioning + awareness-composed fallback) works from the user's own
    direction — not only external trend data.

    The narrative becomes a top-priority ``[HIGH]`` signal (it leads both the
    awareness hook and the body's opening clause because parsed signals keep
    document order); themes become a single ``•`` recommendation bullet.

    Two guards, both load-bearing:
    * The narrative is run through ``_narrative_clause`` first — the awareness
      parsers do NO imperative-lead-in stripping, so an unstripped "write a
      caption about ..." would leak verbatim as a quotable signal.
    * Themes are emitted as a bullet, NEVER as ``#hashtags`` — the distribution
      agent harvests awareness hashtags into a persistent, platform-shared pool
      (``dist:hashtags:*``), so user-specific themes as tags would resurface
      for unrelated artists.
    Colons are neutralised because the hook/body parsers ``split(":")[0]`` and
    would otherwise drop everything after the first colon. Returns "" when
    there is no usable direction.
    """
    lines: List[str] = []
    clause = _narrative_clause(narrative)
    if clause:
        lines.append(f"[HIGH] {clause.replace(':', ' —')}")
    clean_themes = [t.strip() for t in (themes or []) if _norm(t)][:3]
    if clean_themes:
        if len(clean_themes) == 1:
            phrase = clean_themes[0]
        else:
            phrase = ", ".join(clean_themes[:-1]) + " and " + clean_themes[-1]
        lines.append(f"• Woven around {phrase.replace(':', ' —')}")
    return "\n".join(lines)


def _body_candidates(
    topic: str,
    brief: GenerationBrief,
    genre: Optional[str] = None,
    brand_voice: Optional[str] = None,
    agent_body: str = "",
    weave_active: Optional[bool] = None,
) -> List[str]:
    """Value-line candidates composed FROM the brief (keywords, audience,
    tone, genre) instead of echoing the raw topic back as the body."""
    if weave_active is None:
        weave_active = _weave_active()
    topic_n = _norm(topic)
    genre_n = _norm(genre) or "music"
    tone = brief.tone or "authentic"
    out: List[str] = []

    # Agent body competes — unless it is just a raw-topic echo (the exact
    # failure mode this composer exists to fix). Compare on a strong
    # normalisation (alphanumerics only) so punctuation/emoji-only edits
    # of the topic are still rejected as echoes.
    def _skeleton(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    agent_n = _norm(agent_body)
    if agent_n and _skeleton(agent_n) != _skeleton(topic_n):
        out.append(agent_n)

    # ── Distinctive candidates (preferred) ────────────────────────────────────
    # Weave the freeform narrative directive + artist-specific facts (name,
    # track, themes) so the body is unmistakably about THIS release rather than
    # any track with the title swapped in. These are emitted first and, when
    # present, they SUPPRESS the generic keyword/audience templates below — the
    # generic "Every second of this leans into X and Y" line is exactly the
    # placeholder-feel copy this weave exists to replace, so it must not be
    # allowed to out-score genuine, release-specific material.
    #
    # This is hand-authored borrowed weave: it retires with the corpus (see
    # _weave_active). Once the own-corpus has graduated, the model/awareness
    # path renders the direction itself and this stops competing — the generic
    # fallback below remains only as a minimal, low-scoring safety net.
    has_distinctive = False
    if weave_active:
        artist = _norm(brief.artist)
        has_artist = bool(artist) and artist.lower() != "the artist"
        track = _norm(brief.track) or topic_n
        narrative = _narrative_clause(brief.narrative)
        themes = [t for t in (brief.themes or []) if _norm(t)][:3]

        if narrative:
            anchor = track or "this one"
            out.append(f"{narrative} — that's {anchor}.")
            if has_artist:
                lowered = narrative[:1].lower() + narrative[1:]
                out.append(f"{artist} put everything into this: {lowered}.")
            elif len(themes) >= 2:
                out.append(f"{narrative}. {themes[0].capitalize()}, {themes[1]}, no filler.")
        if has_artist and track:
            if len(themes) >= 2:
                out.append(
                    f"{track} is {artist} at their most {themes[0]} — {themes[1]} in every bar."
                )
            else:
                out.append(f"{track} is the {artist} record you didn't know you needed.")
        if len(themes) >= 2:
            out.append(
                f"{themes[0].capitalize()} meets {themes[1]} — {tone} {genre_n} for {brief.audience}."
            )

        has_distinctive = bool(out) and (bool(narrative) or (has_artist and bool(track))
                                         or len(themes) >= 2)

    if not has_distinctive:
        # Fallback: generic brief-reading lines for sparse requests (no
        # narrative, no artist/track, no themes to anchor on).
        kws = [k for k in brief.keywords if k][:2]
        if len(kws) == 2:
            out.append(
                f"Every second of this leans into {kws[0]} and {kws[1]} — "
                f"{tone} from the first bar."
            )
        elif len(kws) == 1:
            out.append(f"Built around one thing: {kws[0]} — {tone}, no filler.")
        out.append(
            f"{tone.capitalize()} {genre_n} made for {brief.audience} — "
            f"no skips, all intent."
        )

    # Brand voice as copy, when the caller supplied one (competes in both modes).
    bv = _norm(brand_voice)
    if bv:
        out.append(f"{bv.rstrip('.')}. This is what that sounds like.")

    return out or [f"{(topic_n or 'New music').capitalize()} — made for {brief.audience}."]


def _cta_candidates(topic: str, brief: GenerationBrief, agent_cta: str = "") -> List[str]:
    """CTA candidates: agent output + intent CTA + research-playbook bank."""
    out: List[str] = []
    if _norm(agent_cta):
        out.append(_norm(agent_cta))
    out.append(brief.suggested_cta)
    # Research-playbook CTAs are borrowed world knowledge — same retirement
    # contract as directives/hooks: once the platform's own corpus graduates
    # (self-sufficiency retired), the playbook stops contributing.
    try:
        from ai_model.quality_awareness import self_sufficiency
        if not self_sufficiency()["retired"]:
            from ai_model.content_playbook import cta_candidates as _pb_ctas
            out.extend(_pb_ctas(brief.intent, topic))
    except Exception:  # noqa: BLE001 - playbook must never break composition
        pass
    return out


def compose_caption(
    topic: str,
    artist: str,
    brief: GenerationBrief,
    genre: Optional[str] = None,
    brand_voice: Optional[str] = None,
    agent_hook: str = "",
    agent_body: str = "",
    agent_cta: str = "",
    variants: int = 1,
) -> Dict[str, Any]:
    """Compose the best caption FROM the brief's intelligence.

    The hook is ranked across agent output + stylistic variants + awareness
    hooks (best_hook); bodies are composed from the brief's keywords,
    audience and strategy rather than echoing the raw topic; CTAs come from
    the agent, the intent library and the research playbook. Every complete
    hook/body/CTA combination is scored as a full caption (structure-aware
    via score_candidate) and the best one wins. Deterministic.
    """
    # Compute the retirement gate ONCE per caption and thread it down, so the
    # weave's self-sufficiency probe doesn't hit the corpus store repeatedly.
    weave_active = _weave_active()
    hook, hook_score, hooks_considered = best_hook(topic, artist, agent_hook, brief,
                                                   weave_active=weave_active)

    bodies = _body_candidates(topic, brief, genre=genre,
                              brand_voice=brand_voice, agent_body=agent_body,
                              weave_active=weave_active)

    # The narrative clause can surface in both the hook and a body candidate;
    # drop bodies that merely restate the chosen hook so the two don't echo.
    # Compare on shared skeleton prefix length (the two sentences diverge only
    # after the common opening clause), not exact-prefix equality.
    def _skel(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    def _shared_prefix(a: str, b: str) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    _hook_skel = _skel(hook)
    _narr_skel = _skel(_narrative_clause(brief.narrative))
    _narr_in_hook = bool(_narr_skel) and _narr_skel in _hook_skel

    def _echoes_hook(b: str) -> bool:
        bs = _skel(b)
        if _hook_skel and _shared_prefix(bs, _hook_skel) >= 30:
            return True
        # If the narrative clause is already in the hook, don't repeat it in the
        # body — force a different distinctive candidate (artist/track/theme).
        if _narr_in_hook and _narr_skel in bs:
            return True
        return False

    _deduped = [b for b in bodies if not _echoes_hook(b)]
    bodies = _deduped or bodies

    ctas = _cta_candidates(topic, brief, agent_cta=agent_cta)

    # Score every complete hook/body/CTA combination, keep them all ranked.
    scored: List[Tuple[str, str, str, float]] = []
    for body in bodies:
        for cta in ctas:
            caption = f"{hook}\n\n{body}\n\n{cta}"
            s = score_candidate(caption, brief)
            scored.append((caption, body, cta, s))
    # Deterministic ordering: score desc, then caption text as a stable tiebreak.
    scored.sort(key=lambda t: (-t[3], t[0]))
    best: Tuple[str, str, str, float] = scored[0] if scored else ("", "", "", -1.0)

    # Build up to `variants` distinct alternatives (by body text) for callers who
    # want A/B/C options. variant[0] is always the winner. Deterministic.
    variant_list: List[Dict[str, Any]] = []
    seen_bodies: set = set()
    want = max(1, int(variants or 1))
    for cap, body, cta, s in scored:
        skel = _skel(body)
        if skel in seen_bodies:
            continue
        seen_bodies.add(skel)
        variant_list.append({
            "caption": cap, "hook": hook, "body": body, "cta": cta,
            "score": round(s, 2), "char_count": len(cap),
        })
        if len(variant_list) >= want:
            break

    return {
        "caption": best[0],
        "hook": hook,
        "body": best[1],
        "cta": best[2],
        "caption_score": best[3],
        "hook_score": hook_score,
        "hooks_considered": hooks_considered,
        "bodies_considered": len(bodies),
        "ctas_considered": len(ctas),
        "variants": variant_list,
    }
