"""
DatasetSampler — scene text generation backed by three tiers of real data:

  1. **pdim corpus** (`mb:phrases:<scene_type>`) — live phrases accumulated from
     prior generations and ingestor feedback.  Grows automatically over time.
  2. **Awareness signals** — live industry context injected by the Node.js layer
     that can contribute additional topical phrases for this video.
  3. **Built-in phrase banks** — curated seed corpus (same vocabulary used for
     model training).  Used only when both pdim and awareness yield nothing.

A `_UsedSet` guard prevents the same phrase appearing twice in one video.
All tiers return `source = "datasets"`.
"""
from __future__ import annotations

import random
import re
from typing import Dict, List, Tuple


# ── phrase bank helpers ────────────────────────────────────────────────────────

def _f(template: str, **kw) -> str:
    """Safe format — leaves unknown {keys} in place."""
    try:
        return template.format(**kw)
    except (KeyError, ValueError):
        return template


class _UsedSet:
    """Prevent duplicate phrases within one video."""
    def __init__(self):
        self._seen: set[str] = set()

    def pick(self, pool: List[str]) -> str:
        candidates = [p for p in pool if p not in self._seen]
        if not candidates:
            candidates = pool          # all used — allow repeats rather than crash
        choice = random.choice(candidates)
        self._seen.add(choice)
        return choice


# ── per-scene-type phrase pools (seed / fallback corpus) ──────────────────────

HOOK_PHRASES: List[str] = [
    "Stop scrolling — you need to hear {idea}",
    "Finally. {artist} delivers {idea}",
    "The {genre} wave just changed with {idea}",
    "This is the drop everyone's been waiting for",
    "POV: you just found your new favourite song",
    "I wasn't ready for this one — {idea}",
    "Your playlist needs {idea} right now",
    "Listen before it gets big: {idea}",
    "The sound everyone's been sleeping on",
    "This song just broke my algorithm",
    "New era. New sound. {artist} is back",
    "{artist} — {idea} — out now",
    "The record that started it all",
    "Turn it up. You'll thank me later",
    "This {genre} hit is everywhere for a reason",
    "Don't skip this one",
    "Real talk — {idea} goes hard",
    "The track that started the conversation",
    "This is what passion sounds like",
    "Before the world found out, we knew: {artist}",
    "{idea} — stream it everywhere",
    "The moment you've been waiting for is here",
]

VERSE_PHRASES: List[str] = [
    "Every late night and early morning went into this",
    "The {genre} vibes on this are real",
    "Been building to this moment for months",
    "Stream {idea} on all platforms — link in bio",
    "Every word in this track means something",
    "The story behind {idea} is bigger than the music",
    "This one's for everyone who believed before anyone else did",
    "Real emotion, real production, real {genre}",
    "From the studio to the streets — {idea} is live",
    "Every bar, every beat — crafted for this moment",
    "The grind never stopped. {idea} is the proof",
    "Low key, this might be the best thing {artist} has dropped",
    "If you know, you know. {idea} hits different",
    "Built this from scratch. Every layer, every sample",
    "The culture needed this record",
    "When the beat drops, everything makes sense",
    "Shoutout everyone who streamed from day one",
    "This record was made for late nights and big dreams",
    "Pure {genre} energy — no compromise",
    "{artist} went all in on {idea}",
    "No features. No gimmicks. Just the music",
    "The sound you've been looking for is finally here",
]

CHORUS_PHRASES: List[str] = [
    "Play it loud — feel every word",
    "This is the moment everything changed",
    "The chorus you won't be able to get out of your head",
    "When this part hits, you know it's special",
    "{artist} made something timeless with {idea}",
    "This hook goes on repeat — you've been warned",
    "The biggest record of the year starts here",
    "Turn it up, close your eyes, feel it",
    "This chorus was built for arenas",
    "Everyone knows the words by now",
    "The melody that defines the moment",
    "Drop everything and press play",
]

BUILD_PHRASES: List[str] = [
    "The energy keeps rising — {idea} is building",
    "Something big is coming. You can feel it",
    "The build-up before the moment that changes everything",
    "Every second leading to this",
    "Patience. The payoff is worth it",
    "The tension before the release",
    "Watch this space — {artist} is not done yet",
    "Building toward something special",
    "The anticipation is real for {idea}",
    "This is what momentum sounds like",
]

DROP_PHRASES: List[str] = [
    "When the bass hits — you'll know",
    "The drop that made everyone stop scrolling",
    "This is the moment the track takes over",
    "The beat switch nobody saw coming",
    "Full send — {idea} does not hold back",
    "The 808 that started a movement",
    "Nobody was ready for this drop",
    "The loudest moment on the record",
    "When {genre} hits its peak — this is it",
    "The drop heard around the timeline",
]

BRIDGE_PHRASES: List[str] = [
    "Every struggle made this moment possible",
    "This one's for the believers — you know who you are",
    "The bridge that reframes everything",
    "When {artist} gets real, the whole room goes quiet",
    "This part of {idea} hits the hardest",
    "The moment of truth in the record",
    "Not everything is about hype — some of it is about heart",
    "The real story behind {idea} comes out right here",
    "Stripped back, raw, honest — this is {artist}",
    "The verse that made it personal",
]

OUTRO_PHRASES: List[str] = [
    "The record that keeps giving — {idea}",
    "Thank you for listening to {idea}",
    "This was just the beginning — {artist} has more coming",
    "Stream {idea} everywhere music lives",
    "The outro you didn't want to end",
    "Replay. Repeat. Share with someone who gets it",
    "The journey continues — stay locked in",
    "From {artist} to you — thank you for the love",
    "The outro sets up what's next",
    "One down. Many more to come",
]

TRANSITION_PHRASES: List[str] = [
    "And now — the next chapter of {idea}",
    "Keep watching — it only gets better",
    "The energy shifts here",
    "From one vibe to the next",
    "Seamless. That's the {artist} difference",
    "The scene changes. The quality doesn't",
    "What comes next will surprise you",
]

CTA_PHRASES: Dict[str, List[str]] = {
    "youtube":         ["Like, subscribe, and hit the bell for more", "Subscribe for daily {genre} content"],
    "tiktok":          ["Follow for more fire content — link in bio", "Duet this if you feel it"],
    "instagram":       ["Double tap if you vibe with this — save for later", "Follow {artist} for more drops"],
    "facebook":        ["Share with someone who needs to hear {idea}", "Like the page for more releases"],
    "twitter":         ["RT if this goes hard — drop your take below", "Quote tweet with your reaction"],
    "linkedin":        ["Follow for music industry insights and new drops", "Comment your thoughts below"],
    "google_business": ["Visit the site and stream {idea} today", "Check the link for tickets and merch"],
    "threads":         ["Repost this if it hits different", "Drop a reply — what do you think of {idea}?"],
    "_default":        ["Stream {idea} on all platforms now", "Follow {artist} — more music coming soon"],
}

# Scene type → primary phrase pool
_POOL_MAP: Dict[str, List[str]] = {
    "hook":       HOOK_PHRASES,
    "body":       VERSE_PHRASES,
    "verse":      VERSE_PHRASES,
    "chorus":     CHORUS_PHRASES,
    "build":      BUILD_PHRASES,
    "drop":       DROP_PHRASES,
    "bridge":     BRIDGE_PHRASES,
    "outro":      OUTRO_PHRASES,
    "transition": TRANSITION_PHRASES,
    "cta":        [],              # handled separately with platform lookup
}

# pdim key prefix for stored phrase corpora
_PDIM_PHRASE_PREFIX = "phrases"


def _personalise(raw: str, idea: str, genre: str, tone: str,
                 platform: str, artist: str) -> str:
    """Substitute context tokens into a phrase template."""
    text = _f(raw,
              idea=idea or "the new drop",
              genre=genre or "music",
              tone=tone or "energetic",
              platform=platform or "all platforms",
              artist=artist or "the artist")
    # Strip any remaining un-substituted {tokens}
    text = re.sub(r"\{[^}]+\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _trim(text: str, max_words: int = 10) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    short = " ".join(words[:max_words])
    m = re.search(r"[.!?]", short)
    return short[: m.start() + 1].strip() if m else short


# ── pdim corpus access ────────────────────────────────────────────────────────

def _pdim_fetch_phrases(scene_type: str, count: int = 20) -> List[str]:
    """
    Fetch stored phrases for this scene type from pdim.
    Returns [] if pdim is offline or no phrases have been accumulated yet.
    """
    try:
        from storage_client import get_storage
        store = get_storage()
        key = f"{_PDIM_PHRASE_PREFIX}:{scene_type}"
        raw = store.lrange(key, 0, count - 1)
        return [r for r in (raw or []) if isinstance(r, str) and len(r) > 5]
    except Exception:
        return []


def _pdim_push_phrase(scene_type: str, phrase: str) -> None:
    """
    Store a successfully-used phrase back to pdim to grow the live corpus.
    Capped at 256 entries per scene type.
    """
    if not phrase or len(phrase) < 5:
        return
    try:
        from storage_client import get_storage
        store = get_storage()
        key = f"{_PDIM_PHRASE_PREFIX}:{scene_type}"
        existing = store.lrange(key, 0, -1) or []
        if phrase not in existing:
            store.lpush(key, phrase)
            store.ltrim(key, 0, 255)
    except Exception:
        pass


def _awareness_phrases_for_scene(
    awareness: str,
    scene_type: str,
    idea: str,
    genre: str,
    artist: str,
) -> List[str]:
    """
    Extract live context phrases from the awareness string that fit this scene.
    These are actionable signals turned into video-ready copy.
    """
    if not awareness:
        return []

    phrases: List[str] = []

    for line in awareness.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("==="):
            continue

        # Signal headlines → good for hook/build scenes
        m = re.match(r"\[(HIGH|MEDIUM)\]\s+(.+)", stripped)
        if m and scene_type in ("hook", "build", "drop"):
            headline = m.group(2).strip().split(":")[0]
            if headline and len(headline) > 10:
                phrases.append(f"{headline} — {idea}" if idea else headline)

        # Action recommendations → good for CTA/outro
        if (stripped.startswith("Action:") or "↳ Action:" in stripped) and scene_type in ("cta", "outro"):
            action = re.sub(r"^(Action:|↳ Action:)\s*", "", stripped).strip()
            if action and len(action) > 10:
                phrases.append(action)

        # Content recommendations → verse/body
        if stripped.startswith("•") and scene_type in ("body", "verse", "bridge"):
            rec = stripped.lstrip("•").strip()
            if rec and len(rec) > 15:
                phrases.append(rec)

        # Trending topics → all scenes
        tags = re.findall(r"#(\w+)", stripped)
        topic_words = [t for t in tags if len(t) > 4][:2]
        if topic_words and scene_type in ("hook", "chorus", "outro"):
            topic_str = " ".join(f"#{t}" for t in topic_words)
            phrases.append(f"{idea} is trending — {topic_str}" if idea else topic_str)

    return phrases[:4]


# ── public API ────────────────────────────────────────────────────────────────

def sample_all_scenes(
    scene_sequence: List[str],
    idea: str,
    genre: str,
    tone: str,
    platform: str,
    artist_name: str,
    awareness: str = "",
) -> Tuple[Dict[int, str], str]:
    """
    Return ({scene_idx: text}, "datasets") for every scene in the sequence.

    Resolution order for each scene:
      1. pdim live corpus (phrases accumulated from real generations)
      2. Awareness-derived phrases (live industry signals)
      3. Built-in seed phrase banks (always available, zero cost)

    All tiers are personalised with artist context before being returned.
    """
    used = _UsedSet()
    ctx = dict(idea=idea or "the drop", genre=genre or "music",
               tone=tone or "energetic", platform=platform or "youtube",
               artist=artist_name or "the artist")

    cta_pool = (CTA_PHRASES.get(platform.lower(), [])
                + CTA_PHRASES["_default"])

    results: Dict[int, str] = {}

    for idx, stype in enumerate(scene_sequence):
        # ── Tier 1: pdim live corpus ──────────────────────────────────────────
        pdim_pool = _pdim_fetch_phrases(stype)
        if pdim_pool:
            raw = used.pick(pdim_pool)
            text = _personalise(raw, **ctx)
            text = _trim(text)
            results[idx] = text
            # Don't re-push pdim phrases — they're already stored
            continue

        # ── Tier 2: awareness-derived phrases ─────────────────────────────────
        awareness_pool = _awareness_phrases_for_scene(
            awareness, stype, idea or "the drop", genre or "music",
            artist_name or "the artist"
        )
        if awareness_pool:
            raw = used.pick(awareness_pool)
            text = _personalise(raw, **ctx)
            text = _trim(text)
            results[idx] = text
            # Persist to pdim so future requests benefit
            _pdim_push_phrase(stype, raw)
            continue

        # ── Tier 3: built-in seed phrase banks ────────────────────────────────
        if stype == "cta":
            pool = cta_pool
        else:
            pool = _POOL_MAP.get(stype, VERSE_PHRASES)

        raw = used.pick(pool)
        text = _personalise(raw, **ctx)
        text = _trim(text, max_words=10)
        results[idx] = text
        # Persist seed phrases to pdim to bootstrap the corpus
        _pdim_push_phrase(stype, raw)

    return results, "datasets"
