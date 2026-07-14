"""Release campaign planner — turn one song into a full rollout.

Research finding (Water & Music survey of 150+ music-industry pros; RouteNote /
NotNoise / Chartlex rollout guides, 2026): the single thing independent artists
and producers most want from a system like this is *promotional tooling that
turns a release into a whole content campaign* so they can spend their time
making music instead of feeding the content machine. The recurring, concrete
asks are:

  * "Turn one release into 20+ pieces of content" (the *content goldmine*).
  * A structured multi-week **rollout timeline** working backward from the
    release date: announce → tease → pre-save → release day → sustain.
  * **Platform-specific** posts (a Reel/TikTok is not an Instagram caption).
  * Ready-to-post **captions with hooks + CTAs**, plus **art direction** for the
    visuals (the visual side is where non-visual creators get stuck).

``build_campaign`` composes the pieces this server already has — the request
brief (``request_intelligence.build_brief``), the caption composer
(``compose_caption``), and the unified Visual-DNA technique engine
(``extract_technique``) — into that rollout. It is **never-raise**: any slot
that fails to generate degrades to a minimal templated post so the returned
campaign is always complete.

Design note (why hooks are templated here, not left to the composer): each post
gets a clean, phase-appropriate *hook* from a content-type template, while the
*body* and *CTA* come from the intelligence composer (driven by a per-post
theme + goal). This keeps hooks distinct and professional across a 15-post
campaign that all shares one title, and — critically — keeps the human-readable
creative *brief* out of the generation inputs so it can never leak verbatim into
copy (a known failure mode when instruction text is templated raw).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Rollout blueprint ────────────────────────────────────────────────────────
# Base plan is a 6-week (42-day) single rollout; ``day`` is days relative to
# release day (negative = before, 0 = release day, positive = after). Pre-release
# offsets scale with the requested campaign length; release day and the sustain
# tail stay fixed. ``theme`` seeds the composed body/CTA; ``brief`` is the
# human-readable creative direction returned to the caller (never fed to the
# generator).
_BLUEPRINT: List[Dict[str, Any]] = [
    # ── ANNOUNCE ────────────────────────────────────────────────────────────
    {"phase": "announce", "type": "announcement", "format": "post",  "day": -35, "goal": "awareness",  "platform": "instagram", "theme": "new single announcement", "brief": "Announce the new single is coming and reveal the title."},
    {"phase": "announce", "type": "teaser",       "format": "reel",  "day": -33, "goal": "awareness",  "platform": "tiktok",    "theme": "sneak peek",             "brief": "Hint that something new is on the way — leave them curious."},
    {"phase": "announce", "type": "story",        "format": "post",  "day": -28, "goal": "connection", "platform": "instagram", "theme": "the story behind the song", "brief": "Share the story and inspiration behind the song."},
    # ── TEASE ─────────────────────────────────────────────────────────────────
    {"phase": "tease",    "type": "audio_teaser", "format": "reel",  "day": -21, "goal": "anticipation", "platform": "tiktok",   "theme": "first snippet of the hook", "brief": "Post the catchiest 8 seconds of the hook."},
    {"phase": "tease",    "type": "lyric_tease",  "format": "reel",  "day": -17, "goal": "anticipation", "platform": "tiktok",   "theme": "standout lyric",            "brief": "Spotlight a single lyric that makes people feel something."},
    {"phase": "tease",    "type": "bts",          "format": "reel",  "day": -14, "goal": "connection",   "platform": "tiktok",   "theme": "behind the scenes",         "brief": "Behind the scenes making the track in the studio."},
    # ── PRE-SAVE ───────────────────────────────────────────────────────────────
    {"phase": "presave",  "type": "presave_push", "format": "post",  "day": -12, "goal": "conversion",   "platform": "instagram", "theme": "pre-save now",              "brief": "The pre-save link is live — tell them why to save it now."},
    {"phase": "presave",  "type": "countdown",    "format": "story", "day": -7,  "goal": "anticipation", "platform": "instagram", "theme": "countdown",                 "brief": "One week to go — build countdown energy."},
    {"phase": "presave",  "type": "countdown",    "format": "reel",  "day": -3,  "goal": "conversion",   "platform": "tiktok",    "theme": "final countdown",           "brief": "Last push to pre-save before the drop."},
    # ── RELEASE ────────────────────────────────────────────────────────────────
    {"phase": "release",  "type": "out_now",      "format": "post",  "day": 0,   "goal": "conversion",   "platform": "instagram", "theme": "out now",                   "brief": "It's out now — celebrate and send them to stream."},
    {"phase": "release",  "type": "out_now",      "format": "reel",  "day": 0,   "goal": "conversion",   "platform": "tiktok",    "theme": "out now stream now",        "brief": "The wait is over — the song is live, use the full hook."},
    {"phase": "release",  "type": "thank_you",    "format": "post",  "day": 2,   "goal": "connection",   "platform": "instagram", "theme": "thank you fans",            "brief": "Thank fans for the first listens and shares."},
    # ── SUSTAIN ────────────────────────────────────────────────────────────────
    {"phase": "sustain",  "type": "reaction",     "format": "reel",  "day": 5,   "goal": "engagement",   "platform": "tiktok",    "theme": "fan reactions",             "brief": "React to fan reactions / duets and encourage more."},
    {"phase": "sustain",  "type": "acoustic",     "format": "reel",  "day": 10,  "goal": "engagement",   "platform": "tiktok",    "theme": "acoustic version",          "brief": "Share a stripped-back or alternate version."},
    {"phase": "sustain",  "type": "milestone",    "format": "post",  "day": 18,  "goal": "social_proof", "platform": "instagram", "theme": "streams milestone",         "brief": "Celebrate a streams / playlist milestone with fans."},
]

# Formats that represent moving video (get a short video teaser when asset
# generation is opted into); everything else is a still image only.
_TEASER_FORMATS = {"reel", "video"}

_PHASE_META = {
    "announce": ("Announcement", "Reveal the release exists and start the story."),
    "tease":    ("Teaser",       "Give tasters of the sound to build anticipation."),
    "presave":  ("Pre-save",     "Convert anticipation into pre-saves and countdown hype."),
    "release":  ("Release",      "The drop — drive streams and thank early fans."),
    "sustain":  ("Sustain",      "Keep the release alive long after day one."),
}

# Clean, phase-appropriate hooks per content type. ``{title}``/``{artist}`` are
# filled; ``countdown`` is handled separately (needs the day count). A value may
# be a list of variants — when the same content type is used by more than one
# slot (e.g. two ``out_now`` posts on release day), each occurrence takes the
# next variant so no two posts in the campaign share an identical hook.
_HOOKS: Dict[str, Any] = {
    "announcement": "New single incoming — '{title}' 🎶",
    "teaser":       "Something new from {artist} is on the way…",
    "story":        "The story behind '{title}'",
    "audio_teaser": "First listen — the hook from '{title}' 🔊",
    "lyric_tease":  "This line from '{title}' hits different…",
    "bts":          "Behind the scenes making '{title}'",
    "presave_push": "'{title}' — pre-save it now (link in bio)",
    "out_now": [
        "'{title}' is OUT NOW 🚀",
        "The wait is over — '{title}' is live everywhere 🎧",
    ],
    "thank_you":    "Thank you for the love on '{title}' 🙏",
    "reaction":     "Your reactions to '{title}' are unreal",
    "acoustic":     "'{title}' — stripped back 🎸",
    "milestone":    "'{title}' just hit a new milestone 🎉",
}


def _countdown_phrase(day_offset: int) -> str:
    n = abs(int(day_offset))
    if n >= 7 and n % 7 == 0:
        w = n // 7
        return "1 week to go" if w == 1 else f"{w} weeks to go"
    if n == 1:
        return "1 day to go"
    return f"{n} days to go"


def _hook_for(ctype: str, title: str, artist: str, day_offset: int,
              variant_index: int = 0) -> str:
    if ctype == "countdown":
        return f"{_countdown_phrase(day_offset)} until '{title}' 🔥"
    tpl = _HOOKS.get(ctype, "'{title}' — {artist}")
    if isinstance(tpl, (list, tuple)):
        tpl = tpl[variant_index % len(tpl)] if tpl else "'{title}' — {artist}"
    try:
        return tpl.format(title=title, artist=artist)
    except Exception:
        return f"{artist} — {title}"


def _body_similarity(a: str, b: str) -> float:
    """Token-set Jaccard similarity (0-1) between two body strings.

    Used to keep each post's chosen body distinct from the bodies already placed
    in the same phase, so a campaign that shares one title never collapses posts
    into near-duplicates (the exact failure this planner guards against)."""
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _compose_body_cta(
    *, title: str, artist: str, platform: str, goal: str, theme: str,
    tone: Optional[str], genre: Optional[str], brand_voice: Optional[str],
    target_audience: Optional[str], awareness: str = "",
) -> Dict[str, Any]:
    """Ranked body+CTA variants from the shared intelligence composer.

    Returns ``{"variants": [{"body", "cta"}, ...], "disclosure_brief": brief}``
    with the composer's variants in rank order (best first). Only the *body* and
    *CTA* come from the composer; the hook is templated per content type.
    ``theme`` (a natural phrase, never an instruction) is fed as the post's theme
    to diversify bodies across the campaign. The caller picks the variant that is
    most distinct from other posts already placed in the same phase. Never raises.
    ``awareness`` carries live chart + platform signals into the brief so every
    campaign post is conditioned on the same real-world context as stand-alone
    social posts.
    """
    try:
        from ai_model import request_intelligence as ri
        brief = ri.build_brief(
            modality="content", platform=platform, topic=title, goal=goal,
            tone=tone, genre=genre, artist=artist,
            extra=" ".join(filter(None, [brand_voice, target_audience])),
            themes=[theme] if theme else None, track=title,
            awareness=awareness,
        )
        composed = ri.compose_caption(
            title, artist, brief, genre=genre, brand_voice=brand_voice, variants=4,
        )
        fallback_cta = composed.get("cta", "") or brief.suggested_cta
        opts: List[Dict[str, str]] = []
        for v in (composed.get("variants") or []):
            body = v.get("body", "")
            if body:
                opts.append({"body": body, "cta": v.get("cta", "") or fallback_cta})
        if not opts:
            opts = [{"body": composed.get("body", "") or theme.capitalize(),
                     "cta": fallback_cta}]
        return {"variants": opts, "disclosure_brief": brief}
    except Exception:
        return {"variants": [{"body": theme.capitalize() if theme else "New music.",
                              "cta": "Follow for more."}],
                "disclosure_brief": None}


def _pick_distinct_body(
    variants: List[Dict[str, str]], placed_bodies: List[str],
) -> Dict[str, str]:
    """Choose the body/CTA variant least similar to bodies already placed in the
    phase; ties break toward the composer's own rank (earlier = higher quality)."""
    if not variants:
        return {"body": "", "cta": ""}
    best = variants[0]
    best_key: Optional[Tuple[float, int]] = None
    for rank, opt in enumerate(variants):
        worst_sim = max((_body_similarity(opt["body"], pb) for pb in placed_bodies),
                        default=0.0)
        key = (worst_sim, rank)
        if best_key is None or key < best_key:
            best_key = key
            best = opt
    return best


def _apply_disclosure(caption: str, brief: Any) -> str:
    if brief is None:
        return caption
    try:
        from ai_model import request_intelligence as ri
        return ri.apply_disclosure(caption, brief)
    except Exception:
        return caption


def _art_direction(*, idea: str, genre: Optional[str], tone: Optional[str],
                   mood: Optional[str], bpm: Optional[float], key: Optional[str],
                   seed: int) -> Optional[Dict[str, Any]]:
    """Visual DNA for the whole campaign's imagery/video, from the technique engine."""
    try:
        from ai_model.generation.technique import extract_technique
        tech = extract_technique(
            idea=idea, genre=genre, tone=tone, mood=mood, bpm=bpm, key=key, seed=seed,
        )
        return {
            "color_scheme": tech.color_scheme(),
            "mood": tech.mood(),
            "energy": round(tech.energy, 3),
            "dna": tech.dna_dict(),
            "source": tech.source,
            "note": "Use this palette/energy across cover art, teasers and video "
                    "so the whole rollout looks like one release.",
        }
    except Exception:
        return None


def build_campaign(
    *,
    artist: str,
    title: str,
    genre: Optional[str] = None,
    tone: Optional[str] = None,
    brand_voice: Optional[str] = None,
    target_audience: Optional[str] = None,
    platforms: Optional[List[str]] = None,
    weeks: int = 6,
    mood: Optional[str] = None,
    bpm: Optional[float] = None,
    key: Optional[str] = None,
    release_date: Optional[str] = None,
    hashtag_fn: Optional[Callable[[str, Optional[str], str], List[str]]] = None,
    normalize_platform_fn: Optional[Callable[[str], str]] = None,
    image_fn: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
    teaser_fn: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
    seed: int = 0,
    awareness: str = "",
) -> Dict[str, Any]:
    """Build a full release rollout campaign for one song. Never raises.

    Returns a structured, multi-week, multi-platform content calendar: phased
    posts each with ready-to-post copy (hook/body/CTA/caption + hashtags), plus
    shared art direction (Visual DNA) for the imagery and video.

    Optional visual asset generation (opt-in via ``image_fn`` / ``teaser_fn``):
    when an ``image_fn`` is supplied, every post is paired with an on-brand image
    conditioned on the campaign's shared ``art_direction`` (so the whole rollout
    looks like one release); when a ``teaser_fn`` is supplied, reel/video slots
    additionally get a short video teaser. Both callables are invoked never-raise
    — a failed asset simply leaves that post text-only, so the calendar is always
    complete. When neither is supplied the plan stays text-only and fast.
    """
    artist = (artist or "the artist").strip() or "the artist"
    title = (title or "the new single").strip() or "the new single"
    try:
        weeks = max(2, min(12, int(weeks)))
    except (TypeError, ValueError):
        weeks = 6

    allowed = [p for p in (platforms or []) if p] or ["instagram", "tiktok"]
    if normalize_platform_fn:
        try:
            allowed = list(dict.fromkeys(normalize_platform_fn(p) for p in allowed)) or allowed
        except Exception:
            pass

    # Resolve release date if given, so posts carry absolute dates.
    rel = None
    if release_date:
        try:
            from datetime import date
            rel = date.fromisoformat(str(release_date)[:10])
        except Exception:
            rel = None

    scale = weeks / 6.0

    def _post_date(day_offset: int) -> Optional[str]:
        if rel is None:
            return None
        try:
            from datetime import timedelta
            return (rel + timedelta(days=day_offset)).isoformat()
        except Exception:
            return None

    # Shared Visual DNA for the whole rollout — computed once up front so the
    # optional per-post asset generators can condition on it (every image/teaser
    # shares one palette/energy) and so it can be returned to the caller as-is.
    art = _art_direction(
        idea=title, genre=genre, tone=tone, mood=mood, bpm=bpm, key=key, seed=seed,
    )

    phases: Dict[str, Dict[str, Any]] = {}
    total = 0
    by_platform: Dict[str, int] = {}
    images_generated = 0
    teasers_queued = 0
    # Anti-repetition state, so a rollout that shares one title never degrades
    # into near-duplicate copy: ``type_seen`` picks a fresh hook variant each
    # time a content type recurs; ``placed_bodies`` tracks the bodies already
    # used in each phase so every new post picks the most *distinct* body variant.
    type_seen: Dict[str, int] = {}
    placed_bodies: Dict[str, List[str]] = {}

    for slot in _BLUEPRINT:
        ctype = slot["type"]
        phase_name = slot["phase"]
        hook_variant = type_seen.get(ctype, 0)
        type_seen[ctype] = hook_variant + 1
        phase_bodies = placed_bodies.setdefault(phase_name, [])
        try:
            pref = slot["platform"]
            platform = pref if pref in allowed else allowed[0]
            # Pre-release offsets stretch/compress with campaign length; keep the
            # release day and sustain tail fixed.
            base_day = int(slot["day"])
            day_offset = int(round(base_day * scale)) if base_day < 0 else base_day

            hook = _hook_for(ctype, title, artist, day_offset,
                             variant_index=hook_variant)
            bc = _compose_body_cta(
                title=title, artist=artist, platform=platform, goal=slot["goal"],
                theme=slot["theme"], tone=tone, genre=genre, brand_voice=brand_voice,
                target_audience=target_audience, awareness=awareness,
            )
            chosen = _pick_distinct_body(bc["variants"], phase_bodies)
            body, cta = chosen["body"], chosen["cta"]
            phase_bodies.append(body)
            caption = f"{hook}\n\n{body}\n\n{cta}".strip()
            caption = _apply_disclosure(caption, bc.get("disclosure_brief"))

            tags: List[str] = []
            if hashtag_fn:
                try:
                    tags = hashtag_fn(title, genre, platform)[:8]
                except Exception:
                    tags = []

            post = {
                "day_offset": day_offset,
                "date": _post_date(day_offset),
                "platform": platform,
                "content_type": slot["type"],
                "format": slot["format"],
                "goal": slot["goal"],
                "brief": slot["brief"],
                "hook": hook,
                "body": body,
                "cta": cta,
                "caption": caption,
                "hashtags": tags,
                "char_count": len(caption),
            }
        except Exception:
            # Absolute last resort: never drop a slot from the calendar.
            post = {
                "day_offset": int(slot.get("day", 0)),
                "date": None,
                "platform": allowed[0],
                "content_type": slot.get("type", "post"),
                "format": slot.get("format", "post"),
                "goal": slot.get("goal", "engagement"),
                "brief": slot.get("brief", ""),
                "hook": f"{artist} — {title}",
                "body": slot.get("theme", "New music.").capitalize(),
                "cta": "Follow for more.",
                "caption": f"{artist} — {title}",
                "hashtags": [],
                "char_count": len(f"{artist} — {title}"),
            }

        # ── Optional generated assets (opt-in), conditioned on the shared art
        # direction so every asset looks like one release. Never-raise: a failed
        # asset just leaves the post text-only and never drops it.
        if image_fn is not None:
            try:
                img = image_fn(
                    topic=title, headline=post["hook"], platform=post["platform"],
                    fmt=post["format"], purpose=post["goal"], art_direction=art,
                )
                if img:
                    post["image"] = img
                    images_generated += 1
            except Exception:
                pass
        if teaser_fn is not None and post["format"] in _TEASER_FORMATS:
            try:
                tsr = teaser_fn(
                    topic=title, hook=post["hook"], body=post["body"],
                    cta=post["cta"], platform=post["platform"],
                    purpose=post["goal"], art_direction=art,
                )
                if tsr:
                    post["teaser"] = tsr
                    teasers_queued += 1
            except Exception:
                pass

        ph = slot["phase"]
        if ph not in phases:
            label, desc = _PHASE_META.get(ph, (ph.title(), ""))
            phases[ph] = {"phase": ph, "label": label, "description": desc, "posts": []}
        phases[ph]["posts"].append(post)
        total += 1
        by_platform[post["platform"]] = by_platform.get(post["platform"], 0) + 1

    ordered = [phases[k] for k in ("announce", "tease", "presave", "release", "sustain") if k in phases]

    return {
        "release": {
            "artist": artist,
            "title": title,
            "genre": genre,
            "weeks": weeks,
            "platforms": allowed,
            "release_date": rel.isoformat() if rel else None,
        },
        "art_direction": art,
        "phases": ordered,
        "summary": {
            "total_posts": total,
            "by_platform": by_platform,
            "by_phase": {p["phase"]: len(p["posts"]) for p in ordered},
            "images_generated": images_generated,
            "teasers_queued": teasers_queued,
        },
        "notes": [
            "Rollout works backward from release day; negative day_offset = days "
            "before release, 0 = release day, positive = after.",
            "Captions are ready to post; pair each with visuals following "
            "art_direction so the whole campaign looks cohesive.",
        ],
    }
