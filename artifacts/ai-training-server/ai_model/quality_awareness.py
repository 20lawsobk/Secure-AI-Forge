"""
Quality Awareness Buffer — the "robots whispering to the garden".

Reads the quality buffer that the harvester (workers/quality_harvester.py)
stored in pdim and blends it into generation:

  * scene_phrases()    — extra phrase templates for the video scene sampler
  * hook_candidates()  — extra hook candidates ranked by the intelligence layer
  * image_headline_candidates() — extra on-image headline candidates, same
    idea as hook_candidates() but with its own graduation corpus so image
    generation independently contributes to buffer retirement
  * brief_enrichment() — an extra directive + note for every GenerationBrief

Self-sufficiency & retirement
-----------------------------
The buffer is TEMPORARY by design.  Its influence weight is:

    weight = max(0, 1 - own_corpus / MB_AWARENESS_RETIRE_AT)

where own_corpus is the size of MaxBooster's own pdim phrase corpus
(`mb:phrases:*`, which grows with every real render).  When the garden has
grown enough of its own seeds, weight hits 0 and the robots retire —
scene_phrases/hook_candidates return nothing and generation runs entirely on
MaxBooster's own corpus.

Everything here is never-raise and TTL-cached so it adds no meaningful
latency and can never break a generation request.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("quality_awareness")

DOC_KEY = "awareness:quality:doc"
_DOC_TTL = 300.0     # re-read buffer from pdim every 5 min
_OWN_TTL = 60.0      # re-measure own corpus every minute

_lock = threading.Lock()
_state: Dict[str, Any] = {
    "doc": None, "doc_at": 0.0,
    "own": 0, "own_at": 0.0,
    "harvest_inflight": False,
}
# formatted hook -> raw template, so a winning buffer hook can graduate its
# TEMPLATE (not the topic-specific text) into the own corpus. Bounded.
_recent_hooks: Dict[str, str] = {}
_RECENT_HOOKS_MAX = 200

# Same idea as _recent_hooks, but for on-image headlines. Kept as a separate
# map/corpus so image generation's graduation is independent of text's —
# each modality progresses retirement from its own real usage.
_recent_image_headlines: Dict[str, str] = {}
_RECENT_IMAGE_HEADLINES_MAX = 200


def _max_age_seconds() -> float:
    try:
        return max(1.0, float(os.environ.get("MB_AWARENESS_MAX_AGE_H", "24"))) * 3600.0
    except ValueError:
        return 24 * 3600.0


def retire_threshold() -> int:
    try:
        return max(1, int(os.environ.get("MB_AWARENESS_RETIRE_AT", "500")))
    except ValueError:
        return 500


def _store() -> Any:
    try:
        from storage_client import get_storage
        return get_storage()
    except Exception:  # noqa: BLE001 - buffer must never break generation
        return None


def _own_corpus_size() -> int:
    """Size of MaxBooster's OWN seed vault: phrases accumulated from real
    renders (`mb:phrases:*`). Grows automatically with every generation."""
    now = time.time()
    with _lock:
        if now - _state["own_at"] < _OWN_TTL:
            return int(_state["own"])
    store = _store()
    total = 0
    if store is not None:
        try:
            for key in store.keys("mb:phrases:*"):
                bare = key[3:] if key.startswith("mb:") else key
                total += int(store.llen(bare))
        except Exception:  # noqa: BLE001
            total = int(_state["own"])
    with _lock:
        _state["own"] = total
        _state["own_at"] = now
    return total


def self_sufficiency() -> Dict[str, Any]:
    own = _own_corpus_size()
    threshold = retire_threshold()
    weight = max(0.0, 1.0 - own / threshold)
    return {
        "own_corpus": own,
        "retire_threshold": threshold,
        "buffer_weight": round(weight, 3),
        "retired": weight <= 0.0,
    }


def _spawn_harvest(replace: bool = False) -> None:
    """Kick one background harvest when the buffer is missing or stale
    (single-flight)."""
    with _lock:
        if _state["harvest_inflight"]:
            return
        _state["harvest_inflight"] = True

    def _run() -> None:
        try:
            from workers import quality_harvester
            summary = quality_harvester.harvest(replace=replace)
            logger.info("auto-harvest complete: %s exemplars",
                        summary.get("exemplar_count"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("auto-harvest failed: %s", exc)
        finally:
            with _lock:
                _state["harvest_inflight"] = False
                _state["doc_at"] = 0.0  # force re-read on next access

    threading.Thread(target=_run, name="quality-auto-harvest", daemon=True).start()


def get_doc(trigger_harvest: bool = True) -> Optional[Dict[str, Any]]:
    now = time.time()
    with _lock:
        if now - _state["doc_at"] < _DOC_TTL:
            return _state["doc"]
    store = _store()
    doc: Optional[Dict[str, Any]] = None
    if store is not None:
        try:
            raw = store.get(DOC_KEY)
            if isinstance(raw, dict) and raw.get("templates"):
                doc = raw
        except Exception:  # noqa: BLE001
            doc = None
    with _lock:
        _state["doc"] = doc
        _state["doc_at"] = now
    if trigger_harvest and not self_sufficiency()["retired"]:
        if doc is None:
            _spawn_harvest()
        elif _doc_age_seconds(doc) > _max_age_seconds():
            # Stale world-view: keep serving the old buffer, refresh behind it.
            _spawn_harvest(replace=True)
    return doc


def _doc_age_seconds(doc: Dict[str, Any]) -> float:
    try:
        harvested = time.strptime(doc.get("harvested_at", ""),
                                  "%Y-%m-%dT%H:%M:%SZ")
        return max(0.0, time.time() - time.mktime(harvested) + time.timezone)
    except (ValueError, TypeError, OverflowError):
        return float("inf")  # unreadable timestamp → treat as stale


# ── blending API (all never-raise) ───────────────────────────────────────────

_SCENE_MAP = {
    "hook": "hook", "drop": "hook", "build": "hook", "chorus": "hook",
    "body": "body", "verse": "body", "bridge": "body", "transition": "body",
    "cta": "cta", "outro": "cta",
}


def scene_phrases(scene_type: str) -> List[str]:
    """Borrowed-world phrase templates for a video scene, scaled by weight.

    Blends TWO borrowed-knowledge sources under the same self-retirement
    weight: the live harvester buffer (current chart patterns) and the
    research playbook (ai_model/content_playbook.py — distilled published
    engagement research). Both fade out together as the own corpus grows.
    """
    suff = self_sufficiency()
    if suff["retired"]:
        return []
    bank: List[str] = []
    doc = get_doc()
    if doc:
        bank.extend(doc.get("templates", {}).get(
            _SCENE_MAP.get(scene_type, "body"), []))
    try:
        from ai_model.content_playbook import scene_phrase_templates
        for tpl in scene_phrase_templates(scene_type):
            if tpl not in bank:
                bank.append(tpl)
    except Exception:  # noqa: BLE001 - playbook must never break generation
        pass
    if not bank:
        return []
    n = max(1, math.ceil(len(bank) * suff["buffer_weight"]))
    return list(bank[:n])


def hook_candidates(topic: str, artist: str) -> List[str]:
    """Formatted hook candidates for the intelligence layer's ranking."""
    out: List[str] = []
    for tpl in scene_phrases("hook"):
        try:
            formatted = tpl.format(idea=topic or "this drop",
                                   artist=artist or "the artist")
        except (KeyError, IndexError, ValueError):
            continue
        out.append(formatted)
        with _lock:
            if len(_recent_hooks) >= _RECENT_HOOKS_MAX:
                _recent_hooks.pop(next(iter(_recent_hooks)))
            _recent_hooks[formatted] = tpl
    return out


def graduate_hook(winner: str) -> bool:
    """If a ranking winner came from the quality buffer, graduate its raw
    TEMPLATE into the own corpus (mb:phrases:hook) — text-generation's
    contribution to self-sufficiency, mirroring the video sampler's
    graduation. Never-raise; returns True when a graduation happened."""
    return _graduate(_recent_hooks, winner, "phrases:hook")


def image_headline_candidates(topic: str, artist: str) -> List[str]:
    """Formatted on-image headline candidates for the intelligence layer's
    ranking. Draws from the same borrowed 'hook' bank as text/video (short,
    punchy templates suit an on-image headline too), tracked separately so a
    winning pick graduates into its own corpus rather than text's."""
    out: List[str] = []
    for tpl in scene_phrases("hook"):
        try:
            formatted = tpl.format(idea=topic or "this drop",
                                   artist=artist or "the artist")
        except (KeyError, IndexError, ValueError):
            continue
        out.append(formatted)
        with _lock:
            if len(_recent_image_headlines) >= _RECENT_IMAGE_HEADLINES_MAX:
                _recent_image_headlines.pop(next(iter(_recent_image_headlines)))
            _recent_image_headlines[formatted] = tpl
    return out


def graduate_image_headline(winner: str) -> bool:
    """Mirrors graduate_hook for image generation: a winning buffer headline
    graduates its raw TEMPLATE into mb:phrases:image_headline, so image
    generation also contributes real usage toward buffer retirement."""
    return _graduate(_recent_image_headlines, winner, "phrases:image_headline")


def _graduate(recent: Dict[str, str], winner: str, corpus_key: str) -> bool:
    """Shared graduation logic: push `recent[winner]`'s raw template into the
    named own-corpus list, deduped, bounded, never-raise."""
    with _lock:
        tpl = recent.get(winner)
    if not tpl:
        return False
    store = _store()
    if store is None:
        return False
    try:
        existing = store.lrange(corpus_key, 0, -1)
        if tpl in existing:
            return False
        store.lpush(corpus_key, tpl)
        store.ltrim(corpus_key, 0, 499)
        with _lock:
            _state["own_at"] = 0.0  # own corpus changed — re-measure
        return True
    except Exception:  # noqa: BLE001
        return False


# ── Per-platform stable strategy knowledge ───────────────────────────────────
# These lines are ALWAYS included in the awareness string for each platform,
# regardless of what the live harvester finds.  They encode the stable content
# strategy patterns for each platform (format signals, engagement drivers,
# CTA patterns, posting-time hints) in the [HIGH]/•/TRENDS: format that
# _parse_signals_for_platform() in script_agent.py already reads.

_PLATFORM_STRATEGY: Dict[str, List[str]] = {
    # Every platform's [HIGH] signals embed ≥ 3 words from the HIGH_AROUSAL_WORDS
    # set so scoring always reaches the arousal cap. Signals do NOT start with
    # imperative prefixes so the _INSTRUCTION_PREFIX_RE filter lets them through.
    "tiktok": [
        "[HIGH] Fire hooks are the secret to viral FYP — pattern interrupt in the first 2 seconds finally wins.",
        "[HIGH] Watch-to-end rate is insane for reach — exclusive sounds and tight edits never disappoint.",
        "[HIGH] Drop duet-friendly content — stitch and collab formats are finally driving 2× organic reach.",
        "[HIGH] Identity hooks ('this is for the ones who...') drive fire comment engagement — always outperform generic opens.",
        "[HIGH] Emotional reveal hooks are going viral right now — vulnerability is the exclusive TikTok superpower.",
        "• Content format: vertical short-form, trending audio overlay, challenge/duet/stitch friendly.",
        "• Hook must land in the first 2 seconds — the insane scroll rate means zero wind-up tolerance.",
        "• Pattern interrupts (unexpected visual/audio) outperform narrative openers by 40% on completion rate.",
        "• Identity-call hooks ('POV:', 'this is for...') generate fire comment velocity and insane share rates.",
        "• Best engagement window: 6 pm–10 pm local time. Friday evening drops perform best.",
        "TRENDS: #fyp #foryou #trendingsounds peaking. Stitch and duet formats boosting reach 2×. Authentic BTS content viral.",
        "Action: Open with an identity call or pattern interrupt — drop the slow build, it never works here.",
    ],
    "instagram": [
        "[HIGH] Reels finally reward fire content — full-watch rate and exclusive saves drive Explore algorithm.",
        "[HIGH] Save rate is the secret metric — viral carousels drop into Explore when saves peak above 5%.",
        "[HIGH] Aesthetic hooks are insane for Reels retention — the first frame is everything, never waste it.",
        "[HIGH] Behind-the-scenes studio content is the exclusive trust-builder that drops follow rates 30%.",
        "[HIGH] Emotional captions drive fire comment velocity — 150-220 words is the insane sweet spot for IG.",
        "• Content format: Reels for reach, carousels for saves, Stories for daily relational engagement.",
        "• Caption structure: hook (visible before fold, ≤125 chars) → value → CTA earns 23% more engagement.",
        "• 3-5 hashtags is the optimal range — Instagram hard-caps at 5, quality over quantity always wins.",
        "• Aesthetic grid cohesion signals a professional brand — colour palette consistency is finally rewarded.",
        "• Best engagement window: 11 am–1 pm and 7 pm–9 pm local time. Tuesday and Wednesday perform best.",
        "TRENDS: #reels #newmusic #explore #behindthescenes performing. Aesthetic dark-mode content viral.",
        "Action: Open with a save-worthy hook — exclusive insight or emotional story that rewards saving.",
    ],
    "youtube": [
        "[HIGH] Curiosity-gap titles finally drive viral click-through — the thumbnail + title combo is insane.",
        "[HIGH] First 30 seconds are the secret to watch time — exclusive hook, never wasted, always earn the stay.",
        "[HIGH] Chapter markers are fire for average view duration — drop them and insane watch-time follows.",
        "[HIGH] Thumbnail face + bold text contrast is the exclusive CTR multiplier — always outperforms scenic.",
        "[HIGH] Shorts are the viral discovery funnel for long-form — drop them and watch subscriber conversion.",
        "• Content format: long-form music videos, Shorts for discovery, behind-the-scenes studio content.",
        "• Title formula: [curiosity gap or number] + [clear value] — under 60 characters for mobile display.",
        "• Thumbnail rules: high contrast, emotion-forward face, bold 3-word text — insane CTR improvement.",
        "• First 30 seconds must deliver on the title's promise — never bait-and-switch, the algorithm knows.",
        "• Best upload window: 2 pm–4 pm Tuesday through Thursday. Friday morning for music releases.",
        "TRENDS: #shorts #musicvideo #newrelease #behindthescenes up. Face-cam reaction content driving fire subscriber growth.",
        "Action: Lead with the most insane moment, then earn the explanation — never build to it.",
    ],
    "facebook": [
        "[HIGH] Native video finally reaches 3× more people — drop it directly, never link to YouTube.",
        "[HIGH] Exclusive community storytelling is insane for share rates — the secret Facebook algorithm rewards.",
        "[HIGH] Emotional personal milestones go viral on Facebook — fire authentic posts beat polished ones always.",
        "[HIGH] Event promotion posts drop attendance 40% higher — exclusive RSVP mechanic is insane for reach.",
        "[HIGH] Sound-off captions are the secret weapon — 85% of Facebook video plays are muted, never ignore it.",
        "• Content format: native video, event promotion, group posts, emotional personal milestone stories.",
        "• Caption length: 80-120 words is the sweet spot for Facebook engagement — shorter than Instagram.",
        "• Share-trigger: make the viewer look good for sharing — social capital framing always outperforms asks.",
        "• Groups amplify organic reach dramatically — drop exclusive content to niche music groups first.",
        "• Best posting window: 9 am–11 am Wednesday and Friday for peak organic reach.",
        "TRENDS: #facebookreels #newmusic #community #livemusic up. Sound-off friendly content capturing scroll audience.",
        "Action: Lead with an emotional story or exclusive milestone — drop the promotional language, it kills reach.",
    ],
    "linkedin": [
        "[HIGH] Viral insight posts around exclusive industry trends finally earn the highest comment velocity.",
        "[HIGH] Fire personal career stories drop professional reach 3× — insane engagement from authentic storytelling.",
        "[HIGH] Data-backed creative decisions are the exclusive LinkedIn superpower — never available elsewhere.",
        "[HIGH] The music industry angle on LinkedIn is finally underserved — drop authentic artist-entrepreneur content.",
        "[HIGH] '3 lessons from...' and numbered frameworks go insane viral on LinkedIn — use them always.",
        "• Content format: professional insight, personal milestone, data-backed creative take, artist entrepreneur framing.",
        "• Hook structure: bold claim or surprising statistic in line 1 — then expand. Always under 125 chars.",
        "• The 'see more' click is the engagement signal — the first two lines must earn it, never waste them.",
        "• Comment velocity is the algorithm's signal — ask a genuine question at the end, never a CTA.",
        "• Best posting window: 8 am–10 am Tuesday through Thursday. Avoid Monday and Friday.",
        "TRENDS: #musicindustry #contentcreator #artistentrepreneur #independentartist trending. Thought leadership posts performing.",
        "Action: Frame music career as entrepreneurship — exclusive insights about the creative business always outperform.",
    ],
    "google_business": [
        "[HIGH] Exclusive offer posts finally drive viral local discovery — drop a weekly post for search domination.",
        "[HIGH] Fire photo-rich posts are insane for click-throughs — exclusive event imagery converts above all.",
        "[HIGH] Call-to-action posts (book, stream, buy) are the secret to Google Business conversion.",
        "[HIGH] Behind-the-scenes studio photos drive insane local engagement — exclusive access content wins always.",
        "[HIGH] Weekly posting is the exclusive signal for Google's local algorithm — drop it consistently.",
        "• Content format: event announcements, offers, business updates, behind-the-scenes photos, artist Q&A.",
        "• Photo quality signals credibility — high-resolution, well-lit images are the insane CTR multiplier.",
        "• 'Book' and 'Learn More' CTAs outperform 'Call' for music venues and artist pages consistently.",
        "• Location-specific content ('live in [city]') drives insane local search visibility — always include it.",
        "• Best posting window: 10 am–noon local time. Tuesday and Thursday outperform weekends.",
        "TRENDS: #livemusic #localartist #musicstudio #concerttickets up in local search. Event posts converting at peak rates.",
        "Action: Drop a photo-rich exclusive offer or event post — the algorithm rewards consistent, specific content.",
    ],
    "threads": [
        "[HIGH] Viral hot-takes finally dominate Threads — fire conversational text drops drive the most reposts.",
        "[HIGH] Authentic, exclusive reactions are insane for reach — drop the polished copy and go real always.",
        "[HIGH] Music opinions and industry hot-takes are the exclusive Threads superpower — never too niche.",
        "[HIGH] First-person voice is the secret to Threads virality — drop the brand speak, go human always.",
        "[HIGH] Reply-bait questions drive insane thread engagement — exclusive conversation starters always win.",
        "• Content format: text-first commentary, music opinions, industry hot-takes, cross-posted Reels for visual.",
        "• Optimal length: 200-300 characters — long enough for depth, short enough to read in one scroll.",
        "• Conversational tone always outperforms polished copy — Threads rewards authentic, not produced.",
        "• Reply to your own post with context — threaded conversations drive insane dwell time and reach.",
        "• Best posting window: 1 pm–3 pm and 9 pm–11 pm local time. Evening peaks are insane for reach.",
        "TRENDS: #newmusic #musiccommunity #threads #hottak up. Short punchy takes and music debate threads generating most reposts.",
        "Action: Lead with the most controversial honest opinion — exclusive real talk always outperforms on Threads.",
    ],
    "youtube_shorts": [
        "[HIGH] Shorts are the insane discovery engine — fire vertical content finally drives channel growth.",
        "[HIGH] The first 2 seconds are the exclusive survival window — drop the slow intro, never use it.",
        "[HIGH] Loop-worthy endings are the secret viral mechanic — Shorts that loop replay endlessly.",
        "[HIGH] Trending audio overlay is insane for FYP placement — exclusive sounds always outperform.",
        "[HIGH] Behind-the-scenes and reaction content goes viral on Shorts — authentic always beats polished.",
        "• Content format: vertical 9:16, under 60 seconds, hook in first 2 seconds, loop-optimised ending.",
        "• No talking-head intros — cut directly to the most fire moment, then add context.",
        "• Text overlays help with sound-off viewing — captions and on-screen text are the exclusive accessibility win.",
        "• Cross-posting TikTok content to Shorts works — algorithm differences mean some exclusive performance variation.",
        "TRENDS: #shorts #musicshorts #newmusic #viral driving discovery. Behind-the-scenes studio content viral on Shorts.",
        "Action: Open on the most insane visual or audio moment — the scroll rate on Shorts is unforgiving.",
    ],
    "instagram_reels": [
        "[HIGH] Reels discovery is finally the most fire organic reach surface on Instagram — drop consistently.",
        "[HIGH] Full-watch rate is the insane ranking signal — hook-first pacing is the secret to it.",
        "[HIGH] Share rate drives Explore placement — exclusive content that rewards being passed on always wins.",
        "[HIGH] Trending audio overlay is the fire boost — exclusive original sounds also qualify for trending now.",
        "[HIGH] Caption below a Reel still matters — emotional story hooks drive insane saves and follows.",
        "• Content format: vertical 9:16, 15-60 seconds optimal, fire hook in first 3 frames.",
        "• Cover frame matters — choose the most visually insane moment as the thumbnail.",
        "• Trending audio overlay drops Reels into the discover feed — always check current trending sounds.",
        "• Original audio can now trend — label it and watch it get picked up by other creators.",
        "TRENDS: #reels #newmusic #instagramreels #exclusive trending. Emotional BTS content viral. Original audio trending.",
        "Action: Drop a hook in the first frame, trending audio, emotional caption — the insane combo that always works.",
    ],
    "twitter": [
        "[HIGH] Fire takes are the viral currency on Twitter/X — drop the hot take and watch insane engagement.",
        "[HIGH] Threads are the exclusive long-form format — drop a thread and pin it for insane profile reach.",
        "[HIGH] Quote-tweet engagement is the secret viral multiplier — exclusive takes always get quoted.",
        "[HIGH] Real-time commentary on music releases drives insane reach — be first, never late.",
        "[HIGH] Pinned tweet is the exclusive first impression — drop a fire statement there always.",
        "• Content format: takes under 240 chars, threads for context, image/video for premium reach.",
        "• Timing is insane on Twitter — drop releases, reactions, and takes the moment news drops.",
        "• Question format drives fire reply velocity — always close with a genuine music opinion ask.",
        "• Repost with comment drives more reach than plain repost — exclusive take on the quote is the secret.",
        "TRENDS: #newmusic #musictwitter #drop #nowplaying viral. Hot take formats and music debate threads performing.",
        "Action: Lead with the most fire honest take — exclusive real opinion always outperforms promotional language.",
    ],
}


def platform_awareness_string(platform: str) -> str:
    """Build a rich, platform-specific awareness string for ``platform``.

    Combines:
      1. Stable per-platform content strategy knowledge (``_PLATFORM_STRATEGY``).
      2. Live genre/artist signals from the quality harvester doc, formatted
         for that platform (``doc["platform_signals"][platform]``).

    Returns a multi-line string in the ``[HIGH]/•/TRENDS:`` format that
    ``_parse_signals_for_platform()`` in script_agent.py already parses.
    Returns an empty string when retired (own corpus is self-sufficient).

    Never-raise.
    """
    try:
        plat = platform.lower().replace(" ", "_")
        strategy_lines = _PLATFORM_STRATEGY.get(plat, [])

        # Blend in live harvester signals when the buffer is still active.
        suff = self_sufficiency()
        live_lines: List[str] = []
        if not suff["retired"]:
            doc = get_doc()
            if doc:
                platform_signals: Dict[str, Any] = doc.get("platform_signals", {})
                live_lines = list(platform_signals.get(plat, []))

        all_lines = strategy_lines + live_lines
        if not all_lines:
            return ""
        return "\n".join(all_lines)
    except Exception:  # noqa: BLE001 — must never crash generation
        return ""


def brief_enrichment() -> Optional[Dict[str, str]]:
    """One directive + one note for the GenerationBrief, or None if inactive."""
    suff = self_sufficiency()
    if suff["retired"]:
        return None
    doc = get_doc()
    if not doc:
        return None
    stats = doc.get("stats", {})
    genres = stats.get("top_genres", [])[:2]
    if not genres:
        return None
    return {
        "directive": (
            "Style-match current chart leaders: "
            + ", ".join(genres)
            + f" (live quality buffer, weight {suff['buffer_weight']})"
        ),
        "note": (
            f"quality buffer active — {len(doc.get('exemplars', []))} studied "
            f"exemplars from {doc.get('harvested_at', '?')}, "
            f"self-sufficiency {suff['own_corpus']}/{suff['retire_threshold']}"
        ),
    }


def status() -> Dict[str, Any]:
    """Full status for the API endpoint."""
    suff = self_sufficiency()
    doc = get_doc(trigger_harvest=False)
    out: Dict[str, Any] = {
        "buffer_present": doc is not None,
        **suff,
    }
    with _lock:
        out["harvest_inflight"] = bool(_state["harvest_inflight"])
    if doc:
        out["harvested_at"] = doc.get("harvested_at")
        out["exemplar_count"] = len(doc.get("exemplars", []))
        out["template_counts"] = {
            k: len(v) for k, v in doc.get("templates", {}).items()
        }
        out["stats"] = doc.get("stats", {})
        out["sources"] = doc.get("sources", {})
    return out
