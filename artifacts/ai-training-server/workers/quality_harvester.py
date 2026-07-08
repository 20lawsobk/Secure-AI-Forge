"""
Quality Harvester — the "robots that explore the world".

Fetches the *best-performing real content* from live public sources (music
charts, top music channels, high-engagement stories), studies it — never
copies it — and distils the patterns into a quality buffer stored in pdim
(`mb:awareness:quality:doc`).

The buffer is a TEMPORARY dataset: generation endpoints blend it in only
while MaxBooster's own pdim corpus is still small.  As the garden grows its
own seeds, the buffer's weight decays to zero and the robots retire
(see ai_model/quality_awareness.py).

No fake data: if every source fails, harvest() raises explicitly — an empty
world-scan is never silently stored as knowledge.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from collections import Counter
from typing import Any, Dict, List

logger = logging.getLogger("quality_harvester")

DOC_KEY = "awareness:quality:doc"
_UA = "MaxBooster/1.0 (content quality research; contact: admin)"
_TIMEOUT = 12

APPLE_SONGS_URL = (
    "https://rss.applemarketingtools.com/api/v2/us/music/most-played/50/songs.json"
)
APPLE_ALBUMS_URL = (
    "https://rss.applemarketingtools.com/api/v2/us/music/most-played/25/albums.json"
)
YOUTUBE_FEEDS: List[Dict[str, str]] = [
    {"name": "NoCopyrightSounds",
     "url": "https://www.youtube.com/feeds/videos.xml?user=NoCopyrightSounds"},
    {"name": "TrapNation",
     "url": "https://www.youtube.com/feeds/videos.xml?user=alltrapnation"},
    {"name": "Proximity",
     "url": "https://www.youtube.com/feeds/videos.xml?user=ProximityM"},
]
HN_URL = (
    "https://hn.algolia.com/api/v1/search"
    "?query=music%20artist&tags=story&hitsPerPage=30"
)


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read()


# ── per-source harvesters (each never-raise; report ok/error) ────────────────

def _harvest_apple_charts() -> Dict[str, Any]:
    """Top-of-market songs + albums: real, ranked, current."""
    out: Dict[str, Any] = {"ok": False, "items": 0, "error": None,
                           "titles": [], "artists": [], "genres": []}
    try:
        for url in (APPLE_SONGS_URL, APPLE_ALBUMS_URL):
            data = json.loads(_fetch(url))
            results = data.get("feed", {}).get("results", [])
            for rank, r in enumerate(results):
                name = (r.get("name") or "").strip()
                artist = (r.get("artistName") or "").strip()
                if name:
                    # weight: chart position → 1.0 at #1, decaying linearly
                    weight = max(0.1, 1.0 - rank / max(1, len(results)))
                    out["titles"].append({"text": name, "weight": round(weight, 3),
                                          "source": "apple_charts"})
                if artist:
                    out["artists"].append(artist)
                for g in r.get("genres", []):
                    gname = (g.get("name") or "").strip()
                    if gname and gname.lower() != "music":
                        out["genres"].append(gname)
        out["items"] = len(out["titles"])
        out["ok"] = out["items"] > 0
    except Exception as exc:  # noqa: BLE001 - per-source isolation
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _harvest_youtube_music() -> Dict[str, Any]:
    """Titles from major music channels, weighted by REAL view counts taken
    from the feed's media:statistics (no engagement data → low honest weight,
    labelled 'curated', never inflated)."""
    out: Dict[str, Any] = {"ok": False, "items": 0, "error": None, "titles": []}
    errors: List[str] = []
    raw: List[Dict[str, Any]] = []
    for feed in YOUTUBE_FEEDS:
        try:
            xml = _fetch(feed["url"]).decode("utf-8", errors="replace")
            for entry in xml.split("<entry>")[1:16]:
                tm = re.search(r"<title>([^<]{4,120})</title>", entry)
                if not tm:
                    continue
                vm = re.search(r'views="(\d+)"', entry)
                raw.append({
                    "text": tm.group(1).strip(),
                    "views": int(vm.group(1)) if vm else None,
                    "channel": feed["name"],
                })
        except Exception as exc:  # noqa: BLE001 - per-source isolation
            errors.append(f"{feed['name']}: {type(exc).__name__}")
    max_views = max((r["views"] for r in raw if r["views"]), default=0)
    for r in raw:
        if not r["text"]:
            continue
        if r["views"] is not None and max_views > 0:
            weight = round(0.2 + 0.8 * (r["views"] / max_views), 3)
            source = f"youtube:{r['channel']}"
        else:
            weight = 0.3  # no measurable engagement — honest low weight
            source = f"youtube_curated:{r['channel']}"
        out["titles"].append({"text": r["text"], "weight": weight,
                              "source": source})
    out["items"] = len(out["titles"])
    out["ok"] = out["items"] > 0
    if errors and not out["ok"]:
        out["error"] = "; ".join(errors)
    return out


def _harvest_hn_engagement() -> Dict[str, Any]:
    """High-engagement music stories — headline patterns with real vote counts."""
    out: Dict[str, Any] = {"ok": False, "items": 0, "error": None, "titles": []}
    try:
        data = json.loads(_fetch(HN_URL))
        hits = data.get("hits", [])
        max_pts = max((h.get("points") or 1 for h in hits), default=1)
        for h in hits:
            title = (h.get("title") or "").strip()
            pts = h.get("points") or 0
            if title and pts > 5:
                out["titles"].append({
                    "text": title,
                    "weight": round(min(1.0, pts / max_pts), 3),
                    "source": "hn_engagement",
                })
        out["items"] = len(out["titles"])
        out["ok"] = out["items"] > 0
    except Exception as exc:  # noqa: BLE001 - per-source isolation
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# ── pattern study (in-house, deterministic) ──────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def _study(titles: List[Dict[str, Any]], artists: List[str],
           genres: List[str]) -> Dict[str, Any]:
    """Distil style statistics from harvested top content — the 'studying'."""
    word_counts = [len(t["text"].split()) for t in titles if t.get("text")]
    word_counts.sort()
    median_words = word_counts[len(word_counts) // 2] if word_counts else 0

    punchy = sum(1 for t in titles
                 if any(c in t["text"] for c in "!?—:")) / max(1, len(titles))

    leading: Counter = Counter()
    for t in titles:
        m = _WORD_RE.search(t["text"])
        if m:
            leading[m.group(0).lower()] += 1

    genre_counts = Counter(g for g in genres)
    artist_counts = Counter(a for a in artists)

    return {
        "median_title_words": median_words,
        "punchy_title_ratio": round(punchy, 3),
        "top_leading_words": [w for w, _ in leading.most_common(8)],
        "top_genres": [g for g, _ in genre_counts.most_common(6)],
        "top_artists": [a for a, _ in artist_counts.most_common(8)],
    }


def _derive_templates(stats: Dict[str, Any]) -> Dict[str, List[str]]:
    """Turn studied patterns into {idea}/{artist} phrase templates.

    Only produced when real data exists (empty stats → empty templates).
    Templates are style guidance conditioned on live chart data — the robots
    whisper "here's how the winners look", they never copy a plant.

    Template FORMS follow the researched hook archetypes catalogued in
    ai_model/content_playbook.py (identity call, curiosity gap, reveal,
    emotion-first story, low-friction CTA) — the playbook supplies the shape,
    the live harvest supplies today's genres/artists to fill it.
    """
    genres = stats.get("top_genres", [])
    artists = stats.get("top_artists", [])
    if not genres and not artists:
        return {"hook": [], "body": [], "cta": []}

    g1 = genres[0] if genres else "the chart"
    g2 = genres[1] if len(genres) > 1 else g1
    a1 = artists[0] if artists else "the biggest artists"

    hooks = [
        # curiosity gap / bold claim forms
        f"{g1} is running the charts — {{idea}} is next",
        f"The {g1} wave is peaking. {{idea}} rides it",
        "Charts move fast — {idea} moves faster",
        # reveal form (live-chart flavored)
        f"While the world loops {a1}, {{artist}} drops {{idea}}",
        # identity-call form — the top-retention archetype in 2026 testing
        f"If {g1} is your whole personality, {{idea}} was made for you",
    ]
    if stats.get("median_title_words", 0) and stats["median_title_words"] <= 4:
        hooks.append("{idea}. That's the post")

    bodies = [
        f"Cut from the sound leading right now: {g1} and {g2}",
        f"{{idea}} — built with the same energy as this week's top {g1} records",
        "Studied what the top of the charts does — then made it ours with {idea}",
    ]
    ctas = [
        # low-friction save / share / stream forms per CTA research
        "Stream {idea} before the algorithm catches up",
        f"Add {{idea}} to the rotation next to {a1}",
        "Early on {idea} now — say you knew first",
        f"Tag the friend who won't stop playing {g1} — {{idea}} is for them",
    ]
    return {"hook": hooks, "body": bodies, "cta": ctas}


# ── main entry point ─────────────────────────────────────────────────────────

def harvest(replace: bool = True) -> Dict[str, Any]:
    """Scan the world, study the winners, store the quality buffer in pdim.

    Raises RuntimeError when EVERY source fails — no fake knowledge.
    """
    t0 = time.time()
    apple = _harvest_apple_charts()
    youtube = _harvest_youtube_music()
    hn = _harvest_hn_engagement()

    sources = {
        "apple_charts": {k: apple[k] for k in ("ok", "items", "error")},
        "youtube_music": {k: youtube[k] for k in ("ok", "items", "error")},
        "hn_engagement": {k: hn[k] for k in ("ok", "items", "error")},
    }
    if not any(s["ok"] for s in sources.values()):
        raise RuntimeError(
            f"quality harvest failed — every source unreachable: {sources}"
        )

    all_titles: List[Dict[str, Any]] = (
        apple["titles"] + youtube["titles"] + hn["titles"]
    )
    all_titles.sort(key=lambda t: t["weight"], reverse=True)
    exemplars = all_titles[:40]

    stats = _study(all_titles, apple.get("artists", []), apple.get("genres", []))
    templates = _derive_templates(stats)

    doc: Dict[str, Any] = {
        "harvested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "harvest_seconds": round(time.time() - t0, 2),
        "sources": sources,
        "exemplars": exemplars,
        "stats": stats,
        "templates": templates,
    }

    from storage_client import get_storage
    store = get_storage()
    if replace or not store.exists(DOC_KEY):
        store.set(DOC_KEY, doc)

    logger.info(
        "quality harvest: %d exemplars, %d hook templates, sources ok: %s",
        len(exemplars), len(templates["hook"]),
        [k for k, v in sources.items() if v["ok"]],
    )
    return {
        "exemplar_count": len(exemplars),
        "template_counts": {k: len(v) for k, v in templates.items()},
        "stats": stats,
        "sources": sources,
        "harvest_seconds": doc["harvest_seconds"],
    }
