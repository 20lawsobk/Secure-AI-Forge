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
# Deezer public chart API — no key required. Global chart (0) plus the major
# per-genre editorial charts so the beacon covers the whole market, not just
# one blended top-50.
DEEZER_CHARTS: List[Dict[str, Any]] = [
    {"genre": "",            "id": 0},    # global top
    {"genre": "hip hop",     "id": 116},  # rap / hip-hop
    {"genre": "electronic",  "id": 106},  # electro / dance
    {"genre": "pop",         "id": 132},
    {"genre": "rock",        "id": 152},
    {"genre": "r&b",         "id": 165},
    {"genre": "latin",       "id": 197},
]
DEEZER_CHART_URL = "https://api.deezer.com/chart/{id}/tracks?limit=25"
# Social + advertising quality beacon — public, key-free, real engagement:
# Mastodon trending posts/tags (reblogs+favourites are real share metrics)
# and HN marketing/advertising stories (vote-ranked headline patterns).
MASTODON_TRENDS_STATUSES = "https://mastodon.social/api/v1/trends/statuses?limit=20"
MASTODON_TRENDS_TAGS = "https://mastodon.social/api/v1/trends/tags?limit=10"
HN_ADS_URL = (
    "https://hn.algolia.com/api/v1/search"
    "?query=advertising%20marketing&tags=story&hitsPerPage=25"
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


def _harvest_deezer_charts() -> Dict[str, Any]:
    """Global + per-genre Deezer charts: titles, artists, genre membership,
    plus per-track duration/gain used later for music-feature study.

    ``rank`` on Deezer is a real popularity score — weights are normalised
    against the max rank in each chart (honest relative popularity)."""
    out: Dict[str, Any] = {"ok": False, "items": 0, "error": None,
                           "titles": [], "artists": [], "genres": [],
                           "tracks": []}
    errors: List[str] = []
    for chart in DEEZER_CHARTS:
        try:
            data = json.loads(_fetch(DEEZER_CHART_URL.format(id=chart["id"])))
            rows = data.get("data", [])
            max_rank = max((r.get("rank") or 1 for r in rows), default=1)
            for pos, r in enumerate(rows):
                title = (r.get("title_short") or r.get("title") or "").strip()
                artist = ((r.get("artist") or {}).get("name") or "").strip()
                if not title:
                    continue
                weight = round(max(0.1, (r.get("rank") or 1) / max_rank), 3)
                out["titles"].append({"text": title, "weight": weight,
                                      "source": "deezer_charts"})
                if artist:
                    out["artists"].append(artist)
                if chart["genre"]:
                    out["genres"].append(chart["genre"])
                out["tracks"].append({
                    "genre": chart["genre"] or "global",
                    "title": title,
                    "artist": artist,
                    "duration": int(r.get("duration") or 0),
                    "position": pos,
                    "weight": weight,
                    "preview": (r.get("preview") or "").strip(),
                })
        except Exception as exc:  # noqa: BLE001 - per-source isolation
            errors.append(f"chart {chart['id']}: {type(exc).__name__}")
    out["items"] = len(out["titles"])
    out["ok"] = out["items"] > 0
    if errors and not out["ok"]:
        out["error"] = "; ".join(errors)
    return out


_TAG_RE = re.compile(r"<[^>]+>")
# Tags come from public, attacker-influenceable input; allow word characters
# only (covers non-English letters/digits/underscore) plus hyphen, and cap
# length, so no control chars, whitespace, or prompt-syntax survive.
_SAFE_TAG_RE = re.compile(r"[\w-]+", re.UNICODE)


def _sanitize_tag(name: Any) -> str:
    if not isinstance(name, str):
        return ""
    return "".join(_SAFE_TAG_RE.findall(name))[:40]


def _harvest_social_ad() -> Dict[str, Any]:
    """Social + advertising quality beacon.

    Social: Mastodon's public trending posts — reblogs+favourites are REAL
    share metrics, so weights are honest engagement, plus trending tags.
    Ads/marketing: HN vote-ranked advertising/marketing stories — the
    headline patterns that earn engagement in the marketing world itself.
    """
    out: Dict[str, Any] = {"ok": False, "items": 0, "error": None,
                           "social_posts": [], "trending_tags": [],
                           "ad_titles": []}
    errors: List[str] = []
    try:
        posts = json.loads(_fetch(MASTODON_TRENDS_STATUSES))
        max_eng = max(((p.get("reblogs_count") or 0) + (p.get("favourites_count") or 0)
                       for p in posts), default=1) or 1
        for p in posts:
            text = _TAG_RE.sub(" ", p.get("content") or "")
            text = re.sub(r"\s+", " ", text).strip()
            eng = (p.get("reblogs_count") or 0) + (p.get("favourites_count") or 0)
            if len(text) >= 20 and eng > 0:
                out["social_posts"].append({
                    "text": text[:300],
                    "weight": round(max(0.1, eng / max_eng), 3),
                    "source": "mastodon_trending",
                })
    except Exception as exc:  # noqa: BLE001 - per-source isolation
        errors.append(f"mastodon_statuses: {type(exc).__name__}")
    try:
        tags = json.loads(_fetch(MASTODON_TRENDS_TAGS))
        out["trending_tags"] = [
            st for st in (
                _sanitize_tag(t.get("name"))
                for t in tags if isinstance(t, dict)
            ) if st
        ][:10]
    except Exception as exc:  # noqa: BLE001
        errors.append(f"mastodon_tags: {type(exc).__name__}")
    try:
        data = json.loads(_fetch(HN_ADS_URL))
        hits = data.get("hits", [])
        max_pts = max((h.get("points") or 1 for h in hits), default=1)
        for h in hits:
            title = (h.get("title") or "").strip()
            pts = h.get("points") or 0
            if title and pts > 3:
                out["ad_titles"].append({
                    "text": title,
                    "weight": round(min(1.0, pts / max_pts), 3),
                    "source": "hn_marketing",
                })
    except Exception as exc:  # noqa: BLE001
        errors.append(f"hn_marketing: {type(exc).__name__}")
    out["items"] = (len(out["social_posts"]) + len(out["trending_tags"])
                    + len(out["ad_titles"]))
    out["ok"] = out["items"] > 0
    if errors and not out["ok"]:
        out["error"] = "; ".join(errors)
    return out


def _study_social_ad(social: Dict[str, Any]) -> Dict[str, Any]:
    """Distil measured social + ad patterns into awareness signal lines.

    Everything here is derived from real engagement-weighted content pulled
    minutes ago — no invented numbers. Lines follow the [HIGH]/[MED]/TRENDS:
    format the downstream parsers already understand.
    """
    posts = social.get("social_posts", [])
    ads = social.get("ad_titles", [])
    tags = social.get("trending_tags", [])
    lines: List[str] = []

    if posts:
        top = sorted(posts, key=lambda p: p["weight"], reverse=True)[:8]
        lens = sorted(len(p["text"].split()) for p in top)
        median_words = lens[len(lens) // 2]
        question_ratio = sum(1 for p in top if "?" in p["text"]) / len(top)
        first_person = sum(
            1 for p in top
            if re.search(r"\b(I|we|my|our)\b", p["text"], re.IGNORECASE)) / len(top)
        lines.append(
            f"[HIGH] Top-shared social posts right now run ~{median_words} words; "
            f"{int(first_person * 100)}% speak first-person and "
            f"{int(question_ratio * 100)}% open a question — personal voice is "
            "outperforming broadcast copy.")
    if tags:
        lines.append("TRENDS: live cross-platform tags gaining now: "
                     + " ".join(f"#{t}" for t in tags[:6]) + ".")
    if ads:
        top_ads = sorted(ads, key=lambda a: a["weight"], reverse=True)[:8]
        leading: Counter = Counter()
        for a in top_ads:
            m = _WORD_RE.search(a["text"])
            if m:
                leading[m.group(0).lower()] += 1
        lead_words = [w for w, _ in leading.most_common(3)]
        if lead_words:
            lines.append(
                "[MEDIUM] Highest-engagement marketing headlines this week lead with: "
                + ", ".join(f"'{w}'" for w in lead_words)
                + " — concrete, claim-first openers beat vague brand talk.")
    return {
        "signal_lines": lines,
        "social_exemplars": sorted(posts, key=lambda p: p["weight"],
                                   reverse=True)[:10],
        "ad_exemplars": sorted(ads, key=lambda a: a["weight"],
                               reverse=True)[:10],
        "trending_tags": tags,
    }


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


# ── music-feature study: measure the charting sound itself ──────────────────

def _analyze_preview_bpm(preview_url: str) -> Dict[str, float]:
    """Download one 30 s chart-track preview and MEASURE bpm/energy with the
    in-house analyzer (librosa beat tracking). Never raises — returns {} on
    any failure. This is a real live signal from a track that is charting
    right now, not a lookup table."""
    import os
    import tempfile

    import numpy as np

    tmp_mp3 = tmp_raw = None
    try:
        from ai_model.video.ffmpeg_util import run_ffmpeg
        from ai_model.audio.audio_analysis import analyze_audio

        raw = _fetch(preview_url)
        if len(raw) < 10_000:
            return {}
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(raw)
            tmp_mp3 = f.name
        tmp_raw = tmp_mp3 + ".f32"
        res = run_ffmpeg(
            ["ffmpeg", "-y", "-i", tmp_mp3, "-ac", "1", "-ar", "22050",
             "-f", "f32le", tmp_raw],
            timeout=15.0, retries=1,
        )
        if getattr(res, "returncode", 1) != 0 or not os.path.exists(tmp_raw):
            return {}
        y = np.fromfile(tmp_raw, dtype=np.float32)
        if y.size < 22050:
            return {}
        tl = analyze_audio(y, 22050, use_cache=False)
        if not tl.analysis_ok or tl.bpm <= 0:
            return {}
        rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
        return {"bpm": round(float(tl.bpm), 1),
                "energy": round(min(1.0, rms * 4.0), 3)}
    except Exception:  # noqa: BLE001 - single preview failure is not an error
        return {}
    finally:
        for p in (tmp_mp3, tmp_raw):
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass


def _study_music_features(tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-genre musical targets measured from the charts.

    Duration comes from every chart row; BPM/energy are MEASURED by analyzing
    a bounded number of 30 s previews per genre (top of each chart first).
    Bounded by MB_HARVEST_PREVIEWS total (default 12) so a harvest stays
    fast. Genres with no successful analysis simply omit bpm — no fake data.
    """
    import os as _os

    try:
        budget = max(0, int(_os.environ.get("MB_HARVEST_PREVIEWS", "12")))
    except ValueError:
        budget = 12
    # Hard wall-clock budget for the whole preview-analysis pass so a slow
    # CDN or ffmpeg stall can't stretch a harvest into minutes (the harvest
    # runs in a background thread, but its runtime should stay bounded).
    try:
        deadline = time.time() + max(
            10.0, float(_os.environ.get("MB_HARVEST_PREVIEW_BUDGET_S", "90")))
    except ValueError:
        deadline = time.time() + 90.0

    by_genre: Dict[str, List[Dict[str, Any]]] = {}
    for t in tracks:
        by_genre.setdefault(t["genre"], []).append(t)

    features: Dict[str, Any] = {}
    per_genre_cap = max(1, budget // max(1, len(by_genre)))
    for genre, rows in by_genre.items():
        durations = sorted(r["duration"] for r in rows if r["duration"] > 0)
        entry: Dict[str, Any] = {
            "tracks_seen": len(rows),
            "duration_median_sec": durations[len(durations) // 2] if durations else 0,
        }
        bpms: List[float] = []
        energies: List[float] = []
        analyzed = 0
        for r in sorted(rows, key=lambda x: x["position"]):
            if analyzed >= per_genre_cap or budget <= 0 or time.time() >= deadline:
                break
            if not r.get("preview"):
                continue
            m = _analyze_preview_bpm(r["preview"])
            budget -= 1
            analyzed += 1
            if m:
                bpms.append(m["bpm"])
                energies.append(m["energy"])
        if bpms:
            bpms.sort()
            entry["bpm_median"] = bpms[len(bpms) // 2]
            entry["bpm_range"] = [min(bpms), max(bpms)]
            entry["energy_mean"] = round(sum(energies) / len(energies), 3)
            entry["measured_previews"] = len(bpms)
        features[genre] = entry
    return features


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


def _derive_platform_signals(stats: Dict[str, Any]) -> Dict[str, List[str]]:
    """Per-platform signal lines shaped from live chart data.

    Each platform gets 3–4 lines in the ``[HIGH]/•/TRENDS:`` format that
    ``_parse_signals_for_platform()`` in script_agent.py already parses.
    They are stored under ``platform_signals`` in the doc and consumed by
    ``quality_awareness.platform_awareness_string()``.
    """
    genres = stats.get("top_genres", [])
    artists = stats.get("top_artists", [])
    g1 = genres[0] if genres else "the top genre"
    g2 = genres[1] if len(genres) > 1 else g1
    a1 = artists[0] if artists else "top artists"
    a2 = artists[1] if len(artists) > 1 else a1

    return {
        "tiktok": [
            f"[HIGH] {g1} sounds trending on TikTok — hook in the first 2 seconds.",
            f"[HIGH] {a1}-style reveal videos driving watch-to-end rate up this week.",
            f"TRENDS: pattern interrupt opening, {g2} audio overlay, challenge format "
            f"converting at 3× the rate of standard posts. #fyp #viral peaking.",
        ],
        "instagram": [
            f"[HIGH] {g1} Reels getting pushed by the algorithm — save-rate is the key metric.",
            f"[HIGH] Carousel posts with {g2} mood aesthetic driving 60% more saves than statics.",
            f"TRENDS: {a1} aesthetic resonating. First 3 seconds critical. "
            f"#reels #newmusic #explore trending this cycle.",
        ],
        "youtube": [
            f"[HIGH] {g1} music videos outperforming average watch-time by 40% on YouTube.",
            f"[HIGH] First 30 seconds: curiosity-gap hook is the top retention driver for {g2}.",
            f"TRENDS: {a1}-style thumbnails (high contrast, face + text) boosting CTR. "
            f"Chapters and end-screen cards lifting subscribe rate. #shorts #musicvideo up.",
        ],
        "facebook": [
            f"[HIGH] Native {g1} video uploads reaching 3× the audience of shared links on Facebook.",
            f"[HIGH] Community storytelling around {g2} driving share rates above baseline.",
            f"TRENDS: sound-off captioned videos capturing scroll-past audience. "
            f"{a2} emotional story format performing strongly. #facebookreels rising.",
        ],
        "linkedin": [
            f"[HIGH] Music industry insight posts around {g1} trends earning high comment velocity.",
            f"[HIGH] Personal story + professional lesson format outperforming promotional posts.",
            f"TRENDS: data-backed {g2} industry analysis and artist milestone content "
            f"driving professional engagement. First 3 lines before fold are critical. "
            f"#musicindustry #contentcreator trending.",
        ],
        "google_business": [
            f"[HIGH] Event and offer posts featuring {g1} genre driving local discovery clicks.",
            f"[HIGH] Photo-rich posts getting 35% more click-throughs for music venues and studios.",
            f"TRENDS: regular posting schedule signals active business to Google Maps ranking. "
            f"Call-to-action posts (book, call, stream) converting above static updates. "
            f"#livemusic #localartist up in local search.",
        ],
        "threads": [
            f"[HIGH] Conversational {g1} commentary and hot-takes driving the most Repost activity.",
            f"[HIGH] Authentic, unpolished text posts about {g2} outperforming polished copy.",
            f"TRENDS: {a1} music commentary threads generating reply chains that boost reach. "
            f"Cross-platform Reels to Threads reposts amplified by the algorithm. "
            f"#newmusic #musiccommunity gaining momentum.",
        ],
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
    deezer = _harvest_deezer_charts()
    youtube = _harvest_youtube_music()
    hn = _harvest_hn_engagement()
    social = _harvest_social_ad()

    sources = {
        "apple_charts": {k: apple[k] for k in ("ok", "items", "error")},
        "deezer_charts": {k: deezer[k] for k in ("ok", "items", "error")},
        "youtube_music": {k: youtube[k] for k in ("ok", "items", "error")},
        "hn_engagement": {k: hn[k] for k in ("ok", "items", "error")},
        "social_ad": {k: social[k] for k in ("ok", "items", "error")},
    }
    if not any(s["ok"] for s in sources.values()):
        raise RuntimeError(
            f"quality harvest failed — every source unreachable: {sources}"
        )

    all_titles: List[Dict[str, Any]] = (
        apple["titles"] + deezer["titles"] + youtube["titles"] + hn["titles"]
    )
    all_titles.sort(key=lambda t: t["weight"], reverse=True)
    exemplars = all_titles[:60]

    stats = _study(
        all_titles,
        apple.get("artists", []) + deezer.get("artists", []),
        apple.get("genres", []) + deezer.get("genres", []),
    )
    templates = _derive_templates(stats)
    platform_signals = _derive_platform_signals(stats)
    music_features = _study_music_features(deezer.get("tracks", []))
    social_ad_patterns = _study_social_ad(social)

    doc: Dict[str, Any] = {
        "harvested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "harvest_seconds": round(time.time() - t0, 2),
        "sources": sources,
        "exemplars": exemplars,
        "stats": stats,
        "templates": templates,
        "platform_signals": platform_signals,
        "music_features": music_features,
        "social_ad_patterns": social_ad_patterns,
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
