"""Platform-specific URL extractors for the Universal URL Parser.

Each extractor receives the raw HTML body (already fetched) and the parsed URL
components, and returns a partially-filled :class:`ParsedUrl`.  The core
orchestrator merges results from the best-matching extractor with generic
og-tag / slug fallbacks.

All extractors are pure functions — never-raise, never make network calls
(fetching is handled by :mod:`core`).
"""
from __future__ import annotations

import html as _html
import json
import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .models import ParsedUrl

# ── Shared HTML helpers ───────────────────────────────────────────────────────

_TAG_RE       = re.compile(r"<[^>]+>")
_WS           = re.compile(r"\s+")
_SCRIPT_STYLE = re.compile(
    r"<(script|style|noscript|nav|footer|header|aside)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_P_TAG  = re.compile(r"<p[^>]*>(.*?)</p>",  re.IGNORECASE | re.DOTALL)
_H1_TAG = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
_JSON_LD = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)

# Trailing platform suffixes to strip from page titles
_TITLE_SUFFIX = re.compile(
    r"\s*[\|–—\-]\s*(?:Spotify|YouTube|YouTube Music|SoundCloud|Apple Music|"
    r"Tidal|Deezer|Bandcamp|Audiomack|DistroKid|TuneCore|Genius|Pitchfork|"
    r"Instagram|TikTok|Twitter|X\.com|Facebook|LinkedIn|Pinterest|Reddit|"
    r"Twitch|Snapchat|BeReal|Substack|SoundOn|Triller|Lemon8|Threads|"
    r"Rolling Stone|XXL|HotNewHipHop|Complex|Hypebeast|Linktree)\s*$",
    re.IGNORECASE,
)

# Music genre keywords → canonical label
_GENRE_KEYWORDS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bhip.?hop\b",          re.I), "hip-hop"),
    (re.compile(r"\btrap\b",              re.I), "trap"),
    (re.compile(r"\br&b\b|rnb\b",         re.I), "r&b"),
    (re.compile(r"\bpop\b",               re.I), "pop"),
    (re.compile(r"\belectronic|edm\b",    re.I), "electronic"),
    (re.compile(r"\bhouse\b",             re.I), "house"),
    (re.compile(r"\btechno\b",            re.I), "techno"),
    (re.compile(r"\bjazz\b",              re.I), "jazz"),
    (re.compile(r"\bsoul\b",              re.I), "soul"),
    (re.compile(r"\brock\b",              re.I), "rock"),
    (re.compile(r"\bindies?\b",           re.I), "indie"),
    (re.compile(r"\bafrobeat[s]?\b",      re.I), "afrobeats"),
    (re.compile(r"\blatin\b",             re.I), "latin"),
    (re.compile(r"\breggaeton\b",         re.I), "reggaeton"),
    (re.compile(r"\bcountry\b",           re.I), "country"),
    (re.compile(r"\bclassical\b",         re.I), "classical"),
    (re.compile(r"\bambie?nt\b",          re.I), "ambient"),
    (re.compile(r"\bdrill\b",             re.I), "drill"),
    (re.compile(r"\bfunk\b",              re.I), "funk"),
    (re.compile(r"\bmetal\b",             re.I), "metal"),
    (re.compile(r"\bpunk\b",              re.I), "punk"),
    (re.compile(r"\bblues\b",             re.I), "blues"),
    (re.compile(r"\bgospel\b",            re.I), "gospel"),
    (re.compile(r"\breggae\b",            re.I), "reggae"),
    (re.compile(r"\bdance\b",             re.I), "dance"),
    (re.compile(r"\blo.?fi\b",            re.I), "lo-fi"),
    (re.compile(r"\bphonk\b",             re.I), "phonk"),
]

# Mood keywords
_MOOD_KEYWORDS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\benergetic\b",         re.I), "energetic"),
    (re.compile(r"\bchill\b|mellow\b",    re.I), "chill"),
    (re.compile(r"\bdark\b|gritty\b",     re.I), "dark"),
    (re.compile(r"\buplifting\b|upbeat\b",re.I), "uplifting"),
    (re.compile(r"\bromantic\b",          re.I), "romantic"),
    (re.compile(r"\bmelanchol\b|sad\b",   re.I), "melancholic"),
    (re.compile(r"\baggressive\b",        re.I), "aggressive"),
    (re.compile(r"\bhype\b",              re.I), "hype"),
    (re.compile(r"\bsmoky\b|smooth\b",    re.I), "smooth"),
    (re.compile(r"\bnostalgic\b",         re.I), "nostalgic"),
    (re.compile(r"\binspiration\b|inspirational\b", re.I), "inspirational"),
    (re.compile(r"\bplayful\b",           re.I), "playful"),
    (re.compile(r"\bdreamy\b",            re.I), "dreamy"),
    (re.compile(r"\bintense\b",           re.I), "intense"),
]

# Musical key patterns
_KEY_RE = re.compile(
    r"\b([A-G][b#]?)\s*(major|minor|maj|min|m)\b",
    re.IGNORECASE,
)

# BPM patterns
_BPM_RE = re.compile(
    r"\b(\d{2,3})\s*(?:bpm|beats?\s+per\s+min)",
    re.IGNORECASE,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(html_frag: str) -> str:
    return _WS.sub(" ", _TAG_RE.sub(" ", _html.unescape(html_frag))).strip()


def _og(html: str, prop: str) -> str:
    for pat in (
        rf'<meta[^>]+property=["\']og:{prop}["\'][^>]+content=["\']([^"\'<>]+)["\']',
        rf'<meta[^>]+content=["\']([^"\'<>]+)["\'][^>]+property=["\']og:{prop}["\']',
        rf'<meta[^>]+name=["\']{prop}["\'][^>]+content=["\']([^"\'<>]+)["\']',
        rf'<meta[^>]+content=["\']([^"\'<>]+)["\'][^>]+name=["\']{prop}["\']',
    ):
        m = re.search(pat, html, re.IGNORECASE)
        if m:
            return _html.unescape(m.group(1)).strip()
    return ""


def _strip_suffix(title: str) -> str:
    return _TITLE_SUFFIX.sub("", title).strip()


def _extract_paragraphs(html: str, max_chars: int = 400) -> str:
    parts: list[str] = []
    total = 0
    for m in _P_TAG.finditer(html):
        text = _clean(m.group(1))
        if len(text) < 20:
            continue
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return " ".join(parts)[:max_chars]


def _detect_genre(text: str) -> str:
    for pat, label in _GENRE_KEYWORDS:
        if pat.search(text):
            return label
    return ""


def _detect_mood(text: str) -> str:
    for pat, label in _MOOD_KEYWORDS:
        if pat.search(text):
            return label
    return ""


def _detect_bpm(text: str) -> Optional[int]:
    m = _BPM_RE.search(text)
    if m:
        v = int(m.group(1))
        if 40 <= v <= 300:
            return v
    return None


def _detect_key(text: str) -> str:
    m = _KEY_RE.search(text)
    if m:
        note    = m.group(1)
        quality = m.group(2).lower()
        quality = "major" if quality in ("major", "maj") else "minor"
        return f"{note} {quality}"
    return ""


def _extract_year(text: str) -> Optional[int]:
    m = re.search(r"\b(19[6-9]\d|20[0-2]\d)\b", text)
    return int(m.group(1)) if m else None


def _load_json_ld(html: str) -> list[dict]:
    """Extract all JSON-LD blocks from page HTML (never-raise)."""
    results: list[dict] = []
    for m in _JSON_LD.finditer(html):
        try:
            data = json.loads(m.group(1).strip())
            if isinstance(data, list):
                results.extend(data)
            elif isinstance(data, dict):
                results.append(data)
        except Exception:
            pass
    return results


def _jld_str(obj: dict, *keys: str) -> str:
    """Safely pull a nested str value from a JSON-LD dict."""
    for k in keys:
        val = obj.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            obj = val
    return ""


# ── Generic extractor (any web page) ─────────────────────────────────────────

def extract_generic(html: str, parsed_url) -> ParsedUrl:
    """Baseline extractor — works on any HTML page."""
    result = ParsedUrl(
        raw_url=parsed_url.geturl(),
        platform="web",
        platform_label="Web",
        content_type="page",
    )
    clean = _SCRIPT_STYLE.sub(" ", html)

    # Title: og:title → <title> → first H1
    og_title = _og(clean, "title")
    if og_title:
        result.title = _strip_suffix(og_title)
    else:
        t = re.search(r"<title[^>]*>([^<]{3,})</title>", clean, re.IGNORECASE)
        if t:
            result.title = _strip_suffix(_html.unescape(t.group(1)).strip())
        else:
            h1 = _H1_TAG.search(clean)
            if h1:
                result.title = _clean(h1.group(1))

    # Description: og:description → meta:description → first ¶s
    og_desc = _og(clean, "description")
    if og_desc:
        result.description = og_desc
    else:
        result.description = _extract_paragraphs(clean, 300)

    combined = result.title + " " + result.description
    result.genre = _detect_genre(combined)
    result.mood  = _detect_mood(combined)
    result.bpm   = _detect_bpm(combined)
    result.key   = _detect_key(combined)

    return result


# ── Spotify extractor ─────────────────────────────────────────────────────────

# Path patterns: /track/{id}, /album/{id}, /playlist/{id}, /artist/{id}
_SPO_PATH = re.compile(
    r"^/(track|album|playlist|artist|user|episode|show)/([A-Za-z0-9]+)",
    re.IGNORECASE,
)

def extract_spotify(html: str, parsed_url) -> ParsedUrl:
    result = ParsedUrl(
        raw_url=parsed_url.geturl(),
        platform="spotify",
        platform_label="Spotify",
        intent="drive_streams",
        goal="streams",
    )
    path = parsed_url.path.rstrip("/")
    m = _SPO_PATH.match(path)
    if m:
        result.content_type = m.group(1)  # track|album|playlist|artist|…

    clean = _SCRIPT_STYLE.sub(" ", html)

    # JSON-LD is the richest source on Spotify
    for ld in _load_json_ld(clean):
        t = ld.get("@type", "")
        if isinstance(t, str):
            t = t.lower()
        if "musicrecording" in t or "song" in t:
            result.title       = _jld_str(ld, "name") or result.title
            by = ld.get("byArtist", {})
            if isinstance(by, dict):
                result.artist  = _jld_str(by, "name") or result.artist
            in_album = ld.get("inAlbum", {})
            if isinstance(in_album, dict):
                result.album   = _jld_str(in_album, "name") or result.album
        elif "musicalbum" in t:
            result.title       = _jld_str(ld, "name") or result.title
            by = ld.get("byArtist", {})
            if isinstance(by, dict):
                result.artist  = _jld_str(by, "name") or result.artist
        elif "musicgroup" in t or "person" in t:
            result.artist      = _jld_str(ld, "name") or result.artist
        elif "playlist" in t:
            result.content_type = "playlist"
            result.title        = _jld_str(ld, "name") or result.title

    # og:title fallback: "Track Name • Artist Name" or "Album • Artist"
    if not result.title:
        og = _strip_suffix(_og(clean, "title"))
        if "•" in og:
            parts = [p.strip() for p in og.split("•", 1)]
            result.title  = parts[0]
            result.artist = parts[1]
        elif " - " in og:
            parts = [p.strip() for p in og.split(" - ", 1)]
            result.title  = parts[0]
            result.artist = parts[1]
        else:
            result.title = og

    result.description = _og(clean, "description") or _extract_paragraphs(clean, 250)

    combined = " ".join(filter(None, [result.title, result.artist, result.album, result.description]))
    result.genre = _detect_genre(combined)
    result.mood  = _detect_mood(combined)
    result.bpm   = _detect_bpm(combined)
    result.key   = _detect_key(combined)
    result.release_year = _extract_year(combined)
    return result


# ── YouTube extractor ─────────────────────────────────────────────────────────

_YT_VIDEO_ID = re.compile(r"[?&]v=([A-Za-z0-9_-]{11})")
_YT_CHANNEL  = re.compile(r"^/(@[\w.-]+|channel/[A-Za-z0-9_-]+|c/[\w.-]+)")
_YT_PLAYLIST = re.compile(r"[?&]list=([A-Za-z0-9_-]+)")


def extract_youtube(html: str, parsed_url) -> ParsedUrl:
    path = parsed_url.path
    qs   = parse_qs(parsed_url.query)

    result = ParsedUrl(
        raw_url=parsed_url.geturl(),
        platform="youtube",
        platform_label="YouTube",
        intent="drive_streams",
        goal="streams",
    )

    # Detect content type from URL
    if "/shorts/" in path:
        result.content_type = "short"
        result.intent = "grow_followers"
        result.goal   = "fanbase"
    elif _YT_VIDEO_ID.search(parsed_url.query or ""):
        result.content_type = "video"
    elif "list=" in (parsed_url.query or ""):
        result.content_type = "playlist"
    elif _YT_CHANNEL.match(path):
        result.content_type = "profile"
        result.intent = "grow_followers"
        result.goal   = "fanbase"

    clean = _SCRIPT_STYLE.sub(" ", html)

    # og:title is reliable on YouTube
    og_title = _strip_suffix(_og(clean, "title"))
    if " - " in og_title:
        parts = [p.strip() for p in og_title.rsplit(" - ", 1)]
        result.title  = parts[0]
        result.artist = parts[1]
    else:
        result.title = og_title

    result.description = _og(clean, "description") or _extract_paragraphs(clean, 300)

    # JSON-LD
    for ld in _load_json_ld(clean):
        t = str(ld.get("@type", "")).lower()
        if "videoobject" in t or "movie" in t:
            result.title        = _jld_str(ld, "name") or result.title
            result.description  = _jld_str(ld, "description") or result.description
            result.release_year = _extract_year(_jld_str(ld, "uploadDate")) or result.release_year
            channel = ld.get("author") or ld.get("creator") or {}
            if isinstance(channel, dict):
                result.artist = _jld_str(channel, "name") or result.artist

    combined = " ".join(filter(None, [result.title, result.artist, result.description]))
    result.genre = _detect_genre(combined)
    result.mood  = _detect_mood(combined)
    result.bpm   = _detect_bpm(combined)
    result.key   = _detect_key(combined)
    return result


# ── TikTok extractor ──────────────────────────────────────────────────────────

_TT_VIDEO = re.compile(r"/@([\w.]+)/video/(\d+)")
_TT_USER  = re.compile(r"/@([\w.]+)")

def extract_tiktok(html: str, parsed_url) -> ParsedUrl:
    path = parsed_url.path
    result = ParsedUrl(
        raw_url=parsed_url.geturl(),
        platform="tiktok",
        platform_label="TikTok",
        intent="grow_followers",
        goal="fanbase",
    )

    vm = _TT_VIDEO.search(path)
    if vm:
        result.artist       = vm.group(1).replace("_", " ").replace(".", " ")
        result.content_type = "video"
    else:
        um = _TT_USER.search(path)
        if um:
            result.artist       = um.group(1).replace("_", " ").replace(".", " ")
            result.content_type = "profile"

    clean = _SCRIPT_STYLE.sub(" ", html)
    og_title = _strip_suffix(_og(clean, "title"))
    if og_title:
        result.title = og_title
    result.description = _og(clean, "description") or ""

    combined = " ".join(filter(None, [result.title, result.artist, result.description]))
    result.genre = _detect_genre(combined)
    result.mood  = _detect_mood(combined)
    return result


# ── Instagram extractor ───────────────────────────────────────────────────────

_IG_POST  = re.compile(r"/p/([A-Za-z0-9_-]+)")
_IG_REEL  = re.compile(r"/reel/([A-Za-z0-9_-]+)")
_IG_USER  = re.compile(r"^/([A-Za-z0-9_\.]+)/?$")

def extract_instagram(html: str, parsed_url) -> ParsedUrl:
    path = parsed_url.path
    result = ParsedUrl(
        raw_url=parsed_url.geturl(),
        platform="instagram",
        platform_label="Instagram",
        intent="grow_followers",
        goal="fanbase",
    )

    if _IG_REEL.search(path):
        result.content_type = "reel"
    elif _IG_POST.search(path):
        result.content_type = "post"
    else:
        um = _IG_USER.match(path)
        if um and um.group(1) not in ("p", "reel", "stories", "explore", "accounts"):
            result.artist       = um.group(1)
            result.content_type = "profile"

    clean = _SCRIPT_STYLE.sub(" ", html)
    og_title = _strip_suffix(_og(clean, "title"))
    if og_title:
        result.title = og_title
    result.description = _og(clean, "description") or ""

    combined = " ".join(filter(None, [result.title, result.artist, result.description]))
    result.genre = _detect_genre(combined)
    result.mood  = _detect_mood(combined)
    return result


# ── SoundCloud extractor ──────────────────────────────────────────────────────

_SC_TRACK = re.compile(r"^/([\w-]+)/([\w-]+)$")
_SC_USER  = re.compile(r"^/([\w-]+)/?$")
_SC_PLAYLIST = re.compile(r"^/([\w-]+)/sets/([\w-]+)$")

def extract_soundcloud(html: str, parsed_url) -> ParsedUrl:
    path = parsed_url.path.rstrip("/")
    result = ParsedUrl(
        raw_url=parsed_url.geturl(),
        platform="soundcloud",
        platform_label="SoundCloud",
        intent="drive_streams",
        goal="streams",
    )

    pl = _SC_PLAYLIST.match(path)
    if pl:
        result.artist       = pl.group(1).replace("-", " ").title()
        result.content_type = "playlist"
    else:
        tk = _SC_TRACK.match(path)
        if tk:
            result.artist       = tk.group(1).replace("-", " ").title()
            result.title        = tk.group(2).replace("-", " ").title()
            result.content_type = "track"
        else:
            um = _SC_USER.match(path)
            if um:
                result.artist       = um.group(1).replace("-", " ").title()
                result.content_type = "profile"

    clean = _SCRIPT_STYLE.sub(" ", html)
    og_title = _strip_suffix(_og(clean, "title"))
    if og_title and not result.title:
        if " by " in og_title:
            parts = og_title.split(" by ", 1)
            result.title  = parts[0].strip()
            result.artist = result.artist or parts[1].strip()
        else:
            result.title = og_title

    result.description = _og(clean, "description") or ""
    combined = " ".join(filter(None, [result.title, result.artist, result.description]))
    result.genre = _detect_genre(combined)
    result.mood  = _detect_mood(combined)
    result.bpm   = _detect_bpm(combined)
    result.key   = _detect_key(combined)
    return result


# ── Apple Music extractor ─────────────────────────────────────────────────────

def extract_apple_music(html: str, parsed_url) -> ParsedUrl:
    path = parsed_url.path
    result = ParsedUrl(
        raw_url=parsed_url.geturl(),
        platform="apple_music",
        platform_label="Apple Music",
        intent="drive_streams",
        goal="streams",
    )

    if "/album/" in path:
        result.content_type = "album"
    elif "/playlist/" in path:
        result.content_type = "playlist"
    elif "/artist/" in path:
        result.content_type = "profile"
    else:
        result.content_type = "track"

    clean = _SCRIPT_STYLE.sub(" ", html)

    for ld in _load_json_ld(clean):
        t = str(ld.get("@type", "")).lower()
        if "musicalbum" in t:
            result.title  = _jld_str(ld, "name") or result.title
            by = ld.get("byArtist", {})
            if isinstance(by, dict):
                result.artist = _jld_str(by, "name") or result.artist
            result.release_year = _extract_year(_jld_str(ld, "datePublished")) or result.release_year
        elif "musicrecording" in t:
            result.title  = _jld_str(ld, "name") or result.title

    og_title = _strip_suffix(_og(clean, "title"))
    if og_title and not result.title:
        result.title = og_title
    result.description = _og(clean, "description") or ""

    combined = " ".join(filter(None, [result.title, result.artist, result.description]))
    result.genre = _detect_genre(combined)
    result.mood  = _detect_mood(combined)
    return result


# ── Bandcamp extractor ────────────────────────────────────────────────────────

def extract_bandcamp(html: str, parsed_url) -> ParsedUrl:
    host = (parsed_url.hostname or "").lower()
    path = parsed_url.path

    result = ParsedUrl(
        raw_url=parsed_url.geturl(),
        platform="bandcamp",
        platform_label="Bandcamp",
        intent="drive_conversion",
        goal="conversion",
    )

    # artist from subdomain: artist.bandcamp.com
    if host != "bandcamp.com" and host.endswith(".bandcamp.com"):
        result.artist = host.replace(".bandcamp.com", "").replace("-", " ").title()

    if "/album/" in path:
        result.content_type = "album"
        slug = path.split("/album/", 1)[1].rstrip("/").replace("-", " ").title()
        result.title = slug
    elif "/track/" in path:
        result.content_type = "track"
        slug = path.split("/track/", 1)[1].rstrip("/").replace("-", " ").title()
        result.title = slug

    clean = _SCRIPT_STYLE.sub(" ", html)

    for ld in _load_json_ld(clean):
        t = str(ld.get("@type", "")).lower()
        if "musicalbum" in t or "musicrecording" in t:
            result.title        = _jld_str(ld, "name") or result.title
            by = ld.get("byArtist", {})
            if isinstance(by, dict):
                result.artist   = _jld_str(by, "name") or result.artist
            result.release_year = _extract_year(_jld_str(ld, "datePublished")) or result.release_year

    og_title = _strip_suffix(_og(clean, "title"))
    if og_title and not result.title:
        result.title = og_title
    result.description = _og(clean, "description") or _extract_paragraphs(clean, 250)

    combined = " ".join(filter(None, [result.title, result.artist, result.description]))
    result.genre = _detect_genre(combined)
    result.mood  = _detect_mood(combined)
    result.bpm   = _detect_bpm(combined)
    result.key   = _detect_key(combined)
    return result


# ── Genius extractor (lyrics page → artist + track) ──────────────────────────

_GENIUS_PATH = re.compile(r"^/([A-Za-z0-9-]+)-lyrics/?$")

def extract_genius(html: str, parsed_url) -> ParsedUrl:
    path = parsed_url.path.rstrip("/")
    result = ParsedUrl(
        raw_url=parsed_url.geturl(),
        platform="genius",
        platform_label="Genius",
        intent="build_awareness",
        goal="awareness",
        content_type="article",
    )

    clean = _SCRIPT_STYLE.sub(" ", html)

    for ld in _load_json_ld(clean):
        t = str(ld.get("@type", "")).lower()
        if "musicrecording" in t:
            result.title       = _jld_str(ld, "name") or result.title
            by = ld.get("byArtist", {})
            if isinstance(by, dict):
                result.artist = _jld_str(by, "name") or result.artist

    og_title = _strip_suffix(_og(clean, "title"))
    if og_title and not result.title:
        # "Song Name Lyrics – Artist Name" or "Artist – Song"
        for sep in (" Lyrics – ", " – ", " - "):
            if sep in og_title:
                a, b = og_title.split(sep, 1)
                if "Lyrics" in b:
                    result.title, result.artist = a.strip(), b.replace("Lyrics", "").strip()
                else:
                    result.title, result.artist = b.strip(), a.strip()
                break
        if not result.title:
            result.title = og_title

    result.description = _og(clean, "description") or ""
    combined = " ".join(filter(None, [result.title, result.artist, result.description]))
    result.genre = _detect_genre(combined)
    result.mood  = _detect_mood(combined)
    return result


# ── Audiomack extractor ───────────────────────────────────────────────────────

def extract_audiomack(html: str, parsed_url) -> ParsedUrl:
    path = parsed_url.path
    result = ParsedUrl(
        raw_url=parsed_url.geturl(),
        platform="audiomack",
        platform_label="Audiomack",
        intent="drive_streams",
        goal="streams",
    )

    parts = [p for p in path.split("/") if p]
    if len(parts) >= 1:
        result.artist = parts[0].replace("-", " ").title()
    if len(parts) >= 3 and parts[1] in ("song", "album", "playlist"):
        result.content_type = parts[1] if parts[1] != "song" else "track"
        result.title        = parts[2].replace("-", " ").title()
    elif len(parts) == 1:
        result.content_type = "profile"

    clean = _SCRIPT_STYLE.sub(" ", html)
    og_title = _strip_suffix(_og(clean, "title"))
    if og_title:
        result.title = og_title
    result.description = _og(clean, "description") or ""

    combined = " ".join(filter(None, [result.title, result.artist, result.description]))
    result.genre = _detect_genre(combined)
    result.mood  = _detect_mood(combined)
    return result


# ── Editorial / media extractor ───────────────────────────────────────────────

def extract_editorial(html: str, parsed_url) -> ParsedUrl:
    result = ParsedUrl(
        raw_url=parsed_url.geturl(),
        platform="editorial",
        platform_label="Editorial",
        intent="build_awareness",
        goal="awareness",
        content_type="article",
    )

    clean = _SCRIPT_STYLE.sub(" ", html)

    for ld in _load_json_ld(clean):
        t = str(ld.get("@type", "")).lower()
        if "article" in t or "newsarticle" in t or "review" in t:
            result.title = _jld_str(ld, "headline") or _jld_str(ld, "name") or result.title
            author = ld.get("author") or {}
            if isinstance(author, dict):
                result.artist = _jld_str(author, "name") or result.artist
            result.description = _jld_str(ld, "description") or result.description

    og_title = _strip_suffix(_og(clean, "title"))
    if og_title and not result.title:
        result.title = og_title
    if not result.description:
        result.description = _og(clean, "description") or _extract_paragraphs(clean, 300)

    combined = " ".join(filter(None, [result.title, result.description]))
    result.genre = _detect_genre(combined)
    result.mood  = _detect_mood(combined)
    return result


# ── Twitter / X extractor ─────────────────────────────────────────────────────

_TW_STATUS = re.compile(r"/([\w]+)/status/(\d+)")
_TW_USER   = re.compile(r"^/([\w]+)/?$")

def extract_twitter(html: str, parsed_url) -> ParsedUrl:
    path = parsed_url.path
    result = ParsedUrl(
        raw_url=parsed_url.geturl(),
        platform="twitter",
        platform_label="Twitter/X",
        intent="drive_engagement",
        goal="engagement",
    )

    sm = _TW_STATUS.search(path)
    if sm:
        result.artist       = sm.group(1)
        result.content_type = "post"
    else:
        um = _TW_USER.match(path)
        if um and um.group(1) not in ("i", "home", "explore", "notifications"):
            result.artist       = um.group(1)
            result.content_type = "profile"

    clean = _SCRIPT_STYLE.sub(" ", html)
    og_title = _strip_suffix(_og(clean, "title"))
    if og_title:
        result.title = og_title
    result.description = _og(clean, "description") or ""

    combined = " ".join(filter(None, [result.title, result.artist, result.description]))
    result.genre = _detect_genre(combined)
    result.mood  = _detect_mood(combined)
    return result


# ── Linktree extractor ────────────────────────────────────────────────────────

def extract_linktree(html: str, parsed_url) -> ParsedUrl:
    path = parsed_url.path.lstrip("/")
    result = ParsedUrl(
        raw_url=parsed_url.geturl(),
        platform="linktree",
        platform_label="Linktree",
        intent="grow_followers",
        goal="fanbase",
        content_type="profile",
    )
    if path:
        result.artist = path.replace("-", " ").replace("_", " ").title()

    clean = _SCRIPT_STYLE.sub(" ", html)
    og_title = _strip_suffix(_og(clean, "title"))
    if og_title:
        result.title = og_title
    result.description = _og(clean, "description") or ""
    return result


# ── Deezer extractor ──────────────────────────────────────────────────────────

def extract_deezer(html: str, parsed_url) -> ParsedUrl:
    path = parsed_url.path
    result = ParsedUrl(
        raw_url=parsed_url.geturl(),
        platform="deezer",
        platform_label="Deezer",
        intent="drive_streams",
        goal="streams",
    )

    if "/track/" in path:
        result.content_type = "track"
    elif "/album/" in path:
        result.content_type = "album"
    elif "/playlist/" in path:
        result.content_type = "playlist"
    elif "/artist/" in path:
        result.content_type = "profile"

    clean = _SCRIPT_STYLE.sub(" ", html)
    og_title = _strip_suffix(_og(clean, "title"))
    if og_title:
        for sep in (" - ", " – ", " by "):
            if sep in og_title:
                p1, p2 = og_title.split(sep, 1)
                result.title  = p1.strip()
                result.artist = p2.strip()
                break
        if not result.title:
            result.title = og_title
    result.description = _og(clean, "description") or ""

    combined = " ".join(filter(None, [result.title, result.artist, result.description]))
    result.genre = _detect_genre(combined)
    result.mood  = _detect_mood(combined)
    return result


# ── DistroKid / TuneCore extractor ───────────────────────────────────────────

def extract_distro(html: str, parsed_url) -> ParsedUrl:
    host = (parsed_url.hostname or "").lower()
    result = ParsedUrl(
        raw_url=parsed_url.geturl(),
        platform="distribution",
        platform_label="DistroKid" if "distrokid" in host else "TuneCore",
        intent="drive_streams",
        goal="streams",
    )
    clean = _SCRIPT_STYLE.sub(" ", html)
    og_title = _strip_suffix(_og(clean, "title"))
    if og_title:
        result.title = og_title
    result.description = _og(clean, "description") or ""
    return result


# ── Spotify URI (non-HTTP) ────────────────────────────────────────────────────

_SPO_URI = re.compile(
    r"spotify:(track|album|playlist|artist):([A-Za-z0-9]+)",
    re.IGNORECASE,
)

def parse_spotify_uri(uri: str) -> Optional[ParsedUrl]:
    """Parse a spotify:track:XXX URI without HTML fetching."""
    m = _SPO_URI.match(uri.strip())
    if not m:
        return None
    content_type = m.group(1).lower()
    spotify_id   = m.group(2)
    return ParsedUrl(
        raw_url=uri,
        canonical_url=f"https://open.spotify.com/{content_type}/{spotify_id}",
        platform="spotify",
        platform_label="Spotify",
        content_type=content_type,
        intent="drive_streams",
        goal="streams",
        fetch_ok=False,
    )


# ── Registry: hostname → extractor function ───────────────────────────────────

def _mk_host_map() -> dict[str, object]:
    return {
        "open.spotify.com": extract_spotify,
        "spotify.com":      extract_spotify,
        "youtube.com":      extract_youtube,
        "www.youtube.com":  extract_youtube,
        "youtu.be":         extract_youtube,
        "m.youtube.com":    extract_youtube,
        "tiktok.com":       extract_tiktok,
        "www.tiktok.com":   extract_tiktok,
        "instagram.com":    extract_instagram,
        "www.instagram.com":extract_instagram,
        "soundcloud.com":   extract_soundcloud,
        "www.soundcloud.com":extract_soundcloud,
        "music.apple.com":  extract_apple_music,
        "bandcamp.com":     extract_bandcamp,
        "genius.com":       extract_genius,
        "audiomack.com":    extract_audiomack,
        "www.audiomack.com":extract_audiomack,
        "deezer.com":       extract_deezer,
        "www.deezer.com":   extract_deezer,
        "twitter.com":      extract_twitter,
        "x.com":            extract_twitter,
        "linktr.ee":        extract_linktree,
        "distrokid.com":    extract_distro,
        "www.distrokid.com":extract_distro,
        "tunecore.com":     extract_distro,
        "www.tunecore.com": extract_distro,
        "pitchfork.com":    extract_editorial,
        "www.pitchfork.com":extract_editorial,
        "rollingstone.com": extract_editorial,
        "www.rollingstone.com":extract_editorial,
        "hypebeast.com":    extract_editorial,
        "complex.com":      extract_editorial,
        "xxlmag.com":       extract_editorial,
        "hotnewhiphop.com": extract_editorial,
    }


_HOST_MAP = _mk_host_map()


def pick_extractor(hostname: str):
    """Return the best extractor function for *hostname* (falls back to generic)."""
    host = hostname.lower().lstrip("www.")
    # Exact match first
    if hostname in _HOST_MAP:
        return _HOST_MAP[hostname]
    # Strip www.
    if host in _HOST_MAP:
        return _HOST_MAP[host]
    # Subdomain match (e.g. artist.bandcamp.com)
    for domain, fn in _HOST_MAP.items():
        d = domain.lstrip("www.")
        if host.endswith("." + d):
            return fn
    return extract_generic
