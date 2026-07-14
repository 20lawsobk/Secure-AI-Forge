"""URL content reader for intent detection.

Fetches a URL and extracts structured text signals: og:title, og:description,
meta:description, first h1, first paragraphs, and platform/goal hints derived
from the hostname.  Returns a :class:`UrlContent` dataclass.  Never raises.
"""
from __future__ import annotations

import html as _html
import re
import urllib.request as _urllib_request
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Platform + goal hint tables
# ---------------------------------------------------------------------------

_PLATFORM_BY_DOMAIN: dict[str, str] = {
    "tiktok.com":       "tiktok",
    "instagram.com":    "instagram",
    "youtube.com":      "youtube",
    "youtu.be":         "youtube",
    "spotify.com":      "spotify",
    "soundcloud.com":   "soundcloud",
    "music.apple.com":  "apple_music",
    "apple.com":        "apple_music",
    "twitter.com":      "twitter",
    "x.com":            "twitter",
    "linkedin.com":     "linkedin",
    "facebook.com":     "facebook",
    "deezer.com":       "deezer",
    "audiomack.com":    "audiomack",
    "bandcamp.com":     "bandcamp",
    "distrokid.com":    "distrokid",
    "tunecore.com":     "tunecore",
    "linktr.ee":        "linktree",
    "genius.com":       "genius",
    "pitchfork.com":    "editorial",
    "rollingstone.com": "editorial",
    "hypebeast.com":    "editorial",
    "complex.com":      "editorial",
    "xxlmag.com":       "editorial",
    "hotnewhiphop.com": "editorial",
}

_GOAL_BY_DOMAIN: dict[str, str] = {
    "spotify.com":      "drive_streams",
    "music.apple.com":  "drive_streams",
    "apple.com":        "drive_streams",
    "soundcloud.com":   "drive_streams",
    "audiomack.com":    "drive_streams",
    "youtube.com":      "drive_streams",
    "youtu.be":         "drive_streams",
    "deezer.com":       "drive_streams",
    "bandcamp.com":     "drive_conversion",
    "ticketmaster.com": "drive_conversion",
    "eventbrite.com":   "drive_conversion",
    "shopify.com":      "drive_conversion",
    "distrokid.com":    "drive_streams",
    "tunecore.com":     "drive_streams",
    "genius.com":       "build_awareness",
    "pitchfork.com":    "build_awareness",
    "rollingstone.com": "build_awareness",
    "hypebeast.com":    "build_awareness",
    "complex.com":      "build_awareness",
    "instagram.com":    "grow_followers",
    "tiktok.com":       "grow_followers",
    "twitter.com":      "drive_engagement",
    "x.com":            "drive_engagement",
}

# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"^https?://[^\s/$.?#][^\s]*$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# HTML cleaning helpers
# ---------------------------------------------------------------------------

_TAG_RE       = re.compile(r"<[^>]+>")
_WHITESPACE   = re.compile(r"\s+")
_SCRIPT_STYLE = re.compile(
    r"<(script|style|noscript|nav|footer|header)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_P_TAG        = re.compile(r"<p[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)
_H1_TAG       = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)


def _clean(html: str) -> str:
    """Strip tags and collapse whitespace from an HTML fragment."""
    return _WHITESPACE.sub(" ", _TAG_RE.sub(" ", _html.unescape(html))).strip()


def _og(html: str, prop: str) -> str:
    """Extract og:<prop> or name=<prop> meta content (both attribute orderings)."""
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


def _meta_desc(html: str) -> str:
    return _og(html, "description") or _og(html, "description")


def _extract_paragraphs(html: str, max_chars: int = 400) -> str:
    """Pull the first few <p> tag texts, stripped of markup, up to max_chars."""
    parts: List[str] = []
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


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class UrlContent:
    """Structured content extracted from a URL fetch."""
    title:        str = ""
    description:  str = ""
    body_text:    str = ""
    platform_hint: str = ""    # tiktok | instagram | youtube | spotify | …
    goal_hint:    str = ""     # drive_streams | grow_followers | …
    hostname:     str = ""

    def combined(self) -> str:
        """Concatenated text for the intent detector to scan."""
        return " ".join(p for p in (self.title, self.description, self.body_text) if p)

    def is_empty(self) -> bool:
        return not (self.title or self.description or self.body_text)


# ---------------------------------------------------------------------------
# Public read function
# ---------------------------------------------------------------------------

def read_url(url: str, timeout: float = 5.0) -> UrlContent:
    """Fetch *url* and return a :class:`UrlContent`.

    Resolution:
    1. Derive platform/goal hints from the hostname (always available).
    2. HTTP GET (text/html only, capped at 64 KB).
    3. Extract og:title → <title> → first h1 as title.
    4. Extract og:description → meta:description → first paragraphs as
       description/body.

    Never raises — returns a partial :class:`UrlContent` on any error.
    """
    content = UrlContent()
    if not url or not _URL_RE.match(url.strip()):
        return content

    url = url.strip()
    try:
        parsed  = urlparse(url)
        host    = (parsed.hostname or "").lower().lstrip("www.")
        content.hostname = host
    except Exception:
        return content

    # Hostname → platform / goal hints (works without a fetch)
    for domain, plat in _PLATFORM_BY_DOMAIN.items():
        if host == domain or host.endswith("." + domain):
            content.platform_hint = plat
            break
    for domain, goal in _GOAL_BY_DOMAIN.items():
        if host == domain or host.endswith("." + domain):
            content.goal_hint = goal
            break

    # HTTP fetch
    try:
        req = _urllib_request.Request(
            url,
            headers={
                "User-Agent":      "MaxCore/1.0 (+https://maxbooster.ai/bot)",
                "Accept":          "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        )
        with _urllib_request.urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("content-type", "")
            if "text/html" not in ct:
                return content
            raw = resp.read(65536).decode("utf-8", errors="replace")
    except Exception:
        return content

    # Strip script/style noise before parsing
    raw = _SCRIPT_STYLE.sub(" ", raw)

    # Title
    og_title = _og(raw, "title")
    if og_title:
        content.title = og_title
    else:
        m = re.search(r"<title[^>]*>([^<]{3,})</title>", raw, re.IGNORECASE)
        if m:
            content.title = _html.unescape(m.group(1)).strip()
        else:
            h1_m = _H1_TAG.search(raw)
            if h1_m:
                content.title = _clean(h1_m.group(1))

    # Description
    og_desc = _og(raw, "description")
    if og_desc:
        content.description = og_desc
    else:
        content.description = _extract_paragraphs(raw, 300)

    # Body (extra paragraph text when description is short)
    if len(content.description) < 150:
        extra = _extract_paragraphs(raw, 400)
        if extra and extra != content.description:
            content.body_text = extra

    return content
