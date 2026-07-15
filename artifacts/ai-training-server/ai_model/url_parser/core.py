"""Universal URL Parser — core orchestrator.

:func:`parse_url` is the single entry point.  It:

1. Validates and normalises the input.
2. Detects Spotify URIs (``spotify:track:…``) without HTTP.
3. Picks the best platform extractor from :mod:`extractors`.
4. Fetches the URL (HTML only, capped at 96 KB, timeout 5 s).
5. Runs the extractor.
6. Fills in remaining gaps with generic og-tag / slug fallbacks.
7. Generates :attr:`ParsedUrl.topic_string` and
   :attr:`ParsedUrl.awareness_text` for direct pipeline injection.

:func:`parse_topic_url` is a thin wrapper used when the ``topic`` field of a
generation request might be a URL — it resolves the URL and returns the
human-readable :attr:`~ParsedUrl.topic_string` (or the original string when
it is not a URL).

Both functions are **never-raise**.
"""
from __future__ import annotations

import html as _html
import ipaddress
import re
import socket
import urllib.request as _req
from urllib.parse import urlparse, urlunparse

from .extractors import (
    _SCRIPT_STYLE,
    extract_generic,
    parse_spotify_uri,
    pick_extractor,
)
from .models import ParsedUrl

# ── URL detection ─────────────────────────────────────────────────────────────

_HTTP_RE     = re.compile(r"^https?://\S+$",                            re.IGNORECASE)
_BARE_URL_RE = re.compile(r"^(www\.[\w.-]+|[\w-]+\.[\w-]{2,})\S*$",    re.IGNORECASE)
_SPO_URI_RE  = re.compile(r"^spotify:(track|album|playlist|artist):",   re.IGNORECASE)

# Known path slug junk words — stripped when building topic_string from URL path
_SLUG_JUNK = {
    "track", "artist", "album", "playlist", "watch", "embed", "user",
    "intl-en", "index.html", "index", "home", "en", "us", "post", "video",
    "reel", "shorts", "status", "p", "s", "song", "sets", "lyrics",
}


def _is_url(text: str) -> bool:
    """Return True when *text* looks like a URL or Spotify URI."""
    t = text.strip()
    return bool(
        _HTTP_RE.match(t)
        or _BARE_URL_RE.match(t)
        or _SPO_URI_RE.match(t)
    )


def _normalise_url(raw: str) -> str:
    """Prepend https:// when the scheme is absent; return the normalised URL."""
    raw = raw.strip()
    if _SPO_URI_RE.match(raw):
        return raw
    if not re.match(r"^https?://", raw, re.IGNORECASE):
        raw = "https://" + raw
    return raw


# ── HTTP fetch ────────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent":      "MaxCore/2.0 Universal-URL-Parser (+https://maxbooster.ai/bot)",
    "Accept":          "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
_MAX_BYTES = 98304   # 96 KB

# ── SSRF protection ───────────────────────────────────────────────────────────
# Only these schemes are allowed; file://, ftp://, gopher://, etc. are blocked.
_ALLOWED_SCHEMES = {"http", "https"}

# Private, loopback, link-local, and other reserved networks that must never
# be reachable from a server-side URL fetch (SSRF hardening).
_BLOCKED_NETWORKS = [
    ipaddress.ip_network(cidr) for cidr in [
        "127.0.0.0/8",       # loopback
        "::1/128",           # IPv6 loopback
        "10.0.0.0/8",        # RFC-1918 private
        "172.16.0.0/12",     # RFC-1918 private
        "192.168.0.0/16",    # RFC-1918 private
        "169.254.0.0/16",    # link-local / AWS metadata
        "fe80::/10",         # IPv6 link-local
        "fc00::/7",          # IPv6 unique-local
        "0.0.0.0/8",         # "this" network
        "100.64.0.0/10",     # shared address (CGN)
        "192.0.0.0/24",      # IANA special
        "192.0.2.0/24",      # TEST-NET-1
        "198.18.0.0/15",     # benchmarking
        "198.51.100.0/24",   # TEST-NET-2
        "203.0.113.0/24",    # TEST-NET-3
        "240.0.0.0/4",       # reserved (future)
        "255.255.255.255/32",# broadcast
    ]
]


def _is_safe_host(hostname: str) -> bool:
    """Return True only when *hostname* resolves to a public routable IP.

    Blocks loopback, private, link-local, and other reserved ranges.
    Never raises — treats resolution failure as unsafe (returns False).
    """
    try:
        # getaddrinfo returns all addresses; block if ANY resolves to a private range.
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        if not infos:
            return False
        for _family, _type, _proto, _canon, sockaddr in infos:
            ip_str = sockaddr[0]
            try:
                addr = ipaddress.ip_address(ip_str)
            except ValueError:
                return False
            for net in _BLOCKED_NETWORKS:
                if addr in net:
                    return False
        return True
    except Exception:
        return False


def _validate_url(url: str) -> bool:
    """Return True when *url* passes all SSRF safety checks."""
    try:
        parsed = urlparse(url)
        # Scheme must be http or https
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        # Reject numeric IP literals that are private without a DNS round-trip
        try:
            addr = ipaddress.ip_address(hostname)
            for net in _BLOCKED_NETWORKS:
                if addr in net:
                    return False
        except ValueError:
            pass  # not a bare IP literal — will check via DNS below
        return _is_safe_host(hostname)
    except Exception:
        return False


class _NoRedirectHandler(_req.HTTPRedirectHandler):
    """Redirect handler that raises immediately instead of following redirects.

    This lets us intercept each hop, validate the destination, and only
    then open a new connection — preventing SSRF via redirect chains.
    """
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise _RedirectTo(newurl, code)


class _RedirectTo(Exception):
    def __init__(self, url: str, code: int) -> None:
        self.url  = url
        self.code = code


_NO_REDIRECT_OPENER = _req.build_opener(_NoRedirectHandler())
_MAX_REDIRECTS = 3


def _fetch_html(url: str, timeout: float = 5.0) -> tuple[str, bool]:
    """Fetch *url* and return ``(html_body, fetch_ok)``.

    SSRF-hardened:
    - Only http/https schemes are permitted.
    - Private/link-local/loopback IP ranges are blocked **before** each
      connection attempt (initial URL *and* every redirect hop).
    - Redirects are followed manually: the opener raises instead of
      following automatically, so we validate the new destination URL
      before opening a second connection.  Max 3 hops.
    Never raises.
    """
    current_url = url
    for _hop in range(_MAX_REDIRECTS + 1):
        # ── Validate BEFORE opening the connection ────────────────────────
        if not _validate_url(current_url):
            return "", False
        try:
            request = _req.Request(current_url, headers=_HEADERS)
            with _NO_REDIRECT_OPENER.open(request, timeout=timeout) as resp:
                ct = resp.headers.get("content-type", "")
                if "text/html" not in ct and "application/xhtml" not in ct:
                    return "", False
                raw = resp.read(_MAX_BYTES).decode("utf-8", errors="replace")
            return raw, True
        except _RedirectTo as redir:
            # Redirect: validate the new URL before following
            current_url = redir.url
            continue
        except Exception:
            return "", False
    # Exceeded max redirect hops
    return "", False


# ── Slug extraction from URL path ─────────────────────────────────────────────

def _slug_from_path(path: str) -> str:
    """Derive a human-readable name from the URL path.  Returns '' when nothing useful."""
    parts = [
        p for p in path.split("/")
        if p and not p.isdigit() and len(p) > 2 and p.lower() not in _SLUG_JUNK
    ]
    if not parts:
        return ""
    raw = parts[-1].split("?")[0].split("#")[0]
    return re.sub(r"[-_]", " ", raw).title().strip()


# ── Content-type → intent mapping ─────────────────────────────────────────────

_CTYPE_TO_INTENT: dict[str, tuple[str, str]] = {
    "track":    ("drive_streams",    "streams"),
    "album":    ("drive_streams",    "streams"),
    "playlist": ("drive_streams",    "streams"),
    "video":    ("drive_streams",    "streams"),
    "short":    ("grow_followers",   "fanbase"),
    "reel":     ("grow_followers",   "fanbase"),
    "post":     ("drive_engagement", "engagement"),
    "profile":  ("grow_followers",   "fanbase"),
    "article":  ("build_awareness",  "awareness"),
    "page":     ("build_awareness",  "awareness"),
}


def _derive_intent(result: ParsedUrl) -> None:
    """Fill result.intent / result.goal when still empty (from content_type)."""
    if not result.intent:
        intent, goal = _CTYPE_TO_INTENT.get(result.content_type, ("build_awareness", "awareness"))
        result.intent = intent
        result.goal   = goal


# ── topic_string builder ──────────────────────────────────────────────────────

def _build_topic_string(result: ParsedUrl, hostname: str) -> str:
    """Compose a clean human-readable topic string from extracted metadata."""
    name_parts: list[str] = []

    if result.title and result.artist:
        name_parts.append(f"{result.title} — {result.artist}")
    elif result.title:
        name_parts.append(result.title)
    elif result.artist:
        name_parts.append(result.artist)

    if not name_parts:
        # Fall back to URL path slug
        slug = _slug_from_path(urlparse(result.canonical_url).path)
        name_parts.append(slug or hostname)

    base = name_parts[0]

    # Append platform + content-type context when informative
    if result.platform_label and result.platform_label not in ("Web", "Editorial"):
        ctype = result.content_type
        if ctype in ("track", "album", "playlist", "video", "short", "reel"):
            return f"{base} ({result.platform_label} {ctype})"
        elif ctype == "profile":
            return f"{base} on {result.platform_label}"
        else:
            return f"{base} — {result.platform_label}"

    return base


# ── awareness_text builder ────────────────────────────────────────────────────

_INTENT_LABELS = {
    "drive_streams":    "Drive streams",
    "grow_followers":   "Grow followers",
    "drive_engagement": "Drive engagement",
    "build_awareness":  "Build awareness",
    "drive_conversion": "Drive conversion",
}


def _build_awareness_text(result: ParsedUrl) -> str:
    """Format a multi-line awareness block ready for ScriptAgent injection."""
    lines: list[str] = []

    # Header signal — [HIGH] so it leads the awareness cascade
    if result.platform_label and result.platform_label not in ("Web",):
        ctype_label = result.content_type.replace("_", " ").title() if result.content_type else "Content"
        lines.append(f"[HIGH] Source: {result.platform_label} {ctype_label}")
    elif result.platform:
        lines.append(f"[HIGH] Source: {result.platform} content")

    # Entity signals
    if result.artist:
        lines.append(f"Artist: {result.artist}")
    if result.title and result.content_type != "profile":
        lines.append(f"Title: {result.title}")
    if result.album:
        lines.append(f"Album: {result.album}")

    # Music metadata
    music_parts: list[str] = []
    if result.genre:
        music_parts.append(f"Genre: {result.genre}")
    if result.mood:
        music_parts.append(f"Mood: {result.mood}")
    if music_parts:
        lines.append(" | ".join(music_parts))

    prod_parts: list[str] = []
    if result.bpm:
        prod_parts.append(f"BPM: {result.bpm}")
    if result.key:
        prod_parts.append(f"Key: {result.key}")
    if result.release_year:
        prod_parts.append(f"Year: {result.release_year}")
    if prod_parts:
        lines.append(" | ".join(prod_parts))

    # Intent
    if result.intent:
        intent_label = _INTENT_LABELS.get(result.intent, result.intent.replace("_", " ").title())
        lines.append(f"Intent: {intent_label}")

    # Description (trimmed)
    if result.description:
        desc = result.description[:200].strip()
        if desc:
            lines.append(f"Context: {desc}")

    return "\n".join(lines)


# ── Themes / topics extractor ─────────────────────────────────────────────────

_THEME_RE = re.compile(
    r"\b(release|new music|dropping|drop|out now|available|stream|listen|video|"
    r"debut|freestyle|collab|collaboration|remix|single|ep|album|tour|live|"
    r"exclusive|official|behind the scenes|acoustic|unplugged|cover|sample|"
    r"announce|announcement|hype|presave|pre-save|merch|tickets)\b",
    re.IGNORECASE,
)


def _extract_themes(result: ParsedUrl) -> list[str]:
    combined = " ".join(filter(None, [result.title, result.description, result.body_text]))
    seen: set[str] = set()
    themes: list[str] = []
    for m in _THEME_RE.finditer(combined):
        t = m.group(1).lower().replace(" ", "_")
        if t not in seen:
            seen.add(t)
            themes.append(t)
    return themes[:8]


# ── Public API ────────────────────────────────────────────────────────────────

def parse_url(url: str, timeout: float = 5.0) -> ParsedUrl:
    """Parse *url* and return a richly-populated :class:`ParsedUrl`.

    Never raises.  On any error returns a partial result with ``error`` set.
    """
    if not url or not url.strip():
        return ParsedUrl(error="empty input")

    raw = url.strip()

    # ── Spotify URI (no fetch needed) ─────────────────────────────────────
    if _SPO_URI_RE.match(raw):
        result = parse_spotify_uri(raw)
        if result is None:
            result = ParsedUrl(raw_url=raw, error="invalid spotify URI")
        _derive_intent(result)
        result.topic_string   = _build_topic_string(result, "spotify.com")
        result.awareness_text = _build_awareness_text(result)
        return result

    # ── Normalise URL ─────────────────────────────────────────────────────
    canonical = _normalise_url(raw)
    try:
        parsed = urlparse(canonical)
        if not parsed.hostname:
            return ParsedUrl(raw_url=raw, error="could not parse hostname")
        hostname = parsed.hostname.lower()
    except Exception as exc:
        return ParsedUrl(raw_url=raw, error=str(exc))

    # ── Pick extractor ────────────────────────────────────────────────────
    extractor = pick_extractor(hostname)

    # ── Fetch HTML ────────────────────────────────────────────────────────
    html_body, fetch_ok = _fetch_html(canonical, timeout=timeout)

    # ── Run extractor ─────────────────────────────────────────────────────
    try:
        if fetch_ok and html_body:
            result = extractor(html_body, parsed)
        else:
            # No HTML — use a bare partial result with platform signals only
            result = extractor("", parsed) if extractor is not extract_generic else ParsedUrl()
            result.raw_url = raw
    except Exception as exc:
        result          = ParsedUrl(raw_url=raw, error=str(exc))

    result.raw_url      = raw
    result.canonical_url = canonical
    result.fetch_ok      = fetch_ok

    # ── Fill gaps with slug fallback ──────────────────────────────────────
    if not result.title and html_body:
        # Already tried by extractor; last-resort: bare og:title
        clean = _SCRIPT_STYLE.sub(" ", html_body)
        og = _html.unescape(
            re.search(
                r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\'<>]+)["\']',
                clean, re.IGNORECASE,
            ).group(1)
            if re.search(
                r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\'<>]+)["\']',
                clean, re.IGNORECASE,
            ) else ""
        ).strip()
        if og:
            result.title = og

    if not result.title:
        slug = _slug_from_path(parsed.path)
        if slug:
            result.title = slug

    # ── Derive intent / goal ──────────────────────────────────────────────
    _derive_intent(result)

    # ── Extract content themes ─────────────────────────────────────────────
    result.content_themes = result.content_themes or _extract_themes(result)

    # ── Build pipeline outputs ────────────────────────────────────────────
    result.topic_string   = _build_topic_string(result, hostname)
    result.awareness_text = _build_awareness_text(result)

    return result


def parse_topic_url(raw_topic: str, timeout: float = 4.0) -> str:
    """If *raw_topic* looks like a URL, resolve it and return the topic string.

    When *raw_topic* is plain text (not a URL), return it unchanged.
    Never raises.
    """
    if not raw_topic or not raw_topic.strip():
        return raw_topic
    t = raw_topic.strip()
    if not _is_url(t):
        return t
    try:
        result = parse_url(t, timeout=timeout)
        return result.topic_string or t
    except Exception:
        return t


def is_url(text: str) -> bool:
    """Return True when *text* looks like a URL the parser can handle."""
    return _is_url(text)


def get_content_from_url(url: str, timeout: float = 4.0,
                         platform: str = "") -> dict:
    """Return the FULL extracted content for *url* as a plain dict, with
    the Veo quality DNA from the awareness systems blended into the
    awareness block.

    This is the rich accessor: where :func:`parse_topic_url` returns only a
    topic string, this returns every extracted field (title, artist, album,
    description, body text, genre/mood/BPM/key, themes, intent/goal) plus:

    - ``awareness_text``  — the parser's own awareness block
    - ``veo_dna``         — Veo scoring DNA from the quality-awareness system
                            (power-word hook rules, 15-60-word length rule,
                            structure + CTA rules, live chart patterns)
    - ``awareness_full``  — awareness_text + veo_dna, ready for direct
                            injection into any generation ``awareness`` field

    Never raises; on non-URL input or parse failure returns a dict with
    ``fetch_ok=False`` and the raw input echoed in ``raw_url``/``topic_string``.
    """
    t = (url or "").strip()

    def _veo_dna_block() -> str:
        try:
            from ai_model.quality_awareness import veo_dna as _veo_dna
            return _veo_dna(platform)
        except Exception:
            return ""

    empty = {
        "raw_url": t, "canonical_url": "", "platform": "", "platform_label": "",
        "content_type": "page", "title": "", "artist": "", "album": "",
        "label": "", "description": "", "body_text": "", "genre": "",
        "mood": "", "bpm": None, "key": "", "release_year": None,
        "topics": [], "content_themes": [], "intent": "", "goal": "",
        "topic_string": t, "awareness_text": "", "fetch_ok": False,
        "error": "not a URL" if t and not _is_url(t) else "empty input",
    }

    if not t or not _is_url(t):
        dna = _veo_dna_block()
        return {**empty, "veo_dna": dna, "awareness_full": dna}

    try:
        r = parse_url(t, timeout=timeout)
        dna = _veo_dna_block()
        awareness_full = "\n".join(
            b for b in (r.awareness_text, dna) if b
        )
        return {
            "raw_url": r.raw_url, "canonical_url": r.canonical_url,
            "platform": r.platform, "platform_label": r.platform_label,
            "content_type": r.content_type, "title": r.title,
            "artist": r.artist, "album": r.album, "label": r.label,
            "description": r.description, "body_text": r.body_text,
            "genre": r.genre, "mood": r.mood, "bpm": r.bpm, "key": r.key,
            "release_year": r.release_year, "topics": list(r.topics),
            "content_themes": list(r.content_themes), "intent": r.intent,
            "goal": r.goal, "topic_string": r.topic_string,
            "awareness_text": r.awareness_text, "fetch_ok": r.fetch_ok,
            "error": r.error, "veo_dna": dna, "awareness_full": awareness_full,
        }
    except Exception as exc:  # noqa: BLE001 — never-raise contract
        dna = _veo_dna_block()
        return {**empty, "error": str(exc)[:200], "veo_dna": dna,
                "awareness_full": dna}
