from __future__ import annotations
import re
import json
import logging
from typing import Dict, Any, Optional
from urllib.parse import urlparse

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

logger = logging.getLogger(__name__)

PLATFORM_PATTERNS = {
    "spotify": [
        r"open\.spotify\.com/track/([a-zA-Z0-9]+)",
        r"open\.spotify\.com/album/([a-zA-Z0-9]+)",
        r"spotify\.link/",
    ],
    "youtube": [
        r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]+)",
        r"music\.youtube\.com/watch\?v=([a-zA-Z0-9_-]+)",
    ],
    "soundcloud": [
        r"soundcloud\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)",
        r"on\.soundcloud\.com/",
    ],
    "apple_music": [
        r"music\.apple\.com/.+/album/.+/(\d+)",
        r"music\.apple\.com/.+/song/.+",
    ],
    "tidal": [
        r"tidal\.com/(?:browse/)?track/(\d+)",
        r"listen\.tidal\.com/track/(\d+)",
    ],
    "deezer": [
        r"deezer\.com/.*/track/(\d+)",
        r"deezer\.page\.link/",
    ],
    "bandcamp": [
        r"([a-zA-Z0-9_-]+)\.bandcamp\.com/track/([a-zA-Z0-9_-]+)",
        r"([a-zA-Z0-9_-]+)\.bandcamp\.com/album/([a-zA-Z0-9_-]+)",
    ],
    "audiomack": [
        r"audiomack\.com/([a-zA-Z0-9_-]+)/song/([a-zA-Z0-9_-]+)",
    ],
    "amazon_music": [
        r"music\.amazon\..+/albums/([a-zA-Z0-9]+)",
        r"amazon\..+/dp/([a-zA-Z0-9]+)",
    ],
    "pandora": [
        r"pandora\.com/artist/.+/(.+)/(.+)",
    ],
    "vevo": [
        r"vevo\.com/watch/(.+)",
    ],
}

OEMBED_ENDPOINTS = {
    "youtube": "https://www.youtube.com/oembed?url={url}&format=json",
    "soundcloud": "https://soundcloud.com/oembed?url={url}&format=json",
}

WEBSITE_PROMO_GOALS = [
    "ad_creative", "promo_reel", "promo_clip", "teaser_trailer",
    "hook_clip", "fan_engagement",
]

BRAND_MOOD_KEYWORDS = {
    "energetic": ["launch", "new", "exciting", "powerful", "boost", "fast", "grow", "scale", "ultimate", "supercharge"],
    "uplifting": ["empower", "success", "achieve", "dream", "inspire", "transform", "create", "build", "future"],
    "dark": ["disrupt", "revolutionary", "bold", "edge", "cutting-edge", "underground"],
    "chill": ["simple", "easy", "seamless", "smooth", "intuitive", "minimal", "clean"],
    "mysterious": ["discover", "explore", "unlock", "secret", "exclusive", "premium", "vip"],
}

MOOD_KEYWORDS = {
    "energetic": ["hype", "energy", "pump", "fire", "lit", "bass", "trap", "edm", "dance", "party", "club", "bounce"],
    "dark": ["dark", "grim", "sinister", "shadow", "night", "doom", "heavy", "goth", "emo"],
    "chill": ["chill", "relax", "vibe", "smooth", "lofi", "lo-fi", "mellow", "calm", "ambient", "peaceful"],
    "romantic": ["love", "romance", "heart", "passion", "tender", "sweet", "intimate", "soulful", "rnb", "r&b"],
    "aggressive": ["rage", "angry", "hard", "metal", "punk", "hardcore", "scream", "intense", "drill"],
    "melancholic": ["sad", "pain", "cry", "tears", "lonely", "broken", "miss", "lost", "blues"],
    "uplifting": ["happy", "joy", "hope", "bright", "upbeat", "positive", "sunshine", "feel good", "euphoria"],
    "mysterious": ["mystery", "ethereal", "dream", "psychedelic", "trippy", "surreal", "cosmic", "space"],
}

GENRE_TO_ERA = {
    "hip-hop": "modern", "rap": "modern", "trap": "modern", "drill": "modern",
    "pop": "modern", "indie": "modern", "electronic": "modern", "edm": "modern",
    "rock": "classic", "metal": "classic", "punk": "classic",
    "jazz": "vintage", "blues": "vintage", "soul": "vintage", "funk": "vintage",
    "classical": "vintage", "r&b": "modern", "rnb": "modern",
    "reggae": "classic", "country": "classic", "folk": "classic",
    "lo-fi": "modern", "lofi": "modern", "ambient": "modern",
}


class UrlMetadataExtractor:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def detect_platform(self, url: str) -> str:
        for platform, patterns in PLATFORM_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return platform

        parsed = urlparse(url)
        host = parsed.hostname or ""
        for platform in PLATFORM_PATTERNS:
            if platform.replace("_", "") in host.replace(".", "").replace("-", ""):
                return platform

        return "unknown"

    def extract_id_from_url(self, url: str, platform: str) -> Optional[str]:
        patterns = PLATFORM_PATTERNS.get(platform, [])
        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1) if match.groups() else None
        return None

    def _fetch_url(self, url: str) -> Optional[str]:
        if not HAS_URLLIB:
            return None
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "MaxBooster/1.0 (Music Campaign Generator)",
                "Accept": "text/html,application/xhtml+xml,application/json",
            })
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def _fetch_oembed(self, url: str, platform: str) -> Optional[Dict[str, Any]]:
        endpoint = OEMBED_ENDPOINTS.get(platform)
        if not endpoint:
            return None

        oembed_url = endpoint.format(url=url)
        raw = self._fetch_url(oembed_url)
        if not raw:
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _extract_og_tags(self, html: str) -> Dict[str, str]:
        tags = {}
        og_pattern = r'<meta\s+(?:property|name)=["\'](?:og:|music:|twitter:)([^"\']+)["\']\s+content=["\']([^"\']*)["\']'
        for match in re.finditer(og_pattern, html, re.IGNORECASE):
            tags[match.group(1)] = match.group(2)

        og_pattern_rev = r'<meta\s+content=["\']([^"\']*?)["\']\s+(?:property|name)=["\'](?:og:|music:|twitter:)([^"\']+)["\']'
        for match in re.finditer(og_pattern_rev, html, re.IGNORECASE):
            tags[match.group(2)] = match.group(1)

        title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if title_match and "title" not in tags:
            tags["title"] = title_match.group(1).strip()

        desc_match = re.search(r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']*)["\']', html, re.IGNORECASE)
        if desc_match and "description" not in tags:
            tags["description"] = desc_match.group(1)

        return tags

    def _parse_title_artist(self, raw_title: str, platform: str) -> Dict[str, str]:
        result = {"title": raw_title, "artist": ""}

        separators = [" - ", " — ", " – ", " by ", " | "]
        for sep in separators:
            if sep in raw_title:
                parts = raw_title.split(sep, 1)
                if platform in ("youtube", "vevo"):
                    result["artist"] = parts[0].strip()
                    result["title"] = parts[1].strip()
                else:
                    result["title"] = parts[0].strip()
                    result["artist"] = parts[1].strip()
                break

        suffixes_to_remove = [
            " - YouTube", " - YouTube Music", " on Apple Music",
            " | Spotify", " on Spotify", " by ", " on SoundCloud",
            " | Free Listening on SoundCloud", " - song and lyrics",
            " | Deezer", " | TIDAL", " on Bandcamp", " | Audiomack",
        ]
        for suffix in suffixes_to_remove:
            if result["title"].endswith(suffix):
                result["title"] = result["title"][:-len(suffix)].strip()
            if result["artist"].endswith(suffix):
                result["artist"] = result["artist"][:-len(suffix)].strip()

        return result

    def _infer_mood(self, text: str) -> str:
        text_lower = text.lower()
        scores = {}
        for mood, keywords in MOOD_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[mood] = score

        if scores:
            return max(scores, key=scores.get)
        return "energetic"

    def _infer_era(self, text: str) -> str:
        text_lower = text.lower()
        for genre, era in GENRE_TO_ERA.items():
            if genre in text_lower:
                return era
        return "modern"

    def _build_story(self, metadata: Dict[str, Any]) -> str:
        parts = []
        title = metadata.get("title", "")
        artist = metadata.get("artist", "")
        mood = metadata.get("mood", "energetic")
        genre = metadata.get("genre", "")

        if artist and title:
            parts.append(f'"{title}" by {artist}')
        elif title:
            parts.append(f'"{title}"')

        if genre:
            parts.append(f"A {genre} track")
        if mood:
            parts.append(f"with a {mood} vibe")

        if not parts:
            return "Music video campaign"

        return " ".join(parts) + ". Generate visuals that capture the energy and mood of the track."

    def _extract_website_metadata(self, url: str, existing: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "description": "",
            "site_name": "",
            "page_type": "",
            "keywords": [],
        }

        html = self._fetch_url(url)
        if not html:
            parsed_url = urlparse(url)
            host = parsed_url.hostname or ""
            domain_name = host.replace("www.", "").split(".")[0].title()
            if not existing.get("title"):
                result["title"] = domain_name
            if not existing.get("artist"):
                result["artist"] = domain_name
            result["extraction_method"] = "url_parsing"
            result["platform"] = "website"
            return result

        og_tags = self._extract_og_tags(html)
        method = "opengraph" if existing.get("extraction_method") == "none" else existing.get("extraction_method", "") + "+opengraph"
        result["extraction_method"] = method
        result["platform"] = "website"

        raw_title = og_tags.get("title", "")
        if raw_title and not existing.get("title"):
            result["title"] = raw_title

        site_name = og_tags.get("site_name", "")
        if site_name:
            result["site_name"] = site_name
            if not existing.get("artist"):
                result["artist"] = site_name

        if not result.get("artist") and not existing.get("artist"):
            parsed_url = urlparse(url)
            host = parsed_url.hostname or ""
            result["artist"] = host.replace("www.", "").split(".")[0].title()

        if og_tags.get("image") and not existing.get("artwork_url"):
            result["artwork_url"] = og_tags["image"]

        desc = og_tags.get("description", "")
        if desc:
            result["description"] = desc

        if not desc:
            meta_desc = re.search(
                r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']*)["\']',
                html, re.IGNORECASE
            )
            if meta_desc:
                result["description"] = meta_desc.group(1)

        kw_match = re.search(
            r'<meta\s+name=["\']keywords["\']\s+content=["\']([^"\']*)["\']',
            html, re.IGNORECASE
        )
        if kw_match:
            result["keywords"] = [k.strip() for k in kw_match.group(1).split(",") if k.strip()]

        page_type = og_tags.get("type", "")
        if page_type:
            result["page_type"] = page_type

        text_content = f"{result.get('title', '')} {result.get('description', '')} {' '.join(result.get('keywords', []))}"
        for genre in GENRE_TO_ERA:
            if genre.lower() in text_content.lower():
                result["genre"] = genre
                break

        if not result.get("genre"):
            result["genre"] = "promotional"

        return result

    def _infer_brand_mood(self, text: str) -> str:
        text_lower = text.lower()
        scores = {}
        for mood, keywords in BRAND_MOOD_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[mood] = score

        if not scores:
            for mood, keywords in MOOD_KEYWORDS.items():
                score = sum(1 for kw in keywords if kw in text_lower)
                if score > 0:
                    scores[mood] = score

        if scores:
            return max(scores, key=scores.get)
        return "energetic"

    def _build_website_story(self, metadata: Dict[str, Any]) -> str:
        parts = []
        title = metadata.get("title", "")
        artist = metadata.get("artist", metadata.get("site_name", ""))
        description = metadata.get("description", "")

        if artist and title:
            if artist.lower() in title.lower():
                parts.append(f"Promote {title}.")
            else:
                parts.append(f"Promote {title} by {artist}.")
        elif title:
            parts.append(f"Promote {title}.")

        if description:
            short_desc = description[:200].rstrip(".")
            parts.append(f"{short_desc}.")

        parts.append("Create eye-catching promotional video content that drives engagement and conversions.")

        return " ".join(parts) if parts else "Promotional video campaign for brand awareness and engagement."

    def extract(self, url: str) -> Dict[str, Any]:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        platform = self.detect_platform(url)
        track_id_from_url = self.extract_id_from_url(url, platform)

        metadata: Dict[str, Any] = {
            "url": url,
            "platform": platform,
            "platform_track_id": track_id_from_url,
            "title": "",
            "artist": "",
            "album": "",
            "artwork_url": "",
            "genre": "",
            "duration_sec": None,
            "mood": "energetic",
            "era": "modern",
            "story": "",
            "confidence": 0.0,
            "missing_fields": [],
            "errors": [],
            "extraction_method": "none",
        }

        oembed_data = self._fetch_oembed(url, platform)
        if oembed_data:
            metadata["extraction_method"] = "oembed"
            if "title" in oembed_data:
                parsed = self._parse_title_artist(oembed_data["title"], platform)
                metadata["title"] = parsed["title"]
                if parsed["artist"]:
                    metadata["artist"] = parsed["artist"]
            if "author_name" in oembed_data:
                metadata["artist"] = oembed_data["author_name"]
            if "thumbnail_url" in oembed_data:
                metadata["artwork_url"] = oembed_data["thumbnail_url"]

        if not metadata["title"]:
            html = self._fetch_url(url)
            if html:
                og_tags = self._extract_og_tags(html)
                metadata["extraction_method"] = "opengraph" if metadata["extraction_method"] == "none" else metadata["extraction_method"] + "+opengraph"

                raw_title = og_tags.get("title", "")
                if raw_title:
                    parsed = self._parse_title_artist(raw_title, platform)
                    if not metadata["title"]:
                        metadata["title"] = parsed["title"]
                    if not metadata["artist"] and parsed["artist"]:
                        metadata["artist"] = parsed["artist"]

                if og_tags.get("site_name") and not metadata["artist"]:
                    site = og_tags["site_name"]
                    if site not in ("YouTube", "SoundCloud", "Spotify", "Apple Music", "Deezer", "TIDAL"):
                        metadata["artist"] = site

                if og_tags.get("image") and not metadata["artwork_url"]:
                    metadata["artwork_url"] = og_tags["image"]

                if og_tags.get("description"):
                    desc = og_tags["description"]

                    if platform == "spotify" and not metadata["artist"]:
                        spotify_match = re.match(r"^(.+?)(?:\s+·\s+|\s+\|\s+)", desc)
                        if spotify_match:
                            metadata["artist"] = spotify_match.group(1).strip()
                        elif " · " in desc:
                            metadata["artist"] = desc.split(" · ")[0].strip()

                    if platform == "soundcloud" and not metadata["artist"]:
                        sc_match = re.search(r"Stream (.+?) by (.+?) on", desc)
                        if sc_match:
                            if not metadata["title"] or metadata["title"] == "SoundCloud":
                                metadata["title"] = sc_match.group(1).strip()
                            metadata["artist"] = sc_match.group(2).strip()

                    if not metadata["genre"]:
                        for genre in GENRE_TO_ERA:
                            if genre.lower() in desc.lower():
                                metadata["genre"] = genre
                                break

                if og_tags.get("duration"):
                    try:
                        metadata["duration_sec"] = float(og_tags["duration"])
                    except ValueError:
                        pass

                if og_tags.get("musician"):
                    musician_val = og_tags["musician"]
                    if not musician_val.startswith("http"):
                        metadata["artist"] = musician_val

        if not metadata["title"] and platform != "unknown":
            metadata["extraction_method"] = "url_parsing"
            parsed_url = urlparse(url)
            path_parts = [p for p in parsed_url.path.split("/") if p]

            if platform == "bandcamp" and parsed_url.hostname:
                metadata["artist"] = parsed_url.hostname.split(".")[0].replace("-", " ").title()
                if len(path_parts) >= 2:
                    metadata["title"] = path_parts[-1].replace("-", " ").title()

            elif platform == "soundcloud" and len(path_parts) >= 2:
                metadata["artist"] = path_parts[0].replace("-", " ").title()
                metadata["title"] = path_parts[1].replace("-", " ").title()

            elif platform == "audiomack" and len(path_parts) >= 3:
                metadata["artist"] = path_parts[0].replace("-", " ").title()
                metadata["title"] = path_parts[2].replace("-", " ").title()

        if platform == "unknown":
            metadata["content_type"] = "website"
            website_data = self._extract_website_metadata(url, metadata)
            metadata.update(website_data)
        else:
            metadata["content_type"] = "music"

        all_text = f"{metadata['title']} {metadata['artist']} {metadata['genre']} {metadata.get('description', '')}"
        if metadata["content_type"] == "website":
            metadata["mood"] = self._infer_brand_mood(all_text)
        else:
            metadata["mood"] = self._infer_mood(all_text)
        metadata["era"] = self._infer_era(all_text)

        if metadata["content_type"] == "website":
            metadata["story"] = self._build_website_story(metadata)
        else:
            metadata["story"] = self._build_story(metadata)

        missing = []
        if not metadata["title"]:
            missing.append("title")
        if not metadata["artist"]:
            missing.append("artist")
        if not metadata["genre"]:
            missing.append("genre")
        if metadata["duration_sec"] is None:
            missing.append("duration_sec")
        metadata["missing_fields"] = missing

        filled = sum(1 for f in ["title", "artist", "genre", "artwork_url"]
                      if metadata.get(f))
        metadata["confidence"] = round(filled / 4.0, 2)

        if platform == "unknown" and metadata.get("content_type") != "website":
            metadata["errors"].append("Could not identify music platform from URL")
            metadata["confidence"] = max(0.0, metadata["confidence"] - 0.25)

        return metadata

    def metadata_to_campaign_request(
        self,
        metadata: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        overrides = overrides or {}

        from ..model.platform_heads import PLATFORM_DEFAULTS

        source_platform = metadata.get("platform", "unknown")
        content_type = metadata.get("content_type", "music")

        default_platforms = overrides.get("primary_platforms")
        if not default_platforms:
            if content_type == "website":
                default_platforms = ["tiktok", "youtube", "instagram", "reels", "shorts", "facebook"]
            else:
                always_platforms = ["tiktok", "youtube", "instagram"]
                if source_platform in PLATFORM_DEFAULTS and source_platform not in always_platforms:
                    default_platforms = always_platforms + [source_platform]
                else:
                    default_platforms = always_platforms

        request = {
            "title": overrides.get("title", metadata.get("title", "Untitled")),
            "artist": overrides.get("artist", metadata.get("artist", "Unknown Artist")),
            "album": overrides.get("album", metadata.get("album")),
            "mood": overrides.get("mood", metadata.get("mood", "energetic")),
            "era": overrides.get("era", metadata.get("era", "modern")),
            "story": overrides.get("story", metadata.get("story", "")),
            "primary_platforms": default_platforms,
            "audio_duration_sec": overrides.get("audio_duration_sec", metadata.get("duration_sec") or 180.0),
            "source_url": metadata.get("url", ""),
            "source_platform": source_platform,
            "content_type": content_type,
        }

        if overrides.get("targets"):
            request["targets"] = overrides["targets"]
        elif content_type == "website":
            targets = []
            promo_goals = {
                "tiktok": "ad_creative", "youtube": "promo_reel",
                "instagram": "promo_clip", "reels": "promo_reel",
                "shorts": "ad_creative", "facebook": "promo_clip",
                "twitter": "promo_clip", "linkedin": "promo_clip",
                "threads": "promo_clip", "snapchat": "ad_creative",
                "pinterest": "promo_reel", "email_campaign": "email_hero",
                "billboard_digital": "billboard_ad", "website_embed": "promo_reel",
            }
            for p in default_platforms:
                if p in PLATFORM_DEFAULTS:
                    defaults = PLATFORM_DEFAULTS[p]
                    goal = promo_goals.get(p, "ad_creative")
                    targets.append({
                        "platform": p,
                        "goal": goal,
                        "duration_sec": min(defaults["duration"], 30.0),
                        "aspect_ratio": defaults["aspect"],
                    })
            if targets:
                request["targets"] = targets

        if overrides.get("lyrics"):
            request["lyrics"] = overrides["lyrics"]

        if overrides.get("brand_notes"):
            request["brand_notes"] = overrides["brand_notes"]
        elif content_type == "website":
            desc = metadata.get("description", "")
            if desc:
                request["brand_notes"] = desc[:300]

        if overrides.get("campaign_notes"):
            request["campaign_notes"] = overrides["campaign_notes"]

        return request
