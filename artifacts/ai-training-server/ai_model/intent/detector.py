"""Multi-layer intent detector.

Converts a free-text ``description`` and/or a ``url`` into structured
:class:`IntentSignals` with per-signal confidence scores.  The signals feed
directly into the generation awareness cascade as ``[INTENT]`` lines that
outrank ``[HIGH]`` chart and ``TRENDS:`` lines, so user intent always gets
the highest priority.

Detection runs four layers in sequence:

1. **Lexical** — keyword dictionaries (genre, mood, visual, platform, goal,
   BPM, key) with synonym expansion and weighted scoring.
2. **Structural** — negation detection ("not too upbeat", "avoid saturation"),
   adjective extraction, and reference artist/track patterns.
3. **Cross-signal inference** — reconciles contradictions (e.g. genre=trap
   but energy=low → keep both, don't override) and lifts confidence when
   signals corroborate.
4. **URL enrichment** — platform/goal hints from the hostname fold in before
   lexical scoring; body text is treated as additional description text.

All operations are never-raise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Canonical genre table  ──────────────────────────────────────────────────
# alias (lower, no punctuation) → canonical slug
# ---------------------------------------------------------------------------

_GENRE_ALIASES: Dict[str, str] = {
    # Hip-hop / rap family
    "hip hop": "hip_hop",     "hiphop": "hip_hop",     "hip-hop": "hip_hop",
    "rap": "hip_hop",         "boom bap": "hip_hop",   "boom-bap": "hip_hop",
    "trap": "trap",           "trap music": "trap",    "drill": "drill",
    "uk drill": "drill",      "brooklyn drill": "drill",
    "melodic rap": "melodic_rap",  "melodic trap": "melodic_rap",
    "mumble rap": "hip_hop",
    "cloud rap": "hip_hop",   "underground rap": "hip_hop",
    "conscious rap": "hip_hop",
    # R&B / soul
    "rnb": "rnb",             "r&b": "rnb",            "r and b": "rnb",
    "rhythm and blues": "rnb", "soul": "soul",
    "neo soul": "neo_soul",   "neo-soul": "neo_soul",
    "funk": "funk",
    # Pop
    "pop": "pop",             "pop music": "pop",      "synth pop": "synth_pop",
    "synth-pop": "synth_pop", "dream pop": "dream_pop","indie pop": "indie_pop",
    "electropop": "electropop",
    # Electronic
    "electronic": "electronic", "edm": "edm",
    "house": "house",         "deep house": "deep_house",
    "tech house": "tech_house", "techno": "techno",
    "trance": "trance",       "drum and bass": "dnb", "dnb": "dnb",
    "dubstep": "dubstep",     "future bass": "future_bass",
    "ambient": "ambient",     "downtempo": "downtempo",
    "chillwave": "chillwave", "synthwave": "synthwave",
    "retrowave": "synthwave", "vaporwave": "vaporwave",
    "phonk": "phonk",         "afro house": "afro_house",
    "amapiano": "amapiano",
    # Lo-fi
    "lofi": "lofi",           "lo fi": "lofi",         "lo-fi": "lofi",
    "lofi hip hop": "lofi",   "lo-fi hip hop": "lofi",
    # Rock / alternative
    "rock": "rock",           "indie rock": "indie_rock", "indie": "indie_rock",
    "alternative": "alternative", "alt rock": "alternative",
    "punk": "punk",           "emo": "emo",             "pop punk": "pop_punk",
    "grunge": "grunge",       "metal": "metal",         "heavy metal": "metal",
    # Country / folk / Americana
    "country": "country",     "folk": "folk",           "americana": "americana",
    "bluegrass": "bluegrass",
    # Latin / Afrobeats / global
    "latin": "latin",         "reggaeton": "reggaeton", "salsa": "latin",
    "cumbia": "latin",        "bachata": "latin",       "bossa nova": "latin",
    "afrobeats": "afrobeats", "afropop": "afrobeats",   "afro pop": "afrobeats",
    "dancehall": "dancehall", "reggae": "reggae",
    # Jazz / blues / gospel
    "jazz": "jazz",           "blues": "blues",         "gospel": "gospel",
    "classical": "classical", "orchestral": "classical",
}

# ---------------------------------------------------------------------------
# Mood table
# ---------------------------------------------------------------------------

_MOOD_ALIASES: Dict[str, str] = {
    "energetic": "energetic",   "hype": "energetic",      "hyped": "energetic",
    "lit": "energetic",         "turnt": "energetic",
    "aggressive": "aggressive", "intense": "aggressive",  "hard": "aggressive",
    "angry": "aggressive",      "gritty": "gritty",       "raw": "gritty",
    "street": "gritty",
    "dark": "dark",             "moody": "dark",           "ominous": "dark",
    "sinister": "dark",         "mysterious": "mysterious",
    "melancholic": "melancholic", "sad": "melancholic",   "heartbroken": "melancholic",
    "heartbreak": "melancholic", "sorrowful": "melancholic",
    "nostalgic": "nostalgic",   "throwback": "nostalgic",
    "romantic": "romantic",     "love": "romantic",       "sensual": "romantic",
    "sexy": "romantic",
    "uplifting": "uplifting",   "inspirational": "uplifting", "triumphant": "uplifting",
    "empowering": "uplifting",  "motivational": "uplifting",
    "happy": "happy",           "joyful": "happy",         "fun": "happy",
    "playful": "happy",         "bubbly": "happy",
    "chill": "chill",           "relaxed": "chill",        "mellow": "chill",
    "laid back": "chill",       "laid-back": "chill",      "easy": "chill",
    "calm": "chill",            "soothing": "chill",
    "dreamy": "dreamy",         "ethereal": "dreamy",      "surreal": "dreamy",
    "atmospheric": "dreamy",    "hypnotic": "dreamy",      "trance-like": "dreamy",
    "smooth": "smooth",         "sophisticated": "smooth", "sultry": "smooth",
    "luxurious": "smooth",
}

# ---------------------------------------------------------------------------
# Energy lexicon  (word → raw score; positive = high energy, negative = low)
# ---------------------------------------------------------------------------

_ENERGY_SCORES: Dict[str, float] = {
    # High energy
    "banger": 0.9,     "bangers": 0.9,    "fire": 0.85,    "hard": 0.8,
    "heat": 0.75,      "lit": 0.8,        "turnt": 0.85,   "hype": 0.82,
    "intense": 0.78,   "aggressive": 0.82, "powerful": 0.75, "explosive": 0.88,
    "anthemic": 0.75,  "energetic": 0.8,  "rage": 0.85,    "rowdy": 0.78,
    "loud": 0.72,      "heavy": 0.78,     "fast": 0.65,    "rapid": 0.65,
    "uptempo": 0.7,    "up-tempo": 0.7,   "high energy": 0.85,
    "high-energy": 0.85, "epic": 0.75,   "stadium": 0.72,  "club": 0.7,
    "dance": 0.65,     "party": 0.7,      "bounce": 0.68,  "hitter": 0.8,
    # Mid energy
    "groovy": 0.55,    "funky": 0.58,     "bouncy": 0.6,   "mid tempo": 0.5,
    "mid-tempo": 0.5,
    # Low energy
    "chill": -0.6,     "mellow": -0.6,    "soft": -0.5,    "calm": -0.55,
    "slow": -0.5,      "gentle": -0.55,   "tender": -0.5,  "quiet": -0.55,
    "peaceful": -0.6,  "relaxed": -0.55,  "laid back": -0.6, "easy": -0.45,
    "dreamy": -0.45,   "ethereal": -0.55, "atmospheric": -0.4, "ambient": -0.6,
    "lofi": -0.55,     "lo-fi": -0.55,    "intimate": -0.5, "acoustic": -0.4,
    "minimal": -0.4,   "minimalist": -0.4, "dark": 0.2,    "moody": 0.15,
}

# ---------------------------------------------------------------------------
# Tone resolution
# ---------------------------------------------------------------------------

_TONE_ALIASES: Dict[str, str] = {
    "aggressive": "aggressive", "intense": "aggressive",  "hard": "aggressive",
    "dark": "dark",             "moody": "dark",
    "playful": "playful",       "fun": "playful",          "lighthearted": "playful",
    "sincere": "sincere",       "heartfelt": "sincere",    "authentic": "sincere",
    "raw": "raw",               "honest": "raw",           "vulnerable": "raw",
    "sophisticated": "sophisticated", "polished": "sophisticated",
    "luxury": "sophisticated",  "luxurious": "sophisticated",
    "inspirational": "inspirational", "motivational": "inspirational",
    "uplifting": "uplifting",   "positive": "uplifting",
    "emotional": "emotional",   "sentimental": "emotional",
    "energetic": "energetic",   "hyped": "energetic",
}

# ---------------------------------------------------------------------------
# Visual / production signals
# ---------------------------------------------------------------------------

_LIGHTING_ALIASES: Dict[str, str] = {
    "cinematic": "cinematic",       "film": "cinematic",         "filmic": "cinematic",
    "movie": "cinematic",           "dramatic": "dramatic",      "high contrast": "dramatic",
    "high-contrast": "dramatic",    "natural": "natural",        "daylight": "natural",
    "outdoor": "natural",           "sunlight": "natural",       "studio": "studio",
    "professional": "studio",       "clean light": "studio",     "clean lighting": "studio",
    "golden hour": "golden_hour",   "golden_hour": "golden_hour","golden light": "golden_hour",
    "sunset": "golden_hour",        "sunrise": "golden_hour",    "warm light": "golden_hour",
    "night": "night",               "dark room": "night",        "dark background": "night",
    "low key": "night",             "low-key": "night",
    "neon": "neon",                 "neon light": "neon",        "neon-lit": "neon",
    "neon lights": "neon",          "led": "neon",               "rgb": "neon",
    "cyberpunk": "neon",
    "vintage": "vintage",           "retro lighting": "vintage", "film grain": "vintage",
    "moody": "dramatic",
}

_CAMERA_ALIASES: Dict[str, str] = {
    "zoom in": "zoom_in",      "zooming in": "zoom_in",    "push in": "zoom_in",
    "zoom out": "zoom_out",    "zooming out": "zoom_out",  "pull out": "zoom_out",
    "pull back": "zoom_out",
    "pan left": "pan_left",    "panning left": "pan_left",
    "pan right": "pan_right",  "panning right": "pan_right",
    "tilt up": "tilt_up",      "tilting up": "tilt_up",
    "tilt down": "tilt_down",  "tilting down": "tilt_down",
    "dolly in": "dolly_in",    "dolly out": "dolly_out",
    "crane up": "crane_up",    "crane shot": "crane_up",
    "crane down": "crane_down",
    "static": "static",        "locked off": "static",    "fixed": "static",
    "stationary": "static",    "handheld": "static",
}

_COLOR_TEMP_ALIASES: Dict[str, str] = {
    "warm": "warm",          "golden": "warm",       "amber": "warm",
    "orange": "warm",        "cozy": "warm",         "earthy": "warm",
    "sepia": "warm",
    "cool": "cool",          "cold": "cool",         "blue": "cool",
    "teal": "cool",          "icy": "cool",          "crisp": "cool",
    "winter": "cool",        "arctic": "cool",
    "neutral": "neutral",    "balanced": "neutral",  "white": "neutral",
    "clean": "neutral",
}

_VISUAL_STYLES: List[Tuple[str, str]] = [
    # (trigger phrase, style slug)
    ("cinematic",     "cinematic"),
    ("minimalist",    "minimalist"),
    ("minimal",       "minimalist"),
    ("aesthetic",     "aesthetic"),
    ("editorial",     "editorial"),
    ("documentary",   "documentary"),
    ("raw footage",   "raw"),
    ("street",        "street"),
    ("urban",         "urban"),
    ("grunge",        "grunge"),
    ("retro",         "retro"),
    ("vintage",       "vintage"),
    ("nostalgic",     "nostalgic"),
    ("futuristic",    "futuristic"),
    ("sci-fi",        "futuristic"),
    ("scifi",         "futuristic"),
    ("cyberpunk",     "cyberpunk"),
    ("luxury",        "luxury"),
    ("luxurious",     "luxury"),
    ("high end",      "luxury"),
    ("high-end",      "luxury"),
    ("abstract",      "abstract"),
    ("surreal",       "surreal"),
    ("dreamy",        "dreamy"),
    ("ethereal",      "ethereal"),
    ("lo fi",         "lofi"),
    ("lofi",          "lofi"),
    ("lo-fi",         "lofi"),
    ("animated",      "animated"),
    ("animation",     "animated"),
    ("motion graphics", "motion_graphics"),
    ("split screen",  "split_screen"),
    ("slow motion",   "slow_motion"),
    ("slow mo",       "slow_motion"),
    ("slo-mo",        "slow_motion"),
    ("time lapse",    "time_lapse"),
    ("timelapse",     "time_lapse"),
]

# ---------------------------------------------------------------------------
# Platform detection from text
# ---------------------------------------------------------------------------

_PLATFORM_TEXT: Dict[str, str] = {
    "tiktok": "tiktok",      "tik tok": "tiktok",     "fyp": "tiktok",
    "for you page": "tiktok", "for you": "tiktok",
    "instagram": "instagram", "ig": "instagram",       "insta": "instagram",
    "reels": "instagram",     "ig reels": "instagram", "instagram reels": "instagram",
    "story": "instagram",
    "youtube": "youtube",     "yt": "youtube",         "shorts": "youtube",
    "youtube shorts": "youtube",
    "twitter": "twitter",     "x.com": "twitter",      "tweet": "twitter",
    "linkedin": "linkedin",   "spotify": "spotify",
    "soundcloud": "soundcloud",
}

# ---------------------------------------------------------------------------
# Goal detection from text
# ---------------------------------------------------------------------------

_GOAL_TEXT: Dict[str, str] = {
    "stream":        "drive_streams",  "streaming":    "drive_streams",
    "streams":       "drive_streams",  "listen":       "drive_streams",
    "plays":         "drive_streams",  "playcount":    "drive_streams",
    "ticket":        "drive_conversion", "tickets":    "drive_conversion",
    "merch":         "drive_conversion", "merchandise":"drive_conversion",
    "buy":           "drive_conversion", "purchase":   "drive_conversion",
    "preorder":      "drive_conversion", "pre-order":  "drive_conversion",
    "presale":       "drive_conversion", "sale":       "drive_conversion",
    "shop":          "drive_conversion", "store":      "drive_conversion",
    "follow":        "grow_followers",   "followers":  "grow_followers",
    "subscribe":     "grow_followers",   "subscribers":"grow_followers",
    "grow":          "grow_followers",
    "engagement":    "drive_engagement", "comments":   "drive_engagement",
    "comment":       "drive_engagement", "shares":     "drive_engagement",
    "share":         "drive_engagement", "likes":      "drive_engagement",
    "viral":         "drive_engagement", "trending":   "build_awareness",
    "announce":      "build_awareness",  "launch":     "build_awareness",
    "reveal":        "build_awareness",  "introduce":  "build_awareness",
    "awareness":     "build_awareness",  "discover":   "build_awareness",
}

# ---------------------------------------------------------------------------
# BPM patterns
# ---------------------------------------------------------------------------

_BPM_EXACT      = re.compile(r"\b(\d{2,3})\s*(?:bpm|beats?\s*per\s*minute)\b", re.IGNORECASE)
_BPM_ROUGH: Dict[str, float] = {
    "very slow": 60.0, "super slow": 60.0, "downtempo": 75.0,
    "slow tempo": 75.0, "slow": 80.0,
    "mid tempo": 105.0, "mid-tempo": 105.0, "midtempo": 105.0, "medium tempo": 105.0,
    "fast tempo": 140.0, "fast": 140.0, "rapid": 145.0,
    "very fast": 155.0, "super fast": 160.0, "hyperspeed": 170.0,
}

# ---------------------------------------------------------------------------
# Musical key patterns
# ---------------------------------------------------------------------------

_KEY_RE = re.compile(
    r"\bin\s+(?:the\s+key\s+of\s+)?([A-Ga-g][b#]?)\s*(major|minor|maj|min|m)?\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Reference patterns
# ---------------------------------------------------------------------------

_REF_RE = re.compile(
    r"\b(?:like|inspired\s+by|in\s+the\s+style\s+of|à\s+la|ala|similar\s+to|sounds?\s+like)\s+"
    r"([A-Z][a-zA-Z\s'&]{2,30}?)(?=\s*[,\.;\!?]|\s+but\b|\s+with\b|$)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Negation context  ──────────────────────────────────────────────────────
# Catch "not too X", "avoid X", "without X", "no X", "less X"
# ---------------------------------------------------------------------------

_NEG_RE = re.compile(
    r"\b(?:not?\s+too|avoid|without|no\s+|less\s+|don'?t\s+want|shouldn'?t\s+be|nothing\s+too)"
    r"\s+([a-z][a-z\s-]{1,30}?)(?=[,\.;\!?\s]|$)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Helper: normalise text
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s'&/\-]")


def _norm(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return _PUNCT.sub(" ", text.lower()).strip()


def _phrase_in(phrase: str, blob: str) -> bool:
    """True when *phrase* occurs in *blob* at a real word boundary.

    Multi-word phrases are matched as substrings (they are naturally specific).
    Single-word phrases require that the boundary character is NOT a hyphen
    or alphanumeric character — this prevents "rap" matching inside "trap",
    "lit" matching inside "neon-lit", "rnb" matching inside "turning", etc.
    """
    if " " in phrase:
        return phrase in blob
    # Boundary: the char before must not be [a-z0-9-], and the char after must not be [a-z0-9-].
    # This is stricter than \b (which treats hyphens as word boundaries).
    return bool(re.search(
        r"(?:(?<=[^a-z0-9\-])|^)" + re.escape(phrase) + r"(?=[^a-z0-9\-]|$)",
        blob,
    ))


def _multi_match(blob: str, table: Dict[str, str]) -> Optional[Tuple[str, float]]:
    """Find the highest-scoring entry from *table* in *blob*.

    Scoring:
    - Each matching phrase contributes ``len(phrase.split())`` points to its
      canonical so multi-word aliases beat shorter synonyms.
    - Ties are broken by **earliest first appearance** in the blob so the
      first adjective the caller wrote wins ("dark trap" → mood=dark, not
      mood=aggressive even though both match).
    - Uses ``_phrase_in`` (word-boundary, no hyphen leakage) for matching.
    """
    hits: Dict[str, int] = {}
    first_pos: Dict[str, int] = {}   # canonical → earliest match start offset

    for phrase, canonical in table.items():
        if " " in phrase:
            idx = blob.find(phrase)
            if idx < 0:
                continue
            pos = idx
        else:
            m = re.search(
                r"(?:(?<=[^a-z0-9\-])|^)" + re.escape(phrase) + r"(?=[^a-z0-9\-]|$)",
                blob,
            )
            if not m:
                continue
            pos = m.start()

        score = len(phrase.split())
        hits[canonical] = hits.get(canonical, 0) + score
        if canonical not in first_pos or pos < first_pos[canonical]:
            first_pos[canonical] = pos

    if not hits:
        return None

    # Primary: highest total score; secondary: earliest position in blob.
    max_score = max(hits.values())
    tied = [c for c in hits if hits[c] == max_score]
    best_canonical = (
        tied[0] if len(tied) == 1
        else min(tied, key=lambda c: first_pos.get(c, len(blob)))
    )

    top   = hits[best_canonical]
    total = sum(hits.values()) or 1
    confidence = min(1.0, 0.5 + top / (total + 2.0))
    return best_canonical, confidence


def _detect_energy(blob: str) -> Optional[Tuple[float, float]]:
    """Return (energy_0_to_1, confidence) or None."""
    total_weight = 0.0
    total_score  = 0.0
    for word, score in _ENERGY_SCORES.items():
        if _phrase_in(word, blob):
            # multi-word matches get more weight
            w = 1 + word.count(" ") * 0.5
            total_weight += w
            total_score  += score * w
    if total_weight < 0.5:
        return None
    raw = total_score / total_weight          # -1 to 1
    energy = max(0.05, min(0.98, (raw + 1.0) / 2.0))   # map to 0–1
    confidence = min(0.95, 0.40 + total_weight * 0.07)
    return energy, confidence


def _detect_bpm(blob: str) -> Optional[float]:
    """Return explicit BPM value or rough estimate from tempo language."""
    m = _BPM_EXACT.search(blob)
    if m:
        v = float(m.group(1))
        if 40 <= v <= 250:
            return v
    for phrase, bpm in sorted(_BPM_ROUGH.items(), key=lambda x: -len(x[0])):
        if phrase in blob:
            return bpm
    return None


def _detect_key(blob: str) -> Optional[str]:
    m = _KEY_RE.search(blob)
    if m:
        note    = m.group(1).capitalize()
        quality = (m.group(2) or "major").lower()
        quality = "minor" if quality in ("minor", "min", "m") else "major"
        return f"{note} {quality}"
    return None


def _detect_negations(blob: str) -> List[str]:
    """Extract things the user wants to avoid."""
    results: List[str] = []
    for m in _NEG_RE.finditer(blob):
        phrase = m.group(1).strip()
        if 2 < len(phrase) < 50 and not any(phrase == r for r in results):
            results.append(phrase)
    return results[:6]


def _detect_reference(original: str) -> Optional[str]:
    """Detect artist/track references ('like Drake', 'inspired by Kendrick')."""
    m = _REF_RE.search(original)
    if m:
        ref = m.group(1).strip().rstrip(".,;")
        if 2 < len(ref) < 40:
            return ref
    return None


def _detect_platforms(blob: str) -> List[str]:
    seen: List[str] = []
    for phrase, plat in sorted(_PLATFORM_TEXT.items(), key=lambda x: -len(x[0])):
        if _phrase_in(phrase, blob) and plat not in seen:
            seen.append(plat)
    return seen[:3]


def _detect_goal(blob: str) -> Optional[Tuple[str, float]]:
    hits: Dict[str, int] = {}
    for word, goal in _GOAL_TEXT.items():
        if _phrase_in(word, blob):
            hits[goal] = hits.get(goal, 0) + 1
    if not hits:
        return None
    best = max(hits, key=lambda g: hits[g])
    total = sum(hits.values()) or 1
    confidence = min(0.9, 0.4 + hits[best] / (total + 1.0))
    return best, confidence


def _detect_visual_styles(blob: str) -> List[str]:
    seen: List[str] = []
    for phrase, slug in sorted(_VISUAL_STYLES, key=lambda x: -len(x[0])):
        if phrase in blob and slug not in seen:
            seen.append(slug)
    return seen[:5]


def _extract_topics(text: str, stop: set) -> List[str]:
    """Frequency-ranked salient keywords, stopwords removed."""
    tokens = re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
    freq: Dict[str, int] = {}
    first_pos: Dict[str, int] = {}
    for idx, tok in enumerate(tokens):
        if len(tok) < 3 or tok in stop:
            continue
        if tok not in first_pos:
            first_pos[tok] = idx
        freq[tok] = freq.get(tok, 0) + 1
    ranked = sorted(freq.keys(), key=lambda t: (-freq[t], first_pos[t]))
    return ranked[:8]


_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "had", "her", "was", "one", "our", "out", "day", "get", "has", "him",
    "his", "how", "man", "new", "now", "old", "see", "two", "way", "who",
    "boy", "did", "its", "let", "put", "say", "she", "too", "use", "with",
    "that", "this", "from", "have", "been", "they", "what", "when", "will",
    "your", "each", "does", "into", "like", "more", "over", "some", "than",
    "them", "then", "time", "very", "want", "well", "were", "also", "just",
    "make", "much", "only", "same", "such", "take", "than", "their", "here",
    "both", "feel", "kind", "song", "track", "music", "sound", "sounds",
    "going", "video", "about", "going", "create", "creating", "please",
    "generate", "make", "want", "need", "should", "would", "could", "really",
    "style", "vibe", "look", "feels",
}

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class IntentSignals:
    """Structured intent extracted from a description + URL combination."""

    # Core creative
    genre:             Optional[str]   = None    # canonical slug e.g. "trap", "rnb"
    genre_confidence:  float           = 0.0
    mood:              Optional[str]   = None    # canonical slug e.g. "dark", "uplifting"
    mood_confidence:   float           = 0.0
    energy:            Optional[float] = None    # 0.0–1.0
    energy_confidence: float           = 0.0
    tone:              Optional[str]   = None    # aggressive / playful / raw / …
    tempo_bpm:         Optional[float] = None
    key:               Optional[str]   = None    # e.g. "C# minor"

    # Visual / production
    lighting:          Optional[str]   = None
    camera_motion:     Optional[str]   = None
    color_temperature: Optional[str]   = None
    visual_styles:     List[str]       = field(default_factory=list)

    # Platform / distribution
    platform_hints:    List[str]       = field(default_factory=list)

    # Goal
    goal:              Optional[str]   = None
    goal_confidence:   float           = 0.0

    # Content
    topics:            List[str]       = field(default_factory=list)
    keywords:          List[str]       = field(default_factory=list)
    negative_signals:  List[str]       = field(default_factory=list)

    # Reference
    reference_artist:  Optional[str]   = None

    # Meta
    confidence:        float           = 0.0     # aggregate signal strength
    source:            str             = "description"   # description | url | both
    raw_text:          str             = ""

    # -------------------------------------------------------------------
    def is_useful(self) -> bool:
        """True when at least one creative signal was detected."""
        return bool(
            self.genre or self.mood or self.energy is not None
            or self.topics or self.goal or self.lighting
            or self.camera_motion or self.color_temperature
        )

    # -------------------------------------------------------------------
    def to_awareness_lines(self) -> List[str]:
        """Serialise as ``[INTENT]`` tagged lines for the awareness parser.

        These lines are prepended to the full awareness string so they
        outrank ``[HIGH]`` chart signals and ``TRENDS:`` lines.
        """
        lines: List[str] = []

        # ── Core creative signal line ─────────────────────────────────
        core: List[str] = []
        if self.genre:
            core.append(f"genre={self.genre}")
        if self.mood:
            core.append(f"mood={self.mood}")
        if self.energy is not None:
            core.append(f"energy={self.energy:.2f}")
        if self.tone:
            core.append(f"tone={self.tone}")
        if self.tempo_bpm:
            core.append(f"bpm={self.tempo_bpm:.0f}")
        if self.key:
            core.append("key=" + self.key.replace(" ", "_"))
        if core:
            lines.append(
                "[INTENT] " + " ".join(core) + f" confidence={self.confidence:.2f}"
            )

        # ── Visual line ───────────────────────────────────────────────
        vis: List[str] = []
        if self.lighting:
            vis.append(f"lighting={self.lighting}")
        if self.camera_motion:
            vis.append(f"camera={self.camera_motion}")
        if self.color_temperature:
            vis.append(f"color_temp={self.color_temperature}")
        if self.visual_styles:
            vis.append("style=" + ",".join(self.visual_styles[:3]))
        if vis:
            lines.append("[INTENT] " + " ".join(vis))

        # ── Topics / keywords line ────────────────────────────────────
        all_topics = list(dict.fromkeys([*self.topics, *self.keywords]))[:6]
        if all_topics:
            lines.append("[INTENT] topics=" + ",".join(all_topics))

        # ── Negative signals ──────────────────────────────────────────
        if self.negative_signals:
            lines.append("[INTENT] avoid=" + ",".join(self.negative_signals[:5]))

        # ── Goal ─────────────────────────────────────────────────────
        if self.goal:
            lines.append(f"[INTENT] goal={self.goal}")

        # ── Reference artist ─────────────────────────────────────────
        if self.reference_artist:
            lines.append(f"[INTENT] reference_artist={self.reference_artist}")

        return lines

    # -------------------------------------------------------------------
    def to_build_brief_kwargs(self) -> dict:
        """Return a dict of override kwargs to pass to build_brief().

        Only includes keys where this detector has a meaningful, confident
        signal so it never blindly clobbers explicit caller values.
        """
        kw: dict = {}
        if self.genre and self.genre_confidence >= 0.45:
            kw["genre"] = self.genre.replace("_", " ")
        if self.mood and self.mood_confidence >= 0.4:
            kw["mood"] = self.mood
        if self.tempo_bpm:
            kw["bpm"] = self.tempo_bpm
        if self.key:
            kw["key"] = self.key
        if self.tone:
            kw["tone"] = self.tone
        if self.lighting:
            kw["lighting"] = self.lighting
        if self.camera_motion:
            kw["camera_motion"] = self.camera_motion
        if self.color_temperature:
            kw["color_temperature"] = self.color_temperature
        if self.negative_signals:
            kw["negative_prompt"] = ", ".join(self.negative_signals)
        if self.goal and self.goal_confidence >= 0.5:
            kw["goal"] = self.goal
        return kw


# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------

def detect_intent(
    description: str = "",
    url: str = "",
    *,
    url_content_text: str = "",   # pre-fetched URL body (skips HTTP call)
    url_platform_hint: str = "",  # pre-fetched platform hint from UrlContent
    url_goal_hint: str = "",      # pre-fetched goal hint from UrlContent
) -> "IntentSignals":
    """Detect intent from a free-text description and/or a URL.

    Args:
        description:       User's free-text description of what to generate.
        url:               URL to fetch and analyse (if not already fetched).
        url_content_text:  Pre-fetched URL body text (avoids duplicate HTTP
                           call when the caller already read the URL).
        url_platform_hint: Pre-fetched platform slug from :class:`UrlContent`.
        url_goal_hint:     Pre-fetched goal slug from :class:`UrlContent`.

    Returns:
        :class:`IntentSignals` — never raises.
    """
    try:
        return _detect_impl(
            description=description or "",
            url=url or "",
            url_content_text=url_content_text,
            url_platform_hint=url_platform_hint,
            url_goal_hint=url_goal_hint,
        )
    except Exception:
        return IntentSignals()


def _detect_impl(
    description: str,
    url: str,
    url_content_text: str,
    url_platform_hint: str,
    url_goal_hint: str,
) -> "IntentSignals":
    # ── 0. Collect raw text corpus ─────────────────────────────────────────
    source_parts: List[str] = []
    if description:
        source_parts.append("description")
    url_fetched = False

    # Optionally fetch URL content
    url_text = url_content_text
    if url and not url_text:
        try:
            from .url_reader import read_url
            uc = read_url(url)
            url_text = uc.combined()
            url_platform_hint = url_platform_hint or uc.platform_hint
            url_goal_hint     = url_goal_hint     or uc.goal_hint
            url_fetched = True
        except Exception:
            pass
    if url_text or url_platform_hint:
        source_parts.append("url")

    source = "+".join(source_parts) if len(source_parts) > 1 else (source_parts[0] if source_parts else "description")

    # Combined text for scanning (description gets higher weight by repetition)
    full_text = " ".join(p for p in [description, description, url_text] if p)
    raw_text  = full_text[:2000]
    blob      = _norm(full_text)[:2000]

    # ── 1. Lexical pass ───────────────────────────────────────────────────
    genre,  genre_conf  = _multi_match(blob, _GENRE_ALIASES)  or (None, 0.0)
    mood,   mood_conf   = _multi_match(blob, _MOOD_ALIASES)   or (None, 0.0)
    tone,   _           = _multi_match(blob, _TONE_ALIASES)   or (None, 0.0)
    lighting,    _      = _multi_match(blob, _LIGHTING_ALIASES) or (None, 0.0)
    camera_motion, _    = _multi_match(blob, _CAMERA_ALIASES) or (None, 0.0)
    color_temperature, _ = _multi_match(blob, _COLOR_TEMP_ALIASES) or (None, 0.0)
    visual_styles       = _detect_visual_styles(blob)
    platform_hints      = _detect_platforms(blob)
    goal, goal_conf     = _detect_goal(blob) or (None, 0.0)
    energy_result       = _detect_energy(blob)
    energy, energy_conf = energy_result if energy_result else (None, 0.0)
    bpm                 = _detect_bpm(blob)
    key                 = _detect_key(blob)

    # ── 2. Structural pass ────────────────────────────────────────────────
    negative_signals = _detect_negations(blob)
    reference_artist = _detect_reference(full_text)   # case-sensitive original
    topics = _extract_topics(full_text, _STOPWORDS)

    # ── 3. URL hint fold-in ───────────────────────────────────────────────
    if url_platform_hint and url_platform_hint not in platform_hints:
        platform_hints.insert(0, url_platform_hint)
    if url_goal_hint and not goal:
        goal      = url_goal_hint
        goal_conf = 0.55     # weaker: derived from hostname, not text

    # ── 4. Cross-signal inference ─────────────────────────────────────────
    # Energy × genre reconciliation — trust explicit energy over genre default
    # but flag inconsistencies with lower confidence rather than forcing.
    if genre in ("lofi", "ambient", "downtempo") and energy is not None and energy > 0.7:
        energy_conf *= 0.6   # lofi + high energy → weak signal, keep both

    # Mood → tone bridge when tone not explicitly detected
    if not tone and mood:
        tone = {
            "energetic": "energetic", "aggressive": "aggressive",
            "dark": "dark",           "gritty": "raw",
            "melancholic": "emotional", "nostalgic": "sincere",
            "romantic": "sincere",    "uplifting": "uplifting",
            "happy": "playful",       "chill": "relaxed",
            "dreamy": "sophisticated", "smooth": "sophisticated",
        }.get(mood)

    # Color temperature from mood/genre when not explicit
    if not color_temperature:
        if mood in ("dark", "aggressive", "gritty"):
            color_temperature = "cool"
        elif genre in ("lofi", "soul", "neo_soul", "gospel") or mood in ("nostalgic", "romantic"):
            color_temperature = "warm"
        elif genre in ("electronic", "techno", "dnb", "dubstep"):
            color_temperature = "cool"

    # Lighting from genre/mood when not explicit
    if not lighting:
        if genre in ("trap", "drill", "phonk") or mood == "aggressive":
            lighting = "dramatic"
        elif genre in ("synthwave", "phonk") or "neon" in blob or "cyberpunk" in blob:
            lighting = "neon"
        elif mood in ("dreamy", "ethereal") or genre in ("lofi", "ambient", "chillwave"):
            lighting = "natural"
        elif genre in ("pop", "edm") and mood not in ("dark",):
            lighting = "studio"

    # ── 5. Aggregate confidence ───────────────────────────────────────────
    confident_signals = sum(1 for v in [
        genre, mood, energy, tone, bpm, key,
        lighting, camera_motion, color_temperature,
        bool(visual_styles), bool(platform_hints), goal,
    ] if v)
    confidence = min(0.98, 0.30 + confident_signals * 0.065)

    # Promote confidence when description is long and specific
    if len(description) > 100:
        confidence = min(0.98, confidence + 0.08)
    if len(description) > 250:
        confidence = min(0.98, confidence + 0.05)

    # ── 6. Keywords (topics minus known genre/mood noise) ─────────────────
    genre_words = set((genre or "").replace("_", " ").split())
    mood_words  = set((mood  or "").replace("_", " ").split())
    keywords = [
        t for t in topics
        if t not in genre_words and t not in mood_words
        and t not in _STOPWORDS
    ][:6]

    return IntentSignals(
        genre              = genre,
        genre_confidence   = genre_conf,
        mood               = mood,
        mood_confidence    = mood_conf,
        energy             = energy,
        energy_confidence  = energy_conf,
        tone               = tone,
        tempo_bpm          = bpm,
        key                = key,
        lighting           = lighting,
        camera_motion      = camera_motion,
        color_temperature  = color_temperature,
        visual_styles      = visual_styles,
        platform_hints     = platform_hints,
        goal               = goal,
        goal_confidence    = goal_conf,
        topics             = topics[:6],
        keywords           = keywords,
        negative_signals   = negative_signals,
        reference_artist   = reference_artist,
        confidence         = confidence,
        source             = source,
        raw_text           = raw_text,
    )
