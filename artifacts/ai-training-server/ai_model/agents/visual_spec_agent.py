from __future__ import annotations
import re
from dataclasses import dataclass
from ..model.creative_model import CreativeModel


@dataclass
class VisualSpecRequest:
    idea: str
    platform: str
    tone: str
    awareness: str = ""


@dataclass
class VisualSpecResponse:
    thumbnail_prompt: str
    color_scheme: str
    layout: str


PLATFORM_LAYOUTS = {
    "tiktok": "vertical_9_16",
    "instagram": "square_1_1",
    "youtube": "landscape_16_9",
    "facebook": "landscape_16_9",
    "twitter": "landscape_2_1",
    "linkedin": "landscape_1200x627",
    "google_business": "landscape_16_9",
    "threads": "square_1_1",
}

TONE_COLORS = {
    "edgy": "dark_neon",
    "playful": "vibrant_pastel",
    "serious": "monochrome",
    "energetic": "high_contrast",
    "professional": "corporate_blue",
    "casual": "warm_earth",
    "promotional": "bold_red_gold",
}

GENRE_COLORS = {
    "phonk": "dark_neon",
    "drill": "dark_neon",
    "trap": "dark_neon",
    "gothic": "dark_neon",
    "dark": "dark_neon",
    "afrobeats": "vibrant_pastel",
    "latin": "vibrant_pastel",
    "tropical": "vibrant_pastel",
    "reggaeton": "vibrant_pastel",
    "hyperpop": "vibrant_pastel",
    "pop": "vibrant_pastel",
    "euphoric": "vibrant_pastel",
    "rnb": "warm_earth",
    "soul": "warm_earth",
    "neo soul": "warm_earth",
    "warm": "warm_earth",
    "intimate": "warm_earth",
    "indie": "monochrome",
    "alternative": "monochrome",
    "melancholic": "monochrome",
    "cinematic": "monochrome",
    "jazz": "monochrome",
    "summer": "vibrant_pastel",
    "vibrant": "vibrant_pastel",
    "bright": "vibrant_pastel",
}


def _extract_genre_mood(awareness: str) -> list[str]:
    """Pull genre and mood keywords from the awareness context string."""
    if not awareness:
        return []
    tokens: list[str] = []
    genre_m = re.search(r"genre[s]?[:\s]+([a-zA-Z ,/&]+)", awareness, re.IGNORECASE)
    mood_m = re.search(r"mood[s]?[:\s]+([a-zA-Z ,/&]+)", awareness, re.IGNORECASE)
    if genre_m:
        tokens += [t.strip().lower() for t in re.split(r"[,/&]", genre_m.group(1)) if t.strip()]
    if mood_m:
        tokens += [t.strip().lower() for t in re.split(r"[,/&]", mood_m.group(1)) if t.strip()]
    return tokens[:6]


def _pick_color_from_awareness(awareness: str, tone: str) -> str:
    """Pick the most contextually accurate color scheme from live awareness signals."""
    if awareness:
        text_lower = awareness.lower()
        for keyword, scheme in GENRE_COLORS.items():
            if keyword in text_lower:
                return scheme
    return TONE_COLORS.get(tone.lower(), "high_contrast")


def _build_thumbnail_prompt(req: VisualSpecRequest) -> str:
    """Build an awareness-enriched thumbnail prompt when the model doesn't produce one."""
    if not req.awareness:
        return f"Eye-catching {req.tone} thumbnail for: {req.idea}"
    tokens = _extract_genre_mood(req.awareness)
    style_desc = ", ".join(tokens[:3]) if tokens else req.tone
    signals = [
        line.strip() for line in req.awareness.splitlines()
        if re.match(r"\[(HIGH|MEDIUM)\]", line.strip())
    ]
    if signals:
        signal_hint = signals[0].split("]", 1)[-1].strip().split(":")[0].strip()[:60]
        return f"{req.tone.capitalize()} {style_desc} visual for: {req.idea} — {signal_hint}"
    return f"{req.tone.capitalize()} {style_desc} thumbnail for: {req.idea}"


def _build_thumbnail_prompt_alt(req: VisualSpecRequest) -> str:
    """A second, differently-phrased deterministic variant.

    Exists purely so the ranking step below always has more than one
    non-model candidate to choose between — mirrors the text pipeline's
    pattern of ranking a model candidate against multiple deterministic
    variants rather than a single fallback.
    """
    tokens = _extract_genre_mood(req.awareness) if req.awareness else []
    descriptor = tokens[0] if tokens else req.tone
    signals = [
        line.strip() for line in req.awareness.splitlines()
        if re.match(r"\[(HIGH|MEDIUM)\]", line.strip())
    ] if req.awareness else []
    if signals:
        hint = signals[0].split("]", 1)[-1].strip().split(":")[0].strip()[:60]
        return f"Cinematic {descriptor} cover art for {req.idea}, capturing {hint.lower()}"
    return f"Bold {descriptor} cover art for {req.idea}, high-contrast and share-ready"


def _score_visual_prompt(prompt: str, awareness: str = "") -> float:
    """Brief-aware quality score (0-100) for an image thumbnail prompt.

    Rewards: sensible length, alignment with live awareness genre/mood
    signals, and concrete visual descriptors. Penalises the bare generic
    fallback phrasing so an awareness-enriched or model candidate wins
    whenever one is available and coherent.
    """
    text = (prompt or "").strip()
    if not text:
        return 0.0
    words = text.split()
    n = len(words)
    length_score = 1.0 if 6 <= n <= 24 else max(0.0, 1.0 - abs(n - 15) / 20.0)

    low = text.lower()
    awareness_tokens = _extract_genre_mood(awareness) if awareness else []
    if awareness_tokens:
        hits = sum(1 for t in awareness_tokens if t in low)
        awareness_score = min(1.0, hits / max(1, len(awareness_tokens)))
    else:
        awareness_score = 0.5  # neutral when there's no live signal to match

    generic_penalty = 0.3 if low.startswith("eye-catching") and "thumbnail for" in low else 0.0

    descriptive_score = 1.0 if any(w in low for w in (
        "neon", "cinematic", "moody", "vibrant", "raw", "gritty", "glow",
        "dramatic", "warm", "cold", "high-contrast", "textured", "bold",
    )) else 0.3

    blended = (
        length_score * 0.25
        + awareness_score * 0.35
        + descriptive_score * 0.25
        - generic_penalty
    )
    return round(max(0.0, min(100.0, blended * 100)), 1)


class VisualSpecAgent:
    def __init__(self, model: CreativeModel):
        self.model = model

    def run(self, req: VisualSpecRequest) -> VisualSpecResponse:
        platform_token = f"<PLATFORM_{req.platform.upper()}>"
        tone_token = f"<TONE_{req.tone.upper()}>"

        awareness_prefix = ""
        if req.awareness:
            tokens = _extract_genre_mood(req.awareness)
            if tokens:
                awareness_prefix = f"Style: {', '.join(tokens[:3])}\n"
            high_signals = [
                line.strip() for line in req.awareness.splitlines()
                if re.match(r"\[HIGH\]", line.strip())
            ]
            if high_signals:
                hint = high_signals[0].split("]", 1)[-1].strip()[:80]
                awareness_prefix += f"Trend: {hint}\n"

        prompt = (
            f"{awareness_prefix}{platform_token} {tone_token} "
            f"Generate thumbnail spec for: {req.idea}\n"
        )

        candidates: list[str] = []
        try:
            output = self.model.generate(prompt)
            if self._is_meaningful(output):
                candidates.append(output)
        except Exception:
            pass

        # Two differently-phrased deterministic variants always compete
        # alongside the model output — mirrors the text pipeline's best-of-N
        # ranking pattern so a weak/undertrained model candidate can't win
        # by default. Guarantees a non-generic floor.
        candidates.append(_build_thumbnail_prompt(req))
        candidates.append(_build_thumbnail_prompt_alt(req))

        scored = {c: _score_visual_prompt(c, req.awareness) for c in candidates if c}
        thumbnail_prompt = max(scored, key=lambda c: scored[c]) if scored else (
            f"Eye-catching {req.tone} thumbnail for: {req.idea}"
        )

        platform_key = req.platform.lower().replace(" ", "_")
        color_scheme = _pick_color_from_awareness(req.awareness, req.tone)

        return VisualSpecResponse(
            thumbnail_prompt=thumbnail_prompt,
            color_scheme=color_scheme,
            layout=PLATFORM_LAYOUTS.get(platform_key, "square_1_1"),
        )

    def _is_meaningful(self, text: str) -> bool:
        if not text or len(text) < 10:
            return False
        control_count = sum(1 for w in text.split() if w.startswith("<") and w.endswith(">"))
        total = len(text.split())
        if total == 0:
            return False
        return control_count / total < 0.5
