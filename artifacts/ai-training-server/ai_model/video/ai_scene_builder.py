"""
ai_scene_builder.py
───────────────────
Generates SceneConfig objects directly from content DNA — no templates.

Every video gets a unique visual style computed from:
  • genre / tone  → energy, darkness, warmth, saturation, grain
  • idea text     → deterministic seed that shifts palette within the genre's
                    emotional space so two trap videos never look identical
  • scene type    → per-scene layout, font size, animation, timing

The system produces an infinite palette space rather than 23 fixed choices.
"""
from __future__ import annotations
import hashlib
import colorsys
from dataclasses import dataclass
from typing import List, Dict, Optional

from .scenes import SceneConfig, TextElement, FONT_PATH


# ── Visual DNA definitions ────────────────────────────────────────────────────

@dataclass
class VisualDNA:
    """Normalised (0–1) parameters that drive all visual decisions."""
    energy:      float = 0.7   # pace / intensity
    darkness:    float = 0.6   # background luminance (1 = very dark)
    warmth:      float = 0.5   # hue axis: 0 = cool blues, 1 = warm reds/golds
    saturation:  float = 0.75  # colour richness
    grain:       float = 0.0   # film-grain amount
    complexity:  float = 0.5   # number of simultaneous visual effects
    seed:        int   = 0     # idea-derived integer for palette micro-variation


# Genre → base DNA values
_GENRE_DNA: Dict[str, Dict] = {
    "trap":       {"energy": 0.90, "darkness": 0.88, "warmth": 0.25, "saturation": 0.80, "grain": 0.3},
    "drill":      {"energy": 0.85, "darkness": 0.92, "warmth": 0.20, "saturation": 0.45, "grain": 0.5},
    "hiphop":     {"energy": 0.80, "darkness": 0.75, "warmth": 0.35, "saturation": 0.70, "grain": 0.2},
    "hip_hop":    {"energy": 0.80, "darkness": 0.75, "warmth": 0.35, "saturation": 0.70, "grain": 0.2},
    "rnb":        {"energy": 0.50, "darkness": 0.60, "warmth": 0.75, "saturation": 0.60, "grain": 0.1},
    "r&b":        {"energy": 0.50, "darkness": 0.60, "warmth": 0.75, "saturation": 0.60, "grain": 0.1},
    "soul":       {"energy": 0.40, "darkness": 0.60, "warmth": 0.80, "saturation": 0.55, "grain": 0.4},
    "jazz":       {"energy": 0.30, "darkness": 0.70, "warmth": 0.65, "saturation": 0.40, "grain": 0.5},
    "pop":        {"energy": 0.70, "darkness": 0.30, "warmth": 0.55, "saturation": 0.90, "grain": 0.0},
    "afrobeats":  {"energy": 0.85, "darkness": 0.45, "warmth": 0.85, "saturation": 0.95, "grain": 0.0},
    "afro":       {"energy": 0.85, "darkness": 0.45, "warmth": 0.85, "saturation": 0.95, "grain": 0.0},
    "reggaeton":  {"energy": 0.85, "darkness": 0.40, "warmth": 0.80, "saturation": 0.90, "grain": 0.0},
    "latin":      {"energy": 0.80, "darkness": 0.40, "warmth": 0.82, "saturation": 0.88, "grain": 0.0},
    "lofi":       {"energy": 0.20, "darkness": 0.55, "warmth": 0.60, "saturation": 0.38, "grain": 0.7},
    "lo_fi":      {"energy": 0.20, "darkness": 0.55, "warmth": 0.60, "saturation": 0.38, "grain": 0.7},
    "indie":      {"energy": 0.38, "darkness": 0.42, "warmth": 0.55, "saturation": 0.48, "grain": 0.3},
    "acoustic":   {"energy": 0.28, "darkness": 0.32, "warmth": 0.62, "saturation": 0.40, "grain": 0.2},
    "electronic": {"energy": 0.92, "darkness": 0.82, "warmth": 0.15, "saturation": 0.88, "grain": 0.1},
    "hyperpop":   {"energy": 1.00, "darkness": 0.25, "warmth": 0.50, "saturation": 1.00, "grain": 0.0},
}

# Tone → additive adjustments applied on top of genre DNA
_TONE_DELTA: Dict[str, Dict] = {
    "energetic":     {"energy": +0.15, "darkness": -0.08, "saturation": +0.05},
    "hype":          {"energy": +0.20, "darkness": -0.12, "saturation": +0.08},
    "edgy":          {"energy": +0.08, "darkness": +0.15, "saturation": -0.05},
    "chill":         {"energy": -0.25, "darkness": -0.08, "saturation": -0.08},
    "professional":  {"energy": -0.10, "darkness": +0.10, "saturation": -0.15, "grain": +0.0},
    "playful":       {"energy": +0.10, "darkness": -0.18, "saturation": +0.12},
    "emotional":     {"energy": -0.08, "darkness": +0.08, "grain": +0.15},
    "promotional":   {"energy": +0.05, "saturation": +0.10},
    "inspirational": {"energy": -0.05, "darkness": -0.10, "warmth": +0.10},
    "serious":       {"energy": -0.15, "darkness": +0.12, "saturation": -0.10, "grain": +0.1},
    "casual":        {"energy": -0.05, "saturation": -0.05},
}

# Background type chosen by (energy, darkness) grid
def _choose_bg_type(energy: float, darkness: float, seed: int) -> str:
    _micro = (seed % 7) / 7.0   # 0-1 micro-variation per video
    if energy > 0.85 and darkness > 0.80:
        return "plasma"
    if energy > 0.85 and darkness <= 0.80:
        return "animated_gradient"
    if energy > 0.70 and darkness > 0.65:
        return ["animated_gradient", "wave"][seed % 2]
    if energy <= 0.35 and darkness <= 0.45:
        return "aurora"
    if energy <= 0.35 and darkness > 0.55:
        return "radial"
    if energy <= 0.55:
        return ["wave", "radial"][seed % 2]
    return ["animated_gradient", "wave"][seed % 2]


# ── Palette generation ────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _hsv_hex(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(_clamp(h), _clamp(s), _clamp(v))
    return f"0x{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _rgb_tuple_to_hex(r: int, g: int, b: int) -> str:
    return f"0x{r:02x}{g:02x}{b:02x}"


@dataclass
class Palette:
    bg1: str       # background colour 1
    bg2: str       # background colour 2 (gradient end)
    text_primary: str    # main text colour
    text_secondary: str  # secondary text colour (artist label etc)
    accent: str          # highlight / CTA colour
    color_grade: str     # cinematic grade tag


def derive_palette(dna: VisualDNA) -> Palette:
    """
    Convert visual DNA into concrete hex colours.

    Warmth maps hue:
      warmth=0.0 → pure blue (240°)  e.g. drill, electronic
      warmth=0.5 → purple/magenta
      warmth=1.0 → orange/red        e.g. afrobeats, reggaeton

    A seed-based micro-shift (±15°) ensures every video differs.
    """
    seed = dna.seed

    # Hue: 0.0–1.0 where 0.67 = blue, 0.0/1.0 = red, 0.17 = yellow
    base_hue = 0.67 - (dna.warmth * 0.55)   # 0.67 (blue) → 0.12 (orange)
    micro_shift = ((seed % 31) - 15) / 360.0
    h1 = (base_hue + micro_shift) % 1.0

    # Background colours: dark and slightly darker
    bg_value1 = 1.0 - dna.darkness * 0.88
    bg_value2 = bg_value1 * 0.65
    bg_sat = dna.saturation * 0.55

    bg1 = _hsv_hex(h1, bg_sat, bg_value1)
    bg2 = _hsv_hex((h1 + 0.08) % 1.0, bg_sat * 1.1, bg_value2)

    # Accent: complementary hue, highly saturated
    accent_hue = (h1 + 0.50 + ((seed % 13) - 6) / 100.0) % 1.0
    accent_val = 0.90 + dna.energy * 0.08
    accent = _hsv_hex(accent_hue, 0.85 + dna.saturation * 0.12, _clamp(accent_val))

    # Text: near-white with slight tint
    txt_r_mix = dna.warmth * 0.15
    text_val = 0.92 + (1.0 - dna.darkness) * 0.06
    text_primary = _hsv_hex(h1, txt_r_mix, _clamp(text_val))
    text_secondary = _hsv_hex(accent_hue, 0.40, 0.80)

    # Colour grade
    if dna.grain > 0.4:
        grade = "vintage"
    elif dna.darkness > 0.75 and dna.saturation > 0.65:
        grade = "neon"
    elif dna.warmth > 0.70:
        grade = "warm"
    elif dna.darkness < 0.40:
        grade = ""
    else:
        grade = "cinematic"

    return Palette(
        bg1=bg1, bg2=bg2,
        text_primary=text_primary, text_secondary=text_secondary,
        accent=accent, color_grade=grade,
    )


# ── DNA derivation ────────────────────────────────────────────────────────────

def build_dna(idea: str, genre: str, tone: str) -> VisualDNA:
    """Derive VisualDNA from idea / genre / tone."""
    # Seed from idea content for palette micro-variation
    raw_seed = int(hashlib.md5(idea.encode()).hexdigest()[:8], 16)

    # Start from genre base
    g = genre.lower().replace("-", "_").replace(" ", "_")
    base = dict(_GENRE_DNA.get(g, {"energy": 0.65, "darkness": 0.60, "warmth": 0.45, "saturation": 0.65, "grain": 0.1}))

    # Apply tone delta
    t = tone.lower()
    for k, delta in _TONE_DELTA.get(t, {}).items():
        base[k] = _clamp(base.get(k, 0.5) + delta)

    return VisualDNA(
        energy=_clamp(base.get("energy", 0.7)),
        darkness=_clamp(base.get("darkness", 0.6)),
        warmth=_clamp(base.get("warmth", 0.5)),
        saturation=_clamp(base.get("saturation", 0.7)),
        grain=_clamp(base.get("grain", 0.0)),
        complexity=_clamp(base.get("energy", 0.7) * 0.7 + base.get("saturation", 0.7) * 0.3),
        seed=raw_seed,
    )


# ── Scene timing ──────────────────────────────────────────────────────────────

_SCENE_WEIGHT: Dict[str, float] = {
    "hook":       1.4,
    "build":      0.9,
    "body":       1.0,
    "drop":       1.3,
    "cta":        1.0,
    "outro":      0.8,
    # Extended types
    "verse":      1.0,
    "chorus":     1.2,
    "bridge":     0.85,
    "transition": 0.7,
}

# Minimum seconds per scene — ensures 20 scenes in a 60s video stay ~3 s each
_MIN_SCENE_DUR = 2.5

def allocate_durations(scene_types: List[str], total: float) -> List[float]:
    """Distribute total duration across scenes by weight, enforcing a minimum per scene."""
    n = len(scene_types)
    if n == 0:
        return []
    # Reserve enough headroom for minimums so we never go below _MIN_SCENE_DUR
    min_total = _MIN_SCENE_DUR * n
    effective_total = max(total, min_total)
    weights = [_SCENE_WEIGHT.get(st, 1.0) for st in scene_types]
    total_w = sum(weights)
    return [max(_MIN_SCENE_DUR, effective_total * w / total_w) for w in weights]


# ── Per-scene font / layout decisions ────────────────────────────────────────

def _font_size_for_scene(scene_type: str, energy: float, text_len: int) -> int:
    base = {
        "hook":       68,
        "drop":       66,
        "chorus":     64,
        "build":      54,
        "body":       50,
        "verse":      50,
        "cta":        58,
        "bridge":     46,
        "outro":      46,
        "transition": 42,
    }.get(scene_type, 52)
    # Scale down for long text
    if text_len > 60:
        base = int(base * 0.72)
    elif text_len > 40:
        base = int(base * 0.86)
    # Energy bump
    base = int(base * (1.0 + energy * 0.12))
    return max(28, min(80, base))


def _y_position(scene_type: str, energy: float) -> str:
    if scene_type == "hook":
        return "(h*0.38)"
    if scene_type in ("build", "body", "verse"):
        return "(h*0.50)"
    if scene_type in ("drop", "chorus"):
        return "(h*0.40)"
    if scene_type == "bridge":
        return "(h*0.55)"
    if scene_type == "cta":
        return "(h*0.70)"
    if scene_type in ("outro", "transition"):
        return "(h*0.80)"
    return "(h-text_h)/2"


def _animation_for_scene(scene_type: str, energy: float, seed: int) -> str:
    if energy > 0.80:
        return {"hook": "scale_in", "drop": "scale_in", "chorus": "scale_in"}.get(
            scene_type, "slide_up"
        )
    if energy > 0.55:
        return {"hook": "slide_up", "cta": "scale_in", "chorus": "slide_up"}.get(
            scene_type, "slide_up"
        )
    return "fade"


# ── Main builder ──────────────────────────────────────────────────────────────

def build_scenes(
    scenes_data: List[Dict],
    idea: str,
    genre: str,
    tone: str,
    platform: str,
    artist_name: str,
    total_duration: float,
    width: int,
    height: int,
    awareness: str = "",
    technique_dna: Optional[Dict] = None,
    # ── Veo-parity controls ───────────────────────────────────────────────
    camera_motion: str = "",          # pan_left/zoom_in/tilt_up/dolly_in/static/auto
    negative_prompt: str = "",        # content/style to exclude (forwarded to diffusion)
    seed_override: Optional[int] = None,  # explicit caller seed; overrides idea-hash seed
    motion_intensity: Optional[float] = None,  # 0–1; overrides genre-derived energy
    lighting: str = "",               # cinematic/dramatic/natural/studio/golden_hour/night/neon
    color_temperature: str = "",      # warm/cool/neutral
    fps: int = 24,                    # output frame rate (8/16/24/30)
) -> List[SceneConfig]:
    """
    Build a list of SceneConfig objects from content DNA.
    scenes_data: [{"type": "hook"|"build"|"body"|"drop"|"cta"|"outro", "text": str}, ...]

    When `awareness` is provided, each SceneConfig gets a `diffusion_meta` dict
    that enables the MaxCore neural diffusion background pipeline in _pil_bg_frame.

    Veo-parity controls (camera_motion, negative_prompt, seed_override,
    motion_intensity, lighting, color_temperature, fps) flow through to
    SceneConfig so the render layer can use them directly.  All are
    never-raise — a bad value leaves the DNA at its genre-derived default.
    """
    if not scenes_data:
        return []

    dna = build_dna(idea, genre, tone)

    # ── Apply Veo-parity caller overrides to the derived DNA ────────────────
    # Never-raise: a bad value leaves the DNA at its genre-derived default.
    if seed_override is not None:
        try:
            dna = VisualDNA(
                energy=dna.energy, darkness=dna.darkness, warmth=dna.warmth,
                saturation=dna.saturation, grain=dna.grain,
                complexity=dna.complexity, seed=int(seed_override),
            )
        except Exception:
            pass

    if motion_intensity is not None:
        try:
            _mi = max(0.0, min(1.0, float(motion_intensity)))
            dna = VisualDNA(
                energy=_mi, darkness=dna.darkness, warmth=dna.warmth,
                saturation=dna.saturation, grain=dna.grain,
                complexity=dna.complexity, seed=dna.seed,
            )
        except Exception:
            pass

    if color_temperature:
        _ct_warmth = {"warm": 0.78, "cool": 0.22, "neutral": 0.50}.get(
            color_temperature.lower()
        )
        if _ct_warmth is not None:
            try:
                dna = VisualDNA(
                    energy=dna.energy, darkness=dna.darkness, warmth=_ct_warmth,
                    saturation=dna.saturation, grain=dna.grain,
                    complexity=dna.complexity, seed=dna.seed,
                )
            except Exception:
                pass

    # Lighting → base colour-grade preset used on non-emphasis scenes.
    # High-energy emphasis scenes (chorus/drop) still override to neon/vintage
    # so dynamic scenes keep their visual punch.
    _LIGHTING_GRADE: Dict[str, str] = {
        "cinematic":    "cinematic",
        "dramatic":     "cinematic",
        "natural":      "",
        "studio":       "",
        "golden_hour":  "warm",
        "night":        "cinematic",
        "neon":         "neon",
        "vintage":      "vintage",
    }
    _lighting_grade_override: Optional[str] = (
        _LIGHTING_GRADE.get(lighting.lower()) if lighting else None
    )

    # fps must be a supported value; fall back to 24 for anything out of range.
    _fps = fps if fps in (8, 16, 24, 30) else 24

    palette = derive_palette(dna)
    base_bg_type = _choose_bg_type(dna.energy, dna.darkness, dna.seed)

    # For large scene counts we cycle through bg types for visual dynamism
    _BG_ROTATION = [
        "animated_gradient", "wave", "plasma",
        "radial", "aurora", "animated_gradient",
    ]
    # Emphasis types always get the high-energy bg
    _EMPHASIS_TYPES = {"chorus", "drop", "hook"}

    scene_types = [s.get("type", "body") for s in scenes_data]
    durations = allocate_durations(scene_types, total_duration)

    configs: List[SceneConfig] = []

    # Grain is film-grain intensity (0–20 FFmpeg scale)
    grain_val = int(dna.grain * 18)
    base_vignette = 0.35 + dna.darkness * 0.30

    # Show progress bar on the scene before the last
    show_progress = dna.energy > 0.70 and len(scenes_data) > 4
    show_border = dna.complexity > 0.72 and dna.saturation > 0.65

    import colorsys as _cs

    def _shift_hex(hex_color: str, shift: float) -> str:
        c = hex_color.lstrip("0x")
        try:
            r, g, b = int(c[0:2], 16) / 255, int(c[2:4], 16) / 255, int(c[4:6], 16) / 255
            h, s, v = _cs.rgb_to_hsv(r, g, b)
            r2, g2, b2 = _cs.hsv_to_rgb((h + shift) % 1.0, s, v)
            return f"0x{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"
        except Exception:
            return hex_color

    many_scenes = len(scenes_data) > 6

    for idx, (scene_info, dur) in enumerate(zip(scenes_data, durations)):
        scene_type = scene_info.get("type", "body")
        text = scene_info.get("text", "").strip()

        if not text:
            continue

        # ── Per-scene bg type ───────────────────────────────────────────
        if many_scenes:
            if scene_type in _EMPHASIS_TYPES:
                sc_bg_type = "plasma" if dna.energy > 0.65 else "animated_gradient"
            else:
                sc_bg_type = _BG_ROTATION[(idx // 4) % len(_BG_ROTATION)]
        else:
            sc_bg_type = base_bg_type

        # ── Per-scene hue micro-shift ────────────────────────────────────
        scene_seed = (dna.seed + idx * 17) % 360
        scene_hue_shift = (scene_seed % 9 - 4) / 360.0

        # Emphasis scenes (chorus/drop) get an amplified hue shift for impact
        if scene_type in _EMPHASIS_TYPES and many_scenes:
            scene_hue_shift *= 3.0

        sc_bg1 = _shift_hex(palette.bg1, scene_hue_shift)
        sc_bg2 = _shift_hex(palette.bg2, scene_hue_shift * 1.5)

        # ── Per-scene vignette variation (±0.08) ────────────────────────
        vignette_shift = (((idx * 7) % 5) - 2) * 0.04
        sc_vignette = _clamp(base_vignette + vignette_shift, 0.15, 0.75)

        # ── Per-scene colour grade ───────────────────────────────────────
        # Emphasis scenes override to neon/vintage for visual impact;
        # all other scenes use the lighting preset (if set) or the
        # palette's DNA-derived grade.
        if scene_type in ("chorus", "drop") and dna.energy > 0.70:
            sc_color_grade = "neon"
        elif scene_type == "bridge" and dna.grain > 0.25:
            sc_color_grade = "vintage"
        elif scene_type in ("outro",) and dna.warmth > 0.60:
            sc_color_grade = "warm"
        else:
            sc_color_grade = (
                _lighting_grade_override
                if _lighting_grade_override is not None
                else palette.color_grade
            )

        # ── Per-scene text colour: accent for high-energy emphasis ───────
        if scene_type in ("chorus", "drop") and many_scenes:
            sc_text_color = palette.accent
        else:
            sc_text_color = palette.text_primary

        font_size = _font_size_for_scene(scene_type, dna.energy, len(text))
        # Slightly larger text on emphasis scenes in long videos
        if scene_type in ("chorus", "drop") and many_scenes:
            font_size = min(80, int(font_size * 1.08))

        y_pos = _y_position(scene_type, dna.energy)
        animation = _animation_for_scene(scene_type, dna.energy, dna.seed + idx)

        # Fade timing: snappier on high-energy scenes
        if scene_type in ("chorus", "drop", "hook"):
            fade_in, fade_out = 0.25, 0.25
        elif dna.energy > 0.70:
            fade_in, fade_out = 0.4, 0.35
        else:
            fade_in, fade_out = 0.7, 0.55

        # Reduce max_chars for many-scene videos (shorter text = more dynamic)
        max_chars = 22 if (many_scenes and font_size > 56) else (28 if font_size > 56 else 34)

        texts = [
            TextElement(
                text=text,
                font=FONT_PATH,
                size=font_size,
                color=sc_text_color,
                x="(w-text_w)/2",
                y=y_pos,
                start=0.0,
                end=dur,
                fade_in=fade_in,
                fade_out=fade_out,
                shadow=True,
                shadow_color="0x000000",
                shadow_offset=max(2, int(font_size * 0.05)),
                max_chars=max_chars,
                animation=animation,
            )
        ]

        # Artist label on outro / cta scenes
        if artist_name and scene_type in ("outro", "cta"):
            label_size = max(22, int(font_size * 0.45))
            texts.append(
                TextElement(
                    text=artist_name.upper(),
                    font=FONT_PATH,
                    size=label_size,
                    color=palette.accent,
                    x="(w-text_w)/2",
                    y="(h*0.88)",
                    start=fade_in,
                    end=dur,
                    fade_in=0.5,
                    fade_out=0.3,
                    shadow=True,
                    shadow_color="0x000000",
                    shadow_offset=2,
                    max_chars=24,
                    animation="fade",
                )
            )

        effects: List[str] = []
        if dna.energy > 0.75:
            effects.append("breathing")

        _diffusion_meta: Dict = {
            "idea": idea,
            "platform": platform,
            "tone": tone,
            "awareness": awareness,
            "brand": artist_name or "",
            "dna": technique_dna or {
                "energy": dna.energy,
                "darkness": dna.darkness,
                "warmth": dna.warmth,
                "saturation": dna.saturation,
            },
            "technique_source": "real_asset" if technique_dna else "genre_prior",
            # ── Veo-parity conditioning fields ──────────────────────────
            "camera_motion":    camera_motion or "",
            "negative_prompt":  negative_prompt or "",
            "lighting":         lighting or "",
            "color_temperature": color_temperature or "",
        } if (awareness or camera_motion or negative_prompt or lighting) else {}

        cfg = SceneConfig(
            duration=dur,
            bg_type=sc_bg_type,
            bg_color1=sc_bg1,
            bg_color2=sc_bg2,
            texts=texts,
            effects=effects,
            vignette=sc_vignette,
            film_grain_amount=grain_val,
            color_grade=sc_color_grade,
            letterbox_ratio=0.035 if dna.energy > 0.70 and dna.darkness > 0.65 else 0.0,
            corner_accent_color=palette.accent if dna.complexity > 0.75 and idx == 0 else "",
            border_color=palette.accent if show_border and idx % 2 == 0 else "",
            breathing=("breathing" in effects),
            show_progress=show_progress and idx == len(scenes_data) - 2,
            progress_color=palette.accent,
            brand=artist_name or "",
            diffusion_meta=_diffusion_meta if _diffusion_meta else None,
            # Veo-parity render controls propagated per-scene
            camera_motion=camera_motion or "",
            negative_prompt=negative_prompt or "",
            fps=_fps,
        )
        configs.append(cfg)

    return configs
