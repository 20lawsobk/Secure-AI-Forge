"""
MaxBooster Image Engine
=======================
Renders platform-sized promotional images entirely in-house using PIL.
No external AI APIs required — powered by:
  - VisualSpecAgent (in-house transformer) for concept generation
  - NumPy gradient fills for cinematic backgrounds
  - PIL for typography, layout, and decorative elements
"""
from __future__ import annotations

import uuid
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter  # noqa: F401
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# ─── Constants ────────────────────────────────────────────────────────────────

_FONT_BOLD   = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT_NORMAL = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

PLATFORM_DIMS: dict[str, Tuple[int, int]] = {
    "square_1_1":        (1080, 1080),
    "vertical_9_16":     (1080, 1920),
    "landscape_16_9":    (1920, 1080),
    "landscape_2_1":     (1200,  600),
    "landscape_1200x627":(1200,  627),
}

# Each color scheme: (grad_top_RGB, grad_bottom_RGB, accent_RGB, text_RGB)
COLOR_SCHEMES: dict[str, tuple] = {
    "dark_neon":       ((10,  0,  30), (0,  0,   0), (180,  0, 255), (255, 255, 255)),
    "vibrant_pastel":  ((255,200,220), (200,180,240), (255, 80, 160), ( 30,  10,  50)),
    "monochrome":      ((30,  30, 30), (0,   0,   0), (200, 200, 200), (255, 255, 255)),
    "high_contrast":   ((5,   5,  60), (0,   0,   0), ( 50, 120, 255), (255, 255, 255)),
    "corporate_blue":  ((10,  30, 80), (5,  15,  40), ( 80, 140, 255), (255, 255, 255)),
    "warm_earth":      ((60,  30,  10), (15,  5,   0), (210, 140,  50), (255, 240, 210)),
    "bold_red_gold":   ((60,   0,   0), (10,  0,   0), (220, 160,   0), (255, 255, 255)),
}

_DEFAULT_SCHEME = ("dark_neon", COLOR_SCHEMES["dark_neon"])

_UPLOADS_DIR = Path(__file__).resolve().parents[2] / "uploads" / "images"


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ImageRequest:
    prompt: str
    color_scheme: str = "dark_neon"
    layout: str = "square_1_1"
    platform: str = "instagram"
    artist_name: str = "MaxBooster"
    intent: str = "promotional"
    style_tags: list = field(default_factory=lambda: ["cinematic"])


@dataclass
class ImageResult:
    success: bool
    filename: str
    url: str
    width: int
    height: int
    color_scheme: str
    layout: str
    prompt_used: str
    error: str = ""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_font(size: int, bold: bool = False) -> "ImageFont.FreeTypeFont":
    path = _FONT_BOLD if bold else _FONT_NORMAL
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _gradient_array(w: int, h: int, top: tuple, bottom: tuple) -> np.ndarray:
    """Create a vertical gradient as an H×W×3 uint8 array."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for i, (a, b) in enumerate(zip(top, bottom)):
        col = np.linspace(a, b, h, dtype=np.float32)
        arr[:, :, i] = col[:, np.newaxis]
    return arr


def _add_noise_texture(arr: np.ndarray, strength: int = 12) -> np.ndarray:
    """Light grain overlay for a film/cinematic feel."""
    noise = np.random.randint(-strength, strength + 1, arr.shape, dtype=np.int16)
    return np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _draw_gradient_bar(draw: "ImageDraw.Draw", x0: int, y0: int, x1: int, y1: int,
                       color_a: tuple, color_b: tuple, steps: int = 8):
    """Horizontal gradient bar (for accent strips)."""
    bar_w = x1 - x0
    seg = bar_w // steps
    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0
        r = int(color_a[0] * (1 - t) + color_b[0] * t)
        g = int(color_a[1] * (1 - t) + color_b[1] * t)
        b = int(color_a[2] * (1 - t) + color_b[2] * t)
        draw.rectangle([x0 + i * seg, y0, x0 + (i + 1) * seg, y1], fill=(r, g, b))


def _wrap_text(text: str, max_chars: int) -> list[str]:
    return textwrap.wrap(text, width=max_chars)


def _scheme_for(name: str) -> tuple:
    return COLOR_SCHEMES.get(name, COLOR_SCHEMES["dark_neon"])


# ─── Core renderer ────────────────────────────────────────────────────────────

class ImageEngine:
    def __init__(self):
        _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    def render(self, req: ImageRequest) -> ImageResult:
        if not _PIL_OK:
            return ImageResult(
                success=False, filename="", url="", width=0, height=0,
                color_scheme=req.color_scheme, layout=req.layout,
                prompt_used=req.prompt, error="PIL not available",
            )

        w, h = PLATFORM_DIMS.get(req.layout, (1080, 1080))
        grad_top, grad_bot, accent, text_col = _scheme_for(req.color_scheme)
        is_portrait = h > w

        # ── Background gradient ────────────────────────────────────────────────
        bg_arr = _gradient_array(w, h, grad_top, grad_bot)
        bg_arr = _add_noise_texture(bg_arr, strength=10)
        img = Image.fromarray(bg_arr, "RGB")
        draw = ImageDraw.Draw(img)

        # ── Top accent bar ─────────────────────────────────────────────────────
        bar_h = max(6, h // 90)
        _draw_gradient_bar(draw, 0, 0, w, bar_h, accent, grad_top)

        # ── Bottom accent bar ──────────────────────────────────────────────────
        _draw_gradient_bar(draw, 0, h - bar_h, w, h, grad_bot, accent)

        # ── Corner brackets ────────────────────────────────────────────────────
        pad   = max(20, w // 40)
        span  = max(40, w // 18)
        thick = max(3, w // 270)
        for (cx, cy, dx, dy) in [
            (pad, pad,             +1, +1),
            (w - pad, pad,         -1, +1),
            (pad, h - pad,         +1, -1),
            (w - pad, h - pad,     -1, -1),
        ]:
            # PIL needs [x0, y0, x1, y1] with x0 ≤ x1, y0 ≤ y1
            ax0, ax1 = sorted([cx, cx + dx * span])
            ay0, ay1 = sorted([cy, cy + dy * thick])
            draw.rectangle([ax0, ay0, ax1, ay1], fill=accent)
            bx0, bx1 = sorted([cx, cx + dx * thick])
            by0, by1 = sorted([cy, cy + dy * span])
            draw.rectangle([bx0, by0, bx1, by1], fill=accent)

        # ── Platform label (top-right) ─────────────────────────────────────────
        plat_font_sz  = max(18, w // 52)
        plat_font     = _load_font(plat_font_sz, bold=False)
        plat_label    = req.platform.upper()
        plat_x        = w - pad - len(plat_label) * (plat_font_sz // 1.7)
        plat_y        = pad + bar_h + 8
        # Shadow
        draw.text((plat_x + 2, plat_y + 2), plat_label, font=plat_font,
                  fill=(0, 0, 0, 120))
        draw.text((plat_x, plat_y), plat_label, font=plat_font, fill=accent)

        # ── Artist / intent badge (top-left) ──────────────────────────────────
        badge_font_sz = max(16, w // 56)
        badge_font    = _load_font(badge_font_sz, bold=True)
        badge_label   = req.artist_name.upper()
        draw.text((pad + 4, plat_y + 2), badge_label, font=badge_font,
                  fill=(0, 0, 0, 120))
        draw.text((pad, plat_y), badge_label, font=badge_font, fill=text_col)

        # ── Central divider line ───────────────────────────────────────────────
        centre_y   = h // 2
        line_thick = max(2, h // 300)
        glow_color = tuple(min(255, int(c * 1.4)) for c in accent)
        draw.rectangle([pad * 3, centre_y - line_thick * 2,
                        w - pad * 3, centre_y + line_thick * 2],
                       fill=(*glow_color, 40))
        draw.rectangle([pad * 3, centre_y - line_thick // 2,
                        w - pad * 3, centre_y + line_thick // 2],
                       fill=accent)

        # ── Main headline text (above divider) ────────────────────────────────
        headline_font_sz = max(32, w // (22 if is_portrait else 28))
        headline_font    = _load_font(headline_font_sz, bold=True)
        max_chars        = max(12, w // (headline_font_sz // 2))
        lines            = _wrap_text(req.prompt, max_chars)[:4]  # cap 4 lines

        line_gap  = int(headline_font_sz * 1.25)
        block_h   = len(lines) * line_gap
        text_y    = centre_y - bar_h * 2 - block_h - int(h * 0.04)

        for i, line in enumerate(lines):
            lx = pad + int(w * 0.05)
            ly = text_y + i * line_gap
            # Drop shadow
            draw.text((lx + 3, ly + 3), line, font=headline_font,
                      fill=(0, 0, 0, 160))
            draw.text((lx, ly), line, font=headline_font, fill=text_col)

        # ── Sub-label (below divider) — intent tag ────────────────────────────
        sub_font_sz = max(20, w // 44)
        sub_font    = _load_font(sub_font_sz, bold=False)
        sub_label   = f"#{req.intent.upper()} • {' • '.join(t.upper() for t in req.style_tags[:3])}"
        sub_y       = centre_y + int(h * 0.04)
        draw.text((pad + int(w * 0.05) + 2, sub_y + 2), sub_label, font=sub_font,
                  fill=(0, 0, 0, 140))
        draw.text((pad + int(w * 0.05), sub_y), sub_label, font=sub_font,
                  fill=accent)

        # ── Style tag dots ────────────────────────────────────────────────────
        dot_y  = sub_y + sub_font_sz + 16
        dot_r  = max(5, w // 160)
        dot_dx = dot_r * 3
        for idx in range(min(len(req.style_tags), 5)):
            cx = pad + int(w * 0.05) + idx * dot_dx * 2
            draw.ellipse([cx - dot_r, dot_y - dot_r, cx + dot_r, dot_y + dot_r],
                         fill=accent)

        # ── Save ──────────────────────────────────────────────────────────────
        fname = f"img_{uuid.uuid4().hex[:16]}.png"
        out_path = _UPLOADS_DIR / fname
        img.save(str(out_path), "PNG", optimize=True)

        return ImageResult(
            success=True,
            filename=fname,
            url=f"/uploads/images/{fname}",
            width=w,
            height=h,
            color_scheme=req.color_scheme,
            layout=req.layout,
            prompt_used=req.prompt,
        )
