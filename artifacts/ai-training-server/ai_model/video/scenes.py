from __future__ import annotations
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Optional, List
from .effects import (
    vignette_filter, color_grade_cinematic, color_grade_warm, color_grade_cool,
    color_grade_neon, color_grade_vintage,
    corner_accents, letterbox, animated_border, progress_bar,
)
from .ffmpeg_util import run_ffmpeg

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_font(name: str) -> str:
    """Return an absolute path to the named font file, or '' if not found.

    Priority:
      1. Bundled fonts shipped inside this package (works in all environments).
      2. Standard Debian/Ubuntu system path (dev container fallback).
    Callers must treat an empty return value as "font unavailable" and omit
    the ``fontfile=`` argument from FFmpeg drawtext filters.
    """
    bundled = os.path.join(_MODULE_DIR, "fonts", name)
    if os.path.exists(bundled):
        return bundled
    for system_path in [
        f"/usr/share/fonts/truetype/dejavu/{name}",
        f"/usr/share/fonts/dejavu/{name}",
        f"/nix/var/nix/profiles/default/share/fonts/truetype/{name}",
    ]:
        if os.path.exists(system_path):
            return system_path
    print(f"[VideoRender][WARN] Font not found: {name} — drawtext will use ffmpeg built-in", file=sys.stderr)
    return ""


FONT_PATH         = _resolve_font("DejaVuSans-Bold.ttf")
FONT_PATH_REGULAR = _resolve_font("DejaVuSans.ttf")

TEMP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "uploads", "videos", ".tmp",
)


def _esc(text: str) -> str:
    """Escape text for ffmpeg drawtext filter inside text='...'.
    Apostrophes/single-quotes CANNOT be inside single-quoted filter strings —
    they must be removed or replaced rather than backslash-escaped.
    """
    # Replace apostrophes / curly quotes with nothing (they break drawtext parsing)
    text = text.replace("'", "").replace("\u2018", "").replace("\u2019", "")
    # Backslash-escape the characters ffmpeg filter grammar requires escaped
    for ch in ["\\", ":", ";", "[", "]", ",", "="]:
        text = text.replace(ch, "\\" + ch)
    return text


def _wrap(text: str, max_chars: int = 28) -> str:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


@dataclass
class TextElement:
    text: str
    font: str = FONT_PATH
    size: int = 48
    color: str = "0xffffff"
    x: str = "(w-text_w)/2"
    y: str = "(h-text_h)/2"
    start: float = 0.0
    end: float = -1.0
    fade_in: float = 0.5
    fade_out: float = 0.5
    shadow: bool = True
    shadow_color: str = "0x000000"
    shadow_offset: int = 3
    max_chars: int = 28
    animation: str = "fade"


@dataclass
class SceneConfig:
    duration: float = 3.0
    bg_type: str = "gradient"
    bg_color1: str = "0x1a1a2e"
    bg_color2: str = "0x16213e"
    texts: List[TextElement] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    vignette: float = 0.4
    film_grain_amount: int = 0
    color_grade: str = ""
    letterbox_ratio: float = 0.0
    corner_accent_color: str = ""
    border_color: str = ""
    breathing: bool = False
    show_progress: bool = False
    progress_color: str = "0xe94560"
    retrieval_conditioned: bool = True
    brand: str = ""


def _build_text_filter(te: TextElement, scene_dur: float) -> List[str]:
    wrapped = _esc(_wrap(te.text, te.max_chars))
    end = te.end if te.end > 0 else scene_dur
    enable = f"between(t\\,{te.start:.2f}\\,{end:.2f})"

    fs = te.fade_in
    fe = te.fade_out
    fade_start = te.start
    fade_end = end

    alpha_expr = (
        f"if(lt(t\\,{fade_start + fs:.2f})\\,"
        f"min(1\\,(t-{fade_start:.2f})/{fs:.2f})\\,"
        f"if(gt(t\\,{fade_end - fe:.2f})\\,"
        f"max(0\\,({fade_end:.2f}-t)/{fe:.2f})\\,1))"
    )

    font_arg = f"fontfile={te.font}:" if te.font and os.path.exists(te.font) else ""

    parts = []
    if te.shadow:
        sx = f"({te.x})+{te.shadow_offset}" if not te.x.replace("-", "").isdigit() else str(int(te.x) + te.shadow_offset)
        sy = f"({te.y})+{te.shadow_offset}" if not te.y.replace("-", "").isdigit() else str(int(te.y) + te.shadow_offset)
        parts.append(
            f"drawtext={font_arg}text='{wrapped}':fontcolor={te.shadow_color}@0.5"
            f":fontsize={te.size}:x={sx}:y={sy}"
            f":enable='{enable}':alpha='{alpha_expr}'"
        )

    if te.animation == "slide_up":
        y_anim = f"if(lt(t\\,{fade_start + fs:.2f})\\,({te.y})+50*(1-(t-{fade_start:.2f})/{fs:.2f})\\,{te.y})"
        parts.append(
            f"drawtext={font_arg}text='{wrapped}':fontcolor={te.color}"
            f":fontsize={te.size}:x={te.x}:y={y_anim}"
            f":enable='{enable}':alpha='{alpha_expr}'"
        )
    elif te.animation == "scale_in":
        size_anim = (
            f"if(lt(t\\,{fade_start + fs:.2f})\\,"
            f"{int(te.size * 0.5)}+{int(te.size * 0.5)}*(t-{fade_start:.2f})/{fs:.2f}\\,"
            f"{te.size})"
        )
        parts.append(
            f"drawtext={font_arg}text='{wrapped}':fontcolor={te.color}"
            f":fontsize={size_anim}:x={te.x}:y={te.y}"
            f":enable='{enable}':alpha='{alpha_expr}'"
        )
    else:
        parts.append(
            f"drawtext={font_arg}text='{wrapped}':fontcolor={te.color}"
            f":fontsize={te.size}:x={te.x}:y={te.y}"
            f":enable='{enable}':alpha='{alpha_expr}'"
        )

    return parts


# ── PIL + NumPy background generation (replaces slow geq ffmpeg filters) ─────

def _parse_hex_color(h: str) -> tuple:
    h = h.strip()
    if h.startswith(("0x", "0X")):
        h = h[2:]
    h = h.lstrip("#").zfill(6)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _np_gradient(c1: tuple, c2: tuple, w: int, h: int):
    import numpy as np
    t = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(-1, 1)
    ch0 = np.broadcast_to(np.clip(c1[0] + (c2[0] - c1[0]) * t, 0, 255).astype(np.uint8), (h, w)).copy()
    ch1 = np.broadcast_to(np.clip(c1[1] + (c2[1] - c1[1]) * t, 0, 255).astype(np.uint8), (h, w)).copy()
    ch2 = np.broadcast_to(np.clip(c1[2] + (c2[2] - c1[2]) * t, 0, 255).astype(np.uint8), (h, w)).copy()
    return np.stack([ch0, ch1, ch2], axis=2)


def _np_radial(c1: tuple, c2: tuple, w: int, h: int):
    import numpy as np
    cx, cy = w / 2.0, h / 2.0
    xs = np.arange(w, dtype=np.float32).reshape(1, -1)
    ys = np.arange(h, dtype=np.float32).reshape(-1, 1)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    d = (d / float(d.max())).astype(np.float32)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for ch in range(3):
        arr[:, :, ch] = np.clip(c1[ch] + (c2[ch] - c1[ch]) * d, 0, 255).astype(np.uint8)
    return arr


def _np_plasma(c1: tuple, c2: tuple, w: int, h: int, style: str = "plasma"):
    import numpy as np
    from PIL import Image
    sw, sh = max(w // 4, 64), max(h // 4, 64)
    xs = np.linspace(0.0, 1.0, sw, dtype=np.float32).reshape(1, -1)
    ys = np.linspace(0.0, 1.0, sh, dtype=np.float32).reshape(-1, 1)
    if style == "aurora":
        field = 0.5 + 0.3 * np.sin(xs * 10.0) + 0.2 * np.cos(ys * 8.0)
    else:
        field = 0.5 + 0.25 * np.sin(xs * 8.0 + 1.0) + 0.25 * np.cos(ys * 6.0 + 0.5)
    field = np.clip(field, 0.0, 1.0).astype(np.float32)
    arr = np.zeros((sh, sw, 3), dtype=np.uint8)
    for ch in range(3):
        arr[:, :, ch] = np.clip(c1[ch] + (c2[ch] - c1[ch]) * field, 0, 255).astype(np.uint8)
    return np.array(Image.fromarray(arr).resize((w, h), Image.BILINEAR))


def _pil_bg_frame(scene: SceneConfig, width: int, height: int) -> str:
    """
    Generate a static background PNG using PIL + NumPy.
    Fast (<0.2 s) — no ffmpeg geq per-frame computation needed.
    Film grain is baked into the PNG via NumPy noise so there is no overhead
    at encode time.
    """
    import numpy as np
    from PIL import Image
    os.makedirs(TEMP_DIR, exist_ok=True)
    bg_path = os.path.join(TEMP_DIR, f"bg_{uuid.uuid4().hex[:8]}.png")

    try:
        c1 = _parse_hex_color(scene.bg_color1)
        c2 = _parse_hex_color(scene.bg_color2)
    except Exception:
        c1, c2 = (26, 26, 46), (22, 33, 62)

    bg_type = getattr(scene, "bg_type", "gradient")
    if bg_type == "radial":
        arr = _np_radial(c1, c2, width, height)
    elif bg_type in ("plasma", "aurora"):
        arr = _np_plasma(c1, c2, width, height, bg_type)
    else:
        arr = _np_gradient(c1, c2, width, height)

    if getattr(scene, "retrieval_conditioned", True):
        try:
            from ai_model.retrieval.rcgs import condition_background
            arr = condition_background(
                arr, width, height,
                brand=(getattr(scene, "brand", "") or None),
            )
        except Exception:
            pass

    grain = getattr(scene, "film_grain_amount", 0)
    if grain > 0:
        std = max(1, int(grain * 1.5))
        noise = np.random.randint(-std, std + 1, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    Image.fromarray(arr).save(bg_path, format="PNG")
    return bg_path


# ── Scene rendering ────────────────────────────────────────────────────────────

def render_scene(scene: SceneConfig, width: int, height: int, scene_id: str = "") -> Optional[str]:
    os.makedirs(TEMP_DIR, exist_ok=True)
    if not scene_id:
        scene_id = uuid.uuid4().hex[:8]
    out_path = os.path.join(TEMP_DIR, f"scene_{scene_id}.mp4")
    return _render_pil_based(scene, width, height, scene.duration, out_path)


def _render_pil_based(
    scene: SceneConfig,
    width: int,
    height: int,
    dur: float,
    out_path: str,
) -> Optional[str]:
    """
    Render one scene:
      1. Generate background PNG via PIL+NumPy (fast, no per-pixel ffmpeg geq).
      2. Encode with ffmpeg using the PNG as a looped still + drawtext overlay.
    Falls back to a solid-colour render if PIL fails.
    """
    bg_png: Optional[str] = None
    try:
        bg_png = _pil_bg_frame(scene, width, height)
    except Exception:
        pass

    if not bg_png or not os.path.exists(bg_png):
        return _render_fallback(scene, width, height, dur, out_path)

    vf_parts: List[str] = []

    if scene.vignette > 0:
        vf_parts.append(vignette_filter(scene.vignette))

    grade_map = {
        "cinematic": color_grade_cinematic,
        "warm":      color_grade_warm,
        "cool":      color_grade_cool,
        "neon":      color_grade_neon,
        "vintage":   color_grade_vintage,
    }
    if scene.color_grade and scene.color_grade in grade_map:
        vf_parts.append(grade_map[scene.color_grade]())

    if scene.letterbox_ratio > 0:
        vf_parts.append(letterbox(width, height, scene.letterbox_ratio))
    if scene.corner_accent_color:
        vf_parts.append(corner_accents(width, height, scene.corner_accent_color))
    if scene.border_color:
        vf_parts.append(animated_border(width, height, scene.border_color))

    for te in scene.texts:
        vf_parts.extend(_build_text_filter(te, dur))

    if scene.show_progress:
        vf_parts.append(progress_bar(width, height, scene.progress_color))

    vf = ",".join(vf_parts) if vf_parts else "null"

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", "24", "-i", bg_png,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-t", str(dur), out_path,
    ]

    try:
        result = run_ffmpeg(cmd, timeout=45)
        _safe_remove(bg_png)
        if result.returncode != 0:
            print(
                f"[VideoRender][ERROR] ffmpeg PIL render failed (rc={result.returncode}):\n{result.stderr[-800:]}",
                file=sys.stderr,
            )
            return _render_fallback(scene, width, height, dur, out_path)
        return out_path
    except Exception as exc:
        print(f"[VideoRender][ERROR] _render_pil_based exception: {exc}", file=sys.stderr)
        _safe_remove(bg_png)
        return _render_fallback(scene, width, height, dur, out_path)


def _render_fallback(
    scene: SceneConfig,
    width: int,
    height: int,
    dur: float,
    out_path: str,
) -> Optional[str]:
    """Last-resort: solid colour background with drawtext, ultrafast encode."""
    bg_color = scene.bg_color1

    vf_parts: List[str] = []
    if scene.vignette > 0:
        vf_parts.append(vignette_filter(scene.vignette))
    for te in scene.texts:
        vf_parts.extend(_build_text_filter(te, dur))

    vf = ",".join(vf_parts) if vf_parts else "null"

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={bg_color}:s={width}x{height}:d={dur}:r=24",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-t", str(dur), out_path,
    ]

    try:
        result = run_ffmpeg(cmd, timeout=45)
        if result.returncode != 0:
            print(
                f"[VideoRender][ERROR] ffmpeg fallback render failed (rc={result.returncode}):\n{result.stderr[-800:]}",
                file=sys.stderr,
            )
            return None
        return out_path
    except Exception as exc:
        print(f"[VideoRender][ERROR] _render_fallback exception: {exc}", file=sys.stderr)
        return None


def _safe_remove(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def cleanup_temp(paths: List[str]):
    for p in paths:
        _safe_remove(p)


# ── Scene compositing ─────────────────────────────────────────────────────────

def composite_scenes(
    scene_paths: List[str],
    output_path: str,
    transition: str = "fade",
    transition_dur: float = 0.5,
    audio_path: Optional[str] = None,
) -> bool:
    """
    Concatenate rendered scene clips into one MP4.
    Uses a simple concat demuxer for speed; transitions are applied via
    xfade when there are exactly 2 clips (generalising further is left as
    an enhancement — concat is already visually clean).
    """
    if not scene_paths:
        return False

    if len(scene_paths) == 1:
        import shutil
        try:
            shutil.copy2(scene_paths[0], output_path)
            return True
        except Exception:
            return False

    concat_list = os.path.join(TEMP_DIR, f"concat_{uuid.uuid4().hex[:8]}.txt")
    os.makedirs(TEMP_DIR, exist_ok=True)

    try:
        with open(concat_list, "w") as f:
            for p in scene_paths:
                f.write(f"file '{p}'\n")

        if audio_path and os.path.exists(audio_path):
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_list,
                "-i", audio_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-shortest", output_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_list,
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                output_path,
            ]

        result = run_ffmpeg(cmd, timeout=120)
        _safe_remove(concat_list)
        return result.returncode == 0
    except Exception:
        _safe_remove(concat_list)
        return False
