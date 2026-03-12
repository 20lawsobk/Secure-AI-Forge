from __future__ import annotations
import subprocess
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional, List
from .effects import (
    animated_gradient_bg, radial_gradient_bg, wave_gradient_bg, plasma_bg, aurora_bg,
    vignette_filter, film_grain, color_grade_cinematic, color_grade_warm, color_grade_cool,
    color_grade_neon, color_grade_vintage,
    corner_accents, letterbox, breathing_brightness, progress_bar, animated_border,
)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_PATH_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads", "videos", ".tmp")

GEQ_SCALE = 4


def _esc(text: str) -> str:
    for ch in ["\\", "'", ":", ";", "[", "]", ","]:
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

    parts = []
    if te.shadow:
        sx = f"({te.x})+{te.shadow_offset}" if not te.x.replace("-", "").isdigit() else str(int(te.x) + te.shadow_offset)
        sy = f"({te.y})+{te.shadow_offset}" if not te.y.replace("-", "").isdigit() else str(int(te.y) + te.shadow_offset)
        parts.append(
            f"drawtext=fontfile={te.font}:text='{wrapped}':fontcolor={te.shadow_color}@0.5"
            f":fontsize={te.size}:x={sx}:y={sy}"
            f":enable='{enable}':alpha='{alpha_expr}'"
        )

    if te.animation == "slide_up":
        y_anim = f"if(lt(t\\,{fade_start + fs:.2f})\\,({te.y})+50*(1-(t-{fade_start:.2f})/{fs:.2f})\\,{te.y})"
        parts.append(
            f"drawtext=fontfile={te.font}:text='{wrapped}':fontcolor={te.color}"
            f":fontsize={te.size}:x={te.x}:y={y_anim}"
            f":enable='{enable}':alpha='{alpha_expr}'"
        )
    elif te.animation == "scale_in":
        size_anim = f"if(lt(t\\,{fade_start + fs:.2f})\\,{int(te.size * 0.5)}+{int(te.size * 0.5)}*(t-{fade_start:.2f})/{fs:.2f}\\,{te.size})"
        parts.append(
            f"drawtext=fontfile={te.font}:text='{wrapped}':fontcolor={te.color}"
            f":fontsize={size_anim}:x={te.x}:y={te.y}"
            f":enable='{enable}':alpha='{alpha_expr}'"
        )
    else:
        parts.append(
            f"drawtext=fontfile={te.font}:text='{wrapped}':fontcolor={te.color}"
            f":fontsize={te.size}:x={te.x}:y={te.y}"
            f":enable='{enable}':alpha='{alpha_expr}'"
        )

    return parts


def _needs_geq(bg_type: str) -> bool:
    return bg_type in ("animated_gradient", "radial", "wave", "plasma", "aurora", "gradient")


def render_scene(scene: SceneConfig, width: int, height: int, scene_id: str = "") -> Optional[str]:
    os.makedirs(TEMP_DIR, exist_ok=True)
    if not scene_id:
        scene_id = uuid.uuid4().hex[:8]

    out_path = os.path.join(TEMP_DIR, f"scene_{scene_id}.mp4")
    dur = scene.duration

    if _needs_geq(scene.bg_type):
        return _render_two_pass(scene, width, height, dur, out_path)
    else:
        return _render_single_pass(scene, width, height, dur, out_path)


def _render_two_pass(scene: SceneConfig, width: int, height: int, dur: float, out_path: str) -> Optional[str]:
    sw = width // GEQ_SCALE
    sh = height // GEQ_SCALE

    if scene.bg_type == "animated_gradient":
        bg_filter = animated_gradient_bg(sw, sh, dur, scene.bg_color1, scene.bg_color2)
    elif scene.bg_type == "radial":
        bg_filter = radial_gradient_bg(sw, sh, scene.bg_color1, scene.bg_color2)
    elif scene.bg_type == "wave":
        bg_filter = wave_gradient_bg(sw, sh, scene.bg_color1, scene.bg_color2)
    elif scene.bg_type == "plasma":
        bg_filter = plasma_bg(sw, sh, scene.bg_color1, scene.bg_color2)
    elif scene.bg_type == "aurora":
        bg_filter = aurora_bg(sw, sh)
    else:
        bg_filter = animated_gradient_bg(sw, sh, dur, scene.bg_color1, scene.bg_color2)

    bg_path = out_path.replace(".mp4", "_bg.mp4")
    bg_vf = f"{bg_filter},scale={width}:{height}:flags=bicubic"
    bg_cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={sw}x{sh}:d={dur}:r=24",
        "-vf", bg_vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-t", str(dur), bg_path,
    ]

    try:
        result = subprocess.run(bg_cmd, capture_output=True, text=True, timeout=45)
        if result.returncode != 0:
            return _render_fallback(scene, width, height, dur, out_path)
    except Exception:
        return _render_fallback(scene, width, height, dur, out_path)

    overlay_parts = []

    if scene.vignette > 0:
        overlay_parts.append(vignette_filter(scene.vignette))

    if scene.film_grain_amount > 0:
        overlay_parts.append(film_grain(scene.film_grain_amount))

    grade_map = {
        "cinematic": color_grade_cinematic, "warm": color_grade_warm,
        "cool": color_grade_cool, "neon": color_grade_neon, "vintage": color_grade_vintage,
    }
    if scene.color_grade and scene.color_grade in grade_map:
        overlay_parts.append(grade_map[scene.color_grade]())

    if scene.letterbox_ratio > 0:
        overlay_parts.append(letterbox(width, height, scene.letterbox_ratio))

    if scene.corner_accent_color:
        overlay_parts.append(corner_accents(width, height, scene.corner_accent_color))

    if scene.border_color:
        overlay_parts.append(animated_border(width, height, scene.border_color))

    if scene.breathing:
        overlay_parts.append(breathing_brightness())

    for eff in scene.effects:
        overlay_parts.append(eff)

    for te in scene.texts:
        text_parts = _build_text_filter(te, dur)
        overlay_parts.extend(text_parts)

    if scene.show_progress:
        overlay_parts.append(progress_bar(width, height, scene.progress_color))

    if not overlay_parts:
        os.rename(bg_path, out_path)
        return out_path

    vf = ",".join(overlay_parts)
    final_cmd = [
        "ffmpeg", "-y", "-i", bg_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-t", str(dur), out_path,
    ]

    try:
        result = subprocess.run(final_cmd, capture_output=True, text=True, timeout=60)
        _safe_remove(bg_path)
        if result.returncode != 0:
            return _render_fallback(scene, width, height, dur, out_path)
        return out_path
    except Exception:
        _safe_remove(bg_path)
        return _render_fallback(scene, width, height, dur, out_path)


def _render_single_pass(scene: SceneConfig, width: int, height: int, dur: float, out_path: str) -> Optional[str]:
    bg_color = scene.bg_color1
    vf_parts = []

    if scene.vignette > 0:
        vf_parts.append(vignette_filter(scene.vignette))

    if scene.film_grain_amount > 0:
        vf_parts.append(film_grain(scene.film_grain_amount))

    grade_map = {
        "cinematic": color_grade_cinematic, "warm": color_grade_warm,
        "cool": color_grade_cool, "neon": color_grade_neon, "vintage": color_grade_vintage,
    }
    if scene.color_grade and scene.color_grade in grade_map:
        vf_parts.append(grade_map[scene.color_grade]())

    if scene.letterbox_ratio > 0:
        vf_parts.append(letterbox(width, height, scene.letterbox_ratio))
    if scene.corner_accent_color:
        vf_parts.append(corner_accents(width, height, scene.corner_accent_color))
    if scene.border_color:
        vf_parts.append(animated_border(width, height, scene.border_color))
    if scene.breathing:
        vf_parts.append(breathing_brightness())
    for eff in scene.effects:
        vf_parts.append(eff)
    for te in scene.texts:
        text_parts = _build_text_filter(te, dur)
        vf_parts.extend(text_parts)
    if scene.show_progress:
        vf_parts.append(progress_bar(width, height, scene.progress_color))

    vf = ",".join(vf_parts) if vf_parts else "null"

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={bg_color}:s={width}x{height}:d={dur}:r=30",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-t", str(dur), out_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return None
        return out_path
    except Exception:
        return None


def _render_fallback(scene: SceneConfig, width: int, height: int, dur: float, out_path: str) -> Optional[str]:
    bg_color = scene.bg_color1

    vf_parts = []
    if scene.vignette > 0:
        vf_parts.append(vignette_filter(scene.vignette))
    for te in scene.texts:
        text_parts = _build_text_filter(te, dur)
        vf_parts.extend(text_parts)

    vf = ",".join(vf_parts) if vf_parts else "null"

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={bg_color}:s={width}x{height}:d={dur}:r=30",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-t", str(dur), out_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        return out_path
    except Exception:
        return None


def _safe_remove(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def composite_scenes(scene_paths: List[str], output_path: str, transition: str = "fadeblack",
                     transition_dur: float = 0.5, audio_path: Optional[str] = None) -> bool:
    if not scene_paths:
        return False

    if len(scene_paths) == 1:
        import shutil
        shutil.copy2(scene_paths[0], output_path)
        return True

    inputs = []
    for sp in scene_paths:
        inputs.extend(["-i", sp])

    durations = []
    for sp in scene_paths:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", sp],
            capture_output=True, text=True, timeout=10
        )
        try:
            durations.append(float(probe.stdout.strip()))
        except ValueError:
            durations.append(3.0)

    fc_parts = []
    current_label = "[0:v]"

    for i in range(1, len(scene_paths)):
        next_label = f"[{i}:v]"
        out_label = f"[v{i}]" if i < len(scene_paths) - 1 else "[vout]"

        offset = sum(durations[:i]) - transition_dur * i

        fc_parts.append(
            f"{current_label}{next_label}xfade=transition={transition}:duration={transition_dur}:offset={offset:.2f}{out_label}"
        )
        current_label = out_label

    filter_complex = ";".join(fc_parts)

    cmd = ["ffmpeg", "-y"] + inputs
    cmd += ["-filter_complex", filter_complex]
    cmd += ["-map", "[vout]"]

    if audio_path and os.path.isfile(audio_path):
        cmd += ["-i", audio_path, "-map", f"{len(scene_paths)}:a", "-c:a", "aac", "-b:a", "128k", "-shortest"]

    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return result.returncode == 0
    except Exception:
        return False


def cleanup_temp(scene_paths: List[str]):
    for p in scene_paths:
        _safe_remove(p)
