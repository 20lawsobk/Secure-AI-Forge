from __future__ import annotations
import base64
import io
import os
import subprocess
import sys
import time
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
    diffusion_meta: Optional[dict] = None
    # ── Veo-parity render controls ──────────────────────────────────────────
    # camera_motion and negative_prompt are passed into diffusion_meta for
    # downstream conditioning; fps drives the ffmpeg encode frame rate.
    camera_motion: str = ""      # pan_left/zoom_in/static/auto/… (metadata + conditioning)
    negative_prompt: str = ""    # forwarded to diffusion pipeline
    fps: int = 24                # output frame rate (8/16/24/30)
    # ── Visual intent hint fields (from GenerationBrief) ───────────────────
    camera_hint: str = ""        # from brief.camera_motion
    lighting_hint: str = ""      # from brief.lighting (warm/cool/dark/neon/…)
    style_hint: str = ""         # from brief.visual_style / cinematography
    # ── Temporal consistency / SDEdit reference ─────────────────────────────
    reference_b64: Optional[str] = None  # base64 PNG; used as SDEdit prior for diffusion


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
    xs: np.ndarray = np.arange(w, dtype=np.float32).reshape(1, -1)
    ys: np.ndarray = np.arange(h, dtype=np.float32).reshape(-1, 1)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    d = (d / float(d.max())).astype(np.float32)
    arr: np.ndarray = np.zeros((h, w, 3), dtype=np.uint8)
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
    arr: np.ndarray = np.zeros((sh, sw, 3), dtype=np.uint8)
    for ch in range(3):
        arr[:, :, ch] = np.clip(c1[ch] + (c2[ch] - c1[ch]) * field, 0, 255).astype(np.uint8)
    return np.array(Image.fromarray(arr).resize((w, h), Image.BILINEAR))


def _pil_bg_frame(scene: SceneConfig, width: int, height: int) -> tuple:
    """
    Generate a static background PNG using PIL + NumPy.

    Pipeline (each step is additive / non-breaking):
      1. Try MaxCore Diffusion (awareness-conditioned latent DiT) — opt-in via
         scene.diffusion_meta.  Falls through silently on any error.
      2. Procedural gradient / plasma / aurora background (always available).
      3. RCGS retrieval conditioning — grounds frame in real assets.
      4. RTA VRC colour grade applied directly on the NumPy array (before save).
      5. Film grain baked in via NumPy.
      6. Save only once; return (bg_path, png_bytes) for pipe-to-ffmpeg.

    Returns:
      (bg_path: str, png_bytes: bytes | None)
      png_bytes is the raw PNG payload for stdin-pipe, or None on error.
    """
    import numpy as np
    from PIL import Image
    os.makedirs(TEMP_DIR, exist_ok=True)
    bg_path = os.path.join(TEMP_DIR, f"bg_{uuid.uuid4().hex[:8]}.png")

    arr: Optional[np.ndarray] = None

    # ── Step 1: MaxCore Neural Diffusion background ────────────────────────────
    dmeta = getattr(scene, "diffusion_meta", None)
    if dmeta is not None:
        try:
            from .diffusion.maxcore_diffusion import get_diffusion_frame
            _ref = getattr(scene, "reference_b64", None)
            diff_frame = get_diffusion_frame(
                idea=dmeta.get("idea", ""),
                platform=dmeta.get("platform", "tiktok"),
                tone=dmeta.get("tone", "hype"),
                awareness=dmeta.get("awareness", ""),
                width=width,
                height=height,
                context=dmeta,
                reference_b64=_ref,
            )
            if diff_frame is not None and diff_frame.shape == (height, width, 3):
                arr = diff_frame
        except Exception:
            pass

    # ── Step 2: Procedural fallback ────────────────────────────────────────────
    if arr is None:
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

    # ── Step 3: RCGS retrieval conditioning ───────────────────────────────────
    if getattr(scene, "retrieval_conditioned", True):
        try:
            from ai_model.retrieval.rcgs import condition_background
            arr = condition_background(
                arr, width, height,
                brand=(getattr(scene, "brand", "") or None),
            )
        except Exception:
            pass

    # ── Step 4: RTA VRC colour grade on array (BEFORE save — one I/O round-trip)
    _vrc_applied = False
    if scene.color_grade and os.environ.get("RTA_VIDEO_GRADE", "1") != "0":
        try:
            from ai_model import rta as _rta
            arr = _rta.api.grade_video_frame(arr, grade=scene.color_grade)
            _vrc_applied = True
        except Exception as _vrc_err:
            print(f"[RTA] VRC grade failed, using ffmpeg grade: {_vrc_err}")
            _vrc_applied = False

    grain = getattr(scene, "film_grain_amount", 0)
    if grain > 0:
        std = max(1, int(grain * 1.5))
        noise = np.random.randint(-std, std + 1, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ── Step 5: Encode to PNG bytes (in-memory) and save once ─────────────────
    try:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        png_bytes = buf.getvalue()
        # Write to disk only as fallback for ffmpeg -i path
        with open(bg_path, "wb") as _f:
            _f.write(png_bytes)
        return bg_path, png_bytes, _vrc_applied
    except Exception:
        return bg_path, None, _vrc_applied


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
         RTA VRC grade is applied on the array BEFORE saving (single I/O write).
      2. Pipe the PNG bytes to ffmpeg via stdin (avoids temp-file re-read).
         Falls back to disk path if the pipe approach fails.
    Falls back to a solid-colour render if PIL fails.
    """
    _t0 = time.time()
    bg_png: Optional[str] = None
    png_bytes: Optional[bytes] = None
    _vrc_applied = False
    try:
        bg_png, png_bytes, _vrc_applied = _pil_bg_frame(scene, width, height)
    except Exception:
        pass
    _t_bg = time.time() - _t0

    if not bg_png or not os.path.exists(bg_png):
        return _render_fallback(scene, width, height, dur, out_path)

    # Grade timing — grade already applied on array inside _pil_bg_frame
    _t_grade = 0.0

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
    if not _vrc_applied and scene.color_grade and scene.color_grade in grade_map:
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

    # ── Try stdin pipe first (avoids disk re-read; only for in-memory PNG) ──
    # 1920×1080 PNG ≈ 6 MB raw, well within the 50 MB safety threshold.
    _pipe_ok = False
    _encode_t0 = time.time()
    if png_bytes is not None and len(png_bytes) < 50 * 1024 * 1024:
        try:
            pipe_cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-framerate", str(scene.fps),
                "-f", "image2pipe", "-vcodec", "png", "-i", "pipe:0",
                "-vf", vf,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-t", str(dur), out_path,
            ]
            proc = subprocess.Popen(
                pipe_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _stdout, _stderr = proc.communicate(input=png_bytes, timeout=20)
            if proc.returncode == 0:
                _pipe_ok = True
            else:
                print(
                    f"[VideoRender][WARN] ffmpeg pipe render failed (rc={proc.returncode}), "
                    f"falling back to disk: {_stderr[-400:].decode('utf-8', errors='replace')}",
                    file=sys.stderr,
                )
        except Exception as _pipe_exc:
            print(f"[VideoRender][WARN] pipe approach failed ({_pipe_exc}), falling back to disk", file=sys.stderr)

    if not _pipe_ok:
        # Disk fallback
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-framerate", str(scene.fps), "-i", bg_png,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-t", str(dur), out_path,
        ]
        try:
            result = run_ffmpeg(cmd, timeout=20, niceness=0)
            if result.returncode != 0:
                print(
                    f"[VideoRender][ERROR] ffmpeg PIL render failed (rc={result.returncode}):\n{result.stderr[-800:]}",
                    file=sys.stderr,
                )
                _safe_remove(bg_png)
                return _render_fallback(scene, width, height, dur, out_path)
        except Exception as exc:
            print(f"[VideoRender][ERROR] _render_pil_based exception: {exc}", file=sys.stderr)
            _safe_remove(bg_png)
            return _render_fallback(scene, width, height, dur, out_path)

    print(
        f"[VideoRender][Timing] scene bg={_t_bg:.1f}s grade={_t_grade:.1f}s "
        f"encode={time.time() - _encode_t0:.1f}s dur={dur:.1f}s {width}x{height} "
        f"pipe={'yes' if _pipe_ok else 'disk'}",
        flush=True,
    )
    _safe_remove(bg_png)
    return out_path


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
        "-f", "lavfi", "-i", f"color=c={bg_color}:s={width}x{height}:d={dur}:r={scene.fps}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-t", str(dur), out_path,
    ]

    try:
        result = run_ffmpeg(cmd, timeout=20, niceness=0)
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


def _extract_last_frame_b64(clip_path: str) -> Optional[str]:
    """Extract the last frame of a video clip as a base64-encoded PNG.

    Runs ffmpeg with sseof=-0.1 to grab the final frame and pipe it to stdout.
    Never raises — returns None on any error.
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-sseof", "-0.1", "-i", clip_path,
            "-vframes", "1",
            "-f", "image2pipe", "-vcodec", "png", "pipe:1",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        png_data, _ = proc.communicate(timeout=10)
        if proc.returncode != 0 or not png_data:
            return None
        return base64.b64encode(png_data).decode("ascii")
    except Exception:
        return None


# ── Scene compositing ─────────────────────────────────────────────────────────

def _get_clip_duration(clip_path: str) -> float:
    """Get the duration of a video clip via ffprobe. Returns 0.0 on failure."""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                clip_path,
            ],
            capture_output=True, text=True, timeout=8,
        )
        return float(proc.stdout.strip())
    except Exception:
        return 0.0


def _xfade_transition_for_genre(genre: str) -> str:
    """Map genre to an xfade transition type."""
    g = genre.lower()
    if any(k in g for k in ("trap", "drill", "phonk")):
        return "wipeleft"
    if any(k in g for k in ("lofi", "lo_fi", "lo-fi", "jazz", "rnb", "r&b")):
        return "dissolve"
    if "cinematic" in g:
        return "circleopen"
    return "fade"


def _composite_xfade(
    scene_paths: List[str],
    output_path: str,
    transition: str,
    transition_dur: float,
    audio_path: Optional[str],
    genre: str,
) -> bool:
    """
    Build an N-clip xfade chain using filter_complex.

    For N inputs with xfade between each pair:
      [0:v][1:v]xfade=transition=X:duration=D:offset=<end0-D>[v01];
      [v01][2:v]xfade=transition=X:duration=D:offset=<cum_offset>[v012];
      ...
    Falls back to concat on any failure.
    """
    n = len(scene_paths)
    if n < 2:
        return False

    xfade_type = _xfade_transition_for_genre(genre) if genre else "fade"
    td = max(0.1, min(float(transition_dur), 1.0))

    # Get durations for offset calculation
    durations: List[float] = []
    for p in scene_paths:
        d = _get_clip_duration(p)
        if d <= 0.0:
            # Can't compute offsets without durations — fail to concat fallback
            return False
        durations.append(d)

    # Build filter_complex
    # Each xfade offset is cumulative: sum(durations[0..i]) - (i+1)*td
    inputs: List[str] = []
    for p in scene_paths:
        inputs += ["-i", p]

    filter_parts: List[str] = []
    cumulative = 0.0
    prev_label = "[0:v]"

    for i in range(n - 1):
        cumulative += durations[i]
        offset = max(0.0, cumulative - td * (i + 1))
        out_label = f"[v{''.join(str(x) for x in range(i+2))}]" if i < n - 2 else "[vout]"
        filter_parts.append(
            f"{prev_label}[{i+1}:v]xfade=transition={xfade_type}"
            f":duration={td:.3f}:offset={offset:.3f}{out_label}"
        )
        prev_label = out_label

    filter_complex = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y"]
    cmd += inputs
    cmd += ["-filter_complex", filter_complex, "-map", "[vout]"]

    if audio_path and os.path.exists(audio_path):
        cmd += ["-i", audio_path, "-c:a", "aac", "-b:a", "128k", "-shortest"]

    cmd += [
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output_path,
    ]

    try:
        result = run_ffmpeg(cmd, timeout=60, niceness=0)
        return result.returncode == 0
    except Exception:
        return False


def composite_scenes(
    scene_paths: List[str],
    output_path: str,
    transition: str = "fade",
    transition_dur: float = 0.5,
    audio_path: Optional[str] = None,
    genre: str = "",
) -> bool:
    """
    Concatenate rendered scene clips into one MP4.

    For N ≥ 2 clips, attempts xfade filter_complex transitions between each pair
    (transition type selected from genre). Falls back to concat demuxer if xfade
    fails for any reason.
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

    # ── Try xfade filter_complex for N clips ──────────────────────────────────
    xfade_ok = False
    try:
        xfade_ok = _composite_xfade(
            scene_paths, output_path, transition, transition_dur, audio_path, genre
        )
    except Exception:
        xfade_ok = False

    if xfade_ok and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True

    # ── Concat demuxer fallback ───────────────────────────────────────────────
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
                # ultrafast: joining pre-encoded H264 segments is I/O-bound,
                # not compute-bound; ultrafast reduces composite wall-time by
                # 10-20 s compared to "fast" with identical output quality for
                # concat-demuxed streams.
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-shortest", output_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_list,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                output_path,
            ]

        result = run_ffmpeg(cmd, timeout=25, niceness=0)
        _safe_remove(concat_list)
        return result.returncode == 0
    except Exception:
        _safe_remove(concat_list)
        return False
