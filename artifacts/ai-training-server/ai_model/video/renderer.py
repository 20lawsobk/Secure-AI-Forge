from __future__ import annotations
import subprocess
import os
import uuid
from dataclasses import dataclass
from typing import Optional

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_PATH_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

ASPECT_RATIOS = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
}

PLATFORM_RATIOS = {
    "tiktok": "9:16",
    "instagram": "1:1",
    "instagram_reels": "9:16",
    "instagram_story": "9:16",
    "youtube": "16:9",
    "youtube_shorts": "9:16",
    "facebook": "1:1",
    "facebook_reels": "9:16",
    "twitter": "16:9",
    "linkedin": "16:9",
    "google_business": "16:9",
    "threads": "1:1",
}

TEMPLATE_STYLES = {
    "promo": {
        "bg_color": "0x1a1a2e",
        "accent_color": "0xe94560",
        "text_color": "0xffffff",
        "hook_size": 64,
        "body_size": 42,
        "cta_size": 48,
        "cta_bg": "0xe94560",
    },
    "lyric": {
        "bg_color": "0x0f0f23",
        "accent_color": "0xffd700",
        "text_color": "0xffffff",
        "hook_size": 56,
        "body_size": 52,
        "cta_size": 40,
        "cta_bg": "0x333366",
    },
    "announcement": {
        "bg_color": "0x16213e",
        "accent_color": "0x0f3460",
        "text_color": "0xe2e2e2",
        "hook_size": 58,
        "body_size": 44,
        "cta_size": 46,
        "cta_bg": "0xe94560",
    },
    "minimal": {
        "bg_color": "0xfafafa",
        "accent_color": "0x333333",
        "text_color": "0x1a1a1a",
        "hook_size": 60,
        "body_size": 40,
        "cta_size": 44,
        "cta_bg": "0x1a1a1a",
    },
    "neon": {
        "bg_color": "0x0d0221",
        "accent_color": "0xff6ec7",
        "text_color": "0x00fff5",
        "hook_size": 62,
        "body_size": 44,
        "cta_size": 46,
        "cta_bg": "0xff6ec7",
    },
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads", "videos")


@dataclass
class VideoRequest:
    hook: str = ""
    body: str = ""
    cta: str = ""
    platform: str = "tiktok"
    aspect_ratio: Optional[str] = None
    template: str = "promo"
    duration: float = 8.0
    bg_color: Optional[str] = None
    text_color: Optional[str] = None
    accent_color: Optional[str] = None
    artist_name: Optional[str] = None
    audio_path: Optional[str] = None


@dataclass
class VideoResult:
    success: bool
    file_path: str = ""
    filename: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    error: str = ""


def _esc(text: str) -> str:
    for ch in ["\\", "'", ":", ";", "[", "]", ","]:
        text = text.replace(ch, "\\" + ch)
    return text


def _wrap(text: str, max_chars: int = 28) -> str:
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def render_video(req: VideoRequest) -> VideoResult:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ratio = req.aspect_ratio or PLATFORM_RATIOS.get(req.platform, "9:16")
    width, height = ASPECT_RATIOS.get(ratio, (1080, 1920))

    style = TEMPLATE_STYLES.get(req.template, TEMPLATE_STYLES["promo"])
    bg = req.bg_color.replace("#", "0x") if req.bg_color else style["bg_color"]
    tc = req.text_color.replace("#", "0x") if req.text_color else style["text_color"]
    ac = req.accent_color.replace("#", "0x") if req.accent_color else style["accent_color"]
    cta_bg = style.get("cta_bg", ac)

    hs = style["hook_size"]
    bs = style["body_size"]
    cs = style["cta_size"]
    if width < 1080:
        s = width / 1080
        hs = int(hs * s)
        bs = int(bs * s)
        cs = int(cs * s)

    dur = max(5.0, min(req.duration, 30.0))
    hook_end = dur * 0.45
    body_start = dur * 0.25
    body_end = dur * 0.75
    cta_start = dur * 0.6

    filename = f"video_{uuid.uuid4().hex[:12]}.mp4"
    out = os.path.join(OUTPUT_DIR, filename)

    mc = max(20, int(width / (bs * 0.6)))

    vf_parts = []

    bar_h = int(height * 0.12)
    vf_parts.append(f"drawbox=x=0:y=0:w={width}:h={bar_h}:color={ac}@0.3:t=fill")
    vf_parts.append(f"drawbox=x=0:y={height - bar_h}:w={width}:h={bar_h}:color={ac}@0.3:t=fill")

    if req.artist_name:
        at = _esc(req.artist_name)
        vf_parts.append(
            f"drawtext=fontfile={FONT_PATH}:text='{at}':fontcolor={ac}:fontsize={int(bs * 0.7)}"
            f":x=(w-text_w)/2:y=h*0.06"
        )

    if req.hook:
        ht = _esc(_wrap(req.hook, mc))
        vf_parts.append(
            f"drawtext=fontfile={FONT_PATH}:text='{ht}':fontcolor={tc}:fontsize={hs}"
            f":x=(w-text_w)/2:y=(h-text_h)/4"
            f":enable='between(t\\,0.3\\,{hook_end:.1f})'"
            f":alpha='if(lt(t\\,0.8)\\,min(1\\,(t-0.3)*2)\\,if(gt(t\\,{hook_end - 0.5:.1f})\\,max(0\\,({hook_end:.1f}-t)*2)\\,1))'"
        )

    if req.body:
        bt = _esc(_wrap(req.body, mc))
        vf_parts.append(
            f"drawtext=fontfile={FONT_PATH_REGULAR}:text='{bt}':fontcolor={tc}:fontsize={bs}"
            f":x=(w-text_w)/2:y=(h-text_h)/2"
            f":enable='between(t\\,{body_start:.1f}\\,{body_end:.1f})'"
            f":alpha='if(lt(t\\,{body_start + 0.5:.1f})\\,min(1\\,(t-{body_start:.1f})*2)\\,if(gt(t\\,{body_end - 0.5:.1f})\\,max(0\\,({body_end:.1f}-t)*2)\\,1))'"
        )

    if req.cta:
        ct = _esc(_wrap(req.cta, mc))
        box_w = int(width * 0.8)
        box_x = int((width - box_w) / 2)
        box_y = int(height * 0.72)
        vf_parts.append(
            f"drawbox=x={box_x}:y={box_y}:w={box_w}:h={cs + 40}:color={cta_bg}@0.85:t=fill"
            f":enable='between(t\\,{cta_start:.1f}\\,{dur:.1f})'"
        )
        vf_parts.append(
            f"drawtext=fontfile={FONT_PATH}:text='{ct}':fontcolor=white:fontsize={cs}"
            f":x=(w-text_w)/2:y=h*0.73"
            f":enable='between(t\\,{cta_start:.1f}\\,{dur:.1f})'"
            f":alpha='if(lt(t\\,{cta_start + 0.3:.1f})\\,min(1\\,(t-{cta_start:.1f})*3)\\,1)'"
        )

    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={bg}:s={width}x{height}:d={dur}:r=30",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-t", str(dur),
        out,
    ]

    if req.audio_path and os.path.isfile(req.audio_path):
        cmd_with_audio = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c={bg}:s={width}x{height}:d={dur}:r=30",
            "-i", req.audio_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", "-t", str(dur),
            out,
        ]
        cmd = cmd_with_audio

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return VideoResult(success=False, error=f"FFmpeg error: {result.stderr[-500:]}")

        return VideoResult(
            success=True, file_path=out, filename=filename,
            duration=dur, width=width, height=height,
        )
    except subprocess.TimeoutExpired:
        return VideoResult(success=False, error="Video rendering timed out")
    except Exception as e:
        return VideoResult(success=False, error=str(e))
