from __future__ import annotations
import os
import uuid
import time
import subprocess
from dataclasses import dataclass
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from .scenes import SceneConfig, render_scene, composite_scenes, cleanup_temp
from .templates_v2 import CINEMATIC_TEMPLATES, get_template_list
from .renderer import ASPECT_RATIOS, PLATFORM_RATIOS, VideoRequest, VideoResult, render_video as render_basic

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads", "videos")


@dataclass
class CinematicRequest:
    hook: str = ""
    body: str = ""
    cta: str = ""
    platform: str = "tiktok"
    aspect_ratio: Optional[str] = None
    template: str = "cinematic_promo"
    duration: float = 10.0
    artist_name: str = ""
    quality: str = "cinematic"
    audio_path: Optional[str] = None


@dataclass
class CinematicResult:
    success: bool
    file_path: str = ""
    filename: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    template_name: str = ""
    scenes_rendered: int = 0
    render_time_ms: float = 0.0
    error: str = ""


def render_cinematic(req: CinematicRequest) -> CinematicResult:
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ratio = req.aspect_ratio or PLATFORM_RATIOS.get(req.platform, "9:16")
    width, height = ASPECT_RATIOS.get(ratio, (1080, 1920))

    template = CINEMATIC_TEMPLATES.get(req.template)
    if not template:
        template = CINEMATIC_TEMPLATES.get("cinematic_promo")
        if not template:
            return CinematicResult(success=False, error=f"Template '{req.template}' not found")

    dur = max(6.0, min(req.duration, 60.0))

    scenes = template.build_scenes(
        hook=req.hook, body=req.body, cta=req.cta,
        artist_name=req.artist_name,
        total_duration=dur, width=width, height=height
    )

    if not scenes:
        return CinematicResult(success=False, error="No scenes generated")

    scene_paths: List[str] = []
    render_errors: List[str] = []

    def _render_one(idx_scene):
        idx, scene = idx_scene
        sid = f"{uuid.uuid4().hex[:6]}_{idx}"
        path = render_scene(scene, width, height, sid)
        return idx, path

    with ThreadPoolExecutor(max_workers=min(3, len(scenes))) as executor:
        futures = {executor.submit(_render_one, (i, s)): i for i, s in enumerate(scenes)}
        results_map = {}
        for future in as_completed(futures):
            idx, path = future.result()
            if path:
                results_map[idx] = path
            else:
                render_errors.append(f"Scene {idx} failed to render")

    for i in range(len(scenes)):
        if i in results_map:
            scene_paths.append(results_map[i])

    if not scene_paths:
        return CinematicResult(
            success=False,
            error=f"All scenes failed to render: {'; '.join(render_errors)}"
        )

    filename = f"cinematic_{uuid.uuid4().hex[:12]}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)

    if len(scene_paths) == 1:
        import shutil
        shutil.copy2(scene_paths[0], output_path)
        success = True
    else:
        success = composite_scenes(
            scene_paths=scene_paths,
            output_path=output_path,
            transition=template.transition,
            transition_dur=template.transition_dur,
            audio_path=req.audio_path,
        )

    cleanup_temp(scene_paths)

    if not success:
        if len(scene_paths) > 0 and os.path.exists(scene_paths[0]):
            import shutil
            shutil.copy2(scene_paths[0], output_path)
            success = True
        else:
            return CinematicResult(success=False, error="Failed to composite scenes")

    render_time = (time.time() - start_time) * 1000

    return CinematicResult(
        success=True,
        file_path=output_path,
        filename=filename,
        duration=dur,
        width=width,
        height=height,
        template_name=template.name,
        scenes_rendered=len(scene_paths),
        render_time_ms=render_time,
    )


def render_video_auto(hook: str = "", body: str = "", cta: str = "",
                      platform: str = "tiktok", aspect_ratio: Optional[str] = None,
                      template: str = "cinematic_promo", duration: float = 10.0,
                      artist_name: str = "", quality: str = "cinematic",
                      audio_path: Optional[str] = None,
                      bg_color: Optional[str] = None,
                      text_color: Optional[str] = None,
                      accent_color: Optional[str] = None) -> CinematicResult:
    if quality == "quick" or template in ["promo", "lyric", "announcement", "minimal", "neon"]:
        basic_req = VideoRequest(
            hook=hook, body=body, cta=cta, platform=platform,
            aspect_ratio=aspect_ratio, template=template if template in ["promo", "lyric", "announcement", "minimal", "neon"] else "promo",
            duration=duration, bg_color=bg_color, text_color=text_color,
            accent_color=accent_color, artist_name=artist_name, audio_path=audio_path,
        )
        result = render_basic(basic_req)
        return CinematicResult(
            success=result.success,
            file_path=result.file_path,
            filename=result.filename,
            duration=result.duration,
            width=result.width,
            height=result.height,
            template_name=template,
            scenes_rendered=1,
            render_time_ms=0,
            error=result.error,
        )

    req = CinematicRequest(
        hook=hook, body=body, cta=cta, platform=platform,
        aspect_ratio=aspect_ratio, template=template, duration=duration,
        artist_name=artist_name, quality=quality, audio_path=audio_path,
    )
    return render_cinematic(req)
