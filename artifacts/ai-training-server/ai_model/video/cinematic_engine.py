from __future__ import annotations
import os
import uuid
import time
from dataclasses import dataclass
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .scenes import SceneConfig, render_scene, composite_scenes, cleanup_temp
from .renderer import ASPECT_RATIOS, PLATFORM_RATIOS

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




def render_cinematic_open(
    scenes: List[SceneConfig],
    width: int,
    height: int,
    total_duration: float,
    audio_path: Optional[str] = None,
    transition: str = "fade",
    transition_dur: float = 0.5,
    label: str = "",
) -> CinematicResult:
    """
    Render a list of pre-built SceneConfig objects with no template involvement.
    All visual parameters are already embedded in the scenes.
    """
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not scenes:
        return CinematicResult(success=False, error="No scenes provided")

    dur = max(6.0, min(total_duration, 120.0))

    scene_paths: List[str] = []
    render_errors: List[str] = []

    def _render_one(idx_scene):
        idx, scene = idx_scene
        sid = f"{uuid.uuid4().hex[:6]}_{idx}"
        path = render_scene(scene, width, height, sid)
        return idx, path

    with ThreadPoolExecutor(max_workers=min(3, len(scenes))) as executor:
        futures = {executor.submit(_render_one, (i, s)): i for i, s in enumerate(scenes)}
        results_map: Dict[int, str] = {}
        for future in as_completed(futures):
            idx, path = future.result()
            if path:
                results_map[idx] = path
            else:
                render_errors.append(f"Scene {idx} failed")

    for i in range(len(scenes)):
        if i in results_map:
            scene_paths.append(results_map[i])

    if not scene_paths:
        return CinematicResult(
            success=False,
            error=f"All scenes failed: {'; '.join(render_errors)}",
        )

    filename = f"ai_{uuid.uuid4().hex[:12]}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)

    if len(scene_paths) == 1:
        import shutil
        shutil.copy2(scene_paths[0], output_path)
        success = True
    else:
        success = composite_scenes(
            scene_paths=scene_paths,
            output_path=output_path,
            transition=transition,
            transition_dur=transition_dur,
            audio_path=audio_path,
        )

    cleanup_temp(scene_paths)

    if not success:
        if scene_paths and os.path.exists(scene_paths[0]):
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
        template_name=label or "ai_generated",
        scenes_rendered=len(scene_paths),
        render_time_ms=render_time,
    )


