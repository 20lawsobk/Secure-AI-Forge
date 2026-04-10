import numpy as np
from typing import Dict, List
from ai_model.gpu.hyper_core import HyperGPU
from ..boostsheets.schema import PlatformTarget
from .video_generator import VideoGenerator

PLATFORM_DEFAULTS = {
    "tiktok":            {"duration": 12.0,  "aspect": "9:16", "fps": 30},
    "reels":             {"duration": 15.0,  "aspect": "9:16", "fps": 30},
    "shorts":            {"duration": 15.0,  "aspect": "9:16", "fps": 30},
    "youtube":           {"duration": 180.0, "aspect": "16:9", "fps": 24},
    "spotify_canvas":    {"duration": 8.0,   "aspect": "9:16", "fps": 24},
    "instagram":         {"duration": 30.0,  "aspect": "1:1",  "fps": 30},
    "instagram_stories": {"duration": 15.0,  "aspect": "9:16", "fps": 30},
    "twitter":           {"duration": 15.0,  "aspect": "16:9", "fps": 30},
    "facebook":          {"duration": 30.0,  "aspect": "16:9", "fps": 30},
    "snapchat":          {"duration": 10.0,  "aspect": "9:16", "fps": 30},
    "pinterest":         {"duration": 15.0,  "aspect": "2:3",  "fps": 30},
    "linkedin":          {"duration": 30.0,  "aspect": "16:9", "fps": 30},
    "threads":           {"duration": 15.0,  "aspect": "1:1",  "fps": 30},
    "twitch":            {"duration": 60.0,  "aspect": "16:9", "fps": 30},
    "triller":           {"duration": 15.0,  "aspect": "9:16", "fps": 30},
    "vevo":              {"duration": 240.0, "aspect": "16:9", "fps": 24},
    "audiomack":         {"duration": 30.0,  "aspect": "1:1",  "fps": 24},
    "soundcloud":        {"duration": 30.0,  "aspect": "1:1",  "fps": 24},
    "apple_music":       {"duration": 8.0,   "aspect": "9:16", "fps": 24},
    "amazon_music":      {"duration": 8.0,   "aspect": "9:16", "fps": 24},
    "tidal":             {"duration": 10.0,  "aspect": "9:16", "fps": 24},
    "deezer":            {"duration": 8.0,   "aspect": "9:16", "fps": 24},
    "pandora":           {"duration": 10.0,  "aspect": "1:1",  "fps": 24},
    "bandcamp":          {"duration": 60.0,  "aspect": "16:9", "fps": 24},
    "website_embed":     {"duration": 30.0,  "aspect": "16:9", "fps": 30},
    "email_campaign":    {"duration": 6.0,   "aspect": "1:1",  "fps": 15},
    "billboard_digital": {"duration": 15.0,  "aspect": "16:9", "fps": 30},
    "live_backdrop":     {"duration": 300.0, "aspect": "16:9", "fps": 24},
}

GOAL_SPECS = {
    "hook_clip": {
        "description": "Short attention-grabbing hook (first 3 seconds critical)",
        "style": "high_energy",
        "fx_intensity": 0.9,
        "text_overlay": True,
        "beat_sync": True,
    },
    "full_video": {
        "description": "Complete music video with intro, verses, chorus, outro",
        "style": "cinematic",
        "fx_intensity": 0.6,
        "text_overlay": False,
        "beat_sync": True,
    },
    "promo_reel": {
        "description": "Promotional reel with release info and artist branding",
        "style": "branded",
        "fx_intensity": 0.7,
        "text_overlay": True,
        "beat_sync": True,
    },
    "promo_clip": {
        "description": "Short promotional clip for feed posts",
        "style": "clean",
        "fx_intensity": 0.5,
        "text_overlay": True,
        "beat_sync": False,
    },
    "loop_visualizer": {
        "description": "Seamless loop with audio-reactive visuals",
        "style": "abstract",
        "fx_intensity": 0.8,
        "text_overlay": False,
        "beat_sync": True,
    },
    "lyric_video": {
        "description": "Animated lyrics synced to the beat with visual effects",
        "style": "typography",
        "fx_intensity": 0.5,
        "text_overlay": True,
        "beat_sync": True,
    },
    "audio_visualizer": {
        "description": "Spectrum/waveform/particle visualizer reacting to audio",
        "style": "reactive",
        "fx_intensity": 0.9,
        "text_overlay": False,
        "beat_sync": True,
    },
    "behind_the_scenes": {
        "description": "BTS-style raw footage aesthetic with overlay text",
        "style": "raw",
        "fx_intensity": 0.3,
        "text_overlay": True,
        "beat_sync": False,
    },
    "teaser_trailer": {
        "description": "Cinematic teaser building anticipation for a release",
        "style": "cinematic",
        "fx_intensity": 0.8,
        "text_overlay": True,
        "beat_sync": True,
    },
    "countdown": {
        "description": "Release countdown timer with visual buildup",
        "style": "hype",
        "fx_intensity": 0.7,
        "text_overlay": True,
        "beat_sync": False,
    },
    "album_unboxing": {
        "description": "Album/EP visual showcase with track listing",
        "style": "editorial",
        "fx_intensity": 0.4,
        "text_overlay": True,
        "beat_sync": False,
    },
    "fan_engagement": {
        "description": "Interactive-style content (polls, challenges, duet bait)",
        "style": "fun",
        "fx_intensity": 0.6,
        "text_overlay": True,
        "beat_sync": True,
    },
    "concert_promo": {
        "description": "Live show/tour promotion with event details",
        "style": "energetic",
        "fx_intensity": 0.8,
        "text_overlay": True,
        "beat_sync": True,
    },
    "merch_showcase": {
        "description": "Merchandise showcase with product visuals",
        "style": "commercial",
        "fx_intensity": 0.4,
        "text_overlay": True,
        "beat_sync": False,
    },
    "collab_announcement": {
        "description": "Collaboration announcement with dual-artist branding",
        "style": "split_screen",
        "fx_intensity": 0.6,
        "text_overlay": True,
        "beat_sync": True,
    },
    "milestone_celebration": {
        "description": "Streaming milestone or achievement celebration",
        "style": "celebratory",
        "fx_intensity": 0.9,
        "text_overlay": True,
        "beat_sync": True,
    },
    "snippet_preview": {
        "description": "Short audio snippet with waveform and release date",
        "style": "minimal",
        "fx_intensity": 0.3,
        "text_overlay": True,
        "beat_sync": True,
    },
    "live_backdrop": {
        "description": "Extended visual backdrop for live performances",
        "style": "ambient",
        "fx_intensity": 0.5,
        "text_overlay": False,
        "beat_sync": True,
    },
    "vertical_mv": {
        "description": "Vertical-format music video optimized for mobile",
        "style": "cinematic",
        "fx_intensity": 0.7,
        "text_overlay": False,
        "beat_sync": True,
    },
    "ad_creative": {
        "description": "Paid ad creative optimized for conversions",
        "style": "commercial",
        "fx_intensity": 0.6,
        "text_overlay": True,
        "beat_sync": True,
    },
    "podcast_visual": {
        "description": "Visual companion for podcast/interview content",
        "style": "editorial",
        "fx_intensity": 0.2,
        "text_overlay": True,
        "beat_sync": False,
    },
    "gif_loop": {
        "description": "Short seamless GIF-style animation for reactions/shares",
        "style": "playful",
        "fx_intensity": 0.5,
        "text_overlay": False,
        "beat_sync": True,
    },
    "email_hero": {
        "description": "Lightweight animated hero for email campaigns",
        "style": "minimal",
        "fx_intensity": 0.3,
        "text_overlay": True,
        "beat_sync": False,
    },
    "billboard_ad": {
        "description": "Digital billboard ad with bold visuals",
        "style": "bold",
        "fx_intensity": 0.7,
        "text_overlay": True,
        "beat_sync": False,
    },
}


class PlatformHeads:
    def __init__(self, gpu: HyperGPU, video_generator: VideoGenerator):
        self.gpu = gpu
        self.video_generator = video_generator

    def generate_for_target(
        self,
        target: PlatformTarget,
        audio_repr: Dict[str, np.ndarray],
        boostsheet_repr: Dict[str, np.ndarray],
    ) -> Dict:
        defaults = PLATFORM_DEFAULTS.get(target.platform, {"duration": 15.0, "aspect": "16:9", "fps": 24})
        duration = target.duration_sec or defaults["duration"]
        aspect = target.aspect_ratio or defaults["aspect"]
        fps = defaults["fps"]

        goal_spec = GOAL_SPECS.get(target.goal, GOAL_SPECS["promo_clip"])

        conditioned_boostsheet = self._apply_goal_conditioning(
            boostsheet_repr, goal_spec, duration, aspect
        )

        frames = self.video_generator.generate_video(
            audio_repr, conditioned_boostsheet, duration_sec=duration, fps=fps
        )

        if goal_spec.get("beat_sync", False):
            frames = self._apply_beat_sync(frames, audio_repr, goal_spec["fx_intensity"])

        return {
            "platform": target.platform,
            "goal": target.goal,
            "goal_description": goal_spec["description"],
            "style": goal_spec["style"],
            "duration_sec": duration,
            "aspect_ratio": aspect,
            "fps": fps,
            "frame_count": frames.shape[0],
            "resolution": f"{frames.shape[2]}x{frames.shape[1]}",
            "has_text_overlay": goal_spec["text_overlay"],
            "beat_synced": goal_spec["beat_sync"],
            "fx_intensity": goal_spec["fx_intensity"],
            "frames": frames,
        }

    def _apply_goal_conditioning(
        self,
        boostsheet_repr: Dict[str, np.ndarray],
        goal_spec: Dict,
        duration: float,
        aspect: str,
    ) -> Dict[str, np.ndarray]:
        conditioned = dict(boostsheet_repr)
        emb = conditioned["boostsheet_embedding"].copy()

        style_hash = hash(goal_spec["style"]) % 1000
        fx = goal_spec["fx_intensity"]
        style_bias = np.ones_like(emb) * (style_hash / 1000.0 - 0.5) * 0.1
        emb = emb + style_bias * fx

        conditioned["boostsheet_embedding"] = emb
        return conditioned

    def _apply_beat_sync(
        self,
        frames: np.ndarray,
        audio_repr: Dict[str, np.ndarray],
        fx_intensity: float,
    ) -> np.ndarray:
        T = frames.shape[0]
        beats = audio_repr.get("beat_positions", np.linspace(0, 1, max(1, T // 15)))
        beat_indices = (beats * (T - 1)).astype(int)
        beat_indices = np.clip(beat_indices, 0, T - 1)

        synced = frames.copy()
        for idx in beat_indices:
            window = min(3, T - idx)
            synced[idx:idx+window] *= (1.0 + fx_intensity * 0.2)

        return np.clip(synced, -1.0, 1.0)

    @staticmethod
    def get_available_platforms() -> Dict[str, Dict]:
        return PLATFORM_DEFAULTS

    @staticmethod
    def get_available_goals() -> Dict[str, Dict]:
        return {k: {kk: vv for kk, vv in v.items()} for k, v in GOAL_SPECS.items()}

    @staticmethod
    def get_recommended_goals(platform: str) -> List[str]:
        recommendations = {
            "tiktok":            ["hook_clip", "lyric_video", "fan_engagement", "audio_visualizer", "snippet_preview", "vertical_mv", "gif_loop"],
            "reels":             ["promo_reel", "lyric_video", "behind_the_scenes", "audio_visualizer", "fan_engagement", "vertical_mv", "snippet_preview"],
            "shorts":            ["hook_clip", "lyric_video", "snippet_preview", "audio_visualizer", "fan_engagement", "vertical_mv"],
            "youtube":           ["full_video", "lyric_video", "audio_visualizer", "teaser_trailer", "behind_the_scenes", "concert_promo", "album_unboxing", "podcast_visual"],
            "spotify_canvas":    ["loop_visualizer", "audio_visualizer", "gif_loop"],
            "instagram":         ["promo_reel", "album_unboxing", "merch_showcase", "milestone_celebration", "collab_announcement", "ad_creative"],
            "instagram_stories": ["teaser_trailer", "countdown", "behind_the_scenes", "snippet_preview", "fan_engagement", "promo_clip"],
            "twitter":           ["promo_clip", "snippet_preview", "teaser_trailer", "milestone_celebration", "collab_announcement"],
            "facebook":          ["promo_clip", "concert_promo", "behind_the_scenes", "ad_creative", "milestone_celebration", "fan_engagement"],
            "snapchat":          ["hook_clip", "behind_the_scenes", "fan_engagement", "snippet_preview", "gif_loop"],
            "pinterest":         ["promo_reel", "album_unboxing", "merch_showcase", "lyric_video"],
            "linkedin":          ["behind_the_scenes", "milestone_celebration", "podcast_visual", "concert_promo"],
            "threads":           ["promo_clip", "snippet_preview", "gif_loop", "milestone_celebration"],
            "twitch":            ["audio_visualizer", "live_backdrop", "fan_engagement", "concert_promo"],
            "triller":           ["hook_clip", "lyric_video", "audio_visualizer", "fan_engagement", "vertical_mv"],
            "vevo":              ["full_video", "lyric_video", "audio_visualizer"],
            "audiomack":         ["audio_visualizer", "loop_visualizer", "promo_clip"],
            "soundcloud":        ["audio_visualizer", "loop_visualizer", "snippet_preview"],
            "apple_music":       ["loop_visualizer", "audio_visualizer"],
            "amazon_music":      ["loop_visualizer", "audio_visualizer"],
            "tidal":             ["loop_visualizer", "audio_visualizer", "full_video"],
            "deezer":            ["loop_visualizer", "audio_visualizer"],
            "pandora":           ["loop_visualizer", "audio_visualizer"],
            "bandcamp":          ["full_video", "audio_visualizer", "behind_the_scenes", "album_unboxing"],
            "website_embed":     ["promo_reel", "teaser_trailer", "audio_visualizer", "full_video"],
            "email_campaign":    ["email_hero", "gif_loop", "snippet_preview"],
            "billboard_digital": ["billboard_ad", "promo_clip"],
            "live_backdrop":     ["live_backdrop", "audio_visualizer"],
        }
        return recommendations.get(platform, ["promo_clip", "hook_clip", "audio_visualizer"])
