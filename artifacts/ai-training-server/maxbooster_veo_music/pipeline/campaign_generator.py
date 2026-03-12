import time
import numpy as np
from typing import Dict, Any

from ..boostsheets.schema import BoostSheet
from ..gpu import gpu
from ..model.audio_encoder import AudioEncoder
from ..model.boostsheet_encoder import BoostSheetEncoder
from ..model.video_latent_vae import VideoLatentVAE
from ..model.video_generator import VideoGenerator
from ..model.platform_heads import PlatformHeads


class CampaignGenerator:
    def __init__(self, gpu_override=None):
        self.gpu = gpu_override or gpu
        self.audio_encoder = AudioEncoder(self.gpu)
        self.boostsheet_encoder = BoostSheetEncoder(self.gpu)
        self.latent_vae = VideoLatentVAE(self.gpu)
        self.video_generator = VideoGenerator(self.gpu, self.latent_vae)
        self.platform_heads = PlatformHeads(self.gpu, self.video_generator)

    def generate_campaign(
        self,
        boostsheet: BoostSheet,
        audio_waveform: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, Any]:
        t0 = time.time()

        audio_repr = self.audio_encoder.encode(audio_waveform, sample_rate)
        bs_repr = self.boostsheet_encoder.encode(boostsheet)

        master_target = None
        for t in boostsheet.targets:
            if t.goal == "full_video" and t.platform == "youtube":
                master_target = t
                break

        master_video = None
        if master_target:
            master_video = self.platform_heads.generate_for_target(
                master_target, audio_repr, bs_repr
            )

        assets = []
        for t in boostsheet.targets:
            if master_target and t is master_target:
                assets.append(master_video)
            else:
                asset = self.platform_heads.generate_for_target(
                    t, audio_repr, bs_repr
                )
                assets.append(asset)

        elapsed = time.time() - t0
        gpu_status = self.gpu.status()

        return {
            "track_id": boostsheet.track_id,
            "artist": boostsheet.artist,
            "title": boostsheet.title,
            "master_video": master_video,
            "assets": assets,
            "generation_time_s": round(elapsed, 2),
            "gpu_ops": gpu_status["total_ops"],
            "gpu_compute_ms": gpu_status["total_compute_ms"],
        }
