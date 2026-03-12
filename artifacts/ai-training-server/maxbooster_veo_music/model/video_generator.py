import numpy as np
from typing import Dict
from ai_model.gpu.hyper_core import HyperGPU
from .video_latent_vae import VideoLatentVAE


class VideoGenerator:
    def __init__(self, gpu: HyperGPU, latent_vae: VideoLatentVAE):
        self.gpu = gpu
        self.latent_vae = latent_vae

    def generate_latents(
        self,
        audio_repr: Dict[str, np.ndarray],
        boostsheet_repr: Dict[str, np.ndarray],
        duration_sec: float,
        fps: int = 24,
    ) -> np.ndarray:
        T = max(1, int(duration_sec * fps // 2))
        H, W, C = 8, 8, self.latent_vae.latent_dim

        bs_emb = boostsheet_repr["boostsheet_embedding"]
        audio_emb = audio_repr["time_embeddings"]

        cond_dim = bs_emb.shape[0]
        audio_pooled = audio_emb.mean(axis=0)

        if audio_pooled.shape[0] != cond_dim:
            W_align = np.random.randn(audio_pooled.shape[0], cond_dim).astype(np.float32) * 0.01
            audio_pooled = self.gpu.gemm(audio_pooled[np.newaxis, :], W_align)[0]

        combined = (bs_emb + audio_pooled) * 0.5
        combined = combined[np.newaxis, :]

        latent_flat_dim = H * W * C
        W_gen = np.random.randn(cond_dim, latent_flat_dim).astype(np.float32) * 0.02

        latents_list = []
        for t in range(T):
            phase = np.array([[float(t) / max(T, 1)]], dtype=np.float32)
            W_phase = np.random.randn(1, cond_dim).astype(np.float32) * 0.01
            time_cond = self.gpu.gemm(phase, W_phase)
            frame_cond = combined + time_cond * 0.1
            frame_latent = self.gpu.gemm(frame_cond, W_gen)
            latents_list.append(frame_latent.reshape(H, W, C))

        latents = np.stack(latents_list, axis=0)
        return latents

    def generate_video(
        self,
        audio_repr: Dict[str, np.ndarray],
        boostsheet_repr: Dict[str, np.ndarray],
        duration_sec: float,
        fps: int = 24,
    ) -> np.ndarray:
        latents = self.generate_latents(audio_repr, boostsheet_repr, duration_sec, fps)
        frames = self.latent_vae.decode(latents)
        return frames
