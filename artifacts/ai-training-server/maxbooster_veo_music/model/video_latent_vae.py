import numpy as np
from ai_model.gpu.hyper_core import HyperGPU


class VideoLatentVAE:
    def __init__(self, gpu: HyperGPU, latent_dim: int = 4):
        self.gpu = gpu
        self.latent_dim = latent_dim

    def encode(self, video_frames: np.ndarray) -> np.ndarray:
        T, H, W, C = video_frames.shape
        T2 = max(1, T // 2)
        H2 = max(1, H // 4)
        W2 = max(1, W // 4)

        flat = video_frames.reshape(T, -1)
        W_enc = np.random.randn(flat.shape[1], self.latent_dim).astype(np.float32) * 0.01

        latent_flat = np.zeros((T, self.latent_dim), dtype=np.float32)
        chunk_size = 16
        for i in range(0, T, chunk_size):
            batch = flat[i:i+chunk_size]
            latent_flat[i:i+chunk_size] = self.gpu.gemm(batch, W_enc)

        latent_downsampled = latent_flat[:T2]
        latents = latent_downsampled.reshape(T2, 1, 1, self.latent_dim)
        latents = np.broadcast_to(latents, (T2, H2, W2, self.latent_dim)).copy()

        return latents

    def decode(self, latents: np.ndarray) -> np.ndarray:
        T2, H2, W2, C = latents.shape
        T = T2 * 2
        H = H2 * 4
        W = W2 * 4

        flat = latents.reshape(T2, -1)
        out_dim = H * W * 3 // T
        out_dim = max(out_dim, 3)

        W_dec = np.random.randn(flat.shape[1], out_dim).astype(np.float32) * 0.01

        decoded_flat = np.zeros((T2, out_dim), dtype=np.float32)
        for i in range(T2):
            decoded_flat[i:i+1] = self.gpu.gemm(flat[i:i+1], W_dec)

        total_pixels = T * H * W * 3
        decoded_raw = decoded_flat.flatten()[:total_pixels]
        if decoded_raw.size < total_pixels:
            decoded_raw = np.resize(decoded_raw, total_pixels)

        frames = np.tanh(decoded_raw.reshape(T, H, W, 3)).astype(np.float32)
        return frames
