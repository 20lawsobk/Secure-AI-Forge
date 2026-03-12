import numpy as np
from typing import Dict
from ai_model.gpu.hyper_core import HyperGPU


class AudioEncoder:
    def __init__(self, gpu: HyperGPU, embed_dim: int = 128, n_frames: int = 256):
        self.gpu = gpu
        self.embed_dim = embed_dim
        self.n_frames = n_frames

    def encode(self, audio_waveform: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        T_audio = self.n_frames
        D = self.embed_dim

        n_samples = audio_waveform.shape[0] if audio_waveform.ndim == 1 else audio_waveform.shape[0]
        hop = max(1, n_samples // T_audio)

        energy_curve = np.zeros(T_audio, dtype=np.float32)
        for i in range(T_audio):
            start = i * hop
            end = min(start + hop, n_samples)
            chunk = audio_waveform[start:end]
            if chunk.ndim > 1:
                chunk = chunk.mean(axis=-1)
            energy_curve[i] = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0

        section_labels = np.zeros(T_audio, dtype=np.int32)
        if T_audio >= 4:
            q = T_audio // 4
            section_labels[0:q] = 0
            section_labels[q:2*q] = 1
            section_labels[2*q:3*q] = 2
            section_labels[3*q:] = 3

        raw_features = np.column_stack([
            energy_curve,
            np.linspace(0, 1, T_audio, dtype=np.float32),
        ])
        W_proj = np.random.randn(raw_features.shape[1], D).astype(np.float32) * 0.02
        time_embeddings = self.gpu.gemm(raw_features, W_proj)

        return {
            "time_embeddings": time_embeddings,
            "section_labels": section_labels,
            "energy_curve": energy_curve,
        }
