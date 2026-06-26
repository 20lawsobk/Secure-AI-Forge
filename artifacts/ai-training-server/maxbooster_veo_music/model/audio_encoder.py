import numpy as np
from typing import Any, Dict
from ai_model.gpu.hyper_core import HyperGPU


class AudioEncoder:
    def __init__(self, gpu: HyperGPU, embed_dim: int = 128, n_frames: int = 256):
        self.gpu = gpu
        self.embed_dim = embed_dim
        self.n_frames = n_frames

    def encode(self, audio_waveform: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        T_audio = self.n_frames
        D = self.embed_dim

        n_samples = audio_waveform.shape[0] if audio_waveform.ndim == 1 else audio_waveform.shape[0]
        hop = max(1, n_samples // T_audio)

        energy_curve: np.ndarray = np.zeros(T_audio, dtype=np.float32)
        for i in range(T_audio):
            start = i * hop
            end = min(start + hop, n_samples)
            chunk = audio_waveform[start:end]
            if chunk.ndim > 1:
                chunk = chunk.mean(axis=-1)
            energy_curve[i] = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0

        section_labels: np.ndarray = np.zeros(T_audio, dtype=np.int32)
        beat_positions = np.linspace(0, 1, max(1, T_audio // 15), dtype=np.float32)
        bpm = 120.0
        key = "C"
        mode = "major"
        try:
            from ai_model.audio.audio_analysis import analyze_audio

            timeline = analyze_audio(audio_waveform, sample_rate)
            dur = timeline.duration_sec
            if timeline.sections and dur > 0:
                last = len(timeline.sections) - 1
                for i in range(T_audio):
                    t = (i / T_audio) * dur
                    for si, sec in enumerate(timeline.sections):
                        if sec.start <= t < sec.end:
                            section_labels[i] = si
                            break
                    else:
                        section_labels[i] = last
            positions = timeline.beat_positions_normalized()
            if positions:
                beat_positions = np.asarray(positions, dtype=np.float32)
            bpm = float(timeline.bpm)
            key = timeline.key
            mode = timeline.mode
        except Exception:
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
            "beat_positions": beat_positions,
            "bpm": np.float32(bpm),
            "key": key,
            "mode": mode,
        }
