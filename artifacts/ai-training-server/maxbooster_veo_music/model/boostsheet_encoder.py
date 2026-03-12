import numpy as np
from ai_model.gpu.hyper_core import HyperGPU
from .text_encoder import TextEncoder
from ..boostsheets.schema import BoostSheet


class BoostSheetEncoder:
    def __init__(self, gpu: HyperGPU, embed_dim: int = 256):
        self.gpu = gpu
        self.embed_dim = embed_dim
        self.text_encoder = TextEncoder(gpu, embed_dim=embed_dim)

    def encode(self, bs: BoostSheet) -> dict:
        story_vec = self.text_encoder.encode_text(bs.story)
        brand_vec = self.text_encoder.encode_text(bs.brand_notes or "")
        mood_vec = self.text_encoder.encode_text(bs.mood)
        era_vec = self.text_encoder.encode_text(bs.era)

        ref_vecs = [self.text_encoder.encode_text(r) for r in bs.references]
        if ref_vecs:
            ref_mean = np.stack(ref_vecs, axis=0).mean(axis=0)
        else:
            ref_mean = np.zeros_like(story_vec)

        fused = np.concatenate([story_vec, brand_vec, mood_vec, era_vec, ref_mean], axis=0)
        fused = fused[np.newaxis, :]

        W = np.random.randn(fused.shape[1], self.embed_dim).astype(np.float32) * 0.01
        bs_embedding = self.gpu.gemm(fused, W)[0]

        return {
            "boostsheet_embedding": bs_embedding,
            "story_vec": story_vec,
            "brand_vec": brand_vec,
            "mood_vec": mood_vec,
            "era_vec": era_vec,
            "ref_vec": ref_mean,
        }
