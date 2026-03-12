import numpy as np
from ai_model.gpu.hyper_core import HyperGPU


class TextEncoder:
    def __init__(self, gpu: HyperGPU, embed_dim: int = 256):
        self.gpu = gpu
        self.embed_dim = embed_dim

    def encode_text(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.embed_dim, dtype=np.float32)

        tokens = text.lower().split()
        token_hash = np.zeros(self.embed_dim, dtype=np.float32)
        for i, tok in enumerate(tokens):
            h = hash(tok) % (2**31)
            rng = np.random.RandomState(h)
            token_hash += rng.randn(self.embed_dim).astype(np.float32) / max(len(tokens), 1)

        norm = np.linalg.norm(token_hash)
        if norm > 0:
            token_hash = token_hash / norm

        return token_hash
