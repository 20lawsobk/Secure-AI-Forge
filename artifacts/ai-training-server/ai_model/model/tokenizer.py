from __future__ import annotations
from typing import List

CONTROL_TOKENS = [
    "<PLATFORM_TIKTOK>",
    "<PLATFORM_INSTAGRAM>",
    "<PLATFORM_YOUTUBE>",
    "<PLATFORM_FACEBOOK>",
    "<PLATFORM_TWITTER>",
    "<PLATFORM_LINKEDIN>",
    "<PLATFORM_GOOGLE_BUSINESS>",
    "<PLATFORM_THREADS>",
    "<GOAL_GROWTH>",
    "<GOAL_CONVERSION>",
    "<GOAL_NURTURE>",
    "<STAGE_HOOK>",
    "<STAGE_BODY>",
    "<STAGE_CTA>",
    "<TONE_EDGY>",
    "<TONE_PLAYFUL>",
    "<TONE_SERIOUS>",
    "<BRAND_A>",
    "<BRAND_B>",
]


class SimpleTokenizer:
    def __init__(self):
        self.special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"] + CONTROL_TOKENS
        self.vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}
        self.next_id = len(self.vocab)
        self._frozen = False

    @property
    def vocab_size(self):
        return self.next_id

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def encode(self, text: str):
        tokens = text.split()
        ids = []
        unk_id = self.vocab["<UNK>"]
        for t in tokens:
            if t in self.vocab:
                ids.append(self.vocab[t])
            elif self._frozen:
                ids.append(unk_id)
            else:
                self.vocab[t] = self.next_id
                self.inv_vocab[self.next_id] = t
                self.next_id += 1
                ids.append(self.vocab[t])
        return type("Enc", (), {"ids": ids})

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.inv_vocab.get(i, "<UNK>") for i in ids)

    def token_to_id(self, tok: str) -> int:
        return self.vocab.get(tok, self.vocab["<UNK>"])
