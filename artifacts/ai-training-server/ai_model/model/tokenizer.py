from __future__ import annotations
from typing import Dict, List, Tuple

CONTROL_TOKENS = [
    # ── Platform ──────────────────────────────────────────────────────
    "<PLATFORM_TIKTOK>",
    "<PLATFORM_INSTAGRAM>",
    "<PLATFORM_YOUTUBE>",
    "<PLATFORM_FACEBOOK>",
    "<PLATFORM_TWITTER>",
    "<PLATFORM_LINKEDIN>",
    "<PLATFORM_GOOGLE_BUSINESS>",
    "<PLATFORM_THREADS>",
    "<PLATFORM_META>",
    # ── Goals ─────────────────────────────────────────────────────────
    "<GOAL_GROWTH>",
    "<GOAL_CONVERSION>",
    "<GOAL_NURTURE>",
    "<GOAL_ENGAGEMENT>",
    "<GOAL_AWARENESS>",
    "<GOAL_STREAMS>",
    "<GOAL_FOLLOWERS>",
    "<GOAL_SALES>",
    # ── Script stages ─────────────────────────────────────────────────
    "<STAGE_HOOK>",
    "<STAGE_BODY>",
    "<STAGE_CTA>",
    # ── Tones — must cover every tone used by all agents ──────────────
    "<TONE_EDGY>",
    "<TONE_PLAYFUL>",
    "<TONE_SERIOUS>",
    "<TONE_ENERGETIC>",
    "<TONE_PROFESSIONAL>",
    "<TONE_CASUAL>",
    "<TONE_PROMOTIONAL>",
    "<TONE_INSPIRATIONAL>",
    "<TONE_CHILL>",
    # ── Brand slots ───────────────────────────────────────────────────
    "<BRAND_A>",
    "<BRAND_B>",
    # ── Content types ─────────────────────────────────────────────────
    "<TYPE_AD>",
    "<TYPE_POST>",
    "<TYPE_STORY>",
    "<TYPE_REEL>",
    "<TYPE_VIDEO>",
    "<TYPE_TRACK>",
    # ── Video production stages ───────────────────────────────────────
    "<VIDEO_BUILD>",
    "<VIDEO_DROP>",
    "<VIDEO_OUTRO>",
    # ── Genre ─────────────────────────────────────────────────────────
    "<GENRE_TRAP>",
    "<GENRE_RNB>",
    "<GENRE_POP>",
    "<GENRE_AFROBEATS>",
    "<GENRE_DRILL>",
    "<GENRE_LOFI>",
    "<GENRE_INDIE>",
    "<GENRE_ELECTRONIC>",
    "<GENRE_SOUL>",
    "<GENRE_REGGAETON>",
    "<GENRE_HYPERPOP>",
    "<GENRE_ACOUSTIC>",
    "<GENRE_LATIN>",
    "<GENRE_JAZZ>",
    "<GENRE_HIPHOP>",
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


_END_OF_WORD = "</w>"


class BPETokenizer:
    """Byte-pair-encoding subword tokenizer trained from scratch on the
    in-house corpus. Replaces SimpleTokenizer's whitespace/word-level scheme.

    Why: a word-level vocab treats every surface form as an atomic unit, so
    it can't generalize across "stream"/"streaming"/"streams" or represent
    any word absent from training data except as <UNK>. Subword merges let
    the model share statistical strength across morphological variants and
    guarantee full coverage of any text built from characters seen in
    training (falls back to characters, never to a blanket <UNK> for whole
    words). Standard, well-documented approach (Sennrich et al. 2016 /
    GPT-2 style byte-pair merges), implemented here without an external
    dependency so control tokens stay first-class atomic vocabulary items.
    """

    def __init__(self):
        self.special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"] + CONTROL_TOKENS
        self._control_set = set(self.special_tokens)
        self.vocab: Dict[str, int] = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.inv_vocab: Dict[int, str] = {i: tok for tok, i in self.vocab.items()}
        self.merges: List[Tuple[str, str]] = []
        self._merge_ranks: Dict[Tuple[str, str], int] = {}
        self._frozen = False

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    @staticmethod
    def _pretokenize(text: str) -> List[str]:
        return text.split()

    @staticmethod
    def _word_symbols(word: str) -> Tuple[str, ...]:
        return tuple(list(word) + [_END_OF_WORD])

    @staticmethod
    def _apply_merge(symbols: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        merged = pair[0] + pair[1]
        out: List[str] = []
        i = 0
        n = len(symbols)
        while i < n:
            if i < n - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return tuple(out)

    def train(self, texts: List[str], vocab_size: int = 4000, min_freq: int = 2) -> None:
        """Learn BPE merges from a corpus of raw training texts."""
        word_freq: Dict[Tuple[str, ...], int] = {}
        for text in texts:
            for raw_word in self._pretokenize(text):
                if raw_word in self._control_set:
                    continue
                symbols = self._word_symbols(raw_word)
                word_freq[symbols] = word_freq.get(symbols, 0) + 1

        chars: set[str] = set()
        for symbols in word_freq:
            chars.update(symbols)
        next_id = len(self.vocab)
        for ch in sorted(chars):
            if ch not in self.vocab:
                self.vocab[ch] = next_id
                next_id += 1

        merges: List[Tuple[str, str]] = []
        target_merges = max(0, vocab_size - len(self.vocab))
        current = dict(word_freq)

        for _ in range(target_merges):
            pair_counts: Dict[Tuple[str, str], int] = {}
            for symbols, freq in current.items():
                for a, b in zip(symbols, symbols[1:]):
                    pair_counts[(a, b)] = pair_counts.get((a, b), 0) + freq
            if not pair_counts:
                break
            best_pair, best_count = max(pair_counts.items(), key=lambda kv: kv[1])
            if best_count < min_freq:
                break

            merged_symbol = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            self.vocab[merged_symbol] = next_id
            next_id += 1

            new_current: Dict[Tuple[str, ...], int] = {}
            for symbols, freq in current.items():
                new_symbols = self._apply_merge(symbols, best_pair)
                new_current[new_symbols] = new_current.get(new_symbols, 0) + freq
            current = new_current

        self.merges = merges
        self._merge_ranks = {p: i for i, p in enumerate(merges)}
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        self._frozen = True

    def _bpe_word(self, word: str) -> List[str]:
        symbols = list(self._word_symbols(word))
        if not self._merge_ranks:
            return symbols
        while len(symbols) > 1:
            pairs = list(zip(symbols, symbols[1:]))
            ranked = [(self._merge_ranks[p], idx) for idx, p in enumerate(pairs) if p in self._merge_ranks]
            if not ranked:
                break
            _, best_idx = min(ranked, key=lambda x: x[0])
            pair = (symbols[best_idx], symbols[best_idx + 1])
            symbols = list(self._apply_merge(tuple(symbols), pair))
        return symbols

    def encode(self, text: str):
        ids: List[int] = []
        unk_id = self.vocab["<UNK>"]
        for raw_word in self._pretokenize(text):
            if raw_word in self.vocab and raw_word in self._control_set:
                ids.append(self.vocab[raw_word])
                continue
            for sym in self._bpe_word(raw_word):
                ids.append(self.vocab.get(sym, unk_id))
        return type("Enc", (), {"ids": ids})

    def decode(self, ids: List[int]) -> str:
        words: List[str] = []
        current = ""
        for i in ids:
            piece = self.inv_vocab.get(i, "<UNK>")
            if piece in self._control_set:
                if current:
                    words.append(current)
                    current = ""
                words.append(piece)
                continue
            if piece.endswith(_END_OF_WORD):
                current += piece[: -len(_END_OF_WORD)]
                words.append(current)
                current = ""
            else:
                current += piece
        if current:
            words.append(current)
        return " ".join(words)

    def token_to_id(self, tok: str) -> int:
        return self.vocab.get(tok, self.vocab["<UNK>"])
