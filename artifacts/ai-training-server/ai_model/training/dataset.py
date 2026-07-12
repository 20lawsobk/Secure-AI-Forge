from __future__ import annotations
import json
import os
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

from ai_model.model.tokenizer import CONTROL_TOKENS

# Map metadata fields present in training JSON (platform/goal/genre/tone) onto
# the reserved <CATEGORY_VALUE> control tokens, so the model actually sees
# conditioning signal during training instead of it being dropped on the
# floor. Without this, CONTROL_TOKENS exist in the vocab but are never used
# as training input, so the model can't learn to condition generation on
# platform/goal/genre/tone at inference time.
_FIELD_TO_CATEGORY = {
    "platform": "PLATFORM",
    "goal": "GOAL",
    "genre": "GENRE",
    "tone": "TONE",
}


def _normalize(value: str) -> str:
    return "".join(ch for ch in value.upper() if ch.isalnum())


def _build_category_lookup() -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    for tok in CONTROL_TOKENS:
        inner = tok.strip("<>")
        if "_" not in inner:
            continue
        category, _, value = inner.partition("_")
        lookup.setdefault(category, {})[_normalize(value)] = tok
    return lookup


_CATEGORY_LOOKUP = _build_category_lookup()


def _control_token_for(field: str, value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    category = _FIELD_TO_CATEGORY.get(field)
    if not category:
        return None
    return _CATEGORY_LOOKUP.get(category, {}).get(_normalize(value))


def extract_text_from_item(item: Any) -> str:
    """Turn one raw training-JSON item into a training text string, prefixed
    with any control tokens its metadata maps to. Standalone (not tied to a
    tokenizer) so it can be reused to build a corpus before a tokenizer
    exists — e.g. training the BPE vocab itself.
    """
    if isinstance(item, str):
        return item

    if not isinstance(item, dict):
        return str(item)

    control_tokens: List[str] = []
    for field in ("platform", "goal", "genre", "tone"):
        tok = _control_token_for(field, item.get(field))
        if tok and tok not in control_tokens:
            control_tokens.append(tok)
    if not control_tokens and isinstance(item.get("platforms"), list):
        for p in item["platforms"]:
            tok = _control_token_for("platform", p)
            if tok and tok not in control_tokens:
                control_tokens.append(tok)

    # When hook/body/cta are separate fields, structure them with stage tokens so
    # the model learns to emit <STAGE_HOOK>, <STAGE_BODY>, <STAGE_CTA> delimiters.
    # This makes ScriptAgent.run()'s stage-token parser work at inference time.
    has_structured = any(item.get(k) for k in ("hook", "body", "cta"))
    if has_structured:
        struct_parts: List[str] = []
        if item.get("hook"):
            struct_parts.append(f"<STAGE_HOOK> {item['hook']}")
        if item.get("body"):
            struct_parts.append(f"<STAGE_BODY> {item['body']}")
        if item.get("cta"):
            struct_parts.append(f"<STAGE_CTA> {item['cta']}")
        body = " ".join(struct_parts)
        # Append any top-level "text" or "content" blob that isn't already
        # covered by the structured fields.
        for key in ("text", "content", "script", "caption"):
            if item.get(key):
                val = str(item[key])
                # Skip if the value is mostly a duplicate of the structured parts
                if val not in body:
                    body = body  # keep structured form only — avoids duplication
                break
    else:
        parts: List[str] = []
        priority = ["text", "content", "script", "caption", "hook", "body", "cta",
                    "headline", "description", "output", "generated", "prompt", "response"]

        for key in priority:
            if key in item and item[key]:
                parts.append(str(item[key]))

        if not parts:
            for v in item.values():
                if isinstance(v, str) and len(v) > 5:
                    parts.append(v)

        body = " ".join(parts)

    if control_tokens:
        return " ".join(control_tokens) + " " + body
    return body


class CreativeDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples: List[str] = []
        self._encoded: List[Dict] = []
        self._load(data_path)
        self._pre_encode()

    def _load(self, path: str):
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return

        if not isinstance(data, list):
            data = [data]

        for item in data:
            text = self._extract_text(item)
            if text and text.strip():
                self.samples.append(text.strip())

    def _extract_text(self, item: Any) -> str:
        return extract_text_from_item(item)

    def _pre_encode(self):
        bos_id = self.tokenizer.token_to_id("<BOS>")
        eos_id = self.tokenizer.token_to_id("<EOS>")
        pad_id = self.tokenizer.token_to_id("<PAD>")

        for text in self.samples:
            enc = self.tokenizer.encode(text)
            token_ids = [bos_id] + enc.ids + [eos_id]

            if len(token_ids) > self.max_len:
                token_ids = token_ids[:self.max_len]

            input_ids = token_ids[:-1]
            labels = token_ids[1:]

            pad_len = self.max_len - 1 - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [pad_id] * pad_len
                labels = labels + [pad_id] * pad_len

            self._encoded.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._encoded[idx]
