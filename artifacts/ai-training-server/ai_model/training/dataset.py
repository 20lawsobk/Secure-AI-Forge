from __future__ import annotations
import json
import os
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset


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
        if isinstance(item, str):
            return item

        if not isinstance(item, dict):
            return str(item)

        parts = []
        priority = ["text", "content", "script", "caption", "hook", "body", "cta",
                    "headline", "description", "output", "generated", "prompt", "response"]

        for key in priority:
            if key in item and item[key]:
                parts.append(str(item[key]))

        if not parts:
            for v in item.values():
                if isinstance(v, str) and len(v) > 5:
                    parts.append(v)

        return " ".join(parts)

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
