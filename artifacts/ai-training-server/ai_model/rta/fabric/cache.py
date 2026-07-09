"""Deterministic, bounded node-result cache for the UMRF scheduler.

Keyed by a content hash of (node type + params + upstream digests). Only nodes
that opt in (``params['cacheable'] = True`` and a stable ``params['cache_key']``)
are memoised, so non-deterministic / unique renders are never wrongly reused.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any, Optional


class NodeCache:
    def __init__(self, max_entries: int = 64):
        self._store: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_entries

    @staticmethod
    def make_key(node_type: str, cache_key: str) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(node_type.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(cache_key).encode("utf-8"))
        return h.hexdigest()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
        return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)
