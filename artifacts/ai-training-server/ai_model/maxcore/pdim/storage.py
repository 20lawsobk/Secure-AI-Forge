"""Durable PDIM storage — result index + job queue + payload store.

Backed by the project's existing Redis-compatible store (``storage_client``):
the result *index* and the job *queue* live there (so dedup/queueing works
fleet-wide), while large result *payloads* are written to local disk
(cross-platform temp dir by default — never a hard-coded ``D:/`` path). This is
the "infinite storage" tier without adding a raw ``redis`` dependency.

Queue semantics: ``storage_client`` exposes ``lpush``/``lrange``/``ltrim`` but
no atomic ``rpop``, so ``dequeue_batch`` reads the oldest slice and trims it
under a process lock. It is therefore a correct FIFO for a *single consumer*
process (the realistic topology here); multi-process consumers would need an
atomic pop primitive added to the store.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from .config import PDIMConfig


class _InProcStore:
    """Minimal in-process fallback implementing the subset of the store API
    used here, for when ``storage_client`` is unavailable (e.g. isolated tests)."""

    def __init__(self) -> None:
        self._kv: dict[str, Any] = {}
        self._lists: dict[str, list] = {}
        self._lock = threading.Lock()

    def set(self, k: str, v: Any, ex: Optional[int] = None) -> bool:
        with self._lock:
            self._kv[k] = v
        return True

    def get(self, k: str) -> Any:
        with self._lock:
            return self._kv.get(k)

    def delete(self, *keys: str) -> int:
        with self._lock:
            c = 0
            for k in keys:
                c += 1 if self._kv.pop(k, None) is not None else 0
                c += 1 if self._lists.pop(k, None) is not None else 0
            return c

    def lpush(self, k: str, *values: Any) -> int:
        with self._lock:
            lst = self._lists.setdefault(k, [])
            for v in values:
                lst.insert(0, v)
            return len(lst)

    def lrange(self, k: str, start: int, stop: int) -> list:
        with self._lock:
            lst = self._lists.get(k, [])
            end = stop + 1 if stop != -1 else None
            return list(lst[start:end])

    def llen(self, k: str) -> int:
        with self._lock:
            return len(self._lists.get(k, []))

    def ltrim(self, k: str, start: int, stop: int) -> None:
        with self._lock:
            lst = self._lists.get(k, [])
            end = stop + 1 if stop != -1 else None
            self._lists[k] = lst[start:end]

    def status(self) -> dict:
        return {"backend": "in-process-fallback"}


def _resolve_store(store):
    if store is not None:
        return store
    try:
        from storage_client import get_storage
        return get_storage()
    except Exception:
        return _InProcStore()


class PDIMStorage:
    def __init__(self, store=None, config: PDIMConfig | None = None):
        self.config = config or PDIMConfig()
        self.store = _resolve_store(store)
        self.ns = self.config.namespace
        self.ttl = self.config.ttl_seconds
        self.base_dir = Path(self.config.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ── keys ──────────────────────────────────────────────────────────────────
    def _rkey(self, h: str) -> str:
        return f"{self.ns}:result:{h}"

    def _qkey(self, queue: str) -> str:
        return f"{self.ns}:queue:{queue}"

    def _payload_path(self, h: str) -> Path:
        return self.base_dir / f"{h}.json"

    # ── dedup hash ─────────────────────────────────────────────────────────────
    @staticmethod
    def make_hash(model_id: str, prompt: str, params: dict, context_sig: str) -> str:
        blob = json.dumps(
            {"m": model_id, "p": prompt, "params": params, "ctx": context_sig},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ── results (disk payload + store index) ───────────────────────────────────
    def store_result(self, h: str, result: dict) -> bool:
        path = self._payload_path(h)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result))
        os.replace(tmp, path)  # atomic publish
        self.store.set(self._rkey(h), str(path), ex=self.ttl)
        return True

    def get_result(self, h: str) -> Optional[dict]:
        indexed = self.store.get(self._rkey(h))
        path = Path(indexed) if indexed else self._payload_path(h)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    # ── job queue ──────────────────────────────────────────────────────────────
    def enqueue_job(self, queue: str, job: dict) -> int:
        return self.store.lpush(self._qkey(queue), job)

    def dequeue_batch(self, queue: str, batch_size: int) -> list[dict]:
        key = self._qkey(queue)
        with self._lock:
            n = self.store.llen(key)
            if n <= 0:
                return []
            take = min(batch_size, n)
            items = self.store.lrange(key, -take, -1)  # oldest `take` items
            if take >= n:
                self.store.delete(key)
            else:
                self.store.ltrim(key, 0, n - take - 1)  # keep the newer head
        items = list(reversed(items))  # FIFO: oldest first
        return [it for it in items if isinstance(it, dict)]

    def queue_len(self, queue: str) -> int:
        return self.store.llen(self._qkey(queue))

    def status(self) -> dict:
        st = self.store.status() if hasattr(self.store, "status") else {}
        return {"namespace": self.ns, "base_dir": str(self.base_dir), "store": st}
