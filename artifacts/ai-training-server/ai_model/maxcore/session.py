"""Sessions, KV stores, and streams for the DigitalGPU layer.

A ``Session`` is a long-lived context (e.g. a generation conversation) that owns
per-session KV stores (for transformer KV-cache or scratch state) and prioritized
streams (ordered work lanes). This is the in-process state layer; durable state
goes through the PDIM store.
"""
from __future__ import annotations

from typing import Any


class KVStore:
    """A simple namespaced key/value store scoped to a session."""

    def __init__(self, name: str):
        self.name = name
        self._d: dict[str, Any] = {}

    def put(self, key: str, value: Any) -> None:
        self._d[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def delete(self, key: str) -> None:
        self._d.pop(key, None)

    def keys(self) -> list[str]:
        return list(self._d.keys())

    def clear(self) -> None:
        self._d.clear()

    def __len__(self) -> int:
        return len(self._d)


class Stream:
    """A prioritized work lane within a session."""

    def __init__(self, stream_id: int, priority: int = 0, policy: dict | None = None):
        self.stream_id = stream_id
        self.priority = priority
        self.policy = policy or {}


class Session:
    def __init__(self, session_id: str, dg=None, policy: dict | None = None):
        self.session_id = session_id
        self.dg = dg
        self.policy = policy or {}
        self._kv: dict[str, KVStore] = {}
        self._streams: list[Stream] = []

    def get_kv_store(self, name: str = "default") -> KVStore:
        if name not in self._kv:
            self._kv[name] = KVStore(name)
        return self._kv[name]

    def create_stream(self, priority: int = 0, policy: dict | None = None) -> Stream:
        stream = Stream(len(self._streams), priority, policy)
        self._streams.append(stream)
        return stream

    def streams(self) -> list[Stream]:
        return sorted(self._streams, key=lambda s: -s.priority)

    def close(self) -> None:
        for kv in self._kv.values():
            kv.clear()
        self._kv.clear()
        self._streams.clear()
