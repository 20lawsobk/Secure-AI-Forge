"""
MaxBooster Storage Client
Connects to the storage server via its Redis-like HTTP exec API.
Falls back gracefully if the storage server is unreachable.
"""

import os
import json
import time
import logging
import threading
import hashlib
import pickle
from typing import Any, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger("storage_client")

STORAGE_HTTP_URL = os.getenv("STORAGE_HTTP_URL", "")
STORAGE_BEARER_TOKEN = os.getenv("STORAGE_BEARER_TOKEN", "")
STORAGE_INSTANCE = os.getenv("STORAGE_INSTANCE", "max-booster-training")

KEY_PREFIX = "mb:"


class StorageClient:
    """
    Redis-like storage client using the MaxBooster storage server HTTP exec API.
    All keys are namespaced under 'mb:' to avoid collisions with other services.
    Falls back to an in-process dict if the storage server is unavailable.
    """

    def __init__(self):
        self._url = STORAGE_HTTP_URL
        self._token = STORAGE_BEARER_TOKEN
        self._available: Optional[bool] = None
        self._lock = threading.Lock()
        self._fallback: dict[str, Any] = {}
        self._check_thread = threading.Thread(
            target=self._periodic_health_check, daemon=True
        )
        self._check_thread.start()

    def _exec(self, *args) -> Any:
        """Execute a Redis command against the storage server."""
        if not self._url or not self._token:
            return None

        payload = json.dumps({"command": list(args)}).encode("utf-8")
        req = Request(
            self._url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8").strip()
                if not body:
                    return None
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    return body
        except (URLError, HTTPError, Exception) as e:
            logger.debug(f"[Storage] exec failed: {e}")
            return None

    def _ns(self, key: str) -> str:
        return f"{KEY_PREFIX}{key}"

    def ping(self) -> bool:
        result = self._exec("PING")
        return result == "PONG" or result == {"result": "PONG"} or str(result).upper() == "PONG"

    def _periodic_health_check(self):
        while True:
            try:
                ok = self.ping()
                if ok != self._available:
                    self._available = ok
                    status = "ONLINE" if ok else "OFFLINE"
                    logger.info(f"[Storage] Server is now {status}")
            except Exception:
                self._available = False
            time.sleep(30)

    @property
    def is_available(self) -> bool:
        if self._available is None:
            self._available = self.ping()
        return bool(self._available)

    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        serialized = json.dumps(value) if not isinstance(value, str) else value
        ns_key = self._ns(key)
        if self.is_available:
            args = ["SET", ns_key, serialized]
            if ex:
                args += ["EX", str(ex)]
            result = self._exec(*args)
            if result is not None:
                return True
        with self._lock:
            self._fallback[ns_key] = value
        return False

    def get(self, key: str) -> Optional[Any]:
        ns_key = self._ns(key)
        if self.is_available:
            result = self._exec("GET", ns_key)
            if result is not None:
                try:
                    return json.loads(result) if isinstance(result, str) else result
                except (json.JSONDecodeError, TypeError):
                    return result
        with self._lock:
            return self._fallback.get(ns_key)

    def delete(self, *keys: str) -> int:
        ns_keys = [self._ns(k) for k in keys]
        if self.is_available:
            result = self._exec("DEL", *ns_keys)
            return int(result) if result else 0
        with self._lock:
            removed = sum(1 for k in ns_keys if self._fallback.pop(k, None) is not None)
        return removed

    def exists(self, key: str) -> bool:
        ns_key = self._ns(key)
        if self.is_available:
            result = self._exec("EXISTS", ns_key)
            return bool(result)
        with self._lock:
            return ns_key in self._fallback

    def lpush(self, key: str, *values: Any) -> int:
        ns_key = self._ns(key)
        serialized = [json.dumps(v) if not isinstance(v, str) else v for v in values]
        if self.is_available:
            result = self._exec("LPUSH", ns_key, *serialized)
            return int(result) if result else 0
        with self._lock:
            lst = self._fallback.setdefault(ns_key, [])
            for v in reversed(values):
                lst.insert(0, v)
        return len(self._fallback[ns_key])

    def lrange(self, key: str, start: int, stop: int) -> list:
        ns_key = self._ns(key)
        if self.is_available:
            result = self._exec("LRANGE", ns_key, str(start), str(stop))
            if result and isinstance(result, list):
                parsed = []
                for item in result:
                    try:
                        parsed.append(json.loads(item) if isinstance(item, str) else item)
                    except (json.JSONDecodeError, TypeError):
                        parsed.append(item)
                return parsed
        with self._lock:
            lst = self._fallback.get(ns_key, [])
            end = stop + 1 if stop != -1 else None
            return lst[start:end]

    def ltrim(self, key: str, start: int, stop: int):
        ns_key = self._ns(key)
        if self.is_available:
            self._exec("LTRIM", ns_key, str(start), str(stop))
        else:
            with self._lock:
                lst = self._fallback.get(ns_key, [])
                self._fallback[ns_key] = lst[start:stop + 1]

    def hset(self, key: str, field: str, value: Any) -> int:
        ns_key = self._ns(key)
        serialized = json.dumps(value) if not isinstance(value, str) else value
        if self.is_available:
            result = self._exec("HSET", ns_key, field, serialized)
            return int(result) if result else 0
        with self._lock:
            d = self._fallback.setdefault(ns_key, {})
            d[field] = value
        return 1

    def hget(self, key: str, field: str) -> Optional[Any]:
        ns_key = self._ns(key)
        if self.is_available:
            result = self._exec("HGET", ns_key, field)
            if result is not None:
                try:
                    return json.loads(result) if isinstance(result, str) else result
                except (json.JSONDecodeError, TypeError):
                    return result
        with self._lock:
            return self._fallback.get(ns_key, {}).get(field)

    def hgetall(self, key: str) -> dict:
        ns_key = self._ns(key)
        if self.is_available:
            result = self._exec("HGETALL", ns_key)
            if result and isinstance(result, dict):
                parsed = {}
                for k, v in result.items():
                    try:
                        parsed[k] = json.loads(v) if isinstance(v, str) else v
                    except (json.JSONDecodeError, TypeError):
                        parsed[k] = v
                return parsed
        with self._lock:
            return dict(self._fallback.get(ns_key, {}))

    def incr(self, key: str) -> int:
        ns_key = self._ns(key)
        if self.is_available:
            result = self._exec("INCR", ns_key)
            return int(result) if result else 1
        with self._lock:
            val = int(self._fallback.get(ns_key, 0)) + 1
            self._fallback[ns_key] = val
            return val

    def keys(self, pattern: str = "*") -> list[str]:
        ns_pattern = self._ns(pattern)
        if self.is_available:
            result = self._exec("KEYS", ns_pattern)
            if result and isinstance(result, list):
                prefix_len = len(KEY_PREFIX)
                return [k[prefix_len:] if k.startswith(KEY_PREFIX) else k for k in result]
        with self._lock:
            prefix_len = len(KEY_PREFIX)
            return [k[prefix_len:] for k in self._fallback if k.startswith(KEY_PREFIX)]

    def expire(self, key: str, seconds: int):
        ns_key = self._ns(key)
        if self.is_available:
            self._exec("EXPIRE", ns_key, str(seconds))

    def status(self) -> dict:
        return {
            "instance": STORAGE_INSTANCE,
            "url_configured": bool(self._url),
            "available": self.is_available,
            "fallback_keys": len(self._fallback),
        }


class DatasetStreamClient:
    """
    Streams datasets from the 7TB storage pool for training.
    Datasets are stored in storage as chunked lists under 'dataset:<name>:chunks'.
    """

    def __init__(self, storage: StorageClient):
        self.storage = storage

    def list_datasets(self) -> list[dict]:
        keys = self.storage.keys("dataset:*:meta")
        datasets = []
        for key in keys:
            meta = self.storage.get(key)
            if meta:
                datasets.append(meta)
        return datasets

    def get_dataset_meta(self, name: str) -> Optional[dict]:
        return self.storage.get(f"dataset:{name}:meta")

    def register_dataset(self, name: str, description: str, size_bytes: int,
                         num_chunks: int, content_type: str = "text"):
        meta = {
            "name": name,
            "description": description,
            "size_bytes": size_bytes,
            "num_chunks": num_chunks,
            "content_type": content_type,
            "registered_at": time.time(),
        }
        self.storage.set(f"dataset:{name}:meta", meta)
        logger.info(f"[Dataset] Registered '{name}' ({size_bytes / 1e9:.2f} GB, {num_chunks} chunks)")

    def stream_chunk(self, name: str, chunk_idx: int) -> Optional[list]:
        chunk = self.storage.get(f"dataset:{name}:chunk:{chunk_idx}")
        if chunk is not None:
            logger.debug(f"[Dataset] Streamed chunk {chunk_idx} from '{name}'")
        return chunk

    def write_chunk(self, name: str, chunk_idx: int, data: list):
        self.storage.set(f"dataset:{name}:chunk:{chunk_idx}", data)

    def stream_all_chunks(self, name: str, max_chunks: Optional[int] = None):
        meta = self.get_dataset_meta(name)
        if not meta:
            logger.warning(f"[Dataset] Dataset '{name}' not found in storage")
            return
        total = meta.get("num_chunks", 0)
        limit = min(total, max_chunks) if max_chunks else total
        for i in range(limit):
            chunk = self.stream_chunk(name, i)
            if chunk is not None:
                yield chunk


class ModelCheckpointClient:
    """
    Saves and loads model weight checkpoints to/from storage.
    Checkpoints are stored as JSON-serialized state under 'checkpoint:<model_id>'.
    """

    def __init__(self, storage: StorageClient):
        self.storage = storage

    def save_checkpoint(self, model_id: str, state: dict, metadata: Optional[dict] = None) -> bool:
        checkpoint = {
            "model_id": model_id,
            "saved_at": time.time(),
            "metadata": metadata or {},
            "state_hash": hashlib.sha256(
                json.dumps(state, default=str, sort_keys=True).encode()
            ).hexdigest()[:16],
        }
        ok1 = self.storage.set(f"checkpoint:{model_id}:meta", checkpoint)
        ok2 = self.storage.set(f"checkpoint:{model_id}:state", state)
        self.storage.lpush("checkpoint:history", {
            "model_id": model_id,
            "saved_at": checkpoint["saved_at"],
            "hash": checkpoint["state_hash"],
        })
        self.storage.ltrim("checkpoint:history", 0, 49)
        logger.info(f"[Checkpoint] Saved '{model_id}' (hash: {checkpoint['state_hash']})")
        return ok1 or ok2

    def load_checkpoint(self, model_id: str) -> Optional[dict]:
        state = self.storage.get(f"checkpoint:{model_id}:state")
        if state:
            logger.info(f"[Checkpoint] Loaded '{model_id}' from storage")
        return state

    def list_checkpoints(self) -> list[dict]:
        return self.storage.lrange("checkpoint:history", 0, 49)

    def get_checkpoint_meta(self, model_id: str) -> Optional[dict]:
        return self.storage.get(f"checkpoint:{model_id}:meta")

    def delete_checkpoint(self, model_id: str) -> bool:
        self.storage.delete(f"checkpoint:{model_id}:meta", f"checkpoint:{model_id}:state")
        logger.info(f"[Checkpoint] Deleted checkpoint '{model_id}'")
        return True


class CurriculumStateClient:
    """
    Stores per-user curriculum training state in storage.
    The autopilot writes engagement signals here; the trainer reads them on next run.
    """

    def __init__(self, storage: StorageClient):
        self.storage = storage

    def record_feedback(self, user_id: str, platform: str, engagement_rate: float,
                        content_type: str, style_tags: list[str]):
        entry = {
            "user_id": user_id,
            "platform": platform,
            "engagement_rate": engagement_rate,
            "content_type": content_type,
            "style_tags": style_tags,
            "timestamp": time.time(),
        }
        self.storage.lpush(f"curriculum:{user_id}:feedback", entry)
        self.storage.ltrim(f"curriculum:{user_id}:feedback", 0, 199)
        self.storage.hset(f"curriculum:{user_id}:stats", "last_feedback_at", time.time())
        self.storage.hset(f"curriculum:{user_id}:stats", "total_signals",
                          self.storage.incr(f"curriculum:{user_id}:signal_count"))

    def get_user_curriculum(self, user_id: str, limit: int = 50) -> list[dict]:
        return self.storage.lrange(f"curriculum:{user_id}:feedback", 0, limit - 1)

    def get_top_performers(self, user_id: str, platform: Optional[str] = None,
                           top_n: int = 10) -> list[dict]:
        all_feedback = self.get_user_curriculum(user_id, limit=200)
        if platform:
            all_feedback = [f for f in all_feedback if f.get("platform") == platform]
        all_feedback.sort(key=lambda x: x.get("engagement_rate", 0), reverse=True)
        return all_feedback[:top_n]

    def save_user_model_state(self, user_id: str, state: dict):
        self.storage.set(f"user:{user_id}:model_state", state, ex=86400 * 30)

    def load_user_model_state(self, user_id: str) -> Optional[dict]:
        return self.storage.get(f"user:{user_id}:model_state")

    def get_user_stats(self, user_id: str) -> dict:
        return self.storage.hgetall(f"curriculum:{user_id}:stats")


_storage_client: Optional[StorageClient] = None
_dataset_client: Optional[DatasetStreamClient] = None
_checkpoint_client: Optional[ModelCheckpointClient] = None
_curriculum_client: Optional[CurriculumStateClient] = None


def get_storage() -> StorageClient:
    global _storage_client
    if _storage_client is None:
        _storage_client = StorageClient()
    return _storage_client


def get_dataset_client() -> DatasetStreamClient:
    global _dataset_client
    if _dataset_client is None:
        _dataset_client = DatasetStreamClient(get_storage())
    return _dataset_client


def get_checkpoint_client() -> ModelCheckpointClient:
    global _checkpoint_client
    if _checkpoint_client is None:
        _checkpoint_client = ModelCheckpointClient(get_storage())
    return _checkpoint_client


def get_curriculum_client() -> CurriculumStateClient:
    global _curriculum_client
    if _curriculum_client is None:
        _curriculum_client = CurriculumStateClient(get_storage())
    return _curriculum_client
