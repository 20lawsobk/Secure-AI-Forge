"""
MaxBooster Storage Client
Connects to the storage server via its Redis-like HTTP exec API.
Falls back gracefully if the storage server is unreachable.

API format: POST /exec  body: {"cmd": "SET", "args": ["key", "value"]}
"""

import os
import json
import time
import logging
import threading
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
    Redis-like storage client for the MaxBooster storage server.
    Uses HTTP exec API: {"cmd": "SET", "args": ["key", "value"]}
    Falls back to an in-process dict if unreachable.
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

    def _exec(self, cmd: str, *args) -> Any:
        """Execute a Redis command via HTTP exec API."""
        if not self._url or not self._token:
            return None

        payload: dict = {"cmd": cmd}
        if args:
            payload["args"] = [str(a) for a in args]

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            self._url,
            data=data,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=8) as resp:
                body = resp.read().decode("utf-8").strip()
                if not body:
                    return None
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    if "error" in parsed:
                        logger.debug(f"[Storage] {cmd} error: {parsed['error']}")
                        return None
                    return parsed.get("result")
                return parsed
        except (URLError, HTTPError, Exception) as e:
            logger.debug(f"[Storage] exec {cmd} failed: {e}")
            return None

    def _ns(self, key: str) -> str:
        """Namespace a key under mb:"""
        if key.startswith(KEY_PREFIX) or key.startswith("mbs:") or key.startswith("mbs_"):
            return key
        return f"{KEY_PREFIX}{key}"

    def ping(self) -> bool:
        result = self._exec("PING")
        return str(result).upper() == "PONG"

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
            if ex:
                result = self._exec("SET", ns_key, serialized, "EX", ex)
            else:
                result = self._exec("SET", ns_key, serialized)
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
        if self.is_available:
            serialized = [json.dumps(v) if not isinstance(v, str) else v for v in values]
            result = self._exec("LPUSH", ns_key, *serialized)
            return int(result) if result else 0
        with self._lock:
            lst = self._fallback.setdefault(ns_key, [])
            for v in reversed(values):
                lst.insert(0, v)
        return len(self._fallback.get(ns_key, []))

    def lrange(self, key: str, start: int, stop: int) -> list:
        ns_key = self._ns(key)
        if self.is_available:
            result = self._exec("LRANGE", ns_key, start, stop)
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

    def llen(self, key: str) -> int:
        ns_key = self._ns(key)
        if self.is_available:
            result = self._exec("LLEN", ns_key)
            return int(result) if result else 0
        with self._lock:
            return len(self._fallback.get(ns_key, []))

    def ltrim(self, key: str, start: int, stop: int):
        ns_key = self._ns(key)
        if self.is_available:
            self._exec("LTRIM", ns_key, start, stop)
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
                return {k: (json.loads(v) if isinstance(v, str) else v)
                        for k, v in result.items()
                        if not (isinstance(v, str) and _safe_json_fail(v))}
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
        if self.is_available:
            result = self._exec("KEYS", pattern)
            if result and isinstance(result, list):
                return result
        with self._lock:
            return list(self._fallback.keys())

    def expire(self, key: str, seconds: int):
        ns_key = self._ns(key)
        if self.is_available:
            self._exec("EXPIRE", ns_key, seconds)

    def type(self, key: str) -> str:
        ns_key = self._ns(key)
        if self.is_available:
            result = self._exec("TYPE", ns_key)
            return str(result) if result else "none"
        return "string"

    def status(self) -> dict:
        return {
            "instance": STORAGE_INSTANCE,
            "url_configured": bool(self._url),
            "available": self.is_available,
            "fallback_keys": len(self._fallback),
        }


def _safe_json_fail(s: str) -> bool:
    try:
        json.loads(s)
        return False
    except Exception:
        return True


class TrainingDataPipeline:
    """
    Pulls the 7TB training dataset from the MaxBooster storage server.
    Reads session metadata from mbs:training:session, streams content
    from mbs:data and mbs:downloads lists, and feeds batches to the trainer.
    """

    SESSION_KEY = "mbs:training:session"
    DATA_KEYS = ["mbs:data", "mbs:downloads", "mbs_training", "mbs_downloads", "mbs:session"]
    STATUS_KEY = "mbs:status"

    def __init__(self, storage: "StorageClient"):
        self.storage = storage
        self._active = False
        self._session: Optional[dict] = None
        self._bytes_pulled: int = 0
        self._batches_processed: int = 0

    def get_session(self) -> Optional[dict]:
        """Read the active training session from storage."""
        items = self.storage.lrange(self.SESSION_KEY, 0, 0)
        if items:
            return items[0] if isinstance(items[0], dict) else None
        return None

    def get_status(self) -> dict:
        """Read the storage status blob."""
        raw = self.storage.get(self.STATUS_KEY)
        if raw:
            try:
                return json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                pass
        return {"state": "unknown", "bytes": 0}

    def acknowledge_session(self, session_id: str):
        """Mark the training session as active in storage."""
        self.storage.set(self.STATUS_KEY, json.dumps({
            "state": "downloading",
            "session_id": session_id,
            "bytes": self._session.get("bytes", 0) if self._session else 0,
            "bytes_pulled": self._bytes_pulled,
            "ts": int(time.time() * 1000),
        }))
        logger.info(f"[Pipeline] Session {session_id} acknowledged — state: downloading")

    def stream_batches(self, batch_size: int = 64):
        """
        Generator — yields training batches pulled from all data keys.
        Each batch is a list of text strings for the model to train on.
        """
        self._session = self.get_session()
        if not self._session:
            logger.warning("[Pipeline] No training session found in storage")
            return

        session_id = self._session.get("id", "unknown")
        total_bytes = self._session.get("bytes", 0)
        logger.info(f"[Pipeline] Starting dataset pull — session {session_id}, "
                    f"size {total_bytes / 1e12:.2f} TB")

        self.acknowledge_session(session_id)
        self._active = True

        for data_key in self.DATA_KEYS:
            if not self._active:
                break
            length = self.storage.llen(data_key)
            if length == 0:
                continue
            logger.info(f"[Pipeline] Pulling {length} records from {data_key}")
            offset = 0
            while offset < length and self._active:
                chunk = self.storage.lrange(data_key, offset, offset + batch_size - 1)
                if not chunk:
                    break
                batch = self._normalize_batch(chunk)
                if batch:
                    self._batches_processed += 1
                    self._bytes_pulled += sum(len(str(t).encode()) for t in batch)
                    yield batch
                offset += batch_size

        self._mark_complete(session_id)

    def _normalize_batch(self, raw_items: list) -> list[str]:
        """Convert raw storage items into text strings for training."""
        texts = []
        for item in raw_items:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                # Flatten dict to text for language model training
                parts = []
                for k, v in item.items():
                    if isinstance(v, str) and len(v) > 2:
                        parts.append(f"{k}: {v}")
                if parts:
                    texts.append(" | ".join(parts))
            elif isinstance(item, list):
                texts.extend(str(x) for x in item if x)
        return [t for t in texts if t and len(t) > 3]

    def _mark_complete(self, session_id: str):
        self.storage.set(self.STATUS_KEY, json.dumps({
            "state": "complete",
            "session_id": session_id,
            "bytes_pulled": self._bytes_pulled,
            "batches_processed": self._batches_processed,
            "ts": int(time.time() * 1000),
        }))
        logger.info(f"[Pipeline] Dataset pull complete — "
                    f"{self._batches_processed} batches, "
                    f"{self._bytes_pulled / 1e6:.1f} MB processed")

    def stop(self):
        self._active = False

    def pipeline_status(self) -> dict:
        return {
            "active": self._active,
            "session": self._session,
            "bytes_pulled": self._bytes_pulled,
            "batches_processed": self._batches_processed,
            "storage_status": self.get_status(),
        }


class DatasetStreamClient:
    """Registers and streams named dataset chunks from storage."""

    def __init__(self, storage: StorageClient):
        self.storage = storage

    def list_datasets(self) -> list[dict]:
        all_keys = self.storage.keys("mb:dataset:*:meta")
        datasets = []
        for key in all_keys:
            meta = self.storage.get(key)
            if meta:
                datasets.append(meta if isinstance(meta, dict) else json.loads(meta))
        return datasets

    def get_dataset_meta(self, name: str) -> Optional[dict]:
        return self.storage.get(f"mb:dataset:{name}:meta")

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
        self.storage.set(f"mb:dataset:{name}:meta", meta)
        logger.info(f"[Dataset] Registered '{name}' ({size_bytes / 1e9:.2f} GB, {num_chunks} chunks)")

    def stream_chunk(self, name: str, chunk_idx: int) -> Optional[list]:
        return self.storage.get(f"mb:dataset:{name}:chunk:{chunk_idx}")

    def write_chunk(self, name: str, chunk_idx: int, data: list):
        self.storage.set(f"mb:dataset:{name}:chunk:{chunk_idx}", data)

    def stream_all_chunks(self, name: str, max_chunks: Optional[int] = None):
        meta = self.get_dataset_meta(name)
        if not meta:
            logger.warning(f"[Dataset] '{name}' not found in storage")
            return
        total = meta.get("num_chunks", 0)
        limit = min(total, max_chunks) if max_chunks else total
        for i in range(limit):
            chunk = self.stream_chunk(name, i)
            if chunk is not None:
                yield chunk


class ModelCheckpointClient:
    """Saves and loads model weight checkpoints to/from storage."""

    def __init__(self, storage: StorageClient):
        self.storage = storage

    def save_checkpoint(self, model_id: str, state: dict, metadata: Optional[dict] = None) -> bool:
        import hashlib
        checkpoint = {
            "model_id": model_id,
            "saved_at": time.time(),
            "metadata": metadata or {},
            "state_hash": hashlib.sha256(
                json.dumps(state, default=str, sort_keys=True).encode()
            ).hexdigest()[:16],
        }
        ok1 = self.storage.set(f"mb:checkpoint:{model_id}:meta", checkpoint)
        ok2 = self.storage.set(f"mb:checkpoint:{model_id}:state", state)
        self.storage.lpush("mb:checkpoint:history", {
            "model_id": model_id,
            "saved_at": checkpoint["saved_at"],
            "hash": checkpoint["state_hash"],
        })
        self.storage.ltrim("mb:checkpoint:history", 0, 49)
        logger.info(f"[Checkpoint] Saved '{model_id}' (hash: {checkpoint['state_hash']})")
        return ok1 or ok2

    def load_checkpoint(self, model_id: str) -> Optional[dict]:
        state = self.storage.get(f"mb:checkpoint:{model_id}:state")
        if state:
            logger.info(f"[Checkpoint] Loaded '{model_id}' from storage")
        return state

    def list_checkpoints(self) -> list[dict]:
        return self.storage.lrange("mb:checkpoint:history", 0, 49)

    def get_checkpoint_meta(self, model_id: str) -> Optional[dict]:
        return self.storage.get(f"mb:checkpoint:{model_id}:meta")

    def delete_checkpoint(self, model_id: str) -> bool:
        self.storage.delete(f"mb:checkpoint:{model_id}:meta", f"mb:checkpoint:{model_id}:state")
        return True


class CurriculumStateClient:
    """Per-user autopilot engagement signals for curriculum-guided training."""

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
        self.storage.lpush(f"mb:curriculum:{user_id}:feedback", entry)
        self.storage.ltrim(f"mb:curriculum:{user_id}:feedback", 0, 199)
        self.storage.hset(f"mb:curriculum:{user_id}:stats", "last_feedback_at", time.time())
        self.storage.incr(f"mb:curriculum:{user_id}:signal_count")

    def get_user_curriculum(self, user_id: str, limit: int = 50) -> list[dict]:
        return self.storage.lrange(f"mb:curriculum:{user_id}:feedback", 0, limit - 1)

    def get_top_performers(self, user_id: str, platform: Optional[str] = None,
                           top_n: int = 10) -> list[dict]:
        all_fb = self.get_user_curriculum(user_id, limit=200)
        if platform:
            all_fb = [f for f in all_fb if f.get("platform") == platform]
        all_fb.sort(key=lambda x: x.get("engagement_rate", 0), reverse=True)
        return all_fb[:top_n]

    def save_user_model_state(self, user_id: str, state: dict):
        self.storage.set(f"mb:user:{user_id}:model_state", state, ex=86400 * 30)

    def load_user_model_state(self, user_id: str) -> Optional[dict]:
        return self.storage.get(f"mb:user:{user_id}:model_state")

    def get_user_stats(self, user_id: str) -> dict:
        return self.storage.hgetall(f"mb:curriculum:{user_id}:stats")


class AdsClient:
    """
    AI Ad System — stores and retrieves ad performance records.
    Identifies peak-performing ad patterns so the model can replicate them.

    Storage keys:
      mb:ads:{user_id}:runs          — list of all ad run records
      mb:ads:{user_id}:peaks         — list of peak performer records (top 20%)
      mb:ads:{user_id}:patterns      — dict of extracted winning patterns
      mb:ads:{user_id}:stats         — aggregate stats hash
      mb:ads:global:patterns         — cross-user peak patterns for cold-start
    """

    PEAK_ROAS_THRESHOLD = 3.0    # ROAS above this = peak performer
    PEAK_CTR_THRESHOLD  = 2.5    # CTR % above this = peak performer
    PEAK_CPC_CEILING    = 1.50   # CPC below this = efficient

    def __init__(self, storage: StorageClient):
        self.storage = storage

    # ── Record & Store ──────────────────────────────────────────────────────

    def record_ad_run(self, user_id: str, record: dict) -> dict:
        """
        Store a completed ad run. Automatically flags peak performers.
        record fields: platform, ad_type, hook, body, cta, headline,
                       audience_tags, ctr, cpc, roas, conversions,
                       spend, impressions, clicks, run_id (optional)
        """
        record = dict(record)
        record.setdefault("recorded_at", time.time())
        record.setdefault("user_id", user_id)
        record.setdefault("run_id", f"run-{int(time.time() * 1000)}")

        ctr   = float(record.get("ctr", 0))
        roas  = float(record.get("roas", 0))
        cpc   = float(record.get("cpc", 999))

        is_peak = (
            roas >= self.PEAK_ROAS_THRESHOLD or
            ctr  >= self.PEAK_CTR_THRESHOLD  or
            (cpc <= self.PEAK_CPC_CEILING and ctr >= 1.5)
        )
        record["is_peak"] = is_peak

        # Store run
        self.storage.lpush(f"mb:ads:{user_id}:runs", record)
        self.storage.ltrim(f"mb:ads:{user_id}:runs", 0, 499)

        # Store in peaks list
        if is_peak:
            self.storage.lpush(f"mb:ads:{user_id}:peaks", record)
            self.storage.ltrim(f"mb:ads:{user_id}:peaks", 0, 99)
            # Also feed into global cross-user peaks
            self.storage.lpush("mb:ads:global:peaks", record)
            self.storage.ltrim("mb:ads:global:peaks", 0, 199)
            self._update_patterns(user_id, record)

        # Update stats
        self.storage.hset(f"mb:ads:{user_id}:stats", "last_run_at", time.time())
        self.storage.hset(f"mb:ads:{user_id}:stats", "total_runs",
                          self.storage.llen(f"mb:ads:{user_id}:runs"))
        self.storage.hset(f"mb:ads:{user_id}:stats", "total_peaks",
                          self.storage.llen(f"mb:ads:{user_id}:peaks"))

        logger.info(f"[Ads] Recorded run {record['run_id']} for {user_id} "
                    f"(peak={is_peak}, ROAS={roas}, CTR={ctr}%)")
        return record

    def _update_patterns(self, user_id: str, peak: dict):
        """Extract and persist winning pattern signatures from a peak performer."""
        existing = self.storage.get(f"mb:ads:{user_id}:patterns") or {}
        if isinstance(existing, str):
            try:
                existing = json.loads(existing)
            except Exception:
                existing = {}

        platform = peak.get("platform", "unknown")
        ad_type  = peak.get("ad_type", "unknown")
        key = f"{platform}:{ad_type}"

        entry = existing.get(key, {
            "platform": platform,
            "ad_type": ad_type,
            "samples": [],
            "avg_roas": 0.0,
            "avg_ctr": 0.0,
            "top_hooks": [],
            "top_ctas": [],
            "top_audience_tags": [],
        })

        # Add this sample
        entry["samples"].append({
            "hook": peak.get("hook", ""),
            "cta":  peak.get("cta", ""),
            "headline": peak.get("headline", ""),
            "roas": peak.get("roas", 0),
            "ctr":  peak.get("ctr", 0),
            "audience_tags": peak.get("audience_tags", []),
        })
        if len(entry["samples"]) > 20:
            entry["samples"] = entry["samples"][:20]

        # Update aggregates
        samples = entry["samples"]
        entry["avg_roas"] = round(sum(s.get("roas", 0) for s in samples) / len(samples), 2)
        entry["avg_ctr"]  = round(sum(s.get("ctr", 0) for s in samples) / len(samples), 2)
        entry["top_hooks"] = list({s["hook"] for s in samples if s.get("hook")})[:5]
        entry["top_ctas"]  = list({s["cta"] for s in samples if s.get("cta")})[:5]
        all_tags = [t for s in samples for t in s.get("audience_tags", [])]
        entry["top_audience_tags"] = list({t: all_tags.count(t) for t in all_tags}.items())
        entry["top_audience_tags"].sort(key=lambda x: x[1], reverse=True)
        entry["top_audience_tags"] = [t for t, _ in entry["top_audience_tags"][:8]]

        existing[key] = entry
        self.storage.set(f"mb:ads:{user_id}:patterns", json.dumps(existing))

    # ── Read ────────────────────────────────────────────────────────────────

    def get_ad_runs(self, user_id: str, limit: int = 50) -> list[dict]:
        return self.storage.lrange(f"mb:ads:{user_id}:runs", 0, limit - 1)

    def get_peak_performers(self, user_id: str, limit: int = 20,
                            platform: Optional[str] = None) -> list[dict]:
        peaks = self.storage.lrange(f"mb:ads:{user_id}:peaks", 0, limit * 2)
        if platform:
            peaks = [p for p in peaks if p.get("platform") == platform]
        peaks.sort(key=lambda x: float(x.get("roas", 0)) + float(x.get("ctr", 0)) * 0.5,
                   reverse=True)
        return peaks[:limit]

    def get_global_peaks(self, limit: int = 20,
                         platform: Optional[str] = None) -> list[dict]:
        peaks = self.storage.lrange("mb:ads:global:peaks", 0, limit * 2)
        if platform:
            peaks = [p for p in peaks if p.get("platform") == platform]
        peaks.sort(key=lambda x: float(x.get("roas", 0)) + float(x.get("ctr", 0)) * 0.5,
                   reverse=True)
        return peaks[:limit]

    def get_patterns(self, user_id: str) -> dict:
        raw = self.storage.get(f"mb:ads:{user_id}:patterns")
        if raw:
            try:
                return json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                pass
        return {}

    def get_stats(self, user_id: str) -> dict:
        return self.storage.hgetall(f"mb:ads:{user_id}:stats")

    # ── Analysis ─────────────────────────────────────────────────────────────

    def analyse_portfolio(self, user_id: str,
                          platform: Optional[str] = None) -> dict:
        """
        Classify every ad run as: scale | maintain | test | kill
        Based on ROAS, CTR, and CPC benchmarks.
        """
        runs = self.get_ad_runs(user_id, limit=200)
        if platform:
            runs = [r for r in runs if r.get("platform") == platform]

        scale    = []
        maintain = []
        test     = []
        kill     = []

        for run in runs:
            roas = float(run.get("roas", 0))
            ctr  = float(run.get("ctr", 0))
            cpc  = float(run.get("cpc", 999))

            if roas >= self.PEAK_ROAS_THRESHOLD and ctr >= self.PEAK_CTR_THRESHOLD:
                scale.append(run)
            elif roas >= 2.0 or ctr >= 1.5:
                maintain.append(run)
            elif roas == 0 and ctr == 0:
                test.append(run)
            else:
                kill.append(run)

        total_spend       = sum(float(r.get("spend", 0)) for r in runs)
        total_conversions = sum(int(r.get("conversions", 0)) for r in runs)
        avg_roas = (sum(float(r.get("roas", 0)) for r in runs) / len(runs)) if runs else 0

        return {
            "total_runs": len(runs),
            "scale":    [r.get("run_id") for r in scale[:5]],
            "maintain": [r.get("run_id") for r in maintain[:5]],
            "test":     [r.get("run_id") for r in test[:5]],
            "kill":     [r.get("run_id") for r in kill[:5]],
            "scale_count":    len(scale),
            "maintain_count": len(maintain),
            "kill_count":     len(kill),
            "total_spend":       round(total_spend, 2),
            "total_conversions": total_conversions,
            "avg_roas":          round(avg_roas, 2),
            "peak_patterns":     self.get_patterns(user_id),
        }

    def get_winning_formula(self, user_id: str,
                            platform: str, ad_type: str) -> Optional[dict]:
        """Return the extracted peak-performer formula for a platform/ad_type combo."""
        patterns = self.get_patterns(user_id)
        key = f"{platform}:{ad_type}"
        if key in patterns:
            return patterns[key]
        # Fall back to global patterns
        global_peaks = self.get_global_peaks(limit=50, platform=platform)
        if global_peaks:
            return {
                "platform": platform,
                "ad_type": ad_type,
                "top_hooks": list({p.get("hook", "") for p in global_peaks if p.get("hook")})[:5],
                "top_ctas":  list({p.get("cta", "") for p in global_peaks if p.get("cta")})[:5],
                "avg_roas":  round(sum(float(p.get("roas", 0)) for p in global_peaks)
                                   / len(global_peaks), 2),
                "avg_ctr":   round(sum(float(p.get("ctr", 0)) for p in global_peaks)
                                   / len(global_peaks), 2),
                "top_audience_tags": list({t for p in global_peaks
                                          for t in p.get("audience_tags", [])}),
                "source": "global",
            }
        return None


# ─── Singletons ──────────────────────────────────────────────────────────────

_storage_client: Optional[StorageClient] = None
_dataset_client: Optional[DatasetStreamClient] = None
_checkpoint_client: Optional[ModelCheckpointClient] = None
_curriculum_client: Optional[CurriculumStateClient] = None
_pipeline: Optional[TrainingDataPipeline] = None
_ads_client: Optional[AdsClient] = None


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


def get_pipeline() -> TrainingDataPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TrainingDataPipeline(get_storage())
    return _pipeline


def get_ads_client() -> AdsClient:
    global _ads_client
    if _ads_client is None:
        _ads_client = AdsClient(get_storage())
    return _ads_client
