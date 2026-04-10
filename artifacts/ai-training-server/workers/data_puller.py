"""
MaxBooster AI Training Server — Auto Data Puller

Pulls training data from two sources:
  1. pdim (MaxBooster storage server) — live ad peaks, social posts, analytics, content
  2. Public datasets — Harrison Dataset, Social Media Instruction, MusicBench, etc.

Stores pulled samples under ai_model/training_data/<source>/<timestamp>.json
and pushes a deduplicated snapshot back to pdim for cross-session continuity.
"""

import json
import logging
import time
import threading
import hashlib
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

logger = logging.getLogger("data_puller")

TRAINING_DATA_DIR = Path(__file__).parent.parent / "ai_model" / "training_data"

# pdim key patterns to scan for live MaxBooster data
PDIM_SCAN_PATTERNS = [
    "mbs:data",
    "mbs:downloads",
    "mbs_training",
    "mbs_downloads",
    "mbs:session",
    "mb:ads:*:peaks",
    "mb:social:posts:*",
    "mb:analytics:*",
    "mb:content:*",
    "mb:dataset:*:chunk:*",
]

# Public dataset sources — raw JSON / API endpoints (no AI APIs)
PUBLIC_SOURCES = [
    {
        "id": "harrison_hooks",
        "name": "Harrison Dataset (Hashtag/Hook Patterns)",
        "url": "https://raw.githubusercontent.com/minstone/HARRISON-Dataset/main/data/hashtags.json",
        "format": "json_list",
        "field": None,
        "category": "social",
    },
    {
        "id": "social_media_instruction",
        "name": "Social Media Instruction (HuggingFace)",
        "url": "https://datasets-server.huggingface.co/rows?dataset=Shekswess%2Fsocial-media-instruction&config=default&split=train&offset=0&length=100",
        "format": "huggingface_rows",
        "field": "rows",
        "category": "social",
    },
    {
        "id": "musicbench",
        "name": "MusicBench (Text-Music Pairs)",
        "url": "https://datasets-server.huggingface.co/rows?dataset=MusicBench%2FMusicBench&config=default&split=train&offset=0&length=100",
        "format": "huggingface_rows",
        "field": "rows",
        "category": "audio",
    },
    {
        "id": "music_caps_captions",
        "name": "MusicCaps (Audio Captions)",
        "url": "https://datasets-server.huggingface.co/rows?dataset=google%2Fmusiccaps&config=default&split=train&offset=0&length=100",
        "format": "huggingface_rows",
        "field": "rows",
        "category": "audio",
    },
]


def _fingerprint(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode()).hexdigest()[:12]


def _flatten_record(record) -> Optional[str]:
    """Convert any record shape into a training text string."""
    if isinstance(record, str) and len(record) > 8:
        return record.strip()
    if isinstance(record, dict):
        parts = []
        priority_fields = ["caption", "description", "text", "hook", "hashtag",
                           "content", "lyrics", "title", "instruction", "output",
                           "prompt", "response", "aspect_list"]
        for f in priority_fields:
            if f in record and isinstance(record[f], str) and len(record[f]) > 3:
                parts.append(record[f])
        if not parts:
            for v in record.values():
                if isinstance(v, str) and len(v) > 8:
                    parts.append(v)
        return " | ".join(parts[:4]) if parts else None
    if isinstance(record, list):
        joined = " ".join(str(x) for x in record if x)
        return joined if len(joined) > 8 else None
    return None


class DataPuller:
    """
    Periodically pulls training data from pdim and public sources.
    Deduplicates by fingerprint and stores locally + back to pdim.
    """

    STATE_KEY = "mb:training:puller:state"
    PULLED_KEY = "mb:training:puller:pulled"
    DATASET_KEY = "mb:training:puller:dataset"

    def __init__(self, storage):
        self.storage = storage
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._seen: set = set()

        self.state = {
            "status": "idle",
            "last_pull": None,
            "next_pull": None,
            "total_samples": 0,
            "pdim_samples": 0,
            "public_samples": 0,
            "pull_count": 0,
            "sources_available": len(PUBLIC_SOURCES),
            "last_error": None,
        }

        TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def start(self, interval_minutes: int = 30):
        """Start the background data pulling loop."""
        with self._lock:
            if self._running:
                return {"already_running": True}
            self._running = True
            self._thread = threading.Thread(
                target=self._loop, args=(interval_minutes,), daemon=True, name="DataPuller"
            )
            self._thread.start()
            logger.info(f"[DataPuller] Started — pulling every {interval_minutes} min")
            return {"started": True, "interval_minutes": interval_minutes}

    def stop(self):
        """Stop the background loop."""
        with self._lock:
            self._running = False
        logger.info("[DataPuller] Stop requested")

    def pull_now(self) -> dict:
        """Trigger a synchronous pull (runs in caller's thread)."""
        return self._do_pull()

    def get_state(self) -> dict:
        with self._lock:
            return dict(self.state)

    def get_local_samples(self, max_samples: int = 2000) -> list[str]:
        """Return locally-cached training texts (merged from all sources)."""
        samples = []
        if TRAINING_DATA_DIR.exists():
            for fpath in sorted(TRAINING_DATA_DIR.rglob("*.json"), reverse=True):
                try:
                    data = json.loads(fpath.read_text())
                    if isinstance(data, list):
                        samples.extend(data)
                    if len(samples) >= max_samples:
                        break
                except Exception:
                    pass
        return samples[:max_samples]

    # ------------------------------------------------------------------ #
    # Internal loop                                                        #
    # ------------------------------------------------------------------ #

    def _loop(self, interval_minutes: int):
        while self._running:
            try:
                self._do_pull()
            except Exception as e:
                logger.error(f"[DataPuller] Pull error: {e}")
                with self._lock:
                    self.state["last_error"] = str(e)
            wait_sec = interval_minutes * 60
            with self._lock:
                self.state["next_pull"] = time.time() + wait_sec
            for _ in range(wait_sec):
                if not self._running:
                    break
                time.sleep(1)

    def _do_pull(self) -> dict:
        with self._lock:
            self.state["status"] = "pulling"
            self.state["last_pull"] = time.time()

        pulled_pdim = self._pull_from_pdim()
        pulled_public = self._pull_from_public_sources()
        all_samples = pulled_pdim + pulled_public

        # Deduplicate
        new_texts = []
        for text in all_samples:
            fp = _fingerprint(text)
            if fp not in self._seen:
                self._seen.add(fp)
                new_texts.append(text)

        if new_texts:
            self._save_locally(new_texts)
            self._push_to_pdim(new_texts)

        with self._lock:
            self.state["status"] = "idle"
            self.state["total_samples"] += len(new_texts)
            self.state["pdim_samples"] += len(pulled_pdim)
            self.state["public_samples"] += len(pulled_public)
            self.state["pull_count"] += 1
            self.state["last_error"] = None

        logger.info(
            f"[DataPuller] Pull #{self.state['pull_count']} done — "
            f"{len(new_texts)} new samples "
            f"(pdim:{len(pulled_pdim)} public:{len(pulled_public)})"
        )
        return {
            "new_samples": len(new_texts),
            "pdim_samples": len(pulled_pdim),
            "public_samples": len(pulled_public),
        }

    # ------------------------------------------------------------------ #
    # pdim pull                                                            #
    # ------------------------------------------------------------------ #

    def _pull_from_pdim(self) -> list[str]:
        texts = []
        if not self.storage.is_available:
            return texts

        # Scan known list keys
        list_keys = ["mbs:data", "mbs:downloads", "mbs_training", "mbs_downloads"]
        for key in list_keys:
            try:
                length = self.storage.llen(key)
                if length > 0:
                    chunk = self.storage.lrange(key, 0, min(length - 1, 499))
                    for item in chunk:
                        t = _flatten_record(item)
                        if t:
                            texts.append(t)
            except Exception as e:
                logger.debug(f"[DataPuller] pdim key {key} error: {e}")

        # Scan wildcard patterns for analytics, ads, social posts
        wildcard_patterns = [
            "mb:ads:*:peaks",
            "mb:social:posts:*",
            "mb:analytics:*",
            "mb:content:*",
        ]
        for pattern in wildcard_patterns:
            try:
                matching = self.storage.keys(pattern)
                for key in matching[:50]:
                    val = self.storage.get(key)
                    if val:
                        t = _flatten_record(val)
                        if t:
                            texts.append(t)
            except Exception as e:
                logger.debug(f"[DataPuller] wildcard {pattern} error: {e}")

        # Pull dataset chunks registered by DatasetStreamClient
        try:
            meta_keys = self.storage.keys("mb:dataset:*:meta")
            for meta_key in meta_keys[:20]:
                meta = self.storage.get(meta_key)
                if not meta:
                    continue
                name = meta.get("name", "") if isinstance(meta, dict) else ""
                num_chunks = meta.get("num_chunks", 0) if isinstance(meta, dict) else 0
                for chunk_idx in range(min(num_chunks, 5)):
                    chunk = self.storage.get(f"mb:dataset:{name}:chunk:{chunk_idx}")
                    if chunk:
                        if isinstance(chunk, list):
                            for item in chunk:
                                t = _flatten_record(item)
                                if t:
                                    texts.append(t)
        except Exception as e:
            logger.debug(f"[DataPuller] dataset chunks error: {e}")

        return texts

    # ------------------------------------------------------------------ #
    # Public dataset pull                                                  #
    # ------------------------------------------------------------------ #

    def _pull_from_public_sources(self) -> list[str]:
        texts = []
        for src in PUBLIC_SOURCES:
            try:
                raw = self._http_get(src["url"], timeout=15)
                if raw is None:
                    continue
                parsed = json.loads(raw)
                extracted = self._extract_from_source(parsed, src)
                texts.extend(extracted)
                logger.info(f"[DataPuller] {src['id']}: {len(extracted)} samples")
            except Exception as e:
                logger.debug(f"[DataPuller] {src['id']} failed: {e}")
        return texts

    def _extract_from_source(self, parsed, src: dict) -> list[str]:
        texts = []
        fmt = src.get("format", "json_list")
        field = src.get("field")

        if fmt == "huggingface_rows":
            rows = parsed.get(field, []) if field else parsed
            for row in rows:
                row_data = row.get("row", row) if isinstance(row, dict) else row
                t = _flatten_record(row_data)
                if t:
                    texts.append(t)

        elif fmt == "json_list":
            items = parsed.get(field, parsed) if field else parsed
            if isinstance(items, list):
                for item in items:
                    t = _flatten_record(item)
                    if t:
                        texts.append(t)
            elif isinstance(items, dict):
                for v in items.values():
                    t = _flatten_record(v)
                    if t:
                        texts.append(t)
        return texts

    # ------------------------------------------------------------------ #
    # Storage                                                              #
    # ------------------------------------------------------------------ #

    def _save_locally(self, texts: list[str]):
        ts = int(time.time())
        path = TRAINING_DATA_DIR / f"pull_{ts}.json"
        path.write_text(json.dumps(texts, ensure_ascii=False))
        # Keep only last 50 pull files (rolling window)
        files = sorted(TRAINING_DATA_DIR.glob("pull_*.json"))
        for old in files[:-50]:
            try:
                old.unlink()
            except Exception:
                pass

    def _push_to_pdim(self, texts: list[str]):
        if not self.storage.is_available:
            return
        try:
            self.storage.set(self.DATASET_KEY, {
                "samples": texts[:500],
                "count": len(texts),
                "updated_at": time.time(),
            })
            self.storage.set(self.STATE_KEY, self.state)
        except Exception as e:
            logger.debug(f"[DataPuller] push to pdim failed: {e}")

    # ------------------------------------------------------------------ #
    # HTTP helper                                                          #
    # ------------------------------------------------------------------ #

    def _http_get(self, url: str, timeout: int = 10) -> Optional[bytes]:
        try:
            req = Request(url, headers={"User-Agent": "MaxBooster-AI-DataPuller/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (URLError, Exception) as e:
            logger.debug(f"[DataPuller] GET {url} failed: {e}")
            return None
