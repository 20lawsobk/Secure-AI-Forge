"""
MaxCore Retrieval Spine — in-house vector index with an all-real cascade.

This is the foundation of "no broken fallback": every query is answered with a
REAL stored asset, never a procedural/empty placeholder. Retrieval walks a
four-rung cascade, each rung made of real pixels:

  ① exact        — content key hit (same asset already ingested)
  ② nearest      — closest real asset within NEAREST_RADIUS
  ③ brand_prior  — closest asset inside the requesting artist's brand subset
  ④ anchor       — always-loaded domain-anchor core (curated real assets)

The anchor rung is the invariant that guarantees a non-empty result whenever the
core is loaded. The coverage watchdog's job is to keep that core loaded and to
turn weak-coverage queries into ingestion targets.

Everything here is deterministic (no RNG) and total (never raises): bad vectors
are sanitized or rejected gracefully, and queries on a non-empty index always
return at least one asset.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Cosine-distance (1 - cosine_similarity) cutoffs. Range is [0, 2].
NEAREST_RADIUS = 0.35          # within this → a confident "nearest" hit
BRAND_RADIUS = 0.55            # within this → an acceptable brand-prior hit
COVERAGE_GOOD_FRACTION = 0.75  # ≥ this share of probes inside NEAREST_RADIUS → healthy
COVERAGE_MIN_FRACTION = 0.40   # < this → critical coverage
MIN_ANCHORS = 1                # the core must hold at least this many real anchors


@dataclass
class RetrievedAsset:
    asset_id: str
    rung: str                       # "exact" | "nearest" | "brand_prior" | "anchor"
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "rung": self.rung,
            "distance": self.distance,
            "metadata": self.metadata,
        }


def _content_key(vector: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(vector).tobytes(), digest_size=16).hexdigest()


class AssetIndex:
    """
    Deterministic, thread-safe vector index over real assets.

    Embedding-agnostic: it stores whatever feature vectors are added (image
    features, palette histograms, etc.) and answers nearest-neighbour queries by
    cosine distance. The cascade logic lives in :meth:`query`.
    """

    def __init__(self, dim: int = 128):
        self.dim = int(dim) if dim and dim > 0 else 128
        self._lock = threading.RLock()

        self._ids: List[str] = []
        self._vecs: List[np.ndarray] = []          # unit-normalized float32
        self._meta: List[Dict[str, Any]] = []
        self._is_anchor: List[bool] = []
        self._brand: List[Optional[str]] = []

        self._id_to_idx: Dict[str, int] = {}
        self._exact_to_idx: Dict[str, int] = {}    # content-hash → idx

        # Per-brand running centroid accumulators (sum of unit vectors + count),
        # maintained incrementally under _lock so a brand's visual identity is
        # O(1) to read for brand-aware retrieval/conditioning.
        self._brand_sum: Dict[str, np.ndarray] = {}
        self._brand_count: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Mutation                                                            #
    # ------------------------------------------------------------------ #

    def _sanitize(self, vector: Any) -> Optional[np.ndarray]:
        """Coerce to a finite unit vector of the right dim, or None if unusable."""
        try:
            v = np.asarray(vector, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if v.shape[0] != self.dim:
            if v.shape[0] > self.dim:
                v = v[: self.dim]
            else:
                v = np.pad(v, (0, self.dim - v.shape[0]))
        if not np.all(np.isfinite(v)):
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        norm = float(np.linalg.norm(v))
        if norm <= 1e-12:
            return None
        return (v / norm).astype(np.float32)

    def _brand_accumulate(self, brand: Optional[str], vec: np.ndarray, sign: int) -> None:
        """Roll a vector into/out of a brand's centroid. Caller must hold _lock."""
        if not brand:
            return
        contrib = np.asarray(vec, dtype=np.float64).reshape(-1)
        if contrib.shape[0] != self.dim:
            return
        if sign > 0:
            cur = self._brand_sum.get(brand)
            self._brand_sum[brand] = contrib.copy() if cur is None else (cur + contrib)
            self._brand_count[brand] = self._brand_count.get(brand, 0) + 1
        else:
            cur = self._brand_sum.get(brand)
            if cur is None:
                return
            self._brand_sum[brand] = cur - contrib
            self._brand_count[brand] = self._brand_count.get(brand, 1) - 1
            if self._brand_count[brand] <= 0:
                self._brand_sum.pop(brand, None)
                self._brand_count.pop(brand, None)

    def _recompute_brands(self) -> None:
        """Rebuild every per-brand accumulator from scratch. Caller must hold _lock."""
        self._brand_sum = {}
        self._brand_count = {}
        for i, b in enumerate(self._brand):
            self._brand_accumulate(b, self._vecs[i], +1)

    def add(
        self,
        asset_id: str,
        vector: Any,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        is_anchor: bool = False,
        brand: Optional[str] = None,
    ) -> bool:
        """Add or replace an asset. Returns False (never raises) if unusable."""
        v = self._sanitize(vector)
        if v is None or not asset_id:
            return False
        meta = dict(metadata or {})
        with self._lock:
            if asset_id in self._id_to_idx:
                idx = self._id_to_idx[asset_id]
                old_key = _content_key(self._vecs[idx])
                if self._exact_to_idx.get(old_key) == idx:
                    del self._exact_to_idx[old_key]
                # Roll the prior vector/brand out of its centroid before replacing.
                self._brand_accumulate(self._brand[idx], self._vecs[idx], -1)
                self._vecs[idx] = v
                self._meta[idx] = meta
                self._is_anchor[idx] = bool(is_anchor)
                self._brand[idx] = brand
            else:
                idx = len(self._ids)
                self._ids.append(asset_id)
                self._vecs.append(v)
                self._meta.append(meta)
                self._is_anchor.append(bool(is_anchor))
                self._brand.append(brand)
                self._id_to_idx[asset_id] = idx
            self._brand_accumulate(brand, v, +1)
            self._exact_to_idx[_content_key(v)] = idx
        return True

    def remove(self, asset_id: str) -> bool:
        with self._lock:
            idx = self._id_to_idx.pop(asset_id, None)
            if idx is None:
                return False
            # Rebuild compactly to keep indices contiguous and deterministic.
            keep = [i for i in range(len(self._ids)) if i != idx]
            self._ids = [self._ids[i] for i in keep]
            self._vecs = [self._vecs[i] for i in keep]
            self._meta = [self._meta[i] for i in keep]
            self._is_anchor = [self._is_anchor[i] for i in keep]
            self._brand = [self._brand[i] for i in keep]
            self._id_to_idx = {aid: i for i, aid in enumerate(self._ids)}
            self._exact_to_idx = {
                _content_key(self._vecs[i]): i for i in range(len(self._ids))
            }
            self._recompute_brands()
            return True

    # ------------------------------------------------------------------ #
    # Stats                                                               #
    # ------------------------------------------------------------------ #

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._ids)

    @property
    def anchor_count(self) -> int:
        with self._lock:
            return sum(1 for a in self._is_anchor if a)

    def brands(self) -> List[str]:
        with self._lock:
            return sorted({b for b in self._brand if b})

    def brand_centroid(self, brand: str) -> Optional[np.ndarray]:
        """
        Unit-normalized mean embedding of every asset carrying ``brand`` — the
        brand's visual "identity vector" for brand-aware retrieval/conditioning.
        Returns None for an unknown/empty brand. Never raises.
        """
        if not brand:
            return None
        with self._lock:
            s = self._brand_sum.get(brand)
            c = int(self._brand_count.get(brand, 0))
            if s is None or c <= 0:
                return None
            mean = s / float(c)
        norm = float(np.linalg.norm(mean))
        if norm <= 1e-12:
            return None
        return (mean / norm).astype(np.float32)

    def brand_count(self, brand: str) -> int:
        """Number of assets contributing to a brand's centroid."""
        with self._lock:
            return int(self._brand_count.get(brand, 0))

    def brand_stats(self) -> Dict[str, int]:
        """Snapshot of {brand: contributing_asset_count} for all known brands."""
        with self._lock:
            return {b: int(c) for b, c in self._brand_count.items()}

    # ------------------------------------------------------------------ #
    # Query — the all-real cascade                                         #
    # ------------------------------------------------------------------ #

    def _nearest_in(
        self, q: np.ndarray, candidate_idxs: List[int]
    ) -> Optional[Tuple[int, float]]:
        """Argmin cosine distance over candidate_idxs (deterministic tie-break by id)."""
        if not candidate_idxs:
            return None
        best_idx = -1
        best_dist = float("inf")
        for i in candidate_idxs:
            sim = float(np.dot(q, self._vecs[i]))
            dist = 1.0 - sim
            if dist < best_dist or (dist == best_dist and self._ids[i] < self._ids[best_idx]):
                best_dist = dist
                best_idx = i
        if best_idx < 0:
            return None
        return best_idx, max(0.0, best_dist)

    def query(
        self,
        vector: Any,
        *,
        brand: Optional[str] = None,
        exact_key: Optional[str] = None,
    ) -> Optional[RetrievedAsset]:
        """
        Resolve one real asset via the cascade. Returns None ONLY when the index
        is completely empty (no assets and no anchors) — the state the coverage
        watchdog exists to prevent. Never raises.
        """
        q = self._sanitize(vector)
        with self._lock:
            if not self._ids:
                return None
            if q is None:
                # Unusable query: still answer with a real asset (anchor first).
                anchors = [i for i, a in enumerate(self._is_anchor) if a]
                fallback_idx = anchors[0] if anchors else 0
                return self._make(fallback_idx, "anchor" if anchors else "nearest", 2.0)

            # ① exact
            if exact_key is not None:
                idx = self._exact_to_idx.get(exact_key)
                if idx is not None:
                    return self._make(idx, "exact", 0.0)

            real_idxs = [i for i in range(len(self._ids)) if not self._is_anchor[i]]
            anchor_idxs = [i for i in range(len(self._ids)) if self._is_anchor[i]]

            # ② nearest over real (non-anchor) assets within radius
            near = self._nearest_in(q, real_idxs)
            if near is not None and near[1] <= NEAREST_RADIUS:
                return self._make(near[0], "nearest", near[1])

            # ③ brand prior
            if brand:
                brand_idxs = [i for i in real_idxs if self._brand[i] == brand]
                bnear = self._nearest_in(q, brand_idxs)
                if bnear is not None and bnear[1] <= BRAND_RADIUS:
                    return self._make(bnear[0], "brand_prior", bnear[1])

            # ④ anchor core (always-real domain fallback)
            anear = self._nearest_in(q, anchor_idxs)
            if anear is not None:
                return self._make(anear[0], "anchor", anear[1])

            # No anchors loaded — degrade to the best real asset rather than empty.
            if near is not None:
                return self._make(near[0], "nearest", near[1])
            return None

    def _make(self, idx: int, rung: str, dist: float) -> RetrievedAsset:
        return RetrievedAsset(
            asset_id=self._ids[idx],
            rung=rung,
            distance=float(dist),
            metadata=dict(self._meta[idx]),
        )

    # ------------------------------------------------------------------ #
    # Coverage gate                                                        #
    # ------------------------------------------------------------------ #

    def coverage_report(self, probes: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Summarize how well the index covers the query space. If probes are given,
        measures the share resolved within NEAREST_RADIUS by a real (non-anchor)
        asset; otherwise reports structural stats only. Never raises.
        """
        with self._lock:
            n_assets = len(self._ids)
            n_anchors = sum(1 for a in self._is_anchor if a)
            n_real = n_assets - n_anchors
            n_brands = len({b for b in self._brand if b})

            within = 0
            total = 0
            mean_dist = 0.0
            if probes:
                real_idxs = [i for i in range(n_assets) if not self._is_anchor[i]]
                dists: List[float] = []
                for p in probes:
                    q = self._sanitize(p)
                    if q is None:
                        continue
                    total += 1
                    near = self._nearest_in(q, real_idxs)
                    if near is not None:
                        dists.append(near[1])
                        if near[1] <= NEAREST_RADIUS:
                            within += 1
                if dists:
                    mean_dist = float(np.mean(dists))

        fraction = (within / total) if total else 0.0
        report: Dict[str, Any] = {
            "n_assets": n_assets,
            "n_real": n_real,
            "n_anchors": n_anchors,
            "n_brands": n_brands,
            "has_anchors": n_anchors >= MIN_ANCHORS,
            "probes": total,
            "fraction_within_radius": fraction,
            "mean_nearest_distance": mean_dist,
        }
        report["gate"] = self.coverage_gate(report)
        return report

    @staticmethod
    def coverage_gate(report: Dict[str, Any]) -> str:
        """Map a coverage report to 'healthy' | 'degraded' | 'critical'."""
        if not report.get("has_anchors", False) or report.get("n_assets", 0) == 0:
            return "critical"
        if report.get("probes", 0) == 0:
            return "healthy" if report.get("n_real", 0) > 0 else "degraded"
        frac = report.get("fraction_within_radius", 0.0)
        if frac >= COVERAGE_GOOD_FRACTION:
            return "healthy"
        if frac >= COVERAGE_MIN_FRACTION:
            return "degraded"
        return "critical"

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def to_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "dim": self.dim,
                "ids": list(self._ids),
                "vecs": [v.astype(np.float32).tolist() for v in self._vecs],
                "meta": [dict(m) for m in self._meta],
                "is_anchor": list(self._is_anchor),
                "brand": list(self._brand),
            }

    def load_state(self, state: Dict[str, Any]) -> bool:
        """Replace contents from a previously serialized state. Never raises."""
        try:
            dim = int(state.get("dim", self.dim))
            ids = list(state.get("ids", []))
            vecs = state.get("vecs", [])
            meta = state.get("meta", [])
            is_anchor = state.get("is_anchor", [])
            brand = state.get("brand", [])
        except Exception:
            return False
        with self._lock:
            self.dim = dim if dim > 0 else self.dim
            self._ids, self._vecs, self._meta = [], [], []
            self._is_anchor, self._brand = [], []
            self._id_to_idx, self._exact_to_idx = {}, {}
            self._brand_sum, self._brand_count = {}, {}
        for i, aid in enumerate(ids):
            v = vecs[i] if i < len(vecs) else None
            m = meta[i] if i < len(meta) else {}
            a = bool(is_anchor[i]) if i < len(is_anchor) else False
            b = brand[i] if i < len(brand) else None
            self.add(aid, v, m, is_anchor=a, brand=b)
        return True
