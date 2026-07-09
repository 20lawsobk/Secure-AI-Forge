"""A from-scratch Monte-Carlo path tracer (IRC / V-Ray-style behaviour).

Pure NumPy + the self-contained Digital GPU. Renders real lit geometry with:

  * pinhole camera ray generation (camera basis applied via a Digital-GPU GEMM),
  * analytic ray/sphere + ray/plane intersection (the per-ray/per-object dot
    products ``D·C`` and ``O·C`` are batched GEMMs on the Digital GPU),
  * Lambertian surfaces with cosine-weighted importance sampling,
  * emissive area lights + an environment (sky) term for global illumination,
  * multi-bounce indirect lighting (colour bleeding, soft shadows),
  * a filmic (ACES-approx) tonemap.

Everything is seeded for deterministic output. Resolution / samples / bounces
are bounded so a render stays tractable on CPU. There is no fake fallback: the
tracer either produces real pixels or raises.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..fabric.compute import RTACompute

_EPS = 1e-4


@dataclass
class Material:
    albedo: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    emission: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Sphere:
    center: Tuple[float, float, float]
    radius: float
    material: Material


@dataclass
class Plane:
    y: float
    material: Material


@dataclass
class Camera:
    origin: Tuple[float, float, float] = (0.0, 1.1, 4.2)
    look_at: Tuple[float, float, float] = (0.0, 0.9, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_deg: float = 42.0


@dataclass
class Scene:
    spheres: List[Sphere] = field(default_factory=list)
    plane: Optional[Plane] = None
    camera: Camera = field(default_factory=Camera)
    sky_top: Tuple[float, float, float] = (0.55, 0.70, 1.0)
    sky_horizon: Tuple[float, float, float] = (0.9, 0.85, 0.8)
    sky_intensity: float = 1.0
    exposure: float = 1.1


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 1e-12, n, 1.0)


class PathTracer:
    def __init__(self, compute: Optional[RTACompute] = None):
        self.compute = compute or RTACompute()

    # ── camera ────────────────────────────────────────────────────────────
    def _generate_rays(self, scene: Scene, width: int, height: int,
                       rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        cam = scene.camera
        origin = np.array(cam.origin, dtype=np.float64)
        forward = _unit(np.array(cam.look_at, dtype=np.float64) - origin)
        right = _unit(np.cross(forward, np.array(cam.up, dtype=np.float64)))
        true_up = np.cross(right, forward)

        aspect = width / float(height)
        half = np.tan(np.radians(cam.fov_deg) * 0.5)

        # jittered pixel centres in [-1,1]
        jx = rng.random(width * height, dtype=np.float64)
        jy = rng.random(width * height, dtype=np.float64)
        xs = (np.tile(np.arange(width), height) + jx) / width
        ys = (np.repeat(np.arange(height), width) + jy) / height
        px = (2.0 * xs - 1.0) * half * aspect
        py = (1.0 - 2.0 * ys) * half

        # camera-space directions [N,3] = px*Xhat + py*Yhat + 1*Zhat
        dir_cam = np.stack([px, py, np.ones_like(px)], axis=1)
        # world basis matrix [3,3]; world dir = dir_cam @ basis  (GEMM on Digital GPU)
        basis = np.stack([right, true_up, forward], axis=0)
        dirs = self.compute.gemm(dir_cam, basis).astype(np.float64)
        dirs = _normalize_rows(dirs)
        origins = np.broadcast_to(origin, dirs.shape).copy()
        return origins, dirs

    # ── intersection ────────────────────────────────────────────────────────
    def _intersect(self, scene: Scene, O: np.ndarray, D: np.ndarray):
        """Return (t, hit_id) per ray. hit_id: -1 miss, 0..M-1 sphere, M plane."""
        n = O.shape[0]
        best_t = np.full(n, np.inf)
        best_id = np.full(n, -1, dtype=np.int64)

        spheres = scene.spheres
        if spheres:
            centers = np.array([s.center for s in spheres], dtype=np.float64)   # [M,3]
            radii = np.array([s.radius for s in spheres], dtype=np.float64)     # [M]
            # Batched dot products via Digital-GPU GEMM.
            DdotC = self.compute.gemm(D, centers.T).astype(np.float64)          # [N,M]
            OdotC = self.compute.gemm(O, centers.T).astype(np.float64)          # [N,M]
            DdotO = np.sum(D * O, axis=1, keepdims=True)                        # [N,1]
            Onorm2 = np.sum(O * O, axis=1, keepdims=True)                       # [N,1]
            Cnorm2 = np.sum(centers * centers, axis=1)[None, :]                 # [1,M]

            b = DdotO - DdotC                                                   # D·(O-C)
            oc2 = Onorm2 - 2.0 * OdotC + Cnorm2                                 # |O-C|^2
            c_term = oc2 - (radii ** 2)[None, :]
            disc = b * b - c_term
            valid = disc > 0.0
            sq = np.sqrt(np.maximum(disc, 0.0))
            t0 = -b - sq
            t1 = -b + sq
            t = np.where(t0 > _EPS, t0, t1)
            t = np.where(valid & (t > _EPS), t, np.inf)                         # [N,M]

            m_idx = np.argmin(t, axis=1)
            m_t = t[np.arange(n), m_idx]
            take = m_t < best_t
            best_t = np.where(take, m_t, best_t)
            best_id = np.where(take, m_idx, best_id)

        if scene.plane is not None:
            dy = D[:, 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                tp = (scene.plane.y - O[:, 1]) / dy
            tp = np.where((np.abs(dy) > 1e-9) & (tp > _EPS), tp, np.inf)
            plane_id = len(spheres)
            take = tp < best_t
            best_t = np.where(take, tp, best_t)
            best_id = np.where(take, plane_id, best_id)

        return best_t, best_id

    # ── shading helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _cosine_hemisphere(normals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = normals.shape[0]
        u1 = rng.random(n)
        u2 = rng.random(n)
        r = np.sqrt(u1)
        theta = 2.0 * np.pi * u2
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.sqrt(np.maximum(0.0, 1.0 - u1))
        # build an orthonormal basis around each normal
        helper = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
        near_x = np.abs(normals[:, 0]) > 0.9
        helper[near_x] = np.array([0.0, 1.0, 0.0])
        tangent = _normalize_rows(np.cross(helper, normals))
        bitangent = np.cross(normals, tangent)
        return _normalize_rows(
            tangent * x[:, None] + bitangent * y[:, None] + normals * z[:, None]
        )

    def _environment(self, scene: Scene, D: np.ndarray) -> np.ndarray:
        t = 0.5 * (D[:, 1] + 1.0)
        top = np.array(scene.sky_top, dtype=np.float64)
        hor = np.array(scene.sky_horizon, dtype=np.float64)
        env = (1.0 - t)[:, None] * hor[None, :] + t[:, None] * top[None, :]
        return env * scene.sky_intensity

    # ── main render ─────────────────────────────────────────────────────────
    def render(self, scene: Scene, width: int = 256, height: int = 256,
               samples: int = 4, max_bounces: int = 2, seed: int = 0,
               clamp: float = 8.0, denoise: bool = True) -> np.ndarray:
        """Return an ``HxWx3`` float64 HDR-ish image (pre-tonemap, ~[0, inf)).

        ``clamp`` bounds each sample's radiance to suppress fireflies; ``denoise``
        applies a light 3x3 median (edge-preserving) to the accumulated image.
        """
        width = int(max(16, min(width, 640)))
        height = int(max(16, min(height, 640)))
        samples = int(max(1, min(samples, 128)))
        max_bounces = int(max(1, min(max_bounces, 6)))

        spheres = scene.spheres
        n_sph = len(spheres)
        albedos = np.array([s.material.albedo for s in spheres], dtype=np.float64) if n_sph else np.zeros((0, 3))
        emissions = np.array([s.material.emission for s in spheres], dtype=np.float64) if n_sph else np.zeros((0, 3))
        plane_albedo = np.array(scene.plane.material.albedo, dtype=np.float64) if scene.plane else None
        plane_emission = np.array(scene.plane.material.emission, dtype=np.float64) if scene.plane else None

        accum = np.zeros((width * height, 3), dtype=np.float64)

        for s in range(samples):
            rng = np.random.default_rng((seed * 1_000_003 + s) & 0x7FFFFFFF)
            O, D = self._generate_rays(scene, width, height, rng)
            radiance = np.zeros((width * height, 3), dtype=np.float64)
            throughput = np.ones((width * height, 3), dtype=np.float64)
            active = np.ones(width * height, dtype=bool)

            for _bounce in range(max_bounces):
                if not active.any():
                    break
                Oa, Da = O[active], D[active]
                t, hid = self._intersect(scene, Oa, Da)

                miss = hid < 0
                if miss.any():
                    env = self._environment(scene, Da[miss])
                    idx_active = np.where(active)[0]
                    miss_global = idx_active[miss]
                    radiance[miss_global] += throughput[miss_global] * env

                hit = ~miss
                if not hit.any():
                    # every remaining ray escaped to the environment
                    active_idx = np.where(active)[0]
                    active[active_idx] = False
                    break

                Oh, Dh, th, idh = Oa[hit], Da[hit], t[hit], hid[hit]
                P = Oh + th[:, None] * Dh

                normals = np.zeros_like(P)
                alb = np.zeros_like(P)
                emit = np.zeros_like(P)

                is_plane = idh == n_sph
                is_sphere = ~is_plane
                if is_sphere.any():
                    sidx = idh[is_sphere]
                    centers_hit = np.array([spheres[i].center for i in sidx], dtype=np.float64)
                    normals[is_sphere] = _normalize_rows(P[is_sphere] - centers_hit)
                    alb[is_sphere] = albedos[sidx]
                    emit[is_sphere] = emissions[sidx]
                if is_plane.any() and scene.plane is not None:
                    normals[is_plane] = np.array([0.0, 1.0, 0.0])
                    # subtle checker to read the ground plane
                    pxz = P[is_plane][:, [0, 2]]
                    checker = (np.floor(pxz[:, 0]) + np.floor(pxz[:, 1])).astype(int) & 1
                    base = plane_albedo[None, :] * np.where(checker[:, None] == 0, 1.0, 0.55)
                    alb[is_plane] = base
                    emit[is_plane] = plane_emission

                idx_active = np.where(active)[0]
                hit_global = idx_active[hit]
                radiance[hit_global] += throughput[hit_global] * emit

                # continue diffuse bounce
                new_dir = self._cosine_hemisphere(normals, rng)
                throughput[hit_global] = throughput[hit_global] * alb
                O = O.copy()
                D = D.copy()
                O[hit_global] = P + normals * _EPS
                D[hit_global] = new_dir

                # deactivate rays that escaped this bounce
                miss_global = idx_active[miss]
                active[miss_global] = False

            # Firefly clamp: bound each sample's radiance so rare high-variance
            # paths (e.g. a ray that grazes an emitter) can't spike a pixel.
            if clamp and clamp > 0:
                radiance = np.clip(radiance, 0.0, clamp)
            accum += radiance

        img = (accum / float(samples)).reshape(height, width, 3) * scene.exposure
        if denoise:
            img = _median3x3(img)
        return img


def _median3x3(img: np.ndarray) -> np.ndarray:
    """Edge-preserving 3x3 median filter (per channel). Kills residual fireflies."""
    h, w, c = img.shape
    pad = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    stack = [pad[i:i + h, j:j + w, :] for i in range(3) for j in range(3)]
    return np.median(np.stack(stack, axis=0), axis=0)


def tonemap_filmic(hdr: np.ndarray) -> np.ndarray:
    """ACES-approx filmic tonemap → uint8 ``HxWx3``."""
    x = np.maximum(hdr, 0.0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = (x * (a * x + b)) / (x * (c * x + d) + e)
    mapped = np.clip(mapped, 0.0, 1.0)
    srgb = np.where(mapped <= 0.0031308, mapped * 12.92,
                    1.055 * np.power(mapped, 1 / 2.4) - 0.055)
    return (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
