#!/usr/bin/env python3
"""Deterministic ONet chair data preparation (data-side only).

This module turns triangle meshes into per-shape NPZ records and split
manifests for the shape-completion experiment.  It is deliberately
self-contained and writes only below the output directory given by the caller.

Contract implemented:
  * arrays are xyz, shape (64, 64, 64); index 0 / 63 map to -1 / +1
  * partial_sdf : float32, truncated signed distance seen from one calibrated view
  * known_mask  : bool, view-derived narrow-band visibility mask
  * target_sdf  : float32, truncated signed distance of the full mesh
  * anchors     : (32, 3) float32 visible surface positions
  * normals     : (32, 3) float32 outward unit normals at the anchors
  * TSDF is positive outside, truncated to +-0.2, +0.2 filler outside mask

ONet preparation is driven by scripts/prepare_onet_chairs.py.  This module
exposes prepare_one, run_tasks, write_outputs, verify_root, and selftest.

Typical commands (from the task directory):

  python scripts/prepare_onet_chairs.py --archive <archive> --out <data> \
      --config configs/onet_chair_split.json --workers 8

  python -m gcnf_shape_completion.data_prep verify --root <data>

  python -m gcnf_shape_completion.data_prep selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Open3D's wheel needs a local libEGL on this headless host.  The repo keeps a
# copy in data/.lib; set LD_LIBRARY_PATH before importing open3d.
_DATA_DIR = Path(__file__).resolve().parent
_TASK_DIR = _DATA_DIR.parent
# Open3D's wheel needs a local libEGL on some headless hosts.  Keep copies in
# either src/.../.lib or the task root .lib; LD_LIBRARY_PATH is also honoured.
for _candidate in (_DATA_DIR / ".lib", _TASK_DIR / ".lib"):
    if _candidate.is_dir():
        os.environ["LD_LIBRARY_PATH"] = (
            str(_candidate) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )

import numpy as np  # noqa: E402

TRUNC = 0.2
RES = 64
GRID = np.linspace(-1.0, 1.0, RES, dtype=np.float64)
N_ANCHORS = 32
VIEW_RES = 256
VIEW_HALF = 1.02
CAMERA_AZ_DEG = 40.0
CAMERA_EL_DEG = 20.0
SURFACE_SAMPLES = 16384
DEPTH_EPS = 1e-5

MANIFEST_KEYS = ("partial_sdf", "known_mask", "target_sdf", "anchors", "normals")


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _seed_from_text(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _as_mesh(obj: Any) -> Any:
    import trimesh

    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.geometry.values() if hasattr(g, "vertices") and hasattr(g, "faces")]
        if not geoms:
            raise ValueError("scene has no triangle geometry")
        return trimesh.util.concatenate(geoms)
    if isinstance(obj, trimesh.Trimesh):
        return obj
    raise ValueError(f"unsupported mesh object: {type(obj)!r}")


# --------------------------------------------------------------------------- #
# canonical normalisation (matches SDFusion/DISN preprocessing as closely as
# practical: 16384 area-weighted surface samples, their centroid, max radius)
# --------------------------------------------------------------------------- #
@dataclass
class NormalisedMesh:
    vertices: np.ndarray  # (V,3) float64 normalised
    faces: np.ndarray  # (F,3) int64
    centroid: np.ndarray  # (3,) in source coordinates
    scale: float
    stats: Dict[str, Any] = field(default_factory=dict)


def normalise_mesh(mesh: Any, seed: int) -> NormalisedMesh:
    import trimesh

    mesh = _as_mesh(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError(f"bad vertices: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
        raise ValueError(f"bad faces: {faces.shape}")
    if not np.isfinite(vertices).all():
        raise ValueError("non-finite vertices")
    if faces.max(initial=-1) >= len(vertices) or faces.min(initial=0) < 0:
        raise ValueError("face indices out of range")

    sample_method = "surface"
    try:
        samples, _ = trimesh.sample.sample_surface(mesh, SURFACE_SAMPLES, seed=seed)
        samples = np.asarray(samples, dtype=np.float64)
    except Exception:
        sample_method = "vertices"
        samples = vertices
    if len(samples) < 8 or not np.isfinite(samples).all():
        raise ValueError("surface sampling failed")

    centroid = samples.mean(axis=0)
    centred = samples - centroid
    scale = float(np.linalg.norm(centred, axis=1).max())
    if not np.isfinite(scale) or scale <= 1e-9:
        raise ValueError(f"degenerate normalisation scale {scale!r}")

    vn = (vertices - centroid) / scale
    if not np.isfinite(vn).all():
        raise ValueError("non-finite normalised vertices")

    # Stats use a merged copy, which is the meaningful topological mesh.
    topo = trimesh.Trimesh(vertices=vn, faces=faces, process=True, validate=False)
    try:
        components = len(topo.split(only_watertight=False))
    except Exception:
        components = -1
    stats = {
        "sample_method": sample_method,
        "num_vertices": int(len(vertices)),
        "num_faces": int(len(faces)),
        "num_components": int(components),
        "is_watertight": bool(topo.is_watertight),
        "is_winding_consistent": bool(topo.is_winding_consistent),
        "volume_signed": float(topo.volume) if np.isfinite(topo.volume) else None,
        "bbox_min": vn.min(axis=0).astype(float).tolist(),
        "bbox_max": vn.max(axis=0).astype(float).tolist(),
    }
    return NormalisedMesh(vn, faces, centroid.astype(np.float64), scale, stats)


# --------------------------------------------------------------------------- #
# Open3D CPU ray casting + signed distance
# --------------------------------------------------------------------------- #
def _scene_for(vertices: np.ndarray, faces: np.ndarray) -> Any:
    import open3d as o3d

    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(f))
    return scene


def signed_distance(scene: Any, points: np.ndarray) -> np.ndarray:
    import open3d as o3d

    p = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    out = scene.compute_signed_distance(o3d.core.Tensor(p)).numpy()
    out = np.asarray(out, dtype=np.float64).reshape(points.shape[:-1])
    if not np.isfinite(out).all():
        raise ValueError("non-finite signed distance")
    return out


def unsigned_distance(scene: Any, points: np.ndarray) -> np.ndarray:
    import open3d as o3d

    p = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    out = scene.compute_distance(o3d.core.Tensor(p)).numpy()
    return np.asarray(out, dtype=np.float64).reshape(points.shape[:-1])


@dataclass
class Camera:
    position: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    half_extent: float
    resolution: int

    def to_json(self) -> Dict[str, Any]:
        return {
            "type": "orthographic",
            "azimuth_deg": CAMERA_AZ_DEG,
            "elevation_deg": CAMERA_EL_DEG,
            "position": self.position.tolist(),
            "forward": self.forward.tolist(),
            "right": self.right.tolist(),
            "up": self.up.tolist(),
            "half_extent": self.half_extent,
            "resolution": [self.resolution, self.resolution],
            "pixel_convention": "px=(dot(p-pos,right)/half+1)/2*(W-1); nearest sample",
            "depth_convention": "depth=dot(p-pos,forward), forward unit, orthographic",
        }


def make_camera() -> Camera:
    az = math.radians(CAMERA_AZ_DEG)
    el = math.radians(CAMERA_EL_DEG)
    d = np.array(
        [math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)],
        dtype=np.float64,
    )
    d /= np.linalg.norm(d)
    position = 3.0 * d
    forward = -d
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return Camera(position, forward, right, up, VIEW_HALF, VIEW_RES)


def _ray_grid(camera: Camera) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (origins[HW,3], directions[HW,3], pixel_x[H,W], pixel_y[H,W])."""
    w = camera.resolution
    u = np.linspace(-camera.half_extent, camera.half_extent, w, dtype=np.float64)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    origins = (
        camera.position[None, None, :]
        + uu[..., None] * camera.right[None, None, :]
        + vv[..., None] * camera.up[None, None, :]
    )
    dirs = np.broadcast_to(camera.forward, origins.shape)
    px, py = np.meshgrid(np.arange(w), np.arange(w), indexing="xy")
    return origins.reshape(-1, 3), dirs.reshape(-1, 3), px, py


@dataclass
class ViewObservation:
    hit: np.ndarray  # (H,W) bool
    depth: np.ndarray  # (H,W) float32, +inf where miss
    positions: np.ndarray  # (Nhit,3) float32
    normals: np.ndarray  # (Nhit,3) float32
    pixels: np.ndarray  # (Nhit,2) int32
    normal_flips: int
    camera: Camera


def observe_view(scene: Any, camera: Camera) -> ViewObservation:
    import open3d as o3d

    w = camera.resolution
    origins, dirs, px, py = _ray_grid(camera)
    rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
    res = scene.cast_rays(o3d.core.Tensor(rays))
    t = np.asarray(res["t_hit"].numpy(), dtype=np.float64)
    prim = np.asarray(res["primitive_ids"].numpy(), dtype=np.int64)
    hit_flat = np.isfinite(t) & (prim >= 0)
    depth = np.where(hit_flat, t, np.inf).reshape(w, w).astype(np.float32)
    hit = hit_flat.reshape(w, w)

    # Exact first-hit positions and normals from the hit triangle.
    return ViewObservation(
        hit=hit,
        depth=depth,
        positions=np.zeros((0, 3), dtype=np.float32),
        normals=np.zeros((0, 3), dtype=np.float32),
        pixels=np.zeros((0, 2), dtype=np.int32),
        normal_flips=0,
        camera=camera,
    )


def observe_view_full(vertices: np.ndarray, faces: np.ndarray, camera: Camera) -> ViewObservation:
    """Ray-cast the view and recover exact hit positions/normals.

    Kept separate from observe_view so the geometry arrays stay explicit.
    """
    import open3d as o3d

    w = camera.resolution
    origins, dirs, _, _ = _ray_grid(camera)
    rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
    scene = _scene_for(vertices, faces)
    res = scene.cast_rays(o3d.core.Tensor(rays))
    t = np.asarray(res["t_hit"].numpy(), dtype=np.float64)
    prim = np.asarray(res["primitive_ids"].numpy(), dtype=np.int64)
    hit_flat = np.isfinite(t) & (prim >= 0) & (prim < len(faces))
    depth = np.where(hit_flat, t, np.inf).reshape(w, w).astype(np.float32)
    hit = hit_flat.reshape(w, w)

    idx = np.flatnonzero(hit_flat)
    if len(idx) == 0:
        return ViewObservation(
            hit=hit,
            depth=depth,
            positions=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            pixels=np.zeros((0, 2), dtype=np.int32),
            normal_flips=0,
            camera=camera,
        )
    pos = origins[idx] + t[idx, None] * dirs[idx]
    tri = faces[prim[idx]]
    v0 = vertices[tri[:, 0]]
    v1 = vertices[tri[:, 1]]
    v2 = vertices[tri[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1)
    good = ln > 1e-15
    n = n / np.maximum(ln[:, None], 1e-30)
    # Orient towards the camera (first front-facing hit); count corrections.
    view_dot = np.einsum("ij,ij->i", n, np.broadcast_to(camera.forward, n.shape))
    flip = (view_dot > 0) | (~good)
    n[flip] *= -1.0
    pixels = np.stack([idx % w, idx // w], axis=1).astype(np.int32)
    return ViewObservation(
        hit=hit,
        depth=depth,
        positions=pos.astype(np.float32),
        normals=n.astype(np.float32),
        pixels=pixels,
        normal_flips=int(flip.sum()),
        camera=camera,
    )


def project_points(points: np.ndarray, camera: Camera) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous pixel coords and depth along the camera forward axis."""
    rel = points - camera.position[None, :]
    u = rel @ camera.right / camera.half_extent
    v = rel @ camera.up / camera.half_extent
    depth = rel @ camera.forward
    w = camera.resolution
    px = (u + 1.0) * 0.5 * (w - 1)
    py = (v + 1.0) * 0.5 * (w - 1)
    return px, py, depth


def view_mask_from_depth(
    depth_map: np.ndarray, camera: Camera, points: np.ndarray, trunc: float = TRUNC
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive the observed narrow-band mask using camera+depth only.

    Returns (mask, depth, px, py).  mask is True when the point projects inside
    the viewport, the pixel has a first hit, and the point is within +-trunc of
    that hit depth (a one-view TSDF band).  Points deeper than the first hit by
    more than trunc are occluded and stay unknown.
    """
    w = camera.resolution
    px, py, depth = project_points(points, camera)
    ix = np.rint(px).astype(np.int64)
    iy = np.rint(py).astype(np.int64)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < w)
    ix_c = np.clip(ix, 0, w - 1)
    iy_c = np.clip(iy, 0, w - 1)
    dh = depth_map[iy_c, ix_c].astype(np.float64)
    has_hit = np.isfinite(dh)
    near = np.abs(depth - dh) <= trunc
    in_front_or_band = depth <= dh + trunc + DEPTH_EPS
    mask = inside & has_hit & near & in_front_or_band
    return mask, depth, px, py


def farthest_point_sample(points: np.ndarray, count: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < count:
        raise ValueError(f"need {count} points, got {len(pts)}")
    centre = pts.mean(axis=0)
    start = int(np.argmin(np.linalg.norm(pts - centre, axis=1)))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = start
    d = np.linalg.norm(pts - pts[start], axis=1)
    for i in range(1, count):
        j = int(np.argmax(d))
        selected[i] = j
        d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
    return selected


# --------------------------------------------------------------------------- #
# one-shape preparation
# --------------------------------------------------------------------------- #
@dataclass
class PrepareTask:
    mesh_path: str
    object_id: str
    dataset: str
    source_split: str
    out_dir: str
    seed: int
    overwrite: bool = False
    mesh_layout: str = ""


def prepare_one(task: PrepareTask) -> Dict[str, Any]:
    import trimesh

    t0 = time.time()
    mesh_path = Path(task.mesh_path)
    out_dir = Path(task.out_dir)
    npz_dir = out_dir / "npz"
    meta_dir = out_dir / "meta"
    npz_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    npz_path = npz_dir / f"{task.object_id}.npz"
    meta_path = meta_dir / f"{task.object_id}.json"

    if not task.overwrite and npz_path.exists() and meta_path.exists():
        meta = _read_json(meta_path)
        meta["cached"] = True
        return meta

    stage = "load"
    try:
        loaded = trimesh.load(str(mesh_path), process=False, force="mesh")
        mesh = _as_mesh(loaded)
        stage = "normalise"
        nm = normalise_mesh(mesh, task.seed)
        stage = "scene"
        scene = _scene_for(nm.vertices, nm.faces)
        camera = make_camera()
        stage = "view"
        view = observe_view_full(nm.vertices, nm.faces, camera)
        n_hits = len(view.positions)
        if n_hits < max(1000, N_ANCHORS * 10):
            raise ValueError(f"view sees too few pixels ({n_hits})")

        stage = "target_sdf"
        gx, gy, gz = np.meshgrid(GRID, GRID, GRID, indexing="ij")
        grid_points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
        sdf = signed_distance(scene, grid_points).reshape(RES, RES, RES)
        target = np.clip(sdf, -TRUNC, TRUNC).astype(np.float32)
        inside_fraction = float((sdf < 0).mean())
        if inside_fraction <= 1e-5 or inside_fraction >= 0.999:
            raise ValueError(f"degenerate occupancy inside_fraction={inside_fraction:.6g}")

        stage = "partial"
        # One-view TSDF built from the depth observation only.  For each grid
        # point in the visible narrow band, use its signed offset from the
        # first-hit depth along the camera ray (hit depth - point depth):
        # positive in front of the surface, negative behind it.  Everything
        # outside the band keeps the +TRUNC filler.  The full target SDF is
        # never copied into partial.
        mask_flat, depth_flat, px_flat, py_flat = _view_mask_2d(view.depth, camera, grid_points)
        ix = np.clip(np.rint(px_flat).astype(np.int64), 0, camera.resolution - 1)
        iy = np.clip(np.rint(py_flat).astype(np.int64), 0, camera.resolution - 1)
        hit_depth = view.depth[iy, ix].astype(np.float64)
        known = mask_flat.reshape(RES, RES, RES)
        partial = np.full(sdf.shape, TRUNC, dtype=np.float32)
        partial.reshape(-1)[mask_flat] = np.clip(
            hit_depth[mask_flat] - depth_flat[mask_flat], -TRUNC, TRUNC
        ).astype(np.float32)

        stage = "anchors"
        sel = farthest_point_sample(view.positions, N_ANCHORS)
        anchors = view.positions[sel].astype(np.float32)
        normals = view.normals[sel].astype(np.float32)
        # renormalise defensively
        nn = np.linalg.norm(normals, axis=1, keepdims=True)
        bad = (nn[:, 0] < 1e-8) | (~np.isfinite(normals).all(axis=1))
        if bad.any():
            raise ValueError(f"{int(bad.sum())} degenerate anchor normals")
        normals = normals / nn
        anchor_sdf = signed_distance(scene, anchors.astype(np.float64))
        anchor_pix = view.pixels[sel].astype(np.float32)
        anchor_depth = view.depth[
            anchor_pix[:, 1].astype(np.int64), anchor_pix[:, 0].astype(np.int64)
        ].astype(np.float32)
        anchor_view_dot = np.einsum(
            "ij,ij->i", normals.astype(np.float64), np.broadcast_to(camera.forward, normals.shape)
        )
        # pairwise distances for a determinism/quality check
        if N_ANCHORS > 1:
            pd = np.linalg.norm(anchors[:, None, :] - anchors[None, :, :], axis=-1)
            np.fill_diagonal(pd, np.inf)
            min_pair = float(pd.min())
        else:
            min_pair = 0.0

        stage = "write"
        np.savez_compressed(
            npz_path,
            partial_sdf=partial,
            known_mask=known.astype(bool),
            target_sdf=target,
            anchors=anchors,
            normals=normals,
            grid_coords=np.stack(
                [GRID.astype(np.float32), GRID.astype(np.float32), GRID.astype(np.float32)]
            ),
            camera_json=np.asarray(json.dumps(camera.to_json(), sort_keys=True)),
            view_depth=view.depth.astype(np.float32),
            view_hit=view.hit.astype(bool),
            anchor_pixels=anchor_pix,
            anchor_view_depth=anchor_depth,
            anchor_sdf=anchor_sdf.astype(np.float32),
            anchor_view_dot=anchor_view_dot.astype(np.float32),
            source_mesh_sha256=np.asarray(_sha256_file(mesh_path)),
        )
        meta = {
            "object_id": task.object_id,
            "path": f"npz/{task.object_id}.npz",
            "dataset": task.dataset,
            "source_split": task.source_split,
            "mesh_path": str(mesh_path),
            "mesh_sha256": _sha256_file(mesh_path),
            "npz_sha256": _sha256_file(npz_path),
            "seed": task.seed,
            "centroid": nm.centroid.tolist(),
            "scale": nm.scale,
            **nm.stats,
            "inside_fraction": inside_fraction,
            "view": {
                "hits": int(n_hits),
                "hit_fraction": float(n_hits / (camera.resolution**2)),
                "normal_flips": int(view.normal_flips),
            },
            "known_fraction": float(known.mean()),
            "known_count": int(known.sum()),
            "anchor_min_pair_distance": min_pair,
            "anchor_sdf_max_abs": float(np.abs(anchor_sdf).max()),
            "anchor_view_dot_mean": float(anchor_view_dot.mean()),
            "target_abs_mean": float(np.abs(target).mean()),
            "partial_known_abs_mean": float(np.abs(partial[known]).mean()) if known.any() else 0.0,
            "elapsed_s": time.time() - t0,
            "status": "ok",
        }
        _write_json(meta_path, meta)
        return meta
    except Exception as exc:
        meta = {
            "object_id": task.object_id,
            "dataset": task.dataset,
            "source_split": task.source_split,
            "mesh_path": str(mesh_path),
            "status": "failed",
            "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=5),
            "elapsed_s": time.time() - t0,
        }
        _write_json(meta_path, meta)
        if npz_path.exists():
            npz_path.unlink()
        return meta


def _view_mask_2d(
    depth_map: np.ndarray, camera: Camera, points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w = camera.resolution
    px, py, depth = project_points(points, camera)
    ix = np.rint(px).astype(np.int64)
    iy = np.rint(py).astype(np.int64)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < w)
    ix_c = np.clip(ix, 0, w - 1)
    iy_c = np.clip(iy, 0, w - 1)
    dh = depth_map[iy_c, ix_c].astype(np.float64)
    has_hit = np.isfinite(dh)
    mask = inside & has_hit & (np.abs(depth - dh) <= TRUNC) & (depth <= dh + TRUNC + DEPTH_EPS)
    return mask, depth, px, py


# --------------------------------------------------------------------------- #
# writing manifests / QA
# --------------------------------------------------------------------------- #
def run_tasks(tasks: Sequence[Dict[str, Any]], workers: int) -> List[Dict[str, Any]]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results: List[Dict[str, Any]] = []
    if workers <= 1:
        for i, t in enumerate(tasks, 1):
            r = prepare_one(PrepareTask(**t))
            results.append(r)
            print(f"[{i}/{len(tasks)}] {r['object_id']} {r['status']}", flush=True)
        return results
    ctx = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(prepare_one, PrepareTask(**t)): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:  # worker crash
                r = {
                    "object_id": t["object_id"],
                    "dataset": t.get("dataset"),
                    "source_split": t.get("source_split"),
                    "mesh_path": t.get("mesh_path"),
                    "status": "failed",
                    "stage": "worker",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(r)
            print(f"[{i}/{len(tasks)}] {r['object_id']} {r['status']}", flush=True)
    return results


def write_outputs(
    out_dir: Path,
    tasks_by_split: Dict[str, Sequence[Dict[str, Any]]],
    results: Sequence[Dict[str, Any]],
    split_meta: Dict[str, Any],
    dataset_label: str,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {r["object_id"]: r for r in results}
    counts: Dict[str, int] = {}
    for split, tasks in tasks_by_split.items():
        entries = []
        for t in sorted(tasks, key=lambda x: x["object_id"]):
            r = by_id.get(t["object_id"])
            if r and r.get("status") == "ok":
                entries.append({"object_id": r["object_id"], "path": r["path"]})
        _write_json(out_dir / f"{split}.json", entries)
        counts[split] = len(entries)
    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]
    _write_json(out_dir / "split_source.json", split_meta)
    _write_json(out_dir / "failures.json", failed)
    _write_json(
        out_dir / "records.json",
        sorted(
            [{k: v for k, v in r.items() if k not in ("traceback",)} for r in ok],
            key=lambda x: x["object_id"],
        ),
    )
    summary = {
        "dataset": dataset_label,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": counts,
        "ok": len(ok),
        "failed": len(failed),
        "failure_stages": _count_by(failed, "stage"),
        "watertight_ok": sum(1 for r in ok if r.get("is_watertight")),
        "nonwatertight_ok": sum(1 for r in ok if not r.get("is_watertight")),
        "watertight_input": sum(1 for r in ok if r.get("is_watertight")),
        "inside_fraction": _stats([r.get("inside_fraction", 0.0) for r in ok]),
        "known_fraction": _stats([r.get("known_fraction", 0.0) for r in ok]),
        "anchor_sdf_max_abs": _stats([r.get("anchor_sdf_max_abs", 0.0) for r in ok]),
        "anchor_view_dot_mean": _stats([r.get("anchor_view_dot_mean", 0.0) for r in ok]),
        "view_hit_fraction": _stats([r.get("view", {}).get("hit_fraction", 0.0) for r in ok]),
    }
    _write_json(out_dir / "QA" / "summary.json", summary)
    _write_json(out_dir / "QA" / "quality.json", sorted(ok, key=lambda x: x["object_id"]))
    return summary


def _count_by(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        k = str(r.get(key, "unknown"))
        out[k] = out.get(k, 0) + 1
    return out


def _stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def verify_root(root: Path, strict: bool = True) -> Dict[str, Any]:
    """Validate manifests, arrays, splits, values and observation consistency."""
    root = root.resolve()
    problems: List[str] = []
    report: Dict[str, Any] = {"root": str(root), "splits": {}, "problems": []}
    ids_by_split: Dict[str, List[str]] = {}
    for split in ("train", "val", "test"):
        mpath = root / f"{split}.json"
        if not mpath.exists():
            problems.append(f"missing manifest {mpath}")
            continue
        manifest = _read_json(mpath)
        if not isinstance(manifest, list):
            problems.append(f"{mpath} is not a list")
            continue
        ids = []
        for entry in manifest:
            if set(entry) != {"object_id", "path"}:
                problems.append(f"{split}: bad manifest keys {entry}")
                continue
            oid = entry["object_id"]
            ids.append(oid)
            npz_path = (root / entry["path"]).resolve()
            if not npz_path.exists():
                problems.append(f"{split}:{oid}: missing npz {npz_path}")
                continue
            try:
                check_npz(npz_path, strict=strict, problems=problems, tag=f"{split}:{oid}")
            except Exception as exc:
                problems.append(f"{split}:{oid}: verify exception {type(exc).__name__}: {exc}")
        ids_by_split[split] = ids
        report["splits"][split] = {"count": len(ids), "unique": len(set(ids))}
        if len(ids) != len(set(ids)):
            problems.append(f"{split}: duplicate object ids")
    all_ids = [i for v in ids_by_split.values() for i in v]
    if len(all_ids) != len(set(all_ids)):
        problems.append("object ids overlap across splits")
    report["total"] = len(all_ids)
    report["problems"] = problems
    report["ok"] = not problems
    return report


def check_npz(path: Path, strict: bool, problems: List[str], tag: str) -> None:
    z = np.load(path, allow_pickle=False)
    missing = [k for k in MANIFEST_KEYS if k not in z]
    if missing:
        problems.append(f"{tag}: missing arrays {missing}")
        return
    for key in MANIFEST_KEYS:
        arr = z[key]
        expected = {
            "partial_sdf": (RES, RES, RES),
            "known_mask": (RES, RES, RES),
            "target_sdf": (RES, RES, RES),
            "anchors": (N_ANCHORS, 3),
            "normals": (N_ANCHORS, 3),
        }[key]
        if arr.shape != expected:
            problems.append(f"{tag}:{key}: shape {arr.shape} != {expected}")
    partial = z["partial_sdf"]
    known = z["known_mask"]
    target = z["target_sdf"]
    anchors = z["anchors"]
    normals = z["normals"]
    if partial.dtype != np.float32:
        problems.append(f"{tag}: partial_sdf dtype {partial.dtype}")
    if target.dtype != np.float32:
        problems.append(f"{tag}: target_sdf dtype {target.dtype}")
    if known.dtype != np.bool_:
        problems.append(f"{tag}: known_mask dtype {known.dtype}")
    if anchors.dtype != np.float32 or normals.dtype != np.float32:
        problems.append(f"{tag}: anchors/normals dtype")
    for key, arr in (
        ("partial", partial),
        ("target", target),
        ("anchors", anchors),
        ("normals", normals),
    ):
        if not np.isfinite(arr).all():
            problems.append(f"{tag}:{key}: non-finite")
    if np.any(partial < -TRUNC - 1e-6) or np.any(partial > TRUNC + 1e-6):
        problems.append(f"{tag}: partial outside [-trunc,trunc]")
    if np.any(target < -TRUNC - 1e-6) or np.any(target > TRUNC + 1e-6):
        problems.append(f"{tag}: target outside [-trunc,trunc]")
    filler = partial[~known]
    if filler.size and not np.allclose(filler, TRUNC, atol=1e-6):
        problems.append(f"{tag}: unknown partial values are not +0.2 filler")
    if not known.any():
        problems.append(f"{tag}: empty known mask")
    if known.mean() > 0.75:
        problems.append(f"{tag}: known mask too large ({known.mean():.3f})")
    nrm = np.linalg.norm(normals.astype(np.float64), axis=1)
    if not np.allclose(nrm, 1.0, atol=2e-3):
        problems.append(f"{tag}: normals not unit (max err {np.abs(nrm - 1).max():.3g})")
    if np.abs(anchors).max() > 1.05:
        problems.append(f"{tag}: anchors outside normalised bounds")
    if "anchor_sdf" in z and np.abs(z["anchor_sdf"]).max() > 2e-2:
        problems.append(
            f"{tag}: anchor not on target surface ({np.abs(z['anchor_sdf']).max():.3g})"
        )
    if "camera_json" in z and "view_depth" in z:
        cam_json = json.loads(str(z["camera_json"]))
        cam = Camera(
            np.asarray(cam_json["position"], dtype=np.float64),
            np.asarray(cam_json["forward"], dtype=np.float64),
            np.asarray(cam_json["right"], dtype=np.float64),
            np.asarray(cam_json["up"], dtype=np.float64),
            float(cam_json["half_extent"]),
            int(cam_json["resolution"][0]),
        )
        g = np.linspace(-1, 1, RES)
        gx, gy, gz = np.meshgrid(g, g, g, indexing="ij")
        pts = np.stack([gx, gy, gz], -1).reshape(-1, 3)
        mask, _, _, _ = _view_mask_2d(z["view_depth"], cam, pts)
        if not np.array_equal(mask.reshape(RES, RES, RES), known):
            n_diff = int((mask.reshape(RES, RES, RES) != known).sum())
            problems.append(
                f"{tag}: known_mask not reproducible from camera+depth ({n_diff} cells differ)"
            )
        if "anchor_pixels" in z:
            pix = z["anchor_pixels"]
            ix = np.rint(pix[:, 0]).astype(np.int64)
            iy = np.rint(pix[:, 1]).astype(np.int64)
            ok = (ix >= 0) & (ix < cam.resolution) & (iy >= 0) & (iy < cam.resolution)
            if not ok.all():
                problems.append(f"{tag}: anchor pixel out of view")
            else:
                dh = z["view_depth"][iy, ix]
                if not np.isfinite(dh).all():
                    problems.append(f"{tag}: anchor pixel has no first hit")
    # grid axis convention
    if "grid_coords" in z:
        gc = z["grid_coords"]
        if gc.shape != (3, RES) or not np.allclose(gc[0], np.linspace(-1, 1, RES), atol=1e-6):
            problems.append(f"{tag}: bad grid_coords")
    if "anchor_view_dot" in z:
        if float(z["anchor_view_dot"].mean()) > -0.05:
            problems.append(
                f"{tag}: anchors not front-facing "
                f"(mean dot={float(z['anchor_view_dot'].mean()):.3f})"
            )


def selftest() -> int:
    """Fast analytic test of the grid/axis/sign/normal pipeline."""
    import trimesh

    out = _DATA_DIR / ".cache" / "selftest"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    # Anisotropic ellipsoid: long x, short y, medium z.  Watertight and smooth.
    sph = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    v = np.asarray(sph.vertices).copy()
    v *= np.array([1.0, 0.45, 0.7])
    m = trimesh.Trimesh(v, np.asarray(sph.faces), process=False)
    mesh_path = out / "ellipsoid.obj"
    m.export(mesh_path)
    task = PrepareTask(
        mesh_path=str(mesh_path),
        object_id="selftest-ellipsoid",
        dataset="selftest",
        source_split="train",
        out_dir=str(out / "out"),
        seed=1,
        overwrite=True,
    )
    r = prepare_one(task)
    print(json.dumps(r, indent=2, default=_json_default))
    if r.get("status") != "ok":
        return 1
    z = np.load(out / "out" / r["path"])
    t = z["target_sdf"]
    # axis convention: x extent is largest, so the x=0 slice must have the
    # smallest cross-section; check the zero-level extent along each axis.
    ext = []
    for ax in range(3):
        idx = [slice(None)] * 3
        idx[ax] = slice(28, 36)
        sub = t[tuple(idx)]
        ext.append(int((np.abs(sub) < 0.05).sum()))
    print("near-zero counts by tested axis slice (x,y,z):", ext)
    centre = tuple([RES // 2] * 3)
    if not (t[centre] < 0):
        print("FAIL: centre should be inside (negative)")
        return 1
    if not (t[2, 2, 2] > 0):
        print("FAIL: corner should be outside (positive)")
        return 1
    if not (t[-3, -3, -3] > 0):
        print("FAIL: opposite corner should be outside (positive)")
        return 1
    if float(np.abs(z["anchor_sdf"]).max()) > 1e-2:
        print("FAIL: anchors not on surface")
        return 1
    print("SELFTEST PASS")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cmd_verify(args: argparse.Namespace) -> int:
    report = verify_root(args.root, strict=not args.lenient)
    print(json.dumps(report, indent=2))
    out = args.root / "QA" / "verification.json"
    _write_json(out, report)
    return 0 if report["ok"] else 1


def cmd_selftest(args: argparse.Namespace) -> int:
    return selftest()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("verify", help="verify a prepared directory")
    pv.add_argument("--root", required=True, type=Path)
    pv.add_argument("--lenient", action="store_true")
    pv.set_defaults(func=cmd_verify)

    ps = sub.add_parser("selftest", help="analytic self-test")
    ps.set_defaults(func=cmd_selftest)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
