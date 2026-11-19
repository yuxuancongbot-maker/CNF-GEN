#!/usr/bin/env python3
"""Compact independent held-out evaluator for SDFusion + CNF chair completion.

Exactly six methods are compared, all from the same observation and shared
per-object DDIM seed: ``original`` (pretrained), ``plain``/``soft`` (matched
fine-tuned arms), ``post`` (Soft + post-hoc CNF), ``plain_post`` (plain +
post-hoc CNF) and ``cnf`` (CNF-trained arm with the same output correction).

The external SDFusion checkout and official checkpoints are configured with
``--sdfusion-root`` / ``--weights-dir`` or ``GCNF_SDFUSION_ROOT`` /
``GCNF_SDFUSION_WEIGHTS``.  Surface-sampling and matched-DDIM seeds are fixed
exactly as in the validated local run; no global RNG state is read or mutated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import METHODS, __version__, axes
from .cnf import HermiteGaussianCNF
from .conditioning import latent_observation_mask
from .conventions import TRUNCATION, grid_points, spacing
from .field import CubicSplineField
from .runtime import (
    ENV_UPSTREAM,
    ENV_WEIGHTS,
    PINNED_COMMIT,
    RuntimePaths,
    load_sdfusion,
)

sys.dont_write_bytecode = True

REQUIRED_NPZ_KEYS = ("partial_sdf", "known_mask", "target_sdf", "anchors", "normals")

SURFACE_LEVEL = 0.0
GRID_RESOLUTION = 64
N_SURFACE_SAMPLES = 10000
HIDDEN_TRUNCATION = TRUNCATION
CNF_SIGMA_SCALE = 2.0

# Fixed, explicit offsets passed straight to trimesh.sample.sample_surface.
# The 16-object pilot used uncontrolled global-RNG samples, so no numeric match
# with it is claimed. CNF and plain_post share one corrected offset so their
# paired comparison uses matched surface-sampling randomness.
SURFACE_SAMPLING_OFFSETS = {
    "original": 101,
    "plain": 211,
    "soft": 307,
    "post": 401,
    "cnf": 503,
    "plain_post": 601,
}
CORRECTED_SAMPLING_OFFSETS = {"post": 907, "cnf": 911, "plain_post": 911}
PAIRED_BOOTSTRAP_SEED = 111
PAIRED_BOOTSTRAP_RESAMPLES = 10000


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _as_array(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _decode_npz_scalar(arr: Any) -> Any:
    if isinstance(arr, np.ndarray):
        if arr.dtype.kind in ("U", "S", "O"):
            if arr.ndim == 0:
                return str(arr.item())
            flat = arr.reshape(-1)
            if flat.size == 1:
                return str(flat[0])
            return [str(v) for v in flat.tolist()]
        if arr.ndim == 0:
            return arr.item()
    return arr


def load_manifest_entries(manifest_path: Path, count: int) -> List[Dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as fh:
        entries = json.load(fh)
    if not isinstance(entries, list):
        raise ValueError(f"manifest is not a JSON list: {manifest_path}")
    if count is not None and count > 0:
        entries = entries[:count]
    return entries


def _parse_budget_from_name(name: str) -> Optional[int]:
    m = re.search(r"(?:^|[_-])s(\d+)(?:[_-]|\.|$)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"steps?(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def select_checkpoints(ckpt_dir: Path) -> Tuple[Dict[str, Path], int]:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {ckpt_dir}")

    all_files = sorted(
        p
        for p in ckpt_dir.iterdir()
        if p.is_file()
        and p.suffix == ".pt"
        and "_periodic" not in p.name
        and ".tmp" not in p.name
        and not p.name.endswith(".tmp")
    )

    selected: Dict[str, Path] = {}
    budgets: Dict[str, int] = {}

    for arm in ("plain", "soft", "cnf"):
        candidates = []
        for p in all_files:
            stem_tokens = p.stem.replace("-", "_").split("_")
            if arm in stem_tokens:
                candidates.append(p)

        if not candidates:
            raise FileNotFoundError(
                f"no checkpoint found for arm '{arm}' in {ckpt_dir} "
                f"(expected a filename containing a '{arm}' token)"
            )

        parsed = [(p, _parse_budget_from_name(p.name)) for p in candidates]
        with_budget = [(p, b) for p, b in parsed if b is not None]
        if not with_budget:
            raise ValueError(
                f"could not parse a step budget from any {arm} checkpoint: "
                f"{[p.name for p in candidates]}"
            )

        max_budget = max(b for _, b in with_budget)
        finalists = [p for p, b in with_budget if b == max_budget]
        if len(finalists) > 1:
            raise ValueError(
                f"ambiguous {arm} checkpoints at budget {max_budget}: {[p.name for p in finalists]}"
            )

        selected[arm] = finalists[0]
        budgets[arm] = max_budget

    if len({budgets[a] for a in budgets}) != 1:
        raise ValueError(f"trained arms do not share one budget: {budgets}")

    return selected, budgets["plain"]


@dataclass
class Camera:
    position: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    half_extent: float
    resolution: int
    depth: np.ndarray
    depth_valid: np.ndarray


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return v
    return v / n


def extract_camera_and_depth(extras: Dict[str, Any]) -> Optional[Camera]:
    depth = None
    depth_valid = None
    if "view_depth" in extras:
        depth = _as_array(extras["view_depth"]).astype(np.float64)
    elif "depth" in extras:
        depth = _as_array(extras["depth"]).astype(np.float64)
    if "view_hit" in extras:
        depth_valid = _as_array(extras["view_hit"]).astype(bool)
    elif "depth_valid" in extras:
        depth_valid = _as_array(extras["depth_valid"]).astype(bool)

    if depth is None:
        return None
    if depth_valid is None:
        depth_valid = np.isfinite(depth)

    position = None
    forward = None
    right = None
    up = None
    half_extent = 1.02
    resolution = int(depth.shape[0])

    if "camera_json" in extras:
        cj = _decode_npz_scalar(extras["camera_json"])
        if isinstance(cj, str):
            cj = json.loads(cj)
        position = np.asarray(cj["position"], dtype=np.float64)
        forward = _normalize(np.asarray(cj["forward"], dtype=np.float64))
        right = _normalize(
            np.asarray(cj.get("right", np.cross(forward, [0, 1, 0])), dtype=np.float64)
        )
        up = _normalize(np.asarray(cj.get("up", np.cross(right, forward)), dtype=np.float64))
        half_extent = float(cj.get("half_extent", half_extent))
        res = cj.get("resolution", [resolution, resolution])
        resolution = int(res[0] if isinstance(res, (list, tuple)) else res)
    elif "camera_origin" in extras:
        position = np.asarray(extras["camera_origin"], dtype=np.float64)
        forward = _normalize(np.asarray(extras["camera_forward"], dtype=np.float64))
        up_hint = np.asarray(extras.get("camera_up", [0, 1, 0]), dtype=np.float64)
        up = _normalize(up_hint)
        right = _normalize(np.cross(up, forward))
        if np.linalg.norm(right) <= 1e-12:
            right = np.asarray([1.0, 0.0, 0.0])
            up = np.asarray([0.0, 1.0, 0.0])
    else:
        position = np.asarray([0.0, 0.0, 3.0], dtype=np.float64)
        forward = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = _normalize(np.cross(up, forward))

    return Camera(
        position=position,
        forward=forward,
        right=right,
        up=up,
        half_extent=half_extent,
        resolution=resolution,
        depth=depth,
        depth_valid=depth_valid,
    )


def classify_visible(points: np.ndarray, cam: Camera) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0,), dtype=bool)
    rel = pts - cam.position[None, :]
    u = rel @ cam.right
    v = rel @ cam.up
    depth = rel @ cam.forward
    w = cam.resolution
    px = (u / cam.half_extent + 1.0) * 0.5 * (w - 1)
    py = (v / cam.half_extent + 1.0) * 0.5 * (w - 1)
    ix = np.rint(px).astype(np.int64)
    iy = np.rint(py).astype(np.int64)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < w)
    ix_c = np.clip(ix, 0, w - 1)
    iy_c = np.clip(iy, 0, w - 1)
    dh = cam.depth[iy_c, ix_c]
    has_hit = cam.depth_valid[iy_c, ix_c]
    near = np.abs(depth - dh) <= HIDDEN_TRUNCATION
    in_front = depth <= dh + HIDDEN_TRUNCATION + 1e-5
    return inside & has_hit & near & in_front


def extract_surface(sdf_xyz: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    from skimage.measure import marching_cubes

    arr = np.asarray(sdf_xyz, dtype=np.float64)
    if arr.shape != (GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION):
        raise ValueError(f"unexpected SDF shape {arr.shape}")
    h = spacing(GRID_RESOLUTION)
    try:
        verts, faces, _normals, _values = marching_cubes(
            arr, level=SURFACE_LEVEL, spacing=(h, h, h)
        )
    except Exception:
        return None
    if verts is None or faces is None or len(verts) == 0 or len(faces) == 0:
        return None
    verts = np.asarray(verts, dtype=np.float64) - 1.0
    faces = np.asarray(faces, dtype=np.int64)
    if not np.isfinite(verts).all():
        return None
    return verts, faces


def sample_surface_points(verts: np.ndarray, faces: np.ndarray, n: int, seed: int) -> np.ndarray:
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # trimesh>=4 derives its sampler RNG via random_generator(seed) and ignores
    # numpy's global RNG state. Pass the seed explicitly; do not read or mutate
    # global np.random state (the unseeded pilot did, which made samples
    # order-dependent and non-reproducible).
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (n, 3):
        if pts.shape[0] >= n:
            pts = pts[:n]
        else:
            pad = n - pts.shape[0]
            pts = np.concatenate([pts, np.repeat(pts[-1:], pad, axis=0)], axis=0)
    return pts


def chamfer_symmetric(pred: np.ndarray, gt: np.ndarray) -> float:
    from scipy.spatial import cKDTree  # optional eval dependency

    if pred is None or gt is None or len(pred) == 0 or len(gt) == 0:
        return float("nan")
    tree_pred = cKDTree(pred)
    d_gt_to_pred, _ = tree_pred.query(gt)
    term1 = float(np.mean(np.square(d_gt_to_pred)))
    tree_gt = cKDTree(gt)
    d_pred_to_gt, _ = tree_gt.query(pred)
    term2 = float(np.mean(np.square(d_pred_to_gt)))
    return term1 + term2


def paired_comparison(
    per_shape: List[Dict[str, Any]],
    metric: str,
    arm_a: str,
    arm_b: str,
    bootstrap_seed: int = PAIRED_BOOTSTRAP_SEED,
    n_boot: int = PAIRED_BOOTSTRAP_RESAMPLES,
) -> Dict[str, Any]:
    """Paired arm_a-vs-arm_b comparison on a lower-is-better metric.

    diff = arm_a - arm_b, so a negative diff means arm_a is better. Only
    shapes with a finite value for both arms are used; missing/non-finite
    shapes are reported rather than silently dropped.
    """
    pairs: List[Tuple[str, float, float]] = []
    skipped: List[str] = []
    for r in per_shape:
        oid = str(r.get("object_id"))
        va = r.get(metric, {}).get(arm_a)
        vb = r.get(metric, {}).get(arm_b)
        if va is None or vb is None:
            skipped.append(oid)
            continue
        va, vb = float(va), float(vb)
        if not (math.isfinite(va) and math.isfinite(vb)):
            skipped.append(oid)
            continue
        pairs.append((oid, va, vb))

    result: Dict[str, Any] = {
        "metric": metric,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "direction": "diff = arm_a - arm_b; negative means arm_a better (lower Chamfer)",
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": n_boot,
    }
    if not pairs:
        result.update({"n": 0, "skipped_object_ids": skipped, "note": "no finite paired shapes"})
        return result

    diffs = np.asarray([a - b for _, a, b in pairs], dtype=np.float64)
    n = int(diffs.size)
    rng = np.random.default_rng(bootstrap_seed)
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[boot_idx].mean(axis=1)
    result.update(
        {
            "n": n,
            "skipped_object_ids": skipped,
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "bootstrap_ci95_mean_diff": [
                float(np.percentile(boot_means, 2.5)),
                float(np.percentile(boot_means, 97.5)),
            ],
            "wins_arm_a": int(np.sum(diffs < 0)),
            "wins_arm_b": int(np.sum(diffs > 0)),
            "ties": int(np.sum(diffs == 0)),
            "per_shape_diff": {oid: float(d) for (oid, _, _), d in zip(pairs, diffs)},
        }
    )
    return result


def anchor_residuals(
    sdf_xyz: np.ndarray, anchors: np.ndarray, normals: np.ndarray
) -> Dict[str, float]:
    field = CubicSplineField(torch.from_numpy(np.asarray(sdf_xyz, dtype=np.float64)))
    a = torch.from_numpy(np.asarray(anchors, dtype=np.float64))
    n = torch.from_numpy(np.asarray(normals, dtype=np.float64))
    with torch.no_grad():
        vals, grads = field.evaluate(a, with_grad=True)
    vals = vals.to(torch.float64)
    grads = grads.to(torch.float64)
    value_err = (vals - 0.0).abs()
    grad_err = (grads - n).abs()
    return {
        "anchor_value_abs_max": float(value_err.max()) if value_err.numel() else float("nan"),
        "anchor_grad_abs_max": float(grad_err.max()) if grad_err.numel() else float("nan"),
        "anchor_value_rms": float(torch.sqrt((value_err**2).mean()))
        if value_err.numel()
        else float("nan"),
        "anchor_grad_rms": float(torch.sqrt((grad_err**2).mean()))
        if grad_err.numel()
        else float("nan"),
    }


def apply_cnf_correction(
    base_sdf_xyz: np.ndarray,
    anchors: np.ndarray,
    normals: np.ndarray,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    cnf = HermiteGaussianCNF.for_resolution(GRID_RESOLUTION, sigma_scale=CNF_SIGMA_SCALE)
    base = np.asarray(base_sdf_xyz, dtype=np.float64)
    a = np.asarray(anchors, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)

    field = CubicSplineField(torch.from_numpy(base))
    anchors_t = torch.from_numpy(a)
    normals_t = torch.from_numpy(nrm)

    with torch.no_grad():
        a_vals, a_grads = field.evaluate(anchors_t, with_grad=True)
    a_vals = a_vals.to(torch.float64)
    a_grads = a_grads.to(torch.float64)

    base_ops = torch.stack([a_vals, a_grads[:, 0], a_grads[:, 1], a_grads[:, 2]], dim=-1)
    targets = torch.zeros_like(base_ops)
    targets[:, 1:] = normals_t.to(torch.float64)

    diag: Dict[str, Any] = {}
    psi = None
    try:
        psi, solve_diag = cnf.solve(base_ops, targets, anchors_t, diagnostics=True)
        diag.update(solve_diag.as_dict())
    except Exception as exc:
        diag["solve_error"] = str(exc)
        if hasattr(exc, "diagnostics"):
            try:
                diag.update(exc.diagnostics.as_dict())
            except Exception:
                pass
        return None, diag

    try:
        corr_val, corr_grad = cnf.correction(psi, anchors_t, anchors_t, with_grad=True)
        corr_val = corr_val.to(torch.float64)
        corr_grad = corr_grad.to(torch.float64)
        corrected_ops = base_ops.to(torch.float64) + torch.stack(
            [corr_val, corr_grad[:, 0], corr_grad[:, 1], corr_grad[:, 2]], dim=-1
        )
        err = corrected_ops - targets.to(torch.float64)
        diag["anchor_value_abs_err_max"] = float(err[:, 0].abs().max())
        diag["anchor_grad_abs_err_max"] = float(err[:, 1:].abs().max())
        diag["anchor_abs_err_max"] = float(err.abs().max())
        diag["n_anchors_checked"] = int(anchors.shape[0])
    except Exception as exc:
        diag["correction_error"] = str(exc)

    try:
        grid = grid_points(GRID_RESOLUTION, dtype=torch.float64)
        grid_pts = grid.reshape(-1, 3)
        corr_vals, _ = cnf.correction(psi, anchors_t, grid_pts, with_grad=False)
        corr_vals = corr_vals.to(torch.float64).reshape(
            GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION
        )
        corrected = base + corr_vals.detach().cpu().numpy().astype(np.float64)
        return corrected.astype(np.float32), diag
    except Exception as exc:
        diag["grid_correction_error"] = str(exc)
        return None, diag


def run_ddim_arm(
    runtime: Any,
    partial_sdf: np.ndarray,
    known_mask: np.ndarray,
    ddim_steps: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    partial_t = (
        torch.from_numpy(np.asarray(partial_sdf, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    )
    known_t = torch.from_numpy(np.asarray(known_mask, dtype=bool)).unsqueeze(0).unsqueeze(0)

    partial_zyx = axes.xyz_to_zyx(partial_t)
    known_zyx = axes.xyz_to_zyx(known_t)

    device = torch.device(runtime.device if hasattr(runtime, "device") else "cuda")
    partial_zyx = partial_zyx.to(device)
    known_zyx = known_zyx.to(device)

    with torch.no_grad():
        z_cond = runtime.vqvae.encode_no_quant(partial_zyx)
        latent_mask = latent_observation_mask(known_zyx, z_cond.shape[-1], reduce="half", erode=0)
        z_cond = z_cond.detach()
        latent_mask = latent_mask.detach()

        if int(latent_mask.sum().item()) <= 0:
            raise RuntimeError("empty latent observation mask")

        z, _intermediates = runtime.inpaint_latent(
            x0=z_cond,
            mask=latent_mask,
            n=1,
            ddim_steps=ddim_steps,
            eta=0.0,
            seed=seed,
        )
        sdf_zyx = runtime.decode_latent(z, quantize=True)

    sdf_xyz = axes.zyx_to_xyz(sdf_zyx)
    sdf_xyz = sdf_xyz.detach().to("cpu").squeeze(0).squeeze(0).to(torch.float32)
    return sdf_xyz.numpy(), {
        "latent_observed_count": int(latent_mask.sum().item()),
        "latent_observed_fraction": float(latent_mask.mean().item()),
        "known_voxel_count": int(known_t.sum().item()),
    }


def evaluate_shape(
    entry: Dict[str, Any],
    base_dir: Path,
    runtime: Any,
    selected_ckpts: Dict[str, Path],
    final_states: Dict[str, Any],
    ddim_steps: int,
    seed: int,
    surface_samples: int,
) -> Dict[str, Any]:
    path = Path(entry["path"])
    if not path.is_absolute():
        path = base_dir / path
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    for key in REQUIRED_NPZ_KEYS:
        if key not in arrays:
            raise ValueError(f"{path}: missing required key '{key}'")

    object_id = str(entry.get("object_id", path.stem))
    partial_sdf = np.asarray(arrays["partial_sdf"], dtype=np.float32)
    known_mask = np.asarray(arrays["known_mask"], dtype=bool)
    target_sdf = np.asarray(arrays["target_sdf"], dtype=np.float32)
    anchors = np.asarray(arrays["anchors"], dtype=np.float32)
    normals = np.asarray(arrays["normals"], dtype=np.float32)

    extras = {k: arrays[k] for k in arrays if k not in REQUIRED_NPZ_KEYS}
    cam = extract_camera_and_depth(extras)

    gt_mesh = extract_surface(target_sdf)
    gt_pts = None
    gt_visible_mask = None
    gt_hidden_pts = None
    if gt_mesh is not None:
        gt_pts = sample_surface_points(gt_mesh[0], gt_mesh[1], surface_samples, seed + 7919)
        if cam is not None:
            gt_visible_mask = classify_visible(gt_pts, cam)
            gt_hidden_pts = (
                gt_pts[~gt_visible_mask] if (~gt_visible_mask).any() else np.zeros((0, 3))
            )

    arm_order = list(METHODS)
    ckpt_key = {
        "original": None,
        "plain": "plain",
        "soft": "soft",
        "post": "soft",
        "cnf": "cnf",
        "plain_post": "plain",
    }

    sdf_cache: Dict[str, np.ndarray] = {}
    arm_latent_stats: Dict[str, Dict[str, Any]] = {}

    for arm in ["original", "plain", "soft", "cnf"]:
        if arm != "original":
            # final_states is loaded once up front (host RAM >> 3 x 2.4 GB);
            # load_state_dict copies into the runtime weights, so reusing the
            # cached tensors yields byte-identical output to a per-object load.
            runtime.unet.load_state_dict(final_states[ckpt_key[arm]], strict=True)
        runtime.unet.eval()
        runtime.vqvae.eval()

        sdf_xyz, latent_stats = run_ddim_arm(runtime, partial_sdf, known_mask, ddim_steps, seed)
        sdf_cache[arm] = sdf_xyz
        arm_latent_stats[arm] = latent_stats

    sdf_cache["post"] = sdf_cache["soft"].copy()
    # plain_post reuses the exact generated plain SDF; no extra DDIM run.
    sdf_cache["plain_post"] = sdf_cache["plain"].copy()

    per_arm_surface: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
    per_arm_points: Dict[str, Optional[np.ndarray]] = {}
    for arm in arm_order:
        per_arm_surface[arm] = extract_surface(sdf_cache[arm])
        if per_arm_surface[arm] is not None:
            off = SURFACE_SAMPLING_OFFSETS[arm]
            per_arm_points[arm] = sample_surface_points(
                per_arm_surface[arm][0], per_arm_surface[arm][1], surface_samples, seed + off
            )
        else:
            per_arm_points[arm] = None

    anchor_residual_map: Dict[str, Any] = {}
    correction_failures: Dict[str, str] = {}
    for arm in arm_order:
        base_resid = anchor_residuals(sdf_cache[arm], anchors, normals)
        entry_resid: Dict[str, Any] = {"base": base_resid}
        if arm in ("post", "cnf", "plain_post"):
            corrected_sdf, diag = apply_cnf_correction(sdf_cache[arm], anchors, normals)
            if corrected_sdf is not None:
                entry_resid["corrected"] = anchor_residuals(corrected_sdf, anchors, normals)
                sdf_cache[arm] = corrected_sdf
                per_arm_surface[arm] = extract_surface(corrected_sdf)
                if per_arm_surface[arm] is not None:
                    off = CORRECTED_SAMPLING_OFFSETS[arm]
                    per_arm_points[arm] = sample_surface_points(
                        per_arm_surface[arm][0],
                        per_arm_surface[arm][1],
                        surface_samples,
                        seed + off,
                    )
                else:
                    # Corrected mesh could not be extracted: do NOT keep the
                    # uncorrected base-surface points sampled earlier.
                    per_arm_surface[arm] = None
                    per_arm_points[arm] = None
                    correction_failures[arm] = "corrected_mesh_extraction_failed"
                entry_resid["cnf_solve"] = diag
            else:
                entry_resid["corrected"] = None
                entry_resid["cnf_solve"] = diag
                # Failed solve: no usable corrected metric for this arm. Drop
                # the base-mesh points sampled before correction.
                per_arm_surface[arm] = None
                per_arm_points[arm] = None
                correction_failures[arm] = "cnf_solve_failed"
        anchor_residual_map[arm] = entry_resid

    full_cd: Dict[str, Optional[float]] = {}
    hidden_cd: Dict[str, Optional[float]] = {}
    failures: Dict[str, str] = {}

    for arm in arm_order:
        pts = per_arm_points[arm]
        if pts is None:
            failures[arm] = correction_failures.get(arm, "no_surface_extracted")
            full_cd[arm] = None
            hidden_cd[arm] = None
            continue
        if gt_pts is None:
            failures[arm] = "no_gt_surface"
            full_cd[arm] = None
            hidden_cd[arm] = None
            continue

        full_cd[arm] = chamfer_symmetric(pts, gt_pts)

        if cam is not None and gt_hidden_pts is not None and len(gt_hidden_pts) > 0:
            pred_visible = classify_visible(pts, cam)
            pred_hidden_pts = pts[~pred_visible]
            if len(pred_hidden_pts) > 0:
                hidden_cd[arm] = chamfer_symmetric(pred_hidden_pts, gt_hidden_pts)
            else:
                hidden_cd[arm] = None
                failures.setdefault(arm, "no_hidden_predicted_points")
        else:
            hidden_cd[arm] = None

    gt_visible_fraction = None
    if gt_visible_mask is not None and len(gt_visible_mask) > 0:
        gt_visible_fraction = float(gt_visible_mask.mean())

    result = {
        "object_id": object_id,
        "npz_path": str(path),
        "seed": seed,
        "latent_stats": arm_latent_stats,
        "gt_surface_extracted": gt_mesh is not None,
        "gt_visible_fraction": gt_visible_fraction,
        "gt_hidden_point_count": int(len(gt_hidden_pts)) if gt_hidden_pts is not None else 0,
        "full_chamfer": full_cd,
        "hidden_chamfer": hidden_cd,
        "anchor_residuals": anchor_residual_map,
        "failures": failures,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--surface-samples", type=int, default=N_SURFACE_SAMPLES)
    parser.add_argument(
        "--sdfusion-root",
        default=os.environ.get(ENV_UPSTREAM),
        help=f"official SDFusion checkout (default: ${ENV_UPSTREAM})",
    )
    parser.add_argument(
        "--weights-dir",
        default=os.environ.get(ENV_WEIGHTS),
        help=f"directory with official .pth checkpoints (default: ${ENV_WEIGHTS})",
    )
    args = parser.parse_args()

    t0 = time.time()

    selected_ckpts, budget = select_checkpoints(args.checkpoint_dir)

    paths = RuntimePaths.resolve(args.sdfusion_root, args.weights_dir)
    runtime = load_sdfusion(device=args.device, renderer=False, verify_checksums=True, paths=paths)
    runtime.eval()

    official_state = {k: v.detach().clone() for k, v in runtime.unet.state_dict().items()}

    # Optional speed: CPU-cache each verified final state once rather than
    # re-reading the 2.4 GB files for every object (79 GB host RAM available).
    final_states = {
        arm: torch.load(selected_ckpts[arm], map_location="cpu", weights_only=True)
        for arm in ("plain", "soft", "cnf")
    }

    entries = load_manifest_entries(args.manifest, args.count)
    base_dir = args.manifest.resolve().parent

    per_shape: List[Dict[str, Any]] = []
    failures_all: List[Dict[str, Any]] = []

    for i, entry in enumerate(entries):
        obj_seed = args.seed + i
        runtime.unet.load_state_dict(official_state, strict=True)
        runtime.unet.eval()
        runtime.vqvae.eval()

        try:
            res = evaluate_shape(
                entry=entry,
                base_dir=base_dir,
                runtime=runtime,
                selected_ckpts=selected_ckpts,
                final_states=final_states,
                ddim_steps=args.ddim_steps,
                seed=obj_seed,
                surface_samples=args.surface_samples,
            )
            per_shape.append(res)
            if res.get("failures"):
                failures_all.append({"object_id": res["object_id"], "failures": res["failures"]})
            print(
                f"[eval] {i + 1}/{len(entries)} {res['object_id']} "
                f"full_cd(cnf)={res['full_chamfer'].get('cnf')}"
            )
        except Exception as exc:
            import traceback

            failures_all.append(
                {
                    "object_id": str(entry.get("object_id", i)),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[eval] {i + 1}/{len(entries)} FAILED: {exc}")
            traceback.print_exc()

    arms = list(METHODS)
    aggregate: Dict[str, Any] = {}
    for arm in arms:
        full_vals = [
            r["full_chamfer"].get(arm) for r in per_shape if r["full_chamfer"].get(arm) is not None
        ]
        hidden_vals = [
            r["hidden_chamfer"].get(arm)
            for r in per_shape
            if r["hidden_chamfer"].get(arm) is not None
        ]
        aggregate[arm] = {
            "full_chamfer_mean": float(np.mean(full_vals)) if full_vals else None,
            "full_chamfer_median": float(np.median(full_vals)) if full_vals else None,
            "full_chamfer_min": float(np.min(full_vals)) if full_vals else None,
            "full_chamfer_max": float(np.max(full_vals)) if full_vals else None,
            "full_chamfer_n": len(full_vals),
            "hidden_chamfer_mean": float(np.mean(hidden_vals)) if hidden_vals else None,
            "hidden_chamfer_median": float(np.median(hidden_vals)) if hidden_vals else None,
            "hidden_chamfer_min": float(np.min(hidden_vals)) if hidden_vals else None,
            "hidden_chamfer_max": float(np.max(hidden_vals)) if hidden_vals else None,
            "hidden_chamfer_n": len(hidden_vals),
        }

    anchor_agg: Dict[str, Any] = {}
    for arm in arms:
        base_vals = [
            r["anchor_residuals"][arm]["base"]["anchor_value_abs_max"]
            for r in per_shape
            if r["anchor_residuals"].get(arm, {}).get("base", {}).get("anchor_value_abs_max")
            is not None
        ]
        base_grads = [
            r["anchor_residuals"][arm]["base"]["anchor_grad_abs_max"]
            for r in per_shape
            if r["anchor_residuals"].get(arm, {}).get("base", {}).get("anchor_grad_abs_max")
            is not None
        ]
        anchor_agg[arm] = {
            "base_anchor_value_abs_max_mean": float(np.mean(base_vals)) if base_vals else None,
            "base_anchor_grad_abs_max_mean": float(np.mean(base_grads)) if base_grads else None,
        }
        if arm in ("post", "cnf", "plain_post"):
            corr_vals = [
                r["anchor_residuals"][arm].get("corrected", {}).get("anchor_value_abs_max")
                for r in per_shape
                if isinstance(r["anchor_residuals"][arm].get("corrected"), dict)
            ]
            corr_grads = [
                r["anchor_residuals"][arm].get("corrected", {}).get("anchor_grad_abs_max")
                for r in per_shape
                if isinstance(r["anchor_residuals"][arm].get("corrected"), dict)
            ]
            anchor_agg[arm]["corrected_anchor_value_abs_max_mean"] = (
                float(np.mean(corr_vals)) if corr_vals else None
            )
            anchor_agg[arm]["corrected_anchor_grad_abs_max_mean"] = (
                float(np.mean(corr_grads)) if corr_grads else None
            )

    cnf_solve_example = None
    for r in per_shape:
        d = r["anchor_residuals"].get("cnf", {}).get("cnf_solve")
        if isinstance(d, dict):
            cnf_solve_example = d
            break
    if cnf_solve_example is None:
        for r in per_shape:
            d = r["anchor_residuals"].get("post", {}).get("cnf_solve")
            if isinstance(d, dict):
                cnf_solve_example = d
                break

    # Paired comparisons on full and unobserved Chamfer: CNF-vs-plain_post,
    # CNF-vs-plain, and the primary plain_post-vs-plain.
    paired: Dict[str, Any] = {}
    for metric in ("full_chamfer", "hidden_chamfer"):
        for other in ("plain_post", "plain"):
            paired[f"cnf_vs_{other}_{metric}"] = paired_comparison(per_shape, metric, "cnf", other)
        paired[f"plain_post_vs_plain_{metric}"] = paired_comparison(
            per_shape, metric, "plain_post", "plain"
        )

    output = {
        "provenance": {
            "script": f"gcnf_shape_completion.evaluate v{__version__}",
            "pinned_sdfusion_commit": PINNED_COMMIT,
            "manifest": str(args.manifest.resolve()),
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "n_requested": args.count,
            "n_evaluated": len(per_shape),
            "ddim_steps": args.ddim_steps,
            "base_seed": args.seed,
            "surface_samples": args.surface_samples,
            "checkpoint_budget_steps": budget,
            "selected_checkpoints": {k: str(v) for k, v in selected_ckpts.items()},
            "selected_checkpoint_sha256": {k: _sha256_file(v) for k, v in selected_ckpts.items()},
            "runtime_upstream_commit": getattr(runtime, "checkpoint_report", {})
            .get("revision", {})
            .get("actual_commit"),
            "elapsed_s": round(time.time() - t0, 2),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "config": {
            "arms": arms,
            "units": (
                "normalized object coordinates in [-1, 1]^3; Chamfer is the sum of "
                "two directed mean squared nearest-neighbour distances (squared "
                "normalized length units). Marching cubes level set 0 on a 64^3 "
                "grid with inclusive endpoints (spacing 2/63)."
            ),
            "surface_extraction": "skimage.measure.marching_cubes level=0 on xyz grid",
            "surface_sampling": (
                f"trimesh.sample.sample_surface(mesh, {args.surface_samples}, seed=explicit) "
                "per (shape, arm); global numpy RNG is neither read nor modified"
            ),
            "hidden_definition": (
                "GT/predicted surface points classified against the original camera "
                "depth map; hidden points are those not on the observed first-hit band."
            ),
            "cnf": {
                "sigma_scale": CNF_SIGMA_SCALE,
                "solver_dtype": "float64",
                "correction_applied": ["post", "cnf", "plain_post"],
            },
            "seed_policy": (
                "one seed per object: base_seed + object index, shared across all arms; "
                "DDIM noise is seeded by that value (eta=0, n=1) and is identical across "
                "the four independently generated arms"
            ),
            "matched_ddim_seeds": {
                "base_seed": args.seed,
                "per_object_seed": "base_seed + object_index (0-based, manifest order)",
                "n_objects": len(entries),
                "seed_range": [args.seed, args.seed + max(0, len(entries) - 1)],
                "ddim_steps": args.ddim_steps,
                "ddim_eta": 0.0,
                "arms_generated_by_ddim": ["original", "plain", "soft", "cnf"],
                "arms_reusing_a_generated_sdf": {
                    "post": "copy of soft SDF (no extra DDIM run)",
                    "plain_post": "copy of plain SDF (no extra DDIM run)",
                },
                "shared_across_arms": [
                    "partial observation and known_mask",
                    "masked-latent DDIM initial noise for each (object, seed)",
                    "query anchors/normals and observation-region definition",
                ],
                "matched_corrected_surface_sampling": {
                    "cnf": "obj_seed + CORRECTED_SAMPLING_OFFSETS[cnf]",
                    "plain_post": "obj_seed + CORRECTED_SAMPLING_OFFSETS[plain_post]",
                    "equal_offset": (
                        CORRECTED_SAMPLING_OFFSETS["cnf"]
                        == CORRECTED_SAMPLING_OFFSETS["plain_post"]
                    ),
                },
            },
            "latent_mask": "half-block reduction of known_mask, zero erosion (matches training)",
            "quantized_decode": True,
            "surface_sampling_offsets": {
                "per_arm_uncorrected": SURFACE_SAMPLING_OFFSETS,
                "per_arm_corrected": CORRECTED_SAMPLING_OFFSETS,
                "plain_post_reuses": "exact generated plain SDF (no extra DDIM run)",
                "gt_surface_seed": "seed + 7919 (unchanged)",
            },
            "paired_comparison": {
                "bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
                "bootstrap_resamples": PAIRED_BOOTSTRAP_RESAMPLES,
                "unobserved_chamfer_field": "hidden_chamfer",
            },
            "note": (
                "Full 64-object held-out evaluation; extends the overnight pilot. "
                "The 16-object pilot used uncontrolled global-RNG surface samples, so its "
                "Chamfer values are not numerically comparable to this seeded run. "
                "No paper claims."
            ),
        },
        "aggregate": aggregate,
        "anchor_residual_aggregate": anchor_agg,
        "paired_comparisons": paired,
        "cnf_solve_example": cnf_solve_example,
        "per_shape": per_shape,
        "failures": failures_all,
    }

    _write_json_atomic(args.out, output)
    print(f"[eval] wrote {args.out} ({len(per_shape)} shapes, {time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
