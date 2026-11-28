"""Core geometry, RKHS coordinate warp, alignment, and metric utilities.

Implements the tangential-free Normal-Only Sobolev H^3 (Matern-5/2 + weakly
penalized affine) RKHS Coordinate Warp CNF:
    f_CNF(x) = f_0(x + u(x)),
    u(x) = sum_{j=1}^M k(x, p_j) w_j,
    k(x, y) = (1 + s + s^2/3) exp(-s) + gamma_tr + gamma_lin (x^T y),
    s = sqrt(5) ||x - y|| / ell.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

GRID_RES = 128
GRID_LO = -1.0
GRID_HI = 1.0
CANONICAL_BOX_BOUND = 1.01
CANONICAL_GRID_BOUND = 1.01
OUTSIDE_VALUE = 1.0
TRUNCATION = 0.2
N_SURFACE = 10000
BOOTSTRAP_SEED = 111
BOOTSTRAP_RESAMPLES = 10000
GAUSSIAN_SIGMA = 2.0 * (2.0 / 127.0)

DEFAULT_ELL = 0.65
DEFAULT_GAMMA_TR = 1.0
DEFAULT_GAMMA_LIN = 0.15
DEFAULT_REG_STAGE1 = 1e-4
DEFAULT_REG_STAGE1_RETRY = 2e-3


# ---------------------------------------------------------------------------
# I/O & hashing helpers
# ---------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Grid, coordinate transforms, surface extraction & Chamfer evaluation
# ---------------------------------------------------------------------------
def grid_points(
    resolution: int = GRID_RES,
    lo: float = GRID_LO,
    hi: float = GRID_HI,
    dtype: Any = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(lo, hi, resolution, dtype=dtype)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1).reshape(-1, 3), axis


def canonical_oob_mask(points: np.ndarray, bound: float = CANONICAL_BOX_BOUND) -> np.ndarray:
    """True where canonical coordinates exceed the valid Fourier-embedding box."""
    return np.max(np.abs(np.asarray(points, dtype=np.float64)), axis=1) > float(bound)


def transform_to_canonical(points: np.ndarray, alignment: dict[str, Any]) -> np.ndarray:
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    r = np.asarray(alignment["rotation"], dtype=np.float64)
    t = np.asarray(alignment["translation"], dtype=np.float64)
    s = float(alignment["scale"])
    return ((p - t[None, :]) @ r) / s


def grad_to_shapenet(grad_q: np.ndarray, alignment: dict[str, Any]) -> np.ndarray:
    r = np.asarray(alignment["rotation"], dtype=np.float64)
    s = float(alignment["scale"])
    return (np.atleast_2d(np.asarray(grad_q, dtype=np.float64)) @ r.T) / s


def camera_from_npz(z: Any) -> dict[str, Any]:
    return json.loads(str(z["camera_json"]))


def classify_visible(
    points: np.ndarray,
    cam: dict[str, Any],
    depth: np.ndarray,
    depth_valid: np.ndarray,
) -> np.ndarray:
    """First-hit depth-band visibility classifier matching the ShapeNet protocol."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0,), dtype=bool)
    pos = np.asarray(cam["position"], dtype=np.float64)
    rel = pts - pos[None, :]
    u = rel @ np.asarray(cam["right"], dtype=np.float64)
    v = rel @ np.asarray(cam["up"], dtype=np.float64)
    d = rel @ np.asarray(cam["forward"], dtype=np.float64)
    res = cam.get("resolution", [256, 256])
    w = int(res[0] if isinstance(res, (list, tuple)) else res)
    px = (u / float(cam["half_extent"]) + 1.0) * 0.5 * (w - 1)
    py = (v / float(cam["half_extent"]) + 1.0) * 0.5 * (w - 1)
    ix = np.rint(px).astype(np.int64)
    iy = np.rint(py).astype(np.int64)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < w)
    ix_c = np.clip(ix, 0, w - 1)
    iy_c = np.clip(iy, 0, w - 1)
    dh = np.asarray(depth, dtype=np.float64)[iy_c, ix_c]
    has_hit = np.asarray(depth_valid, dtype=bool)[iy_c, ix_c]
    near = np.abs(d - dh) <= TRUNCATION
    in_front = d <= dh + TRUNCATION + 1e-5
    return inside & has_hit & near & in_front


def extract_surface(
    volume: np.ndarray,
    lo: float = GRID_LO,
    hi: float = GRID_HI,
    resolution: int = GRID_RES,
    level: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    arr = np.asarray(volume, dtype=np.float64)
    if arr.shape != (resolution, resolution, resolution) or not np.isfinite(arr).all():
        return None
    try:
        spacing = ((hi - lo) / (resolution - 1),) * 3
        verts, faces, _, _ = marching_cubes(arr, level=level, spacing=spacing)
    except Exception:
        return None
    if len(verts) == 0 or len(faces) == 0:
        return None
    return np.asarray(verts, dtype=np.float64) + lo, np.asarray(faces, dtype=np.int64)


def sample_surface(vertices: np.ndarray, faces: np.ndarray, n: int, seed: int) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < n:
        pts = np.concatenate([pts, np.repeat(pts[-1:], n - len(pts), axis=0)], axis=0)
    return pts[:n]


def chamfer_symmetric(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred is None or gt is None or len(pred) == 0 or len(gt) == 0:
        return float("nan")
    d1 = cKDTree(pred).query(gt, k=1)[0]
    d2 = cKDTree(gt).query(pred, k=1)[0]
    return float(np.mean(d1**2) + np.mean(d2**2))


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    vn = np.zeros_like(verts, dtype=np.float64)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def mesh_roughness_fast(verts: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Vectorized adjacent-face dihedral angle stats (deg) & Laplacian roughness."""
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(v) == 0 or len(f) == 0:
        return {
            "dihedral_deg_mean": None,
            "dihedral_deg_p95": None,
            "dihedral_deg_p99": None,
            "laplacian_roughness": None,
            "n_interior_edges": 0,
            "mean_edge_len": None,
        }
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)

    all_e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    e_sorted = np.sort(all_e, axis=1)
    f_ids = np.tile(np.arange(len(f), dtype=np.int64), 3)

    n_v = int(len(v))
    keys = e_sorted[:, 0] * n_v + e_sorted[:, 1]
    order = np.argsort(keys, kind="mergesort")
    ks = keys[order]
    fs = f_ids[order]
    es = e_sorted[order]

    uniq_idx = np.flatnonzero(np.concatenate([[True], ks[1:] != ks[:-1]]))
    counts = np.diff(np.concatenate([uniq_idx, [len(ks)]]))
    ue = es[uniq_idx]
    mean_edge_len = float(np.mean(np.linalg.norm(v[ue[:, 0]] - v[ue[:, 1]], axis=1)))

    idx_int = uniq_idx[counts == 2]
    cos_ang = np.clip(np.einsum("ij,ij->i", fn[fs[idx_int]], fn[fs[idx_int + 1]]), -1.0, 1.0)
    dih_deg = np.degrees(np.arccos(cos_ang))

    deg = np.zeros(n_v, dtype=np.float64)
    nb_sum = np.zeros((n_v, 3), dtype=np.float64)
    np.add.at(deg, ue[:, 0], 1.0)
    np.add.at(deg, ue[:, 1], 1.0)
    np.add.at(nb_sum, ue[:, 0], v[ue[:, 1]])
    np.add.at(nb_sum, ue[:, 1], v[ue[:, 0]])
    has_nb = deg > 0
    lap = np.linalg.norm(v[has_nb] - nb_sum[has_nb] / deg[has_nb, None], axis=1)
    lap_rough = float(np.mean(lap) / max(mean_edge_len, 1e-12)) if len(lap) else None

    return {
        "dihedral_deg_mean": float(dih_deg.mean()) if len(dih_deg) else None,
        "dihedral_deg_p95": float(np.percentile(dih_deg, 95)) if len(dih_deg) else None,
        "dihedral_deg_p99": float(np.percentile(dih_deg, 99)) if len(dih_deg) else None,
        "laplacian_roughness": lap_rough,
        "n_interior_edges": int(len(dih_deg)),
        "mean_edge_len": mean_edge_len,
    }


# ---------------------------------------------------------------------------
# Strictly SPD C^2 Sobolev H^3 (Matern-5/2) + weakly penalized affine RKHS
# ---------------------------------------------------------------------------
def matern52_kernel(
    x: np.ndarray,
    y: np.ndarray,
    ell: float = DEFAULT_ELL,
    gamma_tr: float = DEFAULT_GAMMA_TR,
    gamma_lin: float = DEFAULT_GAMMA_LIN,
) -> np.ndarray:
    """Strictly positive-definite C^2 Matern-5/2 + weakly penalized affine kernel."""
    xa = np.atleast_2d(np.asarray(x, dtype=np.float64))
    ya = np.atleast_2d(np.asarray(y, dtype=np.float64))
    r = np.linalg.norm(xa[:, None, :] - ya[None, :, :], axis=-1)
    s = math.sqrt(5.0) * r / ell
    k = (1.0 + s + (s**2) / 3.0) * np.exp(-s)
    return k + gamma_tr + gamma_lin * (xa @ ya.T)


def eval_rkhs_u(
    x: np.ndarray,
    anchors: np.ndarray,
    weights: np.ndarray,
    ell: float = DEFAULT_ELL,
    gamma_tr: float = DEFAULT_GAMMA_TR,
    gamma_lin: float = DEFAULT_GAMMA_LIN,
    chunk: int = 131072,
) -> np.ndarray:
    """Evaluate displacement field u(x) = sum_j k(x, p_j) w_j."""
    xa = np.atleast_2d(np.asarray(x, dtype=np.float64))
    out = np.empty((len(xa), 3), dtype=np.float64)
    for st in range(0, len(xa), chunk):
        xb = xa[st : st + chunk]
        out[st : st + chunk] = matern52_kernel(xb, anchors, ell, gamma_tr, gamma_lin) @ weights
    return out


def rkhs_jacobian_u(
    x: np.ndarray,
    anchors: np.ndarray,
    weights: np.ndarray,
    ell: float = DEFAULT_ELL,
    gamma_lin: float = DEFAULT_GAMMA_LIN,
    chunk: int = 32768,
) -> np.ndarray:
    """Exact C^2 analytical Jacobian J[n, k, d] = du_d / dx_k (smooth even at x == p_j)."""
    xa = np.atleast_2d(np.asarray(x, dtype=np.float64))
    a = math.sqrt(5.0) / ell
    jac = np.empty((len(xa), 3, 3), dtype=np.float64)
    for st in range(0, len(xa), chunk):
        xb = xa[st : st + chunk]
        diff = xb[:, None, :] - anchors[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        fac = -(a**2 / 3.0) * (1.0 + a * r) * np.exp(-a * r)
        dk_dx = fac[..., None] * diff + gamma_lin * anchors[None, :, :]
        jac[st : st + chunk] = np.einsum("njk,jd->nkd", dk_dx, weights)
    return jac


def folding_audit(
    weights: np.ndarray,
    anchors: np.ndarray,
    raw_verts: np.ndarray,
    ell: float = DEFAULT_ELL,
    gamma_tr: float = DEFAULT_GAMMA_TR,
    gamma_lin: float = DEFAULT_GAMMA_LIN,
) -> dict[str, Any]:
    """Evaluate exact Jacobian determinant det(I + J_u) on surface vertices & anchors."""
    v = np.asarray(raw_verts, dtype=np.float64)
    vs = v[np.linspace(0, len(v) - 1, 10000).astype(int)] if len(v) > 10000 else v
    jv = rkhs_jacobian_u(vs, anchors, weights, ell=ell, gamma_lin=gamma_lin)
    ja = rkhs_jacobian_u(anchors, anchors, weights, ell=ell, gamma_lin=gamma_lin)
    det_s = np.linalg.det(np.eye(3)[None] + jv)
    det_a = np.linalg.det(np.eye(3)[None] + ja)
    disp = np.linalg.norm(
        np.concatenate(
            [
                eval_rkhs_u(vs, anchors, weights, ell, gamma_tr, gamma_lin),
                eval_rkhs_u(anchors, anchors, weights, ell, gamma_tr, gamma_lin),
            ],
            axis=0,
        ),
        axis=1,
    )
    a_lin = np.eye(3) + gamma_lin * (anchors.T @ weights)
    sv = np.linalg.svd(a_lin, compute_uv=False)
    return {
        "surface_min_detJ": float(det_s.min()),
        "surface_p01_detJ": float(np.percentile(det_s, 1)),
        "surface_fold_fraction": float(np.mean(det_s <= 0.0)),
        "anchor_min_detJ": float(det_a.min()),
        "linear_sv_min": float(sv.min()),
        "linear_sv_max": float(sv.max()),
        "warp_displacement_max": float(disp.max()),
        "warp_displacement_rms": float(np.sqrt(np.mean(disp**2))),
        "n_surface_samples": int(len(vs)),
    }


# ---------------------------------------------------------------------------
# Normal-consistent surface matching & RKHS Warp Solvers (Normal-Only & P2P)
# ---------------------------------------------------------------------------
def normal_consistent_closest_full(
    query_pts: np.ndarray,
    anc_norms: np.ndarray,
    surf_pts: np.ndarray,
    surf_norms: np.ndarray,
    cos_thresh: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-surface normal-consistent closest vertex (avoids wrong-side thin-shell snapping)."""
    m = len(query_pts)
    qm = np.empty_like(query_pts)
    nm = np.empty_like(query_pts)
    for i in range(m):
        dots = surf_norms @ anc_norms[i]
        mask = dots >= cos_thresh
        if not np.any(mask):
            mask = dots >= 0.0
        if not np.any(mask):
            mask = np.ones(len(surf_pts), dtype=bool)
        cp = surf_pts[mask]
        cn = surf_norms[mask]
        j = int(np.argmin(np.sum((cp - query_pts[i]) ** 2, axis=1)))
        qm[i] = cp[j]
        nm[i] = cn[j]
    return qm, nm


def solve_rkhs_normal(
    k_mat: np.ndarray,
    g_dir: np.ndarray,
    rhs_u: np.ndarray,
    reg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact M x M SPD Representer Theorem solve: (K * (G G^T) + reg I) alpha = rhs."""
    m = len(k_mat)
    a_sys = k_mat * (g_dir @ g_dir.T)
    if reg > 0.0:
        a_sys = a_sys + reg * np.eye(m)
    try:
        alpha = np.linalg.solve(a_sys, rhs_u)
    except np.linalg.LinAlgError:
        alpha, *_ = np.linalg.lstsq(a_sys, rhs_u, rcond=1e-13)
    weights = g_dir * alpha[:, None]
    return weights, k_mat @ weights


def solve_normal_matern_warp(
    eval_f0_grad_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    anchors: np.ndarray,
    normals: np.ndarray,
    raw_v: np.ndarray,
    raw_vn: np.ndarray,
    ell: float = DEFAULT_ELL,
    gamma_tr: float = DEFAULT_GAMMA_TR,
    gamma_lin: float = DEFAULT_GAMMA_LIN,
    reg_s1: float = DEFAULT_REG_STAGE1,
) -> tuple[np.ndarray, float, dict[str, int]]:
    """32-equation Tangential-Free Normal-Only Matern-5/2 RKHS CNF Coordinate Warp.

    Stage 1a: 6 normal-consistent tangent-plane RKHS iterations (minimum RKHS norm).
    Stage 1b: Pull off-edge / saturated / wrong-side anchors into the smooth linear zone via chord.
    Stage 2 : Unregularized incremental minimum-RKHS-norm Newton polish directly on FP64 f0 = 0.
    """
    m = len(anchors)
    k_mat = matern52_kernel(anchors, anchors, ell, gamma_tr, gamma_lin)
    weights = np.zeros((m, 3), dtype=np.float64)
    u_anc = np.zeros((m, 3), dtype=np.float64)

    for it in range(6):
        qm, nm = normal_consistent_closest_full(anchors + u_anc, normals, raw_v, raw_vn, 0.35)
        wn, _ = solve_rkhs_normal(k_mat, nm, np.einsum("ij,ij->i", nm, qm - anchors), reg=reg_s1)
        damp = 0.6 if it < 2 else 1.0
        weights = (1.0 - damp) * weights + damp * wn
        u_anc = k_mat @ weights

    s1b_iters = 0
    for it in range(12):
        s1b_iters = it + 1
        xc = anchors + u_anc
        fv, gv = eval_f0_grad_fn(xc)
        gn = np.linalg.norm(gv, axis=1)
        g_dir = gv / np.maximum(gn[:, None], 1e-12)
        off = (np.abs(fv) > 0.45) | (gn < 0.2) | (np.einsum("ij,ij->i", g_dir, normals) < 0.0)
        if not np.any(off):
            break
        qm, nm = normal_consistent_closest_full(xc, normals, raw_v, raw_vn, 0.35)
        rhs_u = np.einsum("ij,ij->i", g_dir, u_anc) - fv / np.maximum(gn, 1e-6)
        chord = qm - xc
        cn = np.linalg.norm(chord, axis=1)
        for i in np.flatnonzero(off):
            di = chord[i] / cn[i] if cn[i] > 1e-5 else nm[i]
            g_dir[i] = di
            rhs_u[i] = float(np.dot(di, qm[i] - anchors[i]))
        wn, _ = solve_rkhs_normal(k_mat, g_dir, rhs_u, reg=reg_s1)
        weights = 0.3 * weights + 0.7 * wn
        u_anc = k_mat @ weights

    fv, gv = eval_f0_grad_fn(anchors + u_anc)
    s2_iters = 0
    for it in range(25):
        mf = float(np.max(np.abs(fv)))
        nf = float(np.linalg.norm(fv))
        if mf < 1e-14:
            break
        s2_iters = it + 1
        gn = np.linalg.norm(gv, axis=1)
        g_dir = gv / np.maximum(gn[:, None], 1e-12)
        rhs_du = -fv / np.maximum(gn, 1e-12)
        sat = (np.abs(fv) > 0.85) | (gn < 1e-2)
        if np.any(sat):
            xc = anchors + u_anc
            qm, nm = normal_consistent_closest_full(xc[sat], normals[sat], raw_v, raw_vn, 0.35)
            ch = qm - xc[sat]
            cn = np.linalg.norm(ch, axis=1, keepdims=True)
            ds = np.where(cn > 1e-5, ch / np.maximum(cn, 1e-12), nm)
            g_dir[sat] = ds
            rhs_du[sat] = np.einsum("ij,ij->i", ds, qm - xc[sat])
        dw, _ = solve_rkhs_normal(k_mat, g_dir, rhs_du, reg=0.0)
        best = None
        for st in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            wt = weights + st * dw
            ut = k_mat @ wt
            ft, gt = eval_f0_grad_fn(anchors + ut)
            nft = float(np.linalg.norm(ft))
            if best is None or nft < best[0]:
                best = (nft, wt, ut, ft, gt)
            if nft < nf * (1.0 - 1e-4 * st):
                break
        if best is None or (best[0] >= nf and mf < 1e-12):
            break
        weights, u_anc, fv, gv = best[1:]

    ff, _ = eval_f0_grad_fn(anchors + u_anc)
    return weights, float(np.max(np.abs(ff))), {"s1b_iters": s1b_iters, "s2_iters": s2_iters}


def solve_p2p_matern_warp(
    eval_f0_grad_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    anchors: np.ndarray,
    normals: np.ndarray,
    raw_v: np.ndarray,
    raw_vn: np.ndarray,
    ell: float = DEFAULT_ELL,
    gamma_tr: float = DEFAULT_GAMMA_TR,
    gamma_lin: float = DEFAULT_GAMMA_LIN,
) -> np.ndarray:
    """Ablation: 96-equation Point-to-Point Matern-5/2 RKHS Warp (u(p_i) = q_i - p_i)."""
    qm, _ = normal_consistent_closest_full(anchors, normals, raw_v, raw_vn, cos_thresh=0.35)
    q = qm.copy()
    for _ in range(15):
        fv, gv = eval_f0_grad_fn(q)
        if float(np.max(np.abs(fv))) < 1e-14:
            break
        gn2 = np.sum(gv**2, axis=1)
        active = (np.abs(fv) >= 1e-14) & (gn2 > 1e-24)
        if not np.any(active):
            break
        step_vec = np.zeros_like(q)
        step_vec[active] = -(fv[active] / gn2[active])[:, None] * gv[active]
        cur_abs = np.abs(fv)
        alpha = np.ones(len(q), dtype=np.float64)
        for _ls in range(10):
            q_try = q + alpha[:, None] * step_vec
            f_try, _ = eval_f0_grad_fn(q_try)
            improved = (np.abs(f_try) < cur_abs) | (~active)
            if np.all(improved):
                q = q_try
                break
            alpha[~improved] *= 0.5
        else:
            q_try = q + alpha[:, None] * step_vec
            f_try, _ = eval_f0_grad_fn(q_try)
            better = np.abs(f_try) < cur_abs
            q[better] = q_try[better]
    k_mat = matern52_kernel(anchors, anchors, ell, gamma_tr, gamma_lin)
    return np.linalg.solve(k_mat, q - anchors)


def eval_anchor_residuals(
    f_val: np.ndarray,
    grad_cnf: np.ndarray,
    normals: np.ndarray,
) -> dict[str, Any]:
    gn = np.linalg.norm(grad_cnf, axis=1)
    nn = np.linalg.norm(normals, axis=1)
    cos = np.einsum("ij,ij->i", grad_cnf, normals) / np.maximum(gn * nn, 1e-30)
    return {
        "anchor_value_abs_max": float(np.max(np.abs(f_val))),
        "anchor_value_abs_mean": float(np.mean(np.abs(f_val))),
        "anchor_value_rms": float(np.sqrt(np.mean(f_val**2))),
        "anchor_normal_cosine_mean": float(np.mean(cos)),
        "anchor_normal_cosine_min": float(np.min(cos)),
        "anchor_neg_normal_count": int((cos < 0.0).sum()),
    }


def eval_warp_anchor_residuals(
    eval_f0_grad_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    anchors: np.ndarray,
    normals: np.ndarray,
    weights: np.ndarray,
    ell: float = DEFAULT_ELL,
    gamma_tr: float = DEFAULT_GAMMA_TR,
    gamma_lin: float = DEFAULT_GAMMA_LIN,
) -> dict[str, Any]:
    """Exact FP64 continuous value f0(p+u(p)) and exact chain-rule gradient (I+J_u) grad f0."""
    u_a = eval_rkhs_u(anchors, anchors, weights, ell, gamma_tr, gamma_lin)
    fv, g_f0 = eval_f0_grad_fn(anchors + u_a)
    ja = rkhs_jacobian_u(anchors, anchors, weights, ell=ell, gamma_lin=gamma_lin)
    grad_cnf = np.einsum("nkd,nd->nk", np.eye(3)[None] + ja, g_f0)
    return eval_anchor_residuals(fv, grad_cnf, normals)


# ---------------------------------------------------------------------------
# Local Hermite Gaussian Patch CNF baseline (128 equations)
# ---------------------------------------------------------------------------
def gaussian_system(anchors: np.ndarray, sigma: float = GAUSSIAN_SIGMA) -> np.ndarray:
    c = np.asarray(anchors, dtype=np.float64)
    r = c[:, None, :] - c[None, :, :]
    inv = 1.0 / (sigma**2)
    phi = np.exp(-np.sum(r**2, axis=-1) * inv / 2.0)
    n = len(c)
    mat = np.empty((4 * n, 4 * n), dtype=np.float64)
    mat[0::4, 0::4] = phi
    for k in range(3):
        mat[0::4, k + 1 :: 4] = r[..., k] * inv * phi
    for j in range(3):
        mat[j + 1 :: 4, 0::4] = -r[..., j] * inv * phi
        for k in range(3):
            delta = 1.0 if j == k else 0.0
            mat[j + 1 :: 4, k + 1 :: 4] = (delta * inv - r[..., j] * r[..., k] * (inv**2)) * phi
    return mat


def gaussian_solve(
    base_ops: np.ndarray,
    normals: np.ndarray,
    anchors: np.ndarray,
    sigma: float = GAUSSIAN_SIGMA,
) -> tuple[np.ndarray, dict[str, Any]]:
    mat = gaussian_system(anchors, sigma)
    target = np.zeros_like(base_ops, dtype=np.float64)
    target[:, 1:] = np.asarray(normals, dtype=np.float64)
    rhs = (target - np.asarray(base_ops, dtype=np.float64)).reshape(-1)
    psi = np.linalg.solve(mat, rhs)
    sv = np.linalg.svd(mat, compute_uv=False)
    resid = mat @ psi - rhs
    diag = {
        "rank": int(np.linalg.matrix_rank(mat)),
        "condition": float(sv[0] / max(sv[-1], 1e-30)),
        "sigma_min": float(sv[-1]),
        "sigma_max": float(sv[0]),
        "constraint_residual_inf": float(np.max(np.abs(resid))),
        "constraint_residual_rms": float(np.sqrt(np.mean(resid**2))),
    }
    return psi, diag


def gaussian_correction(
    psi: np.ndarray,
    anchors: np.ndarray,
    points: np.ndarray,
    sigma: float = GAUSSIAN_SIGMA,
    with_grad: bool = False,
    chunk: int = 65536,
) -> tuple[np.ndarray, np.ndarray | None]:
    c = np.asarray(anchors, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    w_v = psi[0::4]
    w_g = np.stack([psi[1::4], psi[2::4], psi[3::4]], axis=1)
    inv = 1.0 / (sigma**2)
    val = np.empty(len(p), dtype=np.float64)
    grad = np.empty((len(p), 3), dtype=np.float64) if with_grad else None
    for st in range(0, len(p), chunk):
        pb = p[st : st + chunk]
        r = pb[:, None, :] - c[None, :, :]
        phi = np.exp(-np.sum(r**2, axis=-1) * inv / 2.0)
        dot = np.einsum("nmk,mk->nm", r, w_g)
        val[st : st + chunk] = phi @ w_v + np.sum(phi * dot, axis=1) * inv
        if grad is not None:
            coeff = (w_v[None, :] + dot * inv) * inv * phi
            grad[st : st + chunk] = -np.einsum("nm,nmk->nk", coeff, r) + (phi * inv) @ w_g
    return val, grad


# ---------------------------------------------------------------------------
# Strictly Visible-Only Symmetric Upright Alignment (+Y up, zero hidden leakage)
# ---------------------------------------------------------------------------
def align_visible_upright_sym(
    pred_pts: np.ndarray,
    pred_norms: np.ndarray,
    vis_pts: np.ndarray,
    anchors: np.ndarray,
    anc_norms: np.ndarray,
    trim_fraction: float = 0.65,
    rng_seed: int = 111,
) -> dict[str, Any]:
    """Symmetric two-way visible-only upright (+Y up) alignment. Zero hidden leakage."""
    rng = np.random.default_rng(rng_seed)
    p = np.asarray(pred_pts, dtype=np.float64)
    pn = np.asarray(pred_norms, dtype=np.float64)
    q = np.asarray(vis_pts, dtype=np.float64)
    if len(p) > 8000:
        idx = rng.choice(len(p), 8000, replace=False)
        p, pn = p[idx], pn[idx]
    if len(q) > 12000:
        q = q[rng.choice(len(q), 12000, replace=False)]
    q_sub = q[::3] if len(q) >= 120 else q
    tree_q = cKDTree(q)
    h_p = float(np.percentile(p[:, 1], 99) - np.percentile(p[:, 1], 1))
    h_q = float(np.percentile(q[:, 1], 99) - np.percentile(q[:, 1], 1))
    s_init = h_q / max(h_p, 1e-6)
    keep_n = max(min(32, len(p)), int(len(p) * trim_fraction))
    keep_r_n = max(min(32, len(q_sub)), int(0.95 * len(q_sub)))

    def fit_yaw_st_sym(s: float, r: np.ndarray, t: np.ndarray):
        moved = s * (p @ r.T) + t
        df, idx_f = tree_q.query(moved, k=1)
        keep_f = np.argpartition(df, keep_n - 1)[:keep_n]
        dr, idx_r = cKDTree(moved).query(q_sub, k=1)
        keep_r = np.argpartition(dr, keep_r_n - 1)[:keep_r_n]
        pa = np.concatenate([p[keep_f], p[idx_r[keep_r]]], axis=0)
        qa = np.concatenate([q[idx_f[keep_f]], q_sub[keep_r]], axis=0)
        w = np.concatenate(
            [
                np.full(len(keep_f), 1.0 / len(keep_f)),
                np.full(len(keep_r), 1.0 / len(keep_r)),
            ]
        )
        sw = w.sum()
        cp = (pa * w[:, None]).sum(0) / sw
        cq = (qa * w[:, None]).sum(0) / sw
        pc = pa - cp
        qc = qa - cq
        c_t = float(np.sum(w * (pc[:, 0] * qc[:, 0] + pc[:, 2] * qc[:, 2])))
        s_t = float(np.sum(w * (pc[:, 2] * qc[:, 0] - pc[:, 0] * qc[:, 2])))
        ang = math.atan2(s_t, c_t)
        ca, sa = math.cos(ang), math.sin(ang)
        r_new = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=np.float64)
        pr = pc @ r_new.T
        denom = max(float(np.sum(w * np.sum(pc**2, axis=1))), 1e-30)
        s_new = float(np.sum(w * np.sum(pr * qc, axis=1)) / denom)
        s_new = float(np.clip(s_new, 0.78 * s_init, 1.35 * s_init))
        t_new = cq - s_new * (r_new @ cp)
        return s_new, r_new, t_new

    def score_sym_normal(s: float, r: np.ndarray, t: np.ndarray) -> float:
        moved = s * (p @ r.T) + t
        moved_n = pn @ r.T
        d_fwd, _ = tree_q.query(moved, k=1)
        tree_m = cKDTree(moved)
        d_rev, _ = tree_m.query(q_sub, k=1)
        d_anc, idx_anc = tree_m.query(anchors, k=1)
        cos_anc = np.einsum("ij,ij->i", moved_n[idx_anc], anc_norms)
        return float(
            np.sqrt(np.mean(np.sort(d_fwd)[:keep_n] ** 2))
            + np.sqrt(np.mean(np.sort(d_rev)[:keep_r_n] ** 2))
            + 0.5 * np.sqrt(np.mean(d_anc**2))
            + 0.08 * np.mean(np.maximum(0.0, 0.5 - cos_anc))
        )

    best = None
    for deg in range(0, 360, 15):
        a = math.radians(deg)
        r = np.array(
            [[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]],
            dtype=np.float64,
        )
        s = s_init
        t = np.median(q, axis=0) - s * np.median(p @ r.T, axis=0)
        for _ in range(10):
            s, r, t = fit_yaw_st_sym(s, r, t)
        sc = score_sym_normal(s, r, t)
        if best is None or sc < best[0]:
            best = (sc, f"yaw_{deg:03d}", s, r, t)
    assert best is not None
    _, key, s, r, t = best
    for _ in range(15):
        s, r, t = fit_yaw_st_sym(s, r, t)
    sc = score_sym_normal(s, r, t)
    moved = s * (p @ r.T) + t
    df, _ = tree_q.query(moved, k=1)
    trimmed_rmse = float(np.sqrt(np.mean(np.sort(df)[:keep_n] ** 2)))
    assert abs(r[1, 1] - 1.0) < 1e-12
    return {
        "rotation_key": key,
        "scale": float(s),
        "rotation": r,
        "translation": t,
        "score": float(sc),
        "trimmed_rmse": trimmed_rmse,
        "s_init": float(s_init),
        "yaw_deg": float(math.degrees(math.atan2(r[0, 2], r[0, 0]))),
        "R11": float(r[1, 1]),
        "zero_hidden_geometry_leakage": True,
    }


# ---------------------------------------------------------------------------
# Aggregate statistics & paired bootstrap
# ---------------------------------------------------------------------------
def stats_summary(vals: list[float | None]) -> dict[str, Any]:
    arr = np.asarray(
        [v for v in vals if v is not None and math.isfinite(float(v))], dtype=np.float64
    )
    if not len(arr):
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def paired_bootstrap(
    per_shape: list[dict[str, Any]],
    get_a: Callable[[dict[str, Any]], float | None],
    get_b: Callable[[dict[str, Any]], float | None],
    name_a: str,
    name_b: str,
    metric_name: str,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    pairs = []
    for r in per_shape:
        a = get_a(r)
        b = get_b(r)
        if a is not None and b is not None and math.isfinite(a) and math.isfinite(b):
            pairs.append((r["object_id"], float(a), float(b)))
    diffs = np.asarray([a - b for _, a, b in pairs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    mean_a = float(np.mean([a for _, a, _ in pairs]))
    mean_b = float(np.mean([b for _, _, b in pairs]))
    return {
        "metric": metric_name,
        "arm_a": name_a,
        "arm_b": name_b,
        "n": len(pairs),
        "mean_arm_a": mean_a,
        "mean_arm_b": mean_b,
        "mean_diff": float(diffs.mean()),
        "median_diff": float(np.median(diffs)),
        "relative_change_percent": float((mean_a / mean_b - 1.0) * 100.0) if mean_b else None,
        "bootstrap_ci95_mean_diff": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
        "wins_a_lower": int((diffs < 0).sum()),
        "wins_b_lower": int((diffs > 0).sum()),
        "ties": int((diffs == 0).sum()),
        "bootstrap_seed": seed,
        "bootstrap_resamples": n_boot,
    }
