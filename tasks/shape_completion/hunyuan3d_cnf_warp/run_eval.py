#!/usr/bin/env python3
"""CLI evaluation runner for 3D Hunyuan3D-2mini + Matern-5/2 Normal-Only CNF Warp.

Evaluates four matched arms sharing identical Hunyuan3D-2mini tokens, identical
visible-only symmetric upright alignment, identical 128^3 grid, and identical
canonical-box OOB masking:
  1. hy3d_raw                : Aligned Hunyuan3D-2mini zero-shot implicit field
  2. hy3d_cnf_gaussian_patch : Local Hermite Gaussian patch CNF (128 eqs)
  3. hy3d_cnf_warp_p2p       : Ablation — 96-eq Point-to-Point Matern-5/2 RKHS warp
  4. hy3d_cnf_normal_warp    : Proposed — 32-eq Tangential-Free Normal-Only Matern-5/2 warp
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
import torch
from hy3d_field import HunyuanFieldEvaluator
from warp_core import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CANONICAL_BOX_BOUND,
    CANONICAL_GRID_BOUND,
    DEFAULT_ELL,
    DEFAULT_GAMMA_LIN,
    DEFAULT_GAMMA_TR,
    DEFAULT_REG_STAGE1,
    DEFAULT_REG_STAGE1_RETRY,
    GAUSSIAN_SIGMA,
    GRID_RES,
    N_SURFACE,
    OUTSIDE_VALUE,
    align_visible_upright_sym,
    atomic_json,
    camera_from_npz,
    canonical_oob_mask,
    chamfer_symmetric,
    classify_visible,
    compute_vertex_normals,
    eval_anchor_residuals,
    eval_warp_anchor_residuals,
    extract_surface,
    folding_audit,
    gaussian_correction,
    gaussian_solve,
    grid_points,
    load_json,
    mesh_roughness_fast,
    paired_bootstrap,
    sample_surface,
    sha256_file,
    solve_normal_matern_warp,
    solve_p2p_matern_warp,
    stats_summary,
    transform_to_canonical,
)

ALL_ARMS = [
    "hy3d_raw",
    "hy3d_cnf_gaussian_patch",
    "hy3d_cnf_warp_p2p",
    "hy3d_cnf_normal_warp",
]


def extract_visible_from_depth(src: Any) -> np.ndarray:
    """Reconstruct visible first-hit 3D surface points from source depth map."""
    cam = camera_from_npz(src)
    depth = np.asarray(src["view_depth"], dtype=np.float64)
    valid = np.asarray(src["view_hit"], dtype=bool)
    h, w = depth.shape
    u = np.linspace(-float(cam["half_extent"]), float(cam["half_extent"]), w, dtype=np.float64)
    v = np.linspace(-float(cam["half_extent"]), float(cam["half_extent"]), h, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    pos = np.asarray(cam["position"], dtype=np.float64)
    right = np.asarray(cam["right"], dtype=np.float64)
    up = np.asarray(cam["up"], dtype=np.float64)
    fwd = np.asarray(cam["forward"], dtype=np.float64)
    pts = (
        pos[None, None, :]
        + uu[..., None] * right
        + vv[..., None] * up
        + depth[..., None] * fwd
    )
    return pts[valid]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate Hunyuan3D-2mini + Matern-5/2 Normal-Only CNF Warp."
    )
    ap.add_argument("--manifest", type=Path, required=True, help="JSON manifest (e.g. test.json)")
    ap.add_argument("--data-root", type=Path, required=True, help="Root dir for manifest NPZ paths")
    ap.add_argument("--hy3d-repo", type=Path, required=True, help="Hunyuan3D-2 checkout directory")
    ap.add_argument("--weights-dir", type=Path, required=True, help="Hunyuan3D-2mini weights dir")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    ap.add_argument(
        "--token-cache-dir",
        type=Path,
        default=None,
        help="Optional dir with cached {oid}.npz tokens (otherwise generated from --rgba-dir)",
    )
    ap.add_argument(
        "--visible-cache-dir",
        type=Path,
        default=None,
        help="Optional dir with {oid}.visible.npz (otherwise reconstructed from view_depth)",
    )
    ap.add_argument(
        "--rgba-dir",
        type=Path,
        default=None,
        help="Dir with upright {oid}.rgba.png inputs (required when --token-cache-dir is unset)",
    )
    ap.add_argument("--steps", type=int, default=30, help="DiT flow-matching steps")
    ap.add_argument("--guidance", type=float, default=5.0, help="CFG scale")
    ap.add_argument("--seed-base", type=int, default=111, help="Base seed (seed = seed_base + idx)")
    ap.add_argument("--ell", type=float, default=DEFAULT_ELL, help="Matern-5/2 lengthscale")
    ap.add_argument("--gamma-tr", type=float, default=DEFAULT_GAMMA_TR, help="Translation weight")
    ap.add_argument("--gamma-lin", type=float, default=DEFAULT_GAMMA_LIN, help="Linear weight")
    ap.add_argument("--reg-s1", type=float, default=DEFAULT_REG_STAGE1, help="Stage-1 reg")
    ap.add_argument("--reg-retry", type=float, default=DEFAULT_REG_STAGE1_RETRY, help="Retry reg")
    ap.add_argument("--limit", type=int, default=0, help="Optional shape limit (0 = all)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Hunyuan3D-2mini evaluation.")
    if args.token_cache_dir is None and args.rgba_dir is None:
        raise ValueError("Specify either --token-cache-dir or --rgba-dir.")

    out_dir = args.out_dir.resolve()
    (out_dir / "inference").mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs" / "run.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    entries = load_json(args.manifest)
    if args.limit > 0:
        entries = entries[: args.limit]
    t_start = time.perf_counter()

    log("Loading Hunyuan3D-2mini ShapeVAE + DiT pipeline...")
    feval = HunyuanFieldEvaluator(
        hy3d_repo=args.hy3d_repo,
        weights_dir=args.weights_dir,
        steps=args.steps,
        guidance=args.guidance,
    )
    xyz_full, _ = grid_points(GRID_RES, -1.0, 1.0, dtype=np.float32)

    per_shape: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    atomic_json(
        out_dir / "status.json",
        {"status": "running", "completed": 0, "requested": len(entries), "failures": []},
    )

    for idx, entry in enumerate(entries):
        oid = entry["object_id"]
        seed = args.seed_base + idx
        rec_path = out_dir / "inference" / f"{oid}.json"
        npz_path = out_dir / "inference" / f"{oid}.npz"
        if rec_path.exists() and npz_path.exists():
            cached = load_json(rec_path)
            if cached.get("status") == "complete":
                per_shape.append(cached)
                log(
                    f"[{idx + 1}/{len(entries)}] {oid[:10]} CACHED "
                    f"raw={cached['full_chamfer']['hy3d_raw']:.5f} "
                    f"nw={cached['full_chamfer']['hy3d_cnf_normal_warp']:.5f}"
                )
                continue

        t0 = time.perf_counter()
        timing: dict[str, float] = {}
        try:
            src_path = args.data_root / entry["path"]
            with np.load(src_path, allow_pickle=False) as src:
                target_sdf = np.asarray(src["target_sdf"], dtype=np.float32)
                cam = camera_from_npz(src)
                view_depth = np.asarray(src["view_depth"], dtype=np.float64)
                view_hit = np.asarray(src["view_hit"], dtype=bool)
                anchors = np.asarray(src["anchors"], dtype=np.float64)
                normals = np.asarray(src["normals"], dtype=np.float64)
                vis_pts = (
                    np.asarray(
                        np.load(args.visible_cache_dir / f"{oid}.visible.npz", allow_pickle=False)[
                            "points"
                        ],
                        dtype=np.float64,
                    )
                    if args.visible_cache_dir is not None
                    else extract_visible_from_depth(src)
                )

            provenance: dict[str, Any] = {
                "source_npz_sha256": sha256_file(src_path),
                **feval.weight_hashes,
            }
            if args.token_cache_dir is not None:
                tok_path = args.token_cache_dir / f"{oid}.npz"
                with np.load(tok_path, allow_pickle=False) as zt:
                    tokens_np = np.asarray(zt["tokens"], dtype=np.float32)
                tokens32 = torch.from_numpy(tokens_np).float().unsqueeze(0).cuda()
                provenance["token_cache_npz_sha256"] = sha256_file(tok_path)
                timing["generation_s"] = 0.0
            else:
                rgba_path = args.rgba_dir / f"{oid}.rgba.png"
                tokens32, gen_t = feval.generate_tokens(rgba_path, seed)
                provenance["input_rgba_sha256"] = sha256_file(rgba_path)
                timing["generation_s"] = gen_t["generation_total_s"]
            tokens64 = tokens32.double()

            # 1. Canonical surface & visible-only symmetric upright alignment
            tc0 = time.perf_counter()
            can_sdf, _ = feval.canonical_sdf_grid(tokens32)
            can_m = extract_surface(can_sdf, -CANONICAL_GRID_BOUND, CANONICAL_GRID_BOUND, GRID_RES)
            if can_m is None:
                raise RuntimeError("canonical surface extraction failed")
            can_v, can_f = can_m
            can_vn = compute_vertex_normals(can_v, can_f)
            sub = np.linspace(0, len(can_v) - 1, min(64, len(can_v))).astype(int)
            id_aln = {"scale": 1.0, "rotation": np.eye(3), "translation": np.zeros(3)}
            _, g_chk = feval.eval_f0_grad_fp64(can_v[sub], tokens64, id_aln)
            if np.median(np.einsum("ij,ij->i", can_vn[sub], g_chk)) < 0:
                can_vn = -can_vn
            timing["canonical_s"] = time.perf_counter() - tc0

            ta0 = time.perf_counter()
            aln = align_visible_upright_sym(can_v, can_vn, vis_pts, anchors, normals)
            timing["alignment_s"] = time.perf_counter() - ta0

            def eval_f0_fn(pts_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                return feval.eval_f0_grad_fp64(pts_np, tokens64, aln)

            # 2. Raw aligned grid & mesh
            tr0 = time.perf_counter()
            vol_raw_1d = feval.eval_grid_fp32(xyz_full, tokens32, aln)
            raw_m = extract_surface(
                vol_raw_1d.reshape(GRID_RES, GRID_RES, GRID_RES), -1.0, 1.0, GRID_RES
            )
            if raw_m is None:
                raise RuntimeError("raw surface extraction failed")
            raw_v, raw_f = raw_m
            raw_vn = compute_vertex_normals(raw_v, raw_f)
            sub2 = np.linspace(0, len(raw_v) - 1, min(64, len(raw_v))).astype(int)
            _, g_chk2 = eval_f0_fn(raw_v[sub2])
            if np.median(np.einsum("ij,ij->i", raw_vn[sub2], g_chk2)) < 0:
                raw_vn = -raw_vn
            timing["grid_eval_raw_s"] = time.perf_counter() - tr0

            f_raw_a, g_raw_a = eval_f0_fn(anchors)
            res_raw = eval_anchor_residuals(f_raw_a, g_raw_a, normals)

            # 3. Gaussian patch baseline (128 eqs, FP64)
            tg0 = time.perf_counter()
            base_ops = np.concatenate([f_raw_a[:, None], g_raw_a], axis=1)
            psi_g, g_mat_diag = gaussian_solve(base_ops, normals, anchors, GAUSSIAN_SIGMA)
            corr_v, corr_g = gaussian_correction(
                psi_g, anchors, anchors, GAUSSIAN_SIGMA, with_grad=True
            )
            res_gauss = eval_anchor_residuals(f_raw_a + corr_v, g_raw_a + corr_g, normals)
            corr_grid, _ = gaussian_correction(
                psi_g, anchors, xyz_full, GAUSSIAN_SIGMA, with_grad=False
            )
            oob_raw = canonical_oob_mask(
                transform_to_canonical(xyz_full.astype(np.float64), aln), CANONICAL_BOX_BOUND
            )
            vol_gauss_1d = vol_raw_1d.astype(np.float64) + corr_grid
            vol_gauss_1d[oob_raw] = OUTSIDE_VALUE
            gauss_m = extract_surface(
                vol_gauss_1d.reshape(GRID_RES, GRID_RES, GRID_RES), -1.0, 1.0, GRID_RES
            )
            timing["gaussian_total_s"] = time.perf_counter() - tg0

            # 4. Ablation: 96-eq Point-to-Point Matern-5/2 RKHS Warp
            tp0 = time.perf_counter()
            w_p2p = solve_p2p_matern_warp(
                eval_f0_fn, anchors, normals, raw_v, raw_vn, args.ell, args.gamma_tr, args.gamma_lin
            )
            timing["p2p_solve_s"] = time.perf_counter() - tp0
            tp1 = time.perf_counter()
            vol_p2p_1d = feval.eval_grid_fp32(
                xyz_full, tokens32, aln, anchors, w_p2p, args.ell, args.gamma_tr, args.gamma_lin
            )
            p2p_m = extract_surface(
                vol_p2p_1d.reshape(GRID_RES, GRID_RES, GRID_RES), -1.0, 1.0, GRID_RES
            )
            timing["grid_eval_p2p_s"] = time.perf_counter() - tp1
            res_p2p = eval_warp_anchor_residuals(
                eval_f0_fn, anchors, normals, w_p2p, args.ell, args.gamma_tr, args.gamma_lin
            )
            fold_p2p = folding_audit(
                w_p2p, anchors, raw_v, args.ell, args.gamma_tr, args.gamma_lin
            )

            # 5. Proposed: 32-eq Tangential-Free Normal-Only Matern-5/2 RKHS Warp
            tn0 = time.perf_counter()
            w_nw, mf_nw, nw_info = solve_normal_matern_warp(
                eval_f0_fn,
                anchors,
                normals,
                raw_v,
                raw_vn,
                args.ell,
                args.gamma_tr,
                args.gamma_lin,
                reg_s1=args.reg_s1,
            )
            fold_nw = folding_audit(w_nw, anchors, raw_v, args.ell, args.gamma_tr, args.gamma_lin)
            used_retry = False
            if mf_nw > 1e-11 or fold_nw["surface_min_detJ"] <= 0.0:
                wr, mfr, infor = solve_normal_matern_warp(
                    eval_f0_fn,
                    anchors,
                    normals,
                    raw_v,
                    raw_vn,
                    args.ell,
                    args.gamma_tr,
                    args.gamma_lin,
                    reg_s1=args.reg_retry,
                )
                fold_r = folding_audit(wr, anchors, raw_v, args.ell, args.gamma_tr, args.gamma_lin)
                used_retry = True
                if mfr < 1e-11 and (
                    mf_nw >= 1e-11 or fold_r["surface_min_detJ"] > fold_nw["surface_min_detJ"]
                ):
                    w_nw, mf_nw, fold_nw, nw_info = wr, mfr, fold_r, infor
            timing["normal_warp_solve_s"] = time.perf_counter() - tn0
            tn1 = time.perf_counter()
            vol_nw_1d = feval.eval_grid_fp32(
                xyz_full, tokens32, aln, anchors, w_nw, args.ell, args.gamma_tr, args.gamma_lin
            )
            nw_m = extract_surface(
                vol_nw_1d.reshape(GRID_RES, GRID_RES, GRID_RES), -1.0, 1.0, GRID_RES
            )
            timing["grid_eval_normal_warp_s"] = time.perf_counter() - tn1
            res_nw = eval_warp_anchor_residuals(
                eval_f0_fn, anchors, normals, w_nw, args.ell, args.gamma_tr, args.gamma_lin
            )

            # 6. Chamfer & smoothness evaluation
            gt_m = extract_surface(target_sdf, -1.0, 1.0, 64)
            assert gt_m is not None
            gt_pts = sample_surface(gt_m[0], gt_m[1], N_SURFACE, seed + 7919)
            gt_hid = gt_pts[~classify_visible(gt_pts, cam, view_depth, view_hit)]

            meshes = {
                "hy3d_raw": raw_m,
                "hy3d_cnf_gaussian_patch": gauss_m,
                "hy3d_cnf_warp_p2p": p2p_m,
                "hy3d_cnf_normal_warp": nw_m,
            }
            full_cd: dict[str, float | None] = {}
            hidden_cd: dict[str, float | None] = {}
            mesh_counts: dict[str, dict[str, int]] = {}
            roughness: dict[str, Any] = {"GT": mesh_roughness_fast(gt_m[0], gt_m[1])}
            for arm, m in meshes.items():
                if m is None:
                    full_cd[arm] = None
                    hidden_cd[arm] = None
                    mesh_counts[arm] = {"vertices": 0, "faces": 0}
                    roughness[arm] = mesh_roughness_fast(
                        np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
                    )
                else:
                    pts = sample_surface(m[0], m[1], N_SURFACE, seed + 10000)
                    full_cd[arm] = chamfer_symmetric(pts, gt_pts)
                    p_hid = pts[~classify_visible(pts, cam, view_depth, view_hit)]
                    hidden_cd[arm] = (
                        chamfer_symmetric(p_hid, gt_hid) if len(p_hid) and len(gt_hid) else None
                    )
                    mesh_counts[arm] = {"vertices": int(len(m[0])), "faces": int(len(m[1]))}
                    roughness[arm] = mesh_roughness_fast(m[0], m[1])

            save_dict: dict[str, Any] = {
                "anchors": anchors,
                "normals": normals,
                "scale": np.asarray(aln["scale"]),
                "rotation": np.asarray(aln["rotation"]),
                "translation": np.asarray(aln["translation"]),
                "W_p2p": w_p2p,
                "W_nw": w_nw,
                "tokens": tokens32[0].cpu().numpy().astype(np.float16),
            }
            for arm, m in meshes.items():
                if m is not None:
                    save_dict[f"{arm}_verts"] = m[0].astype(np.float32)
                    save_dict[f"{arm}_faces"] = m[1].astype(np.int32)
            np.savez_compressed(npz_path, **save_dict)

            timing["total_s"] = time.perf_counter() - t0
            row = {
                "object_id": oid,
                "index": idx,
                "seed": seed,
                "status": "complete",
                "full_chamfer": full_cd,
                "hidden_chamfer": hidden_cd,
                "anchor_residuals": {
                    "hy3d_raw": res_raw,
                    "hy3d_cnf_gaussian_patch": res_gauss,
                    "hy3d_cnf_warp_p2p": res_p2p,
                    "hy3d_cnf_normal_warp": res_nw,
                },
                "folding": {
                    "hy3d_cnf_warp_p2p": fold_p2p,
                    "hy3d_cnf_normal_warp": fold_nw,
                },
                "normal_warp_diagnostics": {
                    "used_retry": used_retry,
                    "w_diff_vs_p2p_fro": float(np.linalg.norm(w_nw - w_p2p)),
                    **nw_info,
                },
                "roughness": roughness,
                "mesh_stats": mesh_counts,
                "alignment": {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in aln.items()
                },
                "gaussian_matrix": g_mat_diag,
                "timing": timing,
                "provenance": provenance,
            }
            atomic_json(rec_path, row)
            per_shape.append(row)
            log(
                f"[{idx + 1}/{len(entries)}] {oid[:10]} raw={full_cd['hy3d_raw']:.5f} "
                f"p2p={full_cd['hy3d_cnf_warp_p2p']:.5f} nw={full_cd['hy3d_cnf_normal_warp']:.5f} "
                f"| hid {hidden_cd['hy3d_raw']:.5f}->{hidden_cd['hy3d_cnf_normal_warp']:.5f} "
                f"| max|f|={res_nw['anchor_value_abs_max']:.2e} "
                f"minJ={fold_nw['surface_min_detJ']:.3f} t={timing['total_s']:.1f}s"
            )
            atomic_json(
                out_dir / "status.json",
                {
                    "status": "running",
                    "completed": len(per_shape),
                    "requested": len(entries),
                    "failures": failures,
                },
            )
        except Exception as exc:
            failures.append(
                {
                    "object_id": oid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            log(f"[{idx + 1}/{len(entries)}] FAILED {oid}: {exc}")

    aggregate: dict[str, Any] = {
        "GT": {
            k: stats_summary([r["roughness"]["GT"][k] for r in per_shape])
            for k in (
                "dihedral_deg_mean",
                "dihedral_deg_p95",
                "dihedral_deg_p99",
                "laplacian_roughness",
            )
        }
    }
    for arm in ALL_ARMS:
        agg: dict[str, Any] = {
            "full_chamfer": stats_summary([r["full_chamfer"][arm] for r in per_shape]),
            "hidden_chamfer": stats_summary([r["hidden_chamfer"][arm] for r in per_shape]),
            "anchor_value_abs_max": stats_summary(
                [r["anchor_residuals"][arm]["anchor_value_abs_max"] for r in per_shape]
            ),
            "anchor_normal_cosine_mean": stats_summary(
                [r["anchor_residuals"][arm]["anchor_normal_cosine_mean"] for r in per_shape]
            ),
            "dihedral_deg_mean": stats_summary(
                [r["roughness"][arm]["dihedral_deg_mean"] for r in per_shape]
            ),
            "dihedral_deg_p95": stats_summary(
                [r["roughness"][arm]["dihedral_deg_p95"] for r in per_shape]
            ),
            "laplacian_roughness": stats_summary(
                [r["roughness"][arm]["laplacian_roughness"] for r in per_shape]
            ),
        }
        if arm in ("hy3d_cnf_warp_p2p", "hy3d_cnf_normal_warp"):
            for fk in (
                "surface_min_detJ",
                "surface_p01_detJ",
                "surface_fold_fraction",
                "anchor_min_detJ",
                "warp_displacement_max",
            ):
                agg[fk] = stats_summary([r["folding"][arm][fk] for r in per_shape])
        aggregate[arm] = agg

    paired = {
        "nw_vs_raw_full_chamfer": paired_bootstrap(
            per_shape,
            lambda r: r["full_chamfer"]["hy3d_cnf_normal_warp"],
            lambda r: r["full_chamfer"]["hy3d_raw"],
            "hy3d_cnf_normal_warp",
            "hy3d_raw",
            "full_chamfer",
        ),
        "nw_vs_raw_hidden_chamfer": paired_bootstrap(
            per_shape,
            lambda r: r["hidden_chamfer"]["hy3d_cnf_normal_warp"],
            lambda r: r["hidden_chamfer"]["hy3d_raw"],
            "hy3d_cnf_normal_warp",
            "hy3d_raw",
            "hidden_chamfer",
        ),
        "nw_vs_gauss_full_chamfer": paired_bootstrap(
            per_shape,
            lambda r: r["full_chamfer"]["hy3d_cnf_normal_warp"],
            lambda r: r["full_chamfer"]["hy3d_cnf_gaussian_patch"],
            "hy3d_cnf_normal_warp",
            "hy3d_cnf_gaussian_patch",
            "full_chamfer",
        ),
        "nw_vs_gauss_hidden_chamfer": paired_bootstrap(
            per_shape,
            lambda r: r["hidden_chamfer"]["hy3d_cnf_normal_warp"],
            lambda r: r["hidden_chamfer"]["hy3d_cnf_gaussian_patch"],
            "hy3d_cnf_normal_warp",
            "hy3d_cnf_gaussian_patch",
            "hidden_chamfer",
        ),
        "nw_vs_p2p_full_chamfer": paired_bootstrap(
            per_shape,
            lambda r: r["full_chamfer"]["hy3d_cnf_normal_warp"],
            lambda r: r["full_chamfer"]["hy3d_cnf_warp_p2p"],
            "hy3d_cnf_normal_warp",
            "hy3d_cnf_warp_p2p",
            "full_chamfer",
        ),
        "nw_vs_p2p_hidden_chamfer": paired_bootstrap(
            per_shape,
            lambda r: r["hidden_chamfer"]["hy3d_cnf_normal_warp"],
            lambda r: r["hidden_chamfer"]["hy3d_cnf_warp_p2p"],
            "hy3d_cnf_normal_warp",
            "hy3d_cnf_warp_p2p",
            "hidden_chamfer",
        ),
        "nw_vs_p2p_min_detJ": paired_bootstrap(
            per_shape,
            lambda r: r["folding"]["hy3d_cnf_normal_warp"]["surface_min_detJ"],
            lambda r: r["folding"]["hy3d_cnf_warp_p2p"]["surface_min_detJ"],
            "hy3d_cnf_normal_warp",
            "hy3d_cnf_warp_p2p",
            "surface_min_detJ",
        ),
    }

    elapsed = time.perf_counter() - t_start
    results = {
        "status": "complete" if (len(per_shape) == len(entries) and not failures) else "partial",
        "config": {
            "arms": ALL_ARMS,
            "kernel": "Matern-5/2 (Sobolev H^3) + weakly penalized affine",
            "ell": args.ell,
            "gamma_tr": args.gamma_tr,
            "gamma_lin": args.gamma_lin,
            "reg_stage1": args.reg_s1,
            "reg_retry": args.reg_retry,
            "grid_res": GRID_RES,
            "canonical_box_bound": CANONICAL_BOX_BOUND,
            "n_surface": N_SURFACE,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "provenance": {
            "n_evaluated": len(per_shape),
            "n_failures": len(failures),
            "elapsed_s": elapsed,
            **feval.weight_hashes,
        },
        "aggregate": aggregate,
        "paired_comparisons": paired,
        "failures": failures,
        "per_shape": per_shape,
    }
    atomic_json(out_dir / "results.json", results)
    atomic_json(
        out_dir / "status.json",
        {
            "status": results["status"],
            "completed": len(per_shape),
            "requested": len(entries),
            "failures": failures,
            "elapsed_s": elapsed,
        },
    )
    log(f"DONE n={len(per_shape)} failures={len(failures)} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
