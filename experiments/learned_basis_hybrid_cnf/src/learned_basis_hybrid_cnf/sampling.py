from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .projection import hard_project_H
from .util_eval import (
    ANCHOR_FRAMES,
    ANCHOR_TOL,
    DEGREE,
    EULER_STEPS,
    EVAL_DS,
    FRAMES,
    FTOL,
    GROUPS,
    HC_TOL,
    HEIGHT_TOL,
    JOINTS,
    LIMIT_TOL,
    MAXITER,
    NOISE_SCALE,
    NONANCHOR_FRAMES,
    OUT,
    QBC_TOL,
    SAFE_Z,
    SOLVER_GUARD_M,
    WAN_SEEDS,
    analytic_jacobian,
    anchor_target_b,
    append_eval_log,
    build_H,
    checkpoint_path_for_basis,
    decode_q,
    decode_video,
    ensure_eval_dirs,
    file_ok,
    fk_all,
    group_info,
    joint_limits_from_scene,
    load_eval_metadata,
    load_eval_q_orig,
    load_eval_refs_v2,
    motion_seeds_for_scene,
    nullspace_Q,
    read_json,
    render_control_and_sidecar,
    sample_motion_id,
    sample_video_id,
    sha256_array,
    sha256_file,
    unflatten_c,
    update_status_eval,
    utcnow,
    write_json,
)


def load_basis_and_generator(kind: str, device: str | torch.device = "cpu") -> dict[str, Any]:
    from cnf_motion.soft_field import PandaSoftFlowModel
    from .basis import NeuralKernelBasis

    ckpt_path = checkpoint_path_for_basis(kind)
    if not file_ok(ckpt_path, min_bytes=1024):
        raise FileNotFoundError(f"Checkpoint missing or incomplete: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint at {ckpt_path} is not a dict")

    required_keys = [
        "schema_version",
        "group",
        "model",
        "basis",
        "mean",
        "std",
        "degree10",
        "joints7",
        "hidden256",
        "anchor_frames",
        "train_seed0",
        "steps6000",
        "training_config",
    ]
    missing_keys = [k for k in required_keys if k not in ckpt]
    if missing_keys:
        raise KeyError(f"Checkpoint {ckpt_path} missing required keys: {missing_keys}")

    if int(ckpt["schema_version"]) != 1:
        raise ValueError(f"Checkpoint {ckpt_path} schema_version={ckpt['schema_version']} != 1")
    if str(ckpt["group"]) != str(kind):
        raise ValueError(f"Checkpoint {ckpt_path} group={ckpt['group']!r} != expected {kind!r}")
    if int(ckpt["steps6000"]) != 6000 or ("steps" in ckpt and int(ckpt["steps"]) != 6000):
        raise ValueError(f"Checkpoint {ckpt_path} steps mismatch: steps6000={ckpt.get('steps6000')}, steps={ckpt.get('steps')}")
    if int(ckpt["train_seed0"]) != 0:
        raise ValueError(f"Checkpoint {ckpt_path} train_seed0={ckpt['train_seed0']} != 0")

    tcfg = ckpt["training_config"]
    if not isinstance(tcfg, dict) or int(tcfg.get("steps", -1)) != 6000 or str(tcfg.get("group")) != str(kind):
        raise ValueError(f"Checkpoint {ckpt_path} training_config invalid steps/group: {tcfg.get('steps')}, {tcfg.get('group')}")

    ell = float(tcfg["ell"])
    basis = NeuralKernelBasis(ell=ell)
    bsd = ckpt["basis"]
    basis.load_state_dict(bsd, strict=True)
    basis.to(device)
    basis.eval()

    with torch.no_grad():
        B_ref_t, R_t = basis.forward_ref()
        B_at_t = basis.forward_B_at(basis.ref_t, R_t)
        Braw_t = basis.Braw_ref()

    B_ref = B_ref_t.detach().cpu().double().numpy()
    R = R_t.detach().cpu().double().numpy()
    B_at = B_at_t.detach().cpu().double().numpy()
    Braw = Braw_t.detach().cpu().double().numpy()

    degree = int(ckpt["degree10"])
    joints = int(ckpt["joints7"])
    hidden = int(ckpt["hidden256"])
    anchors = list(ckpt["anchor_frames"])
    if degree != DEGREE or joints != JOINTS or hidden != 256 or anchors != list(ANCHOR_FRAMES):
        raise ValueError(f"Checkpoint schema mismatch: degree={degree}, joints={joints}, hidden={hidden}, anchors={anchors}")

    model = PandaSoftFlowModel(joints=joints, degree=degree, anchors=len(anchors), hidden=hidden)
    msd = ckpt["model"]
    model.load_state_dict(msd, strict=True)
    model.to(device)
    model.eval()

    mean_val = ckpt["mean"]
    std_val = ckpt["std"]
    mean = float(mean_val.detach().cpu().item()) if torch.is_tensor(mean_val) else float(mean_val)
    std = float(std_val.detach().cpu().item()) if torch.is_tensor(std_val) else float(std_val)
    if not (np.isfinite(mean) and np.isfinite(std) and std > 0):
        raise ValueError(f"Invalid mean/std in checkpoint {ckpt_path}: mean={mean}, std={std}")

    # Basis numerical diagnostics
    sv_Braw = np.linalg.svd(Braw, compute_uv=False)
    sv_B = np.linalg.svd(B_ref, compute_uv=False)
    Htime = B_ref[ANCHOR_FRAMES, :]
    sv_Htime = np.linalg.svd(Htime, compute_uv=False)
    H_full = build_H(B_ref, ANCHOR_FRAMES)
    sv_H = np.linalg.svd(H_full, compute_uv=False)

    num_fail = bool(
        (not np.all(np.isfinite(B_ref)))
        or (not np.all(np.isfinite(Htime)))
        or float(sv_Braw.min()) <= 1e-6
        or float(sv_Htime.min()) <= 1e-8
        or float(sv_Braw.max() / max(sv_Braw.min(), 1e-30)) > 1e6
        or float(sv_Htime.max() / max(sv_Htime.min(), 1e-30)) > 1e6
    )

    basis_diag = {
        "kind": kind,
        "ckpt_path": str(ckpt_path),
        "ckpt_sha256": sha256_file(ckpt_path),
        "schema_version": int(ckpt["schema_version"]),
        "group_in_ckpt": str(ckpt["group"]),
        "train_seed": int(ckpt["train_seed0"]),
        "steps": int(ckpt["steps6000"]),
        "mean": mean,
        "std": std,
        "orthonormality_max_err": float(np.max(np.abs(B_ref.T @ B_ref - np.eye(DEGREE)))),
        "forward_B_at_ref_max_err": float(np.max(np.abs(B_at - B_ref))),
        "Braw_rank": int(np.sum(sv_Braw > 1e-10)),
        "Braw_min_sv": float(sv_Braw.min()),
        "Braw_max_sv": float(sv_Braw.max()),
        "Braw_cond": float(sv_Braw.max() / max(sv_Braw.min(), 1e-30)),
        "B_min_sv": float(sv_B.min()),
        "B_max_sv": float(sv_B.max()),
        "B_cond": float(sv_B.max() / max(sv_B.min(), 1e-30)),
        "Htime_rank": int(np.sum(sv_Htime > 1e-10)),
        "Htime_min_sv": float(sv_Htime.min()),
        "Htime_max_sv": float(sv_Htime.max()),
        "Htime_cond": float(sv_Htime.max() / max(sv_Htime.min(), 1e-30)),
        "H_rank": int(np.sum(sv_H > 1e-10)),
        "H_min_sv": float(sv_H.min()),
        "H_max_sv": float(sv_H.max()),
        "H_cond": float(sv_H.max() / max(sv_H.min(), 1e-30)),
        "W_fro_norm": float(torch.linalg.norm(basis.W.detach()).cpu().item()),
        "numerical_failure_triggered": num_fail,
    }
    if num_fail:
        fail_file = OUT / "motion" / f"basis_numerical_failure_{kind}.json"
        write_json(fail_file, basis_diag)
        raise RuntimeError(
            f"Basis {kind} triggered numerical failure bound (saved evidence to {fail_file}): "
            f"Braw_min_sv={basis_diag['Braw_min_sv']:.3e}, Braw_cond={basis_diag['Braw_cond']:.3e}, "
            f"Htime_min_sv={basis_diag['Htime_min_sv']:.3e}, Htime_cond={basis_diag['Htime_cond']:.3e}"
        )

    return {
        "basis": basis,
        "model": model,
        "ckpt": ckpt,
        "B_ref": B_ref,
        "R": R,
        "H": H_full,
        "mean": mean,
        "std": std,
        "diagnostics": basis_diag,
    }


def hard_project_c_numpy(H: np.ndarray, c: np.ndarray, b: np.ndarray, min_sv: float = 1e-10) -> tuple[np.ndarray, dict[str, Any]]:
    Ht = torch.from_numpy(np.asarray(H, dtype=np.float64))
    ct = torch.from_numpy(np.asarray(c, dtype=np.float64).reshape(1, -1))
    bt = torch.from_numpy(np.asarray(b, dtype=np.float64).reshape(1, -1))
    c2, info = hard_project_H(Ht, ct, bt, min_sv=min_sv)
    out_info = {
        k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
        for k, v in info.items()
    }
    c_proj = c2.detach().cpu().numpy().reshape(-1).astype(np.float64)
    out_info["proj_delta_l2"] = float(np.linalg.norm(c_proj - np.asarray(c, dtype=np.float64).reshape(-1)))
    out_info["Hc_minus_b_max_abs"] = float(np.max(np.abs(H @ c_proj - b)))
    return c_proj, out_info


def sample_normalized_u(
    model: torch.nn.Module,
    q0: np.ndarray,
    anchors_flat: np.ndarray,
    seed: int,
    *,
    steps: int = EULER_STEPS,
    mode: str = "soft",
    H: np.ndarray | None = None,
    b: np.ndarray | None = None,
    mean: float = 0.0,
    std: float = 1.0,
) -> dict[str, Any]:
    """Euler 50-step sampling in normalized u (float32), with float64 hard projection for hybrid."""
    device = next(model.parameters()).device
    # Always generate initial noise on CPU with torch.Generator('cpu') or on model device consistently;
    # let's use CPU generator then move to device so CPU and CUDA produce identical initial noise u0!
    # Wait: let's check soft_field.py: generator = torch.Generator(device=device).manual_seed(seed).
    # Since both Soft and Hybrid run on the same device in this sampling pass, using device generator is 100% identical between Soft and Hybrid!
    gen = torch.Generator(device=device).manual_seed(int(seed))
    dim = int(getattr(model, "dim", DEGREE * JOINTS))
    u0 = torch.randn(1, dim, device=device, generator=gen, dtype=torch.float32) * float(NOISE_SCALE)
    u = u0.clone()
    u0_np = u0.detach().cpu().numpy().reshape(-1).astype(np.float32)

    q0_t = torch.from_numpy(np.asarray(q0, dtype=np.float32)).unsqueeze(0).to(device)
    a_t = torch.from_numpy(np.asarray(anchors_flat, dtype=np.float32)).unsqueeze(0).to(device)

    step_infos = []
    for s in range(steps):
        t = torch.full((1,), float(s) / float(steps), device=device, dtype=torch.float32)
        with torch.no_grad():
            v = model(u, t, q0_t, a_t)
        u = (u + v / float(steps)).to(torch.float32)
        if mode == "hybrid":
            if H is None or b is None:
                raise ValueError("hybrid mode requires H and b")
            c_step_pre = u.detach().cpu().double().numpy().reshape(-1) * float(std) + float(mean)
            c_step_proj, info = hard_project_c_numpy(H, c_step_pre, b)
            u_next = ((c_step_proj - float(mean)) / float(std)).astype(np.float32)
            u = torch.from_numpy(u_next).unsqueeze(0).to(device=device, dtype=torch.float32)
            step_infos.append({"step": int(s), **info})
        elif mode != "soft":
            raise ValueError(f"Unknown sampling mode: {mode}")

    u_before_final_proj = u.detach().cpu().numpy().reshape(-1).astype(np.float32)
    c_before_final_proj = u_before_final_proj.astype(np.float64) * float(std) + float(mean)

    final_proj_info = None
    if mode == "hybrid":
        assert H is not None and b is not None
        c_raw, final_proj_info = hard_project_c_numpy(H, c_before_final_proj, b)
        u_final = ((c_raw - float(mean)) / float(std)).astype(np.float32)
    else:
        c_raw = c_before_final_proj.copy()
        u_final = u_before_final_proj.copy()

    return {
        "u0": u0_np,
        "u0_sha256": sha256_array(u0_np),
        "u_before_final_proj": u_before_final_proj,
        "c_before_final_proj": c_before_final_proj,
        "u_raw": u_final,
        "c_raw": c_raw,
        "step_infos": step_infos,
        "final_proj_info": final_proj_info,
    }


def _safety_objective_and_constraints(
    robot,
    q_linear: np.ndarray,
    q_map: np.ndarray,
    flatmap: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
):
    cache: dict[str, Any] = {}

    def evaluate(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=np.float64)
        key = z.tobytes()
        if cache.get("key") != key:
            q = q_linear + np.einsum("tjn,n->tj", q_map, z)
            ee = fk_all(robot, q)
            cache["key"] = key
            cache["q"] = q
            cache["ee"] = ee
        return cache["q"], cache["ee"]

    def ineq(z: np.ndarray) -> np.ndarray:
        q, ee = evaluate(z)
        return np.concatenate(
            [
                ee[:, 2] - SAFE_Z - SOLVER_GUARD_M,
                (q - lower).ravel(),
                (upper - q).ravel(),
            ]
        )

    def ineq_jac(z: np.ndarray) -> np.ndarray:
        q, _ = evaluate(z)
        height_jac = np.array(
            [analytic_jacobian(robot, q[k])[2] @ q_map[k] for k in range(FRAMES)],
            dtype=np.float64,
        )
        return np.vstack([height_jac, flatmap, -flatmap])

    return evaluate, ineq, ineq_jac


def safety_repair(
    robot,
    B: np.ndarray,
    c_source: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    mode: str,
    H: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> dict[str, Any]:
    """Shared SLSQP safety repair.

    - Soft: update space is I_70, only safety constraints (NO anchor projection).
    - Hybrid: update space is ker(H) starting from exact hard_project_H(H, c_source, b), preserving anchors.
    """
    from scipy.optimize import minimize

    B = np.asarray(B, dtype=np.float64)
    c_source = np.asarray(c_source, dtype=np.float64).reshape(-1)
    if mode == "soft":
        c_lin = c_source.copy()
        N = np.eye(DEGREE * JOINTS, dtype=np.float64)
        proj_info = {"mode": "soft_no_anchor_projection", "affine_correction_l2": 0.0}
    elif mode == "hybrid":
        if H is None or b is None:
            raise ValueError("hybrid repair requires H and b")
        c_lin, pinfo = hard_project_c_numpy(H, c_source, b)
        N = nullspace_Q(H)
        if N.size == 0:
            N = np.zeros((DEGREE * JOINTS, 0), dtype=np.float64)
        proj_info = {"mode": "hybrid_nullspace_anchor_preserving", **pinfo}
    else:
        raise ValueError(f"Unknown mode: {mode}")

    q_linear = decode_q(B, c_lin)
    if N.shape[1] == 0:
        q_map = np.zeros((FRAMES, JOINTS, 0), dtype=np.float64)
        flatmap = np.zeros((FRAMES * JOINTS, 0), dtype=np.float64)
    else:
        q_map = np.einsum("td,djn->tjn", B, N.reshape(DEGREE, JOINTS, N.shape[1]))
        flatmap = q_map.reshape(FRAMES * JOINTS, N.shape[1])

    evaluate, ineq, ineq_jac = _safety_objective_and_constraints(robot, q_linear, q_map, flatmap, lower, upper)
    z0 = np.zeros(N.shape[1], dtype=np.float64)
    vals0 = ineq(z0)
    jac0 = ineq_jac(z0)
    constant = np.linalg.norm(jac0, axis=1) < 1e-12
    if np.any(vals0[constant] < -1e-10):
        q_fail, ee_fail = evaluate(z0)
        return {
            "ok": False,
            "converged": False,
            "reason": "incompatible_fixed_equality_or_safety",
            "message": "constant constraints infeasible",
            "nit": 0,
            "c_linear": c_lin,
            "c_final": c_lin,
            "q_final": q_fail,
            "ee_final": ee_fail,
            "z": z0,
            "proj_info": proj_info,
            "min_height_margin_m": float((ee_fail[:, 2] - SAFE_Z).min()),
            "min_limit_margin_rad": float(np.minimum(q_fail - lower[None, :], upper[None, :] - q_fail).min()),
            "coeff_change_l2": float(np.linalg.norm(c_lin - c_source)),
            "nullspace_change_l2": 0.0,
        }

    variable = ~constant
    if N.shape[1] == 0 or np.all(vals0[variable] >= 0.0):
        zf = z0.copy()
        nit = 0
        success = True
        msg = "raw_or_affine_already_feasible"
    else:
        res = minimize(
            lambda z: 0.5 * float(np.dot(z, z)),
            z0,
            jac=lambda z: np.asarray(z, dtype=np.float64),
            method="SLSQP",
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda z: ineq(z)[variable],
                    "jac": lambda z: ineq_jac(z)[variable],
                }
            ],
            options={"maxiter": MAXITER, "ftol": FTOL},
        )
        zf = np.asarray(res.x, dtype=np.float64)
        nit = int(getattr(res, "nit", 0) or 0)
        success = bool(res.success)
        msg = str(res.message)

    c_final = c_lin + (N @ zf if N.shape[1] else 0.0)
    q_final, ee_final = evaluate(zf)
    lim_margin = np.minimum(q_final - lower[None, :], upper[None, :] - q_final)
    h_margin = ee_final[:, 2] - SAFE_Z
    tol_ok = bool(
        np.all(np.isfinite(q_final))
        and float(h_margin.min()) >= -HEIGHT_TOL
        and float(lim_margin.min()) >= -LIMIT_TOL
    )
    ok = bool(success and tol_ok)
    return {
        "ok": ok,
        "converged": bool(success),
        "reason": None if ok else ("slsqp_not_converged" if not success else "safety_tolerance_violated"),
        "message": msg,
        "nit": nit,
        "c_linear": c_lin,
        "c_final": c_final,
        "q_final": q_final,
        "ee_final": ee_final,
        "z": zf,
        "proj_info": proj_info,
        "min_height_margin_m": float(h_margin.min()),
        "min_limit_margin_rad": float(lim_margin.min()),
        "strict_height_ge0": bool(float(h_margin.min()) >= 0.0),
        "strict_limits_ge0": bool(float(lim_margin.min()) >= 0.0),
        "coeff_change_l2": float(np.linalg.norm(c_final - c_source)),
        "nullspace_change_l2": float(np.linalg.norm(zf)),
    }


def trajectory_motion_stats(q: np.ndarray, ee: np.ndarray) -> dict[str, float]:
    q = np.asarray(q, dtype=np.float64)
    ee = np.asarray(ee, dtype=np.float64)
    dq = np.diff(q, n=1, axis=0)
    d2q = np.diff(q, n=2, axis=0)
    d3q = np.diff(q, n=3, axis=0)
    dee = np.diff(ee, n=1, axis=0)
    return {
        "joint_max_step_rad": float(np.max(np.abs(dq))),
        "joint_vel_rms_per_step": float(np.sqrt(np.mean(dq ** 2))),
        "joint_acc_d2_rms_per_step2": float(np.sqrt(np.mean(d2q ** 2))),
        "joint_acc_d2_mse_per_step2": float(np.mean(d2q ** 2)),
        "joint_jerk_d3_l2_mean": float(np.mean(np.linalg.norm(d3q, axis=-1))),
        "ee_path_length_m": float(np.sum(np.linalg.norm(dee, axis=-1))),
    }


def reference_comparison_stats(
    q: np.ndarray,
    ee: np.ndarray,
    q_ref_v2: np.ndarray,
    ee_ref_v2: np.ndarray,
    q_orig: np.ndarray,
    ee_orig: np.ndarray,
) -> dict[str, float]:
    q = np.asarray(q, dtype=np.float64)
    ee = np.asarray(ee, dtype=np.float64)
    return {
        "anchor_max_abs_vs_v2_rad": float(np.max(np.abs(q[ANCHOR_FRAMES] - q_ref_v2[ANCHOR_FRAMES]))),
        "anchor_mean_abs_vs_v2_rad": float(np.mean(np.abs(q[ANCHOR_FRAMES] - q_ref_v2[ANCHOR_FRAMES]))),
        "anchor_rmse_vs_v2_rad": float(np.sqrt(np.mean((q[ANCHOR_FRAMES] - q_ref_v2[ANCHOR_FRAMES]) ** 2))),
        "nonanchor_rmse_vs_v2_rad": float(np.sqrt(np.mean((q[NONANCHOR_FRAMES] - q_ref_v2[NONANCHOR_FRAMES]) ** 2))),
        "all_frames_rmse_vs_v2_rad": float(np.sqrt(np.mean((q - q_ref_v2) ** 2))),
        "anchor_fk_max_l2_vs_v2_m": float(np.max(np.linalg.norm(ee[ANCHOR_FRAMES] - ee_ref_v2[ANCHOR_FRAMES], axis=-1))),
        "nonanchor_fk_rmse_vs_v2_m": float(np.sqrt(np.mean(np.sum((ee[NONANCHOR_FRAMES] - ee_ref_v2[NONANCHOR_FRAMES]) ** 2, axis=-1)))),
        "all_frames_fk_rmse_vs_v2_m": float(np.sqrt(np.mean(np.sum((ee - ee_ref_v2) ** 2, axis=-1)))),
        # Post-hoc comparison vs original un-repaired q
        "posthoc_all_frames_rmse_vs_orig_rad": float(np.sqrt(np.mean((q - q_orig) ** 2))),
        "posthoc_nonanchor_rmse_vs_orig_rad": float(np.sqrt(np.mean((q[NONANCHOR_FRAMES] - q_orig[NONANCHOR_FRAMES]) ** 2))),
        "posthoc_all_frames_fk_rmse_vs_orig_m": float(np.sqrt(np.mean(np.sum((ee - ee_orig) ** 2, axis=-1)))),
    }


def compute_certificate(
    robot,
    B: np.ndarray,
    c: np.ndarray,
    q_saved: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    H: np.ndarray,
    b: np.ndarray,
    q_ref_v2: np.ndarray,
    ee_ref_v2: np.ndarray,
    q_orig: np.ndarray,
    ee_orig: np.ndarray,
    *,
    mode: str,
    c_raw: np.ndarray | None = None,
    q_raw: np.ndarray | None = None,
    ee_raw: np.ndarray | None = None,
) -> dict[str, Any]:
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    q_recomputed = decode_q(B, c)
    q_saved = np.asarray(q_saved, dtype=np.float64)
    q_bc_err = float(np.max(np.abs(q_saved - q_recomputed)))
    ee = fk_all(robot, q_saved)
    h_margin = ee[:, 2] - SAFE_Z
    lim_margin = np.minimum(q_saved - lower[None, :], upper[None, :] - q_saved)
    hc = H @ c - b
    anchor_err = float(np.max(np.abs(q_saved[ANCHOR_FRAMES] - q_ref_v2[ANCHOR_FRAMES])))
    hc_max = float(np.max(np.abs(hc)))
    hc_l2 = float(np.linalg.norm(hc))

    height_ok = bool(float(h_margin.min()) >= -HEIGHT_TOL)
    limits_ok = bool(float(lim_margin.min()) >= -LIMIT_TOL)
    q_bc_ok = bool(q_bc_err <= QBC_TOL)
    anchor_ok = bool(anchor_err <= ANCHOR_TOL)
    hc_ok = bool(hc_max <= HC_TOL)

    if mode == "hybrid":
        all_required_pass = bool(height_ok and limits_ok and q_bc_ok and anchor_ok and hc_ok)
    else:
        all_required_pass = bool(height_ok and limits_ok and q_bc_ok)

    cert: dict[str, Any] = {
        "mode": mode,
        "all_required_pass": all_required_pass,
        "anchor_max_abs_rad": anchor_err,
        "anchor_ok_le_1e-6": anchor_ok,
        "Hc_minus_b_max_abs": hc_max,
        "Hc_minus_b_l2": hc_l2,
        "Hc_ok_le_1e-9": hc_ok,
        "q_minus_Bc_max_abs": q_bc_err,
        "q_bc_ok_le_1e-9": q_bc_ok,
        "min_safe_z_margin_m": float(h_margin.min()),
        "height_ok_ge_-1e-6": height_ok,
        "strict_height_ge_0": bool(float(h_margin.min()) >= 0.0),
        "min_joint_limit_margin_rad": float(lim_margin.min()),
        "limits_ok_ge_-1e-9": limits_ok,
        "strict_limits_ge_0": bool(float(lim_margin.min()) >= 0.0),
        "per_frame_safe_z_margin_m": h_margin.tolist(),
        "motion_stats": trajectory_motion_stats(q_saved, ee),
        "ref_errors": reference_comparison_stats(q_saved, ee, q_ref_v2, ee_ref_v2, q_orig, ee_orig),
    }
    if c_raw is not None and q_raw is not None and ee_raw is not None:
        c_raw = np.asarray(c_raw, dtype=np.float64).reshape(-1)
        q_raw = np.asarray(q_raw, dtype=np.float64)
        ee_raw = np.asarray(ee_raw, dtype=np.float64)
        cert["repair_delta"] = {
            "coeff_l2": float(np.linalg.norm(c - c_raw)),
            "coeff_max_abs": float(np.max(np.abs(c - c_raw))),
            "q_max_abs_rad": float(np.max(np.abs(q_saved - q_raw))),
            "q_rmse_rad": float(np.sqrt(np.mean((q_saved - q_raw) ** 2))),
            "anchor_q_max_abs_change_rad": float(np.max(np.abs(q_saved[ANCHOR_FRAMES] - q_raw[ANCHOR_FRAMES]))),
            "fk_max_l2_m": float(np.max(np.linalg.norm(ee - ee_raw, axis=-1))),
            "fk_rmse_m": float(np.sqrt(np.mean(np.sum((ee - ee_raw) ** 2, axis=-1)))),
        }
    return cert


def compute_oracle_basis_reconstruction(
    robot,
    B: np.ndarray,
    H: np.ndarray,
    refs_v2: np.ndarray,
    q_orig_all: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    """Oracle basis representation check on the 16 eval v2 trajectories:
    encode cls = B^T q_ref (since B is column-orthonormal on 17-grid, pinv(B) = B^T),
    then hard project onto anchors b = q_ref[anchors].
    Distinguishes basis representation capability from generative sampling capability.
    """
    pinvB = np.linalg.pinv(B)
    rows = []
    for scene in range(16):
        q_ref = refs_v2[scene]
        q_orig = q_orig_all[scene]
        ee_ref = fk_all(robot, q_ref)
        ee_orig = fk_all(robot, q_orig)
        c_ls = (pinvB @ q_ref).reshape(-1)
        q_ls = decode_q(B, c_ls)
        ee_ls = fk_all(robot, q_ls)
        b = anchor_target_b(q_ref, ANCHOR_FRAMES)
        c_proj, pinfo = hard_project_c_numpy(H, c_ls, b)
        q_proj = decode_q(B, c_proj)
        ee_proj = fk_all(robot, q_proj)
        rep = safety_repair(robot, B, c_proj, lower, upper, mode="hybrid", H=H, b=b)
        q_rep = rep["q_final"]
        ee_rep = rep["ee_final"]
        rows.append(
            {
                "scene": scene,
                "ls_all_rmse_rad": float(np.sqrt(np.mean((q_ls - q_ref) ** 2))),
                "ls_nonanchor_rmse_rad": float(np.sqrt(np.mean((q_ls[NONANCHOR_FRAMES] - q_ref[NONANCHOR_FRAMES]) ** 2))),
                "ls_anchor_max_abs_rad": float(np.max(np.abs(q_ls[ANCHOR_FRAMES] - q_ref[ANCHOR_FRAMES]))),
                "proj_all_rmse_rad": float(np.sqrt(np.mean((q_proj - q_ref) ** 2))),
                "proj_nonanchor_rmse_rad": float(np.sqrt(np.mean((q_proj[NONANCHOR_FRAMES] - q_ref[NONANCHOR_FRAMES]) ** 2))),
                "proj_anchor_max_abs_rad": float(np.max(np.abs(q_proj[ANCHOR_FRAMES] - q_ref[ANCHOR_FRAMES]))),
                "proj_fk_nonanchor_rmse_m": float(np.sqrt(np.mean(np.sum((ee_proj[NONANCHOR_FRAMES] - ee_ref[NONANCHOR_FRAMES]) ** 2, axis=-1)))),
                "repaired_ok": bool(rep["ok"]),
                "repaired_nonanchor_rmse_rad": float(np.sqrt(np.mean((q_rep[NONANCHOR_FRAMES] - q_ref[NONANCHOR_FRAMES]) ** 2))),
                "repaired_fk_nonanchor_rmse_m": float(np.sqrt(np.mean(np.sum((ee_rep[NONANCHOR_FRAMES] - ee_ref[NONANCHOR_FRAMES]) ** 2, axis=-1)))),
                "repaired_vs_orig_nonanchor_rmse_rad": float(np.sqrt(np.mean((q_rep[NONANCHOR_FRAMES] - q_orig[NONANCHOR_FRAMES]) ** 2))),
                "repaired_vs_orig_fk_nonanchor_rmse_m": float(np.sqrt(np.mean(np.sum((ee_rep[NONANCHOR_FRAMES] - ee_orig[NONANCHOR_FRAMES]) ** 2, axis=-1)))),
            }
        )
    return {
        "per_scene": rows,
        "mean_ls_nonanchor_rmse_rad": float(np.mean([r["ls_nonanchor_rmse_rad"] for r in rows])),
        "mean_proj_nonanchor_rmse_rad": float(np.mean([r["proj_nonanchor_rmse_rad"] for r in rows])),
        "mean_proj_fk_nonanchor_rmse_m": float(np.mean([r["proj_fk_nonanchor_rmse_m"] for r in rows])),
        "mean_repaired_nonanchor_rmse_rad": float(np.mean([r["repaired_nonanchor_rmse_rad"] for r in rows])),
        "mean_repaired_fk_nonanchor_rmse_m": float(np.mean([r["repaired_fk_nonanchor_rmse_m"] for r in rows])),
        "repaired_pass_count": int(sum(1 for r in rows if r["repaired_ok"])),
    }


def ensure_reference_targets_sidecars(robot, cube, table, refs_v2: np.ndarray, meta_all: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    targets: dict[int, dict[str, Any]] = {}
    for scene in range(16):
        cpath = OUT / "motion" / "reference_targets" / f"ref_scene{scene:02d}.control.mp4"
        spath = OUT / "motion" / "reference_targets" / f"ref_scene{scene:02d}.sidecar.npz"
        tjson = OUT / "motion" / "reference_targets" / f"ref_scene{scene:02d}.targets.json"
        q_ref = refs_v2[scene]
        q_sha = sha256_array(q_ref)
        cached_ok = False
        if file_ok(cpath) and file_ok(spath) and file_ok(tjson):
            try:
                td = read_json(tjson)
                if td.get("q_sha256") == q_sha and len(td.get("bbox_centers", [])) == FRAMES:
                    targets[scene] = td
                    cached_ok = True
            except Exception:
                cached_ok = False
        if not cached_ok:
            info = render_control_and_sidecar(
                robot, cube, table, q_ref, np.asarray(meta_all[scene]["cube_xy"], dtype=np.float64), cpath, spath
            )
            npz_obj = np.load(spath)
            td = {
                "scene": scene,
                "q_sha256": q_sha,
                "control_path": str(cpath),
                "sidecar_path": str(spath),
                "bbox_centers": info["bbox_centers"],
                "centroids": info["centroids"],
                "bboxes": info["bboxes"],
                "anchors_bbox_centers": {str(a): info["bbox_centers"][a] for a in ANCHOR_FRAMES},
                "audit_rgb_centers": [
                    npz_obj["audit_rgb_centers"][f].tolist() if np.all(np.isfinite(npz_obj["audit_rgb_centers"][f])) else None
                    for f in range(FRAMES)
                ],
                "strict_rgb_centers": [
                    npz_obj["strict_rgb_centers"][f].tolist() if np.all(np.isfinite(npz_obj["strict_rgb_centers"][f])) else None
                    for f in range(FRAMES)
                ],
                "audit_ctrl_centers": [
                    npz_obj["audit_ctrl_centers"][f].tolist() if np.all(np.isfinite(npz_obj["audit_ctrl_centers"][f])) else None
                    for f in range(FRAMES)
                ],
                "strict_ctrl_centers": [
                    npz_obj["strict_ctrl_centers"][f].tolist() if np.all(np.isfinite(npz_obj["strict_ctrl_centers"][f])) else None
                    for f in range(FRAMES)
                ],
            }
            write_json(tjson, td)
            targets[scene] = td
    write_json(OUT / "motion" / "reference_targets" / "all_reference_targets.json", {str(k): v for k, v in targets.items()})
    return targets


def build_motion_manifest() -> list[dict[str, Any]]:
    rows = []
    idx = 0
    for group in GROUPS:
        gi = group_info(group)
        for scene in range(16):
            ref_img = EVAL_DS / "ref" / f"{scene:04d}.png"
            if not file_ok(ref_img):
                raise FileNotFoundError(f"Scene reference RGB image missing: {ref_img}")
            for m_idx, ms in enumerate(motion_seeds_for_scene(scene)):
                mid = sample_motion_id(scene, ms, group)
                rows.append(
                    {
                        "index": idx,
                        "motion_id": mid,
                        "group": gi["group"],
                        "group_slug": gi["group_slug"],
                        "basis_kind": gi["basis_kind"],
                        "mode": gi["mode"],
                        "scene": int(scene),
                        "motion_index": int(m_idx),
                        "motion_seed": int(ms),
                        "wan_seeds": list(WAN_SEEDS),
                        "raw_npz_path": str(OUT / "motion" / "raw" / f"{mid}.npz"),
                        "intermediate_json_path": str(OUT / "motion" / "intermediate" / f"{mid}.json"),
                        "final_npz_path": str(OUT / "motion" / "final" / f"{mid}.npz"),
                        "final_q_npy_path": str(OUT / "motion" / "final" / f"{mid}.q.npy"),
                        "certificate_json_path": str(OUT / "motion" / "certificates" / f"{mid}.json"),
                        "control_mp4_path": str(OUT / "motion" / "controls" / f"{mid}.control.mp4"),
                        "sidecar_npz_path": str(OUT / "motion" / "sidecars" / f"{mid}.sidecar.npz"),
                        "ref_image_path": str(ref_img),
                    }
                )
                idx += 1
    assert len(rows) == 128, f"Expected 128 motion rows, got {len(rows)}"
    assert all(file_ok(r["ref_image_path"]) for r in rows), "Not all 128 ref_image_path entries exist!"
    return rows


def build_video_manifest(motion_records: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    by_mid = {}
    if motion_records:
        for r in motion_records:
            by_mid[r["motion_id"]] = r
    else:
        for r in build_motion_manifest():
            by_mid[r["motion_id"]] = r

    vrows = []
    v_idx = 0
    for group in GROUPS:
        gi = group_info(group)
        for scene in range(16):
            ref_img = EVAL_DS / "ref" / f"{scene:04d}.png"
            if not file_ok(ref_img):
                raise FileNotFoundError(f"Scene reference RGB image missing: {ref_img}")
            for m_idx, ms in enumerate(motion_seeds_for_scene(scene)):
                mid = sample_motion_id(scene, ms, group)
                mrec = by_mid[mid]
                m_status = mrec.get("status", "pending")
                m_ok = bool(mrec.get("motion_ok", False))
                for ws in WAN_SEEDS:
                    vid = sample_video_id(scene, ms, group, ws)
                    vrows.append(
                        {
                            "index": v_idx,
                            "video_id": vid,
                            "motion_id": mid,
                            "group": gi["group"],
                            "group_slug": gi["group_slug"],
                            "basis_kind": gi["basis_kind"],
                            "mode": gi["mode"],
                            "scene": int(scene),
                            "motion_index": int(m_idx),
                            "motion_seed": int(ms),
                            "wan_seed": int(ws),
                            "motion_status": m_status,
                            "motion_ok": m_ok,
                            "q_path": mrec["final_q_npy_path"],
                            "final_npz_path": mrec["final_npz_path"],
                            "control_path": mrec["control_mp4_path"],
                            "sidecar_path": mrec["sidecar_npz_path"],
                            "ref_path": str(ref_img),
                            "video_path": str(OUT / "video" / "generated" / f"{vid}.mp4"),
                            "status": "motion_failed" if (m_status == "failed" or (m_status != "pending" and not m_ok)) else "pending",
                            "count": "failed" if (m_status == "failed" or (m_status != "pending" and not m_ok)) else "pending",
                            "failure_type": "motion_failed" if (m_status == "failed" or (m_status != "pending" and not m_ok)) else None,
                        }
                    )
                    v_idx += 1
    assert len(vrows) == 256, f"Expected 256 video rows, got {len(vrows)}"
    assert all(file_ok(r["ref_path"]) for r in vrows), "Not all 256 ref_path entries exist!"
    return vrows


def run_all_sampling_and_verification(device: str = "cpu") -> dict[str, Any]:
    from cnf_motion.scene import build_scene, disconnect_scene

    ensure_eval_dirs()
    refs_v2, refs_info = load_eval_refs_v2()
    q_orig_all = load_eval_q_orig()
    meta_all = load_eval_metadata()

    # Load both checkpoints
    loaded = {
        "fixed": load_basis_and_generator("fixed", device=device),
        "learned": load_basis_and_generator("learned", device=device),
    }

    manifest = build_motion_manifest()
    robot, cube, table = build_scene()
    try:
        lower, upper = joint_limits_from_scene(robot)
        # Verify v2 references compatibility
        ref_checks = []
        ee_refs_v2 = np.zeros((16, FRAMES, 3), dtype=np.float64)
        ee_orig_all = np.zeros((16, FRAMES, 3), dtype=np.float64)
        for s in range(16):
            ee_r = fk_all(robot, refs_v2[s])
            ee_o = fk_all(robot, q_orig_all[s])
            ee_refs_v2[s] = ee_r
            ee_orig_all[s] = ee_o
            hm = ee_r[:, 2] - SAFE_Z
            lm = np.minimum(refs_v2[s] - lower[None, :], upper[None, :] - refs_v2[s])
            ref_checks.append(
                {
                    "scene": s,
                    "q0_unchanged": bool(np.array_equal(refs_v2[s, 0], q_orig_all[s, 0])),
                    "min_safe_z_margin_m": float(hm.min()),
                    "min_limit_margin_rad": float(lm.min()),
                    "strict_ok": bool(hm.min() >= 0.0 and lm.min() >= 0.0 and np.array_equal(refs_v2[s, 0], q_orig_all[s, 0])),
                }
            )
        if not all(c["strict_ok"] for c in ref_checks):
            raise RuntimeError(f"v2 reference compatibility check failed: {ref_checks}")
        write_json(OUT / "motion" / "v2_reference_checks.json", {"info": refs_info, "per_scene": ref_checks})

        # Ensure reference uncompressed targets & sidecars
        ensure_reference_targets_sidecars(robot, cube, table, refs_v2, meta_all)

        # Oracle basis reconstruction comparison (representation vs generation separation)
        oracle_recon = {}
        for kind in ["fixed", "learned"]:
            oracle_recon[kind] = compute_oracle_basis_reconstruction(
                robot, loaded[kind]["B_ref"], loaded[kind]["H"], refs_v2, q_orig_all, lower, upper
            )
        write_json(
            OUT / "motion" / "basis_diagnostics_and_oracle.json",
            {
                "fixed": {"diagnostics": loaded["fixed"]["diagnostics"], "oracle_reconstruction": oracle_recon["fixed"]},
                "learned": {"diagnostics": loaded["learned"]["diagnostics"], "oracle_reconstruction": oracle_recon["learned"]},
            },
        )

        # Fail fast if numerical failure triggered in basis
        for kind in ["fixed", "learned"]:
            if loaded[kind]["diagnostics"]["numerical_failure_triggered"]:
                write_json(OUT / "motion" / f"basis_numerical_failure_{kind}.json", loaded[kind]["diagnostics"])

        completed_records = []
        u0_registry: dict[tuple[int, int, str], str] = {}

        for idx, item in enumerate(manifest):
            mid = item["motion_id"]
            kind = item["basis_kind"]
            mode = item["mode"]
            scene = int(item["scene"])
            ms = int(item["motion_seed"])
            pkg = loaded[kind]
            B = pkg["B_ref"]
            H = pkg["H"]
            mean = pkg["mean"]
            std = pkg["std"]
            ckpt_sha = pkg["diagnostics"]["ckpt_sha256"]

            raw_npz = Path(item["raw_npz_path"])
            inter_json = Path(item["intermediate_json_path"])
            final_npz = Path(item["final_npz_path"])
            final_q_npy = Path(item["final_q_npy_path"])
            cert_json = Path(item["certificate_json_path"])
            ctrl_mp4 = Path(item["control_mp4_path"])
            side_npz = Path(item["sidecar_npz_path"])

            q_ref = refs_v2[scene]
            ee_ref = ee_refs_v2[scene]
            q_orig = q_orig_all[scene]
            ee_orig = ee_orig_all[scene]
            q0 = q_ref[0]
            b = anchor_target_b(q_ref, ANCHOR_FRAMES)

            # Check cache integrity
            cached_valid = False
            if (
                file_ok(raw_npz)
                and file_ok(final_npz)
                and file_ok(final_q_npy)
                and file_ok(cert_json)
                and file_ok(ctrl_mp4)
                and file_ok(side_npz)
            ):
                try:
                    f_obj = np.load(final_npz, allow_pickle=True)
                    r_obj = np.load(raw_npz, allow_pickle=True)
                    q_loaded = np.load(final_q_npy)
                    cert_loaded = read_json(cert_json)
                    ctrl_dec = decode_video(ctrl_mp4)
                    if (
                        str(f_obj["ckpt_sha256"]) == ckpt_sha
                        and q_loaded.shape == (FRAMES, JOINTS)
                        and ctrl_dec.shape == (FRAMES, 256, 256, 3)
                        and cert_loaded.get("motion_ok") is not None
                    ):
                        # Re-verify reloaded certificate
                        re_cert = compute_certificate(
                            robot,
                            B,
                            f_obj["c_final"],
                            q_loaded,
                            lower,
                            upper,
                            H,
                            b,
                            q_ref,
                            ee_ref,
                            q_orig,
                            ee_orig,
                            mode=mode,
                            c_raw=r_obj["c_raw"],
                            q_raw=r_obj["q_raw"],
                            ee_raw=r_obj["ee_raw"],
                        )
                        u0_sha = str(r_obj["u0_sha256"])
                        u0_registry[(scene, ms, item["group"])] = u0_sha
                        rec = {
                            **item,
                            "status": "cached" if re_cert["all_required_pass"] else "failed",
                            "motion_ok": bool(re_cert["all_required_pass"]),
                            "ckpt_sha256": ckpt_sha,
                            "u0_sha256": u0_sha,
                            "raw_certificate": cert_loaded.get("raw_certificate"),
                            "final_certificate": re_cert,
                            "repair_info": cert_loaded.get("repair_info"),
                            "control_decoded_shape": list(ctrl_dec.shape),
                            "control_sha256": sha256_file(ctrl_mp4),
                            "sidecar_sha256": sha256_file(side_npz),
                        }
                        completed_records.append(rec)
                        cached_valid = True
                except Exception:
                    cached_valid = False

            if cached_valid:
                continue

            update_status_eval(
                "motion_sampling",
                done=idx,
                total=len(manifest),
                current_motion_id=mid,
            )
            t0 = time.time()
            try:
                if pkg["diagnostics"]["numerical_failure_triggered"]:
                    raise RuntimeError(f"Basis {kind} triggered numerical failure threshold")

                samp = sample_normalized_u(
                    pkg["model"],
                    q0,
                    b,
                    ms,
                    steps=EULER_STEPS,
                    mode=mode,
                    H=H,
                    b=b,
                    mean=mean,
                    std=std,
                )
                u0_sha = samp["u0_sha256"]
                u0_registry[(scene, ms, item["group"])] = u0_sha

                c_raw = samp["c_raw"]
                q_raw = decode_q(B, c_raw)
                ee_raw = fk_all(robot, q_raw)

                raw_cert = compute_certificate(
                    robot, B, c_raw, q_raw, lower, upper, H, b, q_ref, ee_ref, q_orig, ee_orig, mode=mode
                )

                np.savez(
                    raw_npz,
                    u0=samp["u0"],
                    u0_sha256=np.array(u0_sha),
                    u_before_final_proj=samp["u_before_final_proj"],
                    c_before_final_proj=samp["c_before_final_proj"],
                    u_raw=samp["u_raw"],
                    c_raw=c_raw,
                    q_raw=q_raw,
                    ee_raw=ee_raw,
                    B=B,
                    ckpt_sha256=np.array(ckpt_sha),
                )
                write_json(
                    inter_json,
                    {
                        "motion_id": mid,
                        "u0_sha256": u0_sha,
                        "step_infos": samp["step_infos"],
                        "final_proj_info": samp["final_proj_info"],
                    },
                )

                rep = safety_repair(robot, B, c_raw, lower, upper, mode=mode, H=H, b=b)
                c_final = rep["c_final"]
                q_final = decode_q(B, c_final)
                ee_final = fk_all(robot, q_final)

                np.savez(
                    final_npz,
                    c_linear=rep["c_linear"],
                    c_final=c_final,
                    z=rep["z"],
                    q_final=q_final,
                    ee_final=ee_final,
                    B=B,
                    ckpt_sha256=np.array(ckpt_sha),
                    u0_sha256=np.array(u0_sha),
                )
                np.save(final_q_npy, q_final)

                # Reload from disk to verify final certificate
                q_reloaded = np.load(final_q_npy)
                f_reloaded = np.load(final_npz, allow_pickle=True)
                final_cert = compute_certificate(
                    robot,
                    B,
                    f_reloaded["c_final"],
                    q_reloaded,
                    lower,
                    upper,
                    H,
                    b,
                    q_ref,
                    ee_ref,
                    q_orig,
                    ee_orig,
                    mode=mode,
                    c_raw=c_raw,
                    q_raw=q_raw,
                    ee_raw=ee_raw,
                )

                motion_ok = bool(rep["ok"] and final_cert["all_required_pass"])
                ctrl_info = render_control_and_sidecar(
                    robot,
                    cube,
                    table,
                    q_reloaded,
                    np.asarray(meta_all[scene]["cube_xy"], dtype=np.float64),
                    ctrl_mp4,
                    side_npz,
                )

                repair_summary = {
                    k: rep[k]
                    for k in [
                        "ok",
                        "converged",
                        "reason",
                        "message",
                        "nit",
                        "proj_info",
                        "min_height_margin_m",
                        "min_limit_margin_rad",
                        "strict_height_ge0",
                        "strict_limits_ge0",
                        "coeff_change_l2",
                        "nullspace_change_l2",
                    ]
                }
                cert_bundle = {
                    "motion_id": mid,
                    "group": item["group"],
                    "basis_kind": kind,
                    "mode": mode,
                    "scene": scene,
                    "motion_seed": ms,
                    "ckpt_sha256": ckpt_sha,
                    "u0_sha256": u0_sha,
                    "motion_ok": motion_ok,
                    "raw_certificate": raw_cert,
                    "repair_info": repair_summary,
                    "final_certificate": final_cert,
                    "elapsed_sec": time.time() - t0,
                }
                write_json(cert_json, cert_bundle)

                rec = {
                    **item,
                    "status": "generated" if motion_ok else "failed",
                    "motion_ok": motion_ok,
                    "failure_reason": None if motion_ok else (rep["reason"] or "certificate_failed"),
                    "ckpt_sha256": ckpt_sha,
                    "u0_sha256": u0_sha,
                    "raw_certificate": raw_cert,
                    "repair_info": repair_summary,
                    "final_certificate": final_cert,
                    "control_decoded_shape": ctrl_info["decoded_shape"],
                    "control_sha256": ctrl_info["control_sha256"],
                    "sidecar_sha256": ctrl_info["sidecar_sha256"],
                    "elapsed_sec": time.time() - t0,
                }
                completed_records.append(rec)
                append_eval_log(
                    f"motion [{idx+1}/128] {mid} ok={motion_ok} anchor={final_cert['anchor_max_abs_rad']:.2e} "
                    f"Hc={final_cert['Hc_minus_b_max_abs']:.2e} min_h={final_cert['min_safe_z_margin_m']:.2e}"
                )
            except Exception as e:
                err_rec = {
                    **item,
                    "status": "failed",
                    "motion_ok": False,
                    "failure_reason": "exception",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "elapsed_sec": time.time() - t0,
                }
                write_json(cert_json, err_rec)
                completed_records.append(err_rec)
                append_eval_log(f"motion [{idx+1}/128] {mid} FAILED: {e}")

            write_json(OUT / "manifests" / "motion_manifest.json", completed_records)

        # Verify initial noise pairing across Soft and Hybrid within each basis
        noise_pairing_checks = []
        for scene in range(16):
            for ms in motion_seeds_for_scene(scene):
                u_sf = u0_registry.get((scene, ms, "S-fixed"))
                u_hf = u0_registry.get((scene, ms, "H-fixed"))
                u_sl = u0_registry.get((scene, ms, "S-learned"))
                u_hl = u0_registry.get((scene, ms, "H-learned"))
                noise_pairing_checks.append(
                    {
                        "scene": scene,
                        "motion_seed": ms,
                        "fixed_pair_equal": bool(u_sf is not None and u_sf == u_hf),
                        "learned_pair_equal": bool(u_sl is not None and u_sl == u_hl),
                        "all_four_equal": bool(u_sf is not None and u_sf == u_hf == u_sl == u_hl),
                        "u0_sha256": u_sf,
                    }
                )

        # Aggregate group certificates summary
        group_summaries = {}
        for g in GROUPS:
            grecs = [r for r in completed_records if r["group"] == g]
            valid_recs = [r for r in grecs if r.get("final_certificate")]
            raw_recs = [r for r in grecs if r.get("raw_certificate")]
            group_summaries[g] = {
                "planned": len(grecs),
                "motion_ok_count": int(sum(1 for r in grecs if r.get("motion_ok"))),
                "failed_count": int(sum(1 for r in grecs if not r.get("motion_ok"))),
                "raw_anchor_max_abs_rad_max": float(max(r["raw_certificate"]["anchor_max_abs_rad"] for r in raw_recs)) if raw_recs else None,
                "raw_anchor_max_abs_rad_mean": float(np.mean([r["raw_certificate"]["anchor_max_abs_rad"] for r in raw_recs])) if raw_recs else None,
                "raw_Hc_minus_b_max_abs_max": float(max(r["raw_certificate"]["Hc_minus_b_max_abs"] for r in raw_recs)) if raw_recs else None,
                "raw_nonanchor_rmse_vs_v2_mean": float(np.mean([r["raw_certificate"]["ref_errors"]["nonanchor_rmse_vs_v2_rad"] for r in raw_recs])) if raw_recs else None,
                "raw_fk_nonanchor_rmse_vs_v2_mean": float(np.mean([r["raw_certificate"]["ref_errors"]["nonanchor_fk_rmse_vs_v2_m"] for r in raw_recs])) if raw_recs else None,
                "raw_min_safe_z_margin_m_min": float(min(r["raw_certificate"]["min_safe_z_margin_m"] for r in raw_recs)) if raw_recs else None,
                "raw_strict_height_pass_count": int(sum(1 for r in raw_recs if r["raw_certificate"]["strict_height_ge_0"])),
                "final_anchor_max_abs_rad_max": float(max(r["final_certificate"]["anchor_max_abs_rad"] for r in valid_recs)) if valid_recs else None,
                "final_anchor_max_abs_rad_mean": float(np.mean([r["final_certificate"]["anchor_max_abs_rad"] for r in valid_recs])) if valid_recs else None,
                "final_Hc_minus_b_max_abs_max": float(max(r["final_certificate"]["Hc_minus_b_max_abs"] for r in valid_recs)) if valid_recs else None,
                "final_q_minus_Bc_max_abs_max": float(max(r["final_certificate"]["q_minus_Bc_max_abs"] for r in valid_recs)) if valid_recs else None,
                "final_min_safe_z_margin_m_min": float(min(r["final_certificate"]["min_safe_z_margin_m"] for r in valid_recs)) if valid_recs else None,
                "final_min_limit_margin_rad_min": float(min(r["final_certificate"]["min_joint_limit_margin_rad"] for r in valid_recs)) if valid_recs else None,
                "final_strict_height_pass_count": int(sum(1 for r in valid_recs if r["final_certificate"]["strict_height_ge_0"])),
                "final_strict_limits_pass_count": int(sum(1 for r in valid_recs if r["final_certificate"]["strict_limits_ge_0"])),
                "final_tol_height_pass_count": int(sum(1 for r in valid_recs if r["final_certificate"]["height_ok_ge_-1e-6"])),
                "final_tol_limits_pass_count": int(sum(1 for r in valid_recs if r["final_certificate"]["limits_ok_ge_-1e-9"])),
                "final_nonanchor_rmse_vs_v2_mean": float(np.mean([r["final_certificate"]["ref_errors"]["nonanchor_rmse_vs_v2_rad"] for r in valid_recs])) if valid_recs else None,
                "final_fk_nonanchor_rmse_vs_v2_mean": float(np.mean([r["final_certificate"]["ref_errors"]["nonanchor_fk_rmse_vs_v2_m"] for r in valid_recs])) if valid_recs else None,
                "final_posthoc_nonanchor_rmse_vs_orig_mean": float(np.mean([r["final_certificate"]["ref_errors"]["posthoc_nonanchor_rmse_vs_orig_rad"] for r in valid_recs])) if valid_recs else None,
                "final_joint_acc_d2_mse_mean": float(np.mean([r["final_certificate"]["motion_stats"]["joint_acc_d2_mse_per_step2"] for r in valid_recs])) if valid_recs else None,
                "final_joint_jerk_d3_l2_mean": float(np.mean([r["final_certificate"]["motion_stats"]["joint_jerk_d3_l2_mean"] for r in valid_recs])) if valid_recs else None,
                "repair_coeff_l2_mean": float(np.mean([r["final_certificate"]["repair_delta"]["coeff_l2"] for r in valid_recs])) if valid_recs else None,
                "repair_q_max_abs_mean": float(np.mean([r["final_certificate"]["repair_delta"]["q_max_abs_rad"] for r in valid_recs])) if valid_recs else None,
                "repair_fk_max_l2_mean": float(np.mean([r["final_certificate"]["repair_delta"]["fk_max_l2_m"] for r in valid_recs])) if valid_recs else None,
            }

        summary = {
            "total_planned": 128,
            "motion_ok_total": int(sum(1 for r in completed_records if r.get("motion_ok"))),
            "failed_total": int(sum(1 for r in completed_records if not r.get("motion_ok"))),
            "noise_pairing_all_ok": bool(all(c["fixed_pair_equal"] and c["learned_pair_equal"] for c in noise_pairing_checks)),
            "noise_pairing_checks": noise_pairing_checks,
            "by_group": group_summaries,
            "basis_diagnostics": {
                "fixed": loaded["fixed"]["diagnostics"],
                "learned": loaded["learned"]["diagnostics"],
            },
            "oracle_reconstruction": oracle_recon,
        }
        write_json(OUT / "manifests" / "motion_manifest.json", completed_records)
        write_json(OUT / "motion" / "motion_summary.json", summary)
        if not summary["noise_pairing_all_ok"]:
            raise RuntimeError(f"Initial noise pairing check failed: {noise_pairing_checks}")
        vmanifest = build_video_manifest(completed_records)
        write_json(OUT / "manifests" / "video_manifest.json", vmanifest)
        update_status_eval(
            "motion_sampling_done",
            motion_planned=128,
            motion_ok=summary["motion_ok_total"],
            motion_failed=summary["failed_total"],
        )
        return {"records": completed_records, "summary": summary, "video_manifest": vmanifest}
    finally:
        disconnect_scene()


def run_cpu_numerical_checks() -> dict[str, Any]:
    """Deterministic CPU numerical verification of the evaluation pipeline components
    (without fabricating formal model results).
    """
    from cnf_motion.scene import build_scene, disconnect_scene
    from cnf_motion.soft_field import PandaSoftFlowModel
    from .basis import NeuralKernelBasis

    ensure_eval_dirs()
    refs_v2, refs_info = load_eval_refs_v2()
    q_orig_all = load_eval_q_orig()
    meta_all = load_eval_metadata()

    torch.manual_seed(20261114)
    np.random.seed(20261114)

    basis = NeuralKernelBasis(ell=0.2)
    basis.eval()
    with torch.no_grad():
        B_ref_t, R_t = basis.forward_ref()
        B_at_t = basis.forward_B_at(basis.ref_t, R_t)
    B_ref = B_ref_t.detach().cpu().double().numpy()
    B_at = B_at_t.detach().cpu().double().numpy()
    ortho_err = float(np.max(np.abs(B_ref.T @ B_ref - np.eye(DEGREE))))
    at_err = float(np.max(np.abs(B_at - B_ref)))

    H = build_H(B_ref, ANCHOR_FRAMES)
    q_ref0 = refs_v2[0]
    b0 = anchor_target_b(q_ref0, ANCHOR_FRAMES)

    # Test hard_project_H exactness and orthogonality to ker(H)
    c_rand = np.random.randn(DEGREE * JOINTS).astype(np.float64)
    c_proj, pinfo = hard_project_c_numpy(H, c_rand, b0)
    N = nullspace_Q(H)
    null_ortho_err = float(np.max(np.abs(N.T @ (c_proj - c_rand))))
    hc_res = float(np.max(np.abs(H @ c_proj - b0)))

    # Test sampling loop float32/float64 roundtrip and initial noise matching
    dummy_model = PandaSoftFlowModel(joints=JOINTS, degree=DEGREE, anchors=len(ANCHOR_FRAMES), hidden=256)
    dummy_model.eval()
    s_soft = sample_normalized_u(
        dummy_model, q_ref0[0], b0, 5000, steps=5, mode="soft", H=H, b=b0, mean=0.05, std=0.25
    )
    s_hyb = sample_normalized_u(
        dummy_model, q_ref0[0], b0, 5000, steps=5, mode="hybrid", H=H, b=b0, mean=0.05, std=0.25
    )
    same_u0 = bool(s_soft["u0_sha256"] == s_hyb["u0_sha256"] and np.array_equal(s_soft["u0"], s_hyb["u0"]))
    hyb_raw_hc_err = float(np.max(np.abs(H @ s_hyb["c_raw"] - b0)))
    hyb_raw_q = decode_q(B_ref, s_hyb["c_raw"])
    hyb_raw_anchor_err = float(np.max(np.abs(hyb_raw_q[ANCHOR_FRAMES] - q_ref0[ANCHOR_FRAMES])))

    # Test PyBullet SLSQP safety repair in I_70 (soft) vs ker(H) (hybrid) + uncompressed sidecar
    robot, cube, table = build_scene()
    try:
        lower, upper = joint_limits_from_scene(robot)
        ee_ref0 = fk_all(robot, q_ref0)
        ee_orig0 = fk_all(robot, q_orig_all[0])

        # Construct a slightly perturbed coefficient vector near LS encoding of q_ref0
        c_ls0 = (np.linalg.pinv(B_ref) @ q_ref0).reshape(-1)
        c_pert = c_ls0 + 0.02 * np.random.randn(DEGREE * JOINTS)

        rep_soft = safety_repair(robot, B_ref, c_pert, lower, upper, mode="soft")
        cert_soft = compute_certificate(
            robot,
            B_ref,
            rep_soft["c_final"],
            rep_soft["q_final"],
            lower,
            upper,
            H,
            b0,
            q_ref0,
            ee_ref0,
            q_orig_all[0],
            ee_orig0,
            mode="soft",
            c_raw=c_pert,
            q_raw=decode_q(B_ref, c_pert),
            ee_raw=fk_all(robot, decode_q(B_ref, c_pert)),
        )

        rep_hyb = safety_repair(robot, B_ref, c_pert, lower, upper, mode="hybrid", H=H, b=b0)
        cert_hyb = compute_certificate(
            robot,
            B_ref,
            rep_hyb["c_final"],
            rep_hyb["q_final"],
            lower,
            upper,
            H,
            b0,
            q_ref0,
            ee_ref0,
            q_orig_all[0],
            ee_orig0,
            mode="hybrid",
            c_raw=c_pert,
            q_raw=decode_q(B_ref, c_pert),
            ee_raw=fk_all(robot, decode_q(B_ref, c_pert)),
        )

        # Verify reference targets sidecar rendering on scene 0
        tmp_cpath = OUT / "motion" / "reference_targets" / "ref_scene00.control.mp4"
        tmp_spath = OUT / "motion" / "reference_targets" / "ref_scene00.sidecar.npz"
        side_info = render_control_and_sidecar(
            robot, cube, table, q_ref0, np.asarray(meta_all[0]["cube_xy"], dtype=np.float64), tmp_cpath, tmp_spath
        )
    finally:
        disconnect_scene()

    all_passed = bool(
        ortho_err <= 1e-12
        and at_err <= 1e-12
        and null_ortho_err <= 1e-12
        and hc_res <= 1e-12
        and same_u0
        and s_hyb["u_raw"].dtype == np.float32
        and s_hyb["c_raw"].dtype == np.float64
        and hyb_raw_hc_err <= 1e-9
        and hyb_raw_anchor_err <= 1e-6
        and cert_soft["all_required_pass"]
        and cert_soft["anchor_max_abs_rad"] > 1e-5  # proves Soft repair did NOT secretly project anchors
        and cert_hyb["all_required_pass"]
        and cert_hyb["anchor_max_abs_rad"] <= 1e-6
        and cert_hyb["Hc_minus_b_max_abs"] <= 1e-9
        and cert_hyb["q_minus_Bc_max_abs"] <= 1e-9
        and side_info["decoded_shape"] == [17, 256, 256, 3]
    )

    result = {
        "time": utcnow(),
        "all_passed": all_passed,
        "basis_orthonormality_max_err": ortho_err,
        "basis_forward_B_at_ref_max_err": at_err,
        "hard_project_Hc_minus_b_max_err": hc_res,
        "hard_project_nullspace_orthogonality_max_err": null_ortho_err,
        "same_initial_noise_u0_verified": same_u0,
        "u_dtype": str(s_hyb["u_raw"].dtype),
        "c_dtype": str(s_hyb["c_raw"].dtype),
        "hybrid_sampling_raw_Hc_err": hyb_raw_hc_err,
        "hybrid_sampling_raw_anchor_err_rad": hyb_raw_anchor_err,
        "soft_repair_pass": cert_soft["all_required_pass"],
        "soft_repair_anchor_unprojected_rad": cert_soft["anchor_max_abs_rad"],
        "soft_repair_min_safe_z_margin_m": cert_soft["min_safe_z_margin_m"],
        "hybrid_repair_pass": cert_hyb["all_required_pass"],
        "hybrid_repair_anchor_err_rad": cert_hyb["anchor_max_abs_rad"],
        "hybrid_repair_Hc_err": cert_hyb["Hc_minus_b_max_abs"],
        "hybrid_repair_q_minus_Bc_err": cert_hyb["q_minus_Bc_max_abs"],
        "hybrid_repair_min_safe_z_margin_m": cert_hyb["min_safe_z_margin_m"],
        "control_mp4_decoded_shape": side_info["decoded_shape"],
    }
    write_json(OUT / "protocol" / "cpu_numerical_verification.json", result)
    return result
