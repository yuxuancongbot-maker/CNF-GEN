from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .basis import ANCHOR_FRAMES, FRAMES, JOINTS
from .config import (
    EVAL_DS as _EVAL_DS,
    OUT as _OUT,
    TRAIN_DS as _TRAIN_DS,
    V2_REFS_DIR as _V2_REFS_DIR,
    ConfigError,
)

OUT_DEFAULT = _OUT
TRAIN_DIR = _TRAIN_DS
EVAL_DIR = _EVAL_DS
EVAL_V2_REFS_DIR = _V2_REFS_DIR

SAFE_Z = np.array([0.3655 if 4 <= k <= 13 else 0.332 for k in range(FRAMES)], dtype=np.float64)
SOLVER_GUARD_M = 2e-6
MAXITER = 200
FTOL = 1e-12


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def _fk_all(robot: int, q: np.ndarray) -> np.ndarray:
    from cnf_motion.scene import ee_position

    return np.array([ee_position(robot, row) for row in q], dtype=np.float64)


def _analytic_jac(robot: int, q_row: np.ndarray) -> np.ndarray:
    import pybullet as p

    pos = list(np.asarray(q_row, dtype=np.float64)) + [0.04, 0.04]
    jac, _ = p.calculateJacobian(robot, 11, [0, 0, 0], pos, [0.0] * 9, [0.0] * 9)
    return np.asarray(jac, dtype=np.float64)[:, :7]


def accepted_reference(
    robot: int,
    qold: np.ndarray,
    yaw: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply legacy IK clearance + 2 um positive numerical guard per v2 protocol."""
    from cnf_motion.scene import ee_position, enforce_clearance

    q = enforce_clearance(robot, qold, yaw).astype(np.float64)
    records: list[dict[str, Any]] = []
    for frame in range(FRAMES):
        start = q[frame].copy()
        target = float(SAFE_Z[frame] + SOLVER_GUARD_M)
        if ee_position(robot, start)[2] >= target:
            continue
        res = minimize(
            lambda x: 0.5 * float(np.dot(x - start, x - start)),
            start,
            jac=lambda x: np.asarray(x - start, dtype=np.float64),
            bounds=list(zip(lower, upper)),
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda x, t=target: float(ee_position(robot, x)[2] - t),
                    "jac": lambda x: _analytic_jac(robot, x)[2],
                }
            ],
            method="SLSQP",
            options={"maxiter": MAXITER, "ftol": FTOL},
        )
        q[frame] = np.asarray(res.x, dtype=np.float64)
        records.append(
            {
                "frame": int(frame),
                "converged": bool(res.success),
                "message": str(res.message),
                "iterations": int(getattr(res, "nit", 0) or 0),
                "guard_correction_l2_rad": float(np.linalg.norm(res.x - start)),
            }
        )
        if not res.success:
            raise RuntimeError(f"v2 reference guard solver failed at frame {frame}: {res.message}")

    ee = _fk_all(robot, q)
    if np.any(ee[:, 2] < SAFE_Z) or np.any(q < lower) or np.any(q > upper):
        raise RuntimeError("v2 reference failed strict same-safe_z or joint-limit check")
    if not np.array_equal(q[0], qold[0]):
        raise RuntimeError("v2 reference altered q0")
    return q, records


def load_raw_dataset(ds_dir: Path, expected_count: int) -> tuple[np.ndarray, list[str], list[dict[str, Any]], dict[str, str]]:
    q_dir = ds_dir / "q"
    files = sorted(q_dir.glob("*.npy"))
    if len(files) != expected_count:
        raise FileNotFoundError(f"Expected {expected_count} .npy files in {q_dir}, found {len(files)}")
    names = [f.name for f in files]
    hashes = {f.name: sha256_file(f) for f in files}
    q = np.stack([np.load(f).astype(np.float64) for f in files], axis=0)
    if q.shape != (expected_count, FRAMES, JOINTS):
        raise ValueError(f"Unexpected q shape {q.shape} in {q_dir}")
    meta = json.loads((ds_dir / "metadata.json").read_text(encoding="utf-8"))
    if len(meta) != expected_count:
        raise ValueError(f"Metadata count {len(meta)} != {expected_count} in {ds_dir}")
    return q, names, meta, hashes


def verify_cached_data(out_data_dir: Path) -> dict[str, Any] | None:
    train_npz = out_data_dir / "train_q_v2.npz"
    eval_npz = out_data_dir / "eval_q_v2.npz"
    meta_path = out_data_dir / "data_meta.json"
    if not (train_npz.exists() and eval_npz.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tr = np.load(train_npz, allow_pickle=True)
        ev = np.load(eval_npz, allow_pickle=True)
        q_tr = tr["q"].astype(np.float64)
        q_ev = ev["q"].astype(np.float64)
        if q_tr.shape != (128, FRAMES, JOINTS) or q_ev.shape != (16, FRAMES, JOINTS):
            return None
        if not (np.all(np.isfinite(q_tr)) and np.all(np.isfinite(q_ev))):
            return None
        if sha256_array(q_tr) != meta.get("train_q_v2_sha256"):
            return None
        if sha256_array(q_ev) != meta.get("eval_q_v2_sha256"):
            return None
        if not (meta.get("train_all_strict_ok") and meta.get("eval_all_strict_ok")):
            return None
        return meta
    except Exception:
        return None


def prepare_and_save(out_data_dir: Path | None = None, force: bool = False) -> dict[str, Any]:
    if out_data_dir is None:
        out_data_dir = OUT_DEFAULT / "data"
    out_data_dir = Path(out_data_dir)
    out_data_dir.mkdir(parents=True, exist_ok=True)

    if TRAIN_DIR is None or EVAL_DIR is None or EVAL_V2_REFS_DIR is None:
        raise ConfigError(
            "Data preparation requires TRAIN_DS, EVAL_DS and V2_COND to be configured "
            "(see learned_basis_hybrid_cnf.config)."
        )

    if not force:
        cached = verify_cached_data(out_data_dir)
        if cached is not None:
            return cached

    import pybullet as p
    from cnf_motion.scene import build_scene, disconnect_scene, ee_jacobian

    q_train_old, train_names, train_meta, train_old_hashes = load_raw_dataset(TRAIN_DIR, 128)
    q_eval_old, eval_names, eval_meta, eval_old_hashes = load_raw_dataset(EVAL_DIR, 16)

    q_train_v2 = np.zeros_like(q_train_old, dtype=np.float64)
    q_eval_v2 = np.zeros_like(q_eval_old, dtype=np.float64)
    train_diffs: list[dict[str, Any]] = []
    eval_checks: list[dict[str, Any]] = []
    eval_ref_hashes: dict[str, str] = {}

    robot, _, _ = build_scene()
    try:
        lower = np.array([p.getJointInfo(robot, j)[8] for j in range(JOINTS)], dtype=np.float64)
        upper = np.array([p.getJointInfo(robot, j)[9] for j in range(JOINTS)], dtype=np.float64)

        # 1. Repair all 128 train trajectories using the v2 reference rule
        for idx in range(128):
            qold = q_train_old[idx]
            yaw = float(train_meta[idx].get("yaw", 0.0))
            jac_err = float(np.abs(_analytic_jac(robot, qold[8]) - ee_jacobian(robot, qold[8])).max())
            if jac_err > 1e-7:
                raise RuntimeError(f"Train sample {idx} analytic vs central-diff Jacobian mismatch: {jac_err}")
            qref, guard_records = accepted_reference(robot, qold, yaw, lower, upper)
            q_train_v2[idx] = qref
            old_ee = _fk_all(robot, qold)
            new_ee = _fk_all(robot, qref)
            old_margin = old_ee[:, 2] - SAFE_Z
            new_margin = new_ee[:, 2] - SAFE_Z
            lim_margin = np.minimum(qref - lower[None, :], upper[None, :] - qref)
            anchor_changes = [
                {
                    "frame": int(f),
                    "q_max_change_rad": float(np.abs(qref[f] - qold[f]).max()),
                    "old_safe_z_margin_m": float(old_margin[f]),
                    "new_safe_z_margin_m": float(new_margin[f]),
                }
                for f in ANCHOR_FRAMES
            ]
            train_diffs.append(
                {
                    "sample": int(idx),
                    "name": train_names[idx],
                    "old_sha256": train_old_hashes[train_names[idx]],
                    "new_sha256": sha256_array(qref),
                    "q0_unchanged": bool(np.array_equal(qold[0], qref[0])),
                    "q_full_max_change_rad": float(np.abs(qref - qold).max()),
                    "old_min_safe_z_margin_m": float(old_margin.min()),
                    "new_min_safe_z_margin_m": float(new_margin.min()),
                    "min_joint_limit_margin_rad": float(lim_margin.min()),
                    "strict_height": bool(np.all(new_margin >= 0.0)),
                    "strict_limits": bool(np.all(lim_margin >= 0.0)),
                    "guard_solver_records": guard_records,
                    "analytic_jacobian_max_difference": jac_err,
                    "anchors": anchor_changes,
                }
            )

        # 2. Load and independently verify all 16 v2 eval reference trajectories
        for scene in range(16):
            qold = q_eval_old[scene]
            yaw = float(eval_meta[scene].get("yaw", 0.0))
            ref_path = EVAL_V2_REFS_DIR / f"{scene:04d}.npy"
            if not ref_path.exists():
                raise FileNotFoundError(f"Missing v2 eval reference: {ref_path}")
            qref = np.load(ref_path).astype(np.float64)
            if qref.shape != (FRAMES, JOINTS):
                raise ValueError(f"Eval reference {ref_path} has wrong shape {qref.shape}")
            eval_ref_hashes[f"{scene:04d}.npy"] = sha256_file(ref_path)
            q_recomp, _ = accepted_reference(robot, qold, yaw, lower, upper)
            recomp_diff = float(np.abs(qref - q_recomp).max())
            old_ee = _fk_all(robot, qold)
            new_ee = _fk_all(robot, qref)
            old_margin = old_ee[:, 2] - SAFE_Z
            new_margin = new_ee[:, 2] - SAFE_Z
            lim_margin = np.minimum(qref - lower[None, :], upper[None, :] - qref)
            q0_ok = bool(np.array_equal(qref[0], qold[0]))
            strict_h = bool(np.all(new_margin >= 0.0))
            strict_l = bool(np.all(lim_margin >= 0.0))
            if not (q0_ok and strict_h and strict_l):
                raise RuntimeError(f"Eval v2 reference {scene} failed verification: q0={q0_ok}, height={strict_h}, limits={strict_l}")
            q_eval_v2[scene] = qref
            eval_checks.append(
                {
                    "scene": int(scene),
                    "name": eval_names[scene],
                    "ref_path": str(ref_path),
                    "old_sha256": eval_old_hashes[eval_names[scene]],
                    "ref_file_sha256": eval_ref_hashes[f"{scene:04d}.npy"],
                    "ref_array_sha256": sha256_array(qref),
                    "q0_unchanged": q0_ok,
                    "q_full_max_change_rad": float(np.abs(qref - qold).max()),
                    "recomputed_max_abs_diff_rad": recomp_diff,
                    "old_min_safe_z_margin_m": float(old_margin.min()),
                    "min_safe_z_margin_m": float(new_margin.min()),
                    "min_joint_limit_margin_rad": float(lim_margin.min()),
                    "strict_height": strict_h,
                    "strict_limits": strict_l,
                }
            )
    finally:
        disconnect_scene()

    np.savez_compressed(
        out_data_dir / "train_q_v2.npz",
        q=q_train_v2,
        q_old=q_train_old,
        names=np.array(train_names, dtype=object),
    )
    np.savez_compressed(
        out_data_dir / "eval_q_v2.npz",
        q=q_eval_v2,
        q_old=q_eval_old,
        names=np.array(eval_names, dtype=object),
    )

    write_json_atomic(out_data_dir / "train_condition_diff.json", train_diffs)
    write_json_atomic(out_data_dir / "eval_compatibility_checks.json", eval_checks)

    meta_summary = {
        "protocol": "v2_compatible_reference_repair_20261114",
        "safe_z": SAFE_Z.tolist(),
        "solver_guard_m": SOLVER_GUARD_M,
        "anchor_frames": list(ANCHOR_FRAMES),
        "train_dir": str(TRAIN_DIR),
        "eval_dir": str(EVAL_DIR),
        "eval_v2_refs_dir": str(EVAL_V2_REFS_DIR),
        "train_shape": list(q_train_v2.shape),
        "eval_shape": list(q_eval_v2.shape),
        "train_q_old_sha256": sha256_array(q_train_old),
        "train_q_v2_sha256": sha256_array(q_train_v2),
        "eval_q_old_sha256": sha256_array(q_eval_old),
        "eval_q_v2_sha256": sha256_array(q_eval_v2),
        "train_npz_sha256": sha256_file(out_data_dir / "train_q_v2.npz"),
        "eval_npz_sha256": sha256_file(out_data_dir / "eval_q_v2.npz"),
        "train_all_strict_ok": bool(all(d["q0_unchanged"] and d["strict_height"] and d["strict_limits"] for d in train_diffs)),
        "eval_all_strict_ok": bool(all(c["q0_unchanged"] and c["strict_height"] and c["strict_limits"] for c in eval_checks)),
        "train_max_change_rad": float(max(d["q_full_max_change_rad"] for d in train_diffs)),
        "train_min_safe_z_margin_m": float(min(d["new_min_safe_z_margin_m"] for d in train_diffs)),
        "train_min_joint_limit_margin_rad": float(min(d["min_joint_limit_margin_rad"] for d in train_diffs)),
        "eval_max_change_rad": float(max(c["q_full_max_change_rad"] for c in eval_checks)),
        "eval_min_safe_z_margin_m": float(min(c["min_safe_z_margin_m"] for c in eval_checks)),
        "eval_min_joint_limit_margin_rad": float(min(c["min_joint_limit_margin_rad"] for c in eval_checks)),
        "eval_recomputed_max_diff_rad": float(max(c["recomputed_max_abs_diff_rad"] for c in eval_checks)),
        "train_old_file_hashes": train_old_hashes,
        "eval_old_file_hashes": eval_old_hashes,
        "eval_v2_ref_file_hashes": eval_ref_hashes,
    }
    write_json_atomic(out_data_dir / "data_meta.json", meta_summary)
    return meta_summary


if __name__ == "__main__":
    summary = prepare_and_save(force=True)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=2))
