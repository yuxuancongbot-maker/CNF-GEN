from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    BASE_1_3B,
    DOC_REPORT,
    EVAL_DS,
    EVALUATOR_CAL,
    EVALUATOR_PT,
    EVALUATOR_SPEC,
    OUT,
    PKG,
    TRAIN_DS,
    V2_COND,
    VIDEOX,
    WAN_LORA,
    WAN_MODEL,
)

FRAMES = 17
DEGREE = 10
JOINTS = 7
ANCHOR_FRAMES = [0, 4, 8, 12, 13, 16]
NONANCHOR_FRAMES = [f for f in range(FRAMES) if f not in ANCHOR_FRAMES]
SAFE_Z = np.array([0.3655 if 4 <= k <= 13 else 0.332 for k in range(FRAMES)], dtype=np.float64)
ANCHOR_TOL = 1e-6
HC_TOL = 1e-9
HEIGHT_TOL = 1e-6
LIMIT_TOL = 1e-9
QBC_TOL = 1e-9
SOLVER_GUARD_M = 2e-6
MAXITER = 200
FTOL = 1e-12
EULER_STEPS = 50
NOISE_SCALE = 0.5
WAN_STEPS = 20
WAN_CFG = 5.0
WAN_SHIFT = 5.0
WAN_FPS = 8
WAN_SEEDS = [43, 44]
BLIND_SCENES = [0, 5, 10]
BOOT_SEED = 20261114
BOOT_N = 10000
DIAGONAL = float(256 * math.sqrt(2))

GROUPS = ["S-fixed", "H-fixed", "S-learned", "H-learned"]

CUBE_COLOR = np.array([255, 120, 20], dtype=np.uint8)
ARM_COLOR = np.array([30, 80, 220], dtype=np.uint8)
GRIPPER_COLOR = np.array([30, 200, 120], dtype=np.uint8)


def motion_seeds_for_scene(scene: int) -> list[int]:
    return [5000 + int(scene), 6000 + int(scene)]


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _clean_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean_json(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_clean_json(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_clean_json(obj), ensure_ascii=False, allow_nan=False) + "\n")


def append_eval_log(msg: str) -> None:
    log_path = OUT / "logs" / "eval.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{utcnow()}] {msg}"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def update_status_eval(stage: str, **kw: Any) -> dict[str, Any]:
    p = OUT / "status_eval.json"
    st: dict[str, Any] = {}
    if p.exists():
        try:
            st = read_json(p)
        except Exception:
            st = {}
    st.update(kw)
    st["stage"] = stage
    st["time"] = utcnow()
    st["pid"] = os.getpid()
    write_json(p, st)
    append_jsonl(OUT / "logs" / "status_eval_history.jsonl", st)
    return st


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()


def file_ok(path: str | Path | None, min_bytes: int = 1) -> bool:
    if path is None:
        return False
    p = Path(path)
    return p.exists() and p.is_file() and p.stat().st_size >= min_bytes


def sha256_file_if_exists(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    return sha256_file(p) if p.exists() else None


def ensure_eval_dirs() -> None:
    for d in [
        "data",
        "train",
        "logs",
        "motion",
        "motion/raw",
        "motion/intermediate",
        "motion/final",
        "motion/certificates",
        "motion/controls",
        "motion/sidecars",
        "motion/reference_targets",
        "video",
        "video/replay",
        "video/generated",
        "score",
        "score/frames",
        "score/blind",
        "score/blind/crops",
        "score/blind/overlays",
        "stats",
        "protocol",
        "manifests",
    ]:
        (OUT / d).mkdir(parents=True, exist_ok=True)


def normalize_group_name(group: str) -> str:
    g = group.strip().replace("_", "-")
    if g not in GROUPS:
        raise ValueError(f"Unknown group: {group}")
    return g


def group_slug(group: str) -> str:
    return normalize_group_name(group).replace("-", "_")


def group_info(group: str) -> dict[str, Any]:
    g = normalize_group_name(group)
    if g == "S-fixed":
        return {"group": "S-fixed", "group_slug": "S_fixed", "basis_kind": "fixed", "mode": "soft"}
    if g == "H-fixed":
        return {"group": "H-fixed", "group_slug": "H_fixed", "basis_kind": "fixed", "mode": "hybrid"}
    if g == "S-learned":
        return {"group": "S-learned", "group_slug": "S_learned", "basis_kind": "learned", "mode": "soft"}
    if g == "H-learned":
        return {"group": "H-learned", "group_slug": "H_learned", "basis_kind": "learned", "mode": "hybrid"}
    raise ValueError(group)


def checkpoint_path_for_basis(kind: str) -> Path:
    if kind in ("fixed", "frozen"):
        return OUT / "train" / "fixed.pt"
    if kind == "learned":
        return OUT / "train" / "learned.pt"
    raise ValueError(kind)


def sample_motion_id(scene: int, motion_seed: int, group: str) -> str:
    slug = group_slug(group)
    m_idx = 0 if int(motion_seed) == 5000 + int(scene) else (1 if int(motion_seed) == 6000 + int(scene) else int(motion_seed))
    return f"scene{int(scene):02d}_m{m_idx}_ms{int(motion_seed)}_{slug}"


def sample_video_id(scene: int, motion_seed: int, group: str, wan_seed: int) -> str:
    return f"{sample_motion_id(scene, motion_seed, group)}_wan{int(wan_seed)}"


def load_eval_metadata() -> list[dict[str, Any]]:
    return read_json(EVAL_DS / "metadata.json")


def load_eval_q_orig() -> np.ndarray:
    files = sorted((EVAL_DS / "q").glob("*.npy"))
    if len(files) != 16:
        raise FileNotFoundError(f"expected 16 eval q files, found {len(files)} in {EVAL_DS/'q'}")
    return np.stack([np.load(f).astype(np.float64) for f in files], axis=0)


def load_eval_refs_v2() -> tuple[np.ndarray, dict[str, Any]]:
    refs_cond = []
    for i in range(16):
        p = V2_COND / "references" / f"{i:04d}.npy"
        if not p.exists():
            raise FileNotFoundError(p)
        refs_cond.append(np.load(p).astype(np.float64))
    arr_cond = np.stack(refs_cond, axis=0)
    if arr_cond.shape != (16, FRAMES, JOINTS):
        raise ValueError(f"unexpected eval refs shape {arr_cond.shape}")

    q_orig = load_eval_q_orig()
    q0_exact = bool(np.array_equal(arr_cond[:, 0, :], q_orig[:, 0, :]))
    if not q0_exact:
        raise ValueError("v2 eval references do not preserve exact q0 vs original eval dataset")

    npz_path = OUT / "data" / "eval_q_v2.npz"
    source = str(V2_COND / "references")
    matches_cond = True
    if file_ok(npz_path):
        obj = np.load(npz_path, allow_pickle=True)
        q_npz = obj["q"].astype(np.float64)
        if q_npz.shape != (16, FRAMES, JOINTS):
            raise ValueError(f"eval_q_v2.npz bad shape: {q_npz.shape}")
        max_diff = float(np.max(np.abs(q_npz - arr_cond)))
        if max_diff <= 1e-10 and np.array_equal(q_npz[:, 0, :], q_orig[:, 0, :]):
            arr = q_npz
            source = str(npz_path)
        else:
            # Verify whether q_npz is valid v2 or un-repaired original q
            matches_cond = False
            arr = arr_cond
    else:
        arr = arr_cond

    info = {
        "source": source,
        "sha256": sha256_array(arr),
        "q0_exact_vs_orig": q0_exact,
        "eval_q_v2_npz_exists": file_ok(npz_path),
        "eval_q_v2_npz_matches_v2_cond": matches_cond,
    }
    return arr, info


def flatten_c(c_dj: np.ndarray) -> np.ndarray:
    c_dj = np.asarray(c_dj, dtype=np.float64)
    if c_dj.shape != (DEGREE, JOINTS):
        raise ValueError(c_dj.shape)
    return c_dj.reshape(-1)


def unflatten_c(c_flat: np.ndarray) -> np.ndarray:
    c_flat = np.asarray(c_flat, dtype=np.float64).reshape(-1)
    if c_flat.shape[0] != DEGREE * JOINTS:
        raise ValueError(c_flat.shape)
    return c_flat.reshape(DEGREE, JOINTS)


def build_H(B: np.ndarray, anchor_frames: list[int] | None = None) -> np.ndarray:
    anchor_frames = ANCHOR_FRAMES if anchor_frames is None else list(anchor_frames)
    B = np.asarray(B, dtype=np.float64)
    if B.shape != (FRAMES, DEGREE):
        raise ValueError(B.shape)
    Ba = B[anchor_frames, :]  # [6,10]
    return np.kron(Ba, np.eye(JOINTS, dtype=np.float64))  # [42,70]


def decode_q(B: np.ndarray, c: np.ndarray) -> np.ndarray:
    B = np.asarray(B, dtype=np.float64)
    c_dj = unflatten_c(c) if np.asarray(c).ndim == 1 else np.asarray(c, dtype=np.float64)
    if B.shape != (FRAMES, DEGREE) or c_dj.shape != (DEGREE, JOINTS):
        raise ValueError((B.shape, c_dj.shape))
    return B @ c_dj  # [17,7]


def anchor_target_b(q_ref: np.ndarray, anchor_frames: list[int] | None = None) -> np.ndarray:
    anchor_frames = ANCHOR_FRAMES if anchor_frames is None else list(anchor_frames)
    q_ref = np.asarray(q_ref, dtype=np.float64)
    return q_ref[anchor_frames].reshape(-1)  # [42]


def nullspace_Q(H: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    _, s, Vt = np.linalg.svd(H, full_matrices=True)
    rank = int(np.sum(s > tol))
    return Vt[rank:].T.copy()  # [70, 70-rank]


def joint_limits_from_scene(robot) -> tuple[np.ndarray, np.ndarray]:
    import pybullet as p
    lower = np.array([p.getJointInfo(robot, j)[8] for j in range(JOINTS)], dtype=np.float64)
    upper = np.array([p.getJointInfo(robot, j)[9] for j in range(JOINTS)], dtype=np.float64)
    return lower, upper


def analytic_jacobian(robot, q: np.ndarray) -> np.ndarray:
    import pybullet as p
    pos = list(np.asarray(q, dtype=np.float64)) + [0.04, 0.04]
    jac, _ = p.calculateJacobian(robot, 11, [0, 0, 0], pos, [0.0] * 9, [0.0] * 9)
    return np.asarray(jac, dtype=np.float64)[:3, :7]


def fk_all(robot, q: np.ndarray) -> np.ndarray:
    from cnf_motion.scene import ee_position
    return np.array([ee_position(robot, row) for row in q], dtype=np.float64)


def mask_from_control(control_frame: np.ndarray) -> np.ndarray:
    return np.all(control_frame == CUBE_COLOR.reshape(1, 1, 3), axis=-1).astype(np.uint8)


def fg_union_from_control(control: np.ndarray) -> np.ndarray:
    union = np.zeros(control.shape[1:3], dtype=np.uint8)
    for f in range(control.shape[0]):
        m = (
            np.all(control[f] == CUBE_COLOR.reshape(1, 1, 3), axis=-1)
            | np.all(control[f] == ARM_COLOR.reshape(1, 1, 3), axis=-1)
            | np.all(control[f] == GRIPPER_COLOR.reshape(1, 1, 3), axis=-1)
        ).astype(np.uint8)
        union = np.maximum(union, m)
    return union


def mask_bbox_and_centers(mask: np.ndarray) -> tuple[list[int] | None, np.ndarray | None, np.ndarray | None]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None, None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox = [x0, y0, x1, y1]
    bbox_center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float32)
    centroid = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)
    return bbox, bbox_center, centroid


def select_largest_component(mask: np.ndarray):
    import cv2
    m = (mask > 0).astype(np.uint8)
    ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    comps = int(max(0, ncomp - 1))
    if comps == 0:
        return np.zeros_like(m), None, 0, None
    areas = [(k, int(stats[k, cv2.CC_STAT_AREA])) for k in range(1, ncomp)]
    k_best = max(areas, key=lambda x: x[1])[0]
    sel = (labels == k_best).astype(np.uint8)
    bbox, center, _ = mask_bbox_and_centers(sel)
    return sel, center, comps, bbox


def strict_hsv_detector(frame: np.ndarray):
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 0] >= 3) & (hsv[..., 0] <= 25) & (hsv[..., 1] >= 166) & (hsv[..., 2] >= 100)).astype(np.uint8)
    n, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    cands = [k for k in range(1, n) if 20 <= stats[k, cv2.CC_STAT_AREA] <= 1500]
    if not cands:
        return None, 0
    k = max(cands, key=lambda x: stats[x, cv2.CC_STAT_AREA])
    return centers[k].astype(np.float32), len(cands)


def audit_hsv_detector(frame: np.ndarray):
    from cnf_motion.metrics import detect_cube
    p = detect_cube(frame)
    if p is None:
        return None, 0
    return np.asarray(p, dtype=np.float32), 1


def decode_video(path: str | Path) -> np.ndarray:
    import imageio.v2 as iio
    return np.stack(iio.mimread(str(path)))


def render_control_and_sidecar(
    robot,
    cube,
    table,
    q: np.ndarray,
    cube_xy: np.ndarray,
    control_mp4_path: str | Path,
    sidecar_npz_path: str | Path,
    width: int = 256,
) -> dict[str, Any]:
    import imageio.v2 as iio
    from cnf_motion.scene import render_trajectory

    q_arr = np.asarray(q, dtype=np.float64)
    rgb, control, _ = render_trajectory(
        robot, cube, table, q_arr, np.asarray(cube_xy, dtype=np.float64), width=width
    )
    if control.shape != (FRAMES, width, width, 3):
        raise ValueError(f"Unexpected rendered control shape: {control.shape}")

    cube_masks = np.zeros((FRAMES, width, width), dtype=np.uint8)
    bboxes = np.full((FRAMES, 4), -1, dtype=np.int32)
    bbox_centers = np.full((FRAMES, 2), np.nan, dtype=np.float32)
    centroids = np.full((FRAMES, 2), np.nan, dtype=np.float32)
    audit_rgb_centers = np.full((FRAMES, 2), np.nan, dtype=np.float32)
    strict_rgb_centers = np.full((FRAMES, 2), np.nan, dtype=np.float32)
    audit_ctrl_centers = np.full((FRAMES, 2), np.nan, dtype=np.float32)
    strict_ctrl_centers = np.full((FRAMES, 2), np.nan, dtype=np.float32)

    centers_list = []
    centroids_list = []
    bboxes_list = []
    for f in range(FRAMES):
        cm = mask_from_control(control[f])
        cube_masks[f] = cm
        bbox, bcenter, ccenter = mask_bbox_and_centers(cm)
        if bbox is not None and bcenter is not None and ccenter is not None:
            bboxes[f] = np.array(bbox, dtype=np.int32)
            bbox_centers[f] = bcenter
            centroids[f] = ccenter
            centers_list.append(bcenter.tolist())
            centroids_list.append(ccenter.tolist())
            bboxes_list.append(bbox)
        else:
            centers_list.append(None)
            centroids_list.append(None)
            bboxes_list.append(None)
        ac_rgb, _ = audit_hsv_detector(rgb[f])
        sc_rgb, _ = strict_hsv_detector(rgb[f])
        ac_ctrl, _ = audit_hsv_detector(control[f])
        sc_ctrl, _ = strict_hsv_detector(control[f])
        if ac_rgb is not None:
            audit_rgb_centers[f] = ac_rgb
        if sc_rgb is not None:
            strict_rgb_centers[f] = sc_rgb
        if ac_ctrl is not None:
            audit_ctrl_centers[f] = ac_ctrl
        if sc_ctrl is not None:
            strict_ctrl_centers[f] = sc_ctrl

    fg_union = fg_union_from_control(control)

    control_mp4_path = Path(control_mp4_path)
    sidecar_npz_path = Path(sidecar_npz_path)
    control_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_npz_path.parent.mkdir(parents=True, exist_ok=True)

    iio.mimsave(str(control_mp4_path), control, fps=WAN_FPS)
    decoded = decode_video(control_mp4_path)
    if decoded.shape != (FRAMES, width, width, 3):
        raise RuntimeError(f"Decoded control MP4 shape mismatch: {decoded.shape}")

    np.savez(
        sidecar_npz_path,
        cube_masks=cube_masks,
        fg_union=fg_union,
        bboxes=bboxes,
        bbox_centers=bbox_centers,
        centroids=centroids,
        audit_rgb_centers=audit_rgb_centers,
        strict_rgb_centers=strict_rgb_centers,
        audit_ctrl_centers=audit_ctrl_centers,
        strict_ctrl_centers=strict_ctrl_centers,
        q_sha256=np.array(sha256_array(q_arr)),
    )
    write_json(
        Path(str(control_mp4_path) + ".centers.json"),
        {
            "source": "uncompressed_simulator_instance_mask",
            "centers": centers_list,
            "centroids": centroids_list,
            "bboxes": bboxes_list,
            "q_sha256": sha256_array(q_arr),
        },
    )
    np.save(Path(str(control_mp4_path) + ".fg_union.npy"), fg_union)

    return {
        "control_path": str(control_mp4_path),
        "control_sha256": sha256_file(control_mp4_path),
        "decoded_shape": list(decoded.shape),
        "sidecar_path": str(sidecar_npz_path),
        "sidecar_sha256": sha256_file(sidecar_npz_path),
        "bbox_centers": centers_list,
        "centroids": centroids_list,
        "bboxes": bboxes_list,
    }


def point_err(p: Any, t: Any) -> float | None:
    if p is None or t is None:
        return None
    pa = np.asarray(p, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(ta))):
        return None
    return float(np.linalg.norm(pa - ta))


def build_evaluation_protocol() -> dict[str, Any]:
    proto = {
        "protocol_name": "learned_basis_hybrid_cnf_eval_20261114",
        "date": "2026-11-14",
        "output_dir": str(OUT),
        "status_file": str(OUT / "status_eval.json"),
        "ready_file": str(OUT / "evaluation_ready.json"),
        "report_file": str(DOC_REPORT),
        "groups": [group_info(g) for g in GROUPS],
        "sampling": {
            "scenes": list(range(16)),
            "motion_seeds_rule": "[5000 + scene, 6000 + scene]",
            "total_trajectories": 128,
            "euler_steps": EULER_STEPS,
            "noise_scale": NOISE_SCALE,
            "u_dtype": "float32",
            "c_and_projection_dtype": "float64",
            "allowed_inputs": ["q0", "six_anchors_[0,4,8,12,13,16]", "shared_safety_metadata"],
            "forbidden_inputs": ["gt_nonanchor_velocity", "gt_initialization", "gt_fallback", "best_of_N"],
            "safety_repair": {
                "solver": "SLSQP",
                "maxiter": MAXITER,
                "ftol": FTOL,
                "safe_z": SAFE_Z.tolist(),
                "solver_guard_m": SOLVER_GUARD_M,
                "soft_update_space": "I_70 (no anchor projection)",
                "hybrid_update_space": "ker(H) (anchor-preserving)",
                "representation_preserved": "q_final = B @ c_final",
            },
            "certificates": {
                "hybrid_anchor_tol_rad": ANCHOR_TOL,
                "hybrid_Hc_minus_b_tol": HC_TOL,
                "q_minus_Bc_tol": QBC_TOL,
                "height_tol_m": -HEIGHT_TOL,
                "limit_tol_rad": -LIMIT_TOL,
            },
        },
        "wan_video": {
            "total_planned_videos": 256,
            "wan_seeds": WAN_SEEDS,
            "resolution": [256, 256],
            "frames": FRAMES,
            "fps": WAN_FPS,
            "steps": WAN_STEPS,
            "cfg": WAN_CFG,
            "shift": WAN_SHIFT,
            "model_path": str(WAN_MODEL) if WAN_MODEL is not None else None,
            "lora_path": str(WAN_LORA) if WAN_LORA is not None else None,
            "lora_sha256": sha256_file_if_exists(WAN_LORA),
            "gt_replay_samples": 8,
            "gt_replay_tolerance": {"max_abs": 1, "mae": 0.01},
            "reuse_old_AC_videos": False,
        },
        "scoring_and_statistics": {
            "evaluator_ckpt": str(EVALUATOR_PT) if EVALUATOR_PT is not None else None,
            "evaluator_ckpt_sha256": sha256_file_if_exists(EVALUATOR_PT),
            "evaluator_spec_sha256": sha256_file_if_exists(EVALUATOR_SPEC),
            "evaluator_cal_sha256": sha256_file_if_exists(EVALUATOR_CAL),
            "threshold": 0.5,
            "postprocessing": "largest_connected_component_bbox_center",
            "primary_metric": "anchor_epe_ref_penalized (6 anchors to common v2 reference, missed frames penalized by 256*sqrt(2) px)",
            "primary_contrast": "H-learned - S-learned",
            "auxiliary_contrasts": ["H-fixed - S-fixed", "H-learned - H-fixed", "S-learned - S-fixed"],
            "statistical_unit": "scene (16 scenes, averaging seeds within scene before paired bootstrap)",
            "bootstrap_resamples": BOOT_N,
            "bootstrap_seed": BOOT_SEED,
            "blind_overlays": {
                "scenes": BLIND_SCENES,
                "groups": GROUPS,
                "motion_index": 0,
                "wan_seeds": WAN_SEEDS,
                "frames": ANCHOR_FRAMES,
                "total_videos": 24,
            },
        },
    }
    write_json(OUT / "evaluation_protocol.json", proto)
    return proto
