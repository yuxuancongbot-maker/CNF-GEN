#!/usr/bin/env python3
"""Prepare the 16-scene v2 evaluation references and their render targets.

Consolidates ``make_v2_conditions`` and ``precompute_targets`` from the original
``video_task_validation_20261114.py`` (the upstream that produced the v2 eval
references and the reference target sidecars consumed by the learned-basis
Hybrid-CNF experiment).

Outputs (under ``CNF_V2_V2_COND``):
  references/{scene:04d}.npy   -- 16 v2-compatible joint trajectories
  condition_diff.json          -- per-scene repair provenance
  compatibility_checks.json    -- q0 / safe_z / joint-limit verification
  targets/scene{scene:02d}.json + targets/all_targets.json

Scenes 0..7 are reused from the legacy 8-scene v2 references
(``CNF_V2_OLD_V2_COND``); scenes 8..15 are generated with the same rule. Both
the base eval dataset and the legacy references are binary artifacts that are
not committed to Git.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from learned_basis_hybrid_cnf.basis import ANCHOR_FRAMES, FRAMES, JOINTS
from learned_basis_hybrid_cnf.config import (
    EVAL_DS,
    OLD_V2_COND,
    V2_COND,
    ConfigError,
)
from learned_basis_hybrid_cnf.data import (
    SAFE_Z,
    SOLVER_GUARD_M,
    _analytic_jac,
    _fk_all,
    accepted_reference,
)

CUBE_COLOR = np.array([255, 120, 20], dtype=np.uint8)


def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, Path):
        return str(x)
    return x


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_v2_conditions() -> dict[int, np.ndarray]:
    import pybullet as p
    from cnf_motion.scene import build_scene, disconnect_scene, ee_jacobian

    if EVAL_DS is None or V2_COND is None:
        raise ConfigError("prepare_v2_references requires EVAL_DS and V2_COND to be configured.")

    refs_dir = V2_COND / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((EVAL_DS / "metadata.json").read_text(encoding="utf-8"))

    if OLD_V2_COND is None:
        raise ConfigError(
            "Reusing legacy v2 references for scenes 0..7 requires OLD_V2_COND "
            "(the 8-scene v2 conditions dir with references/{0..7}.npy and condition_diff.json)."
        )
    old_diff = {d["sample"]: d for d in json.loads((OLD_V2_COND / "condition_diff.json").read_text(encoding="utf-8"))}

    refs: dict[int, np.ndarray] = {}
    cond_diff: list[dict[str, Any]] = []

    robot, cube, table = build_scene()
    try:
        lower = np.array([p.getJointInfo(robot, j)[8] for j in range(JOINTS)], dtype=np.float64)
        upper = np.array([p.getJointInfo(robot, j)[9] for j in range(JOINTS)], dtype=np.float64)
        for scene in range(16):
            out_ref = refs_dir / f"{scene:04d}.npy"
            qold = np.load(EVAL_DS / "q" / f"{scene:04d}.npy").astype(np.float64)
            if scene < 8:
                src = OLD_V2_COND / "references" / f"{scene}.npy"
                qref = np.load(src).astype(np.float64)
                np.save(out_ref, qref)
                d = dict(old_diff[scene])
                d["source"] = "reused_v2_0_7"
                d["reused_source_sha256"] = _sha256_file(src)
            else:
                jac_err = float(np.abs(_analytic_jac(robot, qold[8]) - ee_jacobian(robot, qold[8])).max())
                if jac_err > 1e-7:
                    raise RuntimeError(f"jac mismatch scene {scene}: {jac_err}")
                qref, guard = accepted_reference(robot, qold, meta[scene]["yaw"], lower, upper)
                np.save(out_ref, qref)
                old_ee, new_ee = _fk_all(robot, qold), _fk_all(robot, qref)
                old_v, new_v = (old_ee[9] - old_ee[7]) * 8, (new_ee[9] - new_ee[7]) * 8
                anchors = []
                for f in ANCHOR_FRAMES:
                    anchors.append(
                        {
                            "frame": f,
                            "old_q_rad": qold[f].tolist(),
                            "new_q_rad": qref[f].tolist(),
                            "q_max_change_rad": float(np.abs(qref[f] - qold[f]).max()),
                            "old_safe_z_margin_m": float(old_ee[f, 2] - SAFE_Z[f]),
                            "new_safe_z_margin_m": float(new_ee[f, 2] - SAFE_Z[f]),
                        }
                    )
                d = {
                    "sample": scene,
                    "source": "generated_v2_8_15_same_rule",
                    "q0_exactly_unchanged": bool(np.array_equal(qold[0], qref[0])),
                    "q_full_max_change_rad": float(np.abs(qref - qold).max()),
                    "old_velocity_m_per_normalized_t": old_v.tolist(),
                    "new_velocity_m_per_normalized_t": new_v.tolist(),
                    "velocity_target_l2_change": float(np.linalg.norm(new_v - old_v)),
                    "reference_guard_solver": guard,
                    "analytic_jacobian_max_difference": jac_err,
                    "anchors": anchors,
                }
            refs[scene] = qref
            cond_diff.append(d)
            print(f"v2 condition scene {scene}: {d.get('source')}", flush=True)
    finally:
        disconnect_scene()

    _write_json(V2_COND / "condition_diff.json", cond_diff)

    # Compatibility checks
    robot, _, _ = build_scene()
    try:
        lower = np.array([p.getJointInfo(robot, j)[8] for j in range(JOINTS)])
        upper = np.array([p.getJointInfo(robot, j)[9] for j in range(JOINTS)])
        checks = []
        for s in range(16):
            q = refs[s]
            ee = _fk_all(robot, q)
            m = ee[:, 2] - SAFE_Z
            lim = np.minimum(q - lower, upper - q)
            checks.append(
                {
                    "scene": s,
                    "q0_unchanged": bool(np.array_equal(q[0], np.load(EVAL_DS / "q" / f"{s:04d}.npy")[0])),
                    "min_safe_z_margin_m": float(m.min()),
                    "strict_height": bool(np.all(m >= 0)),
                    "min_joint_limit_margin_rad": float(lim.min()),
                    "strict_limits": bool(np.all(lim >= 0)),
                    "sha256": _sha256_file(V2_COND / "references" / f"{s:04d}.npy"),
                }
            )
    finally:
        disconnect_scene()
    _write_json(V2_COND / "compatibility_checks.json", checks)
    if not all(c["strict_height"] and c["strict_limits"] and c["q0_unchanged"] for c in checks):
        raise RuntimeError(f"compatibility failed: {[c for c in checks if not (c['strict_height'] and c['strict_limits'] and c['q0_unchanged'])]}")
    return refs


def _bbox_center(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [(x0 + x1) / 2.0, (y0 + y1) / 2.0]


def precompute_targets(refs: dict[int, np.ndarray]) -> None:
    from cnf_motion.scene import build_scene, disconnect_scene, render_trajectory

    if EVAL_DS is None or V2_COND is None:
        raise ConfigError("prepare_v2_references requires EVAL_DS and V2_COND to be configured.")
    targets_dir = V2_COND / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((EVAL_DS / "metadata.json").read_text(encoding="utf-8"))
    targets: dict[int, dict[str, Any]] = {}
    robot, cube, table = build_scene()
    try:
        for s in range(16):
            _, control, _ = render_trajectory(robot, cube, table, refs[s], np.array(meta[s]["cube_xy"]), width=256)
            centers = []
            for f in range(FRAMES):
                m = np.all(control[f] == CUBE_COLOR.reshape(1, 1, 3), axis=-1).astype(np.uint8)
                centers.append(_bbox_center(m))
            targets[s] = {
                "scene": s,
                "centers_all": centers,
                "anchors": {str(a): centers[a] for a in ANCHOR_FRAMES},
            }
            _write_json(targets_dir / f"scene{s:02d}.json", targets[s])
            print(f"targets scene {s} ready", flush=True)
    finally:
        disconnect_scene()
    _write_json(targets_dir / "all_targets.json", targets)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite existing v2 references/targets")
    args = parser.parse_args()

    if V2_COND is None:
        raise ConfigError("prepare_v2_references requires V2_COND to be configured.")
    refs_dir = V2_COND / "references"
    if not args.force and refs_dir.exists() and list(refs_dir.glob("*.npy")):
        raise SystemExit(
            f"v2 references already exist under {refs_dir}; use --force to overwrite, "
            "or point CNF_V2_V2_COND at a fresh directory."
        )

    refs = make_v2_conditions()
    precompute_targets(refs)
    print(f"v2 references and targets written under {V2_COND}", flush=True)


if __name__ == "__main__":
    main()
