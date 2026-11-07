#!/usr/bin/env python3
"""Generate CNF-controlled pick-and-place trajectories and control videos.

Dense mode reproduces the 7-anchor experiment; sparse mode reproduces the
6-anchor + task-space velocity experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch

from cnf_motion.basis import make_basis, make_derivative_basis
from cnf_motion.config import (
    ANCHOR_FRAMES_DENSE,
    ANCHOR_FRAMES_SPARSE,
    FPS,
    SIZE,
    VELOCITY_FRAME,
)
from cnf_motion.constraints import sample_sparse, task_space_velocity, validate_trajectory
from cnf_motion.dataset import load_metadata
from cnf_motion.metrics import trajectory_metrics
from cnf_motion.motion_field import (
    coefficients_to_trajectory,
    load_motion_prior,
    sample_with_projection,
)
from cnf_motion.scene import (
    build_scene,
    disconnect_scene,
    ee_position,
    enforce_clearance,
    render_trajectory,
    table_clearance,
)


def parse_methods(value: str | None, mode: str) -> list[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    if mode == "dense":
        return ["waypoint", "velocity", "posthoc"]
    return ["sparse_waypoint", "sparse_velocity", "sparse_posthoc"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True, help="held-out dataset directory")
    parser.add_argument("--motion-prior", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["dense", "sparse"], default="sparse")
    parser.add_argument("--methods", default=None)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--derivative-mode",
        default="discrete_autograd",
        choices=["discrete_autograd", "continuous", "finite_difference"],
    )
    args = parser.parse_args()

    methods = parse_methods(args.methods, args.mode)
    output = Path(args.output)
    for subdir in ("rgb", "control", "q"):
        (output / subdir).mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.eval)
    basis, pinv = make_basis()
    _, velocity_basis, _ = make_derivative_basis(mode=args.derivative_mode)
    model, mean, std, _ = load_motion_prior(args.motion_prior, device=args.device)
    robot, cube, table_id = build_scene()
    rows = []
    try:
        for index, meta in enumerate(metadata[: args.samples]):
            q_gt = np.load(Path(args.eval) / "q" / f"{index:04d}.npy")
            qt = torch.tensor(q_gt, dtype=torch.float32)[None]
            q0 = qt[:, 0]
            coefficients_gt = torch.einsum("dt,btj->bdj", pinv, qt)
            gt_joint_velocity = torch.einsum("td,bdj->btj", velocity_basis, coefficients_gt)
            ee_gt = np.array([ee_position(robot, q_gt[frame]) for frame in range(17)])

            if args.mode == "dense":
                anchor_frames = ANCHOR_FRAMES_DENSE
                position_targets = qt[:, anchor_frames]
                task_velocity = None
            else:
                anchor_frames = ANCHOR_FRAMES_SPARSE
                position_targets = qt[:, anchor_frames]
                task_velocity = task_space_velocity(robot, q_gt)

            for method in methods:
                use_velocity = method in {"velocity", "sparse_velocity"}
                posthoc = method in {"posthoc", "sparse_posthoc"}
                for attempt in range(args.max_attempts):
                    seed = (1000 if args.mode == "dense" else 2000) + index + attempt * 100
                    with torch.no_grad():
                        if args.mode == "dense":
                            coefficients = sample_with_projection(
                                model,
                                mean,
                                std,
                                basis,
                                velocity_basis,
                                q0,
                                position_frames=anchor_frames,
                                position_targets=position_targets,
                                velocity_frames=(VELOCITY_FRAME,) if use_velocity else (),
                                velocity_targets=(
                                    gt_joint_velocity[:, [VELOCITY_FRAME]]
                                    if use_velocity
                                    else None
                                ),
                                steps=args.steps,
                                project_every_step=not posthoc,
                                seed=seed,
                            )
                        else:
                            coefficients = sample_sparse(
                                robot,
                                model,
                                mean,
                                std,
                                basis,
                                velocity_basis,
                                q0,
                                anchor_frames=anchor_frames,
                                anchor_targets=position_targets,
                                task_velocity=task_velocity if use_velocity else None,
                                steps=args.steps,
                                project_every_step=not posthoc,
                                seed=seed,
                            )
                    q = coefficients_to_trajectory(coefficients, basis)[0].numpy()
                    q = enforce_clearance(robot, q, meta["yaw"])
                    check = validate_trajectory(robot, q)
                    if check["valid"]:
                        break
                else:
                    raise RuntimeError(
                        f"failed sample {index} method {method} after {args.max_attempts} attempts"
                    )

                rgb, control, cube_positions = render_trajectory(
                    robot, cube, table_id, q, np.array(meta["cube_xy"]), width=SIZE
                )
                ee = np.array([ee_position(robot, q[frame]) for frame in range(17)])
                row = {
                    "sample": index,
                    "method": method,
                    "attempts": attempt,
                    **trajectory_metrics(
                        q,
                        q_gt,
                        ee,
                        ee_gt,
                        anchor_frames=anchor_frames,
                        task_velocity=task_velocity if args.mode == "sparse" else None,
                        cube_positions=cube_positions,
                        target_xy=meta["target_xy"],
                    ),
                    "table_clearance": table_clearance(cube_positions),
                }
                if args.mode == "dense":
                    row["velocity_error"] = float(
                        np.linalg.norm(
                            np.diff(q, axis=0)[VELOCITY_FRAME]
                            - np.diff(q_gt, axis=0)[VELOCITY_FRAME]
                        )
                    )
                else:
                    row["joint_velocity_at_8_error"] = float(
                        np.linalg.norm(
                            np.diff(q, axis=0)[VELOCITY_FRAME]
                            - np.diff(q_gt, axis=0)[VELOCITY_FRAME]
                        )
                    )
                rows.append(row)
                iio.mimsave(str(output / "rgb" / f"{index}_{method}.mp4"), rgb, fps=FPS)
                iio.mimsave(
                    str(output / "control" / f"{index}_{method}.mp4"), control, fps=FPS
                )
                np.save(str(output / "q" / f"{index}_{method}.npy"), q)
                print(row, flush=True)

        (output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
        print("saved", output)
    finally:
        disconnect_scene()


if __name__ == "__main__":
    main()
