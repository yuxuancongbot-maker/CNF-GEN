#!/usr/bin/env python3
"""Evaluate latent-level hard projection on the 6-DOF action world model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np
import torch

from cnf_motion.basis import make_basis, make_derivative_basis
from cnf_motion.config import ANCHOR_FRAMES_SPARSE, FPS, SIZE
from cnf_motion.constraints import sample_sparse, task_space_velocity
from cnf_motion.dataset import load_metadata
from cnf_motion.metrics import mean_metrics, psnr, trajectory_metrics
from cnf_motion.motion_field import (
    coefficients_to_trajectory,
    load_motion_prior,
    sample_unconstrained,
)
from cnf_motion.scene import (
    build_scene,
    disconnect_scene,
    ee_position,
    enforce_clearance,
    render_trajectory,
)
from cnf_motion.soft_field import load_soft_prior, sample_soft
from cnf_motion.video_wm import load_video_wm

METHODS = ["gt", "unconstrained", "soft", "posthoc", "every_step"]
WIDTH = 128


def load_video_128(path: Path) -> np.ndarray:
    frames = np.stack(iio.mimread(str(path)))
    return np.stack(
        [
            cv2.resize(frame, (WIDTH, WIDTH), interpolation=cv2.INTER_AREA)
            for frame in frames
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True)
    parser.add_argument("--motion-prior", required=True)
    parser.add_argument("--soft-prior", required=True)
    parser.add_argument("--video-wm", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--derivative-mode",
        default="discrete_autograd",
        choices=["discrete_autograd", "continuous", "finite_difference"],
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    (output / "videos").mkdir(parents=True, exist_ok=True)

    video_model, _ = load_video_wm(args.video_wm, device=device)
    hard_model, hard_mean, hard_std, _ = load_motion_prior(args.motion_prior, device="cpu")
    soft_model, soft_mean, soft_std, _ = load_soft_prior(args.soft_prior, device="cpu")
    basis, _ = make_basis()
    _, velocity_basis, _ = make_derivative_basis(mode=args.derivative_mode)
    metadata = load_metadata(args.eval)
    robot, cube, table_id = build_scene()

    rows = []
    try:
        for index, meta in enumerate(metadata[: args.samples]):
            q_gt = np.load(Path(args.eval) / "q" / f"{index:04d}.npy")
            gt_video = load_video_128(Path(args.eval) / "train" / f"{index:04d}.mp4")
            qt = torch.tensor(q_gt, dtype=torch.float32)[None]
            q0 = qt[:, 0]
            anchors = qt[:, list(ANCHOR_FRAMES_SPARSE)]
            ee_gt = np.array([ee_position(robot, q_gt[frame]) for frame in range(17)])
            task_velocity = task_space_velocity(robot, q_gt)

            for method in METHODS:
                if method == "gt":
                    q = q_gt.copy()
                elif method == "unconstrained":
                    with torch.no_grad():
                        coefficients = sample_unconstrained(
                            hard_model,
                            hard_mean,
                            hard_std,
                            q0,
                            steps=args.steps,
                            seed=4000 + index,
                        )
                    q = coefficients_to_trajectory(coefficients, basis)[0].numpy()
                    q = enforce_clearance(robot, q, meta["yaw"])
                elif method == "soft":
                    with torch.no_grad():
                        coefficients = sample_soft(
                            soft_model,
                            soft_mean,
                            soft_std,
                            q0,
                            anchors,
                            steps=args.steps,
                            seed=5000 + index,
                        )
                    q = coefficients_to_trajectory(coefficients, basis)[0].numpy()
                    q = enforce_clearance(robot, q, meta["yaw"])
                else:
                    with torch.no_grad():
                        coefficients = sample_sparse(
                            robot,
                            hard_model,
                            hard_mean,
                            hard_std,
                            basis,
                            velocity_basis,
                            q0,
                            anchor_frames=ANCHOR_FRAMES_SPARSE,
                            anchor_targets=anchors,
                            task_velocity=task_velocity,
                            steps=args.steps,
                            project_every_step=(method == "every_step"),
                            seed=6000 + index,
                        )
                    q = coefficients_to_trajectory(coefficients, basis)[0].numpy()
                    q = enforce_clearance(robot, q, meta["yaw"])

                _, control, cube_positions = render_trajectory(
                    robot, cube, table_id, q, np.array(meta["cube_xy"]), width=SIZE
                )
                control = np.stack(
                    [
                        cv2.resize(frame, (WIDTH, WIDTH), interpolation=cv2.INTER_NEAREST)
                        for frame in control
                    ]
                )
                control_t = (
                    torch.tensor(control, dtype=torch.float32).permute(3, 0, 1, 2)[None]
                    / 255.0
                ).to(device)
                q_t = torch.tensor(q, dtype=torch.float32)[None].to(device)
                first = (
                    torch.tensor(gt_video[0], dtype=torch.float32).permute(2, 0, 1)[None]
                    / 255.0
                ).to(device)
                with torch.no_grad():
                    prediction = video_model(first, control_t, q_t)[0]
                prediction = (
                    (prediction.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
                    .round()
                    .astype(np.uint8)
                )

                ee = np.array([ee_position(robot, q[frame]) for frame in range(17)])
                row = {
                    "sample": index,
                    "method": method,
                    "psnr": psnr(prediction, gt_video, data_range=255.0),
                    **trajectory_metrics(
                        q,
                        q_gt,
                        ee,
                        ee_gt,
                        anchor_frames=ANCHOR_FRAMES_SPARSE,
                        task_velocity=task_velocity,
                        cube_positions=cube_positions,
                        target_xy=meta["target_xy"],
                    ),
                }
                rows.append(row)
                print(row, flush=True)
                iio.mimsave(
                    str(output / "videos" / f"{index}_{method}.mp4"), prediction, fps=FPS
                )

        (output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
        print("\n=== means ===")
        metrics = [
            "psnr",
            "anchor_error",
            "joint_rmse",
            "ee_rmse",
            "ee_velocity_error",
            "jerk",
        ]
        for method in METHODS:
            summary = mean_metrics(rows, method, metrics)
            print(method, summary)
    finally:
        disconnect_scene()


if __name__ == "__main__":
    main()
