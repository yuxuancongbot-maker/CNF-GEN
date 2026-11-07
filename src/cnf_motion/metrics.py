"""Trajectory-level and pixel-level metrics.

The 6-DOF pixel evaluator uses a colour threshold plus connected components to
locate the orange cube.  The 1.3B two-link evaluator uses a different colour
detector for the orange link endpoint; see ``docs/04_wan_1_3b.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np

from .config import CUBE_HALF, TABLE_TOP, VELOCITY_FRAME


def psnr(prediction, target, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB."""
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mse = float(((prediction - target) ** 2).mean())
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / mse))


def load_video(path: str | Path) -> np.ndarray:
    return np.stack(iio.mimread(str(path)))


def detect_cube(frame: np.ndarray):
    """Locate the orange cube centroid in an RGB uint8 frame.

    Thresholds: R>0.55, 0.10<G<0.60, B<0.32, R>1.25G.  Connected components
    with area in ``[20, 1500]`` are candidates; the largest is returned.
    """
    image = frame.astype(np.float32) / 255.0
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    mask = (
        (red > 0.55)
        & (green > 0.10)
        & (green < 0.60)
        & (blue < 0.32)
        & (red > green * 1.25)
    ).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    best_area = 0
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        if 20 <= area <= 1500 and area > best_area:
            best = centroids[index]
            best_area = area
    return None if best is None else best.astype(np.float32)


def frame_velocity(positions: np.ndarray, frame: int = VELOCITY_FRAME, frames: int = 17):
    """Central difference in normalised time units."""
    return (positions[frame + 1] - positions[frame - 1]) / 2 * (frames - 1)


def trajectory_metrics(
    q: np.ndarray,
    q_gt: np.ndarray,
    ee: np.ndarray,
    ee_gt: np.ndarray,
    *,
    anchor_frames,
    task_velocity: np.ndarray | None = None,
    cube_positions: np.ndarray | None = None,
    target_xy=None,
    frames: int = 17,
) -> dict:
    """Joint-space and task-space metrics for one trajectory."""
    joint_rmse = float(np.sqrt(((q - q_gt) ** 2).mean()))
    ee_rmse = float(np.sqrt((((ee - ee_gt) ** 2).sum(-1)).mean()))
    jerk = float(np.linalg.norm(np.diff(q, n=3, axis=0), axis=-1).mean())
    row = {
        "anchor_error": float(np.abs(q[list(anchor_frames)] - q_gt[list(anchor_frames)]).max()),
        "joint_rmse": joint_rmse,
        "ee_rmse": ee_rmse,
        "jerk": jerk,
    }
    if task_velocity is not None:
        row["ee_velocity_error"] = float(
            np.linalg.norm(frame_velocity(ee, frames=frames) - np.asarray(task_velocity))
        )
    if cube_positions is not None:
        row["table_clearance"] = float(
            (cube_positions[:, 2] - CUBE_HALF - TABLE_TOP).min()
        )
        if target_xy is not None:
            row["cube_target_error"] = float(
                np.linalg.norm(cube_positions[-1, :2] - np.asarray(target_xy))
            )
    return row


def pixel_metrics(prediction: np.ndarray, target: np.ndarray, control: np.ndarray | None = None) -> dict:
    """PSNR + colour-tracked cube metrics for one video pair."""
    frame_count = min(len(prediction), len(target))
    prediction = prediction[:frame_count]
    target = target[:frame_count]
    control = control[:frame_count] if control is not None else None
    points, gt_points, control_points = [], [], []
    for frame in range(frame_count):
        points.append(detect_cube(prediction[frame]))
        gt_points.append(detect_cube(target[frame]))
        control_points.append(detect_cube(control[frame]) if control is not None else None)

    track = [
        np.linalg.norm(point - gt_point)
        for point, gt_point in zip(points, gt_points)
        if point is not None and gt_point is not None
    ]
    control_track = [
        np.linalg.norm(point - control_point)
        for point, control_point in zip(points, control_points)
        if point is not None and control_point is not None
    ]
    valid = np.array([point for point in points if point is not None])
    jerk = (
        float(np.linalg.norm(np.diff(valid, n=3, axis=0), axis=-1).mean())
        if len(valid) >= 4
        else float("nan")
    )
    final_error = (
        float(np.linalg.norm(points[-1] - gt_points[-1]))
        if points[-1] is not None and gt_points[-1] is not None
        else float("nan")
    )
    return {
        "psnr": psnr(prediction, target, data_range=255.0),
        "cube_track_epe": float(np.mean(track)) if track else float("nan"),
        "control_epe": float(np.mean(control_track)) if control_track else float("nan"),
        "final_error": final_error,
        "cube_jerk": jerk,
        "detected_frames": int(len(valid)),
    }


def paired_delta(
    rows: list[dict],
    method: str,
    baseline: str,
    metric: str,
    *,
    higher_is_better: bool = False,
) -> dict:
    """Paired mean delta ``method - baseline`` with standard error."""
    by = {(row["sample"], row["method"]): row for row in rows}
    samples = sorted({row["sample"] for row in rows})
    deltas = []
    for sample in samples:
        current = by.get((sample, method))
        reference = by.get((sample, baseline))
        if current is None or reference is None:
            continue
        value = float(current[metric])
        ref = float(reference[metric])
        if np.isfinite(value) and np.isfinite(ref):
            deltas.append(value - ref)
    if not deltas:
        return {"mean": float("nan"), "se": float("nan"), "better": 0, "n": 0}
    deltas = np.array(deltas, dtype=float)
    better = int(
        np.sum(deltas > 0) if higher_is_better else np.sum(deltas < 0)
    )
    return {
        "mean": float(deltas.mean()),
        "se": float(deltas.std(ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else float("nan"),
        "better": better,
        "n": int(len(deltas)),
    }


def mean_metrics(rows: list[dict], method: str, metrics: list[str]) -> dict:
    selected = [row for row in rows if row["method"] == method]
    result = {"n": len(selected)}
    for metric in metrics:
        values = np.array([row[metric] for row in selected], dtype=float)
        result[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    return result


def write_metrics(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


__all__ = [
    "psnr",
    "load_video",
    "detect_cube",
    "frame_velocity",
    "trajectory_metrics",
    "pixel_metrics",
    "paired_delta",
    "mean_metrics",
    "write_metrics",
]
