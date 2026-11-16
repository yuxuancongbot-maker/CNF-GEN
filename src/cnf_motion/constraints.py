"""Sparse-waypoint and task-space velocity constraints.

The dense experiment uses joint-space waypoints and an optional joint-space
velocity.  The sparse experiment uses six waypoints and a 3-D end-effector
velocity at frame 8.  The task-space rows are re-linearised three times because
the end-effector Jacobian depends on the current trajectory.
"""

from __future__ import annotations

import numpy as np
import torch

from .basis import trajectory_from_coefficients
from .config import (
    ANCHOR_FRAMES_SPARSE,
    CLEARANCE_FRAMES,
    INIT_SCALE,
    JOINTS,
    PROJECT_EPS,
    VELOCITY_FRAME,
)
from .motion_field import project_coefficients
from .scene import ee_jacobian, ee_position, hand_clearance, check_limits


def build_joint_rows(basis: torch.Tensor, frames, joints: int = JOINTS) -> torch.Tensor:
    """Rows mapping coefficients to joint values at ``frames``.

    Shape ``[len(frames) * joints, degree * joints]``.
    """
    degree = basis.shape[1]
    dim = degree * joints
    rows = []
    for frame in frames:
        for joint in range(joints):
            row = torch.zeros(dim, dtype=basis.dtype, device=basis.device)
            row[joint::joints] = basis[frame]
            rows.append(row)
    return torch.stack(rows, dim=0)


def project_rows(
    coefficients: torch.Tensor,
    rows: torch.Tensor,
    targets: torch.Tensor,
    eps: float = PROJECT_EPS,
) -> torch.Tensor:
    """Minimum-norm projection onto ``rows @ c = targets``."""
    return project_coefficients(coefficients, rows, targets, eps=eps)


def project_sparse(
    robot,
    coefficients: torch.Tensor,
    basis: torch.Tensor,
    velocity_basis: torch.Tensor,
    anchor_frames=ANCHOR_FRAMES_SPARSE,
    anchor_targets: torch.Tensor | None = None,
    task_velocity: torch.Tensor | None = None,
    iterations: int = 3,
    eps: float = PROJECT_EPS,
) -> torch.Tensor:
    """Project one sample onto sparse waypoints and an optional EE velocity.

    ``coefficients`` is ``[1, degree * 7]``; ``anchor_targets`` is
    ``[1, K, 7]``; ``task_velocity`` is ``[3]`` in metres per normalised time
    unit.  The Jacobian is evaluated at the current frame-8 configuration and
    the task-space rows are rebuilt for ``iterations`` passes.
    """
    if coefficients.shape[0] != 1:
        raise ValueError("project_sparse currently supports batch size 1")
    degree = basis.shape[1]
    if anchor_targets is None:
        raise ValueError("anchor_targets is required")
    joints = anchor_targets.shape[-1]
    dim = degree * joints
    anchor_rows = build_joint_rows(basis, anchor_frames, joints).to(coefficients.device)
    target = anchor_targets.reshape(1, -1).to(coefficients.device)
    current = coefficients
    if task_velocity is None:
        return project_rows(current, anchor_rows, target, eps=eps)

    velocity_basis = velocity_basis.to(coefficients.device)
    task_velocity = torch.as_tensor(
        task_velocity, dtype=coefficients.dtype, device=coefficients.device
    ).reshape(1, 3)
    for _ in range(iterations):
        q8 = trajectory_from_coefficients(current, basis)[0, VELOCITY_FRAME]
        jacobian = ee_jacobian(robot, q8.detach().cpu().numpy())
        rows = []
        for axis in range(3):
            row = torch.zeros(dim, dtype=coefficients.dtype, device=coefficients.device)
            for joint in range(joints):
                row[joint::joints] += (
                    float(jacobian[axis, joint]) * velocity_basis[VELOCITY_FRAME]
                )
            rows.append(row)
        velocity_rows = torch.stack(rows, dim=0)
        combined = torch.cat([anchor_rows, velocity_rows], dim=0)
        combined_target = torch.cat([target, task_velocity], dim=1)
        current = project_rows(current, combined, combined_target, eps=eps)
    return current


def sample_sparse(
    robot,
    model,
    mean: torch.Tensor,
    std: torch.Tensor,
    basis: torch.Tensor,
    velocity_basis: torch.Tensor,
    q0: torch.Tensor,
    anchor_frames=ANCHOR_FRAMES_SPARSE,
    anchor_targets: torch.Tensor | None = None,
    task_velocity: torch.Tensor | None = None,
    *,
    steps: int = 50,
    project_every_step: bool = True,
    seed: int = 0,
    iterations: int = 3,
    eps: float = PROJECT_EPS,
) -> torch.Tensor:
    """Sample the hard prior with sparse/task-space projection.

    Returns physical coefficients ``[1, degree * 7]``.  ``project_every_step``
    false is the post-hoc baseline; a final projection is always applied.
    """
    device = q0.device
    mean = mean.to(device)
    std = std.to(device)
    basis = basis.to(device)
    velocity_basis = velocity_basis.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    coefficients = torch.randn(
        q0.shape[0], model.dim, device=device, generator=generator
    ) * INIT_SCALE

    for step in range(steps):
        time = torch.full((q0.shape[0],), step / steps, device=device)
        coefficients = coefficients + model(coefficients, time, q0) / steps
        if project_every_step:
            physical = coefficients * std + mean
            physical = project_sparse(
                robot,
                physical,
                basis,
                velocity_basis,
                anchor_frames=anchor_frames,
                anchor_targets=anchor_targets,
                task_velocity=task_velocity,
                iterations=iterations,
                eps=eps,
            )
            coefficients = (physical - mean) / std
    physical = coefficients * std + mean
    physical = project_sparse(
        robot,
        physical,
        basis,
        velocity_basis,
        anchor_frames=anchor_frames,
        anchor_targets=anchor_targets,
        task_velocity=task_velocity,
        iterations=iterations,
        eps=eps,
    )
    return physical


def task_space_velocity(robot, q, frame: int = VELOCITY_FRAME, frames: int = 17) -> np.ndarray:
    """Central-difference end-effector velocity in normalised time units.

    ``v = (p[q_{frame+1}] - p[q_{frame-1}]) / 2 * (frames - 1)``.
    """
    positions = np.array([ee_position(robot, q[k]) for k in range(frames)])
    return (positions[frame + 1] - positions[frame - 1]) / 2 * (frames - 1)


def validate_trajectory(
    robot,
    q,
    *,
    max_joint_step: float = 0.8,
    clearance_tolerance: float = -0.001,
) -> dict:
    """Check joint limits, per-frame jumps, and gripper/table clearance."""
    joint_ok = check_limits(robot, q)
    max_step = float(np.abs(np.diff(q, axis=0)).max())
    clearance = hand_clearance(robot, q)
    return {
        "joint_limits": bool(joint_ok),
        "max_joint_step": max_step,
        "hand_clearance": clearance,
        "valid": bool(
            joint_ok and max_step < max_joint_step and clearance > clearance_tolerance
        ),
    }


__all__ = [
    "build_joint_rows",
    "project_rows",
    "project_sparse",
    "sample_sparse",
    "task_space_velocity",
    "validate_trajectory",
    "CLEARANCE_FRAMES",
]
