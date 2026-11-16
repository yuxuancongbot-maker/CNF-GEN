"""Flow-matching motion prior and exact joint-space constraint projection.

The prior models a distribution over Legendre coefficients ``c`` conditioned on
the initial joint configuration ``q0``.  Constraints are enforced by projecting
``c`` onto the affine subspace ``A c = b`` in the *physical* coefficient space.
"""

from __future__ import annotations

import time as timer
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .basis import (
    coefficients_from_trajectory,
    make_basis,
    make_derivative_basis,
    trajectory_from_coefficients,
)
from .config import DEGREE, INIT_SCALE, JOINTS, PROJECT_EPS


def sinusoidal_embedding(time: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal time embedding with shape ``[B, dim]``."""
    half = dim // 2
    frequencies = torch.exp(
        -np.log(10000.0)
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / max(half - 1, 1)
    )
    angles = time[:, None] * frequencies[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class PandaFlowModel(torch.nn.Module):
    """MLP velocity field ``v(c, t, q0)`` for coefficient-space flow matching."""

    def __init__(self, joints: int = JOINTS, degree: int = DEGREE, hidden: int = 256):
        super().__init__()
        self.joints = joints
        self.degree = degree
        self.dim = joints * degree
        self.hidden = hidden
        self.state = torch.nn.Sequential(
            torch.nn.Linear(joints * 3, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        )
        self.time = torch.nn.Sequential(
            torch.nn.Linear(32, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.dim + hidden * 2, hidden * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden * 2, hidden * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden * 2, self.dim),
        )

    def forward(self, coefficients: torch.Tensor, time: torch.Tensor, q0: torch.Tensor):
        features = torch.cat([q0, torch.sin(q0), torch.cos(q0)], dim=-1)
        condition = torch.cat(
            [self.state(features), self.time(sinusoidal_embedding(time, 32))],
            dim=-1,
        )
        return self.net(torch.cat([coefficients, condition], dim=-1))


def fit_motion_prior(
    q: torch.Tensor,
    *,
    steps: int = 6000,
    batch: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str | torch.device = "cpu",
    log_every: int = 1000,
) -> dict:
    """Train the motion prior on joint trajectories ``q [N, T, J]``.

    Returns a checkpoint dictionary with model weights and normalisation
    statistics.  The training objective is

        E_{c, eps, t} || v_theta(c_t, t, q0) - (c - eps) ||^2,

    with ``c_t = (1 - t) eps + t c`` and ``eps ~ N(0, 0.5^2)``.
    """
    if q.ndim != 3:
        raise ValueError("q must have shape [N, T, J]")
    device = torch.device(device)
    torch.manual_seed(seed)
    frames = q.shape[1]
    joints = q.shape[2]
    # Keep the reported configuration exact for the pick-and-place data.
    degree = DEGREE if (joints == JOINTS and frames == 17) else degree_for(frames)
    basis, pinv = make_basis(frames, degree)
    coefficients = coefficients_from_trajectory(q, pinv).reshape(q.shape[0], -1)
    mean = coefficients.mean()
    std = coefficients.std().clamp_min(1e-6)
    normalized = (coefficients - mean) / std
    q0 = q[:, 0]

    data = normalized.to(device)
    q0 = q0.to(device)
    model = PandaFlowModel(joints=joints, degree=degree).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.05
    )
    generator = torch.Generator().manual_seed(seed + 1)
    model.train()
    start = timer.time()
    for step in range(steps):
        index = torch.randint(0, data.shape[0], (batch,), generator=generator)
        target = data[index.to(device)]
        noise = (torch.randn(target.shape, generator=generator) * INIT_SCALE).to(device)
        time = torch.rand(batch, generator=generator).to(device)
        interpolated = (1 - time[:, None]) * noise + time[:, None] * target
        velocity = target - noise
        prediction = model(interpolated, time, q0[index.to(device)])
        loss = F.mse_loss(prediction, velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % max(1, log_every) == 0:
            print(
                step + 1,
                "loss",
                round(float(loss), 6),
                "elapsed",
                round(timer.time() - start, 1),
                flush=True,
            )

    return {
        "model": model.state_dict(),
        "mean": mean.cpu(),
        "std": std.cpu(),
        "degree": degree,
        "joints": joints,
        "frames": frames,
        "hidden": model.hidden,
        "steps": steps,
    }


def degree_for(frames: int) -> int:
    """Reasonable default degree for non-standard sequence lengths."""
    return min(10, max(2, frames // 2))


def save_motion_prior(path: str | Path, checkpoint: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_motion_prior(path: str | Path, device: str | torch.device = "cpu"):
    """Load a motion prior; returns ``(model, mean, std, checkpoint)``."""
    device = torch.device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PandaFlowModel(
        joints=int(checkpoint.get("joints", JOINTS)),
        degree=int(checkpoint.get("degree", DEGREE)),
        hidden=int(checkpoint.get("hidden", 256)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["mean"].to(device), checkpoint["std"].to(device), checkpoint


def constraint_matrix(
    basis: torch.Tensor,
    velocity_basis: torch.Tensor,
    position_frames,
    velocity_frames=(),
    joints: int = JOINTS,
) -> torch.Tensor:
    """Build the constraint matrix ``A`` with shape ``[K, degree * joints]``.

    Each position row evaluates one basis column at one frame for one joint:
    ``A[k, d * J + j] = B_d(t_frame)``.  Velocity rows use ``dB_d/dt``.
    """
    degree = basis.shape[1]
    dim = degree * joints
    rows = []
    for frame in position_frames:
        for joint in range(joints):
            row = torch.zeros(dim, dtype=basis.dtype, device=basis.device)
            row[joint::joints] = basis[frame]
            rows.append(row)
    for frame in velocity_frames:
        for joint in range(joints):
            row = torch.zeros(dim, dtype=basis.dtype, device=velocity_basis.device)
            row[joint::joints] = velocity_basis[frame]
            rows.append(row)
    if not rows:
        raise ValueError("at least one constraint frame is required")
    return torch.stack(rows, dim=0)


def flatten_targets(position_targets: torch.Tensor, velocity_targets=None) -> torch.Tensor:
    """Flatten ``[B, K, J]`` targets in the same (frame, joint) row order as A."""
    pieces = [position_targets.reshape(position_targets.shape[0], -1)]
    if velocity_targets is not None and velocity_targets.numel():
        pieces.append(velocity_targets.reshape(velocity_targets.shape[0], -1))
    return torch.cat(pieces, dim=1)


def project_coefficients(
    coefficients: torch.Tensor,
    constraint: torch.Tensor,
    targets: torch.Tensor,
    eps: float = PROJECT_EPS,
) -> torch.Tensor:
    """Minimum-norm correction ``c <- c - A^T (A A^T + eps I)^{-1} (A c - b)``.

    ``coefficients``: ``[B, D]``, ``constraint``: ``[K, D]``,
    ``targets``: ``[B, K]``.  The returned coefficients satisfy
    ``A c' = b`` up to the linear-solve tolerance.
    """
    residual = coefficients @ constraint.T - targets
    normal = constraint @ constraint.T + eps * torch.eye(
        constraint.shape[0], dtype=constraint.dtype, device=constraint.device
    )
    dual = torch.linalg.solve(normal, residual.T).T
    return coefficients - dual @ constraint


def _apply_projection(
    coefficients: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    constraint: torch.Tensor,
    targets: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    physical = coefficients * std + mean
    physical = project_coefficients(physical, constraint, targets, eps=eps)
    return (physical - mean) / std


def sample_with_projection(
    model: PandaFlowModel,
    mean: torch.Tensor,
    std: torch.Tensor,
    basis: torch.Tensor,
    velocity_basis: torch.Tensor,
    q0: torch.Tensor,
    position_frames,
    position_targets: torch.Tensor,
    velocity_frames=(),
    velocity_targets: torch.Tensor | None = None,
    *,
    steps: int = 50,
    project_every_step: bool = True,
    seed: int = 0,
    eps: float = PROJECT_EPS,
    joints: int | None = None,
) -> torch.Tensor:
    """Sample coefficients with optional per-step hard projection.

    Returns physical coefficients with shape ``[B, degree * J]``.  The final
    projection is always applied, so ``project_every_step=False`` is the
    post-hoc baseline.
    """
    device = q0.device
    mean = mean.to(device)
    std = std.to(device)
    batch = q0.shape[0]
    joints = model.joints if joints is None else joints
    constraint = constraint_matrix(
        basis, velocity_basis, position_frames, velocity_frames, joints=joints
    ).to(device)
    targets = flatten_targets(position_targets, velocity_targets).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    coefficients = torch.randn(
        batch, model.dim, device=device, generator=generator
    ) * INIT_SCALE

    for step in range(steps):
        time = torch.full((batch,), step / steps, device=device)
        velocity = model(coefficients, time, q0)
        coefficients = coefficients + velocity / steps
        if project_every_step:
            coefficients = _apply_projection(
                coefficients, mean, std, constraint, targets, eps
            )
    coefficients = _apply_projection(coefficients, mean, std, constraint, targets, eps)
    return coefficients * std + mean


def sample_unconstrained(
    model: PandaFlowModel,
    mean: torch.Tensor,
    std: torch.Tensor,
    q0: torch.Tensor,
    *,
    steps: int = 50,
    seed: int = 0,
) -> torch.Tensor:
    """Euler integration without any projection."""
    device = q0.device
    mean = mean.to(device)
    std = std.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    coefficients = torch.randn(
        q0.shape[0], model.dim, device=device, generator=generator
    ) * INIT_SCALE
    for step in range(steps):
        time = torch.full((q0.shape[0],), step / steps, device=device)
        coefficients = coefficients + model(coefficients, time, q0) / steps
    return coefficients * std + mean


def coefficients_to_trajectory(
    coefficients: torch.Tensor, basis: torch.Tensor, joints: int = JOINTS
) -> torch.Tensor:
    """Evaluate physical coefficients ``[B, D]`` as trajectories ``[B, T, J]``."""
    return trajectory_from_coefficients(coefficients, basis.to(coefficients.device))


def trajectory_to_coefficients(q: torch.Tensor, degree: int = DEGREE) -> torch.Tensor:
    """Least-squares coefficients for ``q [B, T, J]``; returns ``[B, D]``."""
    basis, pinv = make_basis(q.shape[1], degree)
    return coefficients_from_trajectory(q, pinv).reshape(q.shape[0], -1)


def derivative_coefficients(
    coefficients: torch.Tensor, velocity_basis: torch.Tensor, joints: int = JOINTS
) -> torch.Tensor:
    """Joint velocities ``[B, T, J]`` in normalised time."""
    degree = velocity_basis.shape[1]
    c = coefficients.reshape(coefficients.shape[0], degree, joints)
    return torch.einsum("td,bdj->btj", velocity_basis.to(coefficients.device), c)


__all__ = [
    "PandaFlowModel",
    "sinusoidal_embedding",
    "fit_motion_prior",
    "save_motion_prior",
    "load_motion_prior",
    "constraint_matrix",
    "flatten_targets",
    "project_coefficients",
    "sample_with_projection",
    "sample_unconstrained",
    "coefficients_to_trajectory",
    "trajectory_to_coefficients",
    "derivative_coefficients",
    "make_basis",
    "make_derivative_basis",
]
