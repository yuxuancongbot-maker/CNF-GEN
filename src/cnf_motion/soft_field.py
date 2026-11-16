"""Soft-constraint baseline: anchors are inputs plus a differentiable loss.

This baseline sees the sparse anchor values and is trained with
``flow_loss + soft_weight * anchor_mse``.  It never performs a hard projection,
so its anchor residual is non-zero by construction.
"""

from __future__ import annotations

import time as timer
from pathlib import Path

import torch
import torch.nn.functional as F

from .basis import coefficients_from_trajectory, make_basis
from .config import (
    ANCHOR_FRAMES_SPARSE,
    DEGREE,
    INIT_SCALE,
    JOINTS,
    PROJECT_EPS,
)
from .constraints import project_sparse
from .motion_field import (
    project_coefficients,
    sinusoidal_embedding,
)


class PandaSoftFlowModel(torch.nn.Module):
    """Flow model conditioned on ``q0`` and the sparse anchor values."""

    def __init__(
        self,
        joints: int = JOINTS,
        degree: int = DEGREE,
        anchors: int = len(ANCHOR_FRAMES_SPARSE),
        hidden: int = 256,
    ):
        super().__init__()
        self.joints = joints
        self.degree = degree
        self.anchors = anchors
        self.dim = joints * degree
        self.hidden = hidden
        self.state = torch.nn.Sequential(
            torch.nn.Linear(joints * 3, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        )
        self.anchors = torch.nn.Sequential(
            torch.nn.Linear(anchors * joints, hidden),
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
            torch.nn.Linear(self.dim + hidden * 3, hidden * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden * 2, hidden * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden * 2, self.dim),
        )

    def forward(
        self,
        coefficients: torch.Tensor,
        time: torch.Tensor,
        q0: torch.Tensor,
        anchors: torch.Tensor,
    ):
        q0_features = torch.cat([q0, torch.sin(q0), torch.cos(q0)], dim=-1)
        condition = torch.cat(
            [
                self.state(q0_features),
                self.anchors(anchors.flatten(1)),
                self.time(sinusoidal_embedding(time, 32)),
            ],
            dim=-1,
        )
        return self.net(torch.cat([coefficients, condition], dim=-1))


def fit_soft_prior(
    q: torch.Tensor,
    *,
    anchor_frames=ANCHOR_FRAMES_SPARSE,
    soft_weight: float = 20.0,
    steps: int = 6000,
    batch: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 1,
    device: str | torch.device = "cpu",
    log_every: int = 1000,
) -> dict:
    """Train the soft baseline.

    ``q`` is ``[N, T, J]``.  The returned checkpoint has the same normalisation
    fields as the hard prior, so it can be evaluated with the same sampling
    loop.
    """
    if q.ndim != 3:
        raise ValueError("q must have shape [N, T, J]")
    device = torch.device(device)
    torch.manual_seed(seed)
    frames = q.shape[1]
    joints = q.shape[2]
    degree = DEGREE if (joints == JOINTS and frames == 17) else min(10, max(2, frames // 2))
    basis, pinv = make_basis(frames, degree)
    coefficients = coefficients_from_trajectory(q, pinv).reshape(q.shape[0], -1)
    mean = coefficients.mean()
    std = coefficients.std().clamp_min(1e-6)
    normalized = (coefficients - mean) / std
    q0 = q[:, 0]
    anchors = q[:, list(anchor_frames)].flatten(1)
    anchor_targets = q[:, list(anchor_frames)]

    data = normalized.to(device)
    q0 = q0.to(device)
    anchors = anchors.to(device)
    anchor_targets = anchor_targets.to(device)
    model = PandaSoftFlowModel(
        joints=joints,
        degree=degree,
        anchors=len(anchor_frames),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.05
    )
    anchor_basis = basis[list(anchor_frames)].to(device)
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
        prediction = model(interpolated, time, q0[index.to(device)], anchors[index.to(device)])
        flow_loss = F.mse_loss(prediction, velocity)

        clean = interpolated + (1 - time[:, None]) * prediction
        physical = (clean * std + mean).reshape(-1, degree, joints)
        q_anchor = torch.einsum("kd,bdj->bkj", anchor_basis, physical)
        soft_loss = F.mse_loss(q_anchor, anchor_targets[index.to(device)])
        loss = flow_loss + soft_weight * soft_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % max(1, log_every) == 0:
            print(
                step + 1,
                "flow",
                round(float(flow_loss), 6),
                "soft",
                round(float(soft_loss), 6),
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
        "anchor_frames": list(anchor_frames),
        "soft_weight": soft_weight,
        "steps": steps,
    }


def save_soft_prior(path: str | Path, checkpoint: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_soft_prior(path: str | Path, device: str | torch.device = "cpu"):
    """Load a soft prior; returns ``(model, mean, std, checkpoint)``."""
    device = torch.device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    anchor_frames = checkpoint.get("anchor_frames", list(ANCHOR_FRAMES_SPARSE))
    model = PandaSoftFlowModel(
        joints=int(checkpoint.get("joints", JOINTS)),
        degree=int(checkpoint.get("degree", DEGREE)),
        anchors=len(anchor_frames),
        hidden=int(checkpoint.get("hidden", 256)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["mean"].to(device), checkpoint["std"].to(device), checkpoint


def sample_soft(
    model: PandaSoftFlowModel,
    mean: torch.Tensor,
    std: torch.Tensor,
    q0: torch.Tensor,
    anchors: torch.Tensor,
    *,
    steps: int = 50,
    seed: int = 0,
) -> torch.Tensor:
    """Euler sampling without projection; returns physical coefficients."""
    device = q0.device
    mean = mean.to(device)
    std = std.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    coefficients = torch.randn(
        q0.shape[0], model.dim, device=device, generator=generator
    ) * INIT_SCALE
    for step in range(steps):
        time = torch.full((q0.shape[0],), step / steps, device=device)
        velocity = model(coefficients, time, q0, anchors)
        coefficients = coefficients + velocity / steps
    return coefficients * std + mean


def sample_soft_projected(
    robot,
    model: PandaSoftFlowModel,
    mean: torch.Tensor,
    std: torch.Tensor,
    q0: torch.Tensor,
    anchors: torch.Tensor,
    basis: torch.Tensor,
    velocity_basis: torch.Tensor,
    *,
    anchor_frames=ANCHOR_FRAMES_SPARSE,
    anchor_targets: torch.Tensor | None = None,
    task_velocity: torch.Tensor | None = None,
    steps: int = 50,
    project_every_step: bool = True,
    seed: int = 0,
    iterations: int = 3,
    eps: float = PROJECT_EPS,
) -> torch.Tensor:
    """Soft prior + hard projection (the ``hybrid`` baseline).

    The underlying soft prior is identical to :func:`sample_soft`; the only
    difference is the optional per-step projection onto ``A c = b``.  With
    ``project_every_step=False`` this is exactly the soft baseline plus a final
    projection, which isolates the effect of projection frequency.
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
        velocity = model(coefficients, time, q0, anchors)
        coefficients = coefficients + velocity / steps
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
    return project_sparse(
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


def sample_soft_with_constraints(
    model: PandaSoftFlowModel,
    mean: torch.Tensor,
    std: torch.Tensor,
    q0: torch.Tensor,
    anchors: torch.Tensor,
    constraint: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int = 50,
    project_every_step: bool = True,
    seed: int = 0,
    eps: float = PROJECT_EPS,
) -> torch.Tensor:
    """Soft prior with a generic linear hard projection (no robot model).

    ``constraint`` is ``[K, D]`` and ``targets`` is ``[B, K]``.  This is the
    real-data counterpart of :func:`sample_soft_projected`, which uses the
    PyBullet task-space Jacobian.
    """
    device = q0.device
    mean = mean.to(device)
    std = std.to(device)
    constraint = constraint.to(device)
    targets = targets.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    coefficients = torch.randn(
        q0.shape[0], model.dim, device=device, generator=generator
    ) * INIT_SCALE
    for step in range(steps):
        time = torch.full((q0.shape[0],), step / steps, device=device)
        velocity = model(coefficients, time, q0, anchors)
        coefficients = coefficients + velocity / steps
        if project_every_step:
            physical = coefficients * std + mean
            physical = project_coefficients(physical, constraint, targets, eps=eps)
            coefficients = (physical - mean) / std
    physical = coefficients * std + mean
    return project_coefficients(physical, constraint, targets, eps=eps)


__all__ = [
    "PandaSoftFlowModel",
    "fit_soft_prior",
    "save_soft_prior",
    "load_soft_prior",
    "sample_soft",
    "sample_soft_projected",
    "sample_soft_with_constraints",
]
