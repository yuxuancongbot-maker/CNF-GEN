from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

TensorField = Callable[[torch.Tensor], torch.Tensor]
HardProjector = Callable[..., tuple[torch.Tensor, dict[str, Any]]]


def _points3(name: str, value: torch.Tensor) -> torch.Tensor:
    points = torch.as_tensor(value, dtype=torch.float64)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {tuple(points.shape)}")
    if points.shape[0] == 0 or not torch.isfinite(points).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return points


def gaussian_rbf_matrix(
    points: torch.Tensor, centers: torch.Tensor, length_scale: float
) -> torch.Tensor:
    points = _points3("points", points)
    centers = _points3("centers", centers).to(device=points.device)
    if not torch.isfinite(torch.tensor(length_scale)) or length_scale <= 0:
        raise ValueError("length_scale must be positive and finite")
    squared_distance = (points[:, None, :] - centers[None, :, :]).square().sum(dim=-1)
    return torch.exp(-0.5 * squared_distance / (length_scale * length_scale))


@dataclass(frozen=True)
class CorrectedDisplacementField:
    prior: TensorField
    centers: torch.Tensor
    coefficients: torch.Tensor
    length_scale: float

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        points = _points3("points", points).to(device=self.centers.device)
        prior_values = torch.as_tensor(
            self.prior(points), dtype=torch.float64, device=points.device
        )
        if prior_values.shape != points.shape or not torch.isfinite(prior_values).all():
            raise ValueError("prior must return finite displacements with shape [N, 3]")
        basis_values = gaussian_rbf_matrix(points, self.centers, self.length_scale)
        return prior_values + basis_values @ self.coefficients.T


def correct_with_fixed_gaussian_rbf(
    prior: TensorField,
    handle_points: torch.Tensor,
    target_displacements: torch.Tensor,
    *,
    length_scale: float,
    projector: HardProjector,
    max_handle_error: float = 1e-5,
) -> tuple[CorrectedDisplacementField, dict[str, Any]]:
    """Correct a displacement field at fixed handles using the existing hard projector.

    This task adapter constructs a finite RBF coefficient system. It does not
    implement a second projection solver.
    """
    handles = _points3("handle_points", handle_points)
    targets = _points3("target_displacements", target_displacements).to(handles.device)
    if targets.shape != handles.shape:
        raise ValueError("target_displacements must have the same shape as handle_points")
    if not torch.isfinite(torch.tensor(max_handle_error)) or max_handle_error <= 0:
        raise ValueError("max_handle_error must be positive and finite")
    if not torch.isfinite(torch.tensor(length_scale)) or length_scale <= 0:
        raise ValueError("length_scale must be positive and finite")

    pair_distances = torch.cdist(handles, handles)
    pair_distances.fill_diagonal_(float("inf"))
    if pair_distances.min().item() <= 1e-10:
        raise ValueError("duplicate or numerically coincident handle positions")

    constraint_matrix = gaussian_rbf_matrix(handles, handles, length_scale)
    prior_values = torch.as_tensor(prior(handles), dtype=torch.float64, device=handles.device)
    if prior_values.shape != handles.shape or not torch.isfinite(prior_values).all():
        raise ValueError("prior must return finite displacements with shape [N, 3]")

    residual = targets - prior_values
    initial_coefficients = torch.zeros(
        (3, handles.shape[0]), dtype=torch.float64, device=handles.device
    )
    coefficients, solver_info = projector(constraint_matrix, initial_coefficients, residual.T)
    field = CorrectedDisplacementField(
        prior=prior,
        centers=handles,
        coefficients=coefficients,
        length_scale=length_scale,
    )
    corrected_handles = field(handles)
    handle_errors = torch.linalg.vector_norm(corrected_handles - targets, dim=-1)
    max_error = handle_errors.max().item()
    if not torch.isfinite(handle_errors).all() or max_error > max_handle_error:
        raise RuntimeError(f"corrected handle error {max_error:.3e} exceeds {max_handle_error:.3e}")

    diagnostics: dict[str, Any] = {
        "solver": solver_info,
        "constraint_matrix_shape": list(constraint_matrix.shape),
        "handle_count": int(handles.shape[0]),
        "raw_handle_error_max": torch.linalg.vector_norm(prior_values - targets, dim=-1)
        .max()
        .item(),
        "corrected_handle_error_max": max_error,
        "corrected_handle_error_mean": handle_errors.mean().item(),
        "max_handle_error_tolerance": max_handle_error,
        "length_scale": length_scale,
        "dtype": str(handles.dtype),
    }
    return field, diagnostics


def fit_soft_gaussian_rbf_correction(
    prior: TensorField,
    handle_points: torch.Tensor,
    target_displacements: torch.Tensor,
    *,
    length_scale: float,
    ridge_lambda: float,
) -> tuple[CorrectedDisplacementField, dict[str, Any]]:
    """Fit the matched soft baseline with coefficient L2 regularization."""
    handles = _points3("handle_points", handle_points)
    targets = _points3("target_displacements", target_displacements).to(handles.device)
    if targets.shape != handles.shape:
        raise ValueError("target_displacements must have the same shape as handle_points")
    if not torch.isfinite(torch.tensor(ridge_lambda)) or ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive and finite")
    if not torch.isfinite(torch.tensor(length_scale)) or length_scale <= 0:
        raise ValueError("length_scale must be positive and finite")

    pair_distances = torch.cdist(handles, handles)
    pair_distances.fill_diagonal_(float("inf"))
    if pair_distances.min().item() <= 1e-10:
        raise ValueError("duplicate or numerically coincident handle positions")

    kernel = gaussian_rbf_matrix(handles, handles, length_scale)
    prior_values = torch.as_tensor(prior(handles), dtype=torch.float64, device=handles.device)
    if prior_values.shape != handles.shape or not torch.isfinite(prior_values).all():
        raise ValueError("prior must return finite displacements with shape [N, 3]")

    residual = targets - prior_values
    normal_matrix = kernel.T @ kernel + ridge_lambda * torch.eye(
        handles.shape[0], dtype=torch.float64, device=handles.device
    )
    factor = torch.linalg.cholesky(normal_matrix)
    coefficients = torch.cholesky_solve(kernel.T @ residual, factor).T
    field = CorrectedDisplacementField(
        prior=prior,
        centers=handles,
        coefficients=coefficients,
        length_scale=length_scale,
    )

    corrected_handles = field(handles)
    handle_errors = torch.linalg.vector_norm(corrected_handles - targets, dim=-1)
    objective = (
        kernel @ coefficients.T - residual
    ).square().sum() + ridge_lambda * coefficients.square().sum()
    diagnostics: dict[str, Any] = {
        "objective": "squared_handle_residual_plus_ridge_coefficients",
        "ridge_lambda": float(ridge_lambda),
        "constraint_matrix_shape": list(kernel.shape),
        "constraint_matrix_rank": int(torch.linalg.matrix_rank(kernel).item()),
        "constraint_matrix_condition": float(torch.linalg.cond(kernel).item()),
        "regularized_normal_condition": float(torch.linalg.cond(normal_matrix).item()),
        "objective_value": float(objective.item()),
        "coefficient_norm": float(torch.linalg.vector_norm(coefficients).item()),
        "handle_error_max": float(handle_errors.max().item()),
        "handle_error_mean": float(handle_errors.mean().item()),
        "dtype": str(handles.dtype),
    }
    if not torch.isfinite(handle_errors).all() or not torch.isfinite(coefficients).all():
        raise RuntimeError("soft Gaussian-RBF solve produced non-finite values")
    return field, diagnostics
