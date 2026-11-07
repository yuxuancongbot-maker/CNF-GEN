import torch

from cnf_motion.basis import make_basis, make_derivative_basis
from cnf_motion.motion_field import constraint_matrix, flatten_targets, project_coefficients


def test_position_projection_is_exact():
    torch.manual_seed(0)
    basis, _ = make_basis(17, 10)
    _, velocity, _ = make_derivative_basis(17, 10)
    frames = [0, 4, 8, 12, 13, 16]
    coefficients = torch.randn(3, 70)
    targets = torch.randn(3, len(frames), 7)
    constraint = constraint_matrix(basis, velocity, frames, joints=7)
    projected = project_coefficients(coefficients, constraint, flatten_targets(targets))
    residual = projected @ constraint.T - targets.reshape(3, -1)
    assert residual.abs().max() < 1e-5


def test_velocity_projection_is_exact():
    torch.manual_seed(1)
    basis, _ = make_basis(17, 10)
    _, velocity, _ = make_derivative_basis(17, 10)
    frames = [0, 4, 8, 12, 13, 16]
    coefficients = torch.randn(2, 70)
    positions = torch.randn(2, len(frames), 7)
    velocity_targets = torch.randn(2, 1, 7)
    constraint = constraint_matrix(
        basis, velocity, frames, velocity_frames=(8,), joints=7
    )
    targets = flatten_targets(positions, velocity_targets)
    projected = project_coefficients(coefficients, constraint, targets)
    residual = projected @ constraint.T - targets
    assert residual.abs().max() < 1e-5


def test_projection_is_idempotent():
    torch.manual_seed(2)
    basis, _ = make_basis(17, 10)
    _, velocity, _ = make_derivative_basis(17, 10)
    frames = [0, 4, 8, 12, 13, 16]
    coefficients = torch.randn(1, 70)
    targets = torch.randn(1, len(frames), 7)
    constraint = constraint_matrix(basis, velocity, frames, joints=7)
    flat = flatten_targets(targets)
    once = project_coefficients(coefficients, constraint, flat)
    twice = project_coefficients(once, constraint, flat)
    assert torch.allclose(once, twice, atol=1e-6)
