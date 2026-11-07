import torch

from cnf_motion.basis import (
    coefficients_from_trajectory,
    make_basis,
    make_derivative_basis,
    trajectory_from_coefficients,
)


def test_basis_shapes_and_unit_columns():
    basis, pinv = make_basis(17, 10)
    assert basis.shape == (17, 10)
    assert pinv.shape == (10, 17)
    assert torch.allclose(basis.norm(dim=0), torch.ones(10), atol=1e-5)


def test_coefficient_roundtrip_on_basis_span():
    torch.manual_seed(0)
    basis, pinv = make_basis(17, 10)
    coefficients = torch.randn(4, 10, 7)
    q = trajectory_from_coefficients(coefficients, basis)
    recovered = coefficients_from_trajectory(q, pinv)
    assert torch.allclose(coefficients, recovered, atol=1e-5)


def test_derivative_modes_have_expected_shapes():
    for mode in ("discrete_autograd", "continuous", "finite_difference"):
        basis, velocity, acceleration = make_derivative_basis(17, 10, mode=mode)
        assert basis.shape == (17, 10)
        assert velocity.shape == (17, 10)
        assert acceleration.shape == (17, 10)
        assert torch.isfinite(velocity).all()


def test_continuous_derivative_matches_finite_difference():
    basis, velocity, _ = make_derivative_basis(129, 5, mode="continuous")
    finite = (basis[2:] - basis[:-2]) / 2 * 128
    assert torch.allclose(velocity[1:-1], finite, atol=1e-2)
