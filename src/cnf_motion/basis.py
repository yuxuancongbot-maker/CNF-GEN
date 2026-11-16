"""Legendre basis and coefficient <-> trajectory conversions.

The trajectory model is

    q(t) = sum_{d=0}^{D-1} B_d(t) c_d,        q(t) in R^J,

where ``B_d`` are column-normalised Legendre polynomials on ``[0, 1]`` and
``c`` is a ``D * J`` coefficient vector with layout ``c[d * J + j]``.

Note on the discrete grid
-------------------------
The columns are normalised by their *sampled* L2 norm.  With 17 samples and
degree 10 the sampled Gram matrix is not exactly the identity (the columns are
still linearly independent and the pseudoinverse reconstructs coefficients
exactly).  This matches the original experiments.  For velocity rows the
reported runs use ``mode="discrete_autograd"``, which differentiates the
sampled normalised basis with autograd.  ``mode="continuous"`` treats the
normalisation constant as fixed and is the mathematically cleaner operator;
``mode="finite_difference"`` differentiates the sampled rows directly.
"""

from __future__ import annotations

import torch


def legendre_values(t: torch.Tensor, degree: int) -> torch.Tensor:
    """Return ``P_0(t), ..., P_{degree-1}(t)`` with shape ``[..., degree]``.

    ``t`` is mapped to ``x = 2t - 1`` and the standard three-term recursion is
    used:

        P_0(x) = 1,  P_1(x) = x,
        n P_n(x) = (2n - 1) x P_{n-1}(x) - (n - 1) P_{n-2}(x).
    """
    if degree < 1:
        raise ValueError(f"degree must be positive, got {degree}")
    x = 2.0 * t - 1.0
    if degree == 1:
        return torch.ones(*t.shape, 1, dtype=t.dtype, device=t.device)
    values = [torch.ones_like(x), x]
    for order in range(2, degree):
        values.append(
            ((2.0 * order - 1.0) * x * values[-1] - (order - 1.0) * values[-2])
            / order
        )
    return torch.stack(values[:degree], dim=-1)


def legendre_basis(t: torch.Tensor, degree: int) -> torch.Tensor:
    """Column-normalised Legendre basis on the sampled time grid."""
    basis = legendre_values(t, degree)
    return basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-6)


def make_basis(frames: int = 17, degree: int = 10):
    """Return ``(B, B^+)`` for ``frames`` uniform samples on ``[0, 1]``.

    ``B`` has shape ``[frames, degree]``; ``B^+`` is its pseudoinverse with
    shape ``[degree, frames]``.  ``B^+ q`` gives least-squares coefficients.
    """
    t = torch.linspace(0.0, 1.0, frames)
    basis = legendre_basis(t, degree)
    return basis, torch.linalg.pinv(basis)


def make_derivative_basis(
    frames: int = 17,
    degree: int = 10,
    mode: str = "discrete_autograd",
):
    """Return ``(B, dB/dt, d2B/dt2)`` on the uniform time grid.

    Derivatives are with respect to normalised time ``t in [0, 1]``; multiply
    by ``frames - 1`` to compare with per-frame finite differences.

    ``mode``:
      - ``"discrete_autograd"`` (default, used in the reported runs):
        autograd through the sampled normalisation.  This is the exact operator
        used by the original experiments.
      - ``"continuous"``: autograd of the raw Legendre polynomials divided by
        the fixed sampled norm.  This is the derivative of the continuous
        function represented by the basis.
      - ``"finite_difference"``: second-order finite differences of the sampled
        basis rows via ``torch.gradient``.
    """
    if mode not in {"discrete_autograd", "continuous", "finite_difference"}:
        raise ValueError(f"unknown derivative mode: {mode}")
    t = torch.linspace(0.0, 1.0, frames, requires_grad=True)
    raw = legendre_values(t, degree)
    norm = raw.norm(dim=0, keepdim=True).clamp_min(1e-6)
    basis = raw / norm
    if mode == "finite_difference":
        spacing = 1.0 / (frames - 1)
        velocity = torch.gradient(basis.detach(), spacing=spacing, dim=0)[0]
        acceleration = torch.gradient(velocity, spacing=spacing, dim=0)[0]
        return basis.detach(), velocity.detach(), acceleration.detach()

    source = basis if mode == "discrete_autograd" else raw
    velocity = torch.stack(
        [
            torch.autograd.grad(
                source[:, d].sum(), t, retain_graph=True, create_graph=True
            )[0]
            for d in range(degree)
        ],
        dim=1,
    )
    if mode == "continuous":
        velocity = velocity / norm.detach()
    acceleration = torch.stack(
        [
            torch.autograd.grad(velocity[:, d].sum(), t, retain_graph=True)[0]
            for d in range(degree)
        ],
        dim=1,
    )
    return basis.detach(), velocity.detach(), acceleration.detach()


def coefficients_from_trajectory(q: torch.Tensor, pinv: torch.Tensor) -> torch.Tensor:
    """Project joint trajectories ``[B, T, J]`` onto basis coefficients.

    Returns ``[B, degree, J]``.
    """
    return torch.einsum("dt,btj->bdj", pinv, q)


def trajectory_from_coefficients(c: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Evaluate coefficients ``[B, D]`` or ``[B, degree, J]`` on the basis.

    Returns ``[B, T, J]``.
    """
    if c.ndim == 2:
        degree = basis.shape[1]
        if c.shape[1] % degree != 0:
            raise ValueError("coefficient dimension is not a multiple of degree")
        c = c.reshape(c.shape[0], degree, -1)
    return torch.einsum("td,bdj->btj", basis, c)
