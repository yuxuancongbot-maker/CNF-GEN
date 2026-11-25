"""Core components for Darcy Neural-Basis CNF (MLP-289 + PCA alignment + std_proj FM).

Provides:
- Exact 200-anchor Darcy observation layout (20 boundary + 180 interior on 128x128 grid)
- CoordinateBasisMLP (289 continuous basis functions, ~0.41M params)
- Differentiable FP64 KKT constrained-reconstruction loss with H-conditioning control
- FP64 QR orthonormalization and train-only orthogonal PCA coordinate alignment
- Exact zero-jitter FP64 SVD-nullspace constrained lower-bound diagnostic
- DarcyFlowModel (3.34M params) and FP64 QR standardized-space CNF projection
- Darcy divergence and conservative steady-PDE residual diagnostics
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


GRID_N = 128
N_BASIS = 289
N_BOUNDARY = 20
N_INTERIOR = 180
N_OBS = N_BOUNDARY + N_INTERIOR
FOURIER_K = 8
BETA = 0.01


def boundary_points_uniform(n: int = N_BOUNDARY, size: int = GRID_N):
    m = n // 4
    xs = np.linspace(0, size - 1, m).astype(int)
    ys = np.linspace(0, size - 1, m + 2).astype(int)[1:-1]
    top = [(x, 0) for x in xs]
    bottom = [(x, size - 1) for x in xs]
    left = [(0, y) for y in ys]
    right = [(size - 1, y) for y in ys]
    return top + bottom + left + right


def interior_points_uniform(n: int = N_INTERIOR, size: int = GRID_N):
    a = int(np.round(np.sqrt(n)))
    while a > 0 and n % a != 0:
        a -= 1
    b = n // a
    xs = np.linspace(0, size - 1, a + 2).astype(int)[1:-1]
    ys = np.linspace(0, size - 1, b + 2).astype(int)[1:-1]
    return [(x, y) for y in ys for x in xs]


def anchor_indices(size: int = GRID_N) -> Tuple[np.ndarray, np.ndarray]:
    pts = boundary_points_uniform(N_BOUNDARY, size) + interior_points_uniform(N_INTERIOR, size)
    ix = np.array([p[0] for p in pts], dtype=np.int64)
    iy = np.array([p[1] for p in pts], dtype=np.int64)
    return ix, iy


def grid_coordinates(size: int = GRID_N) -> np.ndarray:
    """Return (size*size, 2) array of (x, y) coordinates in [0, 1)^2 (C-order y, x)."""
    gx = np.arange(size, dtype=np.float64) / size
    xf, yf = np.meshgrid(gx, gx)
    return np.stack([xf.ravel(), yf.ravel()], axis=1)


def fourier_basis(size: int = GRID_N, K: int = FOURIER_K) -> np.ndarray:
    """Original 2D real Fourier basis (289, size*size) in FP64."""
    xy = grid_coordinates(size)
    x, y = xy[:, 0], xy[:, 1]
    out = [np.ones_like(x)]
    for k in range(1, K + 1):
        out += [np.cos(2 * np.pi * k * x), np.sin(2 * np.pi * k * x)]
        out += [np.cos(2 * np.pi * k * y), np.sin(2 * np.pi * k * y)]
    for k in range(1, K + 1):
        for l in range(1, K + 1):
            kx = 2 * np.pi * k * x
            ly = 2 * np.pi * l * y
            out += [
                np.cos(kx) * np.cos(ly),
                np.cos(kx) * np.sin(ly),
                np.sin(kx) * np.cos(ly),
                np.sin(kx) * np.sin(ly),
            ]
    return np.stack(out, axis=0).astype(np.float64)


def positional_features(xy: torch.Tensor, n_freq: int = 16, cross_freq: int = 4) -> torch.Tensor:
    """Raw non-periodic (x, y) plus separable and cross Fourier features (130 dims)."""
    x, y = xy[:, 0], xy[:, 1]
    f = torch.arange(1, n_freq + 1, device=xy.device, dtype=xy.dtype)
    ax = 2.0 * math.pi * x[:, None] * f[None, :]
    ay = 2.0 * math.pi * y[:, None] * f[None, :]
    parts = [xy, torch.sin(ax), torch.cos(ax), torch.sin(ay), torch.cos(ay)]
    fc = torch.arange(1, cross_freq + 1, device=xy.device, dtype=xy.dtype)
    cx = 2.0 * math.pi * x[:, None] * fc[None, :]
    cy = 2.0 * math.pi * y[:, None] * fc[None, :]
    for i in range(cross_freq):
        for j in range(cross_freq):
            parts.extend(
                [
                    torch.cos(cx[:, i] + cy[:, j])[:, None],
                    torch.sin(cx[:, i] + cy[:, j])[:, None],
                    torch.cos(cx[:, i] - cy[:, j])[:, None],
                    torch.sin(cx[:, i] - cy[:, j])[:, None],
                ]
            )
    return torch.cat(parts, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.linear = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(self.norm(x)))


class CoordinateBasisMLP(nn.Module):
    """Continuous coordinate neural basis Phi_theta(x, y) -> R^{n_basis} (~0.41M params)."""

    def __init__(
        self,
        n_basis: int = N_BASIS,
        hidden: int = 256,
        n_freq: int = 16,
        cross_freq: int = 4,
        n_layers: int = 4,
    ):
        super().__init__()
        self.n_basis = n_basis
        self.hidden = hidden
        self.n_freq = n_freq
        self.cross_freq = cross_freq
        self.n_layers = n_layers
        feature_dim = 2 + 4 * n_freq + 4 * cross_freq * cross_freq
        self.input = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(n_layers)])
        self.output_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, n_basis)
        self.skip = nn.Linear(feature_dim, n_basis, bias=False)
        nn.init.orthogonal_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.orthogonal_(self.skip.weight)
        with torch.no_grad():
            self.skip.weight.mul_(0.25)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        feats = positional_features(xy, self.n_freq, self.cross_freq)
        h = self.input(feats)
        for block in self.blocks:
            h = block(h)
        return self.head(self.output_norm(h)) + self.skip(feats)

    def eval_normalized_grid(self, grid_xy: torch.Tensor) -> torch.Tensor:
        """Return unit-L2 row-normalized basis matrix P of shape (n_basis, n_points) in FP64."""
        P = self(grid_xy).T.double()
        return P / torch.linalg.vector_norm(P, dim=1, keepdim=True).clamp_min(1e-12)


def exact_cnf_kkt_loss(
    P: torch.Tensor,
    H: torch.Tensor,
    u_batch: torch.Tensor,
    obs_flat: torch.Tensor,
    jitter: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable FP64 KKT solve for min_{c: Hc=b} ||P^T c - u||_2^2.

    Returns (constrained_rel_mse, unconstrained_rel_mse, gram).
    """
    eye_g = torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
    eye_h = torch.eye(H.shape[0], dtype=P.dtype, device=P.device)
    gram = P @ P.T
    eps_g = jitter * gram.diag().mean()
    chol_g = torch.linalg.cholesky(gram + eps_g * eye_g)
    c_ls = torch.cholesky_solve(P @ u_batch.T, chol_g)
    gi_ht = torch.cholesky_solve(H.T, chol_g)
    m = H @ gi_ht
    eps_m = jitter * m.diag().mean()
    chol_m = torch.linalg.cholesky(m + eps_m * eye_h)
    b_batch = u_batch[:, obs_flat]
    lagrange = torch.cholesky_solve(b_batch.T - H @ c_ls, chol_m)
    c_opt = c_ls + gi_ht @ lagrange

    pred_opt = c_opt.T @ P
    pred_ls = c_ls.T @ P
    denom = (u_batch * u_batch).sum(dim=1).clamp_min(1e-30)
    constrained = (((pred_opt - u_batch) ** 2).sum(dim=1) / denom).mean()
    unconstrained = (((pred_ls - u_batch) ** 2).sum(dim=1) / denom).mean()
    return constrained, unconstrained, gram


def orthonormalize_rows(P: np.ndarray) -> np.ndarray:
    """FP64 reduced QR orthonormalization of basis rows so P_orth @ P_orth.T = I."""
    P64 = np.asarray(P, dtype=np.float64)
    Q, _ = np.linalg.qr(P64.T, mode="reduced")
    return np.ascontiguousarray(Q.T)


def pca_align_basis(P: np.ndarray, u_train_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormalize P on grid and apply train-only orthogonal PCA rotation.

    Preserves span(P) identically while decorrelating and energy-ordering coordinates.
    """
    P_orth = orthonormalize_rows(P)
    C_tr = u_train_flat.astype(np.float64) @ P_orth.T
    C_centered = C_tr - C_tr.mean(axis=0, keepdims=True)
    cov = (C_centered.T @ C_centered) / C_tr.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    U_pca = eigvecs[:, order]
    P_pca = np.ascontiguousarray(U_pca.T @ P_orth)
    return P_pca, U_pca


def basis_diagnostics(P: np.ndarray, obs_flat: np.ndarray) -> Dict[str, float]:
    """Compute H and Gram numerical rank / singular value / condition diagnostics."""
    H = P[:, obs_flat].T
    s_H = np.linalg.svd(H, compute_uv=False)
    tol_H = s_H[0] * max(H.shape) * np.finfo(np.float64).eps
    G = P @ P.T
    s_G = np.linalg.svd(G, compute_uv=False)
    return {
        "rank_H": int(np.sum(s_H > tol_H)),
        "sigma_min_H": float(s_H[-1]),
        "sigma_max_H": float(s_H[0]),
        "cond_H": float(s_H[0] / s_H[-1]) if s_H[-1] > 0 else float("inf"),
        "cond_G": float(s_G[0] / s_G[-1]) if s_G[-1] > 0 else float("inf"),
    }


def true_constrained_lower_bound(
    P: np.ndarray,
    u_eval_flat: np.ndarray,
    obs_flat: np.ndarray,
) -> Dict[str, float]:
    """Exact zero-jitter FP64 SVD-nullspace lower bound min_{c: Hc=b} ||c@P - u||^2."""
    P64 = np.asarray(P, dtype=np.float64)
    u64 = np.asarray(u_eval_flat, dtype=np.float64)
    n_obs = len(obs_flat)
    unobs_mask = np.ones(P64.shape[1], dtype=bool)
    unobs_mask[obs_flat] = False

    H = P64[:, obs_flat].T
    b = u64[:, obs_flat]
    U_H, S_H, Vt_H = np.linalg.svd(H, full_matrices=True)
    V1 = Vt_H[:n_obs, :].T
    N_null = Vt_H[n_obs:, :].T
    C_part = ((V1 * (1.0 / S_H)[None, :]) @ (U_H.T @ b.T)).T

    PN = N_null.T @ P64
    G_null = PN @ PN.T
    R_field = u64 - C_part @ P64
    W_opt = np.linalg.solve(G_null, PN @ R_field.T).T
    C_opt = C_part + W_opt @ N_null.T
    pred = C_opt @ P64

    err2 = (pred - u64) ** 2
    full_rel = err2.sum(axis=1) / np.sum(u64 ** 2, axis=1)
    unobs_rel = err2[:, unobs_mask].sum(axis=1) / np.sum(u64[:, unobs_mask] ** 2, axis=1)
    obs_max = float(np.max(np.abs(C_opt @ H.T - b)))
    return {
        "full_rel_mse": float(np.mean(full_rel)),
        "unobs_rel_mse": float(np.mean(unobs_rel)),
        "obs_max_abs": obs_max,
    }


def sinusoidal_embedding(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class DarcyFlowModel(nn.Module):
    """Matched conditional Flow Matching MLP (3.34M params for dim=289, hidden=512)."""

    def __init__(self, dim: int = N_BASIS, cond_dim: int = N_OBS, hidden: int = 512):
        super().__init__()
        self.dim = dim
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.time_net = nn.Sequential(
            nn.Linear(32, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + hidden * 2, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, dim),
        )

    def forward(self, coefficients: torch.Tensor, time: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        te = self.time_net(sinusoidal_embedding(time, 32))
        c = torch.cat([self.cond_net(cond), te], dim=-1)
        return self.net(torch.cat([coefficients, c], dim=-1))


def project_standardized(
    z: torch.Tensor,
    y: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    H_z: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable FP64 QR min-norm affine projection in standardized z-space.

    Solves H_z z_proj = y (where y = b - H @ mean, H_z = H * std).
    Supports y of shape (N_OBS,) or (B, N_OBS).
    """
    zd = z.double()
    yd = y.double()
    if yd.ndim == 1:
        yd = yd.unsqueeze(0)
    r = yd - zd @ H_z.T
    sol = torch.linalg.solve_triangular(R.T, r.T, upper=False)
    zp = (zd.T + Q @ sol).T
    res = zp @ H_z.T - yd
    return zp, res


def darcy_divergence(p: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Compute div(nu * grad(p)) for p of shape (..., H, W) and nu of shape (H, W)."""
    dp_dy, dp_dx = np.gradient(p, axis=(-2, -1))
    fx = nu * dp_dx
    fy = nu * dp_dy
    return np.gradient(fx, axis=-1) + np.gradient(fy, axis=-2)


def steady_pde_residual_rms(
    fields: np.ndarray,
    nu: np.ndarray,
    beta: float = BETA,
    margin: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Conservative finite-volume steady-PDE residual -div(nu grad p) - beta on interior."""
    h = 1.0 / fields.shape[-1]
    inv_h2 = 1.0 / (h * h)
    # Arithmetic face means for permeability
    ax_p = 0.5 * (nu[margin:-margin, margin:-margin] + nu[margin:-margin, margin + 1 : -margin + 1 if margin > 1 else None])
    ax_m = 0.5 * (nu[margin:-margin, margin:-margin] + nu[margin:-margin, margin - 1 : -margin - 1])
    ay_p = 0.5 * (nu[margin:-margin, margin:-margin] + nu[margin + 1 : -margin + 1 if margin > 1 else None, margin:-margin])
    ay_m = 0.5 * (nu[margin:-margin, margin:-margin] + nu[margin - 1 : -margin - 1, margin:-margin])

    p_c = fields[:, margin:-margin, margin:-margin]
    p_xp = fields[:, margin:-margin, margin + 1 : -margin + 1 if margin > 1 else None]
    p_xm = fields[:, margin:-margin, margin - 1 : -margin - 1]
    p_yp = fields[:, margin + 1 : -margin + 1 if margin > 1 else None, margin:-margin]
    p_ym = fields[:, margin - 1 : -margin - 1, margin:-margin]

    lhs = -(
        ax_p[None] * (p_xp - p_c)
        - ax_m[None] * (p_c - p_xm)
        + ay_p[None] * (p_yp - p_c)
        - ay_m[None] * (p_c - p_ym)
    ) * inv_h2
    res = lhs - beta
    raw_rms = np.sqrt(np.mean(res * res, axis=(1, 2)))
    return raw_rms, raw_rms / beta
