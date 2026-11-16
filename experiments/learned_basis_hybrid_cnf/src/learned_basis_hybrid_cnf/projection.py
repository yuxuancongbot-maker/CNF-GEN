from __future__ import annotations

from typing import Any
import torch


def hard_project_H(
    H: torch.Tensor,
    c: torch.Tensor,
    b: torch.Tensor,
    min_sv: float = 1e-8,
    max_cond: float = 1e6,
    res_tol: float = 1e-8,
    precomputed_qr: tuple[torch.Tensor, torch.Tensor] | None = None,
    prechecked_sv: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Differentiable minimum-norm projection onto H c = b in float64 using thin QR of H^T.

    If H^T = Q R (reduced QR, Q: [D, K], R: [K, K] upper-triangular), then
        Pi(c) = c + Q solve_triangular(R^T, (b - c H^T)^T, upper=False)^T.
    No ridge regularization is used. Raises RuntimeError/ValueError on rank deficiency,
    ill-conditioning, non-finite values, or post-projection residual violation.
    """
    H64 = H.to(torch.float64)
    c64 = c.to(torch.float64)
    b64 = b.to(torch.float64)
    if H64.ndim != 2 or c64.ndim != 2 or b64.ndim != 2:
        raise ValueError(f"Expected 2D tensors (H, c, b), got shapes {tuple(H64.shape)}, {tuple(c64.shape)}, {tuple(b64.shape)}")
    K, D = H64.shape
    if K > D or c64.shape[1] != D or b64.shape[1] != K or c64.shape[0] != b64.shape[0]:
        raise ValueError(f"Shape mismatch in hard_project_H: H={tuple(H64.shape)}, c={tuple(c64.shape)}, b={tuple(b64.shape)}")

    if not (torch.isfinite(H64).all() and torch.isfinite(c64).all() and torch.isfinite(b64).all()):
        raise ValueError("Non-finite inputs to hard_project_H")

    if prechecked_sv is not None:
        min_sv_val = float(prechecked_sv["min_sv"])
        max_sv_val = float(prechecked_sv["max_sv"])
        cond_val = float(prechecked_sv["cond"])
        rank_val = int(prechecked_sv["rank"])
        if rank_val < K or min_sv_val <= min_sv:
            raise RuntimeError(f"Hard projection rank-deficient/ill-conditioned: rank={rank_val}/{K}, min_sv={min_sv_val:.3e} <= {min_sv:.3e}")
        if cond_val > max_cond:
            raise RuntimeError(f"Hard projection condition number too large: cond={cond_val:.3e} > {max_cond:.3e}")
    else:
        with torch.no_grad():
            sv = torch.linalg.svdvals(H64)
            if sv.numel() < K or not torch.isfinite(sv).all():
                raise RuntimeError("Invalid singular values in hard_project_H")
            min_sv_val = float(sv[-1].cpu().item())
            max_sv_val = float(sv[0].cpu().item())
            cond_val = max_sv_val / max(min_sv_val, 1e-30)
            rank_val = int((sv > min_sv).sum().cpu().item())
            if rank_val < K or min_sv_val <= min_sv:
                raise RuntimeError(f"Hard projection rank-deficient/ill-conditioned: rank={rank_val}/{K}, min_sv={min_sv_val:.3e} <= {min_sv:.3e}")
            if cond_val > max_cond:
                raise RuntimeError(f"Hard projection condition number too large: cond={cond_val:.3e} > {max_cond:.3e}")

    if precomputed_qr is not None:
        Q, R = precomputed_qr
    else:
        try:
            Q, R = torch.linalg.qr(H64.T, mode="reduced")  # Q: [D, K], R: [K, K] upper-triangular
        except Exception as e:
            raise RuntimeError(f"QR decomposition failed in hard_project_H: {e}") from e

    diagR_min = float(torch.diag(R).detach().abs().min().cpu().item())
    if not (diagR_min > min_sv * 0.1):
        raise RuntimeError(f"R diagonal near zero in hard_project_H: min|diag(R)|={diagR_min:.3e}")

    r = b64 - (c64 @ H64.T)  # [B, K]
    try:
        yT = torch.linalg.solve_triangular(R.T, r.T, upper=False)  # [K, B]
    except Exception as e:
        raise RuntimeError(f"Triangular solve failed in hard_project_H: {e}") from e

    delta = (Q @ yT).T  # [B, D]
    c_proj = c64 + delta

    with torch.no_grad():
        r_after = b64 - (c_proj @ H64.T)
        max_abs_res = float(r_after.abs().max().cpu().item())
        max_l2_res = float(r_after.norm(dim=1).max().cpu().item())
        if not (torch.isfinite(c_proj).all() and max_abs_res <= res_tol and max_l2_res <= res_tol * 10.0):
            raise RuntimeError(
                f"Hard projection residual check failed: max_abs={max_abs_res:.3e}, max_l2={max_l2_res:.3e}, tol={res_tol:.3e}"
            )

    info = {
        "ok": True,
        "rank": rank_val,
        "min_sv_H": min_sv_val,
        "max_sv_H": max_sv_val,
        "cond_H": cond_val,
        "max_residual": max_l2_res,
        "max_abs_residual": max_abs_res,
    }
    return c_proj, info
