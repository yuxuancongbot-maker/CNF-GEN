"""Hermite Gaussian corrective neural field with an exact FP64 solve.

Layout (contract: ``A = L Psi`` with ``L = (value, dx, dy, dz)`` at the
anchors, 128 correction functions for 32 anchors):

* there are ``4 * n`` basis functions for ``n`` anchor centers: one Gaussian
  ``phi_j(x) = exp(-|x - c_j|^2 / (2 sigma^2))`` plus its three derivatives
  with respect to the anchor center ``dphi_j/dc_jk`` (first-order Hermite
  Gaussians), with ``sigma = 2 * grid spacing`` by default,
* there are ``4 * n`` constraint rows: for anchor ``i`` the operators
  ``(value, d/dx, d/dy, d/dz)`` act on the corrected field
  ``f = F + sum_j psi_j B_j``,
* the system solved is ``M psi = b`` with
  ``M[i*4+op, j*4+typ] = L_i B_j`` and ``b = L(targets) - L(F)``, i.e. the
  contract's ``A = L Psi`` written for the coefficient vector ``psi``.

Solve policy: FP64 only, **no ridge/diagonal jitter ever**.  The direct
``torch.linalg.solve`` is attempted first; ``torch.linalg.lstsq`` (also
jitter-free) is only tried when LAPACK refuses the factorisation.  Whatever
the method, the result must pass the residual gate ``||M psi - b||_inf <=
residual_tol * max(1, ||b||_inf)`` or :class:`CNFSolveError` is raised with
residual, rank and condition diagnostics attached.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch

from .conventions import GRID_HI, GRID_LO, spacing

OPS = 4  # (value, d/dx, d/dy, d/dz) rows per anchor
_BASIS_PER_CENTER = 4  # (phi, dphi/dc_x, dphi/dc_y, dphi/dc_z) columns per center


class CNFSolveError(RuntimeError):
    """Raised when the exact FP64 constraint solve cannot be trusted."""

    def __init__(self, message: str, diagnostics: "SolveDiagnostics") -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class SolveDiagnostics:
    """Residual/rank/conditioning report for one (batched) constraint solve."""

    method: str
    n_constraints: int
    dtype: str
    sigma: float
    rank: int
    condition: float
    residual_inf: float
    residual_rms: float
    rhs_inf: float
    ok: bool

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def flatten_ops(x: torch.Tensor) -> torch.Tensor:
    """``(..., n, 4) -> (..., 4n)``; ``(..., 4n)`` unchanged."""
    if x.shape[-1] == OPS:
        return x.reshape(*x.shape[:-2], x.shape[-2] * OPS)
    return x


class HermiteGaussianCNF:
    """Exact Hermite Gaussian correction of a scalar field at anchor points."""

    def __init__(
        self,
        grid_spacing: float,
        sigma_scale: float = 2.0,
        solver_dtype: torch.dtype = torch.float64,
        residual_tol: float = 1e-8,
        lo: float = GRID_LO,
        hi: float = GRID_HI,
    ) -> None:
        if grid_spacing <= 0:
            raise ValueError(f"grid_spacing must be positive, got {grid_spacing}")
        if sigma_scale <= 0:
            raise ValueError(f"sigma_scale must be positive, got {sigma_scale}")
        if residual_tol <= 0:
            raise ValueError("residual_tol must be positive")
        self.grid_spacing = float(grid_spacing)
        self.sigma_scale = float(sigma_scale)
        self.sigma = float(sigma_scale) * float(grid_spacing)
        self.solver_dtype = solver_dtype
        self.residual_tol = float(residual_tol)
        self.lo = float(lo)
        self.hi = float(hi)

    @classmethod
    def for_resolution(
        cls, resolution: int, sigma_scale: float = 2.0, **kw
    ) -> "HermiteGaussianCNF":
        """Build with the node spacing of an inclusive-endpoint grid."""
        lo, hi = kw.pop("lo", GRID_LO), kw.pop("hi", GRID_HI)
        return cls(spacing(resolution, lo, hi), sigma_scale, lo=lo, hi=hi, **kw)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"HermiteGaussianCNF(h={self.grid_spacing:.6g}, sigma={self.sigma:.6g} "
            f"(= {self.sigma_scale:g} * h), solver_dtype={self.solver_dtype}, "
            f"residual_tol={self.residual_tol:g})"
        )

    # ----------------------------------------------------------------- matrix
    def system_matrix(self, centers: torch.Tensor) -> torch.Tensor:
        """``M[i*4+op, j*4+typ] = L_i B_j`` in the solver dtype.

        ``centers``: ``(n, 3)`` or ``(B, n, 3)``.  The matrix is a pure
        function of the centers and is never modified in place.
        """
        if centers.shape[-1] != 3:
            raise ValueError(f"centers must end with dim 3, got {tuple(centers.shape)}")
        c = centers.to(dtype=self.solver_dtype)
        r = c.unsqueeze(-2) - c.unsqueeze(-3)  # (..., i, j, 3) = a_i - c_j
        inv = 1.0 / (self.sigma**2)
        phi = torch.exp(-((r**2).sum(-1)) * (inv / 2.0))  # (..., i, j)
        n = c.shape[-2]
        # Leading batch dims are ``c.shape[:-2]`` (the trailing two are n,3).
        m = torch.empty(*c.shape[:-2], n * OPS, n * OPS, dtype=self.solver_dtype, device=c.device)

        m[..., 0::OPS, 0::OPS] = phi
        for k in range(3):
            m[..., 0::OPS, (k + 1) :: OPS] = (r[..., k] * inv) * phi
        for j in range(3):
            m[..., (j + 1) :: OPS, 0::OPS] = (-r[..., j] * inv) * phi
            for k in range(3):
                delta = 1.0 if j == k else 0.0
                m[..., (j + 1) :: OPS, (k + 1) :: OPS] = (
                    (delta - r[..., k] * r[..., j] * inv) * inv * phi
                )
        return m

    # ------------------------------------------------------------------ solve
    def solve(
        self,
        base_ops: torch.Tensor,
        targets: torch.Tensor,
        centers: torch.Tensor,
        *,
        diagnostics: bool = True,
    ) -> Tuple[torch.Tensor, SolveDiagnostics]:
        """Solve ``M psi = L(targets) - L(F)`` exactly in FP64.

        Args:
            base_ops: ``L(F)`` at the anchors, ``(..., n, 4)`` or ``(..., 4n)``
                (any float dtype, cast to the solver dtype).
            targets: ``L(targets)`` in the same layout: per anchor the value
                constraint followed by the three gradient components.
            centers: ``(n, 3)`` or ``(B, n, 3)``.

        Returns:
            ``(psi, diagnostics)`` with ``psi`` of shape ``(..., 4n)`` in the
            solver dtype.  Raises :class:`CNFSolveError` on singular,
            non-finite, or residual-gate-failing solves.
        """
        b = flatten_ops(targets).to(self.solver_dtype) - flatten_ops(base_ops).to(self.solver_dtype)
        m = self.system_matrix(centers)
        if m.shape[-1] != b.shape[-1]:
            raise ValueError(f"shape mismatch: matrix {tuple(m.shape)} vs rhs {tuple(b.shape)}")

        # Explicit column RHS: a (B, 128) RHS is otherwise interpreted as a
        # matrix by torch.linalg.solve.  Solve against (B, 128, 1) and squeeze.
        method = "solve"
        try:
            psi = torch.linalg.solve(m, b.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            method = "lstsq"
            if m.ndim == 2:
                psi = torch.linalg.lstsq(m, b.unsqueeze(-1)).solution.squeeze(-1)
            else:
                psi = torch.stack(
                    [
                        torch.linalg.lstsq(m[i], b[i].unsqueeze(-1)).solution.squeeze(-1)
                        for i in range(m.shape[0])
                    ],
                    dim=0,
                )

        with torch.no_grad():
            resid = (m @ psi.unsqueeze(-1)).squeeze(-1) - b
            resid_inf = float(resid.abs().amax())
            resid_rms = float((resid**2).mean().sqrt())
            rhs_inf = float(b.abs().amax())
            scale = max(1.0, rhs_inf)
            finite = bool(torch.isfinite(psi).all())
            ok = finite and resid_inf <= self.residual_tol * scale
            rank, cond = self._rank_and_condition(m) if diagnostics else (-1, float("nan"))

        diag = SolveDiagnostics(
            method=method,
            n_constraints=int(b.shape[-1]),
            dtype=str(self.solver_dtype),
            sigma=float(self.sigma),
            rank=rank,
            condition=cond,
            residual_inf=resid_inf,
            residual_rms=resid_rms,
            rhs_inf=rhs_inf,
            ok=ok,
        )
        if not ok:
            raise CNFSolveError(
                "exact FP64 constraint solve failed: "
                f"finite={finite} residual_inf={resid_inf:.3e} "
                f"(gate {self.residual_tol:g} * scale {scale:.3e}), rank={rank}/{b.shape[-1]}, "
                f"method={method}. No diagonal jitter/ridge term is applied by design; "
                "the constraint set or Gaussian kernel is degenerate.",
                diagnostics=diag,
            )
        return psi, diag

    def _rank_and_condition(self, m: torch.Tensor) -> Tuple[int, float]:
        with torch.no_grad():
            if m.ndim == 2:
                return int(torch.linalg.matrix_rank(m)), float(torch.linalg.cond(m))
            ranks, conds = [], []
            for i in range(m.shape[0]):
                ranks.append(int(torch.linalg.matrix_rank(m[i])))
                conds.append(float(torch.linalg.cond(m[i])))
        return int(min(ranks)), float(max(conds))

    # ------------------------------------------------------------- correction
    def correction(
        self,
        psi: torch.Tensor,
        centers: torch.Tensor,
        points: torch.Tensor,
        *,
        with_grad: bool = True,
        chunk: Optional[int] = None,
        budget_bytes: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate ``sum_j psi_j B_j`` (and its spatial gradient) at points.

        ``psi`` is ``(4n,)`` or ``(B, 4n)``, ``centers`` ``(n, 3)`` /
        ``(B, n, 3)``, ``points`` ``(q, 3)`` / ``(B, q, 3)``; singleton batch
        dimensions broadcast.  Returns ``(values, grads)`` of shape
        ``(..., q)`` / ``(..., q, 3)`` in the solver dtype (the batch
        dimension is kept whenever any input is batched).
        """
        if points.shape[-1] != 3:
            raise ValueError(f"points must end with dim 3, got {tuple(points.shape)}")
        psi_single = psi.ndim == 1
        centers_single = centers.ndim == 2
        points_single = points.ndim == 2

        psi_b = psi.reshape(1, -1) if psi_single else psi
        c_b = centers.reshape(1, -1, 3) if centers_single else centers
        p_b = points.reshape(1, -1, 3) if points_single else points

        batch = max(psi_b.shape[0], c_b.shape[0], p_b.shape[0])
        psi_b = self._broadcast(psi_b, batch, "psi")
        c_b = self._broadcast(c_b, batch, "centers")
        p_b = self._broadcast(p_b, batch, "points")
        psi4 = psi_b.to(self.solver_dtype).reshape(batch, -1, _BASIS_PER_CENTER)
        c64 = c_b.to(self.solver_dtype)
        p64 = p_b.to(self.solver_dtype)

        if chunk is None:
            budget = 256 * 1024 * 1024 if budget_bytes is None else budget_bytes
            chunk = max(1, budget // self._bytes_per_point(c64.shape[-2], with_grad))

        values, grads = [], []
        for bi in range(batch):
            vb, gb = self._correction_single(
                psi4[bi], c64[bi], p64[bi], with_grad=with_grad, chunk=chunk
            )
            values.append(vb)
            if with_grad:
                grads.append(gb)
        v = torch.stack(values, dim=0)
        g = torch.stack(grads, dim=0) if with_grad else None
        if psi_single and centers_single and points_single:
            v = v[0]
            g = g[0] if g is not None else None
        return v, g

    @staticmethod
    def _broadcast(x: torch.Tensor, batch: int, name: str) -> torch.Tensor:
        if x.shape[0] == batch:
            return x
        if x.shape[0] == 1:
            return x.expand(batch, *x.shape[1:])
        raise ValueError(f"{name} batch {x.shape[0]} is incompatible with batch {batch}")

    @staticmethod
    def _bytes_per_point(n_centers: int, with_grad: bool) -> int:
        item = 8  # FP64 evaluation
        per = n_centers * 8 + n_centers * 3 * item + n_centers * item
        if with_grad:
            per += n_centers * 3 * item * 3
        return int(per * 1.35)

    def _correction_single(
        self,
        psi4: torch.Tensor,
        centers: torch.Tensor,
        points: torch.Tensor,
        with_grad: bool,
        chunk: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        vals, grads = [], []
        for start in range(0, points.shape[0], chunk):
            p = points[start : start + chunk]
            v, g = self._correction_chunk(psi4, centers, p, with_grad=with_grad)
            vals.append(v)
            if with_grad:
                grads.append(g)
        v = torch.cat(vals, dim=0)
        g = torch.cat(grads, dim=0) if with_grad else None
        return v, g

    def _correction_chunk(
        self, psi4: torch.Tensor, centers: torch.Tensor, points: torch.Tensor, with_grad: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        inv = 1.0 / (self.sigma**2)
        r = points.unsqueeze(-2) - centers.unsqueeze(-3)  # (q, n, 3) = x - c_j
        phi = torch.exp(-((r**2).sum(-1)) * (inv / 2.0))  # (q, n)
        linear = (r * psi4[None, :, 1:] * inv).sum(-1)  # sum_k psi_k r_k / sigma^2
        value = (phi * (psi4[None, :, 0] + linear)).sum(1)
        if not with_grad:
            return value, None
        grad = -(phi * psi4[None, :, 0])[..., None] * r * inv  # psi_0 * grad phi
        grad = grad + phi[..., None] * psi4[None, :, 1:] * inv  # delta term
        grad = grad - (phi * linear)[..., None] * r * inv  # cross term
        return value, grad.sum(1)

    # ------------------------------------------------------------ diagnostics
    def verify(
        self,
        base_ops: torch.Tensor,
        targets: torch.Tensor,
        psi: torch.Tensor,
        centers: torch.Tensor,
        *,
        with_eval_check: bool = True,
    ) -> Dict[str, float]:
        """FP64 residual report for ``L(F) + M psi - L(targets)``.

        When ``with_eval_check`` is set, the basis functions are evaluated
        *independently* at the anchor positions and compared against the
        matrix rows, so an inconsistency between :meth:`system_matrix` and
        :meth:`correction` cannot hide behind a good linear-algebra residual.
        """
        b = flatten_ops(targets).to(self.solver_dtype) - flatten_ops(base_ops).to(self.solver_dtype)
        m = self.system_matrix(centers)
        psi64 = psi.to(self.solver_dtype)
        with torch.no_grad():
            resid = (m @ psi64.unsqueeze(-1)).squeeze(-1) - b
            out: Dict[str, float] = {
                "constraint_residual_inf": float(resid.abs().amax()),
                "constraint_residual_rms": float((resid**2).mean().sqrt()),
                "constraint_rhs_inf": float(b.abs().amax()),
            }
            if with_eval_check:
                rows = self._ops_at_centers(psi64, centers)
                out["basis_consistency_inf"] = float(
                    (rows - (m @ psi64.unsqueeze(-1)).squeeze(-1)).abs().amax()
                )
        return out

    def _ops_at_centers(self, psi64: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Evaluate the correction's ``(value, dx, dy, dz)`` at its own centers."""
        single = psi64.ndim == 1
        psi_b = psi64.reshape(1, -1) if single else psi64
        c_b = centers.reshape(1, -1, 3) if centers.ndim == 2 else centers
        rows = []
        for bi in range(psi_b.shape[0]):
            ci = c_b[bi] if c_b.shape[0] > 1 else c_b[0]
            v, g = self._correction_chunk(
                psi_b[bi].reshape(-1, _BASIS_PER_CENTER), ci, ci, with_grad=True
            )
            ops = torch.stack([v, g[:, 0], g[:, 1], g[:, 2]], dim=-1).reshape(-1)
            rows.append(ops)
        out = torch.stack(rows, dim=0)
        return out[0] if single and centers.ndim == 2 else out
