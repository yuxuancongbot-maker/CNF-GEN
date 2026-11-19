"""Continuous cubic interpolation of an SDF lattice with analytic gradients.

The field is the tensor-product Catmull-Rom style cubic Hermite interpolant of
the node values on an inclusive-endpoint grid (``index 0 <-> lo``,
``index N-1 <-> hi``); see :mod:`sdfcnf.conventions`.

Properties (all covered by ``tests/test_field.py``):

* it *interpolates* the nodes exactly (so ``f(grid node) == grid value`` and
  the reconstruction branch can use the exact nodal identity),
* it reproduces affine fields ``a . x + b`` exactly in value and analytic
  first derivative, including at the boundary cells and at the endpoints,
* it is ``C1`` and differentiable w.r.t. the node values, so losses on field
  values/gradients back-propagate to the SDF lattice (and through it to the
  decoder/U-Net),
* outside ``[lo, hi]`` it uses the constant extension: the value is clamped to
  the boundary value and only the out-of-domain gradient components are
  zeroed (the in-plane components remain the boundary derivatives).

Evaluation is chunked over query points with a budgetable memory estimate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .conventions import GRID_HI, GRID_LO, spacing

_OFFSETS = (-1, 0, 1, 2)  # node offsets of the cell [j, j+1] along one axis
_ITEMSIZE = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
}


def points_per_chunk(
    budget_bytes: int, dtype: torch.dtype = torch.float32, with_grad: bool = True
) -> int:
    """Estimate how many query points fit into ``budget_bytes``.

    The estimate accounts for the 64 gathered node values per point, the
    64-entry weight tensor (four per axis for value + gradient), index
    tensors (int64) and small temporaries, with a 35% safety margin.  It is a
    budget, not an exact allocator: peak memory stays within the same order
    for any realistic grid.
    """
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    item = _ITEMSIZE.get(dtype, 4)
    per_point = 64 * 8 + 64 * item + (64 * 4 * item if with_grad else 64 * item)
    per_point += (16 * item if with_grad else 4 * item) + 24
    per_point = int(per_point * 1.35)
    return max(1, budget_bytes // per_point)


class CubicSplineField:
    """Cubic Hermite spline field defined by SDF node values.

    Args:
        values: node values, shape ``(N, N, N)`` or ``(B, N, N, N)`` in
            ``(x, y, z)`` storage order.
        resolution: grid resolution; inferred from ``values`` when omitted.
        lo/hi: inclusive coordinate endpoints of the grid.
    """

    def __init__(
        self,
        values: torch.Tensor,
        resolution: Optional[int] = None,
        lo: float = GRID_LO,
        hi: float = GRID_HI,
    ) -> None:
        if values.ndim not in (3, 4):
            raise ValueError(f"values must be (N,N,N) or (B,N,N,N), got {tuple(values.shape)}")
        if not values.is_floating_point():
            raise ValueError(f"values must be a floating dtype, got {values.dtype}")
        res = values.shape[-1]
        if values.shape[-3] != res or values.shape[-2] != res:
            raise ValueError(f"values must be cubic, got {tuple(values.shape)}")
        if resolution is not None and resolution != res:
            raise ValueError(
                f"resolution={resolution} does not match values shape {tuple(values.shape)}"
            )
        if not hi > lo:
            raise ValueError(f"hi must be > lo, got lo={lo} hi={hi}")
        if not torch.isfinite(values).all():
            raise ValueError("field values contain non-finite entries")
        self.values = values
        self.resolution = res
        self.lo = float(lo)
        self.hi = float(hi)
        self.h = spacing(res, lo, hi)

    # ------------------------------------------------------------------ setup
    @property
    def batched(self) -> bool:
        return self.values.ndim == 4

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    @property
    def device(self) -> torch.device:
        return self.values.device

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        b = self.values.shape[0] if self.batched else 1
        return (
            f"CubicSplineField(batch={b}, res={self.resolution}, dtype={self.dtype}, "
            f"device={self.device}, lo={self.lo}, hi={self.hi}, h={self.h:.6g})"
        )

    # --------------------------------------------------------------- queries
    def evaluate(
        self,
        points: torch.Tensor,
        *,
        with_grad: bool = True,
        chunk: Optional[int] = None,
        budget_bytes: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate field values (and analytic spatial gradients) at ``points``.

        Args:
            points: ``(Q, 3)`` (broadcast over a batched field) or ``(B, Q, 3)``
                query positions in ``(x, y, z)`` order.
            with_grad: also return analytic ``df/dx, df/dy, df/dz``.
            chunk: maximum number of query points per chunk.
            budget_bytes: memory budget used to derive ``chunk`` when it is
                not given explicitly (see :func:`points_per_chunk`).

        Returns:
            ``(values, grads)`` where ``values`` is ``(Q,)``/``(B, Q)`` and
            ``grads`` is ``(Q, 3)``/``(B, Q, 3)`` (``None`` if
            ``with_grad=False``).
        """
        if points.shape[-1] != 3:
            raise ValueError(f"points must end with dim 3, got {tuple(points.shape)}")
        if chunk is None:
            budget = 512 * 1024 * 1024 if budget_bytes is None else budget_bytes
            chunk = points_per_chunk(budget, self.dtype, with_grad)
        if chunk <= 0:
            raise ValueError("chunk must be positive")

        grid = self.values.unsqueeze(0) if not self.batched else self.values
        batch = grid.shape[0]

        if points.ndim == 2:
            pts = points.unsqueeze(0).expand(batch, -1, -1)
            squeeze_batch = not self.batched
        elif points.ndim == 3:
            if points.shape[0] != batch:
                raise ValueError(f"batched points {tuple(points.shape)} vs field batch {batch}")
            pts = points
            squeeze_batch = False
        else:
            raise ValueError(f"points must be (Q,3) or (B,Q,3), got {tuple(points.shape)}")

        values, grads = [], []
        for b in range(batch):
            vb, gb = self._evaluate_single(grid[b], pts[b], with_grad=with_grad, chunk=chunk)
            values.append(vb)
            if with_grad:
                grads.append(gb)
        v = torch.stack(values, dim=0)
        g = torch.stack(grads, dim=0) if with_grad else None
        if squeeze_batch:
            v = v[0]
            if g is not None:
                g = g[0]
        return v, g

    def value(self, points: torch.Tensor, **kw) -> torch.Tensor:
        """Field values only (thin wrapper around :meth:`evaluate`)."""
        v, _ = self.evaluate(points, with_grad=False, **kw)
        return v

    def nodal_values(self, index_xyz: torch.Tensor) -> torch.Tensor:
        """Exact node values at integer index triples ``index_xyz`` (..., 3).

        Uses the interpolating property of the spline, so this is algebraically
        identical to evaluating the spline at the node coordinates (verified in
        ``tests/test_field.py``) but avoids the interpolation work.
        """
        if index_xyz.shape[-1] != 3:
            raise ValueError(f"indices must end with dim 3, got {tuple(index_xyz.shape)}")
        idx = index_xyz.long()
        grid = self.values.unsqueeze(0) if not self.batched else self.values
        i, j, k = idx[..., 0], idx[..., 1], idx[..., 2]
        if self.batched:
            out = grid[:, i, j, k]
        else:
            out = grid[i, j, k]
        if idx.ndim == 1:
            return out
        return out

    # -------------------------------------------------------------- internals
    def _evaluate_single(
        self, grid: torch.Tensor, points: torch.Tensor, with_grad: bool, chunk: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n = points.shape[0]
        vals, grads = [], []
        for start in range(0, n, chunk):
            piece = points[start : start + chunk]
            v, g = self._evaluate_chunk(grid, piece, with_grad=with_grad)
            vals.append(v)
            if with_grad:
                grads.append(g)
        v = torch.cat(vals, dim=0)
        g = torch.cat(grads, dim=0) if with_grad else None
        return v, g

    def _evaluate_chunk(
        self, grid: torch.Tensor, points: torch.Tensor, with_grad: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        res, lo, h = self.resolution, self.lo, self.h
        dtype = grid.dtype
        p = points.to(dtype=dtype)
        inside = (p >= lo) & (p <= self.hi)
        pc = p.clamp(lo, self.hi)
        flat = grid.reshape(-1)

        cells, weights, dweights, boundary = [], [], [], []
        for ax in range(3):
            x = pc[:, ax]
            j = torch.clamp(torch.floor((x - lo) / h).long(), 0, res - 2)
            t = (x - (lo + j.to(dtype=dtype) * h)) / h
            # Cubic Hermite basis written directly in the four node values
            # p0..p3 of the cell (Catmull-Rom tangents m1=(p2-p0)/2h,
            # m2=(p3-p1)/2h in index units).
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            weights.append(torch.stack([-h10 / 2, h00 - h11 / 2, h01 + h10 / 2, h11 / 2], dim=1))
            if with_grad:
                h00p = 6 * t**2 - 6 * t
                h10p = 3 * t**2 - 4 * t + 1
                h01p = -6 * t**2 + 6 * t
                h11p = 3 * t**2 - 2 * t
                dweights.append(
                    torch.stack([-h10p / 2, h00p - h11p / 2, h01p + h10p / 2, h11p / 2], dim=1) / h
                )
            cells.append(j)
            boundary.append((j == 0, j == res - 2))

        off = torch.tensor(_OFFSETS, device=grid.device)
        i0 = (cells[0][:, None] + off[None, :]).clamp(0, res - 1)  # (Q,4)
        i1 = (cells[1][:, None] + off[None, :]).clamp(0, res - 1)
        i2 = (cells[2][:, None] + off[None, :]).clamp(0, res - 1)
        flat_idx = (
            i0[:, :, None, None] * (res * res) + i1[:, None, :, None] * res + i2[:, None, None, :]
        )
        gathered = flat[flat_idx.reshape(-1)].reshape(-1, 4, 4, 4)

        # One-sided boundary tangents: replace the clamped ghost node so that
        # m1/m2 become forward/backward differences (affine reproduction at the
        # domain edges).
        (j0x, j2x), (j0y, j2y), (j0z, j2z) = boundary
        gathered[:, 0] = torch.where(
            j0x[:, None, None], 2 * gathered[:, 1] - gathered[:, 2], gathered[:, 0]
        )
        gathered[:, 3] = torch.where(
            j2x[:, None, None], 2 * gathered[:, 2] - gathered[:, 1], gathered[:, 3]
        )
        gathered[:, :, 0] = torch.where(
            j0y[:, None, None], 2 * gathered[:, :, 1] - gathered[:, :, 2], gathered[:, :, 0]
        )
        gathered[:, :, 3] = torch.where(
            j2y[:, None, None], 2 * gathered[:, :, 2] - gathered[:, :, 1], gathered[:, :, 3]
        )
        gathered[:, :, :, 0] = torch.where(
            j0z[:, None, None],
            2 * gathered[:, :, :, 1] - gathered[:, :, :, 2],
            gathered[:, :, :, 0],
        )
        gathered[:, :, :, 3] = torch.where(
            j2z[:, None, None],
            2 * gathered[:, :, :, 2] - gathered[:, :, :, 1],
            gathered[:, :, :, 3],
        )

        wx, wy, wz = weights
        value = (wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :] * gathered).sum(
            dim=(1, 2, 3)
        )
        if not with_grad:
            return value, None

        dx, dy, dz = dweights
        gx = (dx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :] * gathered).sum(
            dim=(1, 2, 3)
        )
        gy = (wx[:, :, None, None] * dy[:, None, :, None] * wz[:, None, None, :] * gathered).sum(
            dim=(1, 2, 3)
        )
        gz = (wx[:, :, None, None] * wy[:, None, :, None] * dz[:, None, None, :] * gathered).sum(
            dim=(1, 2, 3)
        )
        # Constant extension outside the domain: zero only the components whose
        # coordinate lies outside; the other components stay boundary derivatives.
        mask = inside.to(dtype=dtype)
        grad = torch.stack([gx * mask[:, 0], gy * mask[:, 1], gz * mask[:, 2]], dim=-1)
        return value, grad


def affine_grid(
    slope: torch.Tensor,
    bias: torch.Tensor | float,
    resolution: int,
    lo: float = GRID_LO,
    hi: float = GRID_HI,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sample the affine field ``slope . x + bias`` on the grid (test helper)."""
    from .conventions import grid_points

    pts = grid_points(resolution, lo, hi, dtype=dtype, device=device)
    return (pts * slope.to(dtype=dtype, device=pts.device)).sum(-1) + float(bias)
