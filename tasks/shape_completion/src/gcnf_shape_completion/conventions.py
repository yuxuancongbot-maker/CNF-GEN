"""Grid/axis conventions shared by the data adapter, field, and CNF modules.

Conventions (from ``coordination/contract.md`` and the official SDFusion
checkout at commit 09801f6d04c04816e5827af981e2ae77be77ce5b):

* Storage order is ``(x, y, z)``: array axis 0 is ``x``, axis 1 is ``y``,
  axis 2 is ``z``.  The official ``utils/demo_util.py:get_partial_shape``
  applies the ``x`` range to spatial dim 0, ``y`` to dim 1, ``z`` to dim 2 of
  ``(B, 1, H, W, D)`` tensors, and ``get_shape_mask`` confirms the same
  mapping (``bottom`` masks dim 1 = ``y``, ``front`` masks dim 2 = ``z``).
* Voxel endpoints are *inclusive*: index 0 is coordinate ``-1`` and index
  ``N-1`` (63 for 64^3) is coordinate ``+1``, so the node spacing is
  ``h = 2 / (N - 1)``.  This follows the contract wording "index 0 and 63
  correspond to normalized object coordinates -1 and +1".  (The official demo
  helper ``get_partial_shape`` converts ranges with the endpoint-exclusive
  map ``index = (c + 1) / 2 * N``; we deliberately do not use that map for
  field queries.)
* Positions and normals are ``(M, 3)`` arrays in the same ``(x, y, z)`` frame
  as the storage axes, so the analytic field gradient
  ``grad f = (df/dx, df/dy, df/dz)`` is compared directly with the packed
  outward normals.
* TSDF: positive outside the surface, truncation ``+/-0.2``; voxels outside
  ``known_mask`` hold the ``+0.2`` filler in the observation grid.
"""

from __future__ import annotations

from typing import Tuple

import torch

AXIS_ORDER: Tuple[str, str, str] = ("x", "y", "z")
GRID_RESOLUTION = 64
GRID_LO = -1.0
GRID_HI = +1.0
TRUNCATION = 0.2
FILLER = +0.2
N_ANCHORS = 32


def spacing(resolution: int, lo: float = GRID_LO, hi: float = GRID_HI) -> float:
    """Node spacing ``h`` for an inclusive-endpoint grid."""
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    if not hi > lo:
        raise ValueError(f"hi must be > lo, got lo={lo} hi={hi}")
    return (hi - lo) / (resolution - 1)


def node_coords(
    resolution: int,
    lo: float = GRID_LO,
    hi: float = GRID_HI,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """1-D node coordinates of shape ``(resolution,)``, inclusive endpoints."""
    return torch.linspace(lo, hi, resolution, dtype=dtype, device=device)


def coord_to_index(
    coord: torch.Tensor, resolution: int, lo: float = GRID_LO, hi: float = GRID_HI
) -> torch.Tensor:
    """Continuous voxel index of ``coord`` (0 at ``lo``, ``N-1`` at ``hi``)."""
    return (coord - lo) / spacing(resolution, lo, hi)


def index_to_coord(
    index: torch.Tensor, resolution: int, lo: float = GRID_LO, hi: float = GRID_HI
) -> torch.Tensor:
    """Coordinate of a (continuous) voxel index."""
    return lo + index * spacing(resolution, lo, hi)


def grid_points(
    resolution: int,
    lo: float = GRID_LO,
    hi: float = GRID_HI,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """All grid node positions in storage order, shape ``(N, N, N, 3)``."""
    c = node_coords(resolution, lo, hi, dtype=dtype, device=device)
    xx, yy, zz = torch.meshgrid(c, c, c, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1)


def containing_voxel(
    positions: torch.Tensor, resolution: int, lo: float = GRID_LO, hi: float = GRID_HI
) -> torch.Tensor:
    """Integer voxel index ``(x, y, z)`` containing ``positions`` (..., 3)."""
    idx = coord_to_index(positions, resolution, lo, hi).round().long()
    return idx.clamp(0, resolution - 1)


def check_positions_normals(positions: torch.Tensor, normals: torch.Tensor | None = None) -> None:
    """Validate ``(..., M, 3)`` position/normal layouts early and loudly."""
    if positions.shape[-1] != 3:
        raise ValueError(f"positions must have trailing dim 3, got shape {tuple(positions.shape)}")
    if positions.ndim not in (2, 3):
        raise ValueError(
            f"positions must be (M, 3) or (B, M, 3), got shape {tuple(positions.shape)}"
        )
    if not torch.isfinite(positions).all():
        raise ValueError("positions contain non-finite values")
    if normals is not None:
        if normals.shape != positions.shape:
            raise ValueError(
                f"normals shape {tuple(normals.shape)} != positions shape {tuple(positions.shape)}"
            )
        if not torch.isfinite(normals).all():
            raise ValueError("normals contain non-finite values")
