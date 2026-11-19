"""The single explicit grid-axis conversion between our xyz storage and SDFusion.

Our data/CNF convention stores a dense grid with array axes ``(x, y, z)``
(``conventions.py``); the official SDFusion VQ-VAE reads/writes its lattice as
``(z, y, x)`` (``runtime/upstream/preprocess/create_sdf.py:sample_sdf`` stores
x fastest, z slowest).  The transform is therefore a reversal of the three
spatial axes, which is its own inverse.

Rules enforced here (see ``coordination/integration-review.md``):

* apply ``xyz_to_zyx`` immediately before **every** upstream VQ-VAE encode and
  before latent-mask pooling;
* apply ``zyx_to_xyz`` immediately after the differentiable upstream decode,
  before any field/CNF query (anchors stay in world xyz and are NEVER
  transposed);
* the operation is a pure index permutation (a view), so it preserves
  autograd and adds no copy on contiguous inputs.
"""

from __future__ import annotations

import torch

__all__ = ["xyz_to_zyx", "zyx_to_xyz", "assert_axis_convention"]


def xyz_to_zyx(t: torch.Tensor) -> torch.Tensor:
    """(..., x, y, z) -> (..., z, y, x).  Last three dims must exist."""
    if t.ndim < 3:
        raise ValueError(f"expected at least 3 spatial dims, got shape {tuple(t.shape)}")
    return t.transpose(-1, -3)


def zyx_to_xyz(t: torch.Tensor) -> torch.Tensor:
    """(..., z, y, x) -> (..., x, y, z).  Same permutation as ``xyz_to_zyx``."""
    return xyz_to_zyx(t)


def assert_axis_convention(
    res: int = 64,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> dict:
    """Non-symmetric/anisotropic self-check; returns a small report.

    Builds a grid whose value is ``ax*x + ay*y + az*z`` with distinct coefficients
    on each axis, then verifies that the zyx tensor produced by ``xyz_to_zyx``
    has the expected values at *known* nodes.  A symmetric volume would pass
    trivially under a wrong permutation, so distinct coefficients are essential.
    """
    from .conventions import node_coords

    c = node_coords(res, dtype=dtype, device=device)
    xx, yy, zz = torch.meshgrid(c, c, c, indexing="ij")
    ax, ay, az = 1.0, -3.0, 7.0
    xyz = ax * xx + ay * yy + az * zz  # (x, y, z)
    zyx = xyz_to_zyx(xyz.unsqueeze(0).unsqueeze(0))[0, 0]  # (z, y, x)

    # A known node: xyz index (2, 3, 4) must equal zyx index (4, 3, 2).
    expected = (ax * c[2] + ay * c[3] + az * c[4]).item()
    got = float(zyx[4, 3, 2])
    roundtrip_ok = torch.equal(xyz, zyx_to_xyz(zyx.unsqueeze(0).unsqueeze(0))[0, 0])
    known_ok = abs(got - expected) <= 1e-10
    if not (roundtrip_ok and known_ok):
        raise AssertionError(
            f"axis convention failed: roundtrip={roundtrip_ok} known_node "
            f"expected={expected:.6g} got={got:.6g}"
        )
    return {
        "anisotropic_coefficients": [ax, ay, az],
        "known_node_xyz_index": [2, 3, 4],
        "known_node_zyx_index": [4, 3, 2],
        "known_node_expected": expected,
        "known_node_got": got,
        "roundtrip_ok": bool(roundtrip_ok),
        "ok": True,
    }
