"""Masked-latent conditioning built **only** from ``partial_sdf`` +
``known_mask``.

Mechanism (contract: "common masked latent conditioning mechanism compatible
with official masked DDIM completion"):

* ``z_cond = encode(partial_sdf)`` with no gradients.  ``partial_sdf`` holds
  the true truncated TSDF inside ``known_mask`` and the documented ``+0.2``
  filler outside it, so ``z_cond`` can never contain hidden ground-truth
  geometry - the target grid/latent never enters this path.
* ``z_target = encode(target_sdf)`` with no gradients, used **only** as the
  diffusion training target.
* the latent observation mask is a conservative reduction of ``known_mask``
  (see :func:`latent_observation_mask`), and training mixes
  ``x_t = q_sample(mask * z_cond + (1 - mask) * z_target, t,
  mask * noise_obs + (1 - mask) * noise_hidden)``.  Because official
  ``q_sample`` is affine in both arguments this is *exactly* the official
  masked replacement ``img = q_sample(x0, t) * mask + (1 - mask) * img`` used
  by ``DDIMSampler.ddim_sampling`` (``SDFusionModel.shape_comp``), sampled at
  training timesteps instead of the DDIM chain.  The official sampler is fed
  the same ``(z_cond, mask)`` pair at inference - see :mod:`sdfcnf.inference`.

Mask conservatism and assumptions
---------------------------------
* ``reduce="half"`` (default): a latent voxel is observed when **at least
  half** of the source voxels of its ``4x4x4`` block are observed.  ``"all"``
  (every voxel observed) and ``"any"`` are kept as explicit alternatives for
  ablations; none of them silently adapts to the data.
* ``erode`` (default 0) optionally requires the full 6-neighbourhood of latent
  voxels to be observed (a correct binary min-pool erosion).  This is OFF by
  default in both preflight and training; when enabled the erosion is a true
  min-pool (`-max_pool(-mask)`), not the always-false `max_pool(-mask)`.
* an empty observed mask is reported (``n_empty_masks``) and raises no
  fallback: we never substitute ground truth for missing observation.  The
  preflight marks the run as failed (and raises) if the condition is empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from . import axes
from .conventions import index_to_coord
from .data import Batch


@dataclass
class Conditioning:
    """Observation-only condition for one batch."""

    z_cond: torch.Tensor  # (B, C, l, l, l)
    latent_mask: torch.Tensor  # (B, 1, l, l, l) float, 1 = observed
    stats: Dict[str, float] = field(default_factory=dict)


def latent_observation_mask(
    known_mask: torch.Tensor,
    latent_res: int,
    *,
    reduce: str = "half",
    erode: int = 0,
) -> torch.Tensor:
    """Reduce a voxel observation mask to latent resolution.

    Args:
        known_mask: ``(B, 1, R, R, R)`` bool observation mask.
        latent_res: latent grid size (``l``), e.g. 16 for 64^3 inputs.
        reduce: ``"half"`` (default; >= 50% of the block's voxels observed),
            ``"all"`` (every voxel observed) or ``"any"`` (kept for ablation).
        erode: number of 6-neighbourhood min-pool erosions in latent space.

    Returns:
        ``(B, 1, l, l, l)`` float mask (1.0 = observed).
    """
    if known_mask.ndim != 5:
        raise ValueError(f"known_mask must be (B,1,R,R,R), got {tuple(known_mask.shape)}")
    res = known_mask.shape[-1]
    if known_mask.shape[-3] != res or known_mask.shape[-2] != res:
        raise ValueError(f"known_mask must be cubic, got {tuple(known_mask.shape)}")
    if res % latent_res != 0:
        raise ValueError(f"grid {res} not divisible by latent resolution {latent_res}")
    if reduce not in ("half", "all", "any"):
        raise ValueError(f"reduce must be 'half', 'all' or 'any', got {reduce!r}")
    if erode < 0:
        raise ValueError("erode must be >= 0")

    b = known_mask.shape[0]
    block = res // latent_res
    k = known_mask.bool()
    # (B,1,l,bs,l,bs,l,bs) -> move block axes together -> reduce over bs^3
    x = k.reshape(b, 1, latent_res, block, latent_res, block, latent_res, block)
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(
        b, 1, latent_res, latent_res, latent_res, block**3
    )
    if reduce == "all":
        mask = x.all(dim=-1)
    elif reduce == "any":
        mask = x.any(dim=-1)
    else:  # "half": at least 50% of the block's voxels are known
        mask = x.sum(dim=-1) >= (block**3) / 2

    for _ in range(erode):
        mf = mask.float()
        # Correct binary min-pool erosion: -max_pool(-mf) is 1 only where every
        # neighbour is 1.  The old `max_pool(-mf) > 0.5` was always false.
        neigh_all = -F.max_pool3d(-mf, kernel_size=3, stride=1, padding=1) > 0.5
        mask = mask & neigh_all
    return mask.float()


def build_condition(
    bundle, batch: Batch, *, mask_reduce: str = "half", mask_erode: int = 0
) -> Conditioning:
    """Encode the observation and build the conservative latent mask (no-grad).

    The single shared xyz->zyx transform is applied at BOTH upstream entry
    points: VQ-VAE encoding and latent-mask pooling.  ``target_sdf`` is never
    read here, so the condition cannot leak the complete label.
    """
    with torch.no_grad():
        partial_zyx = axes.xyz_to_zyx(batch.partial_sdf)
        known_zyx = axes.xyz_to_zyx(batch.known_mask)
        z_cond = bundle.encode(partial_zyx)
        mask = latent_observation_mask(
            known_zyx, z_cond.shape[-1], reduce=mask_reduce, erode=mask_erode
        )
    observed = mask.mean(dim=(1, 2, 3, 4))
    stats = {
        "latent_observed_fraction": float(observed.mean()),
        "latent_observed_fraction_min": float(observed.min()),
        "n_empty_masks": int((observed == 0).sum()),
        "known_fraction": float(batch.known_mask.float().mean()),
        # Exact counts (integration-review: "known mask count").
        "known_voxel_count": int(batch.known_mask.sum()),
        "latent_observed_count": int(mask.sum()),
        "latent_observed_fraction_mean": float(observed.mean()),
    }
    if stats["n_empty_masks"]:
        # Explicit, never a silent fallback: no conditioning signal in those
        # samples; the batch is still trainable (unconditional behaviour).
        stats["empty_mask_warn"] = 1.0
    return Conditioning(z_cond=z_cond, latent_mask=mask, stats=stats)


def masked_q_sample(
    z_cond: torch.Tensor,
    z_target: torch.Tensor,
    mask: torch.Tensor,
    t: torch.Tensor,
    schedule: Any,
    *,
    noise_obs: torch.Tensor,
    noise_hidden: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Official ``q_sample`` on the observation/target mix.

    The two noise tensors are **supplied by the caller**, never drawn here:
    preflight allocates them once and reuses the exact same tensors for every
    arm so the shared diffusion loss is bit-identical.  Fresh draws here were
    the defect that made diffusion losses differ across arms.

    Returns ``(x_t, noise)`` where ``noise`` is the mixture that generated
    ``x_t`` (observed slots carry ``noise_obs`` like the official DDIM
    replacement, hidden slots carry ``noise_hidden``).
    """
    if z_cond.shape != z_target.shape:
        raise ValueError(f"z_cond {tuple(z_cond.shape)} != z_target {tuple(z_target.shape)}")
    if mask.shape[1] != 1 or mask.shape[2:] != z_cond.shape[2:]:
        raise ValueError(f"mask {tuple(mask.shape)} incompatible with latent {tuple(z_cond.shape)}")
    if noise_obs.shape != z_cond.shape or noise_hidden.shape != z_cond.shape:
        raise ValueError(
            f"noise shapes {tuple(noise_obs.shape)}/{tuple(noise_hidden.shape)} "
            f"must match latent {tuple(z_cond.shape)}"
        )
    m = mask.to(dtype=z_cond.dtype)
    x0_mix = m * z_cond + (1.0 - m) * z_target
    noise = m * noise_obs.to(dtype=z_cond.dtype) + (1.0 - m) * noise_hidden.to(dtype=z_cond.dtype)
    return schedule.q_sample(x0_mix, t, noise), noise


def anchor_query_points(index_xyz: torch.Tensor, resolution: int) -> torch.Tensor:
    """Node coordinates (``Q, 3``) of integer grid index triples (``Q, 3``)."""
    return index_to_coord(index_xyz.to(dtype=torch.get_default_dtype()), resolution)
