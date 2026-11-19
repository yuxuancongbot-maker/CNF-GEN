"""Shared train-step entry point for the three matched SDFusion+CNF arms.

The three arms differ ONLY in constraint enforcement:

* ``plain`` - masked-latent denoising + a reconstruction branch on the decoded
  clean SDF (no anchor penalty);
* ``soft``  - ``plain`` + soft anchor value (target exactly 0) and outward-normal
  penalties on the decoded field;
* ``cnf``   - ``plain`` with the reconstruction branch evaluated on the exactly
  corrected field ``f = decode + Hermite-Gaussian correction`` (FP64 solve,
  residuals/anchor errors logged, NO anchor penalty).

In every arm the diffusion condition is built *solely* from ``partial_sdf`` +
``known_mask``; ``target_sdf`` is used only as a training target.  The VQ-VAE
is frozen but decode keeps gradients from the predicted clean latent to the
U-Net.  No optimizer/step or checkpoint write happens in this module.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from . import axes
from .cnf import HermiteGaussianCNF, flatten_ops
from .conditioning import Conditioning, build_condition, masked_q_sample
from .conventions import index_to_coord
from .data import Batch
from .field import CubicSplineField

ARMS = ("plain", "soft", "cnf")


def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    """Same gathering as the official ``extract_into_tensor`` (shape broadcast)."""
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE, shape ``(B,)``."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {tuple(pred.shape)} vs {tuple(target.shape)}")
    return (pred.float() - target.float()).pow(2).flatten(1).mean(dim=1)


def make_query_points(
    anchors: torch.Tensor,
    resolution: int,
    n_grid: int = 224,
    seed: int = 111,
) -> torch.Tensor:
    """Shared query coordinates (xyz): the anchors plus deterministic grid nodes.

    The same tensor is used by every arm, satisfying the "common query
    coordinates" requirement.  Grid nodes are inclusive-endpoint coordinates.
    Returns ``(Q, 3)`` for unbatched anchors, ``(B, Q, 3)`` for batched anchors.
    """
    dev = anchors.device
    idx = torch.arange(resolution, device=dev)
    grid = torch.stack(torch.meshgrid(idx, idx, idx, indexing="ij"), dim=-1).reshape(
        -1, 3
    )  # (R^3, 3) integer indices
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    perm = torch.randperm(grid.shape[0], generator=g)[:n_grid]
    chosen = index_to_coord(grid[perm].to(dtype=anchors.dtype), resolution).to(dev)  # (n_grid, 3)
    if anchors.ndim == 2:
        return torch.cat([anchors, chosen], dim=0)  # (Q, 3)
    if anchors.ndim == 3:
        return torch.cat(
            [anchors, chosen.unsqueeze(0).expand(anchors.shape[0], -1, -1)], dim=1
        )  # (B, Q, 3)
    raise ValueError(f"anchors must be (M,3) or (B,M,3), got {tuple(anchors.shape)}")


class TrainHarness:
    """Adapts ``sdfusion_runtime.SDFusionRuntime`` to a gradient-aware train step."""

    def __init__(
        self,
        runtime,
        *,
        mask_reduce: str = "half",
        mask_erode: int = 0,
        t_weight_mode: str = "alphas_cumprod",
        sigma_scale: float = 2.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.runtime = runtime
        self.device = device or torch.device(runtime.device)
        self.mask_reduce = mask_reduce
        self.mask_erode = mask_erode
        self.t_weight_mode = t_weight_mode
        self.cnf = HermiteGaussianCNF.for_resolution(64, sigma_scale=sigma_scale)
        # Frozen by the loader; assert it here as well.
        if any(p.requires_grad for p in self.vqvae_params()):
            raise RuntimeError("VQ-VAE parameters must be frozen before training")

    # --------------------------------------------------------------- params
    def unet_params(self):
        return [p for p in self.runtime.unet.parameters() if p.requires_grad]

    def vqvae_params(self):
        return list(self.runtime.vqvae.parameters())

    # -------------------------------------------------------------- encode
    def encode(self, grid_zyx: torch.Tensor) -> torch.Tensor:
        """Encode an upstream-layout grid; caller wraps in ``no_grad``."""
        return self.runtime.vqvae.encode_no_quant(grid_zyx)

    def encode_target(self, batch: Batch) -> torch.Tensor:
        """Training target latent only - never enters the condition path."""
        with torch.no_grad():
            return self.encode(axes.xyz_to_zyx(batch.target_sdf))

    # ------------------------------------------------------------ diffusion
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.runtime.model.q_sample(x0, t, noise)

    def predict_x0(self, x_t: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sa = extract_into_tensor(self.runtime.model.sqrt_alphas_cumprod, t, x_t.shape)
        s1a = extract_into_tensor(self.runtime.model.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - s1a * eps) / sa

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Official 3D U-Net forward, WITHOUT no_grad (gradients preserved)."""
        return self.runtime.model.apply_model(x_t, t, None)

    def t_weights(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
        a = self.runtime.model.alphas_cumprod
        if self.t_weight_mode == "one":
            return torch.ones(batch_size, device=t.device, dtype=x_dtype(a))
        if self.t_weight_mode == "alphas_cumprod":
            return extract_into_tensor(a, t, (batch_size,))
        raise ValueError(f"unknown t_weight_mode {self.t_weight_mode!r}")

    # ------------------------------------------------------------- decode
    def decode_xyz(self, z0_pred: torch.Tensor) -> torch.Tensor:
        """Differentiable VQ decode (frozen params, straight-through quantizer).

        Returns the decoded SDF in OUR xyz layout; the only axis conversion is
        the single zyx->xyz permutation immediately after upstream decode.
        """
        sdf_zyx = self.runtime.vqvae.decode_no_quant(z0_pred, force_not_quantize=False)
        return axes.zyx_to_xyz(sdf_zyx)

    # --------------------------------------------------------------- step
    def train_step(
        self,
        batch: Batch,
        arm: str,
        t: torch.Tensor,
        noise_obs: torch.Tensor,
        noise_hidden: torch.Tensor,
        query_points: torch.Tensor,
    ) -> Dict[str, object]:
        """One forward (+ loss) for ``arm``; caller performs backward.

        ``t`` has shape ``(B,)``; the two noise tensors have the latent shape
        ``(B, C, l, l, l)``.  Passing them in makes the three arms exactly
        matched on noise and timestep.
        """
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        cond: Conditioning = build_condition(
            self, batch, mask_reduce=self.mask_reduce, mask_erode=self.mask_erode
        )
        z_target = self.encode_target(batch)  # target-only latent (no grad)
        x_t, noise = masked_q_sample(
            cond.z_cond,
            z_target,
            cond.latent_mask,
            t,
            _ScheduleAdapter(self.runtime),
            noise_obs=noise_obs,
            noise_hidden=noise_hidden,
        )

        eps_pred = self.denoise(x_t, t)
        loss_diff = sample_mse(eps_pred, noise).mean()

        z0_pred = self.predict_x0(x_t, eps_pred, t)
        sdf_xyz = self.decode_xyz(z0_pred)  # (B,1,R,R,R) xyz, requires grad
        w = self.t_weights(t, batch.batch_size)

        field = CubicSplineField(sdf_xyz[:, 0])
        query = query_points.to(dtype=sdf_xyz.dtype)
        with torch.no_grad():
            target_field = CubicSplineField(batch.target_sdf[:, 0])
            target_vals = target_field.value(query)  # (B, Q) supervision only

        # ---- shared reconstruction branch --------------------------------
        pred_vals = field.value(query)  # (B, Q)
        recon = (sample_mse(pred_vals, target_vals) * w).mean()
        geometry = recon
        anchor_value_loss = torch.zeros((), device=sdf_xyz.device)
        anchor_normal_loss = torch.zeros((), device=sdf_xyz.device)

        anchor_ops = None
        solve_diag = None
        anchor_value_err = None
        anchor_grad_err = None
        anchor_abs_err = None
        n_anchors_checked = None

        if arm == "soft":
            # Target value is EXACTLY ZERO (surface hit), not interpolated TSDF.
            a_vals, a_grads = field.evaluate(batch.anchors, with_grad=True)
            zero = torch.zeros_like(a_vals)
            anchor_value_loss = (sample_mse(a_vals, zero) * w).mean()
            anchor_normal_loss = (sample_mse(a_grads, batch.normals) * w).mean()
            geometry = geometry + anchor_value_loss + anchor_normal_loss
            anchor_ops = torch.stack(
                [a_vals, a_grads[..., 0], a_grads[..., 1], a_grads[..., 2]], dim=-1
            )
        elif arm == "cnf":
            a_vals, a_grads = field.evaluate(batch.anchors, with_grad=True)
            base_ops = torch.stack(
                [a_vals, a_grads[..., 0], a_grads[..., 1], a_grads[..., 2]], dim=-1
            )
            targets = torch.zeros_like(base_ops)
            targets[..., 1:] = batch.normals.to(dtype=targets.dtype)  # exact 0 + outward normal
            psi, solve_diag = self.cnf.solve(base_ops, targets, batch.anchors, diagnostics=True)
            # Direct FP64 residual check at ALL anchors: evaluate the correction
            # itself (value + xyz derivatives) at the anchor points instead of
            # reusing system_matrix @ psi, so a sign/layout bug cannot hide
            # behind the same Gram matrix used by the solve.  Report max abs.
            corr_val, corr_grad = self.cnf.correction(
                psi, batch.anchors, batch.anchors, with_grad=True
            )  # (B, M), (B, M, 3) in FP64
            corrected_ops = base_ops.to(torch.float64) + torch.stack(
                [corr_val, corr_grad[..., 0], corr_grad[..., 1], corr_grad[..., 2]], dim=-1
            )  # (B, M, 4)
            err = corrected_ops - targets.to(torch.float64)
            anchor_value_err = float(err[..., 0].abs().max())
            anchor_grad_err = float(err[..., 1:4].abs().max())
            anchor_abs_err = float(err.abs().max())
            n_anchors_checked = int(batch.anchors.shape[1])
            pred_ops = flatten_ops(corrected_ops)
            corr, _ = self.cnf.correction(psi, batch.anchors, query, with_grad=False)  # FP64 (B, Q)
            corrected_vals = pred_vals + corr.to(dtype=pred_vals.dtype)
            geometry = (sample_mse(corrected_vals, target_vals) * w).mean()
            anchor_ops = pred_ops

        loss_total = loss_diff + geometry

        # Gradient attributable to the geometry branch itself (independent of
        # the shared denoising loss), measured w.r.t. the decoded SDF grid.
        geom_grad = torch.autograd.grad(geometry, sdf_xyz, retain_graph=True, allow_unused=True)[0]
        geom_grad_norm = float(geom_grad.norm()) if geom_grad is not None else 0.0

        result: Dict[str, object] = {
            "arm": arm,
            "loss_total": float(loss_total),
            "loss_diffusion": float(loss_diff),
            "loss_geometry": float(geometry),
            "loss_reconstruction": float(recon),
            "loss_anchor_value": float(anchor_value_loss),
            "loss_anchor_normal": float(anchor_normal_loss),
            "geometry_grad_norm_sdf": geom_grad_norm,
            "z0_pred_requires_grad": bool(z0_pred.requires_grad),
            "sdf_requires_grad": bool(sdf_xyz.requires_grad),
            "known_voxel_count": cond.stats["known_voxel_count"],
            "latent_observed_count": cond.stats["latent_observed_count"],
            "latent_observed_fraction": cond.stats["latent_observed_fraction"],
            "n_empty_masks": cond.stats["n_empty_masks"],
            "n_query_points": int(query_points.shape[-2]),
            # Tensors needed by validation-only preflight for separate backward
            # and gradient checks; they are not serialized.
            "_tensors": {
                "loss_total": loss_total,
                "loss_geometry": geometry,
                "unet_params": self.unet_params(),
                "vqvae_params": self.vqvae_params(),
            },
        }
        if solve_diag is not None:
            result["cnf_solve"] = solve_diag.as_dict()
        if anchor_value_err is not None:
            result["anchor_value_abs_err_max"] = anchor_value_err
            result["anchor_grad_abs_err_max"] = anchor_grad_err
            result["anchor_abs_err_max"] = anchor_abs_err
            result["n_anchors_checked"] = n_anchors_checked
        if anchor_ops is not None:
            result["anchor_ops_finite"] = bool(torch.isfinite(anchor_ops).all())
        return result


def x_dtype(t: torch.Tensor) -> torch.dtype:
    return t.dtype


class _ScheduleAdapter:
    """Minimal object exposing ``q_sample`` for ``masked_q_sample``."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime

    def q_sample(self, x_start, t, noise):
        return self.runtime.model.q_sample(x_start, t, noise)
