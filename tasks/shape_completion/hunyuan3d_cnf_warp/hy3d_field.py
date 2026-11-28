"""Pretrained Hunyuan3D-2mini ShapeVAE + DiT continuous implicit field interface.

Provides:
  - Optional zero-shot single-view DiT flow-matching token generation
  - FP32 canonical and aligned/warped 128^3 grid evaluation with canonical-box OOB mask
  - Exact FP64 continuous neural-field value & autograd gradient evaluation at anchor points
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from warp_core import (
    CANONICAL_BOX_BOUND,
    CANONICAL_GRID_BOUND,
    DEFAULT_ELL,
    DEFAULT_GAMMA_LIN,
    DEFAULT_GAMMA_TR,
    GRID_RES,
    OUTSIDE_VALUE,
    canonical_oob_mask,
    eval_rkhs_u,
    grad_to_shapenet,
    grid_points,
    sha256_file,
    transform_to_canonical,
)


def decoder_features(decoder: Any, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    single = queries.ndim == 2
    if single:
        queries = queries.unsqueeze(0)
    qe = decoder.query_proj(decoder.fourier_embedder(queries).to(tokens.dtype))
    x = decoder.cross_attn_decoder(qe, tokens)
    if decoder.enable_ln_post:
        x = decoder.ln_post(x)
    return x[0] if single else x


class HunyuanFieldEvaluator:
    """Frozen Hunyuan3D-2mini ShapeVAE decoder (SDF-oriented: inside < 0, outside > 0)."""

    def __init__(
        self,
        hy3d_repo: Path,
        weights_dir: Path,
        steps: int = 30,
        guidance: float = 5.0,
        device: str = "cuda",
    ) -> None:
        self.steps = steps
        self.guidance = guidance
        self.device = device
        repo_str = str(Path(hy3d_repo).resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        cfg = Path(weights_dir) / "hunyuan3d-dit-v2-mini" / "config.yaml"
        ckpt = Path(weights_dir) / "hunyuan3d-dit-v2-mini" / "model.fp16.safetensors"
        vae_ckpt = Path(weights_dir) / "hunyuan3d-vae-v2-mini" / "model.fp16.safetensors"
        self.weight_hashes = {
            "dit_weights_sha256": sha256_file(ckpt),
            "vae_weights_sha256": sha256_file(vae_ckpt) if vae_ckpt.exists() else None,
        }
        self.pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            str(ckpt), str(cfg), device=device, dtype=torch.float16, use_safetensors=True
        )
        self.vae = self.pipe.vae
        self.decoder = self.vae.geo_decoder
        for module in (self.vae, self.pipe.model, self.pipe.conditioner, self.decoder):
            module.eval()
        self.decoder.requires_grad_(False)
        self.decoder.float()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Negate occupancy logit so inside < 0, outside > 0 (standard SDF orientation)
        self.w0_f32 = -self.decoder.output_proj.weight[0].detach().float()
        self.b_f32 = -float(self.decoder.output_proj.bias[0].detach().float().item())
        self.w0_f64 = -self.decoder.output_proj.weight[0].detach().double()
        self.b_f64 = -float(self.decoder.output_proj.bias[0].detach().double().item())

    @torch.inference_mode()
    def generate_tokens(self, rgba_path: Path, seed: int) -> tuple[torch.Tensor, dict[str, float]]:
        from hy3dgen.shapegen.pipelines import retrieve_timesteps

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        image = Image.open(rgba_path).convert("RGBA")
        cond_inputs = self.pipe.prepare_image(image)
        image_tensor = cond_inputs.pop("image")
        cond = self.pipe.encode_cond(
            image=image_tensor,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=True,
            dual_guidance=False,
        )
        sigmas = np.linspace(0, 1, self.steps)
        timesteps, _ = retrieve_timesteps(
            self.pipe.scheduler, self.steps, self.pipe.device, sigmas=sigmas
        )
        latents = self.pipe.prepare_latents(
            1, self.pipe.dtype, self.pipe.device, torch.manual_seed(seed)
        )
        for t in timesteps:
            latent_in = torch.cat([latents] * 2)
            t_in = (
                t.expand(latent_in.shape[0]).to(latents.dtype)
                / self.pipe.scheduler.config.num_train_timesteps
            )
            pred = self.pipe.model(latent_in, t_in, cond)
            pred_c, pred_u = pred.chunk(2)
            pred = pred_u + self.guidance * (pred_c - pred_u)
            latents = self.pipe.scheduler.step(pred, t, latents).prev_sample
        tokens = self.vae(latents / self.vae.scale_factor).float()
        torch.cuda.synchronize()
        self.decoder.float()
        return tokens, {"generation_total_s": time.perf_counter() - t0}

    @torch.inference_mode()
    def canonical_sdf_grid(
        self, tokens32: torch.Tensor, chunk: int = 16384
    ) -> tuple[np.ndarray, float]:
        """Evaluate canonical SDF-oriented field (-logits) on [-1.01, 1.01]^3."""
        self.decoder.float()
        pts, _ = grid_points(GRID_RES, -CANONICAL_GRID_BOUND, CANONICAL_GRID_BOUND, np.float32)
        q = torch.from_numpy(pts).to(self.device)
        out = torch.empty(len(q), device=self.device, dtype=torch.float32)
        t0 = time.perf_counter()
        for st in range(0, len(q), chunk):
            phi = decoder_features(self.decoder, q[st : st + chunk], tokens32)
            out[st : st + chunk] = phi @ self.w0_f32 + self.b_f32
        torch.cuda.synchronize()
        return out.reshape(GRID_RES, GRID_RES, GRID_RES).cpu().numpy(), time.perf_counter() - t0

    def eval_f0_grad_fp64(
        self,
        pts_np: np.ndarray,
        tokens64: torch.Tensor,
        alignment: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact FP64 continuous value f0(x) and gradient grad_x f0(x) via autograd."""
        q_np = transform_to_canonical(pts_np, alignment)
        self.decoder.double()
        try:
            q = (
                torch.from_numpy(np.ascontiguousarray(q_np, dtype=np.float64))
                .to(self.device)
                .requires_grad_(True)
            )
            phi = decoder_features(self.decoder, q, tokens64)
            f = phi @ self.w0_f64 + self.b_f64
            gq = torch.autograd.grad(f.sum(), q)[0].detach().cpu().numpy().astype(np.float64)
            fv = f.detach().cpu().numpy().astype(np.float64)
            gx = grad_to_shapenet(gq, alignment)
        finally:
            self.decoder.float()
        return fv, gx

    @torch.inference_mode()
    def eval_grid_fp32(
        self,
        xyz_full: np.ndarray,
        tokens32: torch.Tensor,
        alignment: dict[str, Any],
        anchors: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        ell: float = DEFAULT_ELL,
        gamma_tr: float = DEFAULT_GAMMA_TR,
        gamma_lin: float = DEFAULT_GAMMA_LIN,
        chunk: int = 16384,
    ) -> np.ndarray:
        """Evaluate aligned or warped field on 128^3 grid with canonical-box OOB masking."""
        self.decoder.float()
        x = xyz_full.astype(np.float64)
        if weights is not None and anchors is not None:
            x = x + eval_rkhs_u(x, anchors, weights, ell, gamma_tr, gamma_lin)
        q_np = transform_to_canonical(x, alignment)
        in_idx = np.flatnonzero(~canonical_oob_mask(q_np, CANONICAL_BOX_BOUND))
        vol = np.full(len(xyz_full), OUTSIDE_VALUE, dtype=np.float32)
        if len(in_idx):
            qt = torch.from_numpy(np.ascontiguousarray(q_np[in_idx], dtype=np.float32)).to(
                self.device
            )
            for st in range(0, len(in_idx), chunk):
                sl = slice(st, st + chunk)
                vol[in_idx[sl]] = (
                    decoder_features(self.decoder, qt[sl], tokens32) @ self.w0_f32 + self.b_f32
                ).cpu().numpy()
        return vol
