#!/usr/bin/env python3
"""Matched 2-Arm Prior-Preserving Hunyuan3D-2mini DiT LoRA Trainer:
  - `plain_lora`: Prior-Preservation velocity anchoring MSE(v_lora, v_ref) + 10% CFG dropout
                  + Geometry BCE on raw decoded f_0(x).
  - `cnf_lora`  : IDENTICAL Prior-Preservation velocity anchoring + 10% CFG dropout
                  + Geometry BCE on CNF-warped f_0(x + u_det + du) via exact 1-step IFT.

Both arms freeze ShapeVAE, share the exact same RNG stream (b_idx, t, eps, drop_mask),
and use identical hyperparameters (rank=8, lr=3e-5, lambda_geo=0.05, 600 steps, B=4).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
WARP_DIR = HERE.parent / "hunyuan3d_cnf_warp"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WARP_DIR))

from hy3d_field import HunyuanField, decoder_features  # noqa: E402
from lora_ift_core import (  # noqa: E402
    attach_dit_lora,
    disable_lora,
    fast_gpu_normal_warp_det,
    ift_normal_warp_step,
    matern52_torch,
    save_lora_state,
)


def train_one_arm(
    arm: str,
    cache_dir: Path,
    hy3d_repo: Path,
    hy3d_weights: Path,
    out_dir: Path,
    steps: int,
    batch_size: int,
    lr: float,
    lambda_geo: float,
    rank: int,
    cfg_drop: float,
):
    t0 = time.perf_counter()
    x1_prior = torch.load(cache_dir / "x1_prior.pt", weights_only=True)
    cond_main = torch.load(cache_dir / "cond_main.pt", weights_only=True)
    uncond_one = torch.load(cache_dir / "uncond_one.pt", weights_only=True).cuda().to(torch.bfloat16)
    pose = torch.load(cache_dir / "pose_align.pt", weights_only=True)
    scales_all = pose["scales"]
    trans_all = pose["trans"]
    geo = torch.load(cache_dir / "train_geo.pt", weights_only=True)
    sup_pts, sup_occ = geo["sup_pts"], geo["sup_occ"]
    anc_all, nrm_all = geo["anchors"], geo["normals"]
    N = len(x1_prior)

    torch.manual_seed(111)
    np.random.seed(111)
    hy = HunyuanField(repo_dir=hy3d_repo, weights_dir=hy3d_weights, steps=30, guidance=5.0)
    hy.vae.eval().float()
    hy.decoder.eval().float()
    for p in hy.vae.parameters():
        p.requires_grad_(False)
    for p in hy.pipe.conditioner.parameters():
        p.requires_grad_(False)
    hy.model.train().float()
    attached = attach_dit_lora(hy.model, rank=rank, alpha=float(rank))
    lora_params = [p for p in hy.model.parameters() if p.requires_grad]

    w0_sdf = -hy.decoder.output_proj.weight[0].detach().float()
    b_sdf = -float(hy.decoder.output_proj.bias[0].detach().float().item())

    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=1e-2)

    def lr_lambda(st: int) -> float:
        if st < 40:
            return float(st + 1) / 40.0
        prog = float(st - 40) / max(1, steps - 40)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    gen = torch.Generator(device="cuda").manual_seed(111)

    prior_hist, geo_hist, u_hist = [], [], []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        b_idx = torch.randint(0, N, (batch_size,), generator=gen, device="cuda").cpu()
        x1 = x1_prior[b_idx].cuda().float()
        cm = cond_main[b_idx].cuda().to(torch.bfloat16)
        drop_mask = torch.rand((batch_size,), generator=gen, device="cuda") < cfg_drop
        if drop_mask.any():
            cm = torch.where(drop_mask[:, None, None], uncond_one.expand(batch_size, -1, -1), cm)

        eps = torch.randn(x1.shape, generator=gen, device="cuda", dtype=torch.float32)
        t = torch.rand((batch_size,), generator=gen, device="cuda", dtype=torch.float32) * 0.90 + 0.05
        t_view = t.view(batch_size, 1, 1)
        x_t = (1.0 - t_view) * eps + t_view * x1

        with torch.no_grad(), disable_lora(attached), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            v_ref = hy.model(x_t.to(torch.bfloat16), t.to(torch.bfloat16), {"main": cm}).float()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            v_pred = hy.model(x_t.to(torch.bfloat16), t.to(torch.bfloat16), {"main": cm})
        v_pred32 = v_pred.float()
        loss_prior = F.mse_loss(v_pred32, v_ref)

        geo_mask = (t >= 0.45) & (~drop_mask)
        loss_geo = torch.tensor(0.0, device="cuda")
        u_mag = 0.0
        if geo_mask.any() and lambda_geo > 0.0:
            x1_hat = x_t + (1.0 - t_view) * v_pred32
            active_ids = torch.nonzero(geo_mask, as_tuple=False).squeeze(-1)
            geo_losses = []
            for k in active_ids.tolist():
                ci = int(b_idx[k].item())
                wt = float(t[k].item() ** 2)
                s_i = float(scales_all[ci].item())
                t_i = trans_all[ci].cuda()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    tok_k = hy.vae(x1_hat[k : k + 1] / hy.vae.scale_factor)
                xs = sup_pts[ci].cuda()
                yo = sup_occ[ci].cuda()

                if arm == "plain_lora":
                    q_s = (xs - t_i[None, :]) / s_i
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        phi_s = decoder_features(hy.decoder, q_s, tok_k)
                    f_sup = (phi_s.float() @ w0_sdf) + b_sdf
                elif arm == "cnf_lora":
                    anc_k = anc_all[ci].cuda()
                    nrm_k = nrm_all[ci].cuda()
                    tok_det = tok_k.detach()

                    def eval_f0_det(x_world: torch.Tensor) -> torch.Tensor:
                        q_w = (x_world - t_i[None, :]) / s_i
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            phi = decoder_features(hy.decoder, q_w, tok_det)
                        return (phi.float() @ w0_sdf) + b_sdf

                    W_det, u_det, g_dir, gn_safe, K_aa = fast_gpu_normal_warp_det(
                        eval_f0_det, anc_k, nrm_k, n_iters=4
                    )
                    u_mag += float(u_det.norm(dim=1).mean().item())
                    q_a = (anc_k + u_det - t_i[None, :]) / s_i
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        phi_a = decoder_features(hy.decoder, q_a, tok_k)
                    f_a = (phi_a.float() @ w0_sdf) + b_sdf
                    K_sa = matern52_torch(xs, anc_k)
                    u_sup = ift_normal_warp_step(f_a, W_det, g_dir, gn_safe, K_aa, K_sa)
                    q_s = (xs + u_sup - t_i[None, :]) / s_i
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        phi_s = decoder_features(hy.decoder, q_s, tok_k)
                    f_sup = (phi_s.float() @ w0_sdf) + b_sdf
                else:
                    raise ValueError(arm)

                geo_losses.append(wt * F.binary_cross_entropy_with_logits(-f_sup, yo))

            loss_geo = torch.stack(geo_losses).mean()
            if arm == "cnf_lora":
                u_mag /= len(active_ids)

        loss = loss_prior + lambda_geo * loss_geo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        sched.step()

        prior_hist.append(float(loss_prior.item()))
        geo_hist.append(float(loss_geo.item()))
        u_hist.append(u_mag)
        if step in (1, 50, 150, 300, 450, steps):
            print(
                f"  [{arm}] step {step:4d}/{steps} | prior={np.mean(prior_hist[-20:]):.5f} "
                f"| geo={np.mean(geo_hist[-20:]):.4f} | u={np.mean(u_hist[-20:]):.4f} | t={time.perf_counter()-t0:.1f}s",
                flush=True,
            )

    ckpt_path = out_dir / f"{arm}.pt"
    save_lora_state(
        attached,
        ckpt_path,
        {
            "arm": arm,
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "lambda_geo": lambda_geo,
            "rank": rank,
            "cfg_drop": cfg_drop,
            "elapsed_s": time.perf_counter() - t0,
        },
    )
    print(f"Saved {ckpt_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Train matched Prior-Preserving DiT LoRA arms.")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--hy3d-repo", type=Path, required=True)
    ap.add_argument("--hy3d-weights", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["plain_lora", "cnf_lora"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lambda-geo", type=float, default=0.05)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--cfg-drop", type=float, default=0.10)
    args = ap.parse_args()
    for arm in args.arms:
        train_one_arm(
            arm,
            args.cache_dir,
            args.hy3d_repo,
            args.hy3d_weights,
            args.out_dir,
            args.steps,
            args.batch_size,
            args.lr,
            args.lambda_geo,
            args.rank,
            args.cfg_drop,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
