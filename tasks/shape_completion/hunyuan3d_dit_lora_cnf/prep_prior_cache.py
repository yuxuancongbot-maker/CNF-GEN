#!/usr/bin/env python3
"""Prepare clean on-manifold pretrained DiT prior cache (`x1_prior.pt`, `uncond_one.pt`,
`pose_align.pt`, `cond_main.pt`, `train_geo.pt`) for Prior-Preserving DiT LoRA training.

Because the public Hunyuan3D-2mini VAE checkpoint omits `encoder.*` weights, this script avoids
noisy latent inversion by sampling clean pretrained DiT latents `x1_prior` and aligning each
training chair's canonical 32^3 bounding box to GT occupancy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
WARP_DIR = HERE.parent / "hunyuan3d_cnf_warp"
sys.path.insert(0, str(WARP_DIR))

from hy3d_field import HunyuanField  # noqa: E402


def grid_points_np(res: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    ax = np.linspace(lo, hi, res, dtype=np.float32)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)


@torch.no_grad()
def batched_dit_sample(
    hy: HunyuanField,
    cond_b: torch.Tensor,
    uncond_b: torch.Tensor,
    n_steps: int,
    guidance: float,
    seed: int,
) -> torch.Tensor:
    B = cond_b.shape[0]
    device = cond_b.device
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((B, 512, 64), generator=gen, device=device, dtype=torch.float32)
    sigmas = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=torch.float32)
    cond_cat = torch.cat([cond_b, uncond_b], dim=0).half()
    dit = hy.pipe.model
    for k in range(n_steps):
        t_val = sigmas[k]
        dt = sigmas[k + 1] - sigmas[k]
        x_in = torch.cat([x, x], dim=0).half()
        t_in = torch.full((2 * B,), float(t_val), device=device, dtype=torch.float16)
        v_cat = dit(x_in, t_in, {"main": cond_cat}).float()
        v_c, v_u = v_cat.chunk(2, dim=0)
        v = v_u + guidance * (v_c - v_u)
        x = x + dt * v
    return x


def main():
    ap = argparse.ArgumentParser(description="Cache clean pretrained DiT priors and per-chair 32^3 alignment.")
    ap.add_argument("--data-root", type=Path, required=True, help="Directory with train.json and NPZ files.")
    ap.add_argument("--rgba-dir", type=Path, required=True, help="Directory of upright <object_id>.rgba.png images.")
    ap.add_argument("--hy3d-repo", type=Path, required=True, help="Path to Hunyuan3D-2 repository checkout.")
    ap.add_argument("--hy3d-weights", type=Path, required=True, help="Path to Hunyuan3D-2mini weights directory.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output cache directory.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-steps", type=int, default=15)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--n-sup-near", type=int, default=1536)
    ap.add_argument("--n-sup-uni", type=int, default=512)
    args = ap.parse_args()

    t0 = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_entries = json.loads((args.data_root / "train.json").read_text())
    N = len(train_entries)

    hy = HunyuanField(
        repo_dir=args.hy3d_repo,
        weights_dir=args.hy3d_weights,
        steps=args.n_steps,
        guidance=args.guidance,
    )
    hy.vae.eval().half()
    hy.decoder.eval().float()

    xyz64 = grid_points_np(64, -1.0, 1.0)
    n_sup = args.n_sup_near + args.n_sup_uni
    sup_pts = torch.empty((N, n_sup, 3), dtype=torch.float32)
    sup_occ = torch.empty((N, n_sup), dtype=torch.float32)
    anc_all = torch.empty((N, 32, 3), dtype=torch.float32)
    nrm_all = torch.empty((N, 32, 3), dtype=torch.float32)
    cond_main = None
    uncond_one = None

    print(f"[1/2] Encoding DINOv2 conditions & 3D supervision for {N} chairs...", flush=True)
    for i, e in enumerate(train_entries):
        oid = e["object_id"]
        rng = np.random.default_rng(1000 + i)
        rgba = Image.open(args.rgba_dir / f"{oid}.rgba.png").convert("RGBA")
        with torch.no_grad():
            ci = hy.pipe.prepare_image(rgba)
            it = ci.pop("image")
            cond = hy.pipe.encode_cond(
                image=it,
                additional_cond_inputs=ci,
                do_classifier_free_guidance=(i == 0),
                dual_guidance=False,
            )
            cm = cond["main"][0].detach().half().cpu()
            if cond_main is None:
                cond_main = torch.empty((N, *cm.shape), dtype=torch.float16)
                uncond_one = cond["main"][1:2].detach().half().cuda()
            cond_main[i] = cm

        with np.load(args.data_root / e["path"], allow_pickle=False) as z:
            tsdf = np.asarray(z["target_sdf"], dtype=np.float32).reshape(-1)
            anc_all[i] = torch.from_numpy(np.asarray(z["anchors"], dtype=np.float32))
            nrm_all[i] = torch.from_numpy(np.asarray(z["normals"], dtype=np.float32))
        near_idx = np.flatnonzero(np.abs(tsdf) < 0.14)
        if len(near_idx) < args.n_sup_near:
            near_idx = np.flatnonzero(np.abs(tsdf) < 0.195)
        pick = np.concatenate([
            rng.choice(near_idx, size=args.n_sup_near, replace=(len(near_idx) < args.n_sup_near)),
            rng.choice(len(tsdf), size=args.n_sup_uni, replace=False),
        ])
        sup_pts[i] = torch.from_numpy(xyz64[pick])
        sup_occ[i] = torch.from_numpy((tsdf[pick] < 0.0).astype(np.float32))

    xyz32 = torch.from_numpy(grid_points_np(32, -1.0, 1.0)).cuda()
    x1_prior = torch.empty((N, 512, 64), dtype=torch.float16)
    scales = torch.full((N,), 0.85, dtype=torch.float32)
    trans = torch.zeros((N, 3), dtype=torch.float32)

    print(f"[2/2] Sampling {N} clean pretrained DiT priors (B={args.batch_size}, {args.n_steps} steps)...", flush=True)
    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        B = end - start
        cb = cond_main[start:end].cuda()
        ub = uncond_one.expand(B, -1, -1)
        x1_b = batched_dit_sample(hy, cb, ub, args.n_steps, args.guidance, seed=111 + start)
        x1_prior[start:end] = x1_b.half().cpu()

        with torch.no_grad():
            tok_b = hy.vae((x1_b / float(hy.vae.scale_factor)).half()).float()
            for j in range(B):
                idx = start + j
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits32 = hy.decoder(queries=xyz32.unsqueeze(0), latents=tok_b[j : j + 1]).squeeze(0).squeeze(-1).float()
                occ_can = xyz32[logits32 > 0.0]
                gt_occ = sup_pts[idx][sup_occ[idx] > 0.5].cuda()
                if len(occ_can) >= 32 and len(gt_occ) >= 16:
                    c_can = 0.5 * (occ_can.min(0).values + occ_can.max(0).values)
                    c_gt = 0.5 * (gt_occ.min(0).values + gt_occ.max(0).values)
                    ext_can = (occ_can.max(0).values - occ_can.min(0).values).clamp(min=0.1)
                    ext_gt = (gt_occ.max(0).values - gt_occ.min(0).values).clamp(min=0.1)
                    s_est = float(torch.median(ext_gt / ext_can).clamp(0.65, 1.05).item())
                    t_est = (c_gt - s_est * c_can).clamp(-0.35, 0.35)
                    scales[idx] = s_est
                    trans[idx] = t_est.cpu()

    torch.save(cond_main, args.out_dir / "cond_main.pt")
    torch.save(uncond_one.cpu(), args.out_dir / "uncond_one.pt")
    torch.save(x1_prior, args.out_dir / "x1_prior.pt")
    torch.save({"scales": scales, "trans": trans}, args.out_dir / "pose_align.pt")
    torch.save({"sup_pts": sup_pts, "sup_occ": sup_occ, "anchors": anc_all, "normals": nrm_all}, args.out_dir / "train_geo.pt")
    print(f"Saved prior cache to {args.out_dir} in {time.perf_counter() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
