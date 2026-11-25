#!/usr/bin/env python3
"""Stage 2: Train matched std_proj CNF Flow Matching generator on a frozen 289-dim basis."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from darcy_neural_basis import (
    DarcyFlowModel,
    GRID_N,
    N_BASIS,
    N_OBS,
    anchor_indices,
    basis_diagnostics,
    fourier_basis,
    orthonormalize_rows,
    project_standardized,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train matched CNF Flow Matching generator on Darcy.")
    p.add_argument("--data", type=Path, default=Path("datasets/pde/darcy.h5"))
    p.add_argument("--basis-type", choices=("mlp_pca", "fourier", "pod"), default="mlp_pca")
    p.add_argument("--basis-npz", type=Path, default=Path("outputs/darcy_neural_basis/mlp_289_pca.npz"))
    p.add_argument("--out", type=Path, default=Path("outputs/darcy_neural_basis"))
    p.add_argument("--n-train", type=int, default=9000)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eta-min", type=float, default=5e-5)
    p.add_argument("--soft-weight", type=float, default=20.0)
    p.add_argument("--geom-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=211)
    return p.parse_args()


def load_or_build_basis(args, u_train_flat: np.ndarray, device: str) -> np.ndarray:
    if args.basis_type == "mlp_pca":
        with np.load(args.basis_npz, allow_pickle=False) as d:
            return np.ascontiguousarray(d["P_pca"], dtype=np.float64)
    if args.basis_type == "fourier":
        return orthonormalize_rows(fourier_basis(GRID_N))
    # POD-289 fit strictly on training fields
    u_t = torch.tensor(u_train_flat, dtype=torch.float64, device=device)
    _, _, vh = torch.linalg.svd(u_t, full_matrices=False)
    P_pod = vh[:N_BASIS].cpu().numpy().astype(np.float64)
    return orthonormalize_rows(P_pod)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with h5py.File(args.data, "r") as f:
        u_train = f["tensor"][: args.n_train, 0].astype(np.float64)
    u_train_flat = u_train.reshape(args.n_train, -1)
    ix, iy = anchor_indices(GRID_N)
    obs_flat = iy * GRID_N + ix

    P = load_or_build_basis(args, u_train_flat, device)
    diag = basis_diagnostics(P, obs_flat)
    print(
        f"[{args.basis_type}] rank(H)={diag['rank_H']}/{N_OBS}, "
        f"cond(H)={diag['cond_H']:.3f}, cond(G)={diag['cond_G']:.3f}",
        flush=True,
    )

    # Exact LS coefficients for orthonormal basis P @ P.T = I
    C = u_train_flat @ P.T
    mean = C.mean(axis=0)
    std = C.std(axis=0).clip(1e-6)
    X_norm = ((C - mean) / std).astype(np.float32)
    b = u_train[:, iy, ix].astype(np.float32)
    energy = np.sum(u_train_flat ** 2, axis=1)

    H = P[:, obs_flat].T
    H_z = H * std[None, :]
    Y = b.astype(np.float64) - H @ mean

    X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
    B_t = torch.tensor(b, dtype=torch.float32, device=device)
    C_t = torch.tensor(C, dtype=torch.float64, device=device)
    E_t = torch.tensor(np.maximum(energy, 1e-30), dtype=torch.float64, device=device)
    mean32 = torch.tensor(mean, dtype=torch.float32, device=device)
    std32 = torch.tensor(std, dtype=torch.float32, device=device)
    mean64 = torch.tensor(mean, dtype=torch.float64, device=device)
    std64 = torch.tensor(std, dtype=torch.float64, device=device)
    H32 = torch.tensor(H, dtype=torch.float32, device=device)
    H_z_t = torch.tensor(H_z, dtype=torch.float64, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float64, device=device)
    Q_std, R_std = torch.linalg.qr(H_z_t.T)

    model = DarcyFlowModel(dim=N_BASIS, cond_dim=N_OBS, hidden=args.hidden).to(device)
    print(f"DarcyFlowModel params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.eta_min)
    gen_idx = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    gen_dev = torch.Generator(device=device).manual_seed(args.seed + 2)

    model.train()
    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, args.n_train, (args.batch,), generator=gen_idx).to(device)
        target = X_t[idx]
        noise = torch.randn(target.shape, generator=gen_dev, device=device)
        t = torch.rand(args.batch, generator=gen_dev, device=device)

        interp = (1.0 - t[:, None]) * noise + t[:, None] * target
        velocity = target - noise
        pred = model(interp, t, B_t[idx])
        flow = F.mse_loss(pred, velocity)
        clean_z = interp + (1.0 - t[:, None]) * pred

        physical = clean_z * std32 + mean32
        soft = F.mse_loss(physical @ H32.T, B_t[idx])

        z_proj, res = project_standardized(clean_z, Y_t[idx], Q_std, R_std, H_z_t)
        c_proj = z_proj * std64 + mean64
        geom = (((c_proj - C_t[idx]) ** 2).sum(dim=1) / E_t[idx]).mean()
        loss = flow + args.soft_weight * soft + args.geom_weight * geom

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 2000 == 0 or step == args.steps - 1:
            print(
                f"[{args.basis_type}] step {step:5d}  flow={flow.item():.5f}  "
                f"soft={soft.item():.3e}  geom={geom.item():.3e}  "
                f"proj_res={res.detach().abs().max().item():.2e}  {time.time()-t0:.1f}s",
                flush=True,
            )

    ckpt_path = args.out / f"{args.basis_type}_std_cnf_fm.pt"
    torch.save(
        {
            "basis_type": args.basis_type,
            "model": model.state_dict(),
            "P": P,
            "mean": mean,
            "std": std,
            "H": H,
            "constraint_ix": ix,
            "constraint_iy": iy,
            "N_BASIS": N_BASIS,
            "cond_dim": N_OBS,
            "hidden": args.hidden,
            "steps": args.steps,
            "seed": args.seed,
        },
        ckpt_path,
    )
    print(f"Saved {ckpt_path} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
