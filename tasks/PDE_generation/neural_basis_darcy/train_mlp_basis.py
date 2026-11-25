#!/usr/bin/env python3
"""Stage 1: Train MLP-289 CNF-aware neural basis and apply train-only PCA alignment."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from darcy_neural_basis import (
    CoordinateBasisMLP,
    GRID_N,
    N_BASIS,
    N_OBS,
    anchor_indices,
    basis_diagnostics,
    exact_cnf_kkt_loss,
    grid_coordinates,
    pca_align_basis,
    true_constrained_lower_bound,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train CNF-aware MLP-289 neural basis on Darcy.")
    p.add_argument("--data", type=Path, default=Path("datasets/pde/darcy.h5"))
    p.add_argument("--out", type=Path, default=Path("outputs/darcy_neural_basis"))
    p.add_argument("--n-train", type=int, default=9000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eta-min", type=float, default=1e-4)
    p.add_argument("--unconstrained-weight", type=float, default=0.1)
    p.add_argument("--cond-weight", type=float, default=1e-4)
    p.add_argument("--gram-weight", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=211)
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with h5py.File(args.data, "r") as f:
        u_train = f["tensor"][: args.n_train, 0].astype(np.float64)
        u_test = f["tensor"][args.n_train : args.n_train + args.n_test, 0].astype(np.float64)

    u_train_flat = u_train.reshape(args.n_train, -1)
    u_test_flat = u_test.reshape(args.n_test, -1)
    ix, iy = anchor_indices(GRID_N)
    obs_flat_np = iy * GRID_N + ix
    obs_flat = torch.tensor(obs_flat_np, dtype=torch.long, device=device)

    grid_xy = torch.tensor(grid_coordinates(GRID_N), dtype=torch.float32, device=device)
    u_gpu = torch.tensor(u_train_flat, dtype=torch.float64, device=device)

    model = CoordinateBasisMLP(n_basis=N_BASIS, hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CoordinateBasisMLP params: {n_params:,} ({n_params/1e6:.3f}M) on {device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.eta_min)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    eye_h = torch.eye(N_OBS, dtype=torch.float64, device=device)
    eye_g = torch.eye(N_BASIS, dtype=torch.float64, device=device)

    model.train()
    t0 = time.time()
    for step in range(args.steps + 1):
        idx = torch.randint(0, args.n_train, (args.batch,), generator=gen).to(device)
        u_batch = u_gpu[idx]
        P = model.eval_normalized_grid(grid_xy)
        H = P[:, obs_flat].T
        constrained, unconstrained, gram = exact_cnf_kkt_loss(P, H, u_batch, obs_flat)

        # Smooth FP64 surrogate for mean_j 1 / (sigma_j(H)^2 + 1e-4)
        chol_h = torch.linalg.cholesky(H @ H.T + 1e-4 * eye_h)
        cond_reg = torch.cholesky_solve(eye_h, chol_h).diagonal().mean()
        gram_reg = ((gram - eye_g) ** 2).sum() / N_BASIS
        loss = (
            constrained
            + args.unconstrained_weight * unconstrained
            + args.cond_weight * cond_reg
            + args.gram_weight * gram_reg
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 200 == 0 or step == args.steps:
            with torch.no_grad():
                diag = basis_diagnostics(P.detach().cpu().numpy(), obs_flat_np)
            print(
                f"step {step:4d}  constrained={constrained.item():.5e}  "
                f"unconstrained={unconstrained.item():.5e}  "
                f"rank(H)={diag['rank_H']}/{N_OBS}  cond(H)={diag['cond_H']:.3f}  "
                f"{time.time()-t0:.1f}s",
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        P_raw = model.eval_normalized_grid(grid_xy).cpu().numpy()

    # Train-only orthogonal PCA alignment (preserves span(P) identically)
    P_pca, U_pca = pca_align_basis(P_raw, u_train_flat)
    diag_pca = basis_diagnostics(P_pca, obs_flat_np)
    bound_test = true_constrained_lower_bound(P_pca, u_test_flat, obs_flat_np)

    print("\n=== Frozen MLP-289-PCA Diagnostic (1000 held-out test fields) ===", flush=True)
    print(f"H: rank={diag_pca['rank_H']}/{N_OBS}, sigma_min={diag_pca['sigma_min_H']:.4f}, cond(H)={diag_pca['cond_H']:.4f}")
    print(
        f"True constrained lower bound: full={bound_test['full_rel_mse']:.6e}  "
        f"unobs={bound_test['unobs_rel_mse']:.6e}  obs_max={bound_test['obs_max_abs']:.3e}",
        flush=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "n_basis": N_BASIS,
            "hidden": args.hidden,
            "steps": args.steps,
            "seed": args.seed,
            "n_params": n_params,
        },
        args.out / "mlp_289_basis.pt",
    )
    np.savez(
        args.out / "mlp_289_pca.npz",
        P_raw=P_raw,
        P_pca=P_pca,
        U_pca=U_pca,
        ix=ix,
        iy=iy,
    )
    summary = {
        "n_params": n_params,
        "diagnostics": diag_pca,
        "true_constrained_lower_bound_test": bound_test,
        "elapsed_sec": time.time() - t0,
    }
    (args.out / "basis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved basis artifacts to {args.out}", flush=True)


if __name__ == "__main__":
    main()
