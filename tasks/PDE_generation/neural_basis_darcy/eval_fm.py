#!/usr/bin/env python3
"""Stage 3: Evaluate trained Darcy Neural-Basis CNF checkpoint on 1,000 held-out fields."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from darcy_neural_basis import (
    DarcyFlowModel,
    GRID_N,
    darcy_divergence,
    project_standardized,
    steady_pde_residual_rms,
    true_constrained_lower_bound,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Darcy Neural-Basis CNF checkpoint.")
    p.add_argument("--data", type=Path, default=Path("datasets/pde/darcy.h5"))
    p.add_argument("--ckpt", type=Path, default=Path("outputs/darcy_neural_basis/mlp_pca_std_cnf_fm.pt"))
    p.add_argument("--out", type=Path, default=Path("outputs/darcy_neural_basis/eval_results.json"))
    p.add_argument("--test-start", type=int, default=9000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--euler-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    model = DarcyFlowModel(
        dim=int(ck["N_BASIS"]), cond_dim=int(ck["cond_dim"]), hidden=int(ck["hidden"])
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    P = np.asarray(ck["P"], dtype=np.float64)
    mean64 = np.asarray(ck["mean"], dtype=np.float64)
    std64 = np.asarray(ck["std"], dtype=np.float64)
    H64 = np.asarray(ck["H"], dtype=np.float64)
    ix = np.asarray(ck["constraint_ix"], dtype=np.int64)
    iy = np.asarray(ck["constraint_iy"], dtype=np.int64)
    obs_flat = iy * GRID_N + ix
    unobs_mask = np.ones(GRID_N * GRID_N, dtype=bool)
    unobs_mask[obs_flat] = False

    mean32 = torch.tensor(mean64, dtype=torch.float32, device=device)
    std32 = torch.tensor(std64, dtype=torch.float32, device=device)
    mean64_t = torch.tensor(mean64, dtype=torch.float64, device=device)
    std64_t = torch.tensor(std64, dtype=torch.float64, device=device)
    H64_t = torch.tensor(H64, dtype=torch.float64, device=device)
    H_z_t = torch.tensor(H64 * std64[None, :], dtype=torch.float64, device=device)
    Q_std, R_std = torch.linalg.qr(H_z_t.T)

    with h5py.File(args.data, "r") as f:
        p_test = f["tensor"][args.test_start : args.test_start + args.n_test, 0].astype(np.float64)
        nu_test = f["nu"][args.test_start : args.test_start + args.n_test].astype(np.float64)

    bound = true_constrained_lower_bound(P, p_test.reshape(args.n_test, -1), obs_flat)
    arms = ("noproj", "std_proj", "zero_prior_std_proj")
    metrics = {
        a: {k: [] for k in ("full_rel_mse", "unobs_rel_mse", "obs_max_f32", "solve_max_fp64", "div_norm", "darcy_div", "pde_rel_rms")}
        for a in arms
    }
    pair_a, pair_b = np.triu_indices(args.n_samples, k=1)
    t0 = time.time()

    for i in range(args.n_test):
        p_gt = p_test[i]
        nu_i = nu_test[i]
        gt_flat = p_gt.ravel()
        e_full = float(np.sum(gt_flat ** 2))
        e_unobs = float(np.sum(gt_flat[unobs_mask] ** 2))
        gt_rms_unobs = float(np.sqrt(np.mean(gt_flat[unobs_mask] ** 2)))
        gt_div = darcy_divergence(p_gt, nu_i)
        gt_div_norm = float(np.linalg.norm(gt_div))

        b64 = p_gt[iy, ix]
        y_t = torch.tensor(b64 - H64 @ mean64, dtype=torch.float64, device=device)
        b_t = torch.tensor(b64, dtype=torch.float64, device=device)
        cond = torch.tensor(b64.astype(np.float32), device=device).unsqueeze(0).repeat(args.n_samples, 1)

        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        x0 = torch.randn(args.n_samples, int(ck["N_BASIS"]), generator=gen, device=device)

        with torch.no_grad():
            # Batched Euler for noproj + std_proj
            x_np = x0.clone()
            x_sp = x0.clone()
            for s in range(args.euler_steps):
                t = torch.full((args.n_samples,), s / args.euler_steps, device=device)
                x_np = x_np + model(x_np, t, cond) / args.euler_steps
                x_sp = x_sp + model(x_sp, t, cond) / args.euler_steps
                z_sp, _ = project_standardized(x_sp, y_t, Q_std, R_std, H_z_t)
                x_sp = z_sp.float()
            z_sp, _ = project_standardized(x_sp, y_t, Q_std, R_std, H_z_t)

            # Deterministic zero-prior control (z = 0)
            z_zero, _ = project_standardized(
                torch.zeros(1, int(ck["N_BASIS"]), device=device), y_t, Q_std, R_std, H_z_t
            )
            z_zero = z_zero.repeat(args.n_samples, 1)

        states = {
            "noproj": (x_np, None),
            "std_proj": (z_sp.float(), z_sp),
            "zero_prior_std_proj": (z_zero.float(), z_zero),
        }
        for arm, (z_f32, z_f64) in states.items():
            c_phys = (z_f32 * std32 + mean32).cpu().numpy().astype(np.float64)
            fields = (c_phys @ P).reshape(args.n_samples, GRID_N, GRID_N)
            f_flat = fields.reshape(args.n_samples, -1)
            err2 = (f_flat - gt_flat[None]) ** 2
            full = float(np.mean(err2.sum(axis=1) / e_full))
            unobs = float(np.mean(err2[:, unobs_mask].sum(axis=1) / e_unobs))
            obs_f32 = float(np.mean(np.max(np.abs(c_phys @ H64.T - b64[None]), axis=1)))
            if z_f64 is not None:
                c64 = z_f64 * std64_t + mean64_t
                solve_fp64 = float((c64 @ H64_t.T - b_t[None]).abs().max(dim=1).values.mean().item())
            else:
                solve_fp64 = float("nan")

            unobs_f = f_flat[:, unobs_mask]
            div_rms = float(np.mean(np.sqrt(np.mean((unobs_f[pair_a] - unobs_f[pair_b]) ** 2, axis=1))))
            div_pred = darcy_divergence(fields, nu_i)
            darcy_err = float(np.mean(np.linalg.norm(div_pred - gt_div[None], axis=(1, 2)) / gt_div_norm))
            _, pde_rel = steady_pde_residual_rms(fields, nu_i)

            metrics[arm]["full_rel_mse"].append(full)
            metrics[arm]["unobs_rel_mse"].append(unobs)
            metrics[arm]["obs_max_f32"].append(obs_f32)
            metrics[arm]["solve_max_fp64"].append(solve_fp64)
            metrics[arm]["div_norm"].append(div_rms / gt_rms_unobs)
            metrics[arm]["darcy_div"].append(darcy_err)
            metrics[arm]["pde_rel_rms"].append(float(np.mean(pde_rel)))

        if (i + 1) % 200 == 0 or i + 1 == args.n_test:
            print(
                f"  {i+1}/{args.n_test}  std_proj_unobs={np.mean(metrics['std_proj']['unobs_rel_mse']):.5e}  "
                f"({time.time()-t0:.1f}s)",
                flush=True,
            )

    summary = {
        "basis_type": ck.get("basis_type", "unknown"),
        "true_constrained_lower_bound": bound,
        "arms": {
            arm: {k: float(np.nanmean(v)) for k, v in vals.items()}
            for arm, vals in metrics.items()
        },
        "elapsed_sec": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
