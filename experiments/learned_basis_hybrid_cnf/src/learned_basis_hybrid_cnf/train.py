from __future__ import annotations

import hashlib
import json
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cnf_motion.soft_field import PandaSoftFlowModel

from .basis import (
    ANCHOR_FRAMES,
    BRAW_MIN_SV_BOUND,
    DEGREE,
    FRAMES,
    HTIME_MIN_SV_BOUND,
    JOINTS,
    MAX_COND_BOUND,
    NONANCHOR_FRAMES,
    S_CENTERS,
    NeuralKernelBasis,
)
from .data import OUT_DEFAULT, prepare_and_save, sha256_array, sha256_file, write_json_atomic
from .projection import hard_project_H

TRAIN_STEPS = 6000
WARMUP_STEPS = 1000
BATCH_SIZE = 64
GEN_LR = 1e-3
GEN_WD = 1e-4
GEN_ETA_MIN = 5e-5
BASIS_LR = 1e-4
BASIS_WD = 1e-5
BASIS_ETA_MIN = 1e-5
GRAD_CLIP_NORM = 1.0
NOISE_SCALE = 0.5
ELL = 0.2
BASIS_INIT_SEED = 100
GEN_INIT_SEED = 0
STREAM_SEED = 1

WEIGHT_FM = 1.0
WEIGHT_SOFT_RAW = 20.0
WEIGHT_REC_NONANCHOR = 1.0
WEIGHT_MOTION = 0.01
WEIGHT_R_NUM = 0.001
WEIGHT_W_FROB = 0.0001


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_dict_sha256(sd: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        h.update(k.encode("utf-8"))
        arr = np.ascontiguousarray(sd[k].detach().cpu().numpy())
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


@torch.no_grad()
def encode_constrained_c1(
    basis: NeuralKernelBasis,
    q: torch.Tensor,
    anchor_frames: list[int] = ANCHOR_FRAMES,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Encode trajectories q [N,17,7] on current orthonormal reference basis Btheta_ref,
    then project onto anchor constraints H c = b to obtain c1 [N,70] in float64.
    """
    q64 = q.to(dtype=torch.float64, device=basis.ref_t.device)
    Btheta_ref, _ = basis.qr_normalize_ref()  # [17, 10], orthonormal columns
    # Least-squares encoding: since Btheta_ref^T Btheta_ref = I_10, c_ls = Btheta_ref^T q
    c_ls_dj = torch.einsum("td,ntj->ndj", Btheta_ref, q64)  # [N, 10, 7]
    c_ls = c_ls_dj.reshape(q64.shape[0], DEGREE * JOINTS)  # [N, 70] with c[d*7+j]
    B_anchors = Btheta_ref[anchor_frames, :]  # [6, 10]
    H = torch.kron(B_anchors, torch.eye(JOINTS, dtype=torch.float64, device=q64.device))  # [42, 70]
    b = q64[:, anchor_frames, :].reshape(q64.shape[0], -1)  # [N, 42]
    c1, info = hard_project_H(H, c_ls, b)
    return c1, c_ls, info


@torch.no_grad()
def evaluate_oracle_reconstruction(basis: NeuralKernelBasis, q_np: np.ndarray) -> dict[str, Any]:
    """Compute unconstrained LS and anchor-constrained c1 oracle reconstruction metrics."""
    q64 = torch.tensor(q_np, dtype=torch.float64, device=basis.ref_t.device)
    Btheta_ref, _ = basis.qr_normalize_ref()
    c1, c_ls, proj_info = encode_constrained_c1(basis, q64)
    q_ls = torch.einsum("td,ndj->ntj", Btheta_ref, c_ls.reshape(-1, DEGREE, JOINTS))
    q_c1 = torch.einsum("td,ndj->ntj", Btheta_ref, c1.reshape(-1, DEGREE, JOINTS))
    err_ls = q_ls - q64
    err_c1 = q_c1 - q64
    return {
        "ls_all_rmse_rad": float(torch.sqrt(torch.mean(err_ls ** 2)).cpu().item()),
        "ls_nonanchor_rmse_rad": float(torch.sqrt(torch.mean(err_ls[:, NONANCHOR_FRAMES, :] ** 2)).cpu().item()),
        "ls_anchor_max_abs_rad": float(err_ls[:, ANCHOR_FRAMES, :].abs().max().cpu().item()),
        "c1_all_rmse_rad": float(torch.sqrt(torch.mean(err_c1 ** 2)).cpu().item()),
        "c1_nonanchor_rmse_rad": float(torch.sqrt(torch.mean(err_c1[:, NONANCHOR_FRAMES, :] ** 2)).cpu().item()),
        "c1_nonanchor_mse": float(torch.mean(err_c1[:, NONANCHOR_FRAMES, :] ** 2).cpu().item()),
        "c1_anchor_max_abs_rad": float(err_c1[:, ANCHOR_FRAMES, :].abs().max().cpu().item()),
        "c1_projection_max_residual": float(proj_info["max_residual"]),
    }


def compute_training_step_losses(
    model: PandaSoftFlowModel,
    basis: NeuralKernelBasis,
    q_b: torch.Tensor,
    mean_f32: torch.Tensor,
    std_f32: torch.Tensor,
    noise_f32: torch.Tensor,
    s_f32: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Single-step forward graph implementing Section 5 of the handoff specification.

    - Target branch encodes q_b with current Btheta_ref, projects to anchors -> c1, and detaches c1.
    - Flow matching runs in float32 on fixed normalized scale u1 = (c1.detach() - mean) / std.
    - Clean estimate u_hat = us + (1 - s) * v_pred -> physical c_hat in float64.
    - L_soft_raw = MSE(Btheta_ref[anchors] c_hat, b) before projection.
    - Differentiable CNF projection c_proj = Pi_H(c_hat) with H = kron(Btheta_ref[anchors], I7) (WITH grad!).
    - Decoded field q_hat = Btheta_ref c_proj -> L_rec_nonanchor and L_motion (17-frame 2nd difference).
    - R_num on Braw_ref [17,10] and Htime [6,10] + ||W||_F^2.
    """
    device = q_b.device
    q64 = q_b.to(dtype=torch.float64)
    Bsz = q64.shape[0]

    # Differentiable basis evaluation on fixed 17-point reference grid
    Braw_ref = basis.Braw_ref()  # [17, 10], float64
    Btheta_ref, R_ref = basis.qr_normalize_ref()  # [17, 10], [10, 10], float64
    B_anchors = Btheta_ref[ANCHOR_FRAMES, :]  # [6, 10], float64
    eye7 = torch.eye(JOINTS, dtype=torch.float64, device=device)
    H = torch.kron(B_anchors, eye7)  # [42, 70], float64

    # 8 (precheck). Numerical stability regularizer R_num on Braw [17,10] & Htime [6,10] + ||W||_F^2
    r_num, w_frob_sq, num_stats = basis.compute_r_num(Braw_ref, Btheta_ref, anchor_frames=ANCHOR_FRAMES)
    if not num_stats["ok"]:
        raise RuntimeError(f"Numerical stability bound violated: {num_stats['reasons']}")

    htime_stats = num_stats["Htime"]
    prechecked_sv = {
        "rank": int(htime_stats["rank"]) * JOINTS,
        "min_sv": float(htime_stats["min_sv"]),
        "max_sv": float(htime_stats["max_sv"]),
        "cond": float(htime_stats["cond"]),
    }

    Q_time, R_time = torch.linalg.qr(B_anchors.T.contiguous(), mode="reduced")  # [10, 6], [6, 6]
    Q_H = torch.kron(Q_time.contiguous(), eye7)  # [70, 42]
    R_H = torch.kron(R_time.contiguous(), eye7)  # [42, 42]

    b_anchor_target = q64[:, ANCHOR_FRAMES, :]  # [B, 6, 7], float64
    b_flat = b_anchor_target.reshape(Bsz, -1)  # [B, 42], float64

    # 1. Target branch: encode on current orthonormal basis, project to anchors, and DETACH c1
    with torch.no_grad():
        Btheta_det = Btheta_ref.detach()
        H_det = H.detach()
        c_ls_dj = torch.einsum("td,btj->bdj", Btheta_det, q64)  # [B, 10, 7]
        c_ls = c_ls_dj.reshape(Bsz, DEGREE * JOINTS)  # [B, 70]
        c1, target_proj_info = hard_project_H(
            H_det,
            c_ls,
            b_flat,
            precomputed_qr=(Q_H.detach(), R_H.detach()),
            prechecked_sv=prechecked_sv,
        )
        c1_det_f32 = c1.detach().to(dtype=torch.float32)
        u1 = (c1_det_f32 - mean_f32) / std_f32  # [B, 70], float32

    # 2. Flow Matching in float32
    us = (1.0 - s_f32[:, None]) * noise_f32 + s_f32[:, None] * u1  # [B, 70], float32
    v_target = u1 - noise_f32  # [B, 70], float32
    q0_f32 = q64[:, 0, :].to(dtype=torch.float32)  # [B, 7], float32
    anchors_f32 = b_flat.to(dtype=torch.float32)  # [B, 42], float32

    v_pred = model(us, s_f32, q0_f32, anchors_f32)  # [B, 70], float32
    loss_fm = F.mse_loss(v_pred, v_target)

    # 3. Clean estimate u_hat and physical coefficients c_hat in float64
    u_hat = us + (1.0 - s_f32[:, None]) * v_pred  # [B, 70], float32
    c_hat = (u_hat * std_f32 + mean_f32).to(dtype=torch.float64)  # [B, 70], float64

    # 4. Pre-projection soft anchor loss: MSE(Btheta[anchors] c_hat, b)
    c_hat_dj = c_hat.reshape(Bsz, DEGREE, JOINTS)  # [B, 10, 7]
    q_anchor_raw = torch.einsum("ad,bdj->baj", B_anchors, c_hat_dj)  # [B, 6, 7]
    loss_soft_raw = F.mse_loss(q_anchor_raw, b_anchor_target)

    # 5. Differentiable CNF hard projection Pi_H(c_hat) and decoded trajectory q_hat
    c_proj, proj_info = hard_project_H(
        H,
        c_hat,
        b_flat,
        precomputed_qr=(Q_H, R_H),
        prechecked_sv=prechecked_sv,
    )  # [B, 70], float64 (differentiable wrt H and c_hat)
    c_proj_dj = c_proj.reshape(Bsz, DEGREE, JOINTS)  # [B, 10, 7]
    q_hat = torch.einsum("td,bdj->btj", Btheta_ref, c_proj_dj)  # [B, 17, 7], float64

    # 6. Post-projection non-anchor trajectory MSE (L_rec_nonanchor)
    loss_rec_nonanchor = F.mse_loss(q_hat[:, NONANCHOR_FRAMES, :], q64[:, NONANCHOR_FRAMES, :])

    # 7. Fixed 17-frame second-difference MSE (L_motion, discrete normalized-grid 2nd diff, not physical m/s^2)
    d2_q_hat = q_hat[:, 2:, :] - 2.0 * q_hat[:, 1:-1, :] + q_hat[:, :-2, :]
    d2_q_gt = q64[:, 2:, :] - 2.0 * q64[:, 1:-1, :] + q64[:, :-2, :]
    loss_motion = F.mse_loss(d2_q_hat, d2_q_gt)

    total_loss = (
        WEIGHT_FM * loss_fm.to(dtype=torch.float64)
        + WEIGHT_SOFT_RAW * loss_soft_raw
        + WEIGHT_REC_NONANCHOR * loss_rec_nonanchor
        + WEIGHT_MOTION * loss_motion
        + WEIGHT_R_NUM * r_num
        + WEIGHT_W_FROB * w_frob_sq
    )
    if not torch.isfinite(total_loss):
        raise RuntimeError(f"Non-finite total loss: {float(total_loss.detach().cpu().item())}")

    with torch.no_grad():
        post_anchor_err = float((q_hat[:, ANCHOR_FRAMES, :] - b_anchor_target).abs().max().cpu().item())

    details = {
        "loss_total": float(total_loss.detach().cpu().item()),
        "loss_fm": float(loss_fm.detach().cpu().item()),
        "loss_soft_raw": float(loss_soft_raw.detach().cpu().item()),
        "loss_rec_nonanchor": float(loss_rec_nonanchor.detach().cpu().item()),
        "loss_motion": float(loss_motion.detach().cpu().item()),
        "r_num": float(r_num.detach().cpu().item()),
        "w_frob_sq": float(w_frob_sq.detach().cpu().item()),
        "proj_max_residual": float(proj_info["max_residual"]),
        "proj_max_abs_residual": float(proj_info["max_abs_residual"]),
        "target_proj_max_residual": float(target_proj_info["max_residual"]),
        "post_proj_anchor_max_abs_rad": post_anchor_err,
        "num_stats": num_stats,
    }
    return total_loss, details


def build_initial_basis_and_generator(device: torch.device) -> tuple[NeuralKernelBasis, PandaSoftFlowModel, dict[str, str]]:
    """Initialize basis with independent RNG (BASIS_INIT_SEED) and generator with GEN_INIT_SEED=0."""
    set_deterministic_seed(BASIS_INIT_SEED)
    basis = NeuralKernelBasis(ell=ELL).to(device)
    basis_init_hash = state_dict_sha256(basis.state_dict())

    set_deterministic_seed(GEN_INIT_SEED)
    model = PandaSoftFlowModel(
        joints=JOINTS,
        degree=DEGREE,
        anchors=len(ANCHOR_FRAMES),
        hidden=256,
    ).to(device)
    psi_init_hash = state_dict_sha256(model.state_dict())
    return basis, model, {"basis_init_sha256": basis_init_hash, "psi_init_sha256": psi_init_hash}


@torch.no_grad()
def compute_fixed_initial_norm_stats(
    basis: NeuralKernelBasis,
    q_train_np: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compute scalar mean and std from initial-basis constrained c1 on the 128 v2 training trajectories."""
    q_train = torch.tensor(q_train_np, dtype=torch.float64, device=device)
    c1_init, c_ls_init, proj_info = encode_constrained_c1(basis, q_train)
    mean_val = float(c1_init.mean().cpu().item())
    std_val = float(c1_init.std().clamp_min(1e-6).cpu().item())
    mean_t = torch.tensor(mean_val, dtype=torch.float32, device=device)
    std_t = torch.tensor(std_val, dtype=torch.float32, device=device)
    stats = {
        "source": "initial_basis_W_zero_v2_train128_constrained_c1_scalar_mean_std",
        "mean": mean_val,
        "std": std_val,
        "c1_init_shape": list(c1_init.shape),
        "c1_init_sha256": sha256_array(c1_init.detach().cpu().numpy()),
        "c_ls_init_sha256": sha256_array(c_ls_init.detach().cpu().numpy()),
        "c1_init_projection_max_residual": float(proj_info["max_residual"]),
    }
    return mean_t, std_t, stats


def update_status_train(out_dir: Path, stage: str, **extra: Any) -> dict[str, Any]:
    st = {
        "stage": stage,
        "time": utcnow(),
        "pid": os.getpid(),
        **extra,
    }
    write_json_atomic(out_dir / "status_train.json", st)
    return st


def train_group(
    group: str,
    q_train_np: np.ndarray,
    q_eval_np: np.ndarray,
    out_dir: Path,
    device: torch.device,
    steps: int = TRAIN_STEPS,
    warmup_steps: int = WARMUP_STEPS,
    log_every: int = 100,
) -> dict[str, Any]:
    if group not in ("fixed", "learned"):
        raise ValueError(f"group must be 'fixed' or 'learned', got {group}")

    train_dir = out_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    log_path = train_dir / f"train_log_{group}.jsonl"
    if log_path.exists():
        log_path.unlink()

    basis, model, init_hashes = build_initial_basis_and_generator(device)
    init_basis_sd = {k: v.detach().cpu().clone() for k, v in basis.state_dict().items()}
    init_model_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    mean_f32, std_f32, norm_stats = compute_fixed_initial_norm_stats(basis, q_train_np, device)
    init_diag = basis.diagnostics()
    init_oracle_train = evaluate_oracle_reconstruction(basis, q_train_np)
    init_oracle_eval = evaluate_oracle_reconstruction(basis, q_eval_np)

    q_train_d = torch.tensor(q_train_np, dtype=torch.float64, device=device)
    num_train = q_train_d.shape[0]

    # Single generator optimizer and its scheduler
    opt_gen = torch.optim.AdamW(model.parameters(), lr=GEN_LR, weight_decay=GEN_WD)
    sched_gen = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gen, T_max=steps, eta_min=GEN_ETA_MIN)

    # Basis optimizer and scheduler (only stepped in 'learned' after warmup_steps)
    opt_basis = torch.optim.AdamW(basis.parameters(), lr=BASIS_LR, weight_decay=BASIS_WD) if group == "learned" else None
    sched_basis = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt_basis, T_max=max(1, steps - warmup_steps), eta_min=BASIS_ETA_MIN)
        if opt_basis is not None
        else None
    )

    # Paired random stream for batch indices, flow noise, and flow time s
    stream_gen = torch.Generator(device="cpu").manual_seed(STREAM_SEED)

    model.train()
    basis.train()

    t0 = time.time()
    history_summary: list[dict[str, Any]] = []
    max_w_grad_norm = 0.0
    max_hnet_grad_norm = 0.0
    nonzero_w_grad_steps = 0
    nonzero_hnet_grad_steps = 0

    for step in range(steps):
        basis_active = (group == "learned") and (step >= warmup_steps)
        for p in basis.parameters():
            p.requires_grad_(basis_active)

        idx = torch.randint(0, num_train, (BATCH_SIZE,), generator=stream_gen)
        noise_cpu = torch.randn((BATCH_SIZE, DEGREE * JOINTS), generator=stream_gen, dtype=torch.float32) * NOISE_SCALE
        s_cpu = torch.rand((BATCH_SIZE,), generator=stream_gen, dtype=torch.float32)

        q_b = q_train_d[idx.to(device)]
        noise_f32 = noise_cpu.to(device)
        s_f32 = s_cpu.to(device)

        opt_gen.zero_grad(set_to_none=True)
        if opt_basis is not None:
            opt_basis.zero_grad(set_to_none=True)

        try:
            total_loss, details = compute_training_step_losses(
                model=model,
                basis=basis,
                q_b=q_b,
                mean_f32=mean_f32,
                std_f32=std_f32,
                noise_f32=noise_f32,
                s_f32=s_f32,
            )
            total_loss.backward()
        except Exception as e:
            fail_evidence = {
                "group": group,
                "step": int(step + 1),
                "basis_active": basis_active,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "diagnostics": basis.diagnostics(),
                "time": utcnow(),
            }
            fail_path = train_dir / f"numerical_failure_{group}_step{step + 1}.json"
            write_json_atomic(fail_path, fail_evidence)
            update_status_train(
                out_dir,
                "failed_numerical",
                group=group,
                step=int(step + 1),
                fail_evidence_path=str(fail_path),
                error=str(e),
            )
            raise

        gen_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM).detach().cpu().item())
        if not np.isfinite(gen_grad_norm):
            raise RuntimeError(f"Non-finite generator gradient norm at step {step + 1} in group {group}")

        w_grad_norm = 0.0
        hnet_grad_norm = 0.0
        basis_grad_norm = 0.0
        if basis_active and opt_basis is not None and sched_basis is not None:
            if basis.W.grad is not None:
                w_grad_norm = float(basis.W.grad.detach().norm().cpu().item())
            h_grads = [p.grad.detach().norm() ** 2 for p in basis.hnet.parameters() if p.grad is not None]
            if h_grads:
                hnet_grad_norm = float(torch.sqrt(torch.stack(h_grads).sum()).cpu().item())
            basis_grad_norm = float(torch.nn.utils.clip_grad_norm_(basis.parameters(), GRAD_CLIP_NORM).detach().cpu().item())
            if not (np.isfinite(w_grad_norm) and np.isfinite(hnet_grad_norm) and np.isfinite(basis_grad_norm)):
                raise RuntimeError(f"Non-finite basis gradient norm at step {step + 1} in group {group}")
            if w_grad_norm > 0.0:
                nonzero_w_grad_steps += 1
            if hnet_grad_norm > 0.0:
                nonzero_hnet_grad_steps += 1
            max_w_grad_norm = max(max_w_grad_norm, w_grad_norm)
            max_hnet_grad_norm = max(max_hnet_grad_norm, hnet_grad_norm)
            opt_basis.step()
            sched_basis.step()

        opt_gen.step()
        sched_gen.step()

        step_num = step + 1
        should_log = (
            step_num == 1
            or step_num == warmup_steps
            or step_num == warmup_steps + 1
            or step_num == warmup_steps + 2
            or step_num % log_every == 0
            or step_num == steps
        )
        if should_log:
            with torch.no_grad():
                w_frob = float(torch.norm(basis.W.detach()).cpu().item())
                hnet_delta_l2 = float(
                    torch.sqrt(
                        sum(
                            torch.sum((v.detach().cpu() - init_basis_sd[k]) ** 2)
                            for k, v in basis.state_dict().items()
                            if k.startswith("hnet.")
                        )
                    ).item()
                )
            rec = {
                "step": int(step_num),
                "group": group,
                "basis_active": bool(basis_active),
                "lr_gen": float(opt_gen.param_groups[0]["lr"]),
                "lr_basis": float(opt_basis.param_groups[0]["lr"]) if (basis_active and opt_basis is not None) else 0.0,
                "loss_total": details["loss_total"],
                "loss_fm": details["loss_fm"],
                "loss_soft_raw": details["loss_soft_raw"],
                "loss_rec_nonanchor": details["loss_rec_nonanchor"],
                "loss_motion": details["loss_motion"],
                "r_num": details["r_num"],
                "w_frob_sq": details["w_frob_sq"],
                "w_frob": w_frob,
                "hnet_param_delta_l2": hnet_delta_l2,
                "gen_grad_norm": gen_grad_norm,
                "basis_grad_norm": basis_grad_norm,
                "w_grad_norm": w_grad_norm,
                "hnet_grad_norm": hnet_grad_norm,
                "Braw_cond": details["num_stats"]["Braw"]["cond"],
                "Braw_min_sv": details["num_stats"]["Braw"]["min_sv"],
                "Htime_cond": details["num_stats"]["Htime"]["cond"],
                "Htime_min_sv": details["num_stats"]["Htime"]["min_sv"],
                "proj_max_residual": details["proj_max_residual"],
                "post_proj_anchor_max_abs_rad": details["post_proj_anchor_max_abs_rad"],
                "elapsed_sec": round(time.time() - t0, 3),
                "time": utcnow(),
            }
            append_jsonl(log_path, rec)
            history_summary.append(rec)
            if step_num % 500 == 0 or step_num == 1 or step_num == steps:
                update_status_train(
                    out_dir,
                    "training",
                    group=group,
                    step=int(step_num),
                    total_steps=int(steps),
                    loss_total=rec["loss_total"],
                    loss_fm=rec["loss_fm"],
                    loss_rec_nonanchor=rec["loss_rec_nonanchor"],
                    Braw_cond=rec["Braw_cond"],
                    Htime_cond=rec["Htime_cond"],
                    w_frob=rec["w_frob"],
                    elapsed_sec=rec["elapsed_sec"],
                )

    model.eval()
    basis.eval()
    final_diag = basis.diagnostics()
    if not final_diag["ok"]:
        raise RuntimeError(f"Final basis diagnostics failed for group {group}: {final_diag['reasons']}")

    final_oracle_train = evaluate_oracle_reconstruction(basis, q_train_np)
    final_oracle_eval = evaluate_oracle_reconstruction(basis, q_eval_np)

    final_basis_sd = {k: v.detach().cpu() for k, v in basis.state_dict().items()}
    final_model_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    w_param_delta_l2 = float(torch.norm(final_basis_sd["W"] - init_basis_sd["W"]).item())
    hnet_param_delta_l2 = float(
        torch.sqrt(
            sum(
                torch.sum((final_basis_sd[k] - init_basis_sd[k]) ** 2)
                for k in final_basis_sd
                if k.startswith("hnet.")
            )
        ).item()
    )
    model_param_delta_l2 = float(
        torch.sqrt(sum(torch.sum((final_model_sd[k] - init_model_sd[k]) ** 2) for k in final_model_sd)).item()
    )

    if group == "fixed":
        if w_param_delta_l2 != 0.0 or hnet_param_delta_l2 != 0.0:
            raise RuntimeError(f"Fixed group basis changed! w_delta={w_param_delta_l2}, hnet_delta={hnet_param_delta_l2}")
    else:
        if w_param_delta_l2 <= 0.0 or hnet_param_delta_l2 <= 0.0 or nonzero_w_grad_steps == 0 or nonzero_hnet_grad_steps == 0:
            raise RuntimeError(
                f"Learned group basis did not update properly: w_delta={w_param_delta_l2}, hnet_delta={hnet_param_delta_l2}, "
                f"nonzero_w_steps={nonzero_w_grad_steps}, nonzero_hnet_steps={nonzero_hnet_grad_steps}"
            )

    training_config = {
        "protocol_name": "learned_basis_hybrid_cnf_20261114",
        "group": group,
        "ell": ELL,
        "s_centers": [float(x) for x in S_CENTERS.tolist()],
        "frames": FRAMES,
        "degree": DEGREE,
        "joints": JOINTS,
        "hidden": 256,
        "anchor_frames": list(ANCHOR_FRAMES),
        "nonanchor_frames": list(NONANCHOR_FRAMES),
        "steps": int(steps),
        "warmup_steps": int(warmup_steps),
        "batch": BATCH_SIZE,
        "gen_lr": GEN_LR,
        "gen_weight_decay": GEN_WD,
        "gen_eta_min": GEN_ETA_MIN,
        "basis_lr": BASIS_LR if group == "learned" else 0.0,
        "basis_weight_decay": BASIS_WD if group == "learned" else 0.0,
        "basis_eta_min": BASIS_ETA_MIN if group == "learned" else 0.0,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "noise_scale": NOISE_SCALE,
        "basis_init_seed": BASIS_INIT_SEED,
        "train_seed": GEN_INIT_SEED,
        "stream_seed": STREAM_SEED,
        "loss_weights": {
            "L_FM": WEIGHT_FM,
            "L_soft_raw": WEIGHT_SOFT_RAW,
            "L_rec_nonanchor": WEIGHT_REC_NONANCHOR,
            "L_motion": WEIGHT_MOTION,
            "R_num": WEIGHT_R_NUM,
            "W_frob_sq": WEIGHT_W_FROB,
        },
        "motion_loss_note": "MSE(delta^2 q_hat, delta^2 q_train) on fixed 17-frame discrete second differences; not unscaled physical m/s^2 acceleration.",
        "numerical_bounds": {
            "Braw_min_sv_gt": BRAW_MIN_SV_BOUND,
            "Htime_min_sv_gt": HTIME_MIN_SV_BOUND,
            "cond_le": MAX_COND_BOUND,
        },
        "normalization": norm_stats,
        "init_hashes": init_hashes,
        "final_hashes": {
            "basis_final_sha256": state_dict_sha256(final_basis_sd),
            "psi_final_sha256": state_dict_sha256(final_model_sd),
        },
        "param_changes": {
            "w_param_delta_l2": w_param_delta_l2,
            "hnet_param_delta_l2": hnet_param_delta_l2,
            "model_param_delta_l2": model_param_delta_l2,
            "nonzero_w_grad_steps": int(nonzero_w_grad_steps),
            "nonzero_hnet_grad_steps": int(nonzero_hnet_grad_steps),
            "max_w_grad_norm": max_w_grad_norm,
            "max_hnet_grad_norm": max_hnet_grad_norm,
        },
        "diagnostics_initial": init_diag,
        "diagnostics_final": final_diag,
        "oracle_reconstruction_initial": {
            "train_v2": init_oracle_train,
            "eval_v2": init_oracle_eval,
        },
        "oracle_reconstruction_final": {
            "train_v2": final_oracle_train,
            "eval_v2": final_oracle_eval,
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }

    ckpt = {
        "schema_version": 1,
        "group": group,
        "model": final_model_sd,
        "basis": final_basis_sd,
        "mean": mean_f32.detach().cpu(),
        "std": std_f32.detach().cpu(),
        "degree10": DEGREE,
        "joints7": JOINTS,
        "hidden256": 256,
        "degree": DEGREE,
        "joints": JOINTS,
        "hidden": 256,
        "frames": FRAMES,
        "anchor_frames": list(ANCHOR_FRAMES),
        "train_seed0": GEN_INIT_SEED,
        "train_seed": GEN_INIT_SEED,
        "steps6000": int(steps),
        "steps": int(steps),
        "training_config": training_config,
    }

    ckpt_path = train_dir / f"{group}.pt"
    tmp_ckpt = ckpt_path.with_suffix(".pt.tmp")
    torch.save(ckpt, tmp_ckpt)
    tmp_ckpt.replace(ckpt_path)

    summary = {
        "group": group,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "training_config": training_config,
        "final_step_record": history_summary[-1] if history_summary else None,
    }
    write_json_atomic(train_dir / f"summary_{group}.json", summary)
    return summary


def verify_saved_checkpoint(ckpt_path: Path, expected_group: str) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    required_keys = [
        "schema_version",
        "group",
        "model",
        "basis",
        "mean",
        "std",
        "degree10",
        "joints7",
        "hidden256",
        "anchor_frames",
        "train_seed0",
        "steps6000",
        "training_config",
    ]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        raise KeyError(f"Checkpoint {ckpt_path} missing required keys: {missing}")
    if ckpt["schema_version"] != 1 or ckpt["group"] != expected_group:
        raise ValueError(f"Checkpoint header mismatch in {ckpt_path}: schema={ckpt['schema_version']}, group={ckpt['group']}")

    basis = NeuralKernelBasis(ell=float(ckpt["training_config"].get("ell", ELL)))
    basis.load_state_dict(ckpt["basis"], strict=True)
    basis.eval()
    diag = basis.diagnostics()
    if not diag["ok"]:
        raise RuntimeError(f"Reloaded checkpoint basis failed diagnostics: {diag['reasons']}")

    model = PandaSoftFlowModel(
        joints=int(ckpt["joints7"]),
        degree=int(ckpt["degree10"]),
        anchors=len(ckpt["anchor_frames"]),
        hidden=int(ckpt["hidden256"]),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return {
        "path": str(ckpt_path),
        "sha256": sha256_file(ckpt_path),
        "group": ckpt["group"],
        "mean": float(ckpt["mean"].item()),
        "std": float(ckpt["std"].item()),
        "diagnostics": diag,
    }


def run_all_training(out_dir: Path | None = None, device_str: str | None = None, force: bool = False) -> dict[str, Any]:
    if out_dir is None:
        out_dir = OUT_DEFAULT
    out_dir = Path(out_dir)
    train_dir = out_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Refuse to silently overwrite existing training artifacts; use --force to
    # retrain into the same output directory, or use a fresh CNF_V2_OUT.
    existing = [
        str(p) for p in (train_dir / "fixed.pt", train_dir / "learned.pt", train_dir / "train_completion.json")
        if p.exists()
    ]
    if existing and not force:
        raise RuntimeError(
            "Training output already exists: " + ", ".join(existing) + ". "
            "Set a fresh CNF_V2_OUT or pass --force to overwrite."
        )

    if device_str is None:
        device_str = "cpu"
    device = torch.device(device_str)

    update_status_train(out_dir, "preparing_data", device=str(device))
    data_meta = prepare_and_save(out_dir / "data", force=False)
    q_train_np = np.load(out_dir / "data" / "train_q_v2.npz", allow_pickle=True)["q"].astype(np.float64)
    q_eval_np = np.load(out_dir / "data" / "eval_q_v2.npz", allow_pickle=True)["q"].astype(np.float64)

    # Check initial pairing before training
    b_fix, m_fix, h_fix = build_initial_basis_and_generator(device)
    b_lrn, m_lrn, h_lrn = build_initial_basis_and_generator(device)
    if h_fix != h_lrn:
        raise RuntimeError(f"Initial hashes differ between fixed and learned: {h_fix} vs {h_lrn}")
    del b_fix, m_fix, b_lrn, m_lrn

    protocol_obj = {
        "protocol_name": "learned_basis_hybrid_cnf_training_20261114",
        "created_at": utcnow(),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.type == "cuda" else None,
        "data_meta": {k: v for k, v in data_meta.items() if not isinstance(v, dict)},
        "basis_spec": {
            "formula": "Braw_theta(t) = Bleg_norm(t) + K_theta(t, S) W_theta; B_theta(t) = Braw_theta(t) R_theta^{-1}",
            "degree": DEGREE,
            "frames": FRAMES,
            "joints": JOINTS,
            "anchor_frames": list(ANCHOR_FRAMES),
            "nonanchor_frames": list(NONANCHOR_FRAMES),
            "s_centers": [float(x) for x in S_CENTERS.tolist()],
            "ell": ELL,
            "phi_theta": "[t, 0.1 * h_theta(t)], h_theta: 1->32->32->8 Tanh MLP (random init), W_theta: 10x10 zero init",
            "qr_rule": "Reduced QR on fixed 17-point reference grid with positive R diagonal (Qnew=Q*s[None,:], Rnew=s[:,None]*R); query via solve_triangular(R.T, Braw(t).T, upper=False).T",
        },
        "projection_spec": {
            "formula": "c* = c + Q solve_triangular(R^T, (b - c H^T)^T, upper=False)^T where H^T = Q R in float64",
            "ridge_eps": 0.0,
            "silent_fallback": False,
        },
        "training_spec": {
            "steps": TRAIN_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "batch": BATCH_SIZE,
            "generator_optimizer": {"type": "AdamW", "lr": GEN_LR, "weight_decay": GEN_WD, "scheduler": "CosineAnnealingLR", "eta_min": GEN_ETA_MIN},
            "basis_optimizer": {"type": "AdamW", "lr": BASIS_LR, "weight_decay": BASIS_WD, "scheduler": "CosineAnnealingLR_post_warmup", "eta_min": BASIS_ETA_MIN},
            "grad_clip_norm": GRAD_CLIP_NORM,
            "noise_scale": NOISE_SCALE,
            "seeds": {"basis_init_seed": BASIS_INIT_SEED, "gen_init_seed": GEN_INIT_SEED, "stream_seed": STREAM_SEED},
            "initial_hashes": h_fix,
            "loss_weights": {
                "L_FM": WEIGHT_FM,
                "L_soft_raw": WEIGHT_SOFT_RAW,
                "L_rec_nonanchor": WEIGHT_REC_NONANCHOR,
                "L_motion": WEIGHT_MOTION,
                "R_num": WEIGHT_R_NUM,
                "W_frob_sq": WEIGHT_W_FROB,
            },
            "numerical_failure_bounds": {
                "Braw_min_sv_le": BRAW_MIN_SV_BOUND,
                "Htime_min_sv_le": HTIME_MIN_SV_BOUND,
                "cond_gt": MAX_COND_BOUND,
            },
        },
        "source_hashes": {
            "basis_py": sha256_file(Path(__file__).resolve().parent / "basis.py"),
            "projection_py": sha256_file(Path(__file__).resolve().parent / "projection.py"),
            "data_py": sha256_file(Path(__file__).resolve().parent / "data.py"),
            "train_py": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json_atomic(train_dir / "training_protocol.json", protocol_obj)
    write_json_atomic(out_dir / "training_protocol.json", protocol_obj)

    update_status_train(out_dir, "training_fixed_start", initial_hashes=h_fix)
    summary_fixed = train_group("fixed", q_train_np, q_eval_np, out_dir, device)
    ver_fixed = verify_saved_checkpoint(train_dir / "fixed.pt", "fixed")

    update_status_train(out_dir, "training_learned_start", fixed_ckpt_sha256=ver_fixed["sha256"])
    summary_learned = train_group("learned", q_train_np, q_eval_np, out_dir, device)
    ver_learned = verify_saved_checkpoint(train_dir / "learned.pt", "learned")

    # Verify warmup and initial pairing invariants
    if summary_fixed["training_config"]["init_hashes"] != summary_learned["training_config"]["init_hashes"]:
        raise RuntimeError("Initial psi/basis hashes mismatch between fixed and learned runs!")
    if summary_fixed["training_config"]["normalization"]["mean"] != summary_learned["training_config"]["normalization"]["mean"]:
        raise RuntimeError("Fixed normalization mean mismatch between fixed and learned runs!")

    completion = {
        "status": "completed",
        "completed_at": utcnow(),
        "device": str(device),
        "protocol_path": str(train_dir / "training_protocol.json"),
        "data": {
            "train_q_v2_npz": str(out_dir / "data" / "train_q_v2.npz"),
            "eval_q_v2_npz": str(out_dir / "data" / "eval_q_v2.npz"),
            "train_q_v2_sha256": data_meta["train_q_v2_sha256"],
            "eval_q_v2_sha256": data_meta["eval_q_v2_sha256"],
        },
        "checkpoints": {
            "fixed": ver_fixed,
            "learned": ver_learned,
        },
        "pairing_verification": {
            "initial_psi_sha256_equal": (
                summary_fixed["training_config"]["init_hashes"]["psi_init_sha256"]
                == summary_learned["training_config"]["init_hashes"]["psi_init_sha256"]
            ),
            "initial_basis_sha256_equal": (
                summary_fixed["training_config"]["init_hashes"]["basis_init_sha256"]
                == summary_learned["training_config"]["init_hashes"]["basis_init_sha256"]
            ),
            "initial_psi_sha256": summary_fixed["training_config"]["init_hashes"]["psi_init_sha256"],
            "initial_basis_sha256": summary_fixed["training_config"]["init_hashes"]["basis_init_sha256"],
            "fixed_mean": summary_fixed["training_config"]["normalization"]["mean"],
            "fixed_std": summary_fixed["training_config"]["normalization"]["std"],
        },
        "summaries": {
            "fixed": summary_fixed,
            "learned": summary_learned,
        },
    }
    write_json_atomic(train_dir / "train_completion.json", completion)
    write_json_atomic(out_dir / "train_completion.json", completion)
    update_status_train(
        out_dir,
        "completed",
        fixed_pt=str(train_dir / "fixed.pt"),
        learned_pt=str(train_dir / "learned.pt"),
        fixed_sha256=ver_fixed["sha256"],
        learned_sha256=ver_learned["sha256"],
    )
    return completion


if __name__ == "__main__":
    comp = run_all_training()
    print(
        json.dumps(
            {
                "status": comp["status"],
                "fixed_sha256": comp["checkpoints"]["fixed"]["sha256"],
                "learned_sha256": comp["checkpoints"]["learned"]["sha256"],
                "pairing": comp["pairing_verification"],
            },
            indent=2,
        )
    )
