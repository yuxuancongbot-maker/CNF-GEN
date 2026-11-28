"""Core PyTorch building blocks for Prior-Preserving DiT LoRA + 1-Step IFT Differentiable CNF.

Provides:
  - `LoRALinear`, `attach_dit_lora`, `disable_lora`, `save_lora_state`, `load_lora_state`
  - `matern52_torch`: strictly SPD C^2 Matern-5/2 + weakly penalized affine RKHS kernel
  - `fast_gpu_normal_warp_det`: detached 4-step damped Newton normal-only Matern-5/2 warp
  - `ift_normal_warp_step`: exact 1-step Implicit Function Theorem (IFT) differentiable 32x32 SPD solve
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

ELL = 0.65
GAMMA_TR = 1.0
GAMMA_LIN = 0.15


class LoRALinear(nn.Module):
    """Low-rank adapter initialized to exact zero output delta."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 8.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = float(alpha) / float(r)
        self.A = nn.Parameter(
            torch.randn(r, base.in_features, device=base.weight.device, dtype=torch.float32)
            * (1.0 / math.sqrt(base.in_features))
        )
        self.B = nn.Parameter(
            torch.zeros(base.out_features, r, device=base.weight.device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.scale == 0.0:
            return out
        dx = F.linear(F.linear(x.float(), self.A.float()), self.B.float()) * self.scale
        return out + dx.to(out.dtype)


def attach_dit_lora(
    dit: nn.Module,
    rank: int = 8,
    alpha: float | None = None,
    block_prefixes: tuple[str, ...] = ("double_blocks.", "single_blocks."),
    target_suffixes: tuple[str, ...] = ("qkv", "proj", "linear1", "linear2"),
) -> list[tuple[str, LoRALinear]]:
    """Freeze base DiT weights and attach LoRA adapters to target linear layers."""
    if alpha is None:
        alpha = float(rank)
    for p in dit.parameters():
        p.requires_grad_(False)
    attached: list[tuple[str, LoRALinear]] = []
    for mod_name, mod in list(dit.named_modules()):
        if not any(mod_name.startswith(pref) for pref in block_prefixes):
            continue
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_suffixes:
                lora = LoRALinear(child, r=rank, alpha=alpha)
                setattr(mod, child_name, lora)
                attached.append((f"{mod_name}.{child_name}", lora))
    return attached


@contextlib.contextmanager
def disable_lora(attached: list[tuple[str, LoRALinear]]):
    """Temporarily disable LoRA deltas to evaluate the frozen pretrained model with zero extra VRAM."""
    saved = [mod.scale for _, mod in attached]
    for _, mod in attached:
        mod.scale = 0.0
    try:
        yield
    finally:
        for (_, mod), sc in zip(attached, saved):
            mod.scale = sc


def save_lora_state(attached: list[tuple[str, LoRALinear]], path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: {
            "A": mod.A.detach().cpu(),
            "B": mod.B.detach().cpu(),
            "r": mod.r,
            "scale": mod.scale,
        }
        for name, mod in attached
    }
    torch.save({"lora": state, "meta": meta}, path)


def load_lora_state(dit: nn.Module, ckpt_path: Path, device: str | torch.device = "cuda") -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    meta = ckpt["meta"]
    rank = int(meta["rank"])
    attached = attach_dit_lora(dit, rank=rank, alpha=float(rank))
    lora_sd = ckpt["lora"]
    for name, mod in attached:
        mod.A.data.copy_(lora_sd[name]["A"].to(mod.A.device))
        mod.B.data.copy_(lora_sd[name]["B"].to(mod.B.device))
    return meta


def matern52_torch(
    X: torch.Tensor,
    Y: torch.Tensor,
    ell: float = ELL,
    gamma_tr: float = GAMMA_TR,
    gamma_lin: float = GAMMA_LIN,
) -> torch.Tensor:
    """Strictly SPD C^2 Matern-5/2 + weakly penalized affine RKHS kernel."""
    r = torch.cdist(X, Y)
    s = math.sqrt(5.0) * r / ell
    return (1.0 + s + (s ** 2) / 3.0) * torch.exp(-s) + gamma_tr + gamma_lin * (X @ Y.T)


def fast_gpu_normal_warp_det(
    eval_f0: Callable[[torch.Tensor], torch.Tensor],
    anc: torch.Tensor,
    nrm: torch.Tensor,
    n_iters: int = 4,
    ell: float = ELL,
    reg_step: float = 1e-3,
    max_step: float = 0.08,
    max_disp: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detached damped normal-only Matern-5/2 Newton solve to obtain converged (W_det, u_det, g_dir, gn_safe, K_aa).

    Args:
        eval_f0: callable mapping (M, 3) points -> (M,) scalar SDF values (supports autograd w.r.t. coords).
        anc: (M, 3) visible surface anchor coordinates.
        nrm: (M, 3) unit outward normals at anchors.
    """
    M = anc.shape[0]
    K_aa = matern52_torch(anc, anc, ell=ell)
    W = torch.zeros((M, 3), device=anc.device, dtype=anc.dtype)
    eye = torch.eye(M, device=anc.device, dtype=anc.dtype)

    for _ in range(n_iters):
        u_anc = K_aa @ W
        x_cur = (anc + u_anc).detach().requires_grad_(True)
        f_val = eval_f0(x_cur)
        g_val = torch.autograd.grad(f_val.sum(), x_cur)[0].detach()
        f_det = f_val.detach()
        gn = g_val.norm(dim=1)
        sat = (f_det.abs() > 0.85) | (gn < 0.2)
        g_dir = torch.where(sat[:, None], nrm, g_val / gn.clamp(min=1e-6)[:, None])
        raw_step = torch.where(sat, -0.04 * torch.sign(f_det), -f_det / gn.clamp(min=0.2))
        rhs_du = raw_step.clamp(-max_step, max_step)
        A_sys = K_aa * (g_dir @ g_dir.T) + reg_step * eye
        d_alpha = torch.linalg.solve(A_sys, rhs_du)
        W = W + g_dir * d_alpha[:, None]
        u_norm = (K_aa @ W).norm(dim=1).max().item()
        if u_norm > max_disp:
            W = W * (max_disp / u_norm)

    u_det = (K_aa @ W).detach()
    x_fin = (anc + u_det).detach().requires_grad_(True)
    f_fin = eval_f0(x_fin)
    g_fin = torch.autograd.grad(f_fin.sum(), x_fin)[0].detach()
    gn_fin = g_fin.norm(dim=1)
    sat_fin = (f_fin.detach().abs() > 0.85) | (gn_fin < 0.2)
    g_dir_fin = torch.where(sat_fin[:, None], nrm, g_fin / gn_fin.clamp(min=1e-6)[:, None])
    gn_safe = torch.where(sat_fin, torch.full_like(gn_fin, 20.0), gn_fin.clamp(min=0.5))
    return W.detach(), u_det, g_dir_fin.detach(), gn_safe.detach(), K_aa.detach()


def ift_normal_warp_step(
    f_anc_grad: torch.Tensor,
    W_det: torch.Tensor,
    g_dir: torch.Tensor,
    gn_safe: torch.Tensor,
    K_aa: torch.Tensor,
    K_qa: torch.Tensor,
    reg_ift: float = 1e-4,
) -> torch.Tensor:
    """Exact 1-step Implicit Function Theorem (IFT) differentiable warp displacement u(q) at query points q.

    By the IFT around converged equilibrium f_0(p_i + u_det(p_i); theta) = 0, the first-order
    response dW/dtheta satisfies:
        [K_aa * (g_dir g_dir^T) + reg I] d_alpha = -f_anc_grad / ||grad f_0||,
        dW = g_dir * d_alpha[:, None],
        u(q) = (K_qa @ W_det).detach() + K_qa @ dW.
    At convergence f_anc_grad = 0, so forward u(q) equals u_det(q) while backward carries exact d u / d theta.
    """
    M = K_aa.shape[0]
    eye = torch.eye(M, device=K_aa.device, dtype=K_aa.dtype)
    A_sys = K_aa * (g_dir @ g_dir.T) + reg_ift * eye
    d_alpha = torch.linalg.solve(A_sys, -f_anc_grad / gn_safe)
    dW = g_dir * d_alpha[:, None]
    return (K_qa @ W_det).detach() + (K_qa @ dW)
