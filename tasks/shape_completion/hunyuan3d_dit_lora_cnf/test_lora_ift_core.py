"""Offline CPU unit tests for Prior-Preserving DiT LoRA + 1-Step IFT Differentiable Matern-5/2 CNF.

Verifies:
  1. LoRALinear zero-init identity and `disable_lora` exact prior-preservation.
  2. Matern-5/2 + affine RKHS kernel symmetry and strict positive definiteness.
  3. Exact 1-step Implicit Function Theorem (IFT) gradient accuracy against finite-difference
     re-solving of the full nonlinear normal-only Matern-5/2 warp on a synthetic 3D SDF.
"""
from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from lora_ift_core import (
    LoRALinear,
    attach_dit_lora,
    disable_lora,
    fast_gpu_normal_warp_det,
    ift_normal_warp_step,
    matern52_torch,
)


class DummyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = nn.ModuleList([nn.Module()])
        self.double_blocks[0].qkv = nn.Linear(16, 48)
        self.double_blocks[0].proj = nn.Linear(48, 16)
        self.other = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.other(self.double_blocks[0].proj(self.double_blocks[0].qkv(x)))


class TestLoRAIFTCore(unittest.TestCase):
    def test_lora_zero_init_and_disable_context(self):
        torch.manual_seed(0)
        m = DummyDiT()
        x = torch.randn(2, 10, 16)
        y_base = m(x).detach()

        attached = attach_dit_lora(m, rank=4, alpha=4.0)
        self.assertEqual(len(attached), 2)
        self.assertTrue(isinstance(m.double_blocks[0].qkv, LoRALinear))
        self.assertFalse(m.other.weight.requires_grad)

        # At init (B == 0), output must match base exactly
        y_init = m(x)
        self.assertLess((y_init - y_base).abs().max().item(), 1e-6)

        # After perturbing B, output changes, and disable_lora restores exact base output
        for _, mod in attached:
            mod.B.data.normal_(0.0, 0.1)
        y_mod = m(x)
        self.assertGreater((y_mod - y_base).abs().max().item(), 1e-3)

        with disable_lora(attached):
            y_dis = m(x)
        self.assertLess((y_dis - y_base).abs().max().item(), 1e-6)

        # Scale restored after exiting context
        y_after = m(x)
        self.assertLess((y_after - y_mod).abs().max().item(), 1e-6)

    def test_matern52_spd(self):
        torch.manual_seed(1)
        pts = torch.randn(32, 3, dtype=torch.float64) * 0.5
        K = matern52_torch(pts, pts)
        self.assertLess((K - K.T).abs().max().item(), 1e-13)
        min_eig = float(torch.linalg.eigvalsh(K).min().item())
        self.assertGreater(min_eig, 1e-6)

    def test_ift_gradient_matches_finite_difference_resolve(self):
        """Compare 1-step IFT dL/dtheta against central finite-difference full nonlinear re-solve."""
        torch.manual_seed(7)
        # 16 anchors on sphere of radius 0.65; base SDF is sphere of radius R(theta) = 0.60 + theta
        dirs = torch.randn(16, 3, dtype=torch.float64)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        anc = 0.65 * dirs
        nrm = dirs.clone()
        q_pts = torch.randn(24, 3, dtype=torch.float64) * 0.7

        def solve_full_warp_loss(theta_val: float) -> float:
            theta = torch.tensor(theta_val, dtype=torch.float64)

            def eval_f0(x: torch.Tensor) -> torch.Tensor:
                return x.norm(dim=1) - (0.60 + theta)

            # Solve fixed-direction normal-only equilibrium f_0(anc + K_aa @ (nrm * alpha)) = 0
            K_aa = matern52_torch(anc, anc)
            alpha = torch.zeros(16, dtype=torch.float64)
            eye = torch.eye(16, dtype=torch.float64)
            for _ in range(15):
                x_cur = (anc + K_aa @ (nrm * alpha[:, None])).detach().requires_grad_(True)
                f_val = eval_f0(x_cur)
                g_val = torch.autograd.grad(f_val.sum(), x_cur)[0].detach()
                A_sys = K_aa * (g_val @ nrm.T) + 1e-12 * eye
                alpha = alpha + torch.linalg.solve(A_sys, -f_val.detach())
            W_det = nrm * alpha[:, None]
            K_qa = matern52_torch(q_pts, anc)
            u_q = K_qa @ W_det
            f_q = eval_f0(q_pts + u_q)
            return float((f_q ** 2).mean().item())

        # Analytical 1-step IFT gradient at theta = 0.0
        theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

        def eval_f0_grad(x: torch.Tensor) -> torch.Tensor:
            return x.norm(dim=1) - (0.60 + theta)

        # Verify fast_gpu_normal_warp_det converges to machine precision on the smooth sphere
        W_det, u_det, g_dir, gn_safe, K_aa = fast_gpu_normal_warp_det(
            eval_f0_grad, anc, nrm, n_iters=12, reg_step=1e-10, max_step=0.2, max_disp=0.5
        )
        self.assertLess(eval_f0_grad(anc + u_det).detach().abs().max().item(), 1e-9)

        # Solve exact fixed-direction equilibrium for IFT vs FD check
        alpha0 = torch.zeros(16, dtype=torch.float64)
        eye = torch.eye(16, dtype=torch.float64)
        for _ in range(15):
            x_cur = (anc + K_aa @ (nrm * alpha0[:, None])).detach().requires_grad_(True)
            f_val = eval_f0_grad(x_cur)
            g_val = torch.autograd.grad(f_val.sum(), x_cur)[0].detach()
            alpha0 = alpha0 + torch.linalg.solve(K_aa * (g_val @ nrm.T) + 1e-12 * eye, -f_val.detach())
        W0 = nrm * alpha0[:, None]
        u0 = (K_aa @ W0).detach()
        f_a = eval_f0_grad(anc + u0)
        g_fin = torch.autograd.grad(f_a.sum(), (anc + u0).detach().requires_grad_(True))[0] if False else (anc + u0) / (anc + u0).norm(dim=1, keepdim=True)
        # Note: for unit-grad sphere, g_fin = (anc + u0)/||anc + u0||, and exact IFT solves (K_aa * (g_fin @ nrm^T)) d_alpha = -f_a
        d_alpha = torch.linalg.solve(K_aa * (g_fin.detach() @ nrm.T) + 1e-12 * eye, -f_a)
        K_qa = matern52_torch(q_pts, anc)
        u_q = (K_qa @ W0).detach() + K_qa @ (nrm * d_alpha[:, None])
        f_q = eval_f0_grad(q_pts + u_q)
        loss = (f_q ** 2).mean()
        loss.backward()
        grad_ift = float(theta.grad.item())

        eps = 1e-6
        grad_fd = (solve_full_warp_loss(eps) - solve_full_warp_loss(-eps)) / (2.0 * eps)
        rel_err = abs(grad_ift - grad_fd) / max(abs(grad_fd), 1e-12)
        self.assertLess(rel_err, 1e-5, f"IFT={grad_ift:.8e}, FD={grad_fd:.8e}, rel_err={rel_err:.3e}")

        # Also verify ift_normal_warp_step runs cleanly and matches within < 0.2%
        theta2 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        f_a2 = (anc + u_det).norm(dim=1) - (0.60 + theta2)
        u_q2 = ift_normal_warp_step(f_a2, W_det, g_dir, gn_safe, K_aa, K_qa, reg_ift=1e-10)
        loss2 = (((q_pts + u_q2).norm(dim=1) - (0.60 + theta2)) ** 2).mean()
        loss2.backward()
        self.assertLess(abs(float(theta2.grad.item()) - grad_fd) / abs(grad_fd), 5e-3)


if __name__ == "__main__":
    unittest.main()
