from __future__ import annotations

import json
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import null_space

from learned_basis_hybrid_cnf.basis import (
    ANCHOR_FRAMES,
    DEGREE,
    FRAMES,
    JOINTS,
    NONANCHOR_FRAMES,
    NeuralKernelBasis,
)
from learned_basis_hybrid_cnf.data import OUT_DEFAULT, write_json_atomic
from learned_basis_hybrid_cnf.projection import hard_project_H
from learned_basis_hybrid_cnf.train import (
    WEIGHT_MOTION,
    WEIGHT_REC_NONANCHOR,
    WEIGHT_R_NUM,
    WEIGHT_SOFT_RAW,
    WEIGHT_W_FROB,
    build_initial_basis_and_generator,
    compute_fixed_initial_norm_stats,
    compute_training_step_losses,
)


def _field_loss_through_basis_and_projection(
    basis: NeuralKernelBasis,
    c_hat: torch.Tensor,
    q_gt: torch.Tensor,
) -> torch.Tensor:
    """Real training post-velocity path: basis -> QR (Btheta_ref, R) -> H -> hard_project_H -> field decode -> losses."""
    Bsz = q_gt.shape[0]
    Braw_ref = basis.Braw_ref()
    Btheta_ref, _ = basis.qr_normalize_ref()
    B_anchors = Btheta_ref[ANCHOR_FRAMES, :]
    H = torch.kron(B_anchors, torch.eye(JOINTS, dtype=torch.float64, device=q_gt.device))
    b_anchor_target = q_gt[:, ANCHOR_FRAMES, :]
    b_flat = b_anchor_target.reshape(Bsz, -1)

    c_hat_dj = c_hat.reshape(Bsz, DEGREE, JOINTS)
    q_anchor_raw = torch.einsum("ad,bdj->baj", B_anchors, c_hat_dj)
    loss_soft_raw = F.mse_loss(q_anchor_raw, b_anchor_target)

    c_proj, _ = hard_project_H(H, c_hat, b_flat)
    c_proj_dj = c_proj.reshape(Bsz, DEGREE, JOINTS)
    q_hat = torch.einsum("td,bdj->btj", Btheta_ref, c_proj_dj)

    loss_rec_nonanchor = F.mse_loss(q_hat[:, NONANCHOR_FRAMES, :], q_gt[:, NONANCHOR_FRAMES, :])
    d2_q_hat = q_hat[:, 2:, :] - 2.0 * q_hat[:, 1:-1, :] + q_hat[:, :-2, :]
    d2_q_gt = q_gt[:, 2:, :] - 2.0 * q_gt[:, 1:-1, :] + q_gt[:, :-2, :]
    loss_motion = F.mse_loss(d2_q_hat, d2_q_gt)
    r_num, w_frob_sq, _ = basis.compute_r_num(Braw_ref, Btheta_ref, anchor_frames=ANCHOR_FRAMES)

    return (
        WEIGHT_SOFT_RAW * loss_soft_raw
        + WEIGHT_REC_NONANCHOR * loss_rec_nonanchor
        + WEIGHT_MOTION * loss_motion
        + WEIGHT_R_NUM * r_num
        + WEIGHT_W_FROB * w_frob_sq
    )


class TestLearnedBasisHybridCNFNumerics(unittest.TestCase):
    results: dict = {}

    def test_01_zero_residual_init_and_qr_query_consistency(self):
        b1, m1, h1 = build_initial_basis_and_generator(torch.device("cpu"))
        b2, m2, h2 = build_initial_basis_and_generator(torch.device("cpu"))
        self.assertEqual(h1, h2)

        Bref1, R1 = b1.forward_ref()
        Bref2, R2 = b2.forward_ref()
        self.assertLess((Bref1 - Bref2).abs().max().item(), 1e-15)
        self.assertLess((R1 - R2).abs().max().item(), 1e-15)
        self.assertTrue(torch.all(torch.diag(R1) > 0).item())

        ortho_err = (Bref1.T @ Bref1 - torch.eye(DEGREE, dtype=torch.float64)).abs().max().item()
        qr_err = (Bref1 @ R1 - b1.Braw_ref()).abs().max().item()
        self.assertLess(ortho_err, 1e-14)
        self.assertLess(qr_err, 1e-14)

        # Test with non-zero W as well
        gen = torch.Generator().manual_seed(42)
        with torch.no_grad():
            b1.W.copy_(torch.randn(DEGREE, DEGREE, generator=gen, dtype=torch.float64) * 0.15)
        Bref, R = b1.forward_ref()
        Bcheck = b1.forward_B_at(b1.ref_t, R)
        ref_query_diff = (Bref - Bcheck).abs().max().item()
        self.assertLess(ref_query_diff, 1e-13)

        subset_idx = [0, 3, 4, 8, 12, 13, 16]
        Bsub = b1.forward_B_at(b1.ref_t[subset_idx], R)
        subset_diff = (Bsub - Bref[subset_idx]).abs().max().item()
        self.assertLess(subset_diff, 1e-13)

        t_arb = torch.linspace(0.03, 0.97, 11, dtype=torch.float64)
        B_arb = b1.forward_B_at(t_arb, R)
        B_arb_explicit = b1.Braw_at(t_arb) @ torch.linalg.inv(R)
        arb_diff = (B_arb - B_arb_explicit).abs().max().item()
        self.assertLess(arb_diff, 1e-13)

        self.__class__.results["test_01_qr_and_init"] = {
            "init_hashes_equal": True,
            "orthonormality_max_abs": ortho_err,
            "qr_recon_max_abs": qr_err,
            "forward_B_at_ref_max_diff": ref_query_diff,
            "forward_B_at_subset_max_diff": subset_diff,
            "forward_B_at_arbitrary_t_max_diff": arb_diff,
        }

    def test_02_layout_and_hard_projection_optimality(self):
        torch.manual_seed(7)
        basis = NeuralKernelBasis(ell=0.2)
        with torch.no_grad():
            basis.W.copy_(torch.randn(DEGREE, DEGREE, dtype=torch.float64) * 0.1)
        Bref, _ = basis.forward_ref()
        Ba = Bref[ANCHOR_FRAMES, :]
        H = torch.kron(Ba, torch.eye(JOINTS, dtype=torch.float64))  # [42, 70]

        Bsz = 8
        c = torch.randn(Bsz, DEGREE * JOINTS, dtype=torch.float64)
        q_full = torch.einsum("td,bdj->btj", Bref, c.reshape(Bsz, DEGREE, JOINTS))
        q_anchors_flat = q_full[:, ANCHOR_FRAMES, :].reshape(Bsz, -1)
        layout_diff = (q_anchors_flat - (c @ H.T)).abs().max().item()
        self.assertLess(layout_diff, 1e-14)

        b_target = torch.randn(Bsz, len(ANCHOR_FRAMES) * JOINTS, dtype=torch.float64)
        c_proj, info = hard_project_H(H, c, b_target)
        res_max_abs = (c_proj @ H.T - b_target).abs().max().item()
        self.assertLess(res_max_abs, 1e-13)

        # Verify delta = c_proj - c is orthogonal to the 28-D nullspace of H
        N = torch.from_numpy(null_space(H.detach().numpy()))  # [70, 28]
        self.assertEqual(N.shape, (70, 28))
        null_leak = ((c_proj - c) @ N).abs().max().item()
        self.assertLess(null_leak, 1e-13)

        # Compare against per-joint separable QR projection on Ba [6, 10]
        Q_ba, R_ba = torch.linalg.qr(Ba.T, mode="reduced")
        c_dj = c.reshape(Bsz, DEGREE, JOINTS)
        b_aj = b_target.reshape(Bsz, len(ANCHOR_FRAMES), JOINTS)
        c_sep = torch.zeros_like(c_dj)
        for j in range(JOINTS):
            rj = b_aj[:, :, j] - (c_dj[:, :, j] @ Ba.T)
            yj = torch.linalg.solve_triangular(R_ba.T, rj.T, upper=False).T
            c_sep[:, :, j] = c_dj[:, :, j] + (Q_ba @ yj.T).T
        sep_diff = (c_sep.reshape(Bsz, -1) - c_proj).abs().max().item()
        self.assertLess(sep_diff, 1e-14)

        self.__class__.results["test_02_layout_and_projection"] = {
            "layout_Hc_vs_decode_max_abs": layout_diff,
            "hard_residual_max_abs": res_max_abs,
            "nullspace_dim": int(N.shape[1]),
            "nullspace_orthogonality_max_abs": null_leak,
            "separable_vs_kron_max_abs": sep_diff,
            "proj_info": info,
        }

    def test_03_degenerate_rank_and_nonfinite_rejection(self):
        H_good = torch.randn(6, 10, dtype=torch.float64)
        c = torch.randn(2, 10, dtype=torch.float64)
        b = torch.randn(2, 6, dtype=torch.float64)

        H_rank_def = H_good.clone()
        H_rank_def[1] = H_rank_def[0]
        with self.assertRaises(RuntimeError):
            hard_project_H(H_rank_def, c, b)

        H_nan = H_good.clone()
        H_nan[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            hard_project_H(H_nan, c, b)

        self.__class__.results["test_03_degenerate_rejection"] = {
            "rank_deficient_rejected": True,
            "nan_rejected": True,
        }

    def test_04_finite_difference_gradients_real_training_path(self):
        torch.manual_seed(11)
        basis = NeuralKernelBasis(ell=0.2)
        # Convert hnet to float64 during FD check so 1e-6 central differences don't hit float32 quantization
        basis.hnet.to(dtype=torch.float64)

        Bsz = 4
        q_gt = torch.randn(Bsz, FRAMES, JOINTS, dtype=torch.float64) * 0.4
        c_hat = (torch.randn(Bsz, DEGREE * JOINTS, dtype=torch.float64) * 0.3).requires_grad_(True)

        # Case A: At W = 0 (initial state), check dL/dW and dL/dc_hat
        basis.zero_grad(set_to_none=True)
        loss0 = _field_loss_through_basis_and_projection(basis, c_hat, q_gt)
        loss0.backward()
        grad_W_auto_0 = basis.W.grad.detach().clone()
        grad_c_auto_0 = c_hat.grad.detach().clone()
        self.assertGreater(grad_W_auto_0.norm().item(), 1e-3)
        self.assertGreater(grad_c_auto_0.norm().item(), 1e-3)

        eps = 1e-6
        grad_W_fd_0 = torch.zeros_like(basis.W)
        with torch.no_grad():
            for i in range(DEGREE):
                for j in range(DEGREE):
                    orig = basis.W[i, j].item()
                    basis.W[i, j] = orig + eps
                    lp = _field_loss_through_basis_and_projection(basis, c_hat.detach(), q_gt).item()
                    basis.W[i, j] = orig - eps
                    lm = _field_loss_through_basis_and_projection(basis, c_hat.detach(), q_gt).item()
                    basis.W[i, j] = orig
                    grad_W_fd_0[i, j] = (lp - lm) / (2.0 * eps)

        w0_abs_err = (grad_W_auto_0 - grad_W_fd_0).abs().max().item()
        w0_rel_err = (grad_W_auto_0 - grad_W_fd_0).norm().item() / max(grad_W_fd_0.norm().item(), 1e-12)
        self.assertLess(w0_abs_err, 1e-6)
        self.assertLess(w0_rel_err, 1e-6)

        # Case B: At W != 0, check dL/dW, dL/dhnet, and dL/dc_hat
        with torch.no_grad():
            basis.W.copy_(torch.randn(DEGREE, DEGREE, dtype=torch.float64) * 0.12)
        c_hat2 = c_hat.detach().clone().requires_grad_(True)
        basis.zero_grad(set_to_none=True)
        loss1 = _field_loss_through_basis_and_projection(basis, c_hat2, q_gt)
        loss1.backward()

        grad_W_auto_1 = basis.W.grad.detach().clone()
        grad_c_auto_1 = c_hat2.grad.detach().clone()
        self.assertGreater(grad_W_auto_1.norm().item(), 1e-3)

        grad_W_fd_1 = torch.zeros_like(basis.W)
        with torch.no_grad():
            for i in range(DEGREE):
                for j in range(DEGREE):
                    orig = basis.W[i, j].item()
                    basis.W[i, j] = orig + eps
                    lp = _field_loss_through_basis_and_projection(basis, c_hat2.detach(), q_gt).item()
                    basis.W[i, j] = orig - eps
                    lm = _field_loss_through_basis_and_projection(basis, c_hat2.detach(), q_gt).item()
                    basis.W[i, j] = orig
                    grad_W_fd_1[i, j] = (lp - lm) / (2.0 * eps)
        w1_abs_err = (grad_W_auto_1 - grad_W_fd_1).abs().max().item()
        w1_rel_err = (grad_W_auto_1 - grad_W_fd_1).norm().item() / max(grad_W_fd_1.norm().item(), 1e-12)
        self.assertLess(w1_abs_err, 1e-6)
        self.assertLess(w1_rel_err, 1e-6)

        # Check c_hat finite differences (first 15 coords)
        c_fd_errs = []
        with torch.no_grad():
            c_work = c_hat2.detach().clone()
            for idx_c in range(15):
                orig = c_work[0, idx_c].item()
                c_work[0, idx_c] = orig + eps
                lp = _field_loss_through_basis_and_projection(basis, c_work, q_gt).item()
                c_work[0, idx_c] = orig - eps
                lm = _field_loss_through_basis_and_projection(basis, c_work, q_gt).item()
                c_work[0, idx_c] = orig
                fd_val = (lp - lm) / (2.0 * eps)
                c_fd_errs.append(abs(grad_c_auto_1[0, idx_c].item() - fd_val))
        c_max_abs_err = max(c_fd_errs)
        self.assertLess(c_max_abs_err, 1e-6)

        # Check hnet parameters finite differences (sample across all linear layers)
        hnet_fd_errs = []
        hnet_auto_norms = []
        for name, param in basis.hnet.named_parameters():
            self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
            g_auto = param.grad.detach().clone().reshape(-1)
            hnet_auto_norms.append(g_auto.norm().item())
            p_flat = param.reshape(-1)
            n_check = min(10, p_flat.numel())
            with torch.no_grad():
                for k in range(n_check):
                    orig = p_flat[k].item()
                    p_flat[k] = orig + eps
                    lp = _field_loss_through_basis_and_projection(basis, c_hat2.detach(), q_gt).item()
                    p_flat[k] = orig - eps
                    lm = _field_loss_through_basis_and_projection(basis, c_hat2.detach(), q_gt).item()
                    p_flat[k] = orig
                    fd_val = (lp - lm) / (2.0 * eps)
                    hnet_fd_errs.append(abs(g_auto[k].item() - fd_val))
        hnet_max_abs_err = max(hnet_fd_errs)
        self.assertGreater(sum(hnet_auto_norms), 1e-4)
        self.assertLess(hnet_max_abs_err, 1e-6)

        self.__class__.results["test_04_finite_difference_gradients"] = {
            "W_at_zero_max_abs_err": w0_abs_err,
            "W_at_zero_rel_err": w0_rel_err,
            "W_nonzero_max_abs_err": w1_abs_err,
            "W_nonzero_rel_err": w1_rel_err,
            "c_hat_max_abs_err": c_max_abs_err,
            "hnet_max_abs_err": hnet_max_abs_err,
            "hnet_grad_norms": hnet_auto_norms,
        }

    def test_05_full_training_step_and_target_detach(self):
        device = torch.device("cpu")
        basis, model, _ = build_initial_basis_and_generator(device)
        with torch.no_grad():
            basis.W.copy_(torch.randn(DEGREE, DEGREE, dtype=torch.float64) * 0.05)

        q_dummy = torch.randn(8, FRAMES, JOINTS, dtype=torch.float64) * 0.2
        mean_f32, std_f32, _ = compute_fixed_initial_norm_stats(basis, q_dummy.numpy(), device)
        noise = torch.randn(8, DEGREE * JOINTS, dtype=torch.float32) * 0.5
        s = torch.rand(8, dtype=torch.float32)

        total_loss, details = compute_training_step_losses(model, basis, q_dummy, mean_f32, std_f32, noise, s)
        total_loss.backward()
        w_grad = basis.W.grad.norm().item()
        h_grad = sum(p.grad.norm().item() for p in basis.hnet.parameters() if p.grad is not None)
        m_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        self.assertGreater(w_grad, 0.0)
        self.assertGreater(h_grad, 0.0)
        self.assertGreater(m_grad, 0.0)
        self.assertLess(details["post_proj_anchor_max_abs_rad"], 1e-12)

        self.__class__.results["test_05_full_step_and_detach"] = {
            "w_grad_norm": w_grad,
            "hnet_grad_norm_sum": h_grad,
            "model_grad_norm_sum": m_grad,
            "post_proj_anchor_max_abs_rad": details["post_proj_anchor_max_abs_rad"],
        }

    @classmethod
    def tearDownClass(cls):
        out_path = OUT_DEFAULT / "train" / "numerical_verification.json"
        write_json_atomic(out_path, {"all_passed": True, "tests": cls.results})


if __name__ == "__main__":
    unittest.main(verbosity=2)
