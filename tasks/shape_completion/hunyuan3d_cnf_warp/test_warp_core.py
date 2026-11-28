"""Offline deterministic CPU unit tests for Matern-5/2 Normal-Only CNF Warp."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
from warp_core import (
    align_visible_upright_sym,
    compute_vertex_normals,
    eval_rkhs_u,
    eval_warp_anchor_residuals,
    extract_surface,
    folding_audit,
    gaussian_correction,
    gaussian_solve,
    grid_points,
    matern52_kernel,
    mesh_roughness_fast,
    rkhs_jacobian_u,
    solve_normal_matern_warp,
    solve_p2p_matern_warp,
)


def _fibonacci_sphere(n: int, radius: float = 0.5) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n)
    theta = math.pi * (1.0 + math.sqrt(5.0)) * idx
    pts = np.stack(
        [np.sin(phi) * np.cos(theta), np.cos(phi), np.sin(phi) * np.sin(theta)], axis=1
    )
    return radius * pts


class TestMaternNormalWarpCore(unittest.TestCase):
    def test_matern52_spd_and_analytic_jacobian(self) -> None:
        rng = np.random.default_rng(7)
        anchors = rng.uniform(-0.8, 0.8, size=(16, 3))
        k_mat = matern52_kernel(anchors, anchors)
        eigvals = np.linalg.eigvalsh(k_mat)
        self.assertGreater(float(eigvals.min()), 1e-6)

        weights = rng.normal(scale=0.1, size=(16, 3))
        # Test both general points and exact anchor knots (r == 0 C^2 smoothness!)
        queries = np.vstack([rng.uniform(-0.8, 0.8, size=(6, 3)), anchors[:4]])
        j_analytic = rkhs_jacobian_u(queries, anchors, weights)
        eps = 1e-6
        j_fd = np.zeros_like(j_analytic)
        for k in range(3):
            e = np.zeros(3)
            e[k] = eps
            up = eval_rkhs_u(queries + e, anchors, weights)
            um = eval_rkhs_u(queries - e, anchors, weights)
            j_fd[:, k, :] = (up - um) / (2.0 * eps)
        np.testing.assert_allclose(j_analytic, j_fd, atol=1e-7, rtol=1e-6)

    def test_normal_only_warp_exact_machine_precision_and_no_folding(self) -> None:
        # Base field f0: sphere centered at (0.04, -0.02, 0.03) with radius 0.45 (tanh-saturated)
        c0 = np.array([0.04, -0.02, 0.03])
        r0 = 0.45

        def eval_f0(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            diff = pts - c0[None, :]
            dist = np.linalg.norm(diff, axis=1)
            sdf = dist - r0
            # Steep saturating occupancy logit mimicking Hunyuan3D ShapeVAE
            scale = 8.0
            f = np.tanh(scale * sdf)
            sech2 = 1.0 - f**2
            g = (scale * sech2 / np.maximum(dist, 1e-12))[:, None] * diff
            return f, g

        # Target anchors on shifted/scaled sphere (center 0, radius 0.50)
        anchors = _fibonacci_sphere(32, radius=0.50)
        normals = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)

        # Base mesh extracted on 32^3 grid
        xyz, _ = grid_points(32, -1.0, 1.0, dtype=np.float64)
        vol = eval_f0(xyz)[0].reshape(32, 32, 32)
        raw_m = extract_surface(vol, -1.0, 1.0, 32)
        self.assertIsNotNone(raw_m)
        assert raw_m is not None
        raw_v, raw_f = raw_m
        raw_vn = compute_vertex_normals(raw_v, raw_f)
        if np.median(np.einsum("ij,ij->i", raw_vn, raw_v - c0[None, :])) < 0:
            raw_vn = -raw_vn

        w_nw, max_f, _ = solve_normal_matern_warp(eval_f0, anchors, normals, raw_v, raw_vn)
        self.assertLess(max_f, 1e-13)

        fold = folding_audit(w_nw, anchors, raw_v)
        self.assertEqual(fold["surface_fold_fraction"], 0.0)
        self.assertGreater(fold["surface_min_detJ"], 0.5)

        # Verify chain-rule gradient matches finite differences on f_CNF(x) = f0(x + u(x))
        res = eval_warp_anchor_residuals(eval_f0, anchors, normals, w_nw)
        self.assertLess(res["anchor_value_abs_max"], 1e-13)
        self.assertGreater(res["anchor_normal_cosine_mean"], 0.99)

        # Verify P2P ablation also converges cleanly and differs from Normal-Only
        w_p2p = solve_p2p_matern_warp(eval_f0, anchors, normals, raw_v, raw_vn)
        res_p2p = eval_warp_anchor_residuals(eval_f0, anchors, normals, w_p2p)
        self.assertLess(res_p2p["anchor_value_abs_max"], 1e-12)

    def test_symmetric_upright_alignment_recovers_transform(self) -> None:
        # Chair-like asymmetric L-shape (seat + backrest + left arm) so yaw is uniquely determined
        rng = np.random.default_rng(42)
        gx, gz = np.meshgrid(np.linspace(-0.35, 0.35, 16), np.linspace(-0.35, 0.35, 16))
        seat = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)
        seat_n = np.tile([0.0, 1.0, 0.0], (len(seat), 1))
        bx, by = np.meshgrid(np.linspace(-0.35, 0.35, 16), np.linspace(0.0, 0.75, 16))
        back = np.stack([bx.ravel(), by.ravel(), np.full(bx.size, -0.35)], axis=1)
        back_n = np.tile([0.0, 0.0, 1.0], (len(back), 1))
        az, ay = np.meshgrid(np.linspace(-0.35, 0.35, 12), np.linspace(0.0, 0.35, 10))
        arm = np.stack([np.full(az.size, -0.35), ay.ravel(), az.ravel()], axis=1)
        arm_n = np.tile([1.0, 0.0, 0.0], (len(arm), 1))
        p_base = np.vstack([seat, back, arm])
        p_norm = np.vstack([seat_n, back_n, arm_n])

        yaw_true = math.radians(52.0)
        r_true = np.array(
            [
                [math.cos(yaw_true), 0.0, math.sin(yaw_true)],
                [0.0, 1.0, 0.0],
                [-math.sin(yaw_true), 0.0, math.cos(yaw_true)],
            ]
        )
        s_true = 0.85
        t_true = np.array([0.05, -0.04, 0.07])
        q_full = s_true * (p_base @ r_true.T) + t_true
        qn_full = p_norm @ r_true.T
        anc_idx = rng.choice(len(q_full), size=32, replace=False)

        aln = align_visible_upright_sym(
            p_base, p_norm, q_full, q_full[anc_idx], qn_full[anc_idx]
        )
        self.assertAlmostEqual(aln["R11"], 1.0, places=12)
        self.assertLess(abs(aln["scale"] - s_true), 0.02)
        self.assertLess(np.linalg.norm(aln["rotation"] - r_true), 0.03)
        self.assertLess(np.linalg.norm(aln["translation"] - t_true), 0.02)

    def test_gaussian_patch_and_mesh_roughness(self) -> None:
        anchors = _fibonacci_sphere(12, radius=0.5)
        normals = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)
        base_ops = np.hstack([np.full((12, 1), 0.05), 0.7 * normals])
        psi, diag = gaussian_solve(base_ops, normals, anchors)
        self.assertLess(diag["constraint_residual_inf"], 1e-13)
        cv, cg = gaussian_correction(psi, anchors, anchors, with_grad=True)
        self.assertLess(float(np.max(np.abs(base_ops[:, 0] + cv))), 1e-13)
        assert cg is not None
        self.assertLess(float(np.max(np.abs(base_ops[:, 1:] + cg - normals))), 1e-13)

        xyz, _ = grid_points(20, -1.0, 1.0, dtype=np.float64)
        vol = (np.linalg.norm(xyz, axis=1) - 0.5).reshape(20, 20, 20)
        m = extract_surface(vol, -1.0, 1.0, 20)
        assert m is not None
        rough = mesh_roughness_fast(m[0], m[1])
        self.assertGreater(rough["n_interior_edges"], 100)
        self.assertGreater(rough["dihedral_deg_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
