"""CPU-only regression tests: axis convention, analytic/autograd derivatives,
exact FP64 CNF value/gradient corrections, and seeded surface sampling.

No CUDA, no official weights, no data download, no network.

Run:  python -m unittest discover -s tasks/shape_completion/tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gcnf_shape_completion import axes, conventions  # noqa: E402
from gcnf_shape_completion.cnf import HermiteGaussianCNF, flatten_ops  # noqa: E402
from gcnf_shape_completion.conditioning import (  # noqa: E402
    latent_observation_mask,
    masked_q_sample,
)
from gcnf_shape_completion.data import build_loader, validate_sample  # noqa: E402
from gcnf_shape_completion.field import CubicSplineField, affine_grid  # noqa: E402


class AxisTests(unittest.TestCase):
    def test_anisotropic_known_node(self):
        report = axes.assert_axis_convention(res=9, dtype=torch.float64)
        self.assertTrue(report["ok"])

    def test_grid_permutation_direction(self):
        # (x, y, z) value with distinct coefficients; a single known node is
        # enough to rule out a wrong/symmetric permutation.
        r = 5
        c = conventions.node_coords(r, dtype=torch.float64)
        xx, yy, zz = torch.meshgrid(c, c, c, indexing="ij")
        v = 2.0 * xx - 5.0 * yy + 11.0 * zz
        zyx = axes.xyz_to_zyx(v.unsqueeze(0).unsqueeze(0))[0, 0]
        self.assertAlmostEqual(float(zyx[3, 1, 4]), float(v[4, 1, 3]), places=12)
        self.assertTrue(torch.equal(v, axes.zyx_to_xyz(zyx.unsqueeze(0).unsqueeze(0))[0, 0]))

    def test_anchors_not_transposed(self):
        # World anchor arrays are (M,3) and must never be permuted by axes helpers.
        a = torch.tensor([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]])
        with self.assertRaises(ValueError):
            axes.xyz_to_zyx(a)


class FieldDerivativeTests(unittest.TestCase):
    def test_affine_value_and_gradient_exact(self):
        res = 12
        torch.manual_seed(7)
        slope = torch.tensor([1.5, -2.25, 3.75], dtype=torch.float64)
        bias = 0.4
        grid = affine_grid(slope, bias, res, dtype=torch.float64)
        field = CubicSplineField(grid)
        pts = torch.tensor(
            [[-0.9, 0.1, 0.6], [0.0, 0.0, 0.0], [0.33, -0.77, 0.91]], dtype=torch.float64
        )
        vals, grads = field.evaluate(pts, with_grad=True)
        expected = (pts * slope).sum(-1) + bias
        self.assertLess(float((vals - expected).abs().max()), 1e-10)
        self.assertLess(float((grads - slope).abs().max()), 1e-9)

    def test_analytic_gradient_matches_finite_difference(self):
        res = 9
        torch.manual_seed(11)
        values = torch.randn(res, res, res, dtype=torch.float64)
        field = CubicSplineField(values)
        p = torch.tensor([0.123, -0.311, 0.437], dtype=torch.float64)
        _, grad = field.evaluate(p.unsqueeze(0), with_grad=True)
        eps = 1e-5
        for k in range(3):
            pp = p.clone()
            pp[k] += eps
            pm = p.clone()
            pm[k] -= eps
            fd = (field.value(pp.unsqueeze(0)) - field.value(pm.unsqueeze(0))) / (2 * eps)
            self.assertLess(abs(float(grad[0, k]) - float(fd[0])), 2e-5)

    def test_autograd_gradient_wrt_nodes_is_finite(self):
        res = 9
        torch.manual_seed(11)
        values = torch.randn(res, res, res, dtype=torch.float64, requires_grad=True)
        field = CubicSplineField(values)
        p = torch.tensor([0.123, -0.311, 0.437], dtype=torch.float64)
        f = field.value(p.unsqueeze(0))
        g = torch.autograd.grad(f, values)[0]
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(float(g.abs().sum()), 0.0)

    def test_backward_to_nodes(self):
        res = 8
        torch.manual_seed(3)
        values = torch.randn(res, res, res, dtype=torch.float64, requires_grad=True)
        field = CubicSplineField(values)
        q = torch.rand(16, 3, dtype=torch.float64) * 2 - 1
        v, grad = field.evaluate(q, with_grad=True)
        loss = v.pow(2).mean() + grad.pow(2).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertGreater(float(values.grad.abs().sum()), 0.0)


class ConditioningTests(unittest.TestCase):
    def test_mask_reduce_modes(self):
        # res 8, latent_res 2 -> 4^3 blocks.  Build three known blocks with
        # 64, 32 and 1 known voxels respectively.
        k = torch.zeros(1, 1, 8, 8, 8, dtype=torch.bool)
        k[0, 0, :4, :4, :4] = True  # block (0,0,0): 64/64
        k[0, 0, 4:6, 4:8, :4] = True  # block (1,1,0): 32/64
        k[0, 0, 4:5, :4, 4:5] = True  # block (1,0,1): 1/64
        half = latent_observation_mask(k, 2, reduce="half", erode=0)
        allb = latent_observation_mask(k, 2, reduce="all", erode=0)
        anyb = latent_observation_mask(k, 2, reduce="any", erode=0)
        self.assertEqual(int(half.sum()), 2)  # >= 50% known
        self.assertEqual(int(allb.sum()), 1)  # fully known only
        self.assertEqual(int(anyb.sum()), 3)  # any known

    def test_erosion_is_correct_min_pool(self):
        k = torch.ones(1, 1, 12, 12, 12, dtype=torch.bool)
        no_erode = latent_observation_mask(k, 3, reduce="half", erode=0)
        erode = latent_observation_mask(k, 3, reduce="half", erode=1)
        self.assertEqual(int(no_erode.sum()), 27)
        # The old `max_pool(-mf) > 0.5` was always false -> empty mask.  A
        # correct min-pool keeps observed voxels when all (valid) neighbours
        # are observed.
        self.assertEqual(int(erode.sum()), 27)
        # Removing one latent neighbour erodes the centre via min-pool.
        k[0, 0, 0:4, 4:8, 4:8] = False
        erode2 = latent_observation_mask(k, 3, reduce="half", erode=1)
        self.assertEqual(float(erode2[0, 0, 1, 1, 1]), 0.0)
        self.assertLess(int(erode2.sum()), 27)

    def test_masked_q_sample_uses_supplied_noise(self):
        class _Sched:
            def q_sample(self, x0, t, noise):
                return x0 + noise  # identity schedule; easy to inspect

        z = torch.zeros(1, 1, 2, 2, 2)
        mask = torch.tensor([[[[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]]])
        no = torch.ones_like(z) * 3.0
        nh = torch.ones_like(z) * 7.0
        t = torch.zeros(1, dtype=torch.long)
        x_t, noise = masked_q_sample(z, z, mask, t, _Sched(), noise_obs=no, noise_hidden=nh)
        expected = mask * 3.0 + (1.0 - mask) * 7.0
        self.assertTrue(torch.equal(noise, expected))
        self.assertTrue(torch.equal(x_t, expected))


class DataValidationTests(unittest.TestCase):
    """Strict validation must not conflate anchor raster location with validity."""

    @staticmethod
    def _sample_arrays(res: int = 8):
        known = torch.zeros(res, res, res, dtype=torch.bool)
        known[: res // 2] = True  # upper half of the observation lattice unknown
        partial = torch.full((res, res, res), conventions.FILLER, dtype=torch.float32)
        partial[known] = 0.0
        target = torch.zeros((res, res, res), dtype=torch.float32)
        # The second anchor is sub-voxel: coordinate 0.7 rounds to voxel 6,
        # where known_mask is False.  Its exact zero constraint is independent
        # of that grid node.
        anchors = torch.tensor([[0.5, 0.5, 0.5], [0.7, 0.7, 0.7]], dtype=torch.float32)
        normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        return {
            "partial_sdf": partial,
            "known_mask": known,
            "target_sdf": target,
            "anchors": anchors,
            "normals": normals,
        }

    def test_anchor_raster_location_is_not_a_validity_criterion(self):
        import json
        import tempfile

        import numpy as np

        arrays = self._sample_arrays()
        errors, contract = validate_sample(arrays, source="synthetic", resolution=8)
        self.assertEqual(errors, [])
        self.assertEqual(contract, [])

        # The same sample must also load through the strict manifest path.
        with tempfile.TemporaryDirectory() as d:
            np.savez(Path(d) / "sample.npz", **{k: v.numpy() for k, v in arrays.items()})
            manifest = Path(d) / "one.json"
            manifest.write_text(
                json.dumps([{"object_id": "synthetic-subvoxel", "path": "sample.npz"}]),
                encoding="utf-8",
            )
            batch = next(iter(build_loader(manifest, resolution=8, strict=True)))
            self.assertEqual(tuple(batch.anchors.shape), (1, 2, 3))

    def test_genuine_anchor_violations_still_rejected(self):
        arrays = self._sample_arrays()
        arrays["anchors"] = torch.tensor([[1.5, 0.0, 0.0]], dtype=torch.float32)
        arrays["normals"] = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        _, contract = validate_sample(arrays, source="synthetic", resolution=8)
        self.assertTrue(any("outside the normalized" in c for c in contract))

        arrays = self._sample_arrays()
        arrays["normals"] = torch.tensor([[0.0, 0.0, 2.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        _, contract = validate_sample(arrays, source="synthetic", resolution=8)
        self.assertTrue(any("not unit length" in c for c in contract))


class CNFTests(unittest.TestCase):
    def test_exact_zero_and_normal_constraints(self):
        res = 16
        cnf = HermiteGaussianCNF.for_resolution(res, sigma_scale=2.0)
        torch.manual_seed(17)
        centers = torch.rand(12, 3, dtype=torch.float64) * 1.6 - 0.8
        normals = torch.randn(12, 3, dtype=torch.float64)
        normals = normals / normals.norm(dim=-1, keepdim=True)
        base = torch.randn(12, 4, dtype=torch.float64) * 0.1
        target = torch.zeros_like(base)
        target[:, 1:] = normals
        psi, diag = cnf.solve(base, target, centers, diagnostics=True)
        self.assertTrue(diag.ok)
        m = cnf.system_matrix(centers)
        ops = flatten_ops(base) + (m @ psi.unsqueeze(-1)).squeeze(-1)
        err = (ops - flatten_ops(target)).abs().max()
        self.assertLess(float(err), 1e-8)
        report = cnf.verify(base, target, psi, centers)
        self.assertLess(report["basis_consistency_inf"], 1e-8)

    def test_exact_value_and_gradient_at_centers(self):
        """Evaluate the correction independently: value and all xyz derivatives."""
        cnf = HermiteGaussianCNF.for_resolution(64, sigma_scale=2.0)
        torch.manual_seed(29)
        centers = torch.rand(32, 3, dtype=torch.float64) * 1.6 - 0.8
        normals = torch.randn(32, 3, dtype=torch.float64)
        normals = normals / normals.norm(dim=-1, keepdim=True)
        base = torch.randn(32, 4, dtype=torch.float64) * 0.1
        target = torch.zeros_like(base)
        target[:, 1:] = normals
        psi, diag = cnf.solve(base, target, centers, diagnostics=True)
        self.assertTrue(diag.ok)
        corr_val, corr_grad = cnf.correction(psi, centers, centers, with_grad=True)
        corrected = base + torch.stack(
            [corr_val, corr_grad[:, 0], corr_grad[:, 1], corr_grad[:, 2]], dim=-1
        )
        err = (corrected - target).abs()
        self.assertLess(float(err[:, 0].max()), 1e-8)
        self.assertLess(float(err[:, 1:].max()), 1e-8)
        self.assertLess(float(err.max()), 1e-8)

    def test_batched_solve_explicit_column_rhs(self):
        # A (B, 4n) RHS must not be mistaken for a matrix by torch.linalg.solve.
        res = 16
        cnf = HermiteGaussianCNF.for_resolution(res, sigma_scale=2.0)
        torch.manual_seed(23)
        bsz, n = 2, 6
        centers = torch.rand(bsz, n, 3, dtype=torch.float64) * 1.6 - 0.8
        normals = torch.randn(bsz, n, 3, dtype=torch.float64)
        normals = normals / normals.norm(dim=-1, keepdim=True)
        base = torch.randn(bsz, n, 4, dtype=torch.float64) * 0.1
        target = torch.zeros_like(base)
        target[..., 1:] = normals
        psi, diag = cnf.solve(base, target, centers, diagnostics=True)
        self.assertEqual(tuple(psi.shape), (bsz, 4 * n))
        self.assertTrue(diag.ok)
        m = cnf.system_matrix(centers)
        resid = (m @ psi.unsqueeze(-1)).squeeze(-1) - (flatten_ops(target) - flatten_ops(base))
        self.assertLess(float(resid.abs().max()), 1e-8)

    def test_gradient_through_solve_is_finite(self):
        res = 16
        cnf = HermiteGaussianCNF.for_resolution(res, sigma_scale=2.0)
        centers = torch.rand(8, 3, dtype=torch.float64) * 1.6 - 0.8
        normals = torch.randn(8, 3, dtype=torch.float64)
        normals = normals / normals.norm(dim=-1, keepdim=True)
        base = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
        target = torch.zeros_like(base)
        target[:, 1:] = normals
        psi, _ = cnf.solve(base, target, centers)
        loss = psi.pow(2).mean()
        loss.backward()
        self.assertTrue(base.grad is not None)
        self.assertTrue(torch.isfinite(base.grad).all())


if __name__ == "__main__":
    unittest.main()
