from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from gcnf_deformation import (
    NSDPPointQuery,
    correct_with_fixed_gaussian_rbf,
    displacement_from_mapping,
    fit_soft_gaussian_rbf_correction,
    moved_vertex_candidate_pool,
    resolve_explicit_handle_constraints,
    resolve_handle_count,
    resolve_userhandle_region,
    resolve_userhandle_translation,
)
from gcnf_deformation.normalization import gaps_pca_scale_matrix, prepare_off_mesh
from gcnf_deformation.progress import record_progress_event
from learned_basis_hybrid_cnf.projection import hard_project_H


class TestDeformationFieldCorrection(unittest.TestCase):
    def setUp(self) -> None:
        self.handles = torch.tensor(
            [[-1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]],
            dtype=torch.float64,
        )
        self.affine = torch.tensor(
            [[0.05, 0.01, 0.0], [0.0, -0.03, 0.02], [0.01, 0.0, 0.04]],
            dtype=torch.float64,
        )
        self.offset = torch.tensor([0.01, -0.02, 0.0], dtype=torch.float64)

    def prior_displacement(self, points: torch.Tensor) -> torch.Tensor:
        return points @ self.affine.T + self.offset

    def projector(self, H, c, b):
        return hard_project_H(H, c, b)

    def test_nsdp_point_query_passes_normalized_context_and_returns_positions(self):
        class FakeNSDP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.05]))
                self.received = None

            def forward(self, queries, surface_src, surface_tgt, handle_mask):
                self.received = (queries, surface_src, surface_tgt, handle_mask)
                return queries + self.offset

        model = FakeNSDP()
        surface_src = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
        surface_tgt = surface_src + torch.tensor([0.0, 0.1, 0.0])
        handle_mask = torch.tensor([1.0, 0.0])
        adapter = NSDPPointQuery(model, surface_src, surface_tgt, handle_mask)
        queries = torch.tensor([[0.2, 0.3, 0.4], [-0.1, 0.1, 0.0]])

        mapped = adapter.map_points(queries)

        self.assertEqual(tuple(mapped.shape), (2, 3))
        self.assertTrue(torch.allclose(mapped, queries + model.offset))
        self.assertEqual(tuple(model.received[0].shape), (1, 2, 3))
        self.assertEqual(tuple(model.received[1].shape), (1, 2, 3))
        self.assertEqual(tuple(model.received[2].shape), (1, 2, 3))
        self.assertTrue(torch.equal(model.received[3], handle_mask.reshape(1, 2, 1)))

    def test_mapping_to_displacement_subtracts_source_position(self):
        points = self.handles[:2]

        def mapping(query):
            return query + self.prior_displacement(query)

        actual = displacement_from_mapping(mapping, points)
        self.assertTrue(torch.allclose(actual, self.prior_displacement(points)))

    def test_nonfinite_mapping_query_is_rejected(self):
        points = self.handles[:1].clone()
        points[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            displacement_from_mapping(lambda query: query, points)

    def test_fixed_rbf_satisfies_handles_and_matches_direct_solve(self):
        target_offset = torch.tensor(
            [[0.06, 0.0, 0.0], [-0.02, 0.05, 0.0], [0.0, -0.03, 0.04], [0.01, 0.0, -0.05]],
            dtype=torch.float64,
        )
        targets = self.prior_displacement(self.handles) + target_offset
        field, diagnostics = correct_with_fixed_gaussian_rbf(
            self.prior_displacement,
            self.handles,
            targets,
            length_scale=1.5,
            projector=self.projector,
        )

        actual = field(self.handles)
        self.assertLessEqual(torch.linalg.vector_norm(actual - targets, dim=-1).max(), 1e-5)

        delta = self.handles[:, None] - self.handles[None, :]
        matrix = torch.exp(-delta.square().sum(dim=-1) / (2.0 * 1.5**2))
        expected_coefficients = torch.linalg.solve(
            matrix, (targets - self.prior_displacement(self.handles))
        ).T
        self.assertTrue(torch.allclose(field.coefficients, expected_coefficients, atol=1e-12))
        self.assertEqual(diagnostics["solver"]["rank"], self.handles.shape[0])
        self.assertLess(diagnostics["corrected_handle_error_max"], 1e-8)

    def test_soft_rbf_matches_regularized_least_squares_objective(self):
        target_offset = torch.tensor(
            [[0.06, 0.0, 0.0], [-0.02, 0.05, 0.0], [0.0, -0.03, 0.04], [0.01, 0.0, -0.05]],
            dtype=torch.float64,
        )
        targets = self.prior_displacement(self.handles) + target_offset
        ridge_lambda = 0.1
        field, diagnostics = fit_soft_gaussian_rbf_correction(
            self.prior_displacement,
            self.handles,
            targets,
            length_scale=1.5,
            ridge_lambda=ridge_lambda,
        )

        delta = self.handles[:, None] - self.handles[None, :]
        kernel = torch.exp(-delta.square().sum(dim=-1) / (2.0 * 1.5**2))
        residual = target_offset
        expected_coefficients = torch.linalg.solve(
            kernel.T @ kernel + ridge_lambda * torch.eye(4, dtype=torch.float64),
            kernel.T @ residual,
        ).T
        self.assertTrue(torch.allclose(field.coefficients, expected_coefficients, atol=1e-12))
        self.assertGreater(diagnostics["handle_error_max"], 0.0)
        self.assertEqual(diagnostics["constraint_matrix_rank"], 4)

    def test_moved_vertex_candidate_pool_is_deterministic(self):
        vertices = np.array([[float(x), 0.0, 0.0] for x in range(7)])
        moved_indices = np.arange(7)

        all_pool, all_info = moved_vertex_candidate_pool(
            vertices, moved_indices, candidate_pool="all_moved_vertices"
        )
        inner_pool, inner_info = moved_vertex_candidate_pool(
            vertices,
            moved_indices,
            candidate_pool="nearest_radius_quantile",
            radius_quantile=0.5,
        )

        np.testing.assert_array_equal(all_pool, moved_indices)
        np.testing.assert_array_equal(inner_pool, [1, 2, 3, 4, 5])
        self.assertEqual(all_info["candidate_count"], 7)
        self.assertEqual(inner_info["candidate_count"], 5)
        self.assertAlmostEqual(inner_info["radius_cutoff"], 2.0)

    def test_userhandle_translation_override_preserves_other_components(self):
        base = {"xtrans": -0.15, "ytrans": -0.2, "ztrans": -0.2, "head": True}

        resolved = resolve_userhandle_translation(base, {"xtrans": 0.15})

        self.assertEqual(resolved, {"xtrans": 0.15, "ytrans": -0.2, "ztrans": -0.2})
        self.assertTrue(base["head"])
        self.assertEqual(base["xtrans"], -0.15)

    def test_userhandle_translation_override_rejects_unknown_and_nonfinite_values(self):
        base = {"xtrans": -0.15, "ytrans": -0.2, "ztrans": -0.2}
        with self.assertRaisesRegex(ValueError, "only accepts"):
            resolve_userhandle_translation(base, {"head": False})
        with self.assertRaisesRegex(ValueError, "must be finite"):
            resolve_userhandle_translation(base, {"xtrans": float("nan")})

    def test_userhandle_region_override_changes_only_region_flags(self):
        base = {
            "head": True,
            "tail": False,
            "frontleftfoot": False,
            "frontrightfoot": False,
            "behindleftfoot": False,
            "behindrightfoot": False,
            "xtrans": -0.15,
        }

        resolved = resolve_userhandle_region(base, "tail")

        self.assertTrue(resolved["tail"])
        self.assertFalse(any(value for key, value in resolved.items() if key != "tail"))
        self.assertTrue(base["head"])
        self.assertEqual(base["xtrans"], -0.15)
        with self.assertRaisesRegex(ValueError, "unsupported userhandle region"):
            resolve_userhandle_region(base, "whole_mesh")

    def test_length_scale_sensitivity_grid_is_positive_unique_and_contains_default(self):
        candidates = [0.07, 0.1, 0.14]
        default = 0.1

        self.assertTrue(all(np.isfinite(value) and value > 0 for value in candidates))
        self.assertEqual(len({f"{value:.12g}" for value in candidates}), len(candidates))
        self.assertIn(default, candidates)

    def test_handle_count_override_must_be_configured(self):
        self.assertEqual(resolve_handle_count(12, [6, 12, 24], 6), 6)
        with self.assertRaisesRegex(ValueError, "not in the configured candidates"):
            resolve_handle_count(12, [6, 12, 24], 10)
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            resolve_handle_count(12, [], None)
        self.assertEqual(resolve_handle_count(4, [4], 4), 4)
        with self.assertRaisesRegex(ValueError, "not in the configured candidates"):
            resolve_handle_count(4, [4], 3)

    def test_explicit_vertex_handles_hard_project_final_displacements(self):
        config = {
            "vertex_indices": [0, 1, 2, 3],
            "target_displacements_normalized": [
                [0.1, -0.05, 0.0],
                [-0.1, 0.0, 0.05],
                [0.0, 0.1, -0.05],
                [0.05, -0.1, 0.1],
            ],
        }
        indices, targets = resolve_explicit_handle_constraints(
            config, vertex_count=len(self.handles)
        )
        handles = self.handles[indices]
        target_tensor = torch.tensor(targets, dtype=torch.float64)
        field, _ = correct_with_fixed_gaussian_rbf(
            self.prior_displacement,
            handles,
            target_tensor,
            length_scale=1.5,
            projector=self.projector,
        )

        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertTrue(torch.allclose(field(handles), target_tensor, atol=1e-12, rtol=0))

    def test_explicit_vertex_handle_config_rejects_invalid_inputs(self):
        base = {
            "vertex_indices": [0, 1],
            "target_displacements_normalized": [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        }
        invalid = [
            ({**base, "vertex_indices": []}, "non-empty list"),
            ({**base, "vertex_indices": [0, 0]}, "must be unique"),
            ({**base, "vertex_indices": [0, 1.5]}, "must contain integers"),
            ({**base, "vertex_indices": [0, 4]}, "out-of-range"),
            ({**base, "target_displacements_normalized": [[0.1, 0.0, 0.0]]}, "one target"),
            ({**base, "target_displacements_normalized": [[0.1, 0.0], [0.0, 0.1, 0.0]]}, "shape"),
            (
                {
                    **base,
                    "target_displacements_normalized": [[float("nan"), 0.0, 0.0], [0.0, 0.1, 0.0]],
                },
                "finite",
            ),
            ({**base, "unexpected": True}, "requires exactly"),
        ]
        for config, message in invalid:
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, message):
                resolve_explicit_handle_constraints(config, vertex_count=len(self.handles))

    def test_consistent_duplicate_handle_positions_are_rejected(self):
        handles = torch.cat([self.handles[:2], self.handles[:1]], dim=0)
        targets = self.prior_displacement(handles)
        with self.assertRaisesRegex(ValueError, "coincident handle"):
            correct_with_fixed_gaussian_rbf(
                self.prior_displacement,
                handles,
                targets,
                length_scale=1.5,
                projector=self.projector,
            )

    def test_conflicting_duplicate_handle_positions_are_rejected(self):
        handles = torch.cat([self.handles[:2], self.handles[:1]], dim=0)
        targets = self.prior_displacement(handles)
        targets[-1] += 0.1
        with self.assertRaisesRegex(ValueError, "coincident handle"):
            correct_with_fixed_gaussian_rbf(
                self.prior_displacement,
                handles,
                targets,
                length_scale=1.5,
                projector=self.projector,
            )

    def test_nonfinite_handle_target_is_rejected(self):
        targets = torch.zeros_like(self.handles)
        targets[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-empty and finite"):
            correct_with_fixed_gaussian_rbf(
                self.prior_displacement,
                self.handles,
                targets,
                length_scale=1.5,
                projector=self.projector,
            )

    def test_rank_deficient_rbf_system_is_rejected_by_existing_solver(self):
        handles = torch.tensor([[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0]], dtype=torch.float64)
        targets = torch.zeros_like(handles)
        with self.assertRaises(RuntimeError):
            correct_with_fixed_gaussian_rbf(
                self.prior_displacement,
                handles,
                targets,
                length_scale=1.5,
                projector=self.projector,
            )

    def test_gaps_normalization_matrix_centers_and_scales_pca_variance(self):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]],
            dtype=np.float64,
        )
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
        matrix = gaps_pca_scale_matrix(vertices, faces)
        triangles = vertices[faces]
        source_areas = (
            np.linalg.norm(
                np.cross(
                    triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0],
                ),
                axis=1,
            )
            * 0.5
        )
        source_centroid = np.average(triangles.mean(axis=1), axis=0, weights=source_areas)
        scale = np.linalg.norm(matrix[0, :3])
        self.assertTrue(np.allclose(matrix[:3, :3], np.eye(3) * scale, atol=1e-14))
        self.assertTrue(np.allclose(matrix[:3, 3], -scale * source_centroid, atol=1e-14))

        transformed = vertices @ matrix[:3, :3].T + matrix[:3, 3]
        transformed_centers = transformed[faces].mean(axis=1)
        areas = (
            np.linalg.norm(
                np.cross(
                    transformed[faces][:, 1] - transformed[faces][:, 0],
                    transformed[faces][:, 2] - transformed[faces][:, 0],
                ),
                axis=1,
            )
            * 0.5
        )
        centroid = np.average(transformed_centers, axis=0, weights=areas)
        centered = transformed_centers - centroid
        covariance = (centered.T * areas) @ centered / areas.sum()
        self.assertLess(np.linalg.norm(centroid), 1e-14)
        self.assertAlmostEqual(np.linalg.svd(covariance, compute_uv=False)[0], 0.35**2, places=13)

    def test_prepare_off_mesh_writes_obj_and_transform(self):
        off = """OFF
4 4 6
0 0 0
2 0 0
0 1 0
0 0 0.5
3 0 2 1
3 0 1 3
3 0 3 2
3 1 2 3
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "mesh.off"
            input_path.write_text(off, encoding="ascii")
            mesh_path, matrix_path = prepare_off_mesh(input_path, root / "prepared")
            self.assertEqual(
                sum(line.startswith("v ") for line in mesh_path.read_text().splitlines()), 4
            )
            self.assertEqual(
                sum(line.startswith("f ") for line in mesh_path.read_text().splitlines()), 4
            )
            self.assertEqual(np.loadtxt(matrix_path).shape, (4, 4))

    def test_progress_events_append_valid_json_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "progress.jsonl"
            record_progress_event(
                path,
                elapsed_seconds=0.0,
                stage="model_load",
                progress_status="started",
            )
            record_progress_event(
                path,
                elapsed_seconds=1.25,
                stage="model_load",
                progress_status="completed",
                device="cuda:0",
            )
            record_progress_event(
                path,
                elapsed_seconds=1.5,
                stage="run_summary",
                progress_status="completed",
                status="succeeded",
                path="field_queries.npz",
            )

            events = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(
            [event["status"] for event in events],
            ["started", "completed", "completed"],
        )
        self.assertEqual(events[1]["elapsed_seconds"], 1.25)
        self.assertEqual(events[1]["details"]["device"], "cuda:0")
        self.assertEqual(events[2]["status"], "completed")
        self.assertEqual(events[2]["details"]["status"], "succeeded")
        self.assertEqual(events[2]["details"]["path"], "field_queries.npz")
        self.assertTrue(events[0]["timestamp_utc"].endswith("+00:00"))


if __name__ == "__main__":
    unittest.main()
