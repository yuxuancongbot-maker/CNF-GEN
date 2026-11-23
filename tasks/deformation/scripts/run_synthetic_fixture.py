from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from gcnf_deformation import correct_with_fixed_gaussian_rbf, displacement_from_mapping
from learned_basis_hybrid_cnf.projection import hard_project_H

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tasks" / "deformation" / "configs" / "synthetic_fixture.json"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = [*(f"v {x:.12g} {y:.12g} {z:.12g}" for x, y, z in vertices)]
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _mesh_diagnostics(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    reference_normals: torch.Tensor | None = None,
) -> dict[str, float | int]:
    triangles = vertices[faces]
    normals = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = torch.linalg.vector_norm(normals, dim=1) * 0.5
    flipped_count = 0
    if reference_normals is not None:
        flipped_count = int(((normals * reference_normals).sum(dim=-1) <= 0).sum().item())
    return {
        "degenerate_face_count": int((areas <= 1e-12).sum().item()),
        "face_normal_flip_count": flipped_count,
        "min_face_area": areas.min().item(),
        "mean_face_area": areas.mean().item(),
    }


def _expected_failure(
    case_id: str,
    handles: torch.Tensor,
    targets: torch.Tensor,
    prior,
    length_scale: float,
) -> dict[str, str]:
    try:
        correct_with_fixed_gaussian_rbf(
            prior,
            handles,
            targets,
            length_scale=length_scale,
            projector=hard_project_H,
        )
    except (RuntimeError, ValueError) as exc:
        return {"case": case_id, "status": "rejected_as_expected", "reason": str(exc)}
    raise AssertionError(f"failure fixture {case_id!r} was unexpectedly accepted")


def _artifact(path: Path, role: str) -> dict[str, str]:
    return {"uri": path.name, "sha256": _sha256(path), "role": role}


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"-{uuid4().hex[:8]}-synthetic-affine-v1"
    )
    run_dir = ROOT / "runs" / "deformation" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config_out = run_dir / "config.json"
    _write_json(config_out, config)

    git_revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    common = {
        "schema_version": "0.1",
        "run_id": run_id,
        "task": "deformation",
        "method": "synthetic_affine_prior_fixed_gaussian_rbf_hard_projector",
        "git_revision": git_revision,
        "dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
            ).strip()
        ),
        "config_path": "config.json",
        "config_sha256": _sha256(config_out),
        "dataset": {"id": config["fixture_id"], "revision": "fixture-v1", "split": "fixed-handles"},
        "seeds": {"training": None, "sampling": None},
        "attempts": 1,
    }

    try:
        dtype = torch.float64
        vertices = torch.tensor(config["mesh_vertices"], dtype=dtype)
        faces = torch.tensor(config["mesh_faces"], dtype=torch.long)
        indices = config["handle_vertex_indices"]
        handles = vertices[indices]
        linear = torch.tensor(config["prior"]["linear"], dtype=dtype)
        offset = torch.tensor(config["prior"]["offset"], dtype=dtype)

        def mapping(points: torch.Tensor) -> torch.Tensor:
            return points + points @ linear.T + offset

        def prior(points: torch.Tensor) -> torch.Tensor:
            return displacement_from_mapping(mapping, points)

        target_offsets = torch.tensor(config["handle_displacement_offsets"], dtype=dtype)
        targets = prior(handles) + target_offsets
        start = time.perf_counter()
        corrected_field, diagnostics = correct_with_fixed_gaussian_rbf(
            prior,
            handles,
            targets,
            length_scale=config["rbf"]["length_scale"],
            projector=hard_project_H,
            max_handle_error=config["max_handle_error"],
        )
        solver_seconds = time.perf_counter() - start

        probes = torch.tensor(config["probe_points"], dtype=dtype)
        queries = torch.cat([vertices, probes], dim=0)
        raw_query = prior(queries)
        corrected_query = corrected_field(queries)
        raw_vertices = vertices + prior(vertices)
        corrected_vertices = vertices + corrected_field(vertices)
        raw_handle = prior(handles)
        corrected_handle = corrected_field(handles)
        raw_errors = torch.linalg.vector_norm(raw_handle - targets, dim=-1)
        corrected_errors = torch.linalg.vector_norm(corrected_handle - targets, dim=-1)
        correction_norms = torch.linalg.vector_norm(corrected_query - raw_query, dim=-1)
        source_normals = torch.linalg.cross(
            vertices[faces][:, 1] - vertices[faces][:, 0],
            vertices[faces][:, 2] - vertices[faces][:, 0],
        )
        source_mesh = _mesh_diagnostics(vertices, faces)
        raw_mesh = _mesh_diagnostics(raw_vertices, faces, source_normals)
        corrected_mesh = _mesh_diagnostics(corrected_vertices, faces, source_normals)
        rbf_constraint_matrix = torch.exp(
            -torch.cdist(handles, handles).square() / (2.0 * config["rbf"]["length_scale"] ** 2)
        )

        np.savez_compressed(
            run_dir / "field_queries.npz",
            source_vertices=vertices.numpy(),
            faces=faces.numpy(),
            handles=handles.numpy(),
            target_displacements=targets.numpy(),
            query_points=queries.numpy(),
            raw_query_displacements=raw_query.numpy(),
            corrected_query_displacements=corrected_query.numpy(),
            rbf_constraint_matrix=rbf_constraint_matrix.numpy(),
            rbf_coefficients=corrected_field.coefficients.numpy(),
        )
        _write_obj(run_dir / "source.obj", vertices.numpy(), faces.numpy())
        _write_obj(run_dir / "raw_prior.obj", raw_vertices.numpy(), faces.numpy())
        _write_obj(run_dir / "cnf_corrected.obj", corrected_vertices.numpy(), faces.numpy())

        failure_cases = [
            _expected_failure(
                "duplicate_handle_consistent_positions_rejected",
                torch.cat([handles[:2], handles[:1]], dim=0),
                torch.cat([targets[:2], targets[:1]], dim=0),
                prior,
                config["rbf"]["length_scale"],
            ),
            _expected_failure(
                "duplicate_handle_conflicting_positions_rejected",
                torch.cat([handles[:2], handles[:1]], dim=0),
                torch.cat([targets[:2], targets[0:1] + 0.1], dim=0),
                prior,
                config["rbf"]["length_scale"],
            ),
            _expected_failure(
                "near_coincident_handles_rank_deficient_rbf",
                torch.tensor([[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0]], dtype=dtype),
                torch.zeros((2, 3), dtype=dtype),
                prior,
                config["rbf"]["length_scale"],
            ),
        ]
        details = {
            "evidence_level": "synthetic_numerical_interface_validation_only",
            "config": config,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cpu",
                "dtype": str(dtype),
                "solver": "learned_basis_hybrid_cnf.projection.hard_project_H",
                "solver_seconds_including_rbf_build": solver_seconds,
            },
            "constraint_diagnostics": diagnostics,
            "raw_handle_errors": raw_errors.tolist(),
            "corrected_handle_errors": corrected_errors.tolist(),
            "query_correction_norm_mean": correction_norms.mean().item(),
            "query_correction_norm_max": correction_norms.max().item(),
            "mesh_diagnostics": {
                "source": source_mesh,
                "raw_prior": raw_mesh,
                "cnf_corrected": corrected_mesh,
            },
            "expected_failure_cases": failure_cases,
            "scope_limit": (
                "does not validate NSDP, real meshes, global injectivity, or physical plausibility"
            ),
        }
        details_path = run_dir / "diagnostics.json"
        _write_json(details_path, details)

        metrics = []
        for name, values, stage in (
            ("handle_position_error_max", raw_errors, "raw"),
            ("handle_position_error_mean", raw_errors, "raw"),
            ("handle_position_error_max", corrected_errors, "constrained"),
            ("handle_position_error_mean", corrected_errors, "constrained"),
        ):
            statistic = "max" if name.endswith("max") else "mean"
            value = values.max().item() if statistic == "max" else values.mean().item()
            metrics.append(
                {
                    "name": name,
                    "value": value,
                    "unit": "normalized_position_units",
                    "stage": stage,
                    "scope": "finite_constraints",
                    "count": int(values.numel()),
                    "statistic": statistic,
                    **({"tolerance": config["max_handle_error"]} if stage == "constrained" else {}),
                }
            )
        metrics.extend(
            [
                {
                    "name": "query_correction_norm_mean",
                    "value": correction_norms.mean().item(),
                    "unit": "normalized_position_units",
                    "stage": "constrained",
                    "scope": "intermediate_field",
                    "count": int(correction_norms.numel()),
                    "statistic": "mean",
                },
                {
                    "name": "solver_seconds_including_rbf_build",
                    "value": solver_seconds,
                    "unit": "seconds",
                    "stage": "constrained",
                    "scope": "intermediate_field",
                    "count": 1,
                    "statistic": "single",
                },
            ]
        )
        artifacts = [
            _artifact(config_out, "parsed_config"),
            _artifact(run_dir / "field_queries.npz", "field_query_matrix_and_coefficients"),
            _artifact(details_path, "solver_geometry_and_failure_diagnostics"),
            _artifact(run_dir / "source.obj", "source_mesh"),
            _artifact(run_dir / "raw_prior.obj", "raw_prior_mesh"),
            _artifact(run_dir / "cnf_corrected.obj", "cnf_corrected_mesh"),
        ]
        summary = {
            **common,
            "method": "synthetic_affine_prior_fixed_gaussian_rbf_hard_projector",
            "status": "succeeded",
            "failures": [],
            "metrics": metrics,
            "artifacts": artifacts,
        }
        _write_json(run_dir / "run.json", summary)
        print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))
        return 0
    except Exception as exc:
        failure_path = run_dir / "failure.json"
        _write_json(failure_path, {"type": type(exc).__name__, "message": str(exc)})
        summary = {
            **common,
            "status": "failed",
            "failures": [f"{type(exc).__name__}: {exc}"],
            "metrics": [],
            "artifacts": [
                _artifact(config_out, "parsed_config"),
                _artifact(failure_path, "failure_diagnostic"),
            ],
        }
        _write_json(run_dir / "run.json", summary)
        print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
