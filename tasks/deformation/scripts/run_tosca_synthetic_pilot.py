from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "tasks" / "deformation" / "configs" / "tosca_synthetic_pilot.json"
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "deformation"
sys.path.insert(0, str(ROOT / "tasks" / "deformation" / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "learned_basis_hybrid_cnf" / "src"))

from gcnf_deformation.correction import (  # noqa: E402
    correct_with_fixed_gaussian_rbf,
    fit_soft_gaussian_rbf_correction,
)
from learned_basis_hybrid_cnf.projection import hard_project_H  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = [*(f"v {x:.12g} {y:.12g} {z:.12g}" for x, y, z in vertices)]
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] == "f":
            indices = [int(value.split("/")[0]) - 1 for value in fields[1:4]]
            faces.append(indices)
    values = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"invalid triangular OBJ: {path}")
    if not np.isfinite(values).all() or np.any(triangles < 0) or np.any(triangles >= len(values)):
        raise ValueError(f"non-finite or out-of-range OBJ: {path}")
    return values, triangles


def _fps_indices(vertices: np.ndarray, count: int, start_index: int) -> np.ndarray:
    if count <= 0 or count > len(vertices):
        raise ValueError("handle count must be positive and no larger than the vertex count")
    selected = np.empty(count, dtype=np.int64)
    selected[0] = start_index
    distances = np.full(len(vertices), np.inf, dtype=np.float64)
    for step in range(1, count):
        delta = vertices - vertices[selected[step - 1]]
        distances = np.minimum(distances, np.einsum("ij,ij->i", delta, delta))
        distances[selected[:step]] = -1.0
        selected[step] = int(np.argmax(distances))
    return selected


def _affine(points: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    linear = np.asarray(spec["linear"], dtype=np.float64)
    offset = np.asarray(spec["offset"], dtype=np.float64)
    return points @ linear.T + offset


def _field_values(
    points: np.ndarray, spec: dict[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    affine = _affine(points, spec)
    prior_scale = float(spec["prior_scale"])
    prior = prior_scale * affine
    center = np.asarray(spec.get("region_center", spec.get("bump_center")), dtype=np.float64)
    radius = float(spec["region_radius"])
    region = np.linalg.norm(points - center, axis=1) <= radius
    if spec["type"] == "affine":
        truth = affine
    elif spec["type"] == "affine_plus_gaussian":
        bump_center = np.asarray(spec["bump_center"], dtype=np.float64)
        direction = np.asarray(spec["bump_direction"], dtype=np.float64)
        direction = direction / np.linalg.norm(direction)
        sigma = float(spec["bump_sigma"])
        amplitude = float(spec["bump_amplitude"])
        weights = np.exp(-0.5 * np.sum((points - bump_center) ** 2, axis=1) / (sigma**2))
        truth = affine + weights[:, None] * amplitude * direction[None, :]
    else:
        raise ValueError(f"unknown field type: {spec['type']}")
    if not np.isfinite(truth).all() or not np.isfinite(prior).all():
        raise ValueError("field produced non-finite values")
    return prior, truth, region


def _summary(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _ratio_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("geometry ratio is non-positive or non-finite")
    log_values = np.abs(np.log(values))
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p95_factor": float(np.exp(np.percentile(log_values, 95))),
        "max_factor": float(np.exp(np.max(log_values))),
    }


def _geometry_metrics(
    source: np.ndarray,
    deformed: np.ndarray,
    faces: np.ndarray,
    region: np.ndarray,
) -> dict[str, object]:
    source_tri = source[faces]
    deformed_tri = deformed[faces]
    source_cross = np.cross(
        source_tri[:, 1] - source_tri[:, 0], source_tri[:, 2] - source_tri[:, 0]
    )
    deformed_cross = np.cross(
        deformed_tri[:, 1] - deformed_tri[:, 0], deformed_tri[:, 2] - deformed_tri[:, 0]
    )
    source_areas = np.linalg.norm(source_cross, axis=1) * 0.5
    deformed_areas = np.linalg.norm(deformed_cross, axis=1) * 0.5
    if np.any(source_areas <= 1e-12) or np.any(deformed_areas <= 1e-12):
        raise ValueError("source or deformed mesh has degenerate faces")
    face_flips = int(np.count_nonzero(np.einsum("ij,ij->i", source_cross, deformed_cross) <= 0))
    face_region = region[faces].all(axis=1)
    face_complement = ~region[faces].any(axis=1)
    face_edge_pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.unique(np.sort(face_edge_pairs, axis=1), axis=0)
    source_edges = np.linalg.norm(source[edges[:, 0]] - source[edges[:, 1]], axis=1)
    deformed_edges = np.linalg.norm(deformed[edges[:, 0]] - deformed[edges[:, 1]], axis=1)
    edge_ratios = deformed_edges / source_edges
    edge_region = region[edges].all(axis=1)
    edge_complement = ~region[edges].any(axis=1)

    tangent_1 = source_tri[:, 1] - source_tri[:, 0]
    tangent_1 /= np.linalg.norm(tangent_1, axis=1)[:, None]
    tangent_normal = source_cross / np.linalg.norm(source_cross, axis=1)[:, None]
    tangent_2 = np.cross(tangent_normal, tangent_1)
    deformed_tangent_1 = deformed_tri[:, 1] - deformed_tri[:, 0]
    deformed_tangent_1 /= np.linalg.norm(deformed_tangent_1, axis=1)[:, None]
    deformed_tangent_normal = deformed_cross / np.linalg.norm(deformed_cross, axis=1)[:, None]
    deformed_tangent_2 = np.cross(deformed_tangent_normal, deformed_tangent_1)
    source_matrix = np.stack(
        [
            np.stack(
                [
                    np.einsum("ij,ij->i", tangent_1, tangent_1),
                    np.einsum("ij,ij->i", tangent_1, source_tri[:, 2] - source_tri[:, 0]),
                ],
                axis=1,
            ),
            np.stack(
                [
                    np.einsum("ij,ij->i", tangent_2, tangent_1),
                    np.einsum("ij,ij->i", tangent_2, source_tri[:, 2] - source_tri[:, 0]),
                ],
                axis=1,
            ),
        ],
        axis=1,
    )
    deformed_matrix = np.stack(
        [
            np.stack(
                [
                    np.einsum(
                        "ij,ij->i", deformed_tangent_1, deformed_tri[:, 1] - deformed_tri[:, 0]
                    ),
                    np.einsum(
                        "ij,ij->i", deformed_tangent_1, deformed_tri[:, 2] - deformed_tri[:, 0]
                    ),
                ],
                axis=1,
            ),
            np.stack(
                [
                    np.einsum(
                        "ij,ij->i", deformed_tangent_2, deformed_tri[:, 1] - deformed_tri[:, 0]
                    ),
                    np.einsum(
                        "ij,ij->i", deformed_tangent_2, deformed_tri[:, 2] - deformed_tri[:, 0]
                    ),
                ],
                axis=1,
            ),
        ],
        axis=1,
    )
    tangent_map = deformed_matrix @ np.linalg.inv(source_matrix)
    tangent_sv = np.linalg.svd(tangent_map, compute_uv=False)

    def region_geometry(face_mask: np.ndarray, edge_mask: np.ndarray) -> dict[str, object]:
        return {
            "face_count": int(face_mask.sum()),
            "edge_count": int(edge_mask.sum()),
            "edge_length_ratio": _ratio_summary(edge_ratios[edge_mask]),
            "face_area_ratio": _ratio_summary((deformed_areas / source_areas)[face_mask]),
            "tangent_sigma_max": _summary(tangent_sv[face_mask, 0]),
            "tangent_sigma_min": _summary(tangent_sv[face_mask, 1]),
        }

    return {
        "degenerate_face_count": int(np.count_nonzero(deformed_areas <= 1e-12)),
        "face_normal_flip_count": face_flips,
        "edge_length_ratio": _ratio_summary(edge_ratios),
        "face_area_ratio": _ratio_summary(deformed_areas / source_areas),
        "tangent_sigma_max": _summary(tangent_sv[:, 0]),
        "tangent_sigma_min": _summary(tangent_sv[:, 1]),
        "regions": {
            "local": region_geometry(face_region, edge_region),
            "complement": region_geometry(face_complement, edge_complement),
        },
    }


def _artifact(path: Path, role: str) -> dict[str, str]:
    return {"uri": path.name, "sha256": _sha256(path), "role": role}


def _metric(
    name: str,
    value: float,
    unit: str,
    stage: str,
    scope: str,
    count: int,
    statistic: str = "single",
    tolerance: float | None = None,
) -> dict[str, object]:
    metric: dict[str, object] = {
        "name": name,
        "value": float(value),
        "unit": unit,
        "stage": stage,
        "scope": scope,
        "count": int(count),
        "statistic": statistic,
    }
    if tolerance is not None:
        metric["tolerance"] = float(tolerance)
    return metric


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CPU pilot on real TOSCA cat0 with analytic target fields"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    started = time.perf_counter()
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"-{uuid4().hex[:8]}-tosca-cat0-synthetic-pilot"
    )
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    failures: list[str] = []
    metrics: list[dict[str, object]] = []
    artifacts: list[dict[str, str]] = []
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        resolved_config_path = run_dir / "resolved_config.json"
        _write_json(resolved_config_path, config)
        artifacts.append(_artifact(resolved_config_path, "resolved_config"))
        source_path = (ROOT / config["source"]["path"]).resolve()
        vertices, faces = _read_obj(source_path)
        expected = config["source"]
        if vertices.shape != (expected["expected_vertex_count"], 3) or faces.shape != (
            expected["expected_face_count"],
            3,
        ):
            raise ValueError(
                f"unexpected source mesh shape: vertices={vertices.shape}, faces={faces.shape}"
            )
        source_hash = _sha256(source_path)
        _write_obj(run_dir / "source.obj", vertices, faces)
        artifacts.append(_artifact(run_dir / "source.obj", "source_mesh"))
        handle_indices_by_count = {
            int(count): _fps_indices(
                vertices, int(count), int(config["sampling"]["fps_start_index"])
            )
            for count in config["sampling"]["handle_counts"]
        }
        diagnostics: dict[str, object] = {
            "evidence_level": config["evidence_level"],
            "source": {
                "path": str(source_path),
                "sha256": source_hash,
                "vertex_count": len(vertices),
                "face_count": len(faces),
            },
            "sampling": {
                "seed": config["sampling"]["seed"],
                "fps_start_index": config["sampling"]["fps_start_index"],
                "indices": {str(k): v.tolist() for k, v in handle_indices_by_count.items()},
            },
            "fields": {},
            "cells": [],
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cpu",
                "dtype": "float64",
            },
        }
        for field_name, field_spec in config["fields"].items():
            prior_values, truth_values, region = _field_values(vertices, field_spec)
            target_vertices = vertices + truth_values
            target_path = run_dir / f"target_{field_name}.obj"
            _write_obj(target_path, target_vertices, faces)
            artifacts.append(_artifact(target_path, f"analytic_target_{field_name}"))
            diagnostics["fields"][field_name] = {
                "parameters": field_spec,
                "region_vertex_count": int(region.sum()),
                "region_fraction": float(region.mean()),
            }
            for handle_count, indices in handle_indices_by_count.items():
                handles = torch.as_tensor(vertices[indices], dtype=torch.float64)
                targets = torch.as_tensor(truth_values[indices], dtype=torch.float64)

                def prior(points: torch.Tensor, spec=field_spec) -> torch.Tensor:
                    points_np = points.detach().cpu().numpy()
                    query_prior, _, _ = _field_values(points_np, spec)
                    return torch.as_tensor(query_prior, dtype=torch.float64)

                case_prefix = f"case_{field_name}_k{handle_count}"
                hard_field, hard_info = correct_with_fixed_gaussian_rbf(
                    prior,
                    handles,
                    targets,
                    length_scale=float(config["rbf"]["length_scale"]),
                    projector=hard_project_H,
                    max_handle_error=float(config["rbf"]["max_handle_error"]),
                )
                method_fields: dict[str, np.ndarray] = {"raw": prior_values}
                method_fields["hard"] = hard_field(
                    torch.as_tensor(vertices, dtype=torch.float64)
                ).numpy()
                soft_infos: dict[str, object] = {}
                for lambda_value in config["rbf"]["soft_ridge_lambdas"]:
                    soft_field, soft_info = fit_soft_gaussian_rbf_correction(
                        prior,
                        handles,
                        targets,
                        length_scale=float(config["rbf"]["length_scale"]),
                        ridge_lambda=float(lambda_value),
                    )
                    label = f"soft_lambda_{float(lambda_value):g}".replace(".", "p")
                    method_fields[label] = soft_field(
                        torch.as_tensor(vertices, dtype=torch.float64)
                    ).numpy()
                    soft_infos[label] = soft_info
                case: dict[str, object] = {
                    "field": field_name,
                    "handle_count": handle_count,
                    "handle_indices": indices.tolist(),
                    "hard_solver": hard_info,
                    "soft_solvers": soft_infos,
                    "methods": {},
                }
                npz_values: dict[str, np.ndarray] = {
                    "source_vertices": vertices,
                    "faces": faces,
                    "handle_indices": indices,
                    "handle_points": vertices[indices],
                    "target_displacements": truth_values[indices],
                    "target_displacements_all": truth_values,
                    "region_mask": region.astype(np.uint8),
                }
                for method_name, displacement in method_fields.items():
                    method_stage = "raw" if method_name == "raw" else "constrained"
                    handle_error = np.linalg.norm(
                        displacement[indices] - truth_values[indices], axis=1
                    )
                    target_error = np.linalg.norm(displacement - truth_values, axis=1)
                    correction = np.linalg.norm(displacement - prior_values, axis=1)
                    geom = _geometry_metrics(vertices, vertices + displacement, faces, region)
                    case["methods"][method_name] = {
                        "handle_error": _summary(handle_error),
                        "target_error": _summary(target_error),
                        "correction_norm": _summary(correction),
                        "geometry": geom,
                    }
                    npz_values[f"{method_name}_displacements"] = displacement
                    metrics.extend(
                        [
                            _metric(
                                f"{case_prefix}.{method_name}.handle_error_max",
                                handle_error.max(),
                                "normalized_position_units",
                                method_stage,
                                "finite_constraints",
                                len(handle_error),
                                "max",
                                float(config["rbf"]["max_handle_error"])
                                if method_name == "hard"
                                else None,
                            ),
                            _metric(
                                f"{case_prefix}.{method_name}.target_rmse",
                                np.sqrt(np.mean(target_error**2)),
                                "normalized_position_units",
                                method_stage,
                                "continuous_domain",
                                len(target_error),
                            ),
                        ]
                    )
                    mesh_path = run_dir / f"{case_prefix}_{method_name}.obj"
                    _write_obj(mesh_path, vertices + displacement, faces)
                    artifacts.append(_artifact(mesh_path, f"{method_name}_mesh_{case_prefix}"))
                case_npz = run_dir / f"{case_prefix}.npz"
                np.savez_compressed(case_npz, **npz_values)
                artifacts.append(_artifact(case_npz, f"field_query_case_{case_prefix}"))
                diagnostics["cells"].append(case)
        diagnostics_path = run_dir / "diagnostics.json"
        _write_json(diagnostics_path, diagnostics)
        artifacts.append(_artifact(diagnostics_path, "pilot_diagnostics"))
        metrics_path = run_dir / "metrics.json"
        _write_json(metrics_path, metrics)
        artifacts.append(_artifact(metrics_path, "pilot_metrics"))
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
            ).strip()
        )
        summary = {
            "schema_version": "0.1",
            "run_id": run_id,
            "task": "deformation",
            "method": "tosca_cat0_synthetic_target_fixed_rbf_pilot",
            "status": "succeeded",
            "git_revision": revision,
            "dirty": dirty,
            "config_path": "resolved_config.json",
            "config_sha256": _sha256(resolved_config_path),
            "dataset": {
                "id": "TOSCA/cat0-normalized-synthetic-target",
                "revision": source_hash,
                "split": "single_source_analytic_targets",
            },
            "seeds": {"training": None, "sampling": int(config["sampling"]["seed"])},
            "attempts": 1,
            "failures": failures,
            "metrics": metrics,
            "artifacts": artifacts,
        }
        _write_json(run_dir / "run.json", summary)
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "status": "succeeded",
                    "cell_count": len(diagnostics["cells"]),
                    "elapsed_seconds": time.perf_counter() - started,
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        failure_path = run_dir / "failure.json"
        _write_json(failure_path, {"type": type(exc).__name__, "message": str(exc)})
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        summary = {
            "schema_version": "0.1",
            "run_id": run_id,
            "task": "deformation",
            "method": "tosca_cat0_synthetic_target_fixed_rbf_pilot",
            "status": "failed",
            "git_revision": revision,
            "dirty": True,
            "config_path": "resolved_config.json",
            "config_sha256": None,
            "dataset": {
                "id": "TOSCA/cat0-normalized-synthetic-target",
                "revision": "unknown",
                "split": "single_source_analytic_targets",
            },
            "seeds": {"training": None, "sampling": 20261122},
            "attempts": 1,
            "failures": [f"{type(exc).__name__}: {exc}"],
            "metrics": [],
            "artifacts": artifacts + [_artifact(failure_path, "failure_diagnostic")],
        }
        _write_json(run_dir / "run.json", summary)
        print(
            json.dumps({"run_dir": str(run_dir), "status": "failed", "error": str(exc)}, indent=2)
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
