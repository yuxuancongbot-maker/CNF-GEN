from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent / "paired_region_metrics.json"
RUNS = {
    "cat0_spread": "20261117T093749740483Z-26694d72-nsdp-tosca-cat0-head-rbf-spread_fps",
    "cat0_inner50": "20261117T093835380245Z-d463b24e-nsdp-tosca-cat0-head-rbf-inner50_fps",
    "cat1_replay": (
        "20261117T174329923660Z-b7c727f2-nsdp-tosca-cat1-head_drag_1x-spread_fps-k12-ell0p1"
    ),
    "cat1_supplement": (
        "20261117T185507953524Z-4b59a097-nsdp-tosca-cat1-head_drag_1x-spread_fps-k12-ell0p1"
    ),
    "cat1_original": (
        "20261117T170605525891Z-0928a475-nsdp-tosca-cat1-head_drag_1x-spread_fps-k12-ell0p1"
    ),
}
METHOD_KEYS = {
    "raw_prior": "raw_query_displacements",
    "hard_cnf": "corrected_query_displacements",
}
CONSISTENCY_KEYS = (
    "source_vertices",
    "faces",
    "moved_vertex_indices",
    "selected_handle_vertex_indices",
    "target_displacements",
    "raw_query_displacements",
    "corrected_query_displacements",
    "soft_query_displacements",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_run_summary(run_dir: Path, expected_status: str) -> dict[str, Any]:
    summary_path = run_dir / "run.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != expected_status:
        raise ValueError(
            f"{run_dir.name}: expected status {expected_status!r}, got {summary.get('status')!r}"
        )
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{run_dir.name}: run summary has no indexed artifacts")
    for artifact in artifacts:
        relative_path = Path(artifact["uri"])
        artifact_path = run_dir / relative_path
        if not artifact_path.is_file():
            raise FileNotFoundError(f"{run_dir.name}: missing artifact {relative_path}")
        if _sha256(artifact_path) != artifact["sha256"]:
            raise ValueError(f"{run_dir.name}: hash mismatch for {relative_path}")
    return {
        "status": summary["status"],
        "run_json_sha256": _sha256(summary_path),
        "indexed_artifacts_verified": len(artifacts),
    }


def _load_arrays(run_dir: Path) -> dict[str, np.ndarray]:
    query_path = run_dir / "field_queries.npz"
    with np.load(query_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    required = {
        "source_vertices",
        "faces",
        "moved_vertex_indices",
        "selected_handle_vertex_indices",
        "target_displacements",
        "raw_query_displacements",
        "corrected_query_displacements",
        "soft_ridge_lambdas",
        "soft_query_displacements",
    }
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"{run_dir.name}: field_queries.npz missing {sorted(missing)}")
    return arrays


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    face_edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    return np.unique(np.sort(face_edges, axis=1), axis=0)


def _face_normals_and_areas(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return normals, np.linalg.norm(normals, axis=1) * 0.5


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if values.size == 0 or weights.size != values.size or np.any(weights <= 0):
        raise ValueError("weighted quantile needs non-empty values and positive aligned weights")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative_weight = np.cumsum(weights[order])
    cutoff = quantile * cumulative_weight[-1]
    return float(sorted_values[np.searchsorted(cumulative_weight, cutoff, side="left")])


def _principal_stretches(
    source: np.ndarray, deformed: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two singular values of each triangle's piecewise-linear tangent map."""
    source_triangles = source[faces]
    deformed_triangles = deformed[faces]
    source_edge_1 = source_triangles[:, 1] - source_triangles[:, 0]
    source_edge_2 = source_triangles[:, 2] - source_triangles[:, 0]
    edge_1_length = np.linalg.norm(source_edge_1, axis=1)
    if np.any(edge_1_length <= 0):
        raise ValueError("source contains a zero-length triangle edge")
    tangent_1 = source_edge_1 / edge_1_length[:, None]
    edge_2_projection = np.einsum("ij,ij->i", source_edge_2, tangent_1)
    edge_2_height = np.sqrt(
        np.maximum(
            np.einsum("ij,ij->i", source_edge_2, source_edge_2) - edge_2_projection**2,
            0.0,
        )
    )
    if np.any(edge_2_height <= 0):
        raise ValueError("source contains a degenerate triangle")

    deformed_edge_1 = deformed_triangles[:, 1] - deformed_triangles[:, 0]
    deformed_edge_2 = deformed_triangles[:, 2] - deformed_triangles[:, 0]
    map_column_1 = deformed_edge_1 / edge_1_length[:, None]
    map_column_2 = (deformed_edge_2 - map_column_1 * edge_2_projection[:, None]) / edge_2_height[
        :, None
    ]
    tangent_map = np.stack((map_column_1, map_column_2), axis=-1)
    stretches = np.linalg.svd(tangent_map, compute_uv=False)
    return stretches[:, 0], stretches[:, 1]


def _ratio_summary(log_ratios: np.ndarray) -> dict[str, float]:
    absolute_log = np.abs(log_ratios)
    return {
        "p95_absolute_log": float(np.percentile(absolute_log, 95)),
        "p95_multiplicative_factor": float(np.exp(np.percentile(absolute_log, 95))),
        "p99_absolute_log": float(np.percentile(absolute_log, 99)),
        "p99_multiplicative_factor": float(np.exp(np.percentile(absolute_log, 99))),
    }


def _method_metrics(
    source: np.ndarray,
    deformed: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    regions: dict[str, tuple[np.ndarray, np.ndarray]],
    source_normals: np.ndarray,
    source_areas: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    deformed_normals, deformed_areas = _face_normals_and_areas(deformed, faces)
    source_edge_lengths = np.linalg.norm(source[edges[:, 0]] - source[edges[:, 1]], axis=1)
    deformed_edge_lengths = np.linalg.norm(deformed[edges[:, 0]] - deformed[edges[:, 1]], axis=1)
    if np.any(deformed_edge_lengths <= 0):
        raise ValueError("deformed mesh contains a zero-length edge")
    edge_log_ratios = np.log(deformed_edge_lengths / source_edge_lengths)
    area_log_ratios = np.log(deformed_areas / source_areas)
    flips = np.einsum("ij,ij->i", source_normals, deformed_normals) <= 0
    degenerates = deformed_areas <= 1e-12
    principal_max, principal_min = _principal_stretches(source, deformed, faces)
    if np.any(principal_min <= 0):
        raise ValueError("deformed mesh has a zero principal stretch")
    principal_abs_log = np.maximum(np.abs(np.log(principal_max)), np.abs(np.log(principal_min)))

    per_region: dict[str, Any] = {}
    for region_name, (edge_mask, face_mask) in regions.items():
        region_areas = source_areas[face_mask]
        per_region[region_name] = {
            "edge_count": int(edge_mask.sum()),
            "face_count": int(face_mask.sum()),
            "edge_length_ratio": _ratio_summary(edge_log_ratios[edge_mask]),
            "face_area_ratio": _ratio_summary(area_log_ratios[face_mask]),
            "tangential_stretch": {
                "source_area_weighted_p95_max_abs_log": _weighted_quantile(
                    principal_abs_log[face_mask], region_areas, 0.95
                ),
                "source_area_weighted_p99_max_abs_log": _weighted_quantile(
                    principal_abs_log[face_mask], region_areas, 0.99
                ),
                "max_abs_log": float(principal_abs_log[face_mask].max()),
                "minimum_principal_stretch": float(principal_min[face_mask].min()),
                "maximum_principal_stretch": float(principal_max[face_mask].max()),
            },
            "face_normal_flip_count": int(flips[face_mask].sum()),
            "degenerate_face_count_area_le_1e-12": int(degenerates[face_mask].sum()),
        }
    arrays = {
        "edge_log_ratios": edge_log_ratios,
        "area_log_ratios": area_log_ratios,
        "principal_abs_log": principal_abs_log,
        "flips": flips,
        "deformed_areas": deformed_areas,
    }
    return per_region, arrays


def _paired_metrics(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    source_areas: np.ndarray,
    regions: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for region_name, (edge_mask, face_mask) in regions.items():
        raw_edge = reference["edge_log_ratios"][edge_mask]
        method_edge = candidate["edge_log_ratios"][edge_mask]
        raw_area = reference["area_log_ratios"][face_mask]
        method_area = candidate["area_log_ratios"][face_mask]
        raw_strain = reference["principal_abs_log"][face_mask]
        method_strain = candidate["principal_abs_log"][face_mask]
        region_weights = source_areas[face_mask]
        result[region_name] = {
            "edge_length_change_from_raw": _ratio_summary(method_edge - raw_edge),
            "face_area_change_from_raw": _ratio_summary(method_area - raw_area),
            "face_max_abs_log_stretch_change_from_raw": {
                "source_area_weighted_p95_absolute_change": _weighted_quantile(
                    np.abs(method_strain - raw_strain), region_weights, 0.95
                ),
                "source_area_weighted_p99_absolute_change": _weighted_quantile(
                    np.abs(method_strain - raw_strain), region_weights, 0.99
                ),
                "source_area_weighted_increased_fraction": float(
                    region_weights[method_strain > raw_strain].sum() / region_weights.sum()
                ),
            },
            "edge_source_relative_abs_distortion_increased_fraction": float(
                np.mean(np.abs(method_edge) > np.abs(raw_edge))
            ),
            "face_area_source_relative_abs_distortion_increased_fraction": float(
                region_weights[np.abs(method_area) > np.abs(raw_area)].sum() / region_weights.sum()
            ),
            "flips_added_vs_raw": int(
                (candidate["flips"][face_mask] & ~reference["flips"][face_mask]).sum()
            ),
            "flips_removed_vs_raw": int(
                (reference["flips"][face_mask] & ~candidate["flips"][face_mask]).sum()
            ),
        }
    return result


def _validate_known_stretches() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 3.0, 0.0]])
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    angle = 0.73
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rigid = source @ rotation.T + np.asarray([1.5, -0.25, 4.0])
    rigid_stretches = _principal_stretches(source, rigid, faces)
    if not np.allclose(rigid_stretches, 1.0, rtol=1e-12, atol=1e-12):
        raise AssertionError(
            f"rigid transform should preserve tangent stretches: {rigid_stretches}"
        )

    scaling = np.diag([2.0, 0.5, 1.0])
    scaled = source @ scaling.T
    scaled_stretches = _principal_stretches(source, scaled, faces)
    scaled_values = np.stack(scaled_stretches, axis=-1)
    if not np.allclose(scaled_values, [[2.0, 0.5]], rtol=1e-12, atol=1e-12):
        raise AssertionError(f"known in-plane scaling was not recovered: {scaled_stretches}")


def _analyze_condition(condition: str, run_id: str) -> dict[str, Any]:
    run_dir = ROOT / "runs" / "deformation" / run_id
    run_check = _verify_run_summary(run_dir, expected_status="succeeded")
    arrays = _load_arrays(run_dir)
    source = np.asarray(arrays["source_vertices"], dtype=np.float64)
    faces = np.asarray(arrays["faces"], dtype=np.int64)
    vertex_count = len(source)
    if source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all():
        raise ValueError(f"{run_id}: source vertices must be finite [N, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
        raise ValueError(f"{run_id}: expected a non-empty triangle mesh")
    if faces.min() < 0 or faces.max() >= vertex_count:
        raise ValueError(f"{run_id}: face index is outside the source vertex array")
    moved_indices = np.asarray(arrays["moved_vertex_indices"], dtype=np.int64)
    if (
        moved_indices.ndim != 1
        or moved_indices.size == 0
        or np.any(moved_indices < 0)
        or np.any(moved_indices >= vertex_count)
        or np.unique(moved_indices).size != moved_indices.size
    ):
        raise ValueError(f"{run_id}: invalid moved vertex indices")

    edges = _mesh_edges(faces)
    moved_vertex_mask = np.zeros(vertex_count, dtype=bool)
    moved_vertex_mask[moved_indices] = True
    edge_incident = moved_vertex_mask[edges].any(axis=1)
    face_incident = moved_vertex_mask[faces].any(axis=1)
    regions = {
        "full_mesh": (np.ones(len(edges), dtype=bool), np.ones(len(faces), dtype=bool)),
        "moved_vertex_incident_one_ring": (edge_incident, face_incident),
        "outside_moved_vertex_incident_one_ring": (~edge_incident, ~face_incident),
    }

    source_normals, source_areas = _face_normals_and_areas(source, faces)
    if np.any(source_areas <= 0):
        raise ValueError(f"{run_id}: source mesh contains a zero-area triangle")
    source_edge_lengths = np.linalg.norm(source[edges[:, 0]] - source[edges[:, 1]], axis=1)
    if np.any(source_edge_lengths <= 0):
        raise ValueError(f"{run_id}: source mesh contains a zero-length edge")

    displacements: dict[str, np.ndarray] = {}
    for method_name, key in METHOD_KEYS.items():
        displacement = np.asarray(arrays[key], dtype=np.float64)
        if displacement.ndim != 2 or displacement.shape[1] != 3 or len(displacement) < vertex_count:
            raise ValueError(f"{run_id}: invalid {method_name} displacement array")
        displacements[method_name] = displacement[:vertex_count]
    lambdas = np.asarray(arrays["soft_ridge_lambdas"], dtype=np.float64)
    soft_displacements = np.asarray(arrays["soft_query_displacements"], dtype=np.float64)
    if soft_displacements.ndim != 3 or soft_displacements.shape[0] != len(lambdas):
        raise ValueError(f"{run_id}: Soft arrays and lambda values are misaligned")
    if len(lambdas) != 3 or not np.isfinite(lambdas).all():
        raise ValueError(f"{run_id}: expected the three fixed Soft comparison values")
    for index, value in enumerate(lambdas):
        label = f"soft_lambda_{value:.0e}"
        displacements[label] = np.asarray(
            soft_displacements[index, :vertex_count], dtype=np.float64
        )

    method_metrics: dict[str, Any] = {}
    method_arrays: dict[str, dict[str, np.ndarray]] = {}
    for method_name, displacement in displacements.items():
        deformed = source + displacement
        if not np.isfinite(deformed).all():
            raise ValueError(f"{run_id}: {method_name} has non-finite vertices")
        per_region, computed = _method_metrics(
            source,
            deformed,
            faces,
            edges,
            regions,
            source_normals,
            source_areas,
        )
        method_metrics[method_name] = per_region
        method_arrays[method_name] = computed

    paired = {
        method_name: _paired_metrics(
            method_arrays["raw_prior"], method_arrays[method_name], source_areas, regions
        )
        for method_name in displacements
        if method_name != "raw_prior"
    }
    correction_norms: dict[str, Any] = {}
    for method_name, displacement in displacements.items():
        if method_name == "raw_prior":
            continue
        correction_norm = np.linalg.norm(displacement - displacements["raw_prior"], axis=1)
        correction_norms[method_name] = {
            "full_mesh": {
                "mean": float(correction_norm.mean()),
                "p95": float(np.percentile(correction_norm, 95)),
                "max": float(correction_norm.max()),
            },
            "moved_vertices": {
                "mean": float(correction_norm[moved_vertex_mask].mean()),
                "p95": float(np.percentile(correction_norm[moved_vertex_mask], 95)),
                "max": float(correction_norm[moved_vertex_mask].max()),
            },
            "outside_moved_vertices": {
                "mean": float(correction_norm[~moved_vertex_mask].mean()),
                "p95": float(np.percentile(correction_norm[~moved_vertex_mask], 95)),
                "max": float(correction_norm[~moved_vertex_mask].max()),
            },
        }

    return {
        "condition": condition,
        "run_id": run_id,
        "run_summary": run_check,
        "field_queries_sha256": _sha256(run_dir / "field_queries.npz"),
        "vertex_count": int(vertex_count),
        "face_count": int(len(faces)),
        "unique_edge_count": int(len(edges)),
        "moved_vertex_count": int(moved_vertex_mask.sum()),
        "moved_vertex_incident_edge_count": int(edge_incident.sum()),
        "moved_vertex_incident_face_count": int(face_incident.sum()),
        "selected_handle_count": int(len(arrays["selected_handle_vertex_indices"])),
        "regions": {
            "moved_vertex_incident_one_ring": (
                "Mesh edges/faces with at least one directly moved vertex."
            ),
            "outside_moved_vertex_incident_one_ring": (
                "Mesh edges/faces with no directly moved vertex."
            ),
        },
        "methods": method_metrics,
        "paired_change_from_raw_prior": paired,
        "correction_displacement_norm": correction_norms,
    }


def analyze() -> dict[str, Any]:
    _validate_known_stretches()
    run_root = ROOT / "runs" / "deformation"
    original_dir = run_root / RUNS["cat1_original"]
    original_check = _verify_run_summary(original_dir, expected_status="failed")
    original_has_mesh_queries = (original_dir / "field_queries.npz").is_file()
    if original_has_mesh_queries:
        raise ValueError("cat1 original plan run unexpectedly contains mesh query results")

    input_arrays = {
        name: _load_arrays(run_root / run_id)
        for name, run_id in RUNS.items()
        if name in {"cat0_spread", "cat0_inner50", "cat1_replay", "cat1_supplement"}
    }
    replay_check = _verify_run_summary(run_root / RUNS["cat1_replay"], expected_status="succeeded")
    cat0_shared_keys = (
        "source_vertices",
        "faces",
        "moved_vertex_indices",
        "requested_target_vertices",
        "condition_surface_source",
        "condition_surface_target_masked",
        "condition_handle_mask",
        "raw_query_displacements",
    )
    cat0_consistency = {
        key: bool(
            np.array_equal(input_arrays["cat0_spread"][key], input_arrays["cat0_inner50"][key])
        )
        for key in cat0_shared_keys
    }
    cat1_consistency = {
        key: bool(
            np.array_equal(input_arrays["cat1_replay"][key], input_arrays["cat1_supplement"][key])
        )
        for key in CONSISTENCY_KEYS
    }
    if not all(cat0_consistency.values()):
        raise ValueError(f"cat0 layouts do not share the same source/prior: {cat0_consistency}")
    if not all(cat1_consistency.values()):
        raise ValueError(f"cat1 replay and supplement differ scientifically: {cat1_consistency}")

    conditions = [
        _analyze_condition("cat0_spread", RUNS["cat0_spread"]),
        _analyze_condition("cat0_inner50", RUNS["cat0_inner50"]),
        _analyze_condition("cat1_budgeted_supplement", RUNS["cat1_supplement"]),
    ]
    return {
        "schema_version": "deformation-paired-region-metrics-v1",
        "analysis_scope": (
            "CPU-only paired analysis of saved run artifacts; no model or checkpoint loaded."
        ),
        "metric_definitions": {
            "edge_and_area_change": (
                "Per-edge length and per-face area ratios versus source; report p95/p99 "
                "exp(quantile(abs(log(ratio))))."
            ),
            "tangential_stretch": (
                "Singular values of each source-triangle to deformed-triangle "
                "piecewise-linear tangent map; summarize max(abs(log(sigma1)), "
                "abs(log(sigma2))) with source-face-area weights."
            ),
            "paired_change": (
                "Per-element log metric difference between a method and raw prior "
                "under the same run condition."
            ),
            "moved_region": (
                "Edges/faces incident to a saved direct target-moved vertex; the "
                "complement is not assumed to be unaffected."
            ),
            "acceptance_threshold": (
                "Unset: project materials provide no application-specific geometric "
                "quality cutoff or reference distribution."
            ),
        },
        "excluded_original_cat1_plan_run": {
            "run_id": RUNS["cat1_original"],
            "run_summary": original_check,
            "field_queries_present": original_has_mesh_queries,
            "reason": (
                "Failed before model outputs; not a geometric observation and does "
                "not close the original unresolved plan."
            ),
        },
        "same_condition_checks": {
            "cat0_shared_source_faces_moved_mask_and_raw_prior_equal": cat0_consistency,
            "cat1_replay_and_supplement_scientific_arrays_equal": cat1_consistency,
            "cat1_replay_run_summary": replay_check,
            "cat1_replay_is_not_an_independent_sample": True,
        },
        "conditions": conditions,
    }


def main() -> int:
    result = analyze()
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)} for {len(result['conditions'])} paired conditions.")
    print("No geometry acceptance threshold was applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
