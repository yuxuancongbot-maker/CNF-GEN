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

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSET_ROOT = ROOT.parent / "_external" / "nsdp_assets"
DEFAULT_NSDP_ROOT = ROOT.parent / "_external" / "NSDP"
DEFAULT_CONFIG = ROOT / "tasks" / "deformation" / "configs" / "nsdp_tosca_cat0.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _write_obj(path: Path, vertices, faces) -> None:
    import numpy as np

    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    lines = [*(f"v {x:.12g} {y:.12g} {z:.12g}" for x, y, z in vertices)]
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _artifact(path: Path, role: str) -> dict[str, str]:
    return {"uri": path.name, "sha256": _sha256_file(path), "role": role}


def _safe_sample_component(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or not value.replace("_", "").isalnum()
    ):
        raise ValueError(f"{name} must be a non-empty ASCII path component")
    return value


def _mesh_diagnostics(vertices, faces, reference_normals=None) -> dict:
    import torch

    vertices = torch.as_tensor(vertices, dtype=torch.float64, device="cpu")
    faces = torch.as_tensor(faces, dtype=torch.long, device="cpu")
    triangles = vertices[faces]
    normals = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = torch.linalg.vector_norm(normals, dim=-1) * 0.5
    flipped_count = 0
    if reference_normals is not None:
        reference_normals = torch.as_tensor(reference_normals, dtype=torch.float64)
        flipped_count = int(((normals * reference_normals).sum(dim=-1) <= 0).sum().item())
    return {
        "degenerate_face_count": int((areas <= 1e-12).sum().item()),
        "face_normal_flip_count": flipped_count,
        "min_face_area": areas.min().item(),
        "mean_face_area": areas.mean().item(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one NSDP prior versus fixed-RBF CNF deformation application case"
    )
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--nsdp-root", type=Path, default=DEFAULT_NSDP_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "deformation")
    parser.add_argument("--layout", default=None)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--length-scale", type=float, default=None)
    parser.add_argument("--handle-count", type=int, default=None)
    parser.add_argument(
        "--progress", action="store_true", help="write stage progress to the run directory"
    )
    args = parser.parse_args()

    import numpy as np

    app_config = json.loads(args.config.read_text(encoding="utf-8"))
    sample_id = _safe_sample_component(app_config.get("sample_id", "cat0"), "sample_id")
    if not sample_id.startswith("cat") or not sample_id[3:].isdigit():
        raise ValueError(f"Unsupported TOSCA cat sample id: {sample_id!r}")
    sequence_id = _safe_sample_component(
        app_config.get("nsdp_sequence_id", "cat" if sample_id == "cat0" else sample_id),
        "nsdp_sequence_id",
    )
    motion_split = _safe_sample_component(
        app_config.get(
            "nsdp_motion_split",
            "test_unseen_identities" if sample_id == "cat0" else f"{sample_id}_replication",
        ),
        "nsdp_motion_split",
    )
    deformation_src = ROOT / "tasks" / "deformation" / "src"
    sys.path.insert(0, str(deformation_src))
    from gcnf_deformation.condition import (
        resolve_explicit_handle_constraints,
        resolve_handle_count,
    )
    from gcnf_deformation.progress import record_progress_event

    explicit_handle_config = app_config.get("cnf_handle_constraints")
    explicit_handle_indices = None
    if explicit_handle_config is not None:
        explicit_handle_indices, _ = resolve_explicit_handle_constraints(explicit_handle_config)
        if args.layout not in (None, "explicit_vertices"):
            raise ValueError(
                "cnf_handle_constraints selects its own handles; "
                "omit --layout or use explicit_vertices"
            )
        layout_id = "explicit_vertices"
        layout_config = {"selection_method": "explicit_vertex_indices"}
    else:
        layout_configs = app_config.get("handle_layouts", {})
        if layout_configs:
            layout_id = args.layout or next(iter(layout_configs))
            if layout_id not in layout_configs:
                raise ValueError(
                    f"Unknown handle layout {layout_id!r}; choose from {list(layout_configs)}"
                )
            layout_config = layout_configs[layout_id]
        else:
            layout_id = args.layout or "spread_fps"
            if layout_id != "spread_fps":
                raise ValueError("The configured application has no alternate handle layouts")
            layout_config = {"candidate_pool": "all_moved_vertices"}
    if not layout_id.replace("_", "").isalnum():
        raise ValueError(f"Invalid layout id for artifact naming: {layout_id!r}")
    condition_configs = app_config.get("conditions", {})
    if condition_configs:
        condition_id = args.condition or next(iter(condition_configs))
        if condition_id not in condition_configs:
            raise ValueError(
                f"Unknown condition {condition_id!r}; choose from {list(condition_configs)}"
            )
        condition_config = condition_configs[condition_id]
        userhandle_override = condition_config["userhandle_translation"]
        userhandle_region_override = condition_config.get("handle_region")
    else:
        condition_id = args.condition or app_config.get("condition_id", "head_drag_default")
        userhandle_override = app_config.get("nsdp_userhandle_override", {})
        userhandle_region_override = app_config.get("nsdp_userhandle_region_override")
    if not isinstance(condition_id, str) or not condition_id.replace("_", "").isalnum():
        raise ValueError(f"Invalid condition_id for artifact naming: {condition_id!r}")
    if not isinstance(userhandle_override, dict):
        raise ValueError("nsdp_userhandle_override must be a JSON object")
    length_scales = [
        float(value)
        for value in app_config.get("rbf", {}).get(
            "length_scales_normalized",
            [app_config["rbf"]["length_scale_normalized"]],
        )
    ]
    if not length_scales or any(not np.isfinite(value) or value <= 0 for value in length_scales):
        raise ValueError("RBF length scales must be a non-empty list of positive finite values")
    if len({f"{value:.12g}" for value in length_scales}) != len(length_scales):
        raise ValueError("RBF length scales must be unique")
    default_length_scale = float(app_config["rbf"]["length_scale_normalized"])
    selected_length_scale = (
        default_length_scale if args.length_scale is None else float(args.length_scale)
    )
    if selected_length_scale not in length_scales:
        raise ValueError(
            f"length scale {selected_length_scale} is not in the resolved config {length_scales}"
        )
    app_config["rbf"]["length_scale_normalized"] = selected_length_scale
    configured_handle_counts = app_config.get(
        "cnf_handle_counts", [int(app_config["cnf_handle_count"])]
    )
    if not isinstance(configured_handle_counts, list):
        raise ValueError("cnf_handle_counts must be a list")
    if explicit_handle_indices is not None:
        explicit_handle_count = len(explicit_handle_indices)
        handle_count = resolve_handle_count(
            explicit_handle_count,
            [explicit_handle_count],
            args.handle_count,
        )
        app_config["cnf_handle_counts"] = [handle_count]
    else:
        handle_count = resolve_handle_count(
            int(app_config["cnf_handle_count"]),
            configured_handle_counts,
            args.handle_count,
        )
    app_config["cnf_handle_count"] = handle_count
    soft_lambdas = [
        float(value) for value in app_config.get("soft_baseline", {}).get("ridge_lambdas", [])
    ]
    if any(not np.isfinite(value) or value <= 0 for value in soft_lambdas):
        raise ValueError("soft ridge lambdas must be positive and finite")
    asset_root = args.asset_root.resolve()
    nsdp_root = args.nsdp_root.resolve()
    nsdp_config_path = asset_root / "config" / "tosca_cat0_head.yaml"
    checkpoint_path = asset_root / "pretrained" / "arbitrary.pt"
    raw_mesh_path = asset_root / "tosca_sample" / f"{sample_id}.off"
    nsdp_dataset_dir = asset_root / "tosca_sample" / "dataset"
    nsdp_split_dir = asset_root / "splits"
    split_path = nsdp_split_dir / "tosca" / f"{motion_split}.lst"
    sample_dir = nsdp_dataset_dir / sequence_id / "0000"
    normalized_mesh_path = sample_dir / "model_normalized.obj"
    normalization_matrix_path = sample_dir / "orig_to_gaps.txt"
    overlay_root = asset_root / "runtime_overlay"
    required_paths = (
        nsdp_config_path,
        checkpoint_path,
        raw_mesh_path,
        split_path,
        normalized_mesh_path,
        normalization_matrix_path,
        overlay_root / "pointnet2_ops" / "pointnet2_utils.py",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing NSDP demo inputs: {missing}")
    split_entries = [
        line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if split_entries != [sequence_id]:
        raise ValueError(
            f"Expected {split_path} to contain only {sequence_id!r}, got {split_entries}"
        )

    length_scale_label = f"{selected_length_scale:g}".replace(".", "p")
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"-{uuid4().hex[:8]}-nsdp-tosca-{sample_id}-{condition_id}-{layout_id}"
        + f"-k{handle_count}-ell{length_scale_label}"
    )
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    progress_enabled = bool(args.progress)
    progress_path = run_dir / "progress.jsonl"
    run_started_at = time.perf_counter()
    current_stage = "run_initialized"
    run_config = {
        **app_config,
        "progress_logging": {
            "enabled": progress_enabled,
            "path": "progress.jsonl" if progress_enabled else None,
        },
        "resolved_sample": {
            "id": sample_id,
            "nsdp_sequence_id": sequence_id,
            "motion_split": motion_split,
            "split_entry": sequence_id,
        },
        "resolved_condition": {
            "id": condition_id,
            "handle_region": userhandle_region_override,
            "userhandle_translation": userhandle_override,
        },
        "resolved_length_scale_normalized": selected_length_scale,
        "resolved_handle_count": handle_count,
        "resolved_handle_layout": {"id": layout_id, **layout_config},
        "resources": {
            "nsdp_revision": subprocess.check_output(
                ["git", "-C", str(nsdp_root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "nsdp_config_sha256": _sha256_file(nsdp_config_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "raw_mesh_sha256": _sha256_file(raw_mesh_path),
            "split_sha256": _sha256_file(split_path),
            "normalized_mesh_sha256": _sha256_file(normalized_mesh_path),
            "normalization_matrix_sha256": _sha256_file(normalization_matrix_path),
        },
        "resolved_nsdp_paths": {
            "dataset_dir": str(nsdp_dataset_dir),
            "split_dir": str(nsdp_split_dir),
            "experiment_out_dir": str(asset_root / "nsdp_output"),
            "experiment_name": f"tosca_{sample_id}_head_internal",
        },
    }
    config_out = run_dir / "config.json"
    _write_json(config_out, run_config)
    git_revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    common = {
        "schema_version": "0.1",
        "run_id": run_id,
        "task": "deformation",
        "method": (
            "nsdp_prior_fixed_gaussian_rbf_hard_soft_comparison"
            if soft_lambdas
            else "nsdp_prior_fixed_gaussian_rbf_hard_projector"
        ),
        "git_revision": git_revision,
        "dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
            ).strip()
        ),
        "config_path": "config.json",
        "config_sha256": _sha256_file(config_out),
        "dataset": {
            "id": f"TOSCA/{sample_id}",
            "revision": run_config["resources"]["normalized_mesh_sha256"],
            "split": f"{motion_split}:{sequence_id}",
        },
        "seeds": {"training": None, "sampling": int(app_config["query"]["seed"])},
        "attempts": 1,
    }
    artifacts = [_artifact(config_out, "resolved_run_config")]

    def report_progress(stage: str, progress_status: str, **details) -> None:
        nonlocal current_stage
        current_stage = stage
        if not progress_enabled:
            return
        event = record_progress_event(
            progress_path,
            elapsed_seconds=time.perf_counter() - run_started_at,
            stage=stage,
            progress_status=progress_status,
            **details,
        )
        print(
            f"[progress] {event['elapsed_seconds']:.1f}s {stage} {progress_status}",
            flush=True,
        )

    def write_failure(exc: BaseException, *, interrupted: bool) -> None:
        failed_stage = current_stage
        terminal_stage = "run_interrupted" if interrupted else "run_failed"
        message = str(exc) or type(exc).__name__
        if progress_enabled:
            try:
                report_progress(
                    terminal_stage,
                    "failed",
                    failed_stage=failed_stage,
                    exception=type(exc).__name__,
                )
            except OSError as progress_error:
                message = f"{message}; progress log write failed: {progress_error}"
        failure_path = run_dir / "failure.json"
        _write_json(
            failure_path,
            {
                "exception": type(exc).__name__,
                "message": message,
                "stage": failed_stage,
                "elapsed_seconds": time.perf_counter() - run_started_at,
            },
        )
        failure_artifacts = [
            _artifact(config_out, "resolved_run_config"),
            _artifact(failure_path, "failure_diagnostic"),
        ]
        if progress_enabled and progress_path.is_file():
            failure_artifacts.append(_artifact(progress_path, "stage_progress_log"))
        summary = {
            **common,
            "status": "failed",
            "failures": [f"{type(exc).__name__}: {message}"],
            "metrics": [],
            "artifacts": failure_artifacts,
        }
        _write_json(run_dir / "run.json", summary)

    try:
        report_progress("runtime_setup", "started")
        experiment_src = ROOT / "experiments" / "learned_basis_hybrid_cnf" / "src"
        deformation_src = ROOT / "tasks" / "deformation" / "src"
        for import_root in (
            deformation_src,
            experiment_src,
            nsdp_root,
            nsdp_root / "pointnet2_ops_lib",
            overlay_root,
        ):
            sys.path.insert(0, str(import_root))

        import torch
        import trimesh
        from dataset import dataset_dict
        from gcnf_deformation import (
            NSDPPointQuery,
            correct_with_fixed_gaussian_rbf,
            displacement_from_mapping,
            fit_soft_gaussian_rbf_correction,
            moved_vertex_candidate_pool,
            resolve_userhandle_region,
            resolve_userhandle_translation,
        )
        from gcnf_deformation.correction import gaussian_rbf_matrix
        from learned_basis_hybrid_cnf.projection import hard_project_H
        from model import build_model
        from pointnet2_ops.pointnet2_utils import furthest_point_sample
        from utils.generation import define_userhandle_folder_name
        from utils.training_utils import load_config

        device = torch.device(app_config["device"])
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This NSDP application run requires the configured CUDA device")
        torch.set_num_threads(int(app_config["num_threads"]))
        report_progress("runtime_setup", "completed", device=str(device))
        nsdp_config = load_config(str(nsdp_config_path))
        nsdp_config["data"]["dataset_dir"] = str(nsdp_dataset_dir)
        nsdp_config["data"]["split_dir"] = str(nsdp_split_dir)
        nsdp_config["test"]["motion_split"] = motion_split
        nsdp_config["experiment"]["out_dir"] = str(asset_root / "nsdp_output")
        nsdp_config["experiment"]["name"] = f"tosca_{sample_id}_head_internal"
        requested_userhandle_translation = resolve_userhandle_translation(
            nsdp_config["data"]["userhandle"], userhandle_override
        )
        requested_userhandle_region = resolve_userhandle_region(
            nsdp_config["data"]["userhandle"], userhandle_region_override
        )
        nsdp_config["data"]["userhandle"].update(requested_userhandle_translation)
        nsdp_config["data"]["userhandle"].update(requested_userhandle_region)
        dataset_type = nsdp_config["data"]["type"]
        dataset = dataset_dict[dataset_type](
            nsdp_config,
            iden_split=nsdp_config["test"]["iden_split"],
            motion_split=nsdp_config["test"]["motion_split"],
            load_mesh=True,
            num_sampled_pairs=nsdp_config["test"]["num_sampled_pairs"],
        )
        if len(dataset) != 1:
            raise RuntimeError(f"Expected one prepared TOSCA demo pair, found {len(dataset)}")
        sample = dataset[0]
        sample_metadata = dataset.get_metadata(0)
        report_progress(
            "dataset_loaded",
            "completed",
            pair=str(sample_metadata["pair_info"][1:]),
            vertices=int(len(sample["verts_src"])),
            faces=int(len(sample["faces"])),
        )

        model_start = time.perf_counter()
        report_progress("model_load", "started")
        model, _, _, _ = build_model(nsdp_config, weight_file=str(checkpoint_path), device=device)
        model.eval()
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - model_start
        report_progress("model_load", "completed", seconds=model_load_seconds)

        surface_inputs = np.asarray(sample["surface_samples_inputs"], dtype=np.float32)
        source_vertices = np.asarray(sample["verts_src"], dtype=np.float64)
        requested_target_vertices = np.asarray(sample["verts_tgt"], dtype=np.float64)
        faces = np.asarray(sample["faces"], dtype=np.int64)
        if source_vertices.shape != requested_target_vertices.shape:
            raise ValueError(
                "NSDP source and requested target meshes have different vertex layouts"
            )
        surface_source = torch.as_tensor(surface_inputs[:, :3], dtype=torch.float32)
        surface_target_masked = torch.as_tensor(surface_inputs[:, 3:6], dtype=torch.float32)
        surface_handle_mask = torch.as_tensor(surface_inputs[:, 6], dtype=torch.float32)
        adapter = NSDPPointQuery(
            model,
            surface_source,
            surface_target_masked,
            surface_handle_mask,
        )

        requested_displacements = requested_target_vertices - source_vertices
        moved_indices = np.flatnonzero(np.linalg.norm(requested_displacements, axis=1) > 1e-8)
        requested_translation = np.asarray(
            [requested_userhandle_translation[key] for key in ("xtrans", "ytrans", "ztrans")],
            dtype=np.float64,
        )
        observed_translation = requested_displacements[moved_indices]
        translation_fit_error = np.abs(observed_translation - requested_translation).max()
        if translation_fit_error > 2e-7:
            raise RuntimeError(
                "NSDP requested target does not match the configured user-handle translation: "
                f"max_abs_error={translation_fit_error:.3e}"
            )
        if explicit_handle_config is None:
            if moved_indices.size < handle_count:
                raise ValueError(
                    f"Requested deformation moved {moved_indices.size} vertices; "
                    f"need {handle_count} handles"
                )
            candidate_vertex_indices, layout_diagnostics = moved_vertex_candidate_pool(
                source_vertices,
                moved_indices,
                candidate_pool=layout_config["candidate_pool"],
                radius_quantile=layout_config.get("radius_quantile"),
            )
            if candidate_vertex_indices.size < handle_count:
                raise ValueError(
                    f"Handle layout {layout_id!r} produced "
                    f"{candidate_vertex_indices.size} candidates; "
                    f"need {handle_count}"
                )
            candidate_handles = (
                torch.as_tensor(
                    source_vertices[candidate_vertex_indices],
                    dtype=torch.float32,
                    device=device,
                )
                .unsqueeze(0)
                .contiguous()
            )
            local_handle_indices = furthest_point_sample(candidate_handles, handle_count)[0]
            selected_vertex_indices = candidate_vertex_indices[local_handle_indices.cpu().numpy()]
            target_displacement_values = requested_displacements[selected_vertex_indices]
            layout_diagnostics["selection_method"] = "fps_over_moved_prior_vertices"
        else:
            selected_vertex_indices, explicit_targets = resolve_explicit_handle_constraints(
                explicit_handle_config, vertex_count=len(source_vertices)
            )
            selected_vertex_indices = np.asarray(selected_vertex_indices, dtype=np.int64)
            target_displacement_values = np.asarray(explicit_targets, dtype=np.float64)
            if len(selected_vertex_indices) != handle_count:
                raise RuntimeError("resolved explicit handle count changed after mesh loading")
            layout_diagnostics = {
                "selection_method": "explicit_vertex_indices",
                "candidate_count": int(len(selected_vertex_indices)),
            }
        selected_vertex_tensor = torch.as_tensor(
            selected_vertex_indices, dtype=torch.long, device=device
        )

        source_tensor = torch.as_tensor(source_vertices, dtype=torch.float64, device=device)
        handle_points = source_tensor[selected_vertex_tensor]
        target_displacements = torch.as_tensor(
            target_displacement_values, dtype=torch.float64, device=device
        )
        report_progress(
            "handle_selection",
            "completed",
            moved_vertices=int(moved_indices.size),
            handle_count=int(len(selected_vertex_indices)),
        )

        face_triangles = source_vertices[faces]
        face_normals = np.cross(
            face_triangles[:, 1] - face_triangles[:, 0],
            face_triangles[:, 2] - face_triangles[:, 0],
        )
        normal_lengths = np.linalg.norm(face_normals, axis=1)
        valid_face_indices = np.flatnonzero(normal_lengths > 1e-12)
        probe_count = int(app_config["query"]["surface_normal_offset_probe_count"])
        rng = np.random.default_rng(int(app_config["query"]["seed"]))
        probe_face_indices = rng.choice(valid_face_indices, size=probe_count, replace=False)
        probe_centers = face_triangles[probe_face_indices].mean(axis=1)
        probe_normals = face_normals[probe_face_indices] / normal_lengths[probe_face_indices, None]
        bbox_diagonal = float(np.linalg.norm(np.ptp(source_vertices, axis=0)))
        probe_offset = bbox_diagonal * float(
            app_config["query"]["offset_fraction_of_bbox_diagonal"]
        )
        probe_points = probe_centers + probe_offset * probe_normals

        query_points_np = np.concatenate([source_vertices, probe_points], axis=0)
        query_points = torch.as_tensor(query_points_np, dtype=torch.float64, device=device)
        torch.cuda.synchronize()
        prior_start = time.perf_counter()
        report_progress("prior_query", "started", query_count=int(len(query_points_np)))
        prior_mapped_positions = adapter.map_points(query_points)
        torch.cuda.synchronize()
        prior_query_seconds = time.perf_counter() - prior_start
        report_progress("prior_query", "completed", seconds=prior_query_seconds)
        raw_query_displacements = prior_mapped_positions.to(torch.float64) - query_points

        def prior_field(points: torch.Tensor) -> torch.Tensor:
            points = torch.as_tensor(points, dtype=torch.float64, device=device)
            if points.shape == query_points.shape and torch.equal(points, query_points):
                return raw_query_displacements
            if points.shape == handle_points.shape and torch.equal(points, handle_points):
                return raw_query_displacements[selected_vertex_tensor]
            return displacement_from_mapping(adapter.map_points, points)

        length_scale = selected_length_scale
        solve_start = time.perf_counter()
        report_progress("hard_correction", "started")
        corrected_field, correction_diagnostics = correct_with_fixed_gaussian_rbf(
            prior_field,
            handle_points,
            target_displacements,
            length_scale=length_scale,
            projector=hard_project_H,
            max_handle_error=float(app_config["rbf"]["max_handle_error"]),
        )
        corrected_query_displacements = corrected_field(query_points)
        torch.cuda.synchronize()
        correction_seconds = time.perf_counter() - solve_start
        report_progress(
            "hard_correction",
            "completed",
            seconds=correction_seconds,
            max_handle_error=correction_diagnostics["corrected_handle_error_max"],
        )

        soft_results = []
        for ridge_lambda in soft_lambdas:
            soft_start = time.perf_counter()
            report_progress("soft_baseline", "started", ridge_lambda=ridge_lambda)
            soft_field, soft_diagnostics = fit_soft_gaussian_rbf_correction(
                prior_field,
                handle_points,
                target_displacements,
                length_scale=length_scale,
                ridge_lambda=ridge_lambda,
            )
            soft_query_displacements = soft_field(query_points)
            torch.cuda.synchronize()
            soft_seconds = time.perf_counter() - soft_start
            report_progress(
                "soft_baseline",
                "completed",
                ridge_lambda=ridge_lambda,
                seconds=soft_seconds,
                max_handle_error=soft_diagnostics["handle_error_max"],
            )
            soft_handle_errors = torch.linalg.vector_norm(
                soft_query_displacements[selected_vertex_tensor] - target_displacements,
                dim=-1,
            )
            soft_query_correction_norms = torch.linalg.vector_norm(
                soft_query_displacements - raw_query_displacements, dim=-1
            )
            soft_results.append(
                {
                    "ridge_lambda": ridge_lambda,
                    "lambda_label": f"{ridge_lambda:.0e}",
                    "field": soft_field,
                    "diagnostics": soft_diagnostics,
                    "query_displacements": soft_query_displacements,
                    "handle_errors": soft_handle_errors,
                    "query_correction_norms": soft_query_correction_norms,
                    "solve_and_query_seconds": soft_seconds,
                }
            )

        raw_handle_displacements = raw_query_displacements[selected_vertex_tensor]
        corrected_handle_displacements = corrected_query_displacements[selected_vertex_tensor]
        raw_handle_errors = torch.linalg.vector_norm(
            raw_handle_displacements - target_displacements, dim=-1
        )
        corrected_handle_errors = torch.linalg.vector_norm(
            corrected_handle_displacements - target_displacements, dim=-1
        )
        query_correction_norms = torch.linalg.vector_norm(
            corrected_query_displacements - raw_query_displacements, dim=-1
        )
        face_tensor = torch.as_tensor(faces, dtype=torch.long, device=device)
        source_triangles = source_tensor[face_tensor]
        source_normals = torch.linalg.cross(
            source_triangles[:, 1] - source_triangles[:, 0],
            source_triangles[:, 2] - source_triangles[:, 0],
        )
        raw_vertices = source_tensor + raw_query_displacements[: len(source_vertices)]
        corrected_vertices = source_tensor + corrected_query_displacements[: len(source_vertices)]
        source_mesh_diag = _mesh_diagnostics(source_tensor, faces)
        raw_mesh_diag = _mesh_diagnostics(raw_vertices, faces, source_normals.cpu())
        corrected_mesh_diag = _mesh_diagnostics(corrected_vertices, faces, source_normals.cpu())
        for result in soft_results:
            soft_vertices = source_tensor + result["query_displacements"][: len(source_vertices)]
            result["vertices"] = soft_vertices
            result["mesh_diagnostics"] = _mesh_diagnostics(
                soft_vertices, faces, source_normals.cpu()
            )
        report_progress("mesh_diagnostics", "completed")
        constraint_matrix = gaussian_rbf_matrix(handle_points, handle_points, length_scale)

        query_kind = np.array(
            ["mesh_vertex"] * len(source_vertices) + ["surface_normal_offset"] * probe_count,
            dtype="U24",
        )
        soft_query_array = (
            torch.stack([result["query_displacements"] for result in soft_results]).cpu().numpy()
            if soft_results
            else np.empty((0, *query_points.shape), dtype=np.float64)
        )
        soft_coefficient_array = (
            np.stack([result["field"].coefficients.cpu().numpy() for result in soft_results])
            if soft_results
            else np.empty((0, 3, handle_count), dtype=np.float64)
        )
        query_path = run_dir / "field_queries.npz"
        np.savez_compressed(
            query_path,
            source_vertices=source_vertices,
            faces=faces,
            requested_target_vertices=requested_target_vertices,
            moved_vertex_indices=moved_indices,
            selected_handle_vertex_indices=selected_vertex_indices,
            selected_handle_points=handle_points.cpu().numpy(),
            target_displacements=target_displacements.cpu().numpy(),
            condition_surface_source=surface_inputs[:, :3],
            condition_surface_target_masked=surface_inputs[:, 3:6],
            condition_handle_mask=surface_inputs[:, 6],
            query_points=query_points_np,
            query_kind=query_kind,
            prior_mapped_positions=prior_mapped_positions.cpu().numpy(),
            raw_query_displacements=raw_query_displacements.cpu().numpy(),
            corrected_query_displacements=corrected_query_displacements.cpu().numpy(),
            rbf_constraint_matrix=constraint_matrix.cpu().numpy(),
            rbf_coefficients=corrected_field.coefficients.cpu().numpy(),
            soft_ridge_lambdas=np.asarray(soft_lambdas, dtype=np.float64),
            soft_rbf_coefficients=soft_coefficient_array,
            soft_query_displacements=soft_query_array,
        )
        report_progress("field_artifacts", "completed", path=query_path.name)

        source_path = run_dir / "source.obj"
        target_path = run_dir / "requested_target.obj"
        raw_path = run_dir / "raw_prior.obj"
        corrected_path = run_dir / "cnf_corrected.obj"
        _write_obj(source_path, source_vertices, faces)
        _write_obj(target_path, requested_target_vertices, faces)
        _write_obj(raw_path, raw_vertices.cpu().numpy(), faces)
        _write_obj(corrected_path, corrected_vertices.cpu().numpy(), faces)
        for result in soft_results:
            soft_mesh_path = run_dir / f"soft_lambda_{result['lambda_label']}.obj"
            _write_obj(soft_mesh_path, result["vertices"].cpu().numpy(), faces)
            result["mesh_path"] = soft_mesh_path
        report_progress("mesh_artifacts", "completed")

        pair_info = sample_metadata["pair_info"]
        _, _, _, _, source_name, source_frame, target_name, target_frame = pair_info
        official_mesh_dir = (
            Path(nsdp_config["experiment"]["out_dir"])
            / nsdp_config["experiment"]["name"]
            / define_userhandle_folder_name(nsdp_config)
            / nsdp_config["test"]["mesh_folder"]
        )
        official_deformed_path = (
            official_mesh_dir
            / "deformed"
            / (
                f"{source_name}_{source_frame}_to_{target_name}_{target_frame}."
                f"{nsdp_config['test']['mesh_format']}"
            )
        )
        official_target_path = (
            official_mesh_dir
            / "target"
            / (
                f"{source_name}_{source_frame}_to_{target_name}_{target_frame}."
                f"{nsdp_config['test']['mesh_format']}"
            )
        )
        crosscheck = {
            "status": "unavailable",
            "official_deformed_mesh_path": str(official_deformed_path),
            "official_target_mesh_path": str(official_target_path),
        }
        if official_deformed_path.is_file() and official_target_path.is_file():
            official_deformed = trimesh.load_mesh(official_deformed_path, process=False)
            official_target = trimesh.load_mesh(official_target_path, process=False)
            raw_vertices_np = raw_vertices.cpu().numpy()
            faces_equal = np.array_equal(np.asarray(official_deformed.faces), faces)
            target_faces_equal = np.array_equal(np.asarray(official_target.faces), faces)
            if (
                official_deformed.vertices.shape == raw_vertices_np.shape
                and official_target.vertices.shape == requested_target_vertices.shape
                and faces_equal
                and target_faces_equal
            ):
                prior_delta = np.linalg.norm(
                    np.asarray(official_deformed.vertices, dtype=np.float64) - raw_vertices_np,
                    axis=1,
                )
                target_delta = np.linalg.norm(
                    np.asarray(official_target.vertices, dtype=np.float64)
                    - requested_target_vertices,
                    axis=1,
                )
                crosscheck = {
                    "status": "matched",
                    "official_deformed_mesh_path": str(official_deformed_path),
                    "official_deformed_mesh_sha256": _sha256_file(official_deformed_path),
                    "official_target_mesh_path": str(official_target_path),
                    "official_target_mesh_sha256": _sha256_file(official_target_path),
                    "faces_equal": True,
                    "prior_vs_official_vertex_error_max": float(prior_delta.max()),
                    "prior_vs_official_vertex_error_mean": float(prior_delta.mean()),
                    "target_vs_official_vertex_error_max": float(target_delta.max()),
                    "target_vs_official_vertex_error_mean": float(target_delta.mean()),
                }
            else:
                crosscheck["status"] = "vertex_or_face_layout_mismatch"
        crosscheck_path = run_dir / "official_demo_crosscheck.json"
        _write_json(crosscheck_path, crosscheck)
        report_progress("official_crosscheck", "completed", crosscheck_status=crosscheck["status"])

        runtime = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "device": torch.cuda.get_device_name(0),
            "nsdp_model_dtype": str(next(model.parameters()).dtype),
            "cnf_and_solver_dtype": str(handle_points.dtype),
            "model_load_seconds": model_load_seconds,
            "prior_query_seconds": prior_query_seconds,
            "correction_and_corrected_query_seconds": correction_seconds,
            "soft_solve_and_query_seconds": {
                result["lambda_label"]: result["solve_and_query_seconds"] for result in soft_results
            },
        }
        soft_diagnostic_records = [
            {
                "ridge_lambda": result["ridge_lambda"],
                "lambda_label": result["lambda_label"],
                "solver": result["diagnostics"],
                "handle_errors": result["handle_errors"].cpu().tolist(),
                "query_correction_norm_mean": result["query_correction_norms"].mean().item(),
                "query_correction_norm_max": result["query_correction_norms"].max().item(),
                "mesh_diagnostics": result["mesh_diagnostics"],
                "solve_and_query_seconds": result["solve_and_query_seconds"],
            }
            for result in soft_results
        ]
        diagnostics = {
            "evidence_level": "real_nsdp_prior_single_sample_application",
            "cnf_handle_constraint_mode": (
                "explicit_vertex_indices"
                if explicit_handle_config is not None
                else "moved_region_fps"
            ),
            "condition": {
                "id": condition_id,
                "handle_region": next(
                    key for key, active in requested_userhandle_region.items() if active
                ),
                "userhandle_translation": requested_userhandle_translation,
                "moved_vertex_displacement_mean": observed_translation.mean(axis=0).tolist(),
                "max_abs_target_translation_error": float(translation_fit_error),
                "moved_vertex_count": int(moved_indices.size),
            },
            "resolved_length_scale_normalized": selected_length_scale,
            "resolved_handle_count": handle_count,
            "handle_layout": {"id": layout_id, **layout_diagnostics},
            "nsdp_config_path": str(nsdp_config_path),
            "checkpoint_path": str(checkpoint_path),
            "condition_surface_count": int(surface_inputs.shape[0]),
            "condition_handle_mask_count": int((surface_inputs[:, 6] > 0.5).sum()),
            "moved_vertex_candidate_count": int(moved_indices.size),
            "selected_handle_vertex_indices": selected_vertex_indices.tolist(),
            "selected_handle_displacements": target_displacements.cpu().tolist(),
            "bbox_diagonal_normalized": bbox_diagonal,
            "surface_normal_offset": probe_offset,
            "constraint_diagnostics": correction_diagnostics,
            "raw_handle_errors": raw_handle_errors.cpu().tolist(),
            "corrected_handle_errors": corrected_handle_errors.cpu().tolist(),
            "query_correction_norm_mean": query_correction_norms.mean().item(),
            "query_correction_norm_max": query_correction_norms.max().item(),
            "mesh_diagnostics": {
                "source": source_mesh_diag,
                "raw_prior": raw_mesh_diag,
                "cnf_corrected": corrected_mesh_diag,
                "soft_baselines": {
                    result["lambda_label"]: result["mesh_diagnostics"] for result in soft_results
                },
            },
            "soft_baselines": soft_diagnostic_records,
            "official_demo_crosscheck": crosscheck,
            "runtime": runtime,
        }
        diagnostics_path = run_dir / "diagnostics.json"
        _write_json(diagnostics_path, diagnostics)
        comparison_results_path = run_dir / "comparison_results.json"
        _write_json(
            comparison_results_path,
            {
                "layout": diagnostics["handle_layout"],
                "selected_handle_vertex_indices": selected_vertex_indices.tolist(),
                "methods": {
                    "raw_prior": {
                        "handle_error_max": raw_handle_errors.max().item(),
                        "handle_error_mean": raw_handle_errors.mean().item(),
                        "mesh_diagnostics": raw_mesh_diag,
                    },
                    "hard_cnf": {
                        "handle_error_max": corrected_handle_errors.max().item(),
                        "handle_error_mean": corrected_handle_errors.mean().item(),
                        "query_correction_norm_mean": query_correction_norms.mean().item(),
                        "mesh_diagnostics": corrected_mesh_diag,
                    },
                    "soft_ridge": soft_diagnostic_records,
                },
            },
        )
        metrics = []
        for name, values, stage in (
            ("handle_position_error_max", raw_handle_errors, "raw"),
            ("handle_position_error_mean", raw_handle_errors, "raw"),
            ("handle_position_error_max", corrected_handle_errors, "constrained"),
            ("handle_position_error_mean", corrected_handle_errors, "constrained"),
        ):
            statistic = "max" if name.endswith("max") else "mean"
            value = values.max().item() if statistic == "max" else values.mean().item()
            metric = {
                "name": name,
                "value": value,
                "unit": "normalized_position_units",
                "stage": stage,
                "scope": "finite_constraints",
                "count": int(values.numel()),
                "statistic": statistic,
            }
            if stage == "constrained":
                metric["tolerance"] = float(app_config["rbf"]["max_handle_error"])
            metrics.append(metric)
        for result in soft_results:
            label = result["lambda_label"]
            errors = result["handle_errors"]
            correction_norms = result["query_correction_norms"]
            for statistic, value in (
                ("max", errors.max().item()),
                ("mean", errors.mean().item()),
            ):
                metrics.append(
                    {
                        "name": f"handle_position_error_{statistic}_soft_lambda_{label}",
                        "value": value,
                        "unit": "normalized_position_units",
                        "stage": "constrained",
                        "scope": "finite_constraints",
                        "count": int(errors.numel()),
                        "statistic": statistic,
                    }
                )
            metrics.extend(
                [
                    {
                        "name": f"query_correction_norm_mean_soft_lambda_{label}",
                        "value": correction_norms.mean().item(),
                        "unit": "normalized_position_units",
                        "stage": "constrained",
                        "scope": "intermediate_field",
                        "count": int(correction_norms.numel()),
                        "statistic": "mean",
                    },
                    {
                        "name": f"soft_solve_and_query_seconds_lambda_{label}",
                        "value": result["solve_and_query_seconds"],
                        "unit": "seconds",
                        "stage": "constrained",
                        "scope": "intermediate_field",
                        "count": 1,
                        "statistic": "single",
                    },
                ]
            )
        metrics.extend(
            [
                {
                    "name": "query_correction_norm_mean",
                    "value": query_correction_norms.mean().item(),
                    "unit": "normalized_position_units",
                    "stage": "constrained",
                    "scope": "intermediate_field",
                    "count": int(query_correction_norms.numel()),
                    "statistic": "mean",
                },
                {
                    "name": "prior_query_seconds",
                    "value": prior_query_seconds,
                    "unit": "seconds",
                    "stage": "raw",
                    "scope": "intermediate_field",
                    "count": 1,
                    "statistic": "single",
                },
                {
                    "name": "correction_and_corrected_query_seconds",
                    "value": correction_seconds,
                    "unit": "seconds",
                    "stage": "constrained",
                    "scope": "intermediate_field",
                    "count": 1,
                    "statistic": "single",
                },
            ]
        )
        report_progress("run_summary", "completed", status="succeeded")
        artifacts = [
            _artifact(config_out, "resolved_run_config"),
            _artifact(query_path, "field_queries_constraint_matrix_and_coefficients"),
            _artifact(diagnostics_path, "solver_and_mesh_diagnostics"),
            _artifact(comparison_results_path, "matched_hard_soft_comparison"),
            _artifact(crosscheck_path, "official_nsdp_demo_mesh_crosscheck"),
            _artifact(source_path, "source_mesh"),
            _artifact(target_path, "requested_handle_target_mesh"),
            _artifact(raw_path, "nsdp_prior_mesh"),
            _artifact(corrected_path, "cnf_corrected_mesh"),
            *[
                _artifact(result["mesh_path"], f"soft_ridge_mesh_lambda_{result['lambda_label']}")
                for result in soft_results
            ],
        ]
        if progress_enabled:
            artifacts.append(_artifact(progress_path, "stage_progress_log"))
        summary = {
            **common,
            "status": "succeeded",
            "failures": [],
            "metrics": metrics,
            "artifacts": artifacts,
        }
        _write_json(run_dir / "run.json", summary)
        print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))
        return 0
    except KeyboardInterrupt as exc:
        write_failure(exc, interrupted=True)
        return 130
    except Exception as exc:
        write_failure(exc, interrupted=False)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
