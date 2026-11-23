from __future__ import annotations

from typing import Any

import numpy as np


def moved_vertex_candidate_pool(
    source_vertices: np.ndarray,
    moved_vertex_indices: np.ndarray,
    *,
    candidate_pool: str,
    radius_quantile: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(source_vertices, dtype=np.float64)
    moved_indices = np.asarray(moved_vertex_indices, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("source_vertices must be a finite [N, 3] array")
    if moved_indices.ndim != 1 or moved_indices.size == 0:
        raise ValueError("moved_vertex_indices must be a non-empty vector")
    if np.any(moved_indices < 0) or np.any(moved_indices >= len(vertices)):
        raise ValueError("moved_vertex_indices contain out-of-range indices")
    if np.unique(moved_indices).size != moved_indices.size:
        raise ValueError("moved_vertex_indices must be unique")

    if candidate_pool == "all_moved_vertices":
        return moved_indices.copy(), {
            "candidate_pool": candidate_pool,
            "candidate_count": int(moved_indices.size),
        }
    if candidate_pool != "nearest_radius_quantile":
        raise ValueError(f"unsupported candidate_pool: {candidate_pool}")
    if radius_quantile is None or not np.isfinite(radius_quantile) or not 0 < radius_quantile <= 1:
        raise ValueError("radius_quantile must be finite and in (0, 1]")

    points = vertices[moved_indices]
    center = points.mean(axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    cutoff = float(np.quantile(radii, radius_quantile))
    pool = moved_indices[radii <= cutoff]
    if pool.size == 0:
        raise ValueError("radius quantile produced an empty candidate pool")
    return pool, {
        "candidate_pool": candidate_pool,
        "radius_quantile": float(radius_quantile),
        "radius_cutoff": cutoff,
        "candidate_center": center.tolist(),
        "candidate_count": int(pool.size),
    }
