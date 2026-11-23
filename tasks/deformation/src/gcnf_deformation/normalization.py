from __future__ import annotations

from pathlib import Path

import numpy as np


def gaps_pca_scale_matrix(
    vertices: np.ndarray, faces: np.ndarray, target_scale: float = 0.35
) -> np.ndarray:
    """Reproduce GAPS msh2msh scale_by_pca + translate_by_centroid for triangle meshes.

    This follows GAPS commit f78eb23, apps/msh2msh/msh2msh.cpp and
    pkgs/R3Shapes/R3Mesh.cpp: area-weighted triangle-centroid PCA scale, then
    requested scale and centroid translation. No PCA rotation is applied by
    these two flags.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError("vertices must have shape [N, 3] and be non-empty")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError("faces must be non-empty triangles with shape [M, 3]")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contain non-finite values")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("face index is outside the vertex array")
    if not np.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("target_scale must be positive and finite")

    triangles = vertices[faces]
    triangle_areas = (
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        * 0.5
    )
    total_area = triangle_areas.sum()
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("mesh must have positive finite surface area")

    face_centroids = triangles.mean(axis=1)
    centroid = np.average(face_centroids, axis=0, weights=triangle_areas)
    centered = face_centroids - centroid
    covariance = (centered.T * triangle_areas) @ centered / total_area
    singular_values = np.linalg.svd(covariance, compute_uv=False)
    largest_variance = float(singular_values[0])
    if not np.isfinite(largest_variance) or largest_variance <= 0:
        raise ValueError("mesh has no positive principal variance")

    scale = target_scale / np.sqrt(largest_variance)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] *= scale
    matrix[:3, 3] = -scale * centroid
    return matrix


def read_off(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    tokens = []
    for line in Path(path).read_text(encoding="ascii").splitlines():
        tokens.extend(line.split("#", 1)[0].split())
    if not tokens or tokens[0] != "OFF":
        raise ValueError("expected an OFF mesh")
    cursor = 1
    try:
        vertex_count, face_count, _ = map(int, tokens[cursor : cursor + 3])
        cursor += 3
        vertices = np.asarray(
            [
                list(map(float, tokens[cursor + 3 * i : cursor + 3 * i + 3]))
                for i in range(vertex_count)
            ],
            dtype=np.float64,
        )
        cursor += 3 * vertex_count
        faces = []
        for _ in range(face_count):
            vertex_per_face = int(tokens[cursor])
            cursor += 1
            if vertex_per_face != 3:
                raise ValueError("only triangular OFF faces are supported")
            faces.append(list(map(int, tokens[cursor : cursor + 3])))
            cursor += 3
    except (IndexError, ValueError) as exc:
        raise ValueError(f"malformed triangular OFF mesh: {exc}") from exc
    return vertices, np.asarray(faces, dtype=np.int64)


def prepare_off_mesh(
    input_path: str | Path, output_dir: str | Path, target_scale: float = 0.35
) -> tuple[Path, Path]:
    """Write GAPS-compatible normalized OBJ and original-to-normalized matrix."""
    vertices, faces = read_off(input_path)
    matrix = gaps_pca_scale_matrix(vertices, faces, target_scale)
    normalized = vertices @ matrix[:3, :3].T + matrix[:3, 3]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    mesh_path = output_dir / "model_normalized.obj"
    matrix_path = output_dir / "orig_to_gaps.txt"
    lines = [*(f"v {x:.17g} {y:.17g} {z:.17g}" for x, y, z in normalized)]
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    mesh_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    np.savetxt(matrix_path, matrix, fmt="%.17g")
    return mesh_path, matrix_path
