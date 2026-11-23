from __future__ import annotations

from collections.abc import Callable

import torch


def displacement_from_mapping(
    map_points: Callable[[torch.Tensor], torch.Tensor], points: torch.Tensor
) -> torch.Tensor:
    """Convert a source-to-target position map T(x) into displacement T(x) - x."""
    points = torch.as_tensor(points, dtype=torch.float64)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {tuple(points.shape)}")
    if not torch.isfinite(points).all():
        raise ValueError("points contain non-finite values")
    mapped = torch.as_tensor(map_points(points), dtype=torch.float64, device=points.device)
    if mapped.shape != points.shape:
        raise ValueError(f"mapped points must match input shape {tuple(points.shape)}")
    if not torch.isfinite(mapped).all():
        raise ValueError("mapped points contain non-finite values")
    return mapped - points
