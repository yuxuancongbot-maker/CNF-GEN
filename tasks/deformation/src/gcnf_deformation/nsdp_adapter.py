from __future__ import annotations

import torch
from torch import nn


class NSDPPointQuery:
    """Adapt an NSDP FlowArbitrary model to single-object point-position queries.

    The model returns mapped target positions, not displacements. All coordinates
    and conditioning surfaces supplied here must share the NSDP normalized frame.
    """

    def __init__(
        self,
        model: nn.Module,
        surface_samples_src: torch.Tensor,
        surface_samples_tgt: torch.Tensor,
        handle_mask: torch.Tensor,
    ) -> None:
        self.model = model
        try:
            parameter = next(model.parameters())
            self.device = parameter.device
            self.dtype = parameter.dtype
        except StopIteration:
            self.device = torch.device("cpu")
            self.dtype = torch.float32

        self.surface_samples_src = self._context("surface_samples_src", surface_samples_src)
        self.surface_samples_tgt = self._context("surface_samples_tgt", surface_samples_tgt)
        mask = torch.as_tensor(handle_mask, dtype=self.dtype, device=self.device)
        if mask.ndim == 1:
            mask = mask[:, None]
        if mask.ndim != 2 or mask.shape[1] != 1:
            raise ValueError("handle_mask must have shape [S] or [S, 1]")
        if mask.shape[0] != self.surface_samples_src.shape[1]:
            raise ValueError("handle_mask must align with the conditioning surface")
        if not torch.isfinite(mask).all():
            raise ValueError("handle_mask contains non-finite values")
        self.handle_mask = mask.unsqueeze(0)

        if self.surface_samples_tgt.shape != self.surface_samples_src.shape:
            raise ValueError("source and target conditioning surfaces must have the same shape")

    def _context(self, name: str, values: torch.Tensor) -> torch.Tensor:
        points = torch.as_tensor(values, dtype=self.dtype, device=self.device)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError(f"{name} must have shape [S, 3] and be non-empty")
        if not torch.isfinite(points).all():
            raise ValueError(f"{name} contains non-finite values")
        return points.unsqueeze(0)

    @torch.inference_mode()
    def map_points(self, points: torch.Tensor) -> torch.Tensor:
        queries = torch.as_tensor(points, dtype=self.dtype, device=self.device)
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError(f"points must have shape [N, 3], got {tuple(queries.shape)}")
        if queries.shape[0] == 0 or not torch.isfinite(queries).all():
            raise ValueError("points must be non-empty and finite")

        mapped = self.model(
            queries.unsqueeze(0),
            self.surface_samples_src,
            self.surface_samples_tgt,
            self.handle_mask,
        )
        if mapped.shape != (1, queries.shape[0], 3) or not torch.isfinite(mapped).all():
            raise ValueError("NSDP model must return finite target positions with shape [1, N, 3]")
        return mapped[0]
