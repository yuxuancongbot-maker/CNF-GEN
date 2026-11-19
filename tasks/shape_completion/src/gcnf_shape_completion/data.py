"""Partial-observation chair batches from the manifest/NPZ data contract.

Manifest format (``coordination/contract.md``): a JSON list of
``{"object_id": ..., "path": <relative to the manifest>}``.  Each NPZ holds

===================  ====================  =========================
key                  shape                 dtype
===================  ====================  =========================
``partial_sdf``      ``(R, R, R)``         ``float32``
``known_mask``       ``(R, R, R)``         ``bool``
``target_sdf``       ``(R, R, R)``         ``float32``
``anchors``          ``(M, 3)``            ``float32``
``normals``          ``(M, 3)``            ``float32``
===================  ====================  =========================

with ``R = 64`` by default, ``(x, y, z)`` storage order, inclusive endpoints,
TSDF positive outside with truncation ``+/-0.2`` and the ``+0.2`` filler
outside ``known_mask``.  ``target_sdf`` is loaded *only* as training
supervision / evaluation reference; the condition path never reads it (see
``sdfcnf.conditioning``).

Optional extra NPZ fields (e.g. camera/visibility geometry) are preserved per
sample under ``Batch.extras`` and are never used by the condition path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .conventions import (
    FILLER,
    GRID_HI,
    GRID_LO,
    TRUNCATION,
)

REQUIRED_KEYS = ("partial_sdf", "known_mask", "target_sdf", "anchors", "normals")


class DataContractError(ValueError):
    """Raised when a sample violates the data contract in a fatal way."""

    def __init__(self, source: str, errors: Sequence[str]) -> None:
        super().__init__(f"{source}: " + "; ".join(errors))
        self.source = source
        self.errors = list(errors)


@dataclass
class Batch:
    """A collated batch; every tensor is already in ``(x, y, z)`` layout."""

    object_ids: List[str]
    partial_sdf: torch.Tensor  # (B, 1, R, R, R) float32
    known_mask: torch.Tensor  # (B, 1, R, R, R) bool
    target_sdf: torch.Tensor  # (B, 1, R, R, R) float32
    anchors: torch.Tensor  # (B, M, 3) float32
    normals: torch.Tensor  # (B, M, 3) float32
    extras: List[Dict[str, object]] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return self.partial_sdf.shape[0]

    @property
    def resolution(self) -> int:
        return self.partial_sdf.shape[-1]

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "Batch":
        return Batch(
            object_ids=self.object_ids,
            partial_sdf=self.partial_sdf.to(device, non_blocking=non_blocking),
            known_mask=self.known_mask.to(device, non_blocking=non_blocking),
            target_sdf=self.target_sdf.to(device, non_blocking=non_blocking),
            anchors=self.anchors.to(device, non_blocking=non_blocking),
            normals=self.normals.to(device, non_blocking=non_blocking),
            extras=self.extras,
        )

    def stats(self) -> Dict[str, float]:
        known = self.known_mask
        return {
            "batch_size": float(self.batch_size),
            "resolution": float(self.resolution),
            "anchors_per_sample": float(self.anchors.shape[1]),
            "known_fraction_mean": float(known.float().mean()),
            "known_fraction_min": float(known.float().mean(dim=(1, 2, 3, 4)).min()),
            "known_fraction_max": float(known.float().mean(dim=(1, 2, 3, 4)).max()),
            "partial_absmax": float(self.partial_sdf.abs().max()),
            "target_absmax": float(self.target_sdf.abs().max()),
            "normal_norm_max_dev": float((self.normals.norm(dim=-1) - 1.0).abs().max()),
        }


def validate_sample(
    arrays: Dict[str, "np.ndarray"],  # noqa: F821 - numpy only for typing
    source: str,
    resolution: int = 64,
    truncation: float = TRUNCATION,
    filler: float = FILLER,
    lo: float = GRID_LO,
    hi: float = GRID_HI,
    normal_tol: float = 1e-3,
) -> Tuple[List[str], List[str]]:
    """Return ``(fatal_errors, contract_violations)`` for one NPZ sample."""
    errors: List[str] = []
    contract: List[str] = []

    for key in REQUIRED_KEYS:
        if key not in arrays:
            errors.append(f"missing key '{key}'")
    if errors:
        return errors, contract

    partial = arrays["partial_sdf"]
    known = arrays["known_mask"]
    target = arrays["target_sdf"]
    anchors = arrays["anchors"]
    normals = arrays["normals"]

    if tuple(partial.shape) != (resolution, resolution, resolution):
        errors.append(f"partial_sdf shape {tuple(partial.shape)} != {(resolution,) * 3}")
    if tuple(target.shape) != (resolution, resolution, resolution):
        errors.append(f"target_sdf shape {tuple(target.shape)} != {(resolution,) * 3}")
    if tuple(known.shape) != (resolution, resolution, resolution):
        errors.append(f"known_mask shape {tuple(known.shape)} != {(resolution,) * 3}")
    if anchors.ndim != 2 or anchors.shape[-1] != 3:
        errors.append(f"anchors shape {tuple(anchors.shape)} != (M, 3)")
    if normals.shape != anchors.shape:
        errors.append(
            f"normals shape {tuple(normals.shape)} != anchors shape {tuple(anchors.shape)}"
        )
    if errors:
        return errors, contract

    if partial.dtype != torch.float32:
        errors.append(f"partial_sdf dtype {partial.dtype} != torch.float32")
    if target.dtype != torch.float32:
        errors.append(f"target_sdf dtype {target.dtype} != torch.float32")
    if known.dtype != torch.bool:
        errors.append(f"known_mask dtype {known.dtype} != torch.bool")
    if anchors.dtype != torch.float32:
        errors.append(f"anchors dtype {anchors.dtype} != torch.float32")
    if normals.dtype != torch.float32:
        errors.append(f"normals dtype {normals.dtype} != torch.float32")

    for name, arr in (
        ("partial_sdf", partial),
        ("target_sdf", target),
        ("anchors", anchors),
        ("normals", normals),
    ):
        if not bool(torch.isfinite(torch.as_tensor(arr)).all()):
            errors.append(f"{name} contains non-finite values")
    if errors:
        return errors, contract

    if anchors.shape[0] == 0:
        errors.append("no anchors provided")
        return errors, contract

    if not bool(known.any()):
        errors.append("known_mask is empty (no observation)")
    if not bool((~known).any()):
        contract.append("known_mask covers the full grid (observation equals ground truth)")

    known_t = torch.as_tensor(known)
    partial_t = torch.as_tensor(partial)
    outside = partial_t[~known_t]
    if outside.numel():
        dev = (outside - filler).abs().max().item()
        if dev > 1e-6:
            contract.append(f"filler outside known_mask deviates by {dev:.3e} (expected {filler})")
    inside = partial_t[known_t]
    if inside.numel():
        over = (inside.abs() - truncation).max().item()
        if over > 1e-5:
            contract.append(f"partial_sdf inside known_mask exceeds truncation by {over:.3e}")
    target_t = torch.as_tensor(target)
    over_t = (target_t.abs() - truncation).max().item()
    if over_t > 1e-5:
        contract.append(f"target_sdf exceeds truncation by {over_t:.3e}")

    anchors_t = torch.as_tensor(anchors)
    if bool((anchors_t < lo).any()) or bool((anchors_t > hi).any()):
        contract.append("anchors lie outside the normalized [-1, 1] domain")
    # Anchor raster location is NOT a validity criterion.  Anchors are true
    # first-hit surface points sampled at ray resolution (sub-voxel relative to
    # the 64^3 observation lattice), and each carries its own exact zero value
    # plus outward normal.  A nearest/containing grid node being outside
    # known_mask says nothing about that constraint: the model never reads a
    # value from the node or from the partial-TSDF filler.  Only the numerical
    # contract (finite, in-domain, unit normals) is validated here.
    norms = torch.as_tensor(normals).norm(dim=-1)
    dev = float((norms - 1.0).abs().max())
    if dev > normal_tol:
        contract.append(f"normals are not unit length within {normal_tol}: max deviation {dev:.3e}")
    return errors, contract


class ManifestDataset(Dataset):
    """Lazy NPZ dataset over a manifest list, with per-sample validation."""

    def __init__(
        self,
        manifest: str | os.PathLike,
        resolution: int = 64,
        truncation: float = TRUNCATION,
        filler: float = FILLER,
        max_samples: Optional[int] = None,
        strict: bool = True,
    ) -> None:
        self.manifest = str(manifest)
        self.resolution = resolution
        self.truncation = truncation
        self.filler = filler
        self.strict = strict
        with open(self.manifest, "r", encoding="utf-8") as fh:
            entries = json.load(fh)
        if not isinstance(entries, list):
            raise DataContractError(self.manifest, ["manifest is not a JSON list"])
        bad = [i for i, e in enumerate(entries) if not isinstance(e, dict) or "path" not in e]
        if bad:
            raise DataContractError(self.manifest, [f"entries without 'path': {bad[:5]}"])
        if max_samples is not None:
            entries = entries[:max_samples]
        self.entries = entries
        self.base_dir = os.path.dirname(os.path.abspath(self.manifest))
        self.errors: List[str] = []
        self.contract_violations: List[str] = []

    def __len__(self) -> int:
        return len(self.entries)

    def resolve(self, entry: Dict[str, object]) -> str:
        path = str(entry["path"])
        return path if os.path.isabs(path) else os.path.join(self.base_dir, path)

    def __getitem__(self, index: int) -> Dict[str, object]:
        entry = self.entries[index]
        path = self.resolve(entry)
        if not os.path.exists(path):
            raise DataContractError(path, ["file does not exist"])
        npz = _load_npz(path)
        errors, contract = validate_sample(
            npz,
            source=path,
            resolution=self.resolution,
            truncation=self.truncation,
            filler=self.filler,
        )
        source = str(entry.get("object_id", path))
        if errors:
            self.errors.extend(f"{source}: {e}" for e in errors)
            raise DataContractError(path, errors)
        if contract:
            self.contract_violations.extend(f"{source}: {c}" for c in contract)
            if self.strict:
                raise DataContractError(path, contract)

        sample: Dict[str, object] = {
            "object_id": str(entry.get("object_id", os.path.basename(path))),
            "partial_sdf": torch.as_tensor(npz["partial_sdf"]).unsqueeze(0),
            "known_mask": torch.as_tensor(npz["known_mask"]).unsqueeze(0),
            "target_sdf": torch.as_tensor(npz["target_sdf"]).unsqueeze(0),
            "anchors": torch.as_tensor(npz["anchors"]),
            "normals": torch.as_tensor(npz["normals"]),
            "extras": {k: v for k, v in npz.items() if k not in REQUIRED_KEYS},
        }
        return sample


def _load_npz(path: str) -> Dict[str, object]:
    import numpy as np

    # The data contract stores native numeric arrays only; never unpickle.
    with np.load(path, allow_pickle=False) as data:
        out: Dict[str, object] = {}
        for key in data.files:
            arr = data[key]
            if arr.dtype == object:  # unreachable under a valid contract
                raise DataContractError(
                    path,
                    [
                        f"array '{key}' has dtype object; object arrays are forbidden "
                        "(allow_pickle=False)"
                    ],
                )
            if arr.dtype.kind in ("U", "S"):
                # Non-numeric metadata (e.g. camera_json, source_mesh_sha256)
                # is preserved verbatim for Batch.extras and never tensorized.
                out[key] = arr
            else:
                out[key] = torch.from_numpy(np.ascontiguousarray(arr))
        return out


def collate_samples(samples: Sequence[Dict[str, object]]) -> Batch:
    if not samples:
        raise ValueError("empty batch")
    anchor_counts = {s["anchors"].shape[0] for s in samples}
    if len(anchor_counts) != 1:
        raise DataContractError(
            "collate", [f"inconsistent anchor counts in batch: {sorted(anchor_counts)}"]
        )
    return Batch(
        object_ids=[str(s["object_id"]) for s in samples],
        partial_sdf=torch.stack([s["partial_sdf"] for s in samples]).float(),
        known_mask=torch.stack([s["known_mask"] for s in samples]).bool(),
        target_sdf=torch.stack([s["target_sdf"] for s in samples]).float(),
        anchors=torch.stack([s["anchors"] for s in samples]).float(),
        normals=torch.stack([s["normals"] for s in samples]).float(),
        extras=[s.get("extras", {}) for s in samples],
    )


def build_loader(
    manifest: str | os.PathLike,
    batch_size: int = 1,
    shuffle: bool = False,
    seed: int = 111,
    num_workers: int = 0,
    resolution: int = 64,
    max_samples: Optional[int] = None,
    strict: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Deterministic loader (seeded shuffle, ``num_workers=0`` by default)."""
    dataset = ManifestDataset(
        manifest, resolution=resolution, max_samples=max_samples, strict=strict
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_samples,
        generator=generator if shuffle else None,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


def first_batch(loader: DataLoader) -> Batch:
    """First batch of a loader (the batch that preflight verifies)."""
    for batch in loader:
        return batch
    raise DataContractError("loader", ["dataset is empty"])
