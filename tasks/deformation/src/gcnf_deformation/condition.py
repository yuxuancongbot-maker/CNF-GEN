from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

TRANSLATION_KEYS = ("xtrans", "ytrans", "ztrans")
REGION_KEYS = (
    "head",
    "tail",
    "frontleftfoot",
    "frontrightfoot",
    "behindleftfoot",
    "behindrightfoot",
)


def resolve_handle_count(
    default: int, candidates: list[int] | None = None, override: int | None = None
) -> int:
    allowed = [int(value) for value in ([default] if candidates is None else candidates)]
    if not allowed or any(value <= 0 for value in allowed) or len(set(allowed)) != len(allowed):
        raise ValueError(
            "handle count candidates must be a non-empty list of unique positive integers"
        )
    selected = int(default if override is None else override)
    if selected not in allowed:
        raise ValueError(f"handle count {selected} is not in the configured candidates {allowed}")
    return selected


def resolve_userhandle_region(
    userhandle: Mapping[str, Any], override: str | None = None
) -> dict[str, bool]:
    if override is not None:
        if override not in REGION_KEYS:
            raise ValueError(
                f"unsupported userhandle region {override!r}; expected one of {REGION_KEYS}"
            )
        selected = override
    else:
        selected_regions = [key for key in REGION_KEYS if bool(userhandle.get(key, False))]
        if len(selected_regions) != 1:
            raise ValueError(
                f"expected exactly one active userhandle region, found {selected_regions}"
            )
        selected = selected_regions[0]
    return {key: key == selected for key in REGION_KEYS}


def resolve_userhandle_translation(
    userhandle: Mapping[str, Any], override: Mapping[str, float] | None = None
) -> dict[str, float]:
    """Resolve NSDP drag components without changing the configured handle region."""
    override = override or {}
    unexpected = set(override) - set(TRANSLATION_KEYS)
    if unexpected:
        raise ValueError(
            f"userhandle override only accepts {TRANSLATION_KEYS}; got {sorted(unexpected)}"
        )

    resolved: dict[str, float] = {}
    for key in TRANSLATION_KEYS:
        value = float(override[key] if key in override else userhandle[key])
        if not math.isfinite(value):
            raise ValueError(f"userhandle {key} must be finite")
        resolved[key] = value
    return resolved


def resolve_explicit_handle_constraints(
    constraints: Mapping[str, Any], vertex_count: int | None = None
) -> tuple[list[int], list[list[float]]]:
    """Validate explicit normalized-mesh vertex handles and final displacements."""
    if not isinstance(constraints, Mapping):
        raise ValueError("cnf_handle_constraints must be a JSON object")
    required = {"vertex_indices", "target_displacements_normalized"}
    missing = required - constraints.keys()
    unexpected = constraints.keys() - required
    if missing or unexpected:
        raise ValueError(
            "cnf_handle_constraints requires exactly vertex_indices and "
            "target_displacements_normalized"
        )
    if vertex_count is not None and (
        isinstance(vertex_count, bool) or not isinstance(vertex_count, int) or vertex_count <= 0
    ):
        raise ValueError("vertex_count must be a positive integer")

    raw_indices = constraints["vertex_indices"]
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError("vertex_indices must be a non-empty list")
    indices: list[int] = []
    for value in raw_indices:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("vertex_indices must contain integers")
        if value < 0 or (vertex_count is not None and value >= vertex_count):
            raise ValueError("vertex_indices contain an out-of-range index")
        indices.append(value)
    if len(set(indices)) != len(indices):
        raise ValueError("vertex_indices must be unique")

    raw_targets = constraints["target_displacements_normalized"]
    if not isinstance(raw_targets, list) or len(raw_targets) != len(indices):
        raise ValueError("one target displacement is required for each vertex index")
    targets: list[list[float]] = []
    for target in raw_targets:
        if not isinstance(target, list) or len(target) != 3:
            raise ValueError("target_displacements_normalized must have shape [K, 3]")
        resolved_target: list[float] = []
        for component in target:
            if isinstance(component, bool) or not isinstance(component, (int, float)):
                raise ValueError("target displacement components must be numeric")
            value = float(component)
            if not math.isfinite(value):
                raise ValueError("target displacement components must be finite")
            resolved_target.append(value)
        targets.append(resolved_target)
    return indices, targets
