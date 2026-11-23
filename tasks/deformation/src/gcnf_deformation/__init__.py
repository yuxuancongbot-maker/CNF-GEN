"""Task-local adapters for hard-constrained deformation fields."""

from .condition import (
    resolve_explicit_handle_constraints,
    resolve_handle_count,
    resolve_userhandle_region,
    resolve_userhandle_translation,
)
from .correction import (
    CorrectedDisplacementField,
    correct_with_fixed_gaussian_rbf,
    fit_soft_gaussian_rbf_correction,
)
from .field import displacement_from_mapping
from .nsdp_adapter import NSDPPointQuery
from .selection import moved_vertex_candidate_pool

__all__ = [
    "CorrectedDisplacementField",
    "correct_with_fixed_gaussian_rbf",
    "fit_soft_gaussian_rbf_correction",
    "displacement_from_mapping",
    "NSDPPointQuery",
    "moved_vertex_candidate_pool",
    "resolve_explicit_handle_constraints",
    "resolve_userhandle_translation",
    "resolve_userhandle_region",
    "resolve_handle_count",
]
