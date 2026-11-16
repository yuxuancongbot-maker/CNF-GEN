"""Hard-constrained motion fields for video world models.

Main pieces:
  - ``basis``: Legendre trajectory representation.
  - ``motion_field``: flow prior + exact joint-space projection.
  - ``constraints``: sparse waypoints + task-space velocity.
  - ``scene`` / ``dataset``: PyBullet pick-and-place data.
  - ``video_wm``: 12.88M latent-level action world model.
  - ``metrics``: trajectory and pixel metrics.
"""

from .basis import (
    coefficients_from_trajectory,
    make_basis,
    make_derivative_basis,
    trajectory_from_coefficients,
)
from .constraints import project_sparse, sample_sparse, task_space_velocity
from .motion_field import (
    PandaFlowModel,
    constraint_matrix,
    project_coefficients,
    sample_with_projection,
)
from .video_wm import VideoWorldModel6D

__all__ = [
    "make_basis",
    "make_derivative_basis",
    "coefficients_from_trajectory",
    "trajectory_from_coefficients",
    "PandaFlowModel",
    "constraint_matrix",
    "project_coefficients",
    "sample_with_projection",
    "project_sparse",
    "sample_sparse",
    "task_space_velocity",
    "VideoWorldModel6D",
]
