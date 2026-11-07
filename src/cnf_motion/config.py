"""Shared constants for the 6-DOF pick-and-place testbed.

All geometry is in metres.  Time is normalised to ``t in [0, 1]`` with
``FRAMES`` uniformly spaced samples; frame ``k`` corresponds to
``t_k = k / (FRAMES - 1)``.
"""

from __future__ import annotations

FRAMES = 17
JOINTS = 7
DEGREE = 10
DIM = DEGREE * JOINTS

SIZE = 256
FPS = 8

TABLE_TOP = 0.30
TABLE_CENTER = (0.55, 0.0)
TABLE_HALF = (0.55, 0.45)
CUBE_HALF = 0.025
GRASP_OFFSET = 0.040
LIFT_Z = 0.50

FINGER_CLOSED = 0.025
FINGER_OPEN = 0.040
GRASP_FRAMES = (4, 13)

CAMERA = {
    "target": [0.42, 0.02, 0.33],
    "distance": 0.92,
    "yaw": 30,
    "pitch": -24,
}

PROMPT = (
    "A Franka Panda robotic arm picks up an orange cube from a wooden table "
    "and places it at a target location. Smooth, realistic robot motion."
)

# Dense and sparse constraint sets used in the reported experiments.
ANCHOR_FRAMES_DENSE = (0, 3, 4, 6, 10, 13, 16)
ANCHOR_FRAMES_SPARSE = (0, 4, 8, 12, 13, 16)
VELOCITY_FRAME = 8
CLEARANCE_FRAMES = (2, 3, 5, 6, 7, 9, 10, 11, 13, 14)

# Sampling defaults.  ``seed + sample * 100 + attempt * 100`` reproduces the
# original experiment seeds.
STEPS = 50
INIT_SCALE = 0.5
PROJECT_EPS = 1e-8
