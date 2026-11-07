import os

import numpy as np
import pytest

from cnf_motion.constraints import task_space_velocity, validate_trajectory
from cnf_motion.scene import build_scene, disconnect_scene, generate_one, render_trajectory


@pytest.mark.skipif(os.environ.get("RUN_PYBULLET") != "1", reason="set RUN_PYBULLET=1")
def test_pybullet_smoke():
    robot, cube, table_id = build_scene()
    try:
        rng = np.random.default_rng(0)
        sample = generate_one(robot, cube, table_id, rng, width=64)
        assert sample["q"].shape == (17, 7)
        assert sample["rgb"].shape == (17, 64, 64, 3)
        assert sample["control"].shape == (17, 64, 64, 3)
        velocity = task_space_velocity(robot, sample["q"])
        assert velocity.shape == (3,)
        _, control, cube_positions = render_trajectory(
            robot, cube, table_id, sample["q"], sample["cube_xy"], width=64
        )
        assert control.shape == (17, 64, 64, 3)
        assert cube_positions.shape == (17, 3)
        check = validate_trajectory(robot, sample["q"])
        assert "valid" in check
    finally:
        disconnect_scene()
