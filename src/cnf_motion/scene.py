"""PyBullet scene, analytic keyframe motion, IK, and semantic rendering.

The scene is a Franka Panda, a 5 cm orange cube, and a wooden table.  A
trajectory is generated from Cartesian keyframes, interpolated with a
minimum-jerk profile, and solved frame-by-frame with PyBullet IK.
"""

from __future__ import annotations

import numpy as np
import pybullet as p
import pybullet_data

from .config import (
    CAMERA,
    CUBE_HALF,
    FINGER_CLOSED,
    FINGER_OPEN,
    GRASP_FRAMES,
    GRASP_OFFSET,
    LIFT_Z,
    TABLE_CENTER,
    TABLE_HALF,
    TABLE_TOP,
)


def build_scene():
    """Create a direct (headless) PyBullet scene.

    Returns ``(robot, cube, table_id)``.  Call :func:`disconnect_scene` when
    finished to release the connection.
    """
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1 / 240)

    floor_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2, 2, 0.01])
    floor_vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[2, 2, 0.01], rgbaColor=[0.82, 0.82, 0.84, 1]
    )
    p.createMultiBody(0, floor_col, floor_vis, [0, 0, -0.01])

    table_col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[TABLE_HALF[0], TABLE_HALF[1], 0.03]
    )
    table_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[TABLE_HALF[0], TABLE_HALF[1], 0.03],
        rgbaColor=[0.58, 0.44, 0.30, 1],
    )
    table = p.createMultiBody(
        0, table_col, table_vis, [TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP - 0.03]
    )
    for x, y in [(0.08, -0.36), (1.02, -0.36), (0.08, 0.36), (1.02, 0.36)]:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.135])
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.03, 0.03, 0.135],
            rgbaColor=[0.35, 0.28, 0.22, 1],
        )
        p.createMultiBody(0, col, vis, [x, y, 0.135])

    robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    cube_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[CUBE_HALF] * 3)
    cube_vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[CUBE_HALF] * 3, rgbaColor=[0.95, 0.35, 0.05, 1]
    )
    cube = p.createMultiBody(
        0.05, cube_col, cube_vis, [0.46, 0.10, TABLE_TOP + CUBE_HALF]
    )
    return robot, cube, table


def disconnect_scene() -> None:
    if p.isConnected():
        p.disconnect()


def solve_ik(robot, target, prev_q, yaw: float):
    """IK for the downward-pointing gripper at a Cartesian target.

    Returns ``(q, residual)`` where ``q`` is the 7-DOF arm configuration.
    """
    orn = p.getQuaternionFromEuler([np.pi, 0.0, yaw])
    for joint, value in enumerate(prev_q):
        p.resetJointState(robot, joint, value)
    p.resetJointState(robot, 9, FINGER_OPEN)
    p.resetJointState(robot, 10, FINGER_OPEN)
    rest = list(prev_q) + [0.0, 0.0]
    solution = p.calculateInverseKinematics(
        robot,
        11,
        target,
        orn,
        restPoses=rest,
        jointDamping=[0.05] * 9,
        maxNumIterations=500,
        residualThreshold=1e-6,
    )
    q = np.array(solution[:7], dtype=np.float32)
    for joint, value in enumerate(q):
        p.resetJointState(robot, joint, value)
    state = p.getLinkState(robot, 11, computeForwardKinematics=True)
    residual = float(np.linalg.norm(np.array(state[0]) - np.array(target)))
    return q, residual


def minimum_jerk(s: float) -> float:
    return s * s * s * (10 - 15 * s + 6 * s * s)


def interpolate(a, b, s: float):
    return np.array(a) + (np.array(b) - np.array(a)) * minimum_jerk(s)


def make_keys(cube_xy, target_xy):
    """Cartesian keyframes for pick, lift, move, place, and retreat."""
    grasp_z = TABLE_TOP + CUBE_HALF + GRASP_OFFSET
    grasp = [cube_xy[0], cube_xy[1], grasp_z]
    above = [cube_xy[0], cube_xy[1], LIFT_Z]
    place = [target_xy[0], target_xy[1], grasp_z]
    place_above = [target_xy[0], target_xy[1], LIFT_Z]
    return [
        (0, above),
        (3, grasp),
        (4, grasp),
        (6, above),
        (10, place_above),
        (12, place),
        (13, place),
        (16, place_above),
    ]


def end_effector_at(frame: int, keys):
    for index in range(len(keys) - 1):
        f0, a = keys[index]
        f1, b = keys[index + 1]
        if f0 <= frame <= f1:
            if f1 == f0:
                return np.array(a)
            return interpolate(a, b, (frame - f0) / (f1 - f0))
    return np.array(keys[-1][1])


def render(robot, cube, table_id: int, width: int = 256):
    """Render RGB and a 3-channel semantic control image.

    Control colours: table (80, 80, 80), cube (255, 120, 20),
    arm (30, 80, 220), gripper links 8-11 (30, 200, 120).
    """
    view = p.computeViewMatrixFromYawPitchRoll(
        CAMERA["target"],
        CAMERA["distance"],
        CAMERA["yaw"],
        CAMERA["pitch"],
        0,
        2,
    )
    proj = p.computeProjectionMatrixFOV(50, 1.0, 0.05, 5)
    image = p.getCameraImage(
        width,
        width,
        view,
        proj,
        renderer=p.ER_TINY_RENDERER,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    )
    rgb = np.array(image[2]).reshape(width, width, 4)[..., :3].astype(np.uint8)
    seg = np.array(image[4]).reshape(width, width).astype(np.uint32)
    object_id = seg & 0xFFFFFF
    link_index = seg >> 24
    control = np.zeros((width, width, 3), np.uint8)
    control[object_id == table_id] = (80, 80, 80)
    control[object_id == cube] = (255, 120, 20)
    robot_mask = object_id == robot
    control[robot_mask] = (30, 80, 220)
    gripper_mask = robot_mask & np.isin(link_index, [8, 9, 10, 11])
    control[gripper_mask] = (30, 200, 120)
    return rgb, control


def _place_fingers(robot, frame: int) -> None:
    finger = FINGER_CLOSED if GRASP_FRAMES[0] <= frame <= GRASP_FRAMES[1] else FINGER_OPEN
    p.resetJointState(robot, 9, finger)
    p.resetJointState(robot, 10, finger)


def _grasped_cube_pose(robot):
    state = p.getLinkState(robot, 11, computeForwardKinematics=True)
    hand_pos = np.array(state[0])
    hand_orn = np.array(state[1])
    return hand_pos + np.array([0, 0, -GRASP_OFFSET]), hand_orn


def generate_one(robot, cube, table_id: int, rng: np.random.Generator, width: int = 256):
    """Generate one valid pick-and-place sample.

    Rejection sampling is used for the cube/target separation and for IK
    residuals (``<= 1e-3 m``) and per-frame joint jumps (``<= 0.6 rad``).
    """
    for _ in range(20):
        cube_xy = np.array(
            [rng.uniform(0.42, 0.50), rng.uniform(-0.14, 0.14)], dtype=np.float32
        )
        target_xy = np.array(
            [rng.uniform(0.56, 0.68), rng.uniform(-0.18, 0.18)], dtype=np.float32
        )
        if np.linalg.norm(target_xy - cube_xy) < 0.10:
            continue
        yaw = float(rng.choice([0.0, np.pi / 2]))
        keys = make_keys(cube_xy, target_xy)
        p.resetBasePositionAndOrientation(
            cube, [cube_xy[0], cube_xy[1], TABLE_TOP + CUBE_HALF], [0, 0, 0, 1]
        )
        prev_q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.7], dtype=np.float32)
        qs = []
        residuals = []
        ok = True
        for frame in range(17):
            target = end_effector_at(frame, keys)
            q, residual = solve_ik(robot, target, prev_q, yaw)
            residuals.append(residual)
            if residual > 1e-3:
                ok = False
                break
            if qs and np.abs(q - prev_q).max() > 0.6:
                ok = False
                break
            qs.append(q.copy())
            prev_q = q
        if not ok:
            continue

        frames_rgb, frames_control = [], []
        cube_pos = np.array([cube_xy[0], cube_xy[1], TABLE_TOP + CUBE_HALF])
        cube_orn = np.array([0, 0, 0, 1], dtype=np.float64)
        for frame in range(17):
            q = qs[frame]
            for joint, value in enumerate(q):
                p.resetJointState(robot, joint, value)
            _place_fingers(robot, frame)
            if GRASP_FRAMES[0] <= frame <= GRASP_FRAMES[1]:
                cube_pos, cube_orn = _grasped_cube_pose(robot)
            p.resetBasePositionAndOrientation(cube, cube_pos.tolist(), cube_orn.tolist())
            rgb, control = render(robot, cube, table_id, width)
            frames_rgb.append(rgb)
            frames_control.append(control)

        return {
            "cube_xy": cube_xy,
            "target_xy": target_xy,
            "yaw": yaw,
            "rgb": np.stack(frames_rgb),
            "control": np.stack(frames_control),
            "q": np.stack(qs),
            "residuals": residuals,
        }
    raise RuntimeError("failed to generate a valid trajectory")


def render_trajectory(robot, cube, table_id: int, q, cube_start, width: int = 256):
    """Render an existing joint trajectory with the same grasp logic.

    Returns ``(rgb [T,W,W,3], control [T,W,W,3], cube_positions [T,3])``.
    """
    frames_rgb, frames_control = [], []
    cube_pos = np.array([cube_start[0], cube_start[1], TABLE_TOP + CUBE_HALF])
    cube_orn = np.array([0, 0, 0, 1], dtype=np.float64)
    cube_positions = []
    for frame in range(q.shape[0]):
        for joint in range(7):
            p.resetJointState(robot, joint, float(q[frame, joint]))
        _place_fingers(robot, frame)
        if GRASP_FRAMES[0] <= frame <= GRASP_FRAMES[1]:
            cube_pos, cube_orn = _grasped_cube_pose(robot)
        p.resetBasePositionAndOrientation(cube, cube_pos.tolist(), cube_orn.tolist())
        rgb, control = render(robot, cube, table_id, width)
        frames_rgb.append(rgb)
        frames_control.append(control)
        cube_positions.append(cube_pos.copy())
    return np.stack(frames_rgb), np.stack(frames_control), np.stack(cube_positions)


def ee_position(robot, q) -> np.ndarray:
    """Forward kinematics position of link 11 (gripper base)."""
    for joint in range(7):
        p.resetJointState(robot, joint, float(q[joint]))
    return np.array(p.getLinkState(robot, 11, computeForwardKinematics=True)[0])


def ee_jacobian(robot, q, eps: float = 1e-4) -> np.ndarray:
    """Central-difference 3x7 translational Jacobian at ``q``."""
    q = np.array(q, dtype=float)
    jacobian = np.zeros((3, 7))
    for joint in range(7):
        plus = q.copy()
        minus = q.copy()
        plus[joint] += eps
        minus[joint] -= eps
        jacobian[:, joint] = (ee_position(robot, plus) - ee_position(robot, minus)) / (2 * eps)
    return jacobian


def enforce_clearance(robot, q, yaw: float):
    """Post-hoc IK lift for frames whose gripper is below a safe height.

    Safe height is ``0.3655 m`` during grasp frames 4-13 and ``0.332 m``
    otherwise.  This is applied after sampling; it can perturb anchor frames,
    so callers should re-validate anchors afterwards.
    """
    q = np.array(q, dtype=np.float32).copy()
    for frame in range(q.shape[0]):
        safe_z = 0.3655 if GRASP_FRAMES[0] <= frame <= GRASP_FRAMES[1] else 0.332
        ee = ee_position(robot, q[frame])
        if ee[2] < safe_z:
            target = [ee[0], ee[1], safe_z]
            orn = p.getQuaternionFromEuler([np.pi, 0.0, yaw])
            rest = list(q[frame]) + [FINGER_OPEN, FINGER_OPEN]
            for joint in range(7):
                p.resetJointState(robot, joint, float(q[frame, joint]))
            p.resetJointState(robot, 9, FINGER_OPEN)
            p.resetJointState(robot, 10, FINGER_OPEN)
            solution = p.calculateInverseKinematics(
                robot,
                11,
                target,
                orn,
                restPoses=rest,
                jointDamping=[0.05] * 9,
                maxNumIterations=500,
                residualThreshold=1e-6,
            )
            q[frame] = np.array(solution[:7], dtype=np.float32)
    return q


def check_limits(robot, q) -> bool:
    """Check 7-DOF joint limits with a 1e-4 rad tolerance."""
    lower, upper = [], []
    for joint in range(7):
        info = p.getJointInfo(robot, joint)
        lower.append(info[8])
        upper.append(info[9])
    lower = np.array(lower)
    upper = np.array(upper)
    return bool(np.all(q >= lower - 1e-4) and np.all(q <= upper + 1e-4))


def hand_clearance(robot, q) -> float:
    """Minimum gripper height above the nominal grasp point during grasp."""
    hand_z = []
    for frame in range(q.shape[0]):
        for joint in range(7):
            p.resetJointState(robot, joint, float(q[frame, joint]))
        hand_z.append(p.getLinkState(robot, 11, computeForwardKinematics=True)[0][2])
    return float(min(hand_z[4:14]) - (TABLE_TOP + CUBE_HALF + GRASP_OFFSET))


def table_clearance(cube_positions) -> float:
    """Minimum cube-bottom height above the table."""
    return float((cube_positions[:, 2] - CUBE_HALF - TABLE_TOP).min())


__all__ = [
    "build_scene",
    "disconnect_scene",
    "solve_ik",
    "minimum_jerk",
    "interpolate",
    "make_keys",
    "end_effector_at",
    "render",
    "generate_one",
    "render_trajectory",
    "ee_position",
    "ee_jacobian",
    "enforce_clearance",
    "check_limits",
    "hand_clearance",
    "table_clearance",
]
