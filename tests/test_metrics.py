import numpy as np

from cnf_motion.metrics import detect_cube, paired_delta, psnr, trajectory_metrics


def test_psnr_identical_is_infinite():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert psnr(frame, frame) == float("inf")


def test_detect_cube_centroid():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[10:20, 12:22] = (240, 120, 20)
    point = detect_cube(frame)
    assert point is not None
    assert np.allclose(point, [16.5, 14.5], atol=0.5)


def test_trajectory_metrics_zero_for_identical():
    q = np.zeros((17, 7), dtype=np.float32)
    ee = np.zeros((17, 3), dtype=np.float32)
    row = trajectory_metrics(q, q, ee, ee, anchor_frames=[0, 8, 16])
    assert row["anchor_error"] == 0
    assert row["joint_rmse"] == 0
    assert row["ee_rmse"] == 0
    assert row["jerk"] == 0


def test_paired_delta():
    rows = [
        {"sample": 0, "method": "a", "psnr": 2.0},
        {"sample": 0, "method": "b", "psnr": 1.0},
        {"sample": 1, "method": "a", "psnr": 3.0},
        {"sample": 1, "method": "b", "psnr": 3.0},
    ]
    result = paired_delta(rows, "a", "b", "psnr", higher_is_better=True)
    assert result["mean"] == 0.5
    assert result["better"] == 1
    assert result["n"] == 2
