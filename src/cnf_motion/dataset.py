"""Dataset generation and loading for the 6-DOF pick-and-place testbed."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np
import torch

from .config import FPS, FRAMES, PROMPT, SIZE
from .scene import build_scene, disconnect_scene, generate_one


def generate_dataset(
    output: str | Path,
    *,
    count: int = 128,
    seed: int = 0,
    size: int = SIZE,
    fps: int = FPS,
) -> None:
    """Generate ``count`` RGB/control/q samples.

    The per-sample RNG seed is ``seed * 10000 + index``.  The reported splits
    use ``seed=0, count=128`` for training and ``seed=1, count=16`` for
    held-out evaluation.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for subdir in ("train", "control", "q"):
        (output / subdir).mkdir(exist_ok=True)

    robot, cube, table_id = build_scene()
    metadata = []
    try:
        for index in range(count):
            rng = np.random.default_rng(seed * 10000 + index)
            sample = generate_one(robot, cube, table_id, rng, width=size)
            iio.mimsave(str(output / "train" / f"{index:04d}.mp4"), sample["rgb"], fps=fps)
            iio.mimsave(
                str(output / "control" / f"{index:04d}.mp4"), sample["control"], fps=fps
            )
            np.save(str(output / "q" / f"{index:04d}.npy"), sample["q"])
            metadata.append(
                {
                    "file_path": f"train/{index:04d}.mp4",
                    "control_file_path": f"control/{index:04d}.mp4",
                    "text": PROMPT,
                    "type": "video",
                    "width": size,
                    "height": size,
                    "cube_xy": sample["cube_xy"].tolist(),
                    "target_xy": sample["target_xy"].tolist(),
                    "yaw": float(sample["yaw"]),
                }
            )
            if index % 8 == 0:
                print("generated", index, flush=True)
    finally:
        disconnect_scene()

    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved {count} samples to {output}")


def load_metadata(root: str | Path) -> list[dict]:
    return json.loads((Path(root) / "metadata.json").read_text())


def load_q(root: str | Path, count: int | None = None) -> torch.Tensor:
    """Load joint trajectories as float32 ``[N, 17, 7]``."""
    root = Path(root)
    files = sorted((root / "q").glob("*.npy"))
    if count is not None:
        files = files[:count]
    if not files:
        raise FileNotFoundError(f"no q/*.npy under {root}")
    return torch.tensor(np.stack([np.load(file) for file in files]), dtype=torch.float32)


def load_video_split(
    root: str | Path,
    count: int = 128,
    size: int = 128,
):
    """Load RGB, control, and q arrays for the video world model.

    Returns ``(rgb uint8 [N,T,size,size,3], control uint8, q float32)``.
    RGB is area-resized; control is nearest-resized to preserve labels.
    """
    root = Path(root)
    rgb, control, q = [], [], []
    for index in range(count):
        rgb_frames = np.stack(iio.mimread(str(root / "train" / f"{index:04d}.mp4")))
        control_frames = np.stack(iio.mimread(str(root / "control" / f"{index:04d}.mp4")))
        if len(rgb_frames) != FRAMES:
            raise ValueError(f"{root} sample {index}: expected {FRAMES} frames")
        rgb_frames = np.stack(
            [
                cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                for frame in rgb_frames
            ]
        )
        control_frames = np.stack(
            [
                cv2.resize(frame, (size, size), interpolation=cv2.INTER_NEAREST)
                for frame in control_frames
            ]
        )
        rgb.append(rgb_frames)
        control.append(control_frames)
        q.append(np.load(root / "q" / f"{index:04d}.npy"))
    return (
        torch.tensor(np.stack(rgb), dtype=torch.uint8),
        torch.tensor(np.stack(control), dtype=torch.uint8),
        torch.tensor(np.stack(q), dtype=torch.float32),
    )


__all__ = ["generate_dataset", "load_metadata", "load_q", "load_video_split"]
