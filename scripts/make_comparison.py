#!/usr/bin/env python3
"""Compose a labelled comparison video for one held-out sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np


def label(frame: np.ndarray, text: str) -> np.ndarray:
    frame = frame.copy()
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(
        frame, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    return frame


def load(path: Path, size: int = 256) -> np.ndarray:
    frames = np.stack(iio.mimread(str(path)))
    return np.stack(
        [
            cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            for frame in frames
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True)
    parser.add_argument("--videos", required=True, help="dir with {sample}_{method}.mp4")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument(
        "--methods",
        default="gt,unconstrained,soft,posthoc,every_step",
    )
    args = parser.parse_args()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    items = [("target", load(Path(args.eval) / "train" / f"{args.sample:04d}.mp4"))]
    for method in methods:
        path = Path(args.videos) / f"{args.sample}_{method}.mp4"
        if path.exists():
            items.append((method, load(path)))

    frames = []
    for frame_index in range(min(item[1].shape[0] for item in items)):
        row = [label(item[1][frame_index], item[0]) for item in items]
        while len(row) % 3:
            row.append(np.zeros_like(row[0]))
        rows = [np.concatenate(row[start : start + 3], axis=1) for start in range(0, len(row), 3)]
        frames.append(np.concatenate(rows, axis=0))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print("saved", output)


if __name__ == "__main__":
    main()
