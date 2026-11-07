#!/usr/bin/env python3
"""Generate the 6-DOF pick-and-place dataset (RGB + control + q)."""

from __future__ import annotations

import argparse

from cnf_motion.dataset import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="dataset directory")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    generate_dataset(
        args.output,
        count=args.count,
        seed=args.seed,
        size=args.size,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
