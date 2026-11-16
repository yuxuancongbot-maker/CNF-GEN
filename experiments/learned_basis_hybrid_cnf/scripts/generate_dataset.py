#!/usr/bin/env python3
"""Generate the 6-DOF pick-and-place dataset (RGB + control + q).

Thin wrapper over ``cnf_motion.dataset.generate_dataset``. The reported splits
use ``--seed 0 --count 128`` for training and ``--seed 1 --count 16`` for
held-out evaluation.

Refuses to overwrite an existing dataset directory (``metadata.json`` present)
unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cnf_motion.dataset import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="dataset directory")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="overwrite an existing dataset directory")
    args = parser.parse_args()

    out = Path(args.output)
    if (out / "metadata.json").exists() and not args.force:
        raise SystemExit(
            f"dataset already exists at {out} (metadata.json present); use --force to overwrite"
        )

    generate_dataset(
        args.output,
        count=args.count,
        seed=args.seed,
        size=args.size,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
