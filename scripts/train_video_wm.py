#!/usr/bin/env python3
"""Train the 12.88M action-conditioned video world model."""

from __future__ import annotations

import argparse

from cnf_motion.video_wm import train_video_wm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=400)
    args = parser.parse_args()
    train_video_wm(
        args.data,
        args.output,
        count=args.count,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        hidden=args.hidden,
        size=args.size,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
