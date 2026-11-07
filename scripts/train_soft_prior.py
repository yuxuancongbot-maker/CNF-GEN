#!/usr/bin/env python3
"""Train the soft-constraint baseline on sparse anchor values."""

from __future__ import annotations

import argparse

from cnf_motion.config import ANCHOR_FRAMES_SPARSE
from cnf_motion.dataset import load_q
from cnf_motion.soft_field import fit_soft_prior, save_soft_prior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="dataset directory")
    parser.add_argument("--output", required=True, help="checkpoint path")
    parser.add_argument("--soft-weight", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()

    q = load_q(args.data)
    checkpoint = fit_soft_prior(
        q,
        anchor_frames=ANCHOR_FRAMES_SPARSE,
        soft_weight=args.soft_weight,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
    )
    save_soft_prior(args.output, checkpoint)
    print("saved", args.output)


if __name__ == "__main__":
    main()
