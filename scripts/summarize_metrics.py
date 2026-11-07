#!/usr/bin/env python3
"""Print means and paired deltas from a metrics JSON file."""

from __future__ import annotations

import argparse
import json

from cnf_motion.metrics import mean_metrics, paired_delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--reference", default="every_step")
    parser.add_argument("--methods", default=None)
    parser.add_argument("--keys", default="psnr,anchor_error,joint_rmse,ee_rmse,ee_velocity_error,jerk")
    parser.add_argument("--higher", default="psnr")
    args = parser.parse_args()

    rows = json.load(open(args.metrics))
    keys = [key.strip() for key in args.keys.split(",") if key.strip()]
    higher = {key.strip() for key in args.higher.split(",") if key.strip()}
    methods = [row["method"] for row in rows]
    seen = []
    for method in methods:
        if method not in seen:
            seen.append(method)

    available = [key for key in keys if all(key in row for row in rows)]
    print("=== means ===")
    for method in seen:
        print(method, mean_metrics(rows, method, available))

    print(f"\n=== paired deltas vs {args.reference} (method - reference) ===")
    targets = args.methods.split(",") if args.methods else [m for m in seen if m != args.reference]
    for method in targets:
        print(method)
        for key in available:
            result = paired_delta(
                rows, method, args.reference, key, higher_is_better=key in higher
            )
            print(
                f"  {key}: mean {result['mean']:+.5f} "
                f"se {result['se']:.5f} better {result['better']}/{result['n']}"
            )


if __name__ == "__main__":
    main()
