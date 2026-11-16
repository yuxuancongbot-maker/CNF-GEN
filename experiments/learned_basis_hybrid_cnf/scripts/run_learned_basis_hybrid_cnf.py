#!/usr/bin/env python3
"""Thin entry for the Learned-basis Hybrid-CNF evaluation package.

This is the migrated, path-configurable version of the 2026-11-14 v2 experiment
entry (``run_learned_basis_hybrid_cnf_20261114.py``). All external resources
(dataset, Wan model/LoRA, VideoX-Fun, frozen evaluator) and the output directory
are resolved by ``learned_basis_hybrid_cnf.config`` from environment variables
(``CNF_V2_*``) or a JSON config file; there are no hardcoded machine paths.

Stages (run in order for a full reproduction):

    train  ->  motion  ->  video  ->  score  ->  stats   (then psnr_eval.py)

``--run-all`` is a shorthand for ``motion + video + score + stats + report`` and
**assumes training already produced ``OUT/train/fixed.pt`` and ``learned.pt``**;
it does not train and does not compute PSNR. Training is a separate explicit
step (``--stage train``). PSNR is a separate CPU-only script (``psnr_eval.py``).

Run ``--verify-config`` to print the resolved configuration, then ``--verify-ready``
for the CPU-only numerical verification before any GPU work.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

import torch

from learned_basis_hybrid_cnf.config import assert_configured, missing, resolved
from learned_basis_hybrid_cnf.evaluate import prepare_and_verify_ready, run_full_evaluation
from learned_basis_hybrid_cnf.sampling import run_all_sampling_and_verification
from learned_basis_hybrid_cnf.scoring import run_all_scoring
from learned_basis_hybrid_cnf.statistics import run_all_statistics
from learned_basis_hybrid_cnf.train import run_all_training
from learned_basis_hybrid_cnf.util_eval import update_status_eval
from learned_basis_hybrid_cnf.video import run_all_wan_videos


def _print_config() -> None:
    print(json.dumps(resolved(), indent=2, ensure_ascii=False), flush=True)
    print("missing (require_wan=True):", missing(require_wan=True), flush=True)
    print("missing (require_wan=False):", missing(require_wan=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Learned-basis Hybrid-CNF Evaluation Entry")
    parser.add_argument("--verify-config", action="store_true", help="Print resolved configuration and exit")
    parser.add_argument("--verify-ready", action="store_true", help="Run CPU numerical checks and write evaluation_ready.json")
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run motion + 256-video + scoring + stats + report (requires prior --stage train)",
    )
    parser.add_argument(
        "--stage",
        choices=["train", "ready", "motion", "video", "score", "stats", "all"],
        default=None,
        help="Run a specific stage (train | ready | motion | video | score | stats | all)",
    )
    parser.add_argument("--device", default=None, help="Device for motion sampling / training (default: cuda if available else cpu)")
    parser.add_argument("--force", action="store_true", help="Allow overwriting existing training checkpoints in CNF_V2_OUT")
    args = parser.parse_args()

    try:
        if args.verify_config:
            _print_config()
            return 0

        if args.stage == "train":
            assert_configured(require_wan=False)
            comp = run_all_training(
                device_str=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
                force=args.force,
            )
            print(
                json.dumps(
                    {
                        "status": comp["status"],
                        "fixed_sha256": comp["checkpoints"]["fixed"]["sha256"],
                        "learned_sha256": comp["checkpoints"]["learned"]["sha256"],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 0

        if args.verify_ready or args.stage == "ready":
            assert_configured(require_wan=False)
            res = prepare_and_verify_ready()
            print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
            return 0 if res.get("evaluation_code_ready") else 1

        if args.run_all or args.stage == "all":
            assert_configured(require_wan=True)
            comp = run_full_evaluation(device=args.device)
            print(json.dumps(comp, indent=2, ensure_ascii=False), flush=True)
            return 0

        if args.stage == "motion":
            assert_configured(require_wan=False)
            res = run_all_sampling_and_verification(device=args.device or "cpu")
            print(json.dumps(res["summary"], indent=2, ensure_ascii=False), flush=True)
            return 0

        if args.stage == "video":
            assert_configured(require_wan=True)
            res = run_all_wan_videos()
            print(json.dumps(res["summary"], indent=2, ensure_ascii=False), flush=True)
            return 0

        if args.stage == "score":
            assert_configured(require_wan=False)
            res = run_all_scoring()
            print(json.dumps({"scored": len(res["per_video_scores"])}, indent=2), flush=True)
            return 0

        if args.stage == "stats":
            assert_configured(require_wan=False)
            res = run_all_statistics()
            print(json.dumps(res["primary_outcome"], indent=2, ensure_ascii=False), flush=True)
            return 0

        # Default behavior: verify ready, and if checkpoints are ready & GPU free, run full evaluation
        assert_configured(require_wan=False)
        ready = prepare_and_verify_ready()
        if ready.get("checkpoints_ready") and ready.get("gpu_status", {}).get("gpu_free"):
            assert_configured(require_wan=True)
            comp = run_full_evaluation(device=args.device)
            print(json.dumps(comp, indent=2, ensure_ascii=False), flush=True)
            return 0
        print(json.dumps(ready, indent=2, ensure_ascii=False), flush=True)
        return 0
    except Exception as e:
        update_status_eval("failed", error=str(e), traceback=traceback.format_exc())
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
