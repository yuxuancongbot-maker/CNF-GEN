#!/usr/bin/env python3
"""Independent verification gate checker for run_eval.py outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from warp_core import atomic_json, load_json

ARMS = [
    "hy3d_raw",
    "hy3d_cnf_gaussian_patch",
    "hy3d_cnf_warp_p2p",
    "hy3d_cnf_normal_warp",
]


def finite_tree(x: Any) -> bool:
    if isinstance(x, float):
        return math.isfinite(x)
    if isinstance(x, dict):
        return all(finite_tree(v) for v in x.values())
    if isinstance(x, list):
        return all(finite_tree(v) for v in x)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify run_eval.py outputs and constraints.")
    ap.add_argument("--run-dir", type=Path, required=True, help="Output directory of run_eval.py")
    ap.add_argument("--expected-count", type=int, default=64, help="Expected number of shapes")
    ap.add_argument(
        "--residual-tol",
        type=float,
        default=1e-12,
        help="Max allowed FP64 continuous anchor residual for hy3d_cnf_normal_warp",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    res = load_json(run_dir / "results.json")
    per = res["per_shape"]
    nw_max_f = [r["anchor_residuals"]["hy3d_cnf_normal_warp"]["anchor_value_abs_max"] for r in per]
    nw_fold = [r["folding"]["hy3d_cnf_normal_warp"]["surface_fold_fraction"] for r in per]
    w_diffs = [r["normal_warp_diagnostics"]["w_diff_vs_p2p_fro"] for r in per]

    gates = {
        "expected_count_complete": (
            len(per) == args.expected_count and not res["failures"] and res["status"] == "complete"
        ),
        "zero_nan_inf_results": finite_tree(res),
        "all_arms_finite_chamfer": all(
            r["full_chamfer"][a] is not None and r["hidden_chamfer"][a] is not None
            for r in per
            for a in ARMS
        ),
        "all_alignments_upright_R11_eq_1": all(abs(r["alignment"]["R11"] - 1.0) < 1e-12 for r in per),
        "zero_hidden_leakage_flag": all(
            r["alignment"]["zero_hidden_geometry_leakage"] is True for r in per
        ),
        "nw_all_fp64_residual_below_tol": all(v < args.residual_tol for v in nw_max_f),
        "nw_zero_folding_all": all(v == 0.0 for v in nw_fold),
        "zero_silent_p2p_fallback": all(d > 1e-6 for d in w_diffs),
    }
    failed = [k for k, v in gates.items() if not v]
    report = {
        "status": "passed" if not failed else "failed",
        "n_passed": sum(gates.values()),
        "n_gates": len(gates),
        "failed_gates": failed,
        "gates": gates,
        "diagnostics": {
            "nw_max_fp64_residual": max(nw_max_f) if nw_max_f else None,
            "nw_min_detJ": min(
                (r["folding"]["hy3d_cnf_normal_warp"]["surface_min_detJ"] for r in per),
                default=None,
            ),
            "nw_fold_chairs": sum(v > 0.0 for v in nw_fold),
            "min_w_diff_nw_vs_p2p": min(w_diffs) if w_diffs else None,
        },
    }
    atomic_json(run_dir / "verification.json", report)
    print(json.dumps(report, indent=2))
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
