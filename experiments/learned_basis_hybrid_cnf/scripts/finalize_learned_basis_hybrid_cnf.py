#!/usr/bin/env python3
"""Thin CPU-only finalizer for the learned-basis Hybrid-CNF evaluation.

Why this exists
---------------
The formal evaluation process (PID 799365) loaded its Python modules at launch.
After launch, `statistics.py` gained conditional both-valid diversity pair metrics
and `scoring.py` gained the reused-blind-image fix. The process therefore wrote
correct motion/video/scores but statistics/report from the older in-memory code.

This script does NOT re-train, re-sample, re-generate Wan videos, or re-score.
It reuses the saved artifacts and only:
  1. validates completion counts and GT replay,
  2. recomputes statistics with the current source,
  3. rewrites the report markdown and completion.json,
  4. preserves the original full-pipeline elapsed_sec (not finalizer wall time).

Run:
  python experiments/learned_basis_hybrid_cnf/scripts/finalize_learned_basis_hybrid_cnf.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from learned_basis_hybrid_cnf.evaluate import (  # noqa: E402
    write_markdown_report_and_completion,
)
from learned_basis_hybrid_cnf.statistics import (  # noqa: E402
    run_all_statistics,
)
from learned_basis_hybrid_cnf.util_eval import (  # noqa: E402
    DOC_REPORT,
    GROUPS,
    OUT,
    append_eval_log,
    file_ok,
    read_json,
    update_status_eval,
    utcnow,
    write_json,
)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"Finalization precheck failed: {msg}")


def _primary_mean(stats: dict[str, Any], group: str) -> float | None:
    return stats["group_summaries"][group]["metrics_scene_mean"].get("anchor_epe_ref_penalized")


def main() -> int:
    append_eval_log("Finalizer start: CPU-only statistics/report regeneration (no GPU, no rescoring).")

    # ---- Load only already-saved artifacts ----
    motion_manifest = read_json(OUT / "manifests" / "motion_manifest.json")
    video_manifest = read_json(OUT / "manifests" / "video_manifest.json")
    motion_summary = read_json(OUT / "motion" / "motion_summary.json")
    video_summary = read_json(OUT / "video" / "video_counts.json")
    replay_summary = read_json(OUT / "video" / "replay" / "replay_summary.json")
    evaluator_meta = read_json(OUT / "score" / "evaluator_metadata.json")
    per_video_scores = read_json(OUT / "score" / "per_video_scores.json")
    old_completion_path = OUT / "completion.json"
    old_completion = read_json(old_completion_path) if file_ok(old_completion_path) else {}

    # ---- Validate the frozen formal-run evidence ----
    _require(len(motion_manifest) == 128, f"motion manifest len={len(motion_manifest)} != 128")
    _require(motion_summary["motion_ok_total"] == 128, "motion_ok_total != 128")
    _require(motion_summary["failed_total"] == 0, "motion failed_total != 0")
    for g in GROUPS:
        gs = motion_summary["by_group"][g]
        _require(gs["planned"] == 32 and gs["motion_ok_count"] == 32 and gs["failed_count"] == 0, f"{g} motion != 32/32 ok")
    _require(motion_summary["noise_pairing_all_ok"] is True, "initial noise pairing not all ok")

    _require(len(video_manifest) == 256, f"video manifest len={len(video_manifest)} != 256")
    counts = video_summary["counts"]
    _require(counts["planned"] == 256, f"planned={counts['planned']} != 256")
    _require(counts["generated"] == 256, f"generated={counts['generated']} != 256")
    _require(counts["failed"] == 0, f"failed={counts['failed']} != 0")
    _require(counts["pending"] == 0, f"pending={counts['pending']} != 0")

    vgroups = Counter(r["group"] for r in per_video_scores)
    _require(len(per_video_scores) == 256, f"per_video_scores len={len(per_video_scores)} != 256")
    for g in GROUPS:
        _require(vgroups[g] == 64, f"{g} scored videos={vgroups[g]} != 64")
    _require(all(r.get("score_status") == "ok" for r in per_video_scores), "non-ok score_status present")

    missing_videos = []
    for it in video_manifest:
        p = Path(it.get("video_path") or "")
        if not p.is_absolute():
            p = OUT / p
        if not file_ok(p, min_bytes=1024):
            missing_videos.append(str(p))
    _require(not missing_videos, f"missing generated video files: {missing_videos[:5]} (n={len(missing_videos)})")

    _require(replay_summary["replay_tolerance_pass"] is True, "GT replay tolerance failed")
    _require(replay_summary["strict_bitexact_ok"] is True, "GT replay not bit-exact")
    _require(float(replay_summary["max_of_max_abs"]) == 0.0, "GT replay max_abs != 0")
    _require(float(replay_summary["max_of_mae"]) == 0.0, "GT replay MAE != 0")

    # ---- Preserve the real pipeline elapsed time ----
    elapsed_sec = old_completion.get("elapsed_sec")
    _require(isinstance(elapsed_sec, (int, float)) and float(elapsed_sec) > 0, "missing original completion.elapsed_sec")
    elapsed_sec = float(elapsed_sec)
    original_completed_at = (
        (old_completion.get("finalization") or {}).get("original_completed_at")
        or old_completion.get("completed_at")
    )
    old_primary = None
    if isinstance(old_completion.get("primary_outcome"), dict):
        old_primary = {k: old_completion["primary_outcome"].get(k) for k in ("mean_diff", "ci95", "n_scenes")}

    # ---- CPU-only statistics recomputation with current source ----
    stats_res = run_all_statistics(motion_manifest, per_video_scores)
    new_primary = stats_res["primary_outcome"]
    primary_unchanged = True
    if old_primary and old_primary.get("mean_diff") is not None:
        primary_unchanged = (
            abs(float(old_primary["mean_diff"]) - float(new_primary["mean_diff"])) <= 1e-12
            and list(old_primary.get("ci95") or [0, 0]) == list(new_primary.get("ci95") or [0, 0])
        )

    motion_res = {"records": motion_manifest, "summary": motion_summary, "video_manifest": video_manifest}
    video_res = {"manifest": video_manifest, "summary": video_summary}
    score_res = {"per_video_scores": per_video_scores, "evaluator_meta": evaluator_meta}

    completion = write_markdown_report_and_completion(
        motion_res, video_res, score_res, stats_res, elapsed_sec
    )

    # ---- Attach finalization provenance without changing measured metrics ----
    completion = read_json(old_completion_path)
    completion["finalized_at"] = utcnow()
    completion["finalization"] = {
        "script": str(Path(__file__).resolve()),
        "reason": (
            "Formal run PID 799365 loaded modules before current statistics.py/scoring.py fixes; "
            "motion/video/scores were complete and valid. Statistics were recomputed on CPU from saved "
            "artifacts with current source; no resampling, regeneration, rescoring, or threshold changes."
        ),
        "statistics_recomputed_from_saved_artifacts": True,
        "rescoring_performed": False,
        "gpu_used": False,
        "pipeline_elapsed_sec_preserved": elapsed_sec,
        "original_completed_at": original_completed_at,
        "primary_outcome_unchanged_vs_process_written_completion": primary_unchanged,
        "previous_primary_outcome": old_primary,
        "artifacts_used": {
            "motion_summary": str(OUT / "motion" / "motion_summary.json"),
            "motion_manifest": str(OUT / "manifests" / "motion_manifest.json"),
            "video_manifest": str(OUT / "manifests" / "video_manifest.json"),
            "per_video_scores": str(OUT / "score" / "per_video_scores.json"),
            "replay_summary": str(OUT / "video" / "replay" / "replay_summary.json"),
            "evaluator_metadata": str(OUT / "score" / "evaluator_metadata.json"),
        },
        "statistics_bundle": str(OUT / "stats" / "full_statistics_bundle.json"),
        "report": str(DOC_REPORT),
    }
    write_json(old_completion_path, completion)
    update_status_eval(
        "completed",
        finalized=True,
        finalized_at=completion["finalized_at"],
        completion_file=str(old_completion_path),
        report_file=str(DOC_REPORT),
        statistics_recomputed_from_saved_artifacts=True,
        rescoring_performed=False,
        gpu_used=False,
    )

    per_group = {g: _primary_mean(stats_res, g) for g in GROUPS}
    finalize_summary = {
        "time": utcnow(),
        "ok": True,
        "checks": {
            "motion_ok": motion_summary["motion_ok_total"],
            "motion_failed": motion_summary["failed_total"],
            "videos_generated": counts["generated"],
            "videos_failed": counts["failed"],
            "videos_pending": counts["pending"],
            "videos_reused": counts["reused"],
            "scores_scored": len(per_video_scores),
            "gt_replay_max_abs": replay_summary["max_of_max_abs"],
            "gt_replay_max_mae": replay_summary["max_of_mae"],
            "gt_replay_strict_bitexact_ok": replay_summary["strict_bitexact_ok"],
            "per_group_videos": dict(vgroups),
        },
        "per_group_primary_anchor_epe_ref_penalized_px": per_group,
        "primary_outcome": new_primary,
        "auxiliary_outcomes": stats_res["preset_auxiliary_outcomes"],
        "conditional_diversity_valid_pairs": {
            g: {
                "q_pairs_both_ok": stats_res["variation"]["by_group"][g]["motion_seed_variation"].get("n_scene_pairs_q_both_ok"),
                "q_pairs_total": stats_res["variation"]["by_group"][g]["motion_seed_variation"].get("n_scene_pairs_q"),
                "video_motionseed_pairs_both_le10px": stats_res["variation"]["by_group"][g]["motion_seed_variation"].get("n_video_pairs_both_anchor_le_10px_ref"),
                "video_motionseed_pairs_total": stats_res["variation"]["by_group"][g]["motion_seed_variation"].get("n_video_pairs"),
                "wanseed_pairs_both_le10px": stats_res["variation"]["by_group"][g]["wan_seed_variation"].get("n_control_pairs_both_wan_anchor_le_10px_ref"),
                "wanseed_pairs_total": stats_res["variation"]["by_group"][g]["wan_seed_variation"].get("n_control_pairs"),
            }
            for g in GROUPS
        },
        "pipeline_elapsed_sec": elapsed_sec,
        "primary_outcome_unchanged_vs_process_written_completion": primary_unchanged,
    }
    write_json(OUT / "finalize_summary.json", finalize_summary)

    print(json.dumps(finalize_summary, indent=2, ensure_ascii=False))
    append_eval_log(
        f"Finalizer done: 128/128 motion, 256/256 videos, 256 scores, replay max_abs=0 MAE=0, "
        f"primary unchanged={primary_unchanged}, elapsed_sec={elapsed_sec:.1f}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
