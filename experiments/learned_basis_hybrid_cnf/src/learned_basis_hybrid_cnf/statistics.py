from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .util_eval import (
    ANCHOR_FRAMES,
    BOOT_N,
    BOOT_SEED,
    FRAMES,
    GROUPS,
    NONANCHOR_FRAMES,
    OUT,
    WAN_SEEDS,
    ensure_eval_dirs,
    file_ok,
    motion_seeds_for_scene,
    read_json,
    update_status_eval,
    utcnow,
    write_json,
)

VIDEO_METRICS_SPEC = [
    # (metric_key, higher_is_better, default_on_fail_if_total_denominator)
    ("anchor_epe_ref_penalized", False),
    ("anchor_epe_ref_valid", False),
    ("anchor_coverage", True),
    ("anchor_miss_rate", False),
    ("anchor_all_le_5px_ref", True),
    ("anchor_all_le_10px_ref", True),
    ("frame13_epe_ref_penalized", False),
    ("frame13_epe_ref_valid", False),
    ("frame16_epe_ref_penalized", False),
    ("frame16_epe_ref_valid", False),
    ("hold_drift_13_16_penalized_px", False),
    ("hold_drift_13_16_max_px", False),
    ("hold_stable_le_3px", True),
    ("nonanchor_epe_ref_penalized", False),
    ("nonanchor_epe_ref_valid", False),
    ("all_frames_epe_ref_penalized", False),
    ("all_frames_epe_ref_valid", False),
    ("all_frames_coverage", True),
    ("all_frames_miss_rate", False),
    ("anchor_epe_self_penalized", False),
    ("anchor_epe_self_valid", False),
    ("anchor_all_le_5px_self", True),
    ("anchor_all_le_10px_self", True),
    ("nonanchor_epe_self_penalized", False),
    ("nonanchor_epe_self_valid", False),
    ("audit_hsv_anchor_epe_ref_penalized", False),
    ("audit_hsv_anchor_epe_ref_valid", False),
    ("audit_hsv_anchor_coverage", True),
    ("audit_hsv_anchor_all_le_5px_ref", True),
    ("audit_hsv_anchor_all_le_10px_ref", True),
    ("audit_hsv_anchor_epe_self_penalized", False),
    ("audit_hsv_anchor_epe_self_valid", False),
    ("strict_hsv_anchor_epe_ref_penalized", False),
    ("strict_hsv_anchor_epe_ref_valid", False),
    ("strict_hsv_anchor_coverage", True),
    ("strict_hsv_anchor_all_le_5px_ref", True),
    ("strict_hsv_anchor_all_le_10px_ref", True),
    ("strict_hsv_anchor_epe_self_penalized", False),
    ("strict_hsv_anchor_epe_self_valid", False),
    ("background_flicker_l1", False),
    ("eval_components_gt1_rate", False),
]

CONTRASTS = [
    ("H-learned_minus_S-learned", "H-learned", "S-learned", "primary: CNF hard projection effect in learned basis"),
    ("H-fixed_minus_S-fixed", "H-fixed", "S-fixed", "auxiliary: CNF hard projection effect in fixed basis"),
    ("H-learned_minus_H-fixed", "H-learned", "H-fixed", "auxiliary: learned basis effect under hybrid CNF"),
    ("S-learned_minus_S-fixed", "S-learned", "S-fixed", "auxiliary: learned basis effect under soft sampling"),
]


def paired_scene_bootstrap(
    diffs: list[float | None],
    *,
    higher_is_better: bool = False,
    seed: int = BOOT_SEED,
    n_boot: int = BOOT_N,
) -> dict[str, Any]:
    valid_pairs = [(idx, float(d)) for idx, d in enumerate(diffs) if d is not None and np.isfinite(d)]
    if not valid_pairs:
        return {
            "n_scenes": 0,
            "mean_diff": None,
            "se": None,
            "ci95": [None, None],
            "improved_scenes": 0,
            "per_scene_diffs": diffs,
        }
    x = np.array([v for _, v in valid_pairs], dtype=np.float64)
    n = len(x)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[boot_idx].mean(axis=1)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    mean_diff = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    improved = int(np.sum(x > 0.0) if higher_is_better else np.sum(x < 0.0))
    unchanged = int(np.sum(np.abs(x) <= 1e-12))
    return {
        "n_scenes": n,
        "mean_diff": mean_diff,
        "se": se,
        "ci95": [ci_low, ci_high],
        "ci95_excludes_zero_in_beneficial_direction": bool(ci_low > 0.0 if higher_is_better else ci_high < 0.0),
        "improved_scenes": improved,
        "unchanged_scenes": unchanged,
        "worse_scenes": int(n - improved - unchanged),
        "per_scene_diffs": [float(d) if (d is not None and np.isfinite(d)) else None for d in diffs],
    }


def compute_scene_group_means(per_video_scores: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, float | None]]]:
    """For each group and scene (0..15), average across the 4 seeds within that scene."""
    out: dict[str, dict[int, dict[str, float | None]]] = {g: {} for g in GROUPS}
    for g in GROUPS:
        for s in range(16):
            rows = [r for r in per_video_scores if r["group"] == g and int(r["scene"]) == s]
            if len(rows) != 4:
                raise ValueError(f"Expected 4 videos for group={g} scene={s}, got {len(rows)}")
            s_dict: dict[str, float | None] = {}
            for mkey, _ in VIDEO_METRICS_SPEC:
                vals = []
                for r in rows:
                    v = r.get(mkey)
                    if isinstance(v, bool):
                        vals.append(1.0 if v else 0.0)
                    elif v is not None and np.isfinite(float(v)):
                        vals.append(float(v))
                s_dict[mkey] = float(np.mean(vals)) if vals else None
            out[g][s] = s_dict
    return out


def compute_group_overall_summaries(
    per_video_scores: list[dict[str, Any]],
    scene_means: dict[str, dict[int, dict[str, float | None]]],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for g in GROUPS:
        rows = [r for r in per_video_scores if r["group"] == g]
        n_total = len(rows)
        n_ok = sum(1 for r in rows if r.get("score_status") == "ok")
        g_sum: dict[str, Any] = {
            "planned_videos": n_total,
            "scored_ok_videos": n_ok,
            "failed_videos": n_total - n_ok,
            "metrics_scene_mean": {},
            "metrics_sample_mean": {},
        }
        for mkey, _ in VIDEO_METRICS_SPEC:
            s_vals = [scene_means[g][s][mkey] for s in range(16) if scene_means[g][s][mkey] is not None]
            g_sum["metrics_scene_mean"][mkey] = float(np.mean(s_vals)) if s_vals else None

            raw_vals = []
            for r in rows:
                v = r.get(mkey)
                if isinstance(v, bool):
                    raw_vals.append(1.0 if v else 0.0)
                elif v is not None and np.isfinite(float(v)):
                    raw_vals.append(float(v))
            g_sum["metrics_sample_mean"][mkey] = float(np.mean(raw_vals)) if raw_vals else None
        summaries[g] = g_sum
    return summaries


def compute_video_paired_contrasts(
    scene_means: dict[str, dict[int, dict[str, float | None]]],
) -> dict[str, Any]:
    all_contrasts: dict[str, Any] = {}
    for cname, g_target, g_base, desc in CONTRASTS:
        c_res: dict[str, Any] = {
            "contrast": cname,
            "target_group": g_target,
            "baseline_group": g_base,
            "description": desc,
            "metrics": {},
        }
        for mkey, higher_is_better in VIDEO_METRICS_SPEC:
            diffs = []
            for s in range(16):
                vt = scene_means[g_target][s].get(mkey)
                vb = scene_means[g_base][s].get(mkey)
                if vt is not None and vb is not None and np.isfinite(vt) and np.isfinite(vb):
                    diffs.append(float(vt - vb))
                else:
                    diffs.append(None)
            c_res["metrics"][mkey] = paired_scene_bootstrap(
                diffs, higher_is_better=higher_is_better, seed=BOOT_SEED, n_boot=BOOT_N
            )
        all_contrasts[cname] = c_res
    return all_contrasts


def compute_motion_paired_contrasts(motion_records: list[dict[str, Any]]) -> dict[str, Any]:
    motion_metrics_spec = [
        ("final_anchor_max_abs_rad", lambda r: r["final_certificate"]["anchor_max_abs_rad"], False),
        ("final_Hc_minus_b_max_abs", lambda r: r["final_certificate"]["Hc_minus_b_max_abs"], False),
        ("final_nonanchor_rmse_vs_v2_rad", lambda r: r["final_certificate"]["ref_errors"]["nonanchor_rmse_vs_v2_rad"], False),
        ("final_fk_nonanchor_rmse_vs_v2_m", lambda r: r["final_certificate"]["ref_errors"]["nonanchor_fk_rmse_vs_v2_m"], False),
        ("final_posthoc_nonanchor_rmse_vs_orig_rad", lambda r: r["final_certificate"]["ref_errors"]["posthoc_nonanchor_rmse_vs_orig_rad"], False),
        ("raw_anchor_max_abs_rad", lambda r: r["raw_certificate"]["anchor_max_abs_rad"], False),
        ("raw_nonanchor_rmse_vs_v2_rad", lambda r: r["raw_certificate"]["ref_errors"]["nonanchor_rmse_vs_v2_rad"], False),
        ("raw_fk_nonanchor_rmse_vs_v2_m", lambda r: r["raw_certificate"]["ref_errors"]["nonanchor_fk_rmse_vs_v2_m"], False),
        ("repair_coeff_l2", lambda r: r["final_certificate"]["repair_delta"]["coeff_l2"], False),
        ("repair_q_max_abs_rad", lambda r: r["final_certificate"]["repair_delta"]["q_max_abs_rad"], False),
        ("repair_fk_max_l2_m", lambda r: r["final_certificate"]["repair_delta"]["fk_max_l2_m"], False),
        ("final_joint_acc_d2_mse", lambda r: r["final_certificate"]["motion_stats"]["joint_acc_d2_mse_per_step2"], False),
        ("final_joint_jerk_d3_l2_mean", lambda r: r["final_certificate"]["motion_stats"]["joint_jerk_d3_l2_mean"], False),
        ("motion_ok", lambda r: 1.0 if r.get("motion_ok") else 0.0, True),
    ]

    scene_group: dict[str, dict[int, dict[str, float | None]]] = {g: {} for g in GROUPS}
    for g in GROUPS:
        for s in range(16):
            recs = [r for r in motion_records if r["group"] == g and int(r["scene"]) == s]
            s_dict: dict[str, float | None] = {}
            for mname, fn, _ in motion_metrics_spec:
                vals = []
                for r in recs:
                    try:
                        v = fn(r)
                        if v is not None and np.isfinite(float(v)):
                            vals.append(float(v))
                    except Exception:
                        pass
                s_dict[mname] = float(np.mean(vals)) if vals else None
            scene_group[g][s] = s_dict

    contrasts_out: dict[str, Any] = {}
    for cname, g_target, g_base, desc in CONTRASTS:
        c_res: dict[str, Any] = {
            "contrast": cname,
            "target_group": g_target,
            "baseline_group": g_base,
            "description": desc,
            "metrics": {},
        }
        for mname, _, higher_is_better in motion_metrics_spec:
            diffs = []
            for s in range(16):
                vt = scene_group[g_target][s].get(mname)
                vb = scene_group[g_base][s].get(mname)
                if vt is not None and vb is not None and np.isfinite(vt) and np.isfinite(vb):
                    diffs.append(float(vt - vb))
                else:
                    diffs.append(None)
            c_res["metrics"][mname] = paired_scene_bootstrap(
                diffs, higher_is_better=higher_is_better, seed=BOOT_SEED, n_boot=BOOT_N
            )
        contrasts_out[cname] = c_res

    # Also compare oracle reconstruction between learned and fixed basis
    oracle_path = OUT / "motion" / "basis_diagnostics_and_oracle.json"
    if file_ok(oracle_path):
        od = read_json(oracle_path)
        f_rows = {int(r["scene"]): r for r in od["fixed"]["oracle_reconstruction"]["per_scene"]}
        l_rows = {int(r["scene"]): r for r in od["learned"]["oracle_reconstruction"]["per_scene"]}
        oracle_contrasts = {}
        for okey in [
            "ls_nonanchor_rmse_rad",
            "proj_nonanchor_rmse_rad",
            "proj_fk_nonanchor_rmse_m",
            "repaired_nonanchor_rmse_rad",
            "repaired_fk_nonanchor_rmse_m",
        ]:
            diffs = [float(l_rows[s][okey] - f_rows[s][okey]) for s in range(16)]
            oracle_contrasts[okey] = paired_scene_bootstrap(diffs, higher_is_better=False, seed=BOOT_SEED, n_boot=BOOT_N)
        contrasts_out["oracle_learned_minus_fixed"] = oracle_contrasts

    return contrasts_out


def _load_video_eval_centers(video_id: str) -> list[np.ndarray | None] | None:
    p = OUT / "score" / "frames" / f"{video_id}.json"
    if not file_ok(p):
        return None
    d = read_json(p)
    frames = d.get("frames") or []
    if len(frames) != FRAMES:
        return None
    return [
        np.asarray(fr["eval_center"], dtype=np.float64) if fr.get("eval_center") is not None else None
        for fr in frames
    ]


def compute_diversity_and_variation(
    motion_records: list[dict[str, Any]],
    per_video_scores: list[dict[str, Any]],
) -> dict[str, Any]:
    """Separate motion-seed variation (different motion seeds, same scene/group/wan_seed)
    from Wan-seed variation (same control trajectory, different Wan seed 43 vs 44),
    always reporting the proportion where BOTH samples satisfy constraints.
    """
    m_by_key = {(r["group"], int(r["scene"]), int(r["motion_seed"])): r for r in motion_records}
    v_by_key = {(r["group"], int(r["scene"]), int(r["motion_seed"]), int(r["wan_seed"])): r for r in per_video_scores}

    motion_q_pairs = []
    motion_video_pairs = []
    wan_video_pairs = []

    for g in GROUPS:
        for s in range(16):
            ms0, ms1 = motion_seeds_for_scene(s)
            m0 = m_by_key.get((g, s, ms0))
            m1 = m_by_key.get((g, s, ms1))
            q_nonanchor_rmse = None
            ee_nonanchor_diff = None
            both_m_ok = bool(m0 and m1 and m0.get("motion_ok") and m1.get("motion_ok"))
            if m0 and m1 and file_ok(m0["final_npz_path"]) and file_ok(m1["final_npz_path"]):
                f0 = np.load(m0["final_npz_path"])
                f1 = np.load(m1["final_npz_path"])
                q0, q1 = f0["q_final"], f1["q_final"]
                ee0, ee1 = f0["ee_final"], f1["ee_final"]
                q_nonanchor_rmse = float(np.sqrt(np.mean((q0[NONANCHOR_FRAMES] - q1[NONANCHOR_FRAMES]) ** 2)))
                ee_nonanchor_diff = float(
                    np.mean(np.linalg.norm(ee0[NONANCHOR_FRAMES] - ee1[NONANCHOR_FRAMES], axis=-1))
                )
            motion_q_pairs.append(
                {
                    "group": g,
                    "scene": s,
                    "motion_seeds": [ms0, ms1],
                    "nonanchor_q_rmse_rad": q_nonanchor_rmse,
                    "nonanchor_ee_path_diff_m": ee_nonanchor_diff,
                    "both_motion_constraints_ok": both_m_ok,
                }
            )

            for ws in WAN_SEEDS:
                v0 = v_by_key.get((g, s, ms0, ws))
                v1 = v_by_key.get((g, s, ms1, ws))
                if not (v0 and v1):
                    continue
                c0 = _load_video_eval_centers(v0["video_id"])
                c1 = _load_video_eval_centers(v1["video_id"])
                na_diff = None
                all_diff = None
                if c0 and c1:
                    na_ds = [
                        float(np.linalg.norm(c0[f] - c1[f]))
                        for f in NONANCHOR_FRAMES
                        if c0[f] is not None and c1[f] is not None
                    ]
                    all_ds = [
                        float(np.linalg.norm(c0[f] - c1[f]))
                        for f in range(FRAMES)
                        if c0[f] is not None and c1[f] is not None
                    ]
                    na_diff = float(np.mean(na_ds)) if na_ds else None
                    all_diff = float(np.mean(all_ds)) if all_ds else None
                motion_video_pairs.append(
                    {
                        "group": g,
                        "scene": s,
                        "wan_seed": ws,
                        "motion_seeds": [ms0, ms1],
                        "nonanchor_cube_path_diff_px": na_diff,
                        "all_frames_cube_path_diff_px": all_diff,
                        "both_motion_constraints_ok": both_m_ok,
                        "both_videos_anchor_le_5px_ref": bool(
                            both_m_ok and v0.get("anchor_all_le_5px_ref") and v1.get("anchor_all_le_5px_ref")
                        ),
                        "both_videos_anchor_le_10px_ref": bool(
                            both_m_ok and v0.get("anchor_all_le_10px_ref") and v1.get("anchor_all_le_10px_ref")
                        ),
                    }
                )

            for ms in [ms0, ms1]:
                m_single = m_by_key.get((g, s, ms))
                m_single_ok = bool(m_single and m_single.get("motion_ok"))
                w0 = v_by_key.get((g, s, ms, 43))
                w1 = v_by_key.get((g, s, ms, 44))
                if not (w0 and w1):
                    continue
                c0 = _load_video_eval_centers(w0["video_id"])
                c1 = _load_video_eval_centers(w1["video_id"])
                anc_diff = None
                na_diff = None
                all_diff = None
                if c0 and c1:
                    anc_ds = [
                        float(np.linalg.norm(c0[f] - c1[f]))
                        for f in ANCHOR_FRAMES
                        if c0[f] is not None and c1[f] is not None
                    ]
                    na_ds = [
                        float(np.linalg.norm(c0[f] - c1[f]))
                        for f in NONANCHOR_FRAMES
                        if c0[f] is not None and c1[f] is not None
                    ]
                    all_ds = [
                        float(np.linalg.norm(c0[f] - c1[f]))
                        for f in range(FRAMES)
                        if c0[f] is not None and c1[f] is not None
                    ]
                    anc_diff = float(np.mean(anc_ds)) if anc_ds else None
                    na_diff = float(np.mean(na_ds)) if na_ds else None
                    all_diff = float(np.mean(all_ds)) if all_ds else None
                wan_video_pairs.append(
                    {
                        "group": g,
                        "scene": s,
                        "motion_seed": ms,
                        "wan_seeds": [43, 44],
                        "motion_constraints_ok": m_single_ok,
                        "anchor_cube_path_diff_px": anc_diff,
                        "nonanchor_cube_path_diff_px": na_diff,
                        "all_frames_cube_path_diff_px": all_diff,
                        "both_wan_anchor_le_5px_ref": bool(
                            m_single_ok and w0.get("anchor_all_le_5px_ref") and w1.get("anchor_all_le_5px_ref")
                        ),
                        "both_wan_anchor_le_10px_ref": bool(
                            m_single_ok and w0.get("anchor_all_le_10px_ref") and w1.get("anchor_all_le_10px_ref")
                        ),
                    }
                )

    summary_by_group = {}
    for g in GROUPS:
        mq = [r for r in motion_q_pairs if r["group"] == g]
        mv = [r for r in motion_video_pairs if r["group"] == g]
        wv = [r for r in wan_video_pairs if r["group"] == g]

        mq_valid = [r for r in mq if r.get("both_motion_constraints_ok")]
        mv_m_ok = [r for r in mv if r.get("both_motion_constraints_ok")]
        mv_le10 = [r for r in mv if r.get("both_videos_anchor_le_10px_ref")]
        mv_le5 = [r for r in mv if r.get("both_videos_anchor_le_5px_ref")]
        wv_m_ok = [r for r in wv if r.get("motion_constraints_ok")]
        wv_le10 = [r for r in wv if r.get("both_wan_anchor_le_10px_ref")]
        wv_le5 = [r for r in wv if r.get("both_wan_anchor_le_5px_ref")]

        def _mean_valid(rows: list[dict[str, Any]], key: str) -> float | None:
            vs = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))]
            return float(np.mean(vs)) if vs else None

        summary_by_group[g] = {
            "motion_seed_variation": {
                "n_scene_pairs_q": len(mq),
                "n_scene_pairs_q_both_ok": len(mq_valid),
                "nonanchor_q_rmse_rad_mean": _mean_valid(mq, "nonanchor_q_rmse_rad"),
                "nonanchor_q_rmse_rad_cond_both_motion_ok": _mean_valid(mq_valid, "nonanchor_q_rmse_rad"),
                "nonanchor_ee_path_diff_m_mean": _mean_valid(mq, "nonanchor_ee_path_diff_m"),
                "nonanchor_ee_path_diff_m_cond_both_motion_ok": _mean_valid(mq_valid, "nonanchor_ee_path_diff_m"),
                "both_motion_constraints_ok_rate": float(np.mean([1.0 if r["both_motion_constraints_ok"] else 0.0 for r in mq])),
                "n_video_pairs": len(mv),
                "n_video_pairs_both_motion_ok": len(mv_m_ok),
                "n_video_pairs_both_anchor_le_10px_ref": len(mv_le10),
                "n_video_pairs_both_anchor_le_5px_ref": len(mv_le5),
                "nonanchor_cube_path_diff_px_mean": _mean_valid(mv, "nonanchor_cube_path_diff_px"),
                "nonanchor_cube_path_diff_px_cond_both_motion_ok": _mean_valid(mv_m_ok, "nonanchor_cube_path_diff_px"),
                "nonanchor_cube_path_diff_px_cond_both_anchor_le_10px_ref": _mean_valid(mv_le10, "nonanchor_cube_path_diff_px"),
                "nonanchor_cube_path_diff_px_cond_both_anchor_le_5px_ref": _mean_valid(mv_le5, "nonanchor_cube_path_diff_px"),
                "all_frames_cube_path_diff_px_mean": _mean_valid(mv, "all_frames_cube_path_diff_px"),
                "all_frames_cube_path_diff_px_cond_both_anchor_le_10px_ref": _mean_valid(mv_le10, "all_frames_cube_path_diff_px"),
                "both_videos_anchor_le_5px_ref_rate": float(
                    np.mean([1.0 if r["both_videos_anchor_le_5px_ref"] else 0.0 for r in mv])
                ),
                "both_videos_anchor_le_10px_ref_rate": float(
                    np.mean([1.0 if r["both_videos_anchor_le_10px_ref"] else 0.0 for r in mv])
                ),
            },
            "wan_seed_variation": {
                "n_control_pairs": len(wv),
                "n_control_pairs_motion_ok": len(wv_m_ok),
                "n_control_pairs_both_wan_anchor_le_10px_ref": len(wv_le10),
                "n_control_pairs_both_wan_anchor_le_5px_ref": len(wv_le5),
                "anchor_cube_path_diff_px_mean": _mean_valid(wv, "anchor_cube_path_diff_px"),
                "anchor_cube_path_diff_px_cond_both_anchor_le_10px_ref": _mean_valid(wv_le10, "anchor_cube_path_diff_px"),
                "nonanchor_cube_path_diff_px_mean": _mean_valid(wv, "nonanchor_cube_path_diff_px"),
                "nonanchor_cube_path_diff_px_cond_both_anchor_le_10px_ref": _mean_valid(wv_le10, "nonanchor_cube_path_diff_px"),
                "all_frames_cube_path_diff_px_mean": _mean_valid(wv, "all_frames_cube_path_diff_px"),
                "all_frames_cube_path_diff_px_cond_both_anchor_le_10px_ref": _mean_valid(wv_le10, "all_frames_cube_path_diff_px"),
                "both_wan_anchor_le_5px_ref_rate": float(
                    np.mean([1.0 if r["both_wan_anchor_le_5px_ref"] else 0.0 for r in wv])
                ),
                "both_wan_anchor_le_10px_ref_rate": float(
                    np.mean([1.0 if r["both_wan_anchor_le_10px_ref"] else 0.0 for r in wv])
                ),
            },
        }

    return {
        "note": "Two motion seeds and two Wan seeds per scene measure paired sample variation alongside dual-constraint satisfaction; they do not constitute proof of full distribution coverage.",
        "by_group": summary_by_group,
        "motion_q_pairs": motion_q_pairs,
        "motion_video_pairs": motion_video_pairs,
        "wan_video_pairs": wan_video_pairs,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})


def run_all_statistics(
    motion_records: list[dict[str, Any]] | None = None,
    per_video_scores: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ensure_eval_dirs()
    if motion_records is None:
        motion_records = read_json(OUT / "manifests" / "motion_manifest.json")
    if per_video_scores is None:
        per_video_scores = read_json(OUT / "score" / "per_video_scores.json")

    update_status_eval("computing_statistics")
    scene_means = compute_scene_group_means(per_video_scores)
    group_summaries = compute_group_overall_summaries(per_video_scores, scene_means)
    video_contrasts = compute_video_paired_contrasts(scene_means)
    motion_contrasts = compute_motion_paired_contrasts(motion_records)
    variation = compute_diversity_and_variation(motion_records, per_video_scores)

    # Export CSVs
    video_csv_fields = [
        "index",
        "video_id",
        "motion_id",
        "group",
        "basis_kind",
        "mode",
        "scene",
        "motion_index",
        "motion_seed",
        "wan_seed",
        "status",
        "score_status",
        "anchor_epe_ref_penalized",
        "anchor_epe_ref_valid",
        "anchor_coverage",
        "anchor_miss_rate",
        "anchor_all_le_5px_ref",
        "anchor_all_le_10px_ref",
        "frame13_epe_ref_penalized",
        "frame13_epe_ref_valid",
        "frame16_epe_ref_penalized",
        "frame16_epe_ref_valid",
        "hold_drift_13_16_max_px",
        "hold_stable_le_3px",
        "nonanchor_epe_ref_penalized",
        "nonanchor_epe_ref_valid",
        "anchor_epe_self_penalized",
        "anchor_epe_self_valid",
        "anchor_all_le_5px_self",
        "anchor_all_le_10px_self",
        "audit_hsv_anchor_epe_ref_penalized",
        "audit_hsv_anchor_epe_ref_valid",
        "strict_hsv_anchor_epe_ref_penalized",
        "strict_hsv_anchor_epe_ref_valid",
        "background_flicker_l1",
        "eval_components_gt1_rate",
    ]
    _write_csv(OUT / "stats" / "per_video_metrics.csv", per_video_scores, video_csv_fields)

    scene_csv_rows = []
    for g in GROUPS:
        for s in range(16):
            scene_csv_rows.append({"group": g, "scene": s, **scene_means[g][s]})
    scene_csv_fields = ["group", "scene"] + [m for m, _ in VIDEO_METRICS_SPEC]
    _write_csv(OUT / "stats" / "per_scene_group_means.csv", scene_csv_rows, scene_csv_fields)

    contrast_csv_rows = []
    for cname, cobj in video_contrasts.items():
        for mkey, _ in VIDEO_METRICS_SPEC:
            st = cobj["metrics"][mkey]
            contrast_csv_rows.append(
                {
                    "contrast": cname,
                    "metric": mkey,
                    "n_scenes": st["n_scenes"],
                    "mean_diff": st["mean_diff"],
                    "se": st["se"],
                    "ci95_low": st["ci95"][0],
                    "ci95_high": st["ci95"][1],
                    "improved_scenes": st["improved_scenes"],
                    "worse_scenes": st.get("worse_scenes"),
                    "ci95_excludes_zero_beneficial": st.get("ci95_excludes_zero_in_beneficial_direction"),
                }
            )
    _write_csv(
        OUT / "stats" / "paired_contrasts_summary.csv",
        contrast_csv_rows,
        [
            "contrast",
            "metric",
            "n_scenes",
            "mean_diff",
            "se",
            "ci95_low",
            "ci95_high",
            "improved_scenes",
            "worse_scenes",
            "ci95_excludes_zero_beneficial",
        ],
    )

    primary_contrast = video_contrasts["H-learned_minus_S-learned"]["metrics"]["anchor_epe_ref_penalized"]
    aux_hf_sf = video_contrasts["H-fixed_minus_S-fixed"]["metrics"]["anchor_epe_ref_penalized"]
    aux_hl_hf = video_contrasts["H-learned_minus_H-fixed"]["metrics"]["anchor_epe_ref_penalized"]

    bundle = {
        "time": utcnow(),
        "bootstrap_config": {"n_resamples": BOOT_N, "seed": BOOT_SEED, "unit": "scene (16 scenes)"},
        "primary_outcome": {
            "contrast": "H-learned - S-learned",
            "metric": "anchor_epe_ref_penalized",
            **primary_contrast,
            "ci95_upper_lt_0": bool(
                primary_contrast["ci95"][1] is not None and float(primary_contrast["ci95"][1]) < 0.0
            ),
        },
        "preset_auxiliary_outcomes": {
            "H-fixed_minus_S-fixed_anchor_epe_ref_penalized": aux_hf_sf,
            "H-learned_minus_H-fixed_anchor_epe_ref_penalized": aux_hl_hf,
        },
        "group_summaries": group_summaries,
        "scene_group_means": {g: {str(s): scene_means[g][s] for s in range(16)} for g in GROUPS},
        "video_paired_contrasts": video_contrasts,
        "motion_paired_contrasts": motion_contrasts,
        "variation": variation,
    }
    write_json(OUT / "stats" / "group_summaries.json", group_summaries)
    write_json(OUT / "stats" / "scene_paired_contrasts.json", video_contrasts)
    write_json(OUT / "stats" / "motion_paired_contrasts.json", motion_contrasts)
    write_json(OUT / "stats" / "variation_stats.json", variation)
    write_json(OUT / "stats" / "full_statistics_bundle.json", bundle)
    update_status_eval("statistics_done")
    return bundle
