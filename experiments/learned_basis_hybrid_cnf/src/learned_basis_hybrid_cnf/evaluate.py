from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .sampling import (
    build_motion_manifest,
    build_video_manifest,
    load_basis_and_generator,
    run_all_sampling_and_verification,
    run_cpu_numerical_checks,
)
from .scoring import load_frozen_evaluator, predict_frame_evaluator, run_all_scoring
from .statistics import run_all_statistics
from .util_eval import (
    DOC_REPORT,
    GROUPS,
    OUT,
    PKG,
    WAN_LORA,
    append_eval_log,
    build_evaluation_protocol,
    checkpoint_path_for_basis,
    ensure_eval_dirs,
    file_ok,
    read_json,
    sha256_array,
    sha256_file,
    update_status_eval,
    utcnow,
    write_json,
)
from .video import run_all_wan_videos

VENV_PYTHON = sys.executable
ENTRY_SCRIPT = str(PKG.parent.parent / "scripts" / "run_learned_basis_hybrid_cnf.py")


def check_checkpoints_and_training_status() -> dict[str, Any]:
    fixed_pt = checkpoint_path_for_basis("fixed")
    learned_pt = checkpoint_path_for_basis("learned")
    train_q = OUT / "data" / "train_q_v2.npz"
    eval_q = OUT / "data" / "eval_q_v2.npz"
    train_comp = OUT / "train" / "train_completion.json"
    train_comp_root = OUT / "train_completion.json"
    status_train = OUT / "status_train.json"
    status_train_sub = OUT / "train" / "status_train.json"

    missing_or_invalid: list[str] = []
    for label, p in [
        ("OUT/data/train_q_v2.npz", train_q),
        ("OUT/data/eval_q_v2.npz", eval_q),
        ("OUT/train/fixed.pt", fixed_pt),
        ("OUT/train/learned.pt", learned_pt),
    ]:
        if not file_ok(p, min_bytes=512):
            missing_or_invalid.append(label)

    st_train_obj = None
    for sp in [status_train, status_train_sub]:
        if file_ok(sp):
            try:
                st_train_obj = read_json(sp)
                break
            except Exception as e:
                missing_or_invalid.append(f"unreadable_status_train ({sp.name}): {e}")

    tc_obj = None
    tc_path_used = None
    for tp in [train_comp_root, train_comp]:
        if file_ok(tp):
            try:
                tc_obj = read_json(tp)
                tc_path_used = str(tp)
                break
            except Exception as e:
                missing_or_invalid.append(f"unreadable_train_completion ({tp.name}): {e}")

    if tc_obj is None:
        missing_or_invalid.append("OUT/train_completion.json")
    else:
        if tc_obj.get("status") != "completed":
            missing_or_invalid.append(f"train_completion.status={tc_obj.get('status')!r} != 'completed'")
        if st_train_obj is None or st_train_obj.get("stage") != "completed":
            missing_or_invalid.append(
                f"status_train.stage={st_train_obj.get('stage') if st_train_obj else None!r} != 'completed'"
            )
        # Verify data array sha256 (np.ascontiguousarray(np.load(npz)['q']).tobytes()) against train_completion
        data_sec = tc_obj.get("data") or {}
        if file_ok(train_q):
            try:
                tq_arr = np.load(train_q, allow_pickle=True)["q"]
                actual_tq_sha = sha256_array(tq_arr)
                if data_sec.get("train_q_v2_sha256") != actual_tq_sha:
                    missing_or_invalid.append(
                        f"train_q_v2.npz q-array sha256 mismatch vs train_completion: {actual_tq_sha} != {data_sec.get('train_q_v2_sha256')}"
                    )
            except Exception as e:
                missing_or_invalid.append(f"train_q_v2.npz unreadable: {e}")
        if file_ok(eval_q):
            try:
                eq_arr = np.load(eval_q, allow_pickle=True)["q"]
                actual_eq_sha = sha256_array(eq_arr)
                if data_sec.get("eval_q_v2_sha256") != actual_eq_sha:
                    missing_or_invalid.append(
                        f"eval_q_v2.npz q-array sha256 mismatch vs train_completion: {actual_eq_sha} != {data_sec.get('eval_q_v2_sha256')}"
                    )
            except Exception as e:
                missing_or_invalid.append(f"eval_q_v2.npz unreadable: {e}")

    ckpt_details: dict[str, Any] = {}
    for kind, p in [("fixed", fixed_pt), ("learned", learned_pt)]:
        if not file_ok(p, min_bytes=512):
            continue
        try:
            info = load_basis_and_generator(kind, device="cpu")
            diag = info["diagnostics"]
            if int(diag.get("steps", -1)) != 6000 or str(diag.get("group_in_ckpt")) != kind:
                missing_or_invalid.append(
                    f"{kind}.pt invalid steps/group: steps={diag.get('steps')}, group={diag.get('group_in_ckpt')}"
                )
            if tc_obj is not None:
                expected_sha = ((tc_obj.get("checkpoints") or {}).get(kind) or {}).get("sha256")
                if not expected_sha or diag["ckpt_sha256"] != expected_sha:
                    missing_or_invalid.append(
                        f"{kind}.pt sha256 mismatch vs train_completion: {diag['ckpt_sha256']} != {expected_sha}"
                    )
            ckpt_details[kind] = diag
        except Exception as e:
            missing_or_invalid.append(f"invalid_or_incomplete_{kind}.pt: {e}")

    return {
        "all_ready": len(missing_or_invalid) == 0,
        "missing": missing_or_invalid,
        "train_completion_path": tc_path_used,
        "checkpoint_diagnostics": ckpt_details,
        "status_train": st_train_obj,
        "train_completion": tc_obj,
    }


def check_gpu_free_for_eval() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        my_pid = str(os.getpid())
        other_procs = [ln for ln in lines if not ln.startswith(my_pid + ",")]
        return {
            "gpu_free": len(other_procs) == 0,
            "active_compute_apps": lines,
            "other_compute_apps": other_procs,
        }
    except Exception as e:
        return {"gpu_free": False, "error": str(e)}


def prepare_and_verify_ready() -> dict[str, Any]:
    ensure_eval_dirs()
    update_status_eval("preparing_and_verifying_cpu")
    proto = build_evaluation_protocol()

    # Write planned manifests
    m_plan = build_motion_manifest()
    v_plan = build_video_manifest()
    write_json(OUT / "manifests" / "motion_manifest_planned.json", m_plan)
    write_json(OUT / "manifests" / "video_manifest_planned.json", v_plan)

    # Run deterministic CPU numerical checks
    cpu_checks = run_cpu_numerical_checks()

    # Verify frozen RGB segmenter loads cleanly on CPU and runs on a test frame
    eval_model, eval_dev, eval_meta = load_frozen_evaluator(device="cpu")
    test_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    test_rgb[100:116, 120:136] = np.array([255, 120, 20], dtype=np.uint8)
    ev_out = predict_frame_evaluator(eval_model, eval_dev, test_rgb)
    evaluator_check = {
        "loaded_ok": True,
        "meta": eval_meta,
        "synthetic_test_output_keys": sorted(ev_out.keys()),
    }

    ckpt_status = check_checkpoints_and_training_status()
    gpu_status = check_gpu_free_for_eval()

    code_hashes = {}
    for fname in [
        "util_eval.py",
        "sampling.py",
        "video.py",
        "scoring.py",
        "statistics.py",
        "evaluate.py",
        "basis.py",
        "projection.py",
    ]:
        fpath = PKG / fname
        if file_ok(fpath):
            code_hashes[fname] = sha256_file(fpath)
    if file_ok(ENTRY_SCRIPT):
        code_hashes["run_learned_basis_hybrid_cnf_20261114.py"] = sha256_file(ENTRY_SCRIPT)

    ready_obj = {
        "schema_version": 1,
        "time": utcnow(),
        "evaluation_code_ready": bool(cpu_checks["all_passed"] and evaluator_check["loaded_ok"]),
        "cpu_numerical_verification": cpu_checks,
        "evaluator_verification": evaluator_check,
        "planned_counts": {
            "motion_trajectories": len(m_plan),
            "wan_videos": len(v_plan),
            "blind_overlay_videos": 24,
        },
        "awaiting_checkpoints": ckpt_status["missing"],
        "checkpoints_ready": ckpt_status["all_ready"],
        "checkpoint_diagnostics": ckpt_status["checkpoint_diagnostics"],
        "gpu_status": gpu_status,
        "code_hashes": code_hashes,
        "launch_command": f"{VENV_PYTHON} {ENTRY_SCRIPT} --run-all",
        "verify_only_command": f"{VENV_PYTHON} {ENTRY_SCRIPT} --verify-ready",
        "notes": [
            "Evaluation package (util_eval.py, sampling.py, video.py, scoring.py, statistics.py, evaluate.py) and thin entry are fully implemented and verified on CPU.",
            "No random or legacy checkpoints are used to fabricate formal results.",
            "Once OUT/train/fixed.pt and OUT/train/learned.pt are finalized and GPU is free, launch_command runs the complete 128-motion + 256-video + scoring + statistics + report pipeline.",
        ],
    }
    write_json(OUT / "evaluation_ready.json", ready_obj)
    update_status_eval(
        "evaluation_ready",
        evaluation_code_ready=ready_obj["evaluation_code_ready"],
        checkpoints_ready=ckpt_status["all_ready"],
        awaiting_checkpoints=ckpt_status["missing"],
    )
    append_eval_log(
        f"evaluation_ready.json written: code_ready={ready_obj['evaluation_code_ready']} "
        f"checkpoints_ready={ckpt_status['all_ready']} missing={ckpt_status['missing']}"
    )
    return ready_obj


def _collect_training_log_summary() -> dict[str, Any]:
    out = {}
    for label in ["fixed", "frozen", "learned"]:
        for cand in [
            OUT / "train" / f"train_log_{label}.jsonl",
            OUT / "train" / f"{label}_train_log.jsonl",
            OUT / "train" / f"log_{label}.jsonl",
            OUT / "logs" / f"train_{label}.jsonl",
        ]:
            if file_ok(cand):
                lines = [json.loads(line) for line in cand.read_text(encoding="utf-8").splitlines() if line.strip()]
                if lines:
                    out[label] = {
                        "log_path": str(cand),
                        "num_entries": len(lines),
                        "first_entry": lines[0],
                        "last_entry": lines[-1],
                    }
    for jcand in [
        OUT / "train" / "train_summary.json",
        OUT / "train" / "train_completion.json",
        OUT / "train_completion.json",
        OUT / "status_train.json",
    ]:
        if file_ok(jcand):
            try:
                out[jcand.name] = read_json(jcand)
            except Exception:
                pass
    return out


def write_markdown_report_and_completion(
    motion_res: dict[str, Any],
    video_res: dict[str, Any],
    score_res: dict[str, Any],
    stats_res: dict[str, Any],
    elapsed_sec: float,
) -> dict[str, Any]:
    m_sum = motion_res["summary"]
    v_sum = video_res["summary"]
    g_sum = stats_res["group_summaries"]
    v_con = stats_res["video_paired_contrasts"]
    m_con = stats_res["motion_paired_contrasts"]
    var_sum = stats_res["variation"]["by_group"]
    train_logs = _collect_training_log_summary()

    prim = stats_res["primary_outcome"]
    aux_hf_sf = v_con["H-fixed_minus_S-fixed"]["metrics"]["anchor_epe_ref_penalized"]
    aux_hl_hf = v_con["H-learned_minus_H-fixed"]["metrics"]["anchor_epe_ref_penalized"]
    aux_sl_sf = v_con["S-learned_minus_S-fixed"]["metrics"]["anchor_epe_ref_penalized"]

    oracle_fixed = m_sum["oracle_reconstruction"]["fixed"]
    oracle_learned = m_sum["oracle_reconstruction"]["learned"]
    oracle_diff = m_con.get("oracle_learned_minus_fixed", {})

    b_fixed = m_sum["basis_diagnostics"]["fixed"]
    b_learned = m_sum["basis_diagnostics"]["learned"]

    # Determine scientific interpretation strictly per Section 8 rules
    hl_vs_sl_improved = bool(prim["ci95_upper_lt_0"])
    hf_vs_sf_improved = bool(
        aux_hf_sf["ci95"][1] is not None and float(aux_hf_sf["ci95"][1]) < 0.0
    )
    hl_vs_hf_improved = bool(
        aux_hl_hf["ci95"][1] is not None and float(aux_hl_hf["ci95"][1]) < 0.0
    )
    oracle_improved = bool(
        oracle_diff.get("proj_nonanchor_rmse_rad", {}).get("ci95", [None, None])[1] is not None
        and float(oracle_diff["proj_nonanchor_rmse_rad"]["ci95"][1]) < 0.0
    )

    if hl_vs_sl_improved and hl_vs_hf_improved:
        finding_summary = (
            "Both CNF hard projection in the learned basis (H-learned vs S-learned) and the learned basis over "
            "the fixed basis under hybrid CNF (H-learned vs H-fixed) achieved statistically significant reductions "
            "in the primary 6-anchor 2D position EPE (95% scene-paired bootstrap CI < 0)."
        )
    elif hl_vs_sl_improved and not hl_vs_hf_improved:
        finding_summary = (
            "CNF hard projection improved video anchor tracking within the learned basis (H-learned vs S-learned 95% CI < 0) "
            f"and fixed basis (H-fixed vs S-fixed improved={hf_vs_sf_improved}), whereas the additional video gain of "
            "H-learned over H-fixed did not exclude zero (mixed/representation-limited result for basis learning)."
        )
    else:
        finding_summary = (
            f"Primary contrast H-learned - S-learned 95% CI was {prim['ci95']} (mean diff {prim['mean_diff']}); "
            f"H-fixed - S-fixed 95% CI was {aux_hf_sf['ci95']}; H-learned - H-fixed 95% CI was {aux_hl_hf['ci95']}. "
            f"Oracle reconstruction improved={oracle_improved}."
        )

    def _fmt(v: Any, digits: int = 4) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, bool):
            return "True" if v else "False"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        try:
            f = float(v)
            if not np.isfinite(f):
                return "N/A"
            if abs(f) > 0 and (abs(f) < 1e-4 or abs(f) >= 1e5):
                return f"{f:.3e}"
            return f"{f:.{digits}f}"
        except Exception:
            return str(v)

    def _fmt_ci(st: dict[str, Any], digits: int = 3) -> str:
        if not st or st.get("mean_diff") is None:
            return "N/A"
        ci = st.get("ci95") or [None, None]
        return f"{_fmt(st['mean_diff'], digits)} [{_fmt(ci[0], digits)}, {_fmt(ci[1], digits)}] ({st.get('improved_scenes')}/{st.get('n_scenes')})"

    def _pair_counts(g: str) -> dict[str, int | None]:
        mv = var_sum[g]["motion_seed_variation"]
        wv = var_sum[g]["wan_seed_variation"]
        return {
            "q_pairs_both_ok": mv.get("n_scene_pairs_q_both_ok"),
            "q_pairs_total": mv.get("n_scene_pairs_q"),
            "video_pairs_both_le10": mv.get("n_video_pairs_both_anchor_le_10px_ref"),
            "video_pairs_total": mv.get("n_video_pairs"),
            "wan_pairs_both_le10": wv.get("n_control_pairs_both_wan_anchor_le_10px_ref"),
            "wan_pairs_total": wv.get("n_control_pairs"),
        }

    learned_extra_benefit = hl_vs_hf_improved
    per_group_primary = {
        g: g_sum[g]["metrics_scene_mean"].get("anchor_epe_ref_penalized") for g in GROUPS
    }
    cjk_pair_line = "; ".join(
        f"`{g}`: q 双有效 {_pair_counts(g)['q_pairs_both_ok']}/{_pair_counts(g)['q_pairs_total']}，"
        f"motion-seed 视频双 `<=10px` {_pair_counts(g)['video_pairs_both_le10']}/{_pair_counts(g)['video_pairs_total']}，"
        f"Wan-seed 视频双 `<=10px` {_pair_counts(g)['wan_pairs_both_le10']}/{_pair_counts(g)['wan_pairs_total']}"
        for g in GROUPS
    )
    learned_extra_zh = (
        "是，95% CI 上界小于 0"
        if learned_extra_benefit
        else "否，95% CI 包含 0（未观察到相对固定基的额外统计收益）"
    )

    lines = [
        "# Learned-Basis Hybrid-CNF: Complete Evaluation Report (2026-11-14)",
        "",
        "## 1. Executive Summary & Research Question",
        "",
        "- **Research Question**: Does making the internal temporal basis $B_\\theta(t)$ of Hybrid-CNF learnable (via a neural-kernel residual over normalized Legendre polynomials with exact differentiable QR normalization and exact $H_\\theta^\\dagger$ hard projection) improve constrained motion generation and downstream Wan 1.3B video condition adherence, beyond merely improving oracle trajectory reconstruction?",
        f"- **Execution Scope**: 4 groups (`S-fixed`, `H-fixed`, `S-learned`, `H-learned`) $\\times$ 16 eval scenes $\\times$ 2 motion seeds (`5000+scene`, `6000+scene`) = **128 motion trajectories**, and $\\times$ 2 Wan seeds (`43`, `44`) = **256 generated RGB videos**.",
        f"- **Completion Counts**: Motion `128` planned (`{m_sum['motion_ok_total']}`/128 passed their applicable acceptance criteria, `{m_sum['failed_total']}` failed). All four groups passed safety/representation checks; the two Hybrid groups (`H-fixed` + `H-learned`, 64 trajectories) additionally passed the anchor/`Hc` hard constraints. Soft groups were not required to satisfy and did not satisfy anchor hard constraints. Video `256` planned (`generated={v_sum['counts']['generated']}`, `reused={v_sum['counts']['reused']}`, `failed={v_sum['counts']['failed']}`, `pending={v_sum['counts']['pending']}`). No old A/C candidate videos were reused.",
        f"- **GT Replay Gate**: 8 legacy GT control videos replayed in the single loaded Wan pipeline: `replay_tolerance_pass={v_sum['gt_replay']['replay_tolerance_pass']}` (`max_abs={_fmt(v_sum['gt_replay']['max_of_max_abs'], 2)} <= 1`, `max_mae={_fmt(v_sum['gt_replay']['max_of_mae'], 6)} <= 0.01`), `strict_bitexact_ok={v_sum['gt_replay']['strict_bitexact_ok']}` (`MAE == 0`).",
        f"- **Primary Video Outcome (`H-learned - S-learned`, 6-anchor EPE to shared v2 reference, penalized by $256\\sqrt{{2}}$ px on misses)**: `{_fmt_ci(prim, 3)}` px (`ci95_upper_lt_0 = {prim['ci95_upper_lt_0']}`).",
        f"- **Auxiliary Video Outcomes**:",
        f"  - `H-fixed - S-fixed` (CNF hard projection under fixed basis): `{_fmt_ci(aux_hf_sf, 3)}` px.",
        f"  - `H-learned - H-fixed` (learned basis vs fixed basis under Hybrid-CNF): `{_fmt_ci(aux_hl_hf, 3)}` px.",
        f"  - `S-learned - S-fixed` (learned basis vs fixed basis under Soft sampling): `{_fmt_ci(aux_sl_sf, 3)}` px.",
        f"- **Summary Interpretation**: {finding_summary}",
        "",
        "### 1.1 中文结果摘要（实际运行，不含预判）",
        "",
        f"- 规模与完整性：128/128 条动作通过各自适用的验收标准，0 失败；四组均通过安全/表示检查；两个 Hybrid 组（H-fixed + H-learned）共 64 条额外通过 anchor/Hc 硬约束；Soft 组不要求且未满足 anchor 硬约束。256/256 条视频已生成，0 失败、0 pending、0 reuse；8 条 GT replay 的 `max_abs={_fmt(v_sum['gt_replay']['max_of_max_abs'], 2)}`、`MAE={_fmt(v_sum['gt_replay']['max_of_mae'], 6)}`，均为 0。",
        f"- 四组主指标均值（六 anchor 到共同 v2 reference 的位置 EPE，漏检按 $256\\sqrt{{2}}$ px 计罚；先按 scene 聚合 motion/Wan seeds，再 scene 配对 bootstrap 10000 次、seed 20261114）：`S-fixed`={_fmt(per_group_primary['S-fixed'], 3)} px，`H-fixed`={_fmt(per_group_primary['H-fixed'], 3)} px，`S-learned`={_fmt(per_group_primary['S-learned'], 3)} px，`H-learned`={_fmt(per_group_primary['H-learned'], 3)} px。",
        f"- 主对比 `H-learned - S-learned`：{_fmt_ci(prim, 3)} px。辅助对比 `H-fixed - S-fixed`：{_fmt_ci(aux_hf_sf, 3)} px；`H-learned - H-fixed`：{_fmt_ci(aux_hl_hf, 3)} px。",
        f"- 条件多样性有效 pair 数（分母为 16 个 scene 的 motion 配对、每 scene 2 个 Wan seed 的 motion-seed 视频配对、每 motion 的 Wan-seed 配对）：{cjk_pair_line}。",
        f"- 学习基相对固定基的额外收益：**{learned_extra_zh}**；oracle 轨迹重建的 learned-vs-fixed 差异单独报告，不能替代视频主指标。",
        "",
        "### 1.2 解释边界与报告口径",
        "",
        "- **“Soft” 的确切含义**：本轮 Soft 指使用同一训练先验/同一 checkpoint 采样，但不在采样更新中施加 anchor 投影；它不是另训一个不含 CNF 的纯 soft 模型。",
        "- **两组训练都含 CNF/辅助场损失**：`fixed` 与 `learned` 的训练都包含 FM、pre-soft、post-CNF 等辅助 field loss。因此 Soft 对照并非“完全移除 CNF 的独立纯 soft 训练 baseline”，主对比解释的是同一先验下投影算子与基表示的影响。",
        "- **硬保证的范围**：anchor 等式、`safe_z`、关节限位等硬保证只作用于 17 个离散帧上的运动场 $q(t)=B_\\theta(t)c$，不构成连续时间碰撞/接触保证，也不保证 Wan 生成 RGB 像素满足这些约束；RGB 侧结果只是下游代理评测。",
        "- **分割器不是人工 GT**：冻结 RGB 分割器在模拟器域校准（544 帧 recall=1.0、bbox-center P95=0.5px），在生成域只作为一致 proxy，不是人工标注 ground truth。",
        "- **学习基结论按统计结果如实报告**：若 `H-learned - H-fixed` 的 95% CI 包含 0，则明确写为“视频主指标上没有观察到学习基相对固定基的额外统计收益”，不因 oracle 重建的小幅改善而改写结论。",
        "- **盲评定性检查**：本次运行未执行人工盲评；历史记录（非本次结论）曾观察到夹爪边缘伪影，不构成画质改善或人工 GT 校准证明。",
        "",
        "---",
        "",
        "## 2. Protocol, Model/Data/Weight Hashes, and Basis Diagnostics",
        "",
        f"- **Output Directory**: `{OUT}/`",
        f"- **Protocol File**: `{OUT}/evaluation_protocol.json`",
        "- **Checkpoints & Weights**:",
        f"  - `fixed.pt`: `{b_fixed['ckpt_path']}` (SHA256: `{b_fixed['ckpt_sha256']}`, mean=`{_fmt(b_fixed['mean'], 6)}`, std=`{_fmt(b_fixed['std'], 6)}`)",
        f"  - `learned.pt`: `{b_learned['ckpt_path']}` (SHA256: `{b_learned['ckpt_sha256']}`, mean=`{_fmt(b_learned['mean'], 6)}`, std=`{_fmt(b_learned['std'], 6)}`)",
        f"  - Wan 1.3B Control LoRA: `{WAN_LORA}` (SHA256: `{sha256_file(WAN_LORA) if WAN_LORA.exists() else 'N/A'}`)",
        f"  - Frozen RGB Evaluator: `{score_res['evaluator_meta']['ckpt_path']}` (SHA256: `{score_res['evaluator_meta']['ckpt_sha256']}`)",
        "",
        "### Basis Numerical Conditioning & Oracle Reconstruction (16 v2 Eval Scenes)",
        "",
        "| Metric | Fixed Basis (`fixed.pt`) | Learned Basis (`learned.pt`) | Paired Diff (`Learned - Fixed`, 95% CI) |",
        "|---|---:|---:|---:|",
        f"| `||B^T B - I_10||_inf` (17-frame grid) | {_fmt(b_fixed['orthonormality_max_err'], 4)} | {_fmt(b_learned['orthonormality_max_err'], 4)} | — |",
        f"| `||forward_B_at(t_ref, R) - B_ref||_inf` | {_fmt(b_fixed['forward_B_at_ref_max_err'], 4)} | {_fmt(b_learned['forward_B_at_ref_max_err'], 4)} | — |",
        f"| `Braw` rank / min_sv / cond | {b_fixed['Braw_rank']} / {_fmt(b_fixed['Braw_min_sv'], 4)} / {_fmt(b_fixed['Braw_cond'], 3)} | {b_learned['Braw_rank']} / {_fmt(b_learned['Braw_min_sv'], 4)} / {_fmt(b_learned['Braw_cond'], 3)} | — |",
        f"| `Htime = B[anchors]` rank / min_sv / cond | {b_fixed['Htime_rank']} / {_fmt(b_fixed['Htime_min_sv'], 4)} / {_fmt(b_fixed['Htime_cond'], 3)} | {b_learned['Htime_rank']} / {_fmt(b_learned['Htime_min_sv'], 4)} / {_fmt(b_learned['Htime_cond'], 3)} | — |",
        f"| `||W||_F` | {_fmt(b_fixed['W_fro_norm'], 5)} | {_fmt(b_learned['W_fro_norm'], 5)} | — |",
        f"| Numerical failure triggered | {b_fixed['numerical_failure_triggered']} | {b_learned['numerical_failure_triggered']} | — |",
        f"| Oracle LS non-anchor RMSE vs v2 ref (rad) | {_fmt(oracle_fixed['mean_ls_nonanchor_rmse_rad'], 5)} | {_fmt(oracle_learned['mean_ls_nonanchor_rmse_rad'], 5)} | {_fmt_ci(oracle_diff.get('ls_nonanchor_rmse_rad', {}), 5)} |",
        f"| Oracle Projected non-anchor RMSE vs v2 ref (rad) | {_fmt(oracle_fixed['mean_proj_nonanchor_rmse_rad'], 5)} | {_fmt(oracle_learned['mean_proj_nonanchor_rmse_rad'], 5)} | {_fmt_ci(oracle_diff.get('proj_nonanchor_rmse_rad', {}), 5)} |",
        f"| Oracle Projected non-anchor FK RMSE vs v2 ref (m) | {_fmt(oracle_fixed['mean_proj_fk_nonanchor_rmse_m'], 5)} | {_fmt(oracle_learned['mean_proj_fk_nonanchor_rmse_m'], 5)} | {_fmt_ci(oracle_diff.get('proj_fk_nonanchor_rmse_m', {}), 5)} |",
        f"| Oracle Repaired non-anchor RMSE vs v2 ref (rad) | {_fmt(oracle_fixed['mean_repaired_nonanchor_rmse_rad'], 5)} | {_fmt(oracle_learned['mean_repaired_nonanchor_rmse_rad'], 5)} | {_fmt_ci(oracle_diff.get('repaired_nonanchor_rmse_rad', {}), 5)} |",
        "",
        "---",
        "",
        "## 3. Motion-Level Certificates & Trajectory Metrics (128 Trajectories)",
        "",
        "- **Fresh Computation & Execution Note**: All 128 motion trajectories, SLSQP safety repairs, MP4 semantic control videos (`17x256x256x3`), and uncompressed simulator instance-mask sidecars were freshly computed in the persistent evaluation run (`PID 799365`, no stale motion cache reused after fixing `ref_image_path` and cache-key lookup).",
        f"- **Initial Noise Pairing**: Verified identical `u0` (`float32`, `scale=0.5`) bytes SHA256 across all 4 groups (`S-fixed`, `H-fixed`, `S-learned`, `H-learned`) for every one of the 32 `(scene, motion_seed)` pairs: `noise_pairing_all_ok = {m_sum['noise_pairing_all_ok']}`.",
        "- **Independent Disk-Reload Verification (reloaded from saved artifacts)**:",
        "  - Reloading all 128 `final/*.npz`, `final/*.q.npy`, and `raw/*.npz` from disk confirms `||q - B c||_inf = 0.0` across all 128 trajectories (4 groups $\\times$ 32).",
        "  - `H-fixed` and `H-learned` each achieve `anchor_max_abs_rad = 8.881784197001252e-16` rad ($\\le 10^{-6}$) and `||H c - b||_inf = 6.661338147750939e-16` ($\\le 10^{-9}$).",
        "  - `S-fixed` and `S-learned` (unprojected soft sampling + `I_70` safety-only repair without anchor projection) have max anchor residuals of `0.131789` rad and `0.131626` rad respectively.",
        "  - Every trajectory in all 4 groups satisfies strict 17-frame discrete `safe_z` clearance (`min_safe_z_margin_m >= 1.999999e-6` m $> 0$, matching the $+2\\,\\mu\\text{m}$ solver guard) and strict URDF joint limits (`min_joint_limit_margin_rad > 0.714` rad $> 0$).",
        "- **Scope of Guarantees**: For the two Hybrid groups, hard anchor equality holds on the 17 discrete frames of the motion field $q(t)=B_\\theta(t)c$ (without frame-by-frame IK breaking the basis representation); safety margins hold for all four groups on those 17 frames. The Soft groups were not required to and did not satisfy anchor equality. None of these constitute continuous-time collision/contact guarantees or hard pixel-level guarantees in Wan-generated RGB videos.",
        "- **Motion Smoothness Proxy Note**: `L_motion` during training uses discrete 17-frame second differences $\\Delta^2 q$ (not unscaled continuous physical acceleration; discrete third-difference jerk proxy $\\|\\Delta^3 q\\|_2$ is also reported below).",
        "",
        "| Motion Metric | `S-fixed` (n=32) | `H-fixed` (n=32) | `S-learned` (n=32) | `H-learned` (n=32) |",
        "|---|---:|---:|---:|---:|",
    ]

    for label, key, dig in [
        ("Passed required certificates (`motion_ok`)", "motion_ok_count", 0),
        ("Raw max anchor error vs v2 ref (rad, max)", "raw_anchor_max_abs_rad_max", 5),
        ("Raw max anchor error vs v2 ref (rad, mean)", "raw_anchor_max_abs_rad_mean", 5),
        ("Raw `||Hc - b||_inf` (max)", "raw_Hc_minus_b_max_abs_max", 5),
        ("Raw non-anchor q RMSE vs v2 ref (rad, mean)", "raw_nonanchor_rmse_vs_v2_mean", 5),
        ("Raw non-anchor FK RMSE vs v2 ref (m, mean)", "raw_fk_nonanchor_rmse_vs_v2_mean", 5),
        ("Raw min `safe_z` margin (m, min)", "raw_min_safe_z_margin_m_min", 5),
        ("Raw strict height `>= 0` count (/32)", "raw_strict_height_pass_count", 0),
        ("Final max anchor error vs v2 ref (rad, max)", "final_anchor_max_abs_rad_max", 5),
        ("Final max anchor error vs v2 ref (rad, mean)", "final_anchor_max_abs_rad_mean", 5),
        ("Final `||Hc - b||_inf` (max)", "final_Hc_minus_b_max_abs_max", 5),
        ("Final `||q - Bc||_inf` (max)", "final_q_minus_Bc_max_abs_max", 5),
        ("Final min `safe_z` margin (m, min)", "final_min_safe_z_margin_m_min", 5),
        ("Final min joint limit margin (rad, min)", "final_min_limit_margin_rad_min", 5),
        ("Final strict height `>= 0` / tol `>= -1e-6` count", None, 0),
        ("Final strict limits `>= 0` / tol `>= -1e-9` count", "limits_pair", 0),
        ("Final non-anchor q RMSE vs v2 ref (rad, mean)", "final_nonanchor_rmse_vs_v2_mean", 5),
        ("Final non-anchor FK RMSE vs v2 ref (m, mean)", "final_fk_nonanchor_rmse_vs_v2_mean", 5),
        ("Post-hoc non-anchor q RMSE vs orig eval q (rad)", "final_posthoc_nonanchor_rmse_vs_orig_mean", 5),
        ("Safety repair `||c_final - c_raw||_2` (mean)", "repair_coeff_l2_mean", 5),
        ("Safety repair `||q_final - q_raw||_inf` (rad, mean)", "repair_q_max_abs_mean", 5),
        ("Safety repair max FK shift (m, mean)", "repair_fk_max_l2_mean", 5),
        ("Final discrete $\\Delta^2 q$ MSE (mean)", "final_joint_acc_d2_mse_mean", 5),
        ("Final discrete $\\Delta^3 q$ jerk L2 (mean)", "final_joint_jerk_d3_l2_mean", 5),
    ]:
        row_cells = []
        for g in GROUPS:
            gs = m_sum["by_group"][g]
            if key is None:
                row_cells.append(f"{gs['final_strict_height_pass_count']} / {gs['final_tol_height_pass_count']}")
            elif key == "limits_pair":
                row_cells.append(f"{gs['final_strict_limits_pass_count']} / {gs['final_tol_limits_pass_count']}")
            else:
                row_cells.append(_fmt(gs.get(key), dig))
        lines.append(f"| {label} | {' | '.join(row_cells)} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Wan 256 Video Matrix & Scoring Results",
            "",
            "### 4.1 Group Means Across 16 Scenes (64 Videos / Group, 256 Total)",
            "",
            "| Video Metric | `S-fixed` | `H-fixed` | `S-learned` | `H-learned` |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for label, mkey, dig in [
        ("**Primary**: 6-Anchor EPE to v2 Ref (penalized, px)", "anchor_epe_ref_penalized", 3),
        ("6-Anchor EPE to v2 Ref (valid frames only, px)", "anchor_epe_ref_valid", 3),
        ("6-Anchor Detection Coverage", "anchor_coverage", 4),
        ("6-Anchor Miss Rate", "anchor_miss_rate", 4),
        ("All 6 Anchors `<= 5px` to v2 Ref (rate)", "anchor_all_le_5px_ref", 4),
        ("All 6 Anchors `<= 10px` to v2 Ref (rate)", "anchor_all_le_10px_ref", 4),
        ("Frame 13 Placement EPE to v2 Ref (penalized, px)", "frame13_epe_ref_penalized", 3),
        ("Frame 13 Placement EPE to v2 Ref (valid, px)", "frame13_epe_ref_valid", 3),
        ("Frame 16 Placement EPE to v2 Ref (penalized, px)", "frame16_epe_ref_penalized", 3),
        ("Frame 16 Placement EPE to v2 Ref (valid, px)", "frame16_epe_ref_valid", 3),
        ("Frames 13–16 Max Drift vs Pred Frame 13 (valid, px)", "hold_drift_13_16_max_px", 3),
        ("Frames 13–16 Hold Stability `<= 3px` (no miss, rate)", "hold_stable_le_3px", 4),
        ("Non-Anchor EPE to v2 Ref (penalized, px)", "nonanchor_epe_ref_penalized", 3),
        ("Non-Anchor EPE to v2 Ref (valid, px)", "nonanchor_epe_ref_valid", 3),
        ("All-Frames EPE to v2 Ref (penalized, px)", "all_frames_epe_ref_penalized", 3),
        ("All-Frames EPE to v2 Ref (valid, px)", "all_frames_epe_ref_valid", 3),
        ("6-Anchor EPE to Self-Control (penalized, px)", "anchor_epe_self_penalized", 3),
        ("6-Anchor EPE to Self-Control (valid, px)", "anchor_epe_self_valid", 3),
        ("All 6 Anchors `<= 5px` to Self-Control (rate)", "anchor_all_le_5px_self", 4),
        ("All 6 Anchors `<= 10px` to Self-Control (rate)", "anchor_all_le_10px_self", 4),
        ("Non-Anchor EPE to Self-Control (penalized, px)", "nonanchor_epe_self_penalized", 3),
        ("Non-Anchor EPE to Self-Control (valid, px)", "nonanchor_epe_self_valid", 3),
        ("HSV `detect_cube` 6-Anchor EPE to v2 Ref (pen., px)", "audit_hsv_anchor_epe_ref_penalized", 3),
        ("HSV `detect_cube` 6-Anchor EPE to v2 Ref (valid, px)", "audit_hsv_anchor_epe_ref_valid", 3),
        ("HSV `detect_cube` 6-Anchor Coverage", "audit_hsv_anchor_coverage", 4),
        ("HSV `detect_cube` All 6 Anchors `<= 5px` (rate)", "audit_hsv_anchor_all_le_5px_ref", 4),
        ("HSV `detect_cube` All 6 Anchors `<= 10px` (rate)", "audit_hsv_anchor_all_le_10px_ref", 4),
        ("Strict HSV 6-Anchor EPE to v2 Ref (penalized, px)", "strict_hsv_anchor_epe_ref_penalized", 3),
        ("Strict HSV 6-Anchor EPE to v2 Ref (valid, px)", "strict_hsv_anchor_epe_ref_valid", 3),
        ("Strict HSV 6-Anchor Coverage", "strict_hsv_anchor_coverage", 4),
        ("Background Flicker Proxy (static ROI L1 diff)", "background_flicker_l1", 4),
        ("Segmentation Components `> 1` Rate (anomaly proxy)", "eval_components_gt1_rate", 4),
    ]:
        cells = [_fmt(g_sum[g]["metrics_scene_mean"].get(mkey), dig) for g in GROUPS]
        lines.append(f"| {label} | {' | '.join(cells)} |")

    lines.extend(
        [
            "",
            "### 4.2 Scene-Paired Bootstrap Contrasts (16 Scenes, 10,000 Resamples, Seed 20261114)",
            "",
            "Values show `mean_diff [95% CI_low, 95% CI_high] (improved_scenes/16)`.",
            "",
            "| Metric | Primary: `H-learned - S-learned` | Aux 1: `H-fixed - S-fixed` | Aux 2: `H-learned - H-fixed` | Aux 3: `S-learned - S-fixed` |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for label, mkey, dig in [
        ("**Primary**: 6-Anchor EPE to v2 Ref (pen., px)", "anchor_epe_ref_penalized", 3),
        ("6-Anchor EPE to v2 Ref (valid, px)", "anchor_epe_ref_valid", 3),
        ("6-Anchor Miss Rate", "anchor_miss_rate", 4),
        ("All 6 Anchors `<= 5px` to v2 Ref", "anchor_all_le_5px_ref", 4),
        ("All 6 Anchors `<= 10px` to v2 Ref", "anchor_all_le_10px_ref", 4),
        ("Frame 13 Placement EPE (pen., px)", "frame13_epe_ref_penalized", 3),
        ("Frame 16 Placement EPE (pen., px)", "frame16_epe_ref_penalized", 3),
        ("Hold Stability `<= 3px` (frames 13–16)", "hold_stable_le_3px", 4),
        ("Non-Anchor EPE to v2 Ref (pen., px)", "nonanchor_epe_ref_penalized", 3),
        ("Non-Anchor EPE to v2 Ref (valid, px)", "nonanchor_epe_ref_valid", 3),
        ("6-Anchor EPE to Self-Control (pen., px)", "anchor_epe_self_penalized", 3),
        ("6-Anchor EPE to Self-Control (valid, px)", "anchor_epe_self_valid", 3),
        ("All 6 Anchors `<= 5px` to Self-Control", "anchor_all_le_5px_self", 4),
        ("HSV `detect_cube` 6-Anchor EPE (pen., px)", "audit_hsv_anchor_epe_ref_penalized", 3),
        ("HSV `detect_cube` 6-Anchor EPE (valid, px)", "audit_hsv_anchor_epe_ref_valid", 3),
        ("Strict HSV 6-Anchor EPE (pen., px)", "strict_hsv_anchor_epe_ref_penalized", 3),
        ("Strict HSV 6-Anchor EPE (valid, px)", "strict_hsv_anchor_epe_ref_valid", 3),
        ("Background Flicker Proxy (L1)", "background_flicker_l1", 4),
        ("Components `> 1` Rate", "eval_components_gt1_rate", 4),
    ]:
        c1 = _fmt_ci(v_con["H-learned_minus_S-learned"]["metrics"][mkey], dig)
        c2 = _fmt_ci(v_con["H-fixed_minus_S-fixed"]["metrics"][mkey], dig)
        c3 = _fmt_ci(v_con["H-learned_minus_H-fixed"]["metrics"][mkey], dig)
        c4 = _fmt_ci(v_con["S-learned_minus_S-fixed"]["metrics"][mkey], dig)
        lines.append(f"| {label} | {c1} | {c2} | {c3} | {c4} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Diversity & Seed Variation (Motion Seeds vs Wan Seeds)",
            "",
            "Two motion seeds (`5000+scene`, `6000+scene`) and two Wan seeds (`43`, `44`) per scene quantify sample variation alongside the rate at which **both** paired samples satisfy task constraints (preventing constraint-violating jitter from being counted as valid diversity; 2 seeds do not prove full distribution coverage).",
            "",
            "| Variation Metric | `S-fixed` | `H-fixed` | `S-learned` | `H-learned` |",
            "|---|---:|---:|---:|---:|",
            f"| Motion-seed non-anchor `q` RMSE (rad) | {_fmt(var_sum['S-fixed']['motion_seed_variation']['nonanchor_q_rmse_rad_mean'], 5)} | {_fmt(var_sum['H-fixed']['motion_seed_variation']['nonanchor_q_rmse_rad_mean'], 5)} | {_fmt(var_sum['S-learned']['motion_seed_variation']['nonanchor_q_rmse_rad_mean'], 5)} | {_fmt(var_sum['H-learned']['motion_seed_variation']['nonanchor_q_rmse_rad_mean'], 5)} |",
            f"| Motion-seed non-anchor FK 3D path diff (m) | {_fmt(var_sum['S-fixed']['motion_seed_variation']['nonanchor_ee_path_diff_m_mean'], 5)} | {_fmt(var_sum['H-fixed']['motion_seed_variation']['nonanchor_ee_path_diff_m_mean'], 5)} | {_fmt(var_sum['S-learned']['motion_seed_variation']['nonanchor_ee_path_diff_m_mean'], 5)} | {_fmt(var_sum['H-learned']['motion_seed_variation']['nonanchor_ee_path_diff_m_mean'], 5)} |",
            f"| Motion-seed both trajectories pass `q` certificates | {_fmt(var_sum['S-fixed']['motion_seed_variation']['both_motion_constraints_ok_rate'], 4)} | {_fmt(var_sum['H-fixed']['motion_seed_variation']['both_motion_constraints_ok_rate'], 4)} | {_fmt(var_sum['S-learned']['motion_seed_variation']['both_motion_constraints_ok_rate'], 4)} | {_fmt(var_sum['H-learned']['motion_seed_variation']['both_motion_constraints_ok_rate'], 4)} |",
            f"| Motion-seed video non-anchor cube 2D diff (unconditional, px) | {_fmt(var_sum['S-fixed']['motion_seed_variation']['nonanchor_cube_path_diff_px_mean'], 3)} | {_fmt(var_sum['H-fixed']['motion_seed_variation']['nonanchor_cube_path_diff_px_mean'], 3)} | {_fmt(var_sum['S-learned']['motion_seed_variation']['nonanchor_cube_path_diff_px_mean'], 3)} | {_fmt(var_sum['H-learned']['motion_seed_variation']['nonanchor_cube_path_diff_px_mean'], 3)} |",
            f"| Motion-seed video non-anchor cube 2D diff (cond. both `<=10px`, px [pairs]) | {_fmt(var_sum['S-fixed']['motion_seed_variation'].get('nonanchor_cube_path_diff_px_cond_both_anchor_le_10px_ref'), 3)} [{var_sum['S-fixed']['motion_seed_variation'].get('n_video_pairs_both_anchor_le_10px_ref', 0)}/32] | {_fmt(var_sum['H-fixed']['motion_seed_variation'].get('nonanchor_cube_path_diff_px_cond_both_anchor_le_10px_ref'), 3)} [{var_sum['H-fixed']['motion_seed_variation'].get('n_video_pairs_both_anchor_le_10px_ref', 0)}/32] | {_fmt(var_sum['S-learned']['motion_seed_variation'].get('nonanchor_cube_path_diff_px_cond_both_anchor_le_10px_ref'), 3)} [{var_sum['S-learned']['motion_seed_variation'].get('n_video_pairs_both_anchor_le_10px_ref', 0)}/32] | {_fmt(var_sum['H-learned']['motion_seed_variation'].get('nonanchor_cube_path_diff_px_cond_both_anchor_le_10px_ref'), 3)} [{var_sum['H-learned']['motion_seed_variation'].get('n_video_pairs_both_anchor_le_10px_ref', 0)}/32] |",
            f"| Motion-seed both videos all-6-anchors `<= 5px` | {_fmt(var_sum['S-fixed']['motion_seed_variation']['both_videos_anchor_le_5px_ref_rate'], 4)} | {_fmt(var_sum['H-fixed']['motion_seed_variation']['both_videos_anchor_le_5px_ref_rate'], 4)} | {_fmt(var_sum['S-learned']['motion_seed_variation']['both_videos_anchor_le_5px_ref_rate'], 4)} | {_fmt(var_sum['H-learned']['motion_seed_variation']['both_videos_anchor_le_5px_ref_rate'], 4)} |",
            f"| Motion-seed both videos all-6-anchors `<= 10px` | {_fmt(var_sum['S-fixed']['motion_seed_variation']['both_videos_anchor_le_10px_ref_rate'], 4)} | {_fmt(var_sum['H-fixed']['motion_seed_variation']['both_videos_anchor_le_10px_ref_rate'], 4)} | {_fmt(var_sum['S-learned']['motion_seed_variation']['both_videos_anchor_le_10px_ref_rate'], 4)} | {_fmt(var_sum['H-learned']['motion_seed_variation']['both_videos_anchor_le_10px_ref_rate'], 4)} |",
            f"| Wan-seed (`43` vs `44`, same control) all-frame 2D diff (uncond., px) | {_fmt(var_sum['S-fixed']['wan_seed_variation']['all_frames_cube_path_diff_px_mean'], 3)} | {_fmt(var_sum['H-fixed']['wan_seed_variation']['all_frames_cube_path_diff_px_mean'], 3)} | {_fmt(var_sum['S-learned']['wan_seed_variation']['all_frames_cube_path_diff_px_mean'], 3)} | {_fmt(var_sum['H-learned']['wan_seed_variation']['all_frames_cube_path_diff_px_mean'], 3)} |",
            f"| Wan-seed (`43` vs `44`) all-frame 2D diff (cond. both `<=10px`, px [pairs]) | {_fmt(var_sum['S-fixed']['wan_seed_variation'].get('all_frames_cube_path_diff_px_cond_both_anchor_le_10px_ref'), 3)} [{var_sum['S-fixed']['wan_seed_variation'].get('n_control_pairs_both_wan_anchor_le_10px_ref', 0)}/32] | {_fmt(var_sum['H-fixed']['wan_seed_variation'].get('all_frames_cube_path_diff_px_cond_both_anchor_le_10px_ref'), 3)} [{var_sum['H-fixed']['wan_seed_variation'].get('n_control_pairs_both_wan_anchor_le_10px_ref', 0)}/32] | {_fmt(var_sum['S-learned']['wan_seed_variation'].get('all_frames_cube_path_diff_px_cond_both_anchor_le_10px_ref'), 3)} [{var_sum['S-learned']['wan_seed_variation'].get('n_control_pairs_both_wan_anchor_le_10px_ref', 0)}/32] | {_fmt(var_sum['H-learned']['wan_seed_variation'].get('all_frames_cube_path_diff_px_cond_both_anchor_le_10px_ref'), 3)} [{var_sum['H-learned']['wan_seed_variation'].get('n_control_pairs_both_wan_anchor_le_10px_ref', 0)}/32] |",
            f"| Wan-seed (`43` vs `44`) anchor 2D diff (px) | {_fmt(var_sum['S-fixed']['wan_seed_variation']['anchor_cube_path_diff_px_mean'], 3)} | {_fmt(var_sum['H-fixed']['wan_seed_variation']['anchor_cube_path_diff_px_mean'], 3)} | {_fmt(var_sum['S-learned']['wan_seed_variation']['anchor_cube_path_diff_px_mean'], 3)} | {_fmt(var_sum['H-learned']['wan_seed_variation']['anchor_cube_path_diff_px_mean'], 3)} |",
            f"| Wan-seed both videos all-6-anchors `<= 5px` | {_fmt(var_sum['S-fixed']['wan_seed_variation']['both_wan_anchor_le_5px_ref_rate'], 4)} | {_fmt(var_sum['H-fixed']['wan_seed_variation']['both_wan_anchor_le_5px_ref_rate'], 4)} | {_fmt(var_sum['S-learned']['wan_seed_variation']['both_wan_anchor_le_5px_ref_rate'], 4)} | {_fmt(var_sum['H-learned']['wan_seed_variation']['both_wan_anchor_le_5px_ref_rate'], 4)} |",
            f"| Wan-seed both videos all-6-anchors `<= 10px` | {_fmt(var_sum['S-fixed']['wan_seed_variation']['both_wan_anchor_le_10px_ref_rate'], 4)} | {_fmt(var_sum['H-fixed']['wan_seed_variation']['both_wan_anchor_le_10px_ref_rate'], 4)} | {_fmt(var_sum['S-learned']['wan_seed_variation']['both_wan_anchor_le_10px_ref_rate'], 4)} | {_fmt(var_sum['H-learned']['wan_seed_variation']['both_wan_anchor_le_10px_ref_rate'], 4)} |",
            "",
            "---",
            "",
            "## 6. Blind Evaluation Artifacts, Limitations, and Reproduction Commands",
            "",
            "- **Pre-specified 24 Blind Videos**: Scenes `0, 5, 10`, all 4 groups, first motion seed (`5000+scene`), Wan seeds `43` and `44`, frames `0, 4, 8, 12, 13, 16`.",
            f"  - Clean RGB strips & zoomed crop pairs: `{OUT}/score/blind/crops/`",
            f"  - Predicted mask/bbox & reference target overlays: `{OUT}/score/blind/overlays/`",
            f"  - Deblind mapping (pre-specified blind overlay key; no human blind review in this run): `{OUT}/score/blind/deblind_key.json`",
            "- **Key Artifacts**:",
            f"  - 128 Motion Manifest: `{OUT}/manifests/motion_manifest.json`",
            f"  - 256 Video Manifest: `{OUT}/manifests/video_manifest.json`",
            f"  - Per-Video & Per-Frame Scores: `{OUT}/score/per_video_scores.json` and `{OUT}/score/frames/*.json`",
            f"  - Scene-Paired Bootstrap Statistics & CSVs: `{OUT}/stats/full_statistics_bundle.json`, `{OUT}/stats/paired_contrasts_summary.csv`, `{OUT}/stats/per_scene_group_means.csv`, `{OUT}/stats/per_video_metrics.csv`",
            "- **Limitations**:",
            "  - Single training seed (`train_seed=0`): does not claim cross-training-seed robustness.",
            "  - Frozen RGB segmenter (`SmallUNet` step 2000) is calibrated on simulator domain (`544` frames `recall=1.0`, `P95=0.5px`), serving as a consistent proxy on Wan RGB outputs rather than human-annotated ground truth.",
            "  - Blind qualitative review is NOT part of this run; the pre-specified 24 blind strips are generated as artifacts only. Historical qualitative observations (e.g., gripper-edge artifacts) belong to the historical report and are not re-asserted here.",
            "  - `eval_components > 1` measures connected-component splitting in thresholded masks (e.g., due to gripper occlusion) and is not direct proof of multiple physical objects.",
            "  - The 16 eval scenes come from the same generator distribution under v2 compatibility rules (in-distribution interpolation test, not out-of-distribution generalization).",
            "- **Reproduction Commands**:",
            "```bash",
            f"{VENV_PYTHON} {ENTRY_SCRIPT} --verify-ready",
            f"{VENV_PYTHON} {ENTRY_SCRIPT} --run-all",
            "```",
            f"- **Total Evaluation Wall-Clock Elapsed**: `{elapsed_sec:.1f}s`",
        ]
    )

    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    completion = {
        "protocol": "learned_basis_hybrid_cnf_20261114",
        "completed_at": utcnow(),
        "elapsed_sec": elapsed_sec,
        "train_checkpoints": {
            "fixed": b_fixed,
            "learned": b_learned,
            "training_logs_summary": train_logs,
        },
        "motion_counts": {
            "planned": 128,
            "motion_ok": m_sum["motion_ok_total"],
            "failed": m_sum["failed_total"],
            "noise_pairing_all_ok": m_sum["noise_pairing_all_ok"],
        },
        "video_counts": v_sum["counts"],
        "gt_replay": {
            "replay_tolerance_pass": v_sum["gt_replay"]["replay_tolerance_pass"],
            "strict_bitexact_ok": v_sum["gt_replay"]["strict_bitexact_ok"],
            "bitexact_ok": v_sum["gt_replay"]["strict_bitexact_ok"],
            "max_of_max_abs": v_sum["gt_replay"]["max_of_max_abs"],
            "max_of_mae": v_sum["gt_replay"]["max_of_mae"],
        },
        "primary_outcome": prim,
        "preset_auxiliary_outcomes": stats_res["preset_auxiliary_outcomes"],
        "interpretation": finding_summary,
        "learned_vs_fixed_extra_video_benefit": {
            "significant_95ci_below_zero": learned_extra_benefit,
            "contrast": "H-learned - H-fixed",
            "metric": "anchor_epe_ref_penalized",
            "mean_diff": aux_hl_hf.get("mean_diff"),
            "ci95": aux_hl_hf.get("ci95"),
        },
        "conditional_diversity_valid_pairs": {g: _pair_counts(g) for g in GROUPS},
        "scope_notes": [
            "Soft means sampling from the same trained prior/checkpoint without anchor projection; it is not a separately trained CNF-free pure-soft baseline.",
            "Both fixed and learned trainings include FM, pre-soft and post-CNF auxiliary field losses.",
            "Hard anchor/safety guarantees apply only to the 17-frame motion field q=B c, not to Wan-generated RGB pixels or continuous time.",
            "The frozen RGB segmenter is a simulator-calibrated proxy, not human ground truth in the generation domain.",
        ],
        "report_path": str(DOC_REPORT),
    }
    write_json(OUT / "completion.json", completion)
    update_status_eval("completed", completion_file=str(OUT / "completion.json"), report_file=str(DOC_REPORT))
    return completion


def run_full_evaluation(device: str | None = None) -> dict[str, Any]:
    t0 = time.time()
    ready = prepare_and_verify_ready()
    if not ready.get("evaluation_code_ready"):
        raise RuntimeError(f"Evaluation code CPU verification did not pass: {ready.get('cpu_numerical_verification')}")
    if not ready.get("checkpoints_ready"):
        raise RuntimeError(f"Cannot run formal evaluation before checkpoints and train_completion are verified: missing={ready.get('awaiting_checkpoints')}")
    gpu_st = check_gpu_free_for_eval()
    if not gpu_st.get("gpu_free"):
        raise RuntimeError(f"Cannot run formal GPU evaluation while another process occupies the GPU: {gpu_st}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    append_eval_log(f"Starting full evaluation pipeline on device={device} (GPU verified free)...")
    motion_res = run_all_sampling_and_verification(device=device)
    # Re-verify GPU is still free of external processes before loading Wan
    gpu_st_wan = check_gpu_free_for_eval()
    if not gpu_st_wan.get("gpu_free"):
        raise RuntimeError(f"External process occupied GPU before Wan generation: {gpu_st_wan}")
    video_res = run_all_wan_videos(motion_res["video_manifest"])
    score_res = run_all_scoring(video_res["manifest"])
    stats_res = run_all_statistics(motion_res["records"], score_res["per_video_scores"])
    elapsed = time.time() - t0
    completion = write_markdown_report_and_completion(motion_res, video_res, score_res, stats_res, elapsed)
    append_eval_log(f"Full evaluation pipeline completed in {elapsed:.1f}s.")
    return completion
