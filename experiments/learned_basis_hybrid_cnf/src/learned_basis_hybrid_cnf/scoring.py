from __future__ import annotations

import random
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from .util_eval import (
    ANCHOR_FRAMES,
    BLIND_SCENES,
    BOOT_SEED,
    DIAGONAL,
    EVALUATOR_CAL,
    EVALUATOR_PT,
    EVALUATOR_SPEC,
    FRAMES,
    GROUPS,
    NONANCHOR_FRAMES,
    OUT,
    V2_COND,
    WAN_SEEDS,
    append_eval_log,
    audit_hsv_detector,
    decode_video,
    ensure_eval_dirs,
    file_ok,
    point_err,
    read_json,
    select_largest_component,
    sha256_file,
    strict_hsv_detector,
    update_status_eval,
    utcnow,
    write_json,
)


class SmallUNet(nn.Module):
    """Exact frozen segmenter architecture from video_task_validation_20261114."""

    def __init__(self, in_ch: int = 3, base: int = 16):
        super().__init__()

        def block(ic: int, oc: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.e1 = block(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = block(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = block(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)
        self.b = block(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = block(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        b = self.b(self.p3(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def load_frozen_evaluator(device: str | torch.device | None = None) -> tuple[SmallUNet, torch.device, dict[str, Any]]:
    if not file_ok(EVALUATOR_PT):
        raise FileNotFoundError(f"Frozen evaluator checkpoint missing: {EVALUATOR_PT}")
    if not file_ok(EVALUATOR_SPEC) or not file_ok(EVALUATOR_CAL):
        raise FileNotFoundError("Frozen evaluator spec or calibration file missing")

    spec = read_json(EVALUATOR_SPEC)
    cal = read_json(EVALUATOR_CAL)
    if float(spec.get("threshold", -1.0)) != 0.5:
        raise ValueError(f"Unexpected evaluator threshold in spec: {spec.get('threshold')}")
    if not bool(cal.get("passed", False)):
        raise ValueError(f"Frozen evaluator calibration did not pass: {cal}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    ckpt = torch.load(EVALUATOR_PT, map_location=device, weights_only=False)
    model = SmallUNet(in_ch=3, base=16).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    meta = {
        "ckpt_path": str(EVALUATOR_PT),
        "ckpt_sha256": sha256_file(EVALUATOR_PT),
        "spec_path": str(EVALUATOR_SPEC),
        "spec_sha256": sha256_file(EVALUATOR_SPEC),
        "calibration_path": str(EVALUATOR_CAL),
        "calibration_sha256": sha256_file(EVALUATOR_CAL),
        "threshold": 0.5,
        "calibration_metrics": cal.get("metrics"),
        "note": "Simulator-calibrated RGB segmentation proxy (544 frames recall=1.0, bbox-center P95=0.5px); not human ground truth in generation domain.",
    }
    return model, device, meta


def predict_frame_evaluator(model: SmallUNet, device: torch.device, frame_rgb: np.ndarray) -> dict[str, Any]:
    x = torch.from_numpy((frame_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1))[None].float().to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()
    pred_raw = (prob >= 0.5).astype(np.uint8)
    sel_mask, center, comps, bbox = select_largest_component(pred_raw)
    return {
        "mask": sel_mask,
        "raw_mask": pred_raw,
        "center": center,
        "bbox": bbox,
        "components": int(comps),
        "detected": center is not None,
    }


def _subset_metrics(
    frame_rows: list[dict[str, Any]],
    frames_idx: list[int],
    det_key: str,
    err_key: str,
    penalty: float = DIAGONAL,
) -> dict[str, Any]:
    n = len(frames_idx)
    valid_errs = []
    pen_errs = []
    det_count = 0
    for f in frames_idx:
        r = frame_rows[f]
        det = bool(r.get(det_key, False))
        err = r.get(err_key)
        if det and err is not None and np.isfinite(err):
            det_count += 1
            valid_errs.append(float(err))
            pen_errs.append(float(err))
        else:
            pen_errs.append(float(penalty))
    all_det = det_count == n
    return {
        "n_frames": n,
        "detected_count": det_count,
        "missing_count": n - det_count,
        "coverage": float(det_count) / float(n) if n > 0 else 0.0,
        "miss_rate": float(n - det_count) / float(n) if n > 0 else 1.0,
        "all_detected": all_det,
        "epe_penalized": float(np.mean(pen_errs)) if pen_errs else float(penalty),
        "epe_valid": float(np.mean(valid_errs)) if valid_errs else None,
        "max_err_penalized": float(np.max(pen_errs)) if pen_errs else float(penalty),
        "max_err_valid": float(np.max(valid_errs)) if valid_errs else None,
        "all_le_5px": bool(all_det and all(e <= 5.0 for e in valid_errs)),
        "all_le_10px": bool(all_det and all(e <= 10.0 for e in valid_errs)),
    }


def load_reference_targets_verified() -> dict[int, dict[str, Any]]:
    ref_json = OUT / "motion" / "reference_targets" / "all_reference_targets.json"
    if not file_ok(ref_json):
        raise FileNotFoundError(f"Reference targets missing: {ref_json}")
    raw = read_json(ref_json)
    out: dict[int, dict[str, Any]] = {int(k): v for k, v in raw.items()}

    # Cross-check against V2_COND/targets/all_targets.json if present
    v2_t_path = V2_COND / "targets" / "all_targets.json"
    if file_ok(v2_t_path):
        v2_t = read_json(v2_t_path)
        for s in range(16):
            c_our = np.asarray(out[s]["bbox_centers"], dtype=np.float64)
            c_old = np.asarray(v2_t[str(s)]["centers_all"], dtype=np.float64)
            if not np.allclose(c_our, c_old, atol=1e-5):
                raise ValueError(f"Scene {s} reference bbox centers mismatch vs V2_COND/targets")
    return out


def compute_scene_static_rois(video_manifest: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    kernel = np.ones((9, 9), np.uint8)
    rois: dict[int, np.ndarray] = {}
    for scene in range(16):
        union = np.zeros((256, 256), dtype=np.uint8)
        ref_side = OUT / "motion" / "reference_targets" / f"ref_scene{scene:02d}.sidecar.npz"
        if file_ok(ref_side):
            m = np.load(ref_side)["fg_union"].astype(np.uint8)
            union = np.maximum(union, (m > 0).astype(np.uint8))

        seen_sidecars = set()
        for it in video_manifest:
            if int(it["scene"]) != scene:
                continue
            sp = Path(it["sidecar_path"])
            if str(sp) in seen_sidecars or not file_ok(sp):
                continue
            seen_sidecars.add(str(sp))
            m = np.load(sp)["fg_union"].astype(np.uint8)
            if m.shape == (256, 256):
                union = np.maximum(union, (m > 0).astype(np.uint8))

        dilated = cv2.dilate(union, kernel, iterations=1)
        rois[scene] = (dilated == 0).astype(np.uint8)
        np.save(OUT / "score" / f"static_roi_scene{scene:02d}.npy", rois[scene])

    write_json(
        OUT / "score" / "static_roi_summary.json",
        {
            "definition": "Exterior of 9x9 dilated union of uncompressed simulator (robot + gripper + cube) foreground masks across all candidates and v2 reference of each scene.",
            "scene_static_fraction": {str(s): float(rois[s].mean()) for s in range(16)},
        },
    )
    return rois


def compute_background_flicker(vid: np.ndarray, roi_mask: np.ndarray) -> float | None:
    roi = roi_mask.astype(bool)
    if int(roi.sum()) == 0:
        return None
    vf = vid.astype(np.float32)
    diffs = [float(np.abs(vf[f] - vf[f - 1])[roi].mean()) for f in range(1, vf.shape[0])]
    return float(np.mean(diffs)) if diffs else None


def score_single_video(
    item: dict[str, Any],
    ref_target: dict[str, Any],
    roi_mask: np.ndarray,
    eval_model: SmallUNet,
    eval_device: torch.device,
) -> dict[str, Any]:
    vpath = Path(item["video_path"])
    spath = Path(item["sidecar_path"])
    vid = decode_video(vpath)
    if vid.shape != (FRAMES, 256, 256, 3):
        raise ValueError(f"Bad video shape {vid.shape} for {vpath}")
    if not file_ok(spath):
        raise FileNotFoundError(f"Uncompressed control sidecar missing: {spath}")

    side = np.load(spath)
    self_bbox_centers = side["bbox_centers"]  # [17, 2]
    self_audit_ctrl = side["audit_ctrl_centers"]  # [17, 2]
    self_strict_ctrl = side["strict_ctrl_centers"]  # [17, 2]

    ref_bbox_centers = ref_target["bbox_centers"]
    ref_audit_ctrl = ref_target["audit_ctrl_centers"]
    ref_strict_ctrl = ref_target["strict_ctrl_centers"]

    frame_rows = []
    for f in range(FRAMES):
        ev = predict_frame_evaluator(eval_model, eval_device, vid[f])
        pr_c = ev["center"]
        pr_det = bool(ev["detected"])
        pr_comps = int(ev["components"])
        pr_bbox = ev["bbox"]

        aud_c, aud_n = audit_hsv_detector(vid[f])
        str_c, str_n = strict_hsv_detector(vid[f])

        t_ref = ref_bbox_centers[f]
        t_self = self_bbox_centers[f].tolist() if np.all(np.isfinite(self_bbox_centers[f])) else None

        t_aud_ref = ref_audit_ctrl[f]
        t_aud_self = self_audit_ctrl[f].tolist() if np.all(np.isfinite(self_audit_ctrl[f])) else None

        t_str_ref = ref_strict_ctrl[f]
        t_str_self = self_strict_ctrl[f].tolist() if np.all(np.isfinite(self_strict_ctrl[f])) else None

        frame_rows.append(
            {
                "frame": f,
                "is_anchor": f in ANCHOR_FRAMES,
                "eval_detected": pr_det,
                "eval_center": pr_c.tolist() if pr_c is not None else None,
                "eval_bbox": pr_bbox,
                "eval_components": pr_comps,
                "target_ref_bbox_center": t_ref,
                "target_self_bbox_center": t_self,
                "err_eval_to_ref": point_err(pr_c, t_ref),
                "err_eval_to_self": point_err(pr_c, t_self),
                # Independent HSV trackers (centroid-based, never mixed with bbox-center primary metric)
                "audit_hsv_detected": aud_c is not None,
                "audit_hsv_center": aud_c.tolist() if aud_c is not None else None,
                "audit_hsv_components": int(aud_n),
                "err_audit_hsv_to_ref": point_err(aud_c, t_aud_ref),
                "err_audit_hsv_to_self": point_err(aud_c, t_aud_self),
                "strict_hsv_detected": str_c is not None,
                "strict_hsv_center": str_c.tolist() if str_c is not None else None,
                "strict_hsv_components": int(str_n),
                "err_strict_hsv_to_ref": point_err(str_c, t_str_ref),
                "err_strict_hsv_to_self": point_err(str_c, t_str_self),
            }
        )

    anchor_eval_ref = _subset_metrics(frame_rows, ANCHOR_FRAMES, "eval_detected", "err_eval_to_ref")
    nonanchor_eval_ref = _subset_metrics(frame_rows, NONANCHOR_FRAMES, "eval_detected", "err_eval_to_ref")
    all_eval_ref = _subset_metrics(frame_rows, list(range(FRAMES)), "eval_detected", "err_eval_to_ref")

    anchor_eval_self = _subset_metrics(frame_rows, ANCHOR_FRAMES, "eval_detected", "err_eval_to_self")
    nonanchor_eval_self = _subset_metrics(frame_rows, NONANCHOR_FRAMES, "eval_detected", "err_eval_to_self")
    all_eval_self = _subset_metrics(frame_rows, list(range(FRAMES)), "eval_detected", "err_eval_to_self")

    anchor_audit_ref = _subset_metrics(frame_rows, ANCHOR_FRAMES, "audit_hsv_detected", "err_audit_hsv_to_ref")
    nonanchor_audit_ref = _subset_metrics(frame_rows, NONANCHOR_FRAMES, "audit_hsv_detected", "err_audit_hsv_to_ref")
    anchor_audit_self = _subset_metrics(frame_rows, ANCHOR_FRAMES, "audit_hsv_detected", "err_audit_hsv_to_self")
    nonanchor_audit_self = _subset_metrics(frame_rows, NONANCHOR_FRAMES, "audit_hsv_detected", "err_audit_hsv_to_self")

    anchor_strict_ref = _subset_metrics(frame_rows, ANCHOR_FRAMES, "strict_hsv_detected", "err_strict_hsv_to_ref")
    nonanchor_strict_ref = _subset_metrics(frame_rows, NONANCHOR_FRAMES, "strict_hsv_detected", "err_strict_hsv_to_ref")
    anchor_strict_self = _subset_metrics(frame_rows, ANCHOR_FRAMES, "strict_hsv_detected", "err_strict_hsv_to_self")
    nonanchor_strict_self = _subset_metrics(frame_rows, NONANCHOR_FRAMES, "strict_hsv_detected", "err_strict_hsv_to_self")

    # Placement accuracy at frames 13 and 16 (separate from hold stability)
    f13_det = bool(frame_rows[13]["eval_detected"])
    f13_err = frame_rows[13]["err_eval_to_ref"]
    f16_det = bool(frame_rows[16]["eval_detected"])
    f16_err = frame_rows[16]["err_eval_to_ref"]

    # Hold stability: frames 13..16 max drift relative to predicted frame 13
    hold_frames = [13, 14, 15, 16]
    hold_all_det = all(bool(frame_rows[f]["eval_detected"]) and frame_rows[f]["eval_center"] is not None for f in hold_frames)
    if hold_all_det:
        p13 = np.asarray(frame_rows[13]["eval_center"], dtype=np.float64)
        hold_drift = float(
            max(np.linalg.norm(np.asarray(frame_rows[f]["eval_center"], dtype=np.float64) - p13) for f in hold_frames)
        )
        hold_stable_3px = bool(hold_drift <= 3.0)
        hold_drift_pen = hold_drift
    else:
        hold_drift = None
        hold_stable_3px = False
        hold_drift_pen = float(DIAGONAL)

    flicker = compute_background_flicker(vid, roi_mask)
    comp_gt1_rate = float(sum(1 for r in frame_rows if r["eval_components"] > 1)) / float(FRAMES)

    summary = {
        "score_status": "ok",
        # Primary metric & coverage
        "anchor_epe_ref_penalized": anchor_eval_ref["epe_penalized"],
        "anchor_epe_ref_valid": anchor_eval_ref["epe_valid"],
        "anchor_max_ref_penalized": anchor_eval_ref["max_err_penalized"],
        "anchor_max_ref_valid": anchor_eval_ref["max_err_valid"],
        "anchor_detected_count": anchor_eval_ref["detected_count"],
        "anchor_coverage": anchor_eval_ref["coverage"],
        "anchor_miss_rate": anchor_eval_ref["miss_rate"],
        "anchor_all_le_5px_ref": anchor_eval_ref["all_le_5px"],
        "anchor_all_le_10px_ref": anchor_eval_ref["all_le_10px"],
        # Non-anchor to common v2 reference
        "nonanchor_epe_ref_penalized": nonanchor_eval_ref["epe_penalized"],
        "nonanchor_epe_ref_valid": nonanchor_eval_ref["epe_valid"],
        "nonanchor_coverage": nonanchor_eval_ref["coverage"],
        # All frames to common v2 reference
        "all_frames_epe_ref_penalized": all_eval_ref["epe_penalized"],
        "all_frames_epe_ref_valid": all_eval_ref["epe_valid"],
        "all_frames_coverage": all_eval_ref["coverage"],
        "all_frames_miss_rate": all_eval_ref["miss_rate"],
        # Self-control metrics
        "anchor_epe_self_penalized": anchor_eval_self["epe_penalized"],
        "anchor_epe_self_valid": anchor_eval_self["epe_valid"],
        "anchor_all_le_5px_self": anchor_eval_self["all_le_5px"],
        "anchor_all_le_10px_self": anchor_eval_self["all_le_10px"],
        "nonanchor_epe_self_penalized": nonanchor_eval_self["epe_penalized"],
        "nonanchor_epe_self_valid": nonanchor_eval_self["epe_valid"],
        "all_frames_epe_self_penalized": all_eval_self["epe_penalized"],
        "all_frames_epe_self_valid": all_eval_self["epe_valid"],
        # Placement vs Hold stability (separate!)
        "frame13_detected": f13_det,
        "frame13_epe_ref_valid": f13_err if f13_det else None,
        "frame13_epe_ref_penalized": float(f13_err) if (f13_det and f13_err is not None) else float(DIAGONAL),
        "frame16_detected": f16_det,
        "frame16_epe_ref_valid": f16_err if f16_det else None,
        "frame16_epe_ref_penalized": float(f16_err) if (f16_det and f16_err is not None) else float(DIAGONAL),
        "hold_13_16_all_detected": hold_all_det,
        "hold_drift_13_16_max_px": hold_drift,
        "hold_drift_13_16_penalized_px": hold_drift_pen,
        "hold_stable_le_3px": hold_stable_3px,
        # Independent HSV metrics
        "audit_hsv_anchor_epe_ref_penalized": anchor_audit_ref["epe_penalized"],
        "audit_hsv_anchor_epe_ref_valid": anchor_audit_ref["epe_valid"],
        "audit_hsv_anchor_coverage": anchor_audit_ref["coverage"],
        "audit_hsv_anchor_all_le_5px_ref": anchor_audit_ref["all_le_5px"],
        "audit_hsv_anchor_all_le_10px_ref": anchor_audit_ref["all_le_10px"],
        "audit_hsv_nonanchor_epe_ref_penalized": nonanchor_audit_ref["epe_penalized"],
        "audit_hsv_nonanchor_epe_ref_valid": nonanchor_audit_ref["epe_valid"],
        "audit_hsv_anchor_epe_self_penalized": anchor_audit_self["epe_penalized"],
        "audit_hsv_anchor_epe_self_valid": anchor_audit_self["epe_valid"],
        "audit_hsv_nonanchor_epe_self_penalized": nonanchor_audit_self["epe_penalized"],
        "audit_hsv_nonanchor_epe_self_valid": nonanchor_audit_self["epe_valid"],
        "strict_hsv_anchor_epe_ref_penalized": anchor_strict_ref["epe_penalized"],
        "strict_hsv_anchor_epe_ref_valid": anchor_strict_ref["epe_valid"],
        "strict_hsv_anchor_coverage": anchor_strict_ref["coverage"],
        "strict_hsv_anchor_all_le_5px_ref": anchor_strict_ref["all_le_5px"],
        "strict_hsv_anchor_all_le_10px_ref": anchor_strict_ref["all_le_10px"],
        "strict_hsv_nonanchor_epe_ref_penalized": nonanchor_strict_ref["epe_penalized"],
        "strict_hsv_nonanchor_epe_ref_valid": nonanchor_strict_ref["epe_valid"],
        "strict_hsv_anchor_epe_self_penalized": anchor_strict_self["epe_penalized"],
        "strict_hsv_anchor_epe_self_valid": anchor_strict_self["epe_valid"],
        "strict_hsv_nonanchor_epe_self_penalized": nonanchor_strict_self["epe_penalized"],
        "strict_hsv_nonanchor_epe_self_valid": nonanchor_strict_self["epe_valid"],
        # Visual stability & segmentation anomaly proxies
        "background_flicker_l1": flicker,
        "eval_components_gt1_rate": comp_gt1_rate,
        "subsets_detail": {
            "anchor_eval_ref": anchor_eval_ref,
            "nonanchor_eval_ref": nonanchor_eval_ref,
            "anchor_eval_self": anchor_eval_self,
            "nonanchor_eval_self": nonanchor_eval_self,
            "anchor_audit_ref": anchor_audit_ref,
            "anchor_strict_ref": anchor_strict_ref,
        },
    }
    return {"frames": frame_rows, "summary": summary}


def _failed_video_score_summary(reason: str) -> dict[str, Any]:
    return {
        "score_status": reason,
        "anchor_epe_ref_penalized": float(DIAGONAL),
        "anchor_epe_ref_valid": None,
        "anchor_max_ref_penalized": float(DIAGONAL),
        "anchor_max_ref_valid": None,
        "anchor_detected_count": 0,
        "anchor_coverage": 0.0,
        "anchor_miss_rate": 1.0,
        "anchor_all_le_5px_ref": False,
        "anchor_all_le_10px_ref": False,
        "nonanchor_epe_ref_penalized": float(DIAGONAL),
        "nonanchor_epe_ref_valid": None,
        "nonanchor_coverage": 0.0,
        "all_frames_epe_ref_penalized": float(DIAGONAL),
        "all_frames_epe_ref_valid": None,
        "all_frames_coverage": 0.0,
        "all_frames_miss_rate": 1.0,
        "anchor_epe_self_penalized": float(DIAGONAL),
        "anchor_epe_self_valid": None,
        "anchor_all_le_5px_self": False,
        "anchor_all_le_10px_self": False,
        "nonanchor_epe_self_penalized": float(DIAGONAL),
        "nonanchor_epe_self_valid": None,
        "all_frames_epe_self_penalized": float(DIAGONAL),
        "all_frames_epe_self_valid": None,
        "frame13_detected": False,
        "frame13_epe_ref_valid": None,
        "frame13_epe_ref_penalized": float(DIAGONAL),
        "frame16_detected": False,
        "frame16_epe_ref_valid": None,
        "frame16_epe_ref_penalized": float(DIAGONAL),
        "hold_13_16_all_detected": False,
        "hold_drift_13_16_max_px": None,
        "hold_drift_13_16_penalized_px": float(DIAGONAL),
        "hold_stable_le_3px": False,
        "audit_hsv_anchor_epe_ref_penalized": float(DIAGONAL),
        "audit_hsv_anchor_epe_ref_valid": None,
        "audit_hsv_anchor_coverage": 0.0,
        "audit_hsv_anchor_all_le_5px_ref": False,
        "audit_hsv_anchor_all_le_10px_ref": False,
        "audit_hsv_nonanchor_epe_ref_penalized": float(DIAGONAL),
        "audit_hsv_nonanchor_epe_ref_valid": None,
        "audit_hsv_anchor_epe_self_penalized": float(DIAGONAL),
        "audit_hsv_anchor_epe_self_valid": None,
        "audit_hsv_nonanchor_epe_self_penalized": float(DIAGONAL),
        "audit_hsv_nonanchor_epe_self_valid": None,
        "strict_hsv_anchor_epe_ref_penalized": float(DIAGONAL),
        "strict_hsv_anchor_epe_ref_valid": None,
        "strict_hsv_anchor_coverage": 0.0,
        "strict_hsv_anchor_all_le_5px_ref": False,
        "strict_hsv_anchor_all_le_10px_ref": False,
        "strict_hsv_nonanchor_epe_ref_penalized": float(DIAGONAL),
        "strict_hsv_nonanchor_epe_ref_valid": None,
        "strict_hsv_anchor_epe_self_penalized": float(DIAGONAL),
        "strict_hsv_anchor_epe_self_valid": None,
        "strict_hsv_nonanchor_epe_self_penalized": float(DIAGONAL),
        "strict_hsv_nonanchor_epe_self_valid": None,
        "background_flicker_l1": None,
        "eval_components_gt1_rate": None,
    }


def make_blind_evaluation_artifacts(
    manifest_items: list[dict[str, Any]],
    ref_targets: dict[int, dict[str, Any]],
    eval_model: SmallUNet,
    eval_device: torch.device,
) -> dict[str, Any]:
    """Pre-specified blind evaluation:
    scenes 0, 5, 10; 4 groups; first motion seed (motion_index == 0); Wan seeds 43, 44 -> 24 videos.
    Frames 0, 4, 8, 12, 13, 16 per video.
    Saves clean original crops/strips AND predicted mask/bbox overlays with group names hidden.
    """
    selected = [
        it
        for it in manifest_items
        if int(it["scene"]) in BLIND_SCENES
        and int(it["motion_index"]) == 0
        and int(it["wan_seed"]) in WAN_SEEDS
        and it["group"] in GROUPS
    ]
    selected.sort(key=lambda r: (int(r["scene"]), int(r["wan_seed"]), GROUPS.index(r["group"])))
    if len(selected) != 24:
        raise ValueError(f"Expected 24 blind evaluation videos, found {len(selected)}")

    rng = random.Random(BOOT_SEED)
    shuffled = list(selected)
    rng.shuffle(shuffled)

    deblind_rows = []
    crop_half = 48  # 96x96 zoomed crop around target cube center

    for b_idx, it in enumerate(shuffled):
        blind_id = f"blind_{b_idx:02d}"
        scn = int(it["scene"])
        ref_t = ref_targets[scn]["bbox_centers"]
        vpath = Path(it["video_path"])

        raw_strip_frames = []
        crop_strip_frames = []
        overlay_strip_frames = []

        is_valid_vid = bool(it.get("count") in ("generated", "reused") and file_ok(vpath))
        if is_valid_vid:
            vid = decode_video(vpath)
        else:
            vid = np.zeros((FRAMES, 256, 256, 3), dtype=np.uint8)

        frame_meta = []
        for f in ANCHOR_FRAMES:
            rgb = vid[f].copy()
            ev = predict_frame_evaluator(eval_model, eval_device, rgb) if is_valid_vid else {
                "mask": np.zeros((256, 256), dtype=np.uint8),
                "center": None,
                "bbox": None,
                "components": 0,
                "detected": False,
            }
            tc = ref_t[f]
            pc = ev["center"]
            pb = ev["bbox"]
            mask = ev["mask"]

            # 1. Clean full frame with frame index only (no group name)
            clean_full = rgb.copy()
            cv2.putText(clean_full, f"{blind_id} f{f}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            raw_strip_frames.append(clean_full)

            # 2. Clean 96x96 crop centered at reference target center (with predicted mask & bbox crop below it)
            cx = int(round(tc[0])) if tc is not None else 128
            cy = int(round(tc[1])) if tc is not None else 128
            x0 = max(0, min(256 - 2 * crop_half, cx - crop_half))
            y0 = max(0, min(256 - 2 * crop_half, cy - crop_half))
            x1 = x0 + 2 * crop_half
            y1 = y0 + 2 * crop_half
            clean_crop = rgb[y0:y1, x0:x1].copy()

            ov_full = rgb.copy()
            # Tint predicted mask in red/yellow blend
            if mask.sum() > 0:
                tint = np.zeros_like(ov_full)
                tint[..., 0] = 255
                tint[..., 1] = 60
                m_bool = mask > 0
                ov_full[m_bool] = (0.65 * ov_full[m_bool] + 0.35 * tint[m_bool]).astype(np.uint8)
            if pb is not None:
                cv2.rectangle(ov_full, (int(pb[0]), int(pb[1])), (int(pb[2]), int(pb[3])), (255, 80, 0), 1)
            if tc is not None:
                cv2.circle(ov_full, (int(round(tc[0])), int(round(tc[1]))), 3, (0, 255, 0), -1)
            if pc is not None:
                cv2.circle(ov_full, (int(round(pc[0])), int(round(pc[1]))), 3, (255, 0, 0), -1)
            cv2.putText(ov_full, f"{blind_id} f{f} c={ev['components']}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
            overlay_strip_frames.append(ov_full)

            ov_crop = ov_full[y0:y1, x0:x1].copy()
            pair_crop = np.concatenate([clean_crop, ov_crop], axis=0)  # [192, 96, 3] top=raw crop, bottom=overlay crop
            crop_strip_frames.append(pair_crop)

            frame_meta.append(
                {
                    "frame": f,
                    "crop_box_xyxy": [x0, y0, x1, y1],
                    "target_ref_center": tc,
                    "pred_center": pc.tolist() if pc is not None else None,
                    "pred_bbox": pb,
                    "detected": bool(ev["detected"]),
                    "components": int(ev["components"]),
                    "epe_to_ref_px": point_err(pc, tc),
                }
            )

        raw_strip = np.concatenate(raw_strip_frames, axis=1)
        ov_strip = np.concatenate(overlay_strip_frames, axis=1)
        crop_strip = np.concatenate(crop_strip_frames, axis=1)

        raw_out = OUT / "score" / "blind" / "crops" / f"{blind_id}_scene{scn:02d}_rgb_strip.png"
        crop_out = OUT / "score" / "blind" / "crops" / f"{blind_id}_scene{scn:02d}_crop_pair_strip.png"
        ov_out = OUT / "score" / "blind" / "overlays" / f"{blind_id}_scene{scn:02d}_overlay_strip.png"

        cv2.imwrite(str(raw_out), cv2.cvtColor(raw_strip, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(crop_out), cv2.cvtColor(crop_strip, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(ov_out), cv2.cvtColor(ov_strip, cv2.COLOR_RGB2BGR))

        deblind_rows.append(
            {
                "blind_id": blind_id,
                "video_id": it["video_id"],
                "motion_id": it["motion_id"],
                "group": it["group"],
                "basis_kind": it["basis_kind"],
                "mode": it["mode"],
                "scene": scn,
                "motion_index": int(it["motion_index"]),
                "motion_seed": int(it["motion_seed"]),
                "wan_seed": int(it["wan_seed"]),
                "rgb_strip_path": str(raw_out),
                "crop_pair_strip_path": str(crop_out),
                "overlay_strip_path": str(ov_out),
                "frames": frame_meta,
            }
        )

    summary = {
        "total_blind_videos": len(deblind_rows),
        "scenes": BLIND_SCENES,
        "groups_hidden_in_filenames": True,
        "shuffle_seed": BOOT_SEED,
        "frames_per_video": ANCHOR_FRAMES,
        "legend": {
            "green_dot": "common v2 reference bbox center (from uncompressed simulator instance mask)",
            "red_dot": "predicted largest-component bbox center (frozen SmallUNet threshold 0.5)",
            "orange_box_and_tint": "predicted largest-component mask and bounding box",
            "component_count_note": "c>1 indicates multiple connected components in raw thresholded mask (segmentation anomaly proxy, e.g. gripper occlusion splitting mask; not direct proof of duplicate objects).",
        },
        "deblind_key": deblind_rows,
    }
    write_json(OUT / "score" / "blind" / "deblind_key.json", summary)
    return summary


def run_all_scoring(video_manifest: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    ensure_eval_dirs()
    manifest_path = OUT / "manifests" / "video_manifest.json"
    if video_manifest is None:
        if not file_ok(manifest_path):
            raise FileNotFoundError(f"Video manifest missing: {manifest_path}")
        items = read_json(manifest_path)
    else:
        items = [dict(x) for x in video_manifest]

    if len(items) != 256:
        raise ValueError(f"Expected 256 video manifest rows, got {len(items)}")

    eval_model, eval_device, eval_meta = load_frozen_evaluator()
    write_json(OUT / "score" / "evaluator_metadata.json", eval_meta)

    ref_targets = load_reference_targets_verified()
    rois = compute_scene_static_rois(items)

    results = []
    for idx, it in enumerate(items):
        vid = it["video_id"]
        scn = int(it["scene"])
        frame_out = OUT / "score" / "frames" / f"{vid}.json"

        base_info = {
            k: it.get(k)
            for k in [
                "index",
                "video_id",
                "motion_id",
                "group",
                "group_slug",
                "basis_kind",
                "mode",
                "scene",
                "motion_index",
                "motion_seed",
                "wan_seed",
                "motion_status",
                "motion_ok",
                "status",
                "count",
                "failure_type",
                "q_path",
                "control_path",
                "sidecar_path",
                "video_path",
                "video_sha256",
                "control_sha256",
            ]
        }

        if it.get("count") == "failed" or not file_ok(it["video_path"]):
            reason = it.get("failure_type") or "failed_or_missing_video"
            fail_sum = _failed_video_score_summary(reason)
            rec = {**base_info, **fail_sum}
            write_json(frame_out, {"item": base_info, "frames": [], "summary": fail_sum})
            results.append(rec)
            continue

        # Check cached frame score integrity
        if file_ok(frame_out):
            try:
                cached = read_json(frame_out)
                c_item = cached.get("item") or {}
                if (
                    c_item.get("video_sha256") == it.get("video_sha256")
                    and c_item.get("control_sha256") == it.get("control_sha256")
                    and cached.get("summary", {}).get("score_status") == "ok"
                    and len(cached.get("frames", [])) == FRAMES
                ):
                    results.append({**base_info, **cached["summary"]})
                    continue
            except Exception:
                pass

        update_status_eval("scoring_videos", done=idx, total=len(items), current_video_id=vid)
        try:
            scored = score_single_video(it, ref_targets[scn], rois[scn], eval_model, eval_device)
            write_json(frame_out, {"item": base_info, "frames": scored["frames"], "summary": scored["summary"]})
            rec = {**base_info, **scored["summary"]}
            results.append(rec)
            if (idx + 1) % 16 == 0 or idx == len(items) - 1:
                append_eval_log(
                    f"scored [{idx+1}/256] {vid} epe_ref_pen={scored['summary']['anchor_epe_ref_penalized']:.2f}"
                )
        except Exception as e:
            fail_sum = _failed_video_score_summary("scoring_exception")
            fail_sum["error"] = str(e)
            fail_sum["traceback"] = traceback.format_exc()
            rec = {**base_info, **fail_sum}
            write_json(frame_out, {"item": base_info, "frames": [], "summary": fail_sum})
            results.append(rec)
            append_eval_log(f"score error [{idx+1}/256] {vid}: {e}")

    write_json(OUT / "score" / "per_video_scores.json", results)

    update_status_eval("generating_blind_overlays", total_blind=24)
    blind_summary = make_blind_evaluation_artifacts(items, ref_targets, eval_model, eval_device)
    update_status_eval("scoring_done", total_scored=len(results))
    return {
        "per_video_scores": results,
        "evaluator_meta": eval_meta,
        "blind_summary": blind_summary,
    }
