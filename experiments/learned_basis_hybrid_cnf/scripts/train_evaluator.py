#!/usr/bin/env python3
"""Train and calibrate the frozen RGB segmenter (SmallUNet) used for scoring.

Consolidates ``train_and_calibrate_evaluator`` from the original
``video_task_validation_20261114.py``. Trains a small 3-level U-Net for 2000
steps on the train split (episodes 0..95) and calibrates it on episodes 96..127
of the base pick-and-place dataset, producing the frozen checkpoint, spec, and
calibration artifacts consumed by ``learned_basis_hybrid_cnf.scoring``.

Outputs are written next to ``CNF_V2_EVALUATOR_PT`` (spec to
``CNF_V2_EVALUATOR_SPEC``, calibration to ``CNF_V2_EVALUATOR_CAL``).
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learned_basis_hybrid_cnf.basis import FRAMES
from learned_basis_hybrid_cnf.config import (
    EVALUATOR_CAL,
    EVALUATOR_PT,
    EVALUATOR_SPEC,
    TRAIN_DS,
    ConfigError,
)

CUBE_COLOR = np.array([255, 120, 20], dtype=np.uint8)
EVAL_SEED = 20261114


class SmallUNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 16):
        super().__init__()

        def block(ic, oc):
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

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        b = self.b(self.p3(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def _dice_bce_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3))
    den = (p + target).sum(dim=(1, 2, 3)) + 1e-6
    return bce + (1 - (num / den).mean())


def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, Path):
        return str(x)
    return x


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _bbox_center(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float32)


def _select_largest_component(mask: np.ndarray):
    m = (mask > 0).astype(np.uint8)
    ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    comps = int(max(0, ncomp - 1))
    if comps == 0:
        return np.zeros_like(m), None, 0
    areas = [(k, int(stats[k, cv2.CC_STAT_AREA])) for k in range(1, ncomp)]
    k_best = max(areas, key=lambda x: x[1])[0]
    sel = (labels == k_best).astype(np.uint8)
    return sel, _bbox_center(sel), comps


def _point_err(p, t):
    if p is None or t is None:
        return None
    return float(np.linalg.norm(np.asarray(p, float) - np.asarray(t, float)))


def _render_train_episodes(episode_ids: list[int]):
    from cnf_motion.scene import build_scene, disconnect_scene, render_trajectory

    if TRAIN_DS is None:
        raise ConfigError("train_evaluator requires TRAIN_DS to be configured.")
    meta = json.loads((TRAIN_DS / "metadata.json").read_text(encoding="utf-8"))
    data = {}
    robot, cube, table = build_scene()
    try:
        for ep in episode_ids:
            q = np.load(TRAIN_DS / "q" / f"{ep:04d}.npy")
            rgb, control, _ = render_trajectory(robot, cube, table, q, np.array(meta[ep]["cube_xy"]), width=256)
            masks = np.stack(
                [np.all(control[f] == CUBE_COLOR.reshape(1, 1, 3), axis=-1).astype(np.uint8) for f in range(FRAMES)]
            )
            data[ep] = (rgb, masks)
            print(f"rendered episode {ep}", flush=True)
    finally:
        disconnect_scene()
    return data


def train_and_calibrate_evaluator(force: bool = False) -> bool:
    if EVALUATOR_PT is None or EVALUATOR_SPEC is None or EVALUATOR_CAL is None:
        raise ConfigError("train_evaluator requires EVALUATOR_PT, EVALUATOR_SPEC and EVALUATOR_CAL to be configured.")
    ev = EVALUATOR_PT.parent
    ev.mkdir(parents=True, exist_ok=True)

    spec = {
        "architecture": "SmallUNet(in_ch=3,base=16),3 levels+bottleneck",
        "threshold": 0.5,
        "loss": "BCEWithLogits + (1-Dice)",
        "optimizer": "Adam(lr=1e-3)",
        "scheduler": "none",
        "steps": 2000,
        "batch": 16,
        "seed": EVAL_SEED,
        "split": {"fit": list(range(0, 96)), "calibration": list(range(96, 128))},
        "augmentation": "RGB only: brightness U(0.8,1.2), per-channel U(0.85,1.15), optional vertical shadow gradient U(0.7,1.0); labels unchanged",
        "component_criteria": "binarize>=0.5; if multiple connected components use largest-area component; report component count. component_count>1 treated as segmentation anomaly proxy (occlusion can split same object, not evidence of multiple cubes). Same post-processing used in calibration, sensitivity, and generation-domain scoring.",
        "position": "bbox center of selected (largest-component) predicted mask",
        "frozen_weight_selection": "fixed final step 2000",
    }
    _write_json(EVALUATOR_SPEC, spec)

    if EVALUATOR_PT.exists() and EVALUATOR_CAL.exists():
        try:
            passed = json.loads(EVALUATOR_CAL.read_text()).get("passed", False)
            print(f"evaluator cache hit: passed={passed}; no retraining", flush=True)
            return passed
        except Exception:
            print("evaluator cache unreadable, retraining", flush=True)
    elif EVALUATOR_PT.exists() and not force:
        raise RuntimeError(
            f"evaluator checkpoint already exists at {EVALUATOR_PT} but calibration is missing; "
            "use --force to overwrite, or remove the checkpoint first."
        )

    fit_eps = list(range(0, 96))
    cal_eps = list(range(96, 128))
    fit_data = _render_train_episodes(fit_eps)
    cal_data = _render_train_episodes(cal_eps)
    _write_json(ev / "split.json", {"fit": fit_eps, "calibration": cal_eps})

    random.seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    fit_samples = [(ep, f) for ep in fit_eps for f in range(FRAMES)]
    rng = np.random.default_rng(EVAL_SEED)

    def aug(img):
        img = img.copy()
        img = np.clip(img * rng.uniform(0.8, 1.2), 0, 1)
        for c in range(3):
            img[..., c] = np.clip(img[..., c] * rng.uniform(0.85, 1.15), 0, 1)
        if rng.random() < 0.5:
            y = np.linspace(rng.uniform(0.7, 1.0), rng.uniform(0.7, 1.0), img.shape[0], dtype=np.float32)
            if rng.random() < 0.5:
                y = y[::-1]
            img = np.clip(img * y[:, None, None], 0, 1)
        return img

    log = []
    model.train()
    for step in range(1, 2001):
        idx = rng.integers(0, len(fit_samples), size=16)
        xs, ys = [], []
        for i in idx:
            ep, f = fit_samples[int(i)]
            rgb, msk = fit_data[ep]
            im = aug(rgb[f].astype(np.float32) / 255.0)
            xs.append(torch.from_numpy(im.transpose(2, 0, 1)).float())
            ys.append(torch.from_numpy(msk[f][None].astype(np.float32)))
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        loss = _dice_bce_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0 or step == 2000:
            rec = {"step": step, "loss": float(loss.item()), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            log.append(rec)
            print(f"eval train step {step}/2000 loss={loss.item():.4f}", flush=True)
    _write_json(ev / "train_log.json", log)

    torch.save({"model": model.state_dict(), "spec": spec, "step": 2000}, EVALUATOR_PT)

    # Calibration
    model.eval()
    cal_rows = []
    with torch.no_grad():
        for ep in cal_eps:
            rgb, msk = cal_data[ep]
            for f in range(FRAMES):
                im = rgb[f].astype(np.float32) / 255.0
                x = torch.from_numpy(im.transpose(2, 0, 1))[None].float().to(device)
                prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                pred_raw = (prob >= 0.5).astype(np.uint8)
                pred, pr_c, comps = _select_largest_component(pred_raw)
                gt = msk[f]
                inter = float((pred & gt).sum())
                union = float((pred | gt).sum())
                iou = inter / union if union > 0 else (1.0 if gt.sum() == 0 else 0.0)
                gt_c = _bbox_center(gt)
                cal_rows.append(
                    {
                        "episode": ep,
                        "frame": f,
                        "iou": iou,
                        "detected": pr_c is not None,
                        "components": comps,
                        "pos_error_px": _point_err(pr_c, gt_c),
                        "gt_visible": gt_c is not None,
                        "gt_center": gt_c.tolist() if gt_c is not None else None,
                        "pred_center": pr_c.tolist() if pr_c is not None else None,
                    }
                )
    _write_json(ev / "calibration_rows.json", cal_rows)

    valid = [r for r in cal_rows if r["gt_visible"]]
    errs = [r["pos_error_px"] for r in valid if r["pos_error_px"] is not None]
    recall = sum(1 for r in valid if r["detected"]) / len(valid) if valid else 0.0
    p95 = float(np.percentile(errs, 95)) if errs else None
    metrics = {
        "recall": recall,
        "miss_rate": 1 - recall,
        "pos_error_p95_px": p95,
        "pos_error_mean_px": float(np.mean(errs)) if errs else None,
        "iou_mean": float(np.mean([r["iou"] for r in valid])) if valid else None,
        "n_valid": len(valid),
    }
    _write_json(ev / "calibration_metrics.json", metrics)

    # Perturbation sensitivity
    sens = []
    with torch.no_grad():
        for ep in cal_eps:
            rgb, msk = cal_data[ep]
            for f in range(FRAMES):
                base = rgb[f].astype(np.float32) / 255.0
                gt_c = _bbox_center(msk[f])
                if gt_c is None:
                    continue
                for name, img in [
                    ("bright_down", np.clip(base * 0.8, 0, 1)),
                    ("bright_up", np.clip(base * 1.2, 0, 1)),
                    ("color_shift", np.clip(base * np.array([1.15, 0.9, 1.0]), 0, 1)),
                ]:
                    x = torch.from_numpy(img.transpose(2, 0, 1))[None].float().to(device)
                    pred_raw = (torch.sigmoid(model(x))[0, 0].cpu().numpy() >= 0.5).astype(np.uint8)
                    _, pr_c, comps = _select_largest_component(pred_raw)
                    sens.append(
                        {
                            "episode": ep,
                            "frame": f,
                            "perturb": name,
                            "detected": pr_c is not None,
                            "pos_error_px": _point_err(pr_c, gt_c),
                            "components": comps,
                        }
                    )
    ss = {}
    for name in ["bright_down", "bright_up", "color_shift"]:
        rs = [r for r in sens if r["perturb"] == name]
        es = [r["pos_error_px"] for r in rs if r["pos_error_px"] is not None]
        ss[name] = {
            "n": len(rs),
            "recall": sum(1 for r in rs if r["detected"]) / len(rs) if rs else None,
            "pos_error_p95_px": float(np.percentile(es, 95)) if es else None,
        }
    _write_json(ev / "calibration_sensitivity.json", ss)

    passed = (recall >= 0.99) and (p95 is not None and p95 <= 2.0)
    _write_json(
        EVALUATOR_CAL,
        {
            "passed": passed,
            "metrics": metrics,
            "criteria": {"recall_ge_0.99": recall >= 0.99, "p95_le_2px": p95 is not None and p95 <= 2.0},
            "note": "Frozen final-step weights; no hard-frame dropping.",
        },
    )
    print(f"calibration passed={passed} recall={recall} p95={p95}", flush=True)
    return passed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite an existing evaluator checkpoint")
    args = parser.parse_args()

    passed = train_and_calibrate_evaluator(force=args.force)
    if not passed:
        raise SystemExit("evaluator calibration failed")


if __name__ == "__main__":
    main()
