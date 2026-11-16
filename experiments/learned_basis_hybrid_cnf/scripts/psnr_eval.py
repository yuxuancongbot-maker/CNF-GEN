#!/usr/bin/env python3
"""Supplemental CPU PSNR evaluation of the 256 existing Wan videos (2026-11-14).

Fixed protocol (docs/learned_basis_hybrid_psnr_handoff_20261114.md):

* Primary target: the shared per-scene v2 joint trajectory rendered once with
  the existing ``cnf_motion.scene.render_trajectory`` RGB renderer.  This is
  simulator RGB, not the semantic control map and not a Wan GT replay.
* Auxiliary target: the original dataset ``pickplace256_eval/train/{scene}.mp4``
  decoded RGB.  Reported separately; never averaged together with the primary
  target.
* PSNR = 10*log10(255^2 / MSE), MSE accumulated in float64 over RGB pixels,
  sum first then log.  No upper clipping; zero MSE is marked separately.

The script is self-contained (no training, no Wan, no GPU) and writes all
artifacts under ``OUT/psnr/``.  It refuses to overwrite existing metrics or
videos: only files inside ``OUT/psnr/`` are written.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import imageio.v2 as iio
import numpy as np

from learned_basis_hybrid_cnf.config import (
    DOC_REPORT as _PSNR_DOC,
    EVAL_DS,
    OUT,
)

import cnf_motion.scene as _cnf_scene

OUT_MAIN = OUT
PSNR_OUT = OUT_MAIN / "psnr"
MANIFEST = OUT_MAIN / "manifests/video_manifest.json"
EVAL_Q_V2 = OUT_MAIN / "data/eval_q_v2.npz"
EVAL_META = EVAL_DS / "metadata.json"
REF_TARGET_DIR = OUT_MAIN / "motion/reference_targets"
SCENE_MODULE = Path(_cnf_scene.__file__)

FRAMES = 17
WIDTH = 256
CHANNELS = 3
ANCHORS = [0, 4, 8, 12, 13, 16]
NON_ANCHORS = [f for f in range(FRAMES) if f not in ANCHORS]
GROUPS = ["S-fixed", "H-fixed", "S-learned", "H-learned"]
SUBSETS = {
    "all17": list(range(FRAMES)),
    "excl_first16": list(range(1, FRAMES)),
    "anchors6": ANCHORS,
    "nonanchors11": NON_ANCHORS,
}
PRIMARY_CONTRASTS = [
    ("H-learned", "S-learned"),
    ("H-fixed", "S-fixed"),
    ("H-learned", "H-fixed"),
]
BOOT_N = 10000
BOOT_SEED = 20261114


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def frame_sse(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-frame float64 squared error sum over all RGB pixels."""
    diff = pred.astype(np.float64) - ref.astype(np.float64)
    return np.sum(diff * diff, axis=(1, 2, 3))


def pooled_mse_from_sse(sse: np.ndarray, frames: list[int]) -> float:
    pixels = len(frames) * WIDTH * WIDTH * CHANNELS
    return float(np.sum(sse[np.asarray(frames, dtype=int)]) / pixels)


def psnr_from_mse(mse: float) -> float | None:
    if mse <= 0.0:
        return None
    return float(10.0 * math.log10((255.0 * 255.0) / mse))


def metrics_from_sse(sse: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, frames in SUBSETS.items():
        mse = pooled_mse_from_sse(sse, frames)
        out[f"mse_{name}"] = mse
        out[f"psnr_{name}"] = psnr_from_mse(mse)
    per_frame_mse = sse / float(WIDTH * WIDTH * CHANNELS)
    per_frame_psnr = [
        psnr_from_mse(float(v)) if v > 0.0 else None for v in per_frame_mse
    ]
    out["per_frame_mse"] = per_frame_mse.tolist()
    out["per_frame_psnr"] = per_frame_psnr
    out["zero_mse_frames"] = [i for i, v in enumerate(per_frame_mse) if v <= 0.0]
    return out


def bbox_from_control(control: np.ndarray, cube_rgb=(255, 120, 20)) -> np.ndarray:
    mask = np.all(control == np.asarray(cube_rgb, dtype=np.uint8), axis=-1)
    bboxes = np.full((FRAMES, 4), -1, dtype=np.int32)
    for f in range(FRAMES):
        ys, xs = np.where(mask[f])
        if len(xs):
            bboxes[f] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return bboxes


def paired_bootstrap(a: list[float], b: list[float]) -> dict[str, Any]:
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(diffs)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(BOOT_N, n))
    boot_means = diffs[idx].mean(axis=1)
    return {
        "n_scenes": int(n),
        "mean_diff": float(diffs.mean()),
        "ci95": [
            float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)),
        ],
        "se": float(diffs.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        "improved_scenes": int(np.sum(diffs > 0.0)),
        "unchanged_scenes": int(np.sum(np.abs(diffs) <= 1e-12)),
        "higher_is_better": True,
        "n_boot": BOOT_N,
        "seed": BOOT_SEED,
        "per_scene_diffs": diffs.tolist(),
    }


def build_protocol(manifest: list[dict[str, Any]], q_data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return {
        "protocol": "learned_basis_hybrid_psnr_20261114",
        "created": now(),
        "scope": (
            "CPU-only supplemental PSNR of the 256 existing Wan videos; no retraining, "
            "no resampling, no Wan generation, no GPU."
        ),
        "inputs": {
            "video_manifest": str(MANIFEST),
            "video_manifest_sha256": sha256_file(MANIFEST),
            "eval_q_v2": str(EVAL_Q_V2),
            "eval_q_v2_sha256": sha256_file(EVAL_Q_V2),
            "eval_metadata": str(EVAL_META),
            "eval_metadata_sha256": sha256_file(EVAL_META),
            "scene_module": str(SCENE_MODULE),
            "scene_module_sha256": sha256_file(SCENE_MODULE),
            "n_manifest_entries": len(manifest),
            "groups": GROUPS,
            "n_scenes": int(len(np.unique([e["scene"] for e in manifest]))),
            "frames": FRAMES,
            "width": WIDTH,
        },
        "primary_reference": {
            "construction": (
                "For each scene render the common v2 q (eval_q_v2.npz) exactly once with "
                "cnf_motion.scene.render_trajectory(robot, cube, table, q, metadata[scene]['cube_xy'], "
                "width=256); use the returned RGB array (17x256x256x3 uint8)."
            ),
            "explicitly_not": [
                "semantic control image",
                "Wan-generated GT replay",
                "per-video frame alignment / translation / frame selection",
            ],
            "saved": "psnr/references/v2_render_rgb.npz (lossless, keys scene00..scene15)",
            "auxiliary_target": (
                "Decoded RGB of datasets/pickplace256_eval/train/{scene:04d}.mp4, one decode per scene. "
                "It uses the original q and is reported separately; never averaged with the primary target."
            ),
        },
        "metric": {
            "formula": "PSNR = 10*log10(255^2/MSE)",
            "mse_accumulation": "float64 squared error summed over all pixels of the selected frames, then divided by pixel count; log taken after pooling (MSE-first).",
            "clipping": "none; true zero MSE reported as null psnr with zero_mse_frames listing",
            "frame_subsets": {
                "all17": list(range(FRAMES)),
                "excl_first16": list(range(1, FRAMES)),
                "anchors6": ANCHORS,
                "nonanchors11": NON_ANCHORS,
            },
            "per_frame": "per-frame MSE and PSNR saved separately; per-frame PSNR means are not the pooled metric.",
            "primary_table": (
                "per-video pooled PSNR over all 17 frames; average the 4 seed videos within each scene, "
                "then average the 16 scenes equally."
            ),
            "also_reported": "dataset-level pooled PSNR from total SSE does not replace the scene-equal-weight table.",
            "direction": "higher is better",
        },
        "statistics": {
            "primary": "scene is the independent unit (n=16)",
            "contrasts": [
                {"name": f"{a} - {b}", "a": a, "b": b}
                for a, b in PRIMARY_CONTRASTS
            ],
            "bootstrap": {"n_boot": BOOT_N, "seed": BOOT_SEED, "ci": "percentile 2.5/97.5"},
        },
        "verification": [
            "256 unique manifest entries, 64 per group, 4 per scene per group",
            "all 256 videos exist, decode to 17x256x256x3, and match manifest video SHA256",
            "main reference frame 0 compared with ref PNG; differences recorded without alignment",
            "fresh v2 cube bboxes compared with motion/reference_targets/ref_sceneNN.sidecar.npz",
            "lossless reference arrays and all relevant hashes saved",
        ],
        "out_of_scope": [
            "SSIM/LPIPS/FVD",
            "constraint satisfaction or generation-diversity claims",
            "pixel-level ground truth beyond the stated simulator / dataset targets",
        ],
    }


def main() -> int:
    t0 = time.time()
    PSNR_OUT.mkdir(parents=True, exist_ok=True)
    (PSNR_OUT / "references").mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    q_data = np.load(EVAL_Q_V2, allow_pickle=True)
    q_all = np.asarray(q_data["q"], dtype=np.float64)
    q_old_all = np.asarray(q_data["q_old"], dtype=np.float64)
    meta = json.loads(EVAL_META.read_text(encoding="utf-8"))

    if len(manifest) != 256:
        raise SystemExit(f"expected 256 manifest entries, got {len(manifest)}")
    scenes = sorted({int(e["scene"]) for e in manifest})
    if scenes != list(range(16)):
        raise SystemExit(f"unexpected scenes: {scenes}")
    if q_all.shape != (16, FRAMES, 7):
        raise SystemExit(f"unexpected eval_q_v2 q shape: {q_all.shape}")

    write_json(PSNR_OUT / "protocol.json", build_protocol(manifest, q_data))
    print(f"[{now()}] protocol written", flush=True)

    # ---- Render the shared v2 reference once per scene -------------------
    from cnf_motion.scene import build_scene, disconnect_scene, render_trajectory

    ref_rgb: dict[int, np.ndarray] = {}
    ref_arrays: dict[str, np.ndarray] = {}
    ref_bboxes: dict[int, np.ndarray] = {}
    verification: list[dict[str, Any]] = []

    robot, cube, table = build_scene()
    try:
        for scene in scenes:
            q = q_all[scene]
            cube_xy = np.asarray(meta[scene]["cube_xy"], dtype=np.float64)
            rgb, control, cube_positions = render_trajectory(
                robot, cube, table, q, cube_xy, width=WIDTH
            )
            rgb = np.asarray(rgb, dtype=np.uint8)
            control = np.asarray(control, dtype=np.uint8)
            if rgb.shape != (FRAMES, WIDTH, WIDTH, CHANNELS):
                raise RuntimeError(f"bad rendered RGB shape at scene {scene}: {rgb.shape}")
            ref_rgb[scene] = rgb
            ref_arrays[f"scene{scene:02d}"] = rgb
            bboxes = bbox_from_control(control)
            ref_bboxes[scene] = bboxes

            ref_png_path = EVAL_DS / "ref" / f"{scene:04d}.png"
            ref_png = np.asarray(iio.imread(ref_png_path), dtype=np.uint8)
            aux_path = EVAL_DS / "train" / f"{scene:04d}.mp4"
            aux_video = np.stack(iio.mimread(str(aux_path)))
            aux_video = np.asarray(aux_video, dtype=np.uint8)
            if aux_video.shape != (FRAMES, WIDTH, WIDTH, CHANNELS):
                raise RuntimeError(f"bad auxiliary train video shape at scene {scene}: {aux_video.shape}")

            # Same renderer applied to the original q at frame 0 (v2 q0 is unchanged).
            # This tests whether the raw simulator target itself is stable, isolating
            # the ref-PNG difference from any codec effect in the dataset MP4.
            rgb_old0, _, _ = render_trajectory(
                robot, cube, table, q_old_all[scene][:1], cube_xy, width=WIDTH
            )
            rgb_old0 = np.asarray(rgb_old0, dtype=np.uint8)

            d_png = np.abs(rgb[0].astype(np.int32) - ref_png.astype(np.int32))
            d_train = np.abs(aux_video[0].astype(np.int32) - ref_png.astype(np.int32))
            d_old = np.abs(rgb[0].astype(np.int32) - rgb_old0[0].astype(np.int32))
            sidecar_path = REF_TARGET_DIR / f"ref_scene{scene:02d}.sidecar.npz"
            q_sha = sha256_array(q)
            side: dict[str, Any] = {
                "scene": scene,
                "q_sha256": q_sha,
                "ref_png": str(ref_png_path),
                "ref_png_sha256": sha256_file(ref_png_path),
                "rendered_frame0_vs_ref_png": {
                    "max_abs_diff": int(d_png.max()),
                    "mean_abs_diff": float(d_png.mean()),
                    "identical": bool(np.array_equal(rgb[0], ref_png)),
                },
                "ref_png_vs_train_mp4_frame0": {
                    "max_abs_diff": int(d_train.max()),
                    "mean_abs_diff": float(d_train.mean()),
                    "identical": bool(np.array_equal(aux_video[0], ref_png)),
                },
                "v2_raw_frame0_vs_old_q_raw_frame0": {
                    "max_abs_diff": int(d_old.max()),
                    "mean_abs_diff": float(d_old.mean()),
                    "identical": bool(np.array_equal(rgb[0], rgb_old0[0])),
                    "note": (
                        "Same render_trajectory function; v2 q0 is unchanged, so a nonzero "
                        "difference would indicate renderer nondeterminism. This does not "
                        "explain the ref-PNG difference, which is expected to be codec/legacy-render related."
                    ),
                },
                "aux_train_mp4": str(aux_path),
                "aux_train_mp4_sha256": sha256_file(aux_path),
                "aux_train_shape": list(aux_video.shape),
                "reference_sidecar": str(sidecar_path),
            }
            if sidecar_path.exists():
                side_npz = np.load(sidecar_path)
                side_bboxes = np.asarray(side_npz["bboxes"], dtype=np.int32)
                side_q_sha = str(side_npz["q_sha256"].item())
                mismatch = [
                    int(f) for f in range(FRAMES)
                    if not np.array_equal(bboxes[f], side_bboxes[f])
                ]
                side["reference_sidecar_verification"] = {
                    "q_sha256_match": bool(side_q_sha == q_sha),
                    "bbox_equal_all_frames": bool(np.array_equal(bboxes, side_bboxes)),
                    "bbox_max_abs_diff": int(np.abs(bboxes.astype(np.int64) - side_bboxes.astype(np.int64)).max()),
                    "bbox_mismatch_frames": mismatch,
                }
            else:
                side["reference_sidecar_verification"] = {
                    "q_sha256_match": None,
                    "bbox_equal_all_frames": None,
                    "bbox_max_abs_diff": None,
                    "bbox_mismatch_frames": None,
                    "note": "reference sidecar missing; difference not evaluated",
                }
            side["fresh_bboxes"] = bboxes.tolist()
            side["rendered_rgb_sha256"] = sha256_array(rgb)
            side["cube_positions_first_last"] = [
                np.asarray(cube_positions[0], dtype=float).tolist(),
                np.asarray(cube_positions[-1], dtype=float).tolist(),
            ]
            verification.append(side)
            print(
                f"[{now()}] reference scene {scene:02d}: "
                f"frame0_vs_png max={d_png.max()} mean={d_png.mean():.4f}; "
                f"bbox_equal={side['reference_sidecar_verification']['bbox_equal_all_frames']}",
                flush=True,
            )
    finally:
        disconnect_scene()

    ref_npz = PSNR_OUT / "references" / "v2_render_rgb.npz"
    np.savez_compressed(ref_npz, **ref_arrays)
    ref_npz_sha = sha256_file(ref_npz)
    write_json(PSNR_OUT / "references" / "v2_render_manifest.json", {
        "npz_path": str(ref_npz),
        "npz_sha256": ref_npz_sha,
        "scenes": {
            f"scene{s:02d}": {
                "shape": list(ref_rgb[s].shape),
                "dtype": str(ref_rgb[s].dtype),
                "rgb_sha256": sha256_array(ref_rgb[s]),
                "bboxes": ref_bboxes[s].tolist(),
            }
            for s in scenes
        },
    })
    write_json(PSNR_OUT / "reference_verification.json", {
        "created": now(),
        "note": (
            "Differences are recorded as observed. No per-video alignment, shift, frame "
            "selection, or target optimization was applied."
        ),
        "scenes": verification,
    })
    print(f"[{now()}] reference npz saved: {ref_npz} sha256={ref_npz_sha}", flush=True)

    # ---- Auxiliary dataset references: decode each train video once ------
    aux_ref: dict[int, np.ndarray] = {}
    for scene in scenes:
        p = EVAL_DS / "train" / f"{scene:04d}.mp4"
        vid = np.asarray(np.stack(iio.mimread(str(p))), dtype=np.uint8)
        if vid.shape != (FRAMES, WIDTH, WIDTH, CHANNELS):
            raise RuntimeError(f"bad aux train video shape at scene {scene}: {vid.shape}")
        aux_ref[scene] = vid
        print(f"[{now()}] auxiliary train video scene {scene:02d} decoded {vid.shape}", flush=True)

    # ---- Score all 256 videos -------------------------------------------
    per_video: list[dict[str, Any]] = []
    per_frame_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()

    for entry in manifest:
        video_id = str(entry["video_id"])
        if video_id in seen_video_ids:
            raise SystemExit(f"duplicate video_id in manifest: {video_id}")
        seen_video_ids.add(video_id)
        scene = int(entry["scene"])
        video_path = Path(entry["video_path"])
        if not video_path.exists():
            failures.append({"video_id": video_id, "error": "missing video"})
            continue
        actual_sha = sha256_file(video_path)
        if actual_sha != str(entry["video_sha256"]):
            failures.append({
                "video_id": video_id,
                "error": "video sha256 mismatch",
                "expected": entry["video_sha256"],
                "actual": actual_sha,
            })
            continue
        try:
            video = np.stack(iio.mimread(str(video_path)))
            video = np.asarray(video, dtype=np.uint8)
        except Exception as exc:  # pragma: no cover - defensive
            failures.append({"video_id": video_id, "error": f"decode failed: {exc}"})
            continue
        if video.shape != (FRAMES, WIDTH, WIDTH, CHANNELS):
            failures.append({"video_id": video_id, "error": f"bad shape {video.shape}"})
            continue

        sse_main = frame_sse(video, ref_rgb[scene])
        sse_aux = frame_sse(video, aux_ref[scene])
        main_m = metrics_from_sse(sse_main)
        aux_m = metrics_from_sse(sse_aux)

        rec: dict[str, Any] = {
            "video_id": video_id,
            "group": str(entry["group"]),
            "group_slug": str(entry["group_slug"]),
            "basis_kind": str(entry["basis_kind"]),
            "mode": str(entry["mode"]),
            "scene": scene,
            "motion_index": int(entry["motion_index"]),
            "motion_seed": int(entry["motion_seed"]),
            "wan_seed": int(entry["wan_seed"]),
            "video_path": str(video_path),
            "video_manifest_sha256": str(entry["video_sha256"]),
            "video_actual_sha256": actual_sha,
            "decoded_shape": list(video.shape),
            "main": main_m,
            "aux_train": aux_m,
        }
        per_video.append(rec)

        for f in range(FRAMES):
            per_frame_rows.append({
                "video_id": video_id,
                "group": str(entry["group"]),
                "scene": scene,
                "motion_seed": int(entry["motion_seed"]),
                "wan_seed": int(entry["wan_seed"]),
                "frame": f,
                "main_mse": main_m["per_frame_mse"][f],
                "main_psnr": main_m["per_frame_psnr"][f],
                "aux_mse": aux_m["per_frame_mse"][f],
                "aux_psnr": aux_m["per_frame_psnr"][f],
            })
        if len(per_video) % 32 == 0:
            print(f"[{now()}] scored {len(per_video)}/256 videos", flush=True)

    if failures:
        write_json(PSNR_OUT / "failures.json", {"failures": failures})
        raise SystemExit(f"{len(failures)} videos failed; see psnr/failures.json")
    if len(per_video) != 256:
        raise SystemExit(f"scored {len(per_video)} videos, expected 256")

    # ---- Completeness checks --------------------------------------------
    from collections import Counter, defaultdict

    group_counts = Counter(r["group"] for r in per_video)
    scene_group_counts = Counter((r["scene"], r["group"]) for r in per_video)
    video_ids = {r["video_id"] for r in per_video}
    completeness = {
        "manifest_entries": len(manifest),
        "unique_video_ids": len(video_ids),
        "scored": len(per_video),
        "failures": 0,
        "frames_per_video": sorted({r["decoded_shape"][0] for r in per_video}),
        "shapes": sorted({tuple(r["decoded_shape"]) for r in per_video}),
        "group_counts": dict(sorted(group_counts.items())),
        "scene_group_counts_all_4": all(
            scene_group_counts[(s, g)] == 4 for s in scenes for g in GROUPS
        ),
        "n_scenes": len(scenes),
        "groups": GROUPS,
    }
    expected_ok = (
        completeness["manifest_entries"] == 256
        and completeness["unique_video_ids"] == 256
        and completeness["scored"] == 256
        and completeness["group_counts"] == {g: 64 for g in GROUPS}
        and completeness["scene_group_counts_all_4"]
        and completeness["frames_per_video"] == [FRAMES]
        and completeness["shapes"] == [(FRAMES, WIDTH, WIDTH, CHANNELS)]
    )
    completeness["all_expected_ok"] = bool(expected_ok)
    if not expected_ok:
        raise SystemExit(f"completeness check failed: {completeness}")

    # ---- Per-video CSV/JSON ---------------------------------------------
    video_metrics = ["psnr_all17", "psnr_excl_first16", "psnr_anchors6", "psnr_nonanchors11"]
    mse_metrics = ["mse_all17", "mse_excl_first16", "mse_anchors6", "mse_nonanchors11"]
    video_rows = []
    for r in per_video:
        row = {
            "video_id": r["video_id"],
            "group": r["group"],
            "scene": r["scene"],
            "motion_seed": r["motion_seed"],
            "wan_seed": r["wan_seed"],
            "video_sha256": r["video_actual_sha256"],
        }
        for prefix, bucket in (("main", r["main"]), ("aux", r["aux_train"])):
            for m in mse_metrics:
                row[f"{prefix}_{m}"] = bucket[m]
            for m in video_metrics:
                row[f"{prefix}_{m}"] = bucket[m]
        video_rows.append(row)
    video_fields = list(video_rows[0].keys())
    write_csv(PSNR_OUT / "per_video.csv", video_rows, video_fields)
    write_json(PSNR_OUT / "per_video.json", {"videos": per_video})

    frame_fields = [
        "video_id", "group", "scene", "motion_seed", "wan_seed", "frame",
        "main_mse", "main_psnr", "aux_mse", "aux_psnr",
    ]
    write_csv(PSNR_OUT / "per_frame.csv", per_frame_rows, frame_fields)
    write_json(PSNR_OUT / "per_frame.json", {"frames": per_frame_rows})

    # ---- Per-scene and per-group aggregation ----------------------------
    per_scene_rows = []
    scene_values: dict[str, dict[int, dict[str, float | None]]] = {}
    for group in GROUPS:
        scene_values[group] = {}
        for scene in scenes:
            recs = [r for r in per_video if r["group"] == group and r["scene"] == scene]
            if len(recs) != 4:
                raise RuntimeError(f"{group} scene {scene}: {len(recs)} videos")
            means: dict[str, float | None] = {}
            for subset in SUBSETS:
                vals = [r["main"][f"psnr_{subset}"] for r in recs]
                means[f"mean_psnr_{subset}"] = (
                    float(np.mean([v for v in vals if v is not None]))
                    if all(v is not None for v in vals) else None
                )
            pixels_per_frame = WIDTH * WIDTH * CHANNELS
            # Accumulate SSE as sum(per-frame MSE) * pixels-per-frame; do not
            # multiply the pooled MSE by pixels without the frame count.
            total_sse = sum(
                float(np.sum(np.asarray(r["main"]["per_frame_mse"]) * pixels_per_frame))
                for r in recs
            )
            total_pixels = len(recs) * FRAMES * pixels_per_frame
            means["dataset_pooled_psnr_all17"] = psnr_from_mse(total_sse / total_pixels)
            means["n_videos"] = len(recs)
            scene_values[group][scene] = means
            row: dict[str, Any] = {"scene": scene, "group": group, "n_videos": len(recs)}
            row.update({k: v for k, v in means.items() if k != "n_videos"})
            per_scene_rows.append(row)
    write_csv(
        PSNR_OUT / "per_scene.csv",
        per_scene_rows,
        ["scene", "group", "n_videos"] + [f"mean_psnr_{s}" for s in SUBSETS] + ["dataset_pooled_psnr_all17"],
    )
    write_json(PSNR_OUT / "per_scene.json", {"scenes": per_scene_rows})

    def group_stats(group: str, bucket_key: str) -> dict[str, Any]:
        recs = [r for r in per_video if r["group"] == group]
        result: dict[str, Any] = {"n_scenes": 16, "n_videos": len(recs)}
        for subset in SUBSETS:
            # Scene-equal-weight aggregation for whichever target bucket is requested.
            scene_means = []
            for s in scenes:
                vals = [
                    r[bucket_key][f"psnr_{subset}"]
                    for r in recs if r["scene"] == s
                ]
                scene_means.append(
                    float(np.mean(vals)) if all(v is not None for v in vals) else None
                )
            video_means = [r[bucket_key][f"psnr_{subset}"] for r in recs]
            result[f"scene16_mean_psnr_{subset}"] = (
                float(np.mean(scene_means)) if all(v is not None for v in scene_means) else None
            )
            finite_video_means = [v for v in video_means if v is not None]
            result[f"video64_mean_psnr_{subset}"] = (
                float(np.mean(finite_video_means))
                if len(finite_video_means) == len(video_means) else None
            )
            pixels_per_frame = WIDTH * WIDTH * CHANNELS
            subset_frames = np.asarray(SUBSETS[subset], dtype=int)
            sses = [
                float(np.sum(np.asarray(r[bucket_key]["per_frame_mse"])[subset_frames] * pixels_per_frame))
                for r in recs
            ]
            total_pixels = len(recs) * len(subset_frames) * pixels_per_frame
            result[f"dataset_pooled_psnr_{subset}"] = psnr_from_mse(sum(sses) / total_pixels)
        return result

    summary: dict[str, Any] = {
        "protocol": "learned_basis_hybrid_psnr_20261114",
        "created": now(),
        "primary_metric": "scene16_mean_psnr_all17 (per-video pooled PSNR over all 17 frames, 4 seeds averaged per scene, 16 scenes equally weighted)",
        "higher_is_better": True,
        "groups": {g: group_stats(g, "main") for g in GROUPS},
        "aux_train": {g: group_stats(g, "aux_train") for g in GROUPS},
        "contrasts_primary_all17": {},
        "contrasts_aux_train_all17": {},
        "completeness": completeness,
        "bootstrap": {"n_boot": BOOT_N, "seed": BOOT_SEED, "unit": "scene", "n_scenes": 16},
    }
    for a, b in PRIMARY_CONTRASTS:
        name = f"{a} - {b}"
        a_vals = [scene_values[a][s]["mean_psnr_all17"] for s in scenes]
        b_vals = [scene_values[b][s]["mean_psnr_all17"] for s in scenes]
        summary["contrasts_primary_all17"][name] = paired_bootstrap(a_vals, b_vals)

        aux_scene_values: dict[str, dict[int, float]] = {a: {}, b: {}}
        for group in (a, b):
            for scene in scenes:
                recs = [r for r in per_video if r["group"] == group and r["scene"] == scene]
                aux_scene_values[group][scene] = float(np.mean(
                    [r["aux_train"]["psnr_all17"] for r in recs]
                ))
        summary["contrasts_aux_train_all17"][name] = paired_bootstrap(
            [aux_scene_values[a][s] for s in scenes],
            [aux_scene_values[b][s] for s in scenes],
        )

    # Per-scene contrast table is convenient for audit.
    contrast_rows = []
    for a, b in PRIMARY_CONTRASTS:
        for scene in scenes:
            contrast_rows.append({
                "contrast": f"{a} - {b}",
                "scene": scene,
                "a_group": a,
                "b_group": b,
                "a_psnr_all17": scene_values[a][scene]["mean_psnr_all17"],
                "b_psnr_all17": scene_values[b][scene]["mean_psnr_all17"],
                "diff_a_minus_b": (
                    scene_values[a][scene]["mean_psnr_all17"]
                    - scene_values[b][scene]["mean_psnr_all17"]
                ),
            })
    write_csv(
        PSNR_OUT / "contrast_per_scene.csv",
        contrast_rows,
        ["contrast", "scene", "a_group", "b_group", "a_psnr_all17", "b_psnr_all17", "diff_a_minus_b"],
    )
    write_json(PSNR_OUT / "summary.json", summary)

    # ---- Hashes and completion ------------------------------------------
    artifacts = {}
    for p in sorted(PSNR_OUT.rglob("*")):
        # run.log is written by the external tee and keeps growing until process exit;
        # completion.json/PSNR_COMPLETE are written after this snapshot.
        if p.is_file() and p.name not in {"completion.json", "PSNR_COMPLETE", "run.log"}:
            artifacts[str(p.relative_to(PSNR_OUT))] = sha256_file(p)
    hashes = {
        "created": now(),
        "inputs": {
            "video_manifest": sha256_file(MANIFEST),
            "eval_q_v2": sha256_file(EVAL_Q_V2),
            "eval_metadata": sha256_file(EVAL_META),
            "scene_module": sha256_file(SCENE_MODULE),
            "script": sha256_file(Path(__file__).resolve()),
            "reference_sidecars": {
                f"scene{s:02d}": sha256_file(REF_TARGET_DIR / f"ref_scene{s:02d}.sidecar.npz")
                for s in scenes
            },
        },
        "lossless_reference_npz": ref_npz_sha,
        "reference_rgb_per_scene": {
            f"scene{s:02d}": sha256_array(ref_rgb[s]) for s in scenes
        },
        "aux_train_videos": {
            f"scene{s:02d}": sha256_file(EVAL_DS / "train" / f"{s:04d}.mp4")
            for s in scenes
        },
        "videos": {r["video_id"]: r["video_actual_sha256"] for r in per_video},
        "artifacts": artifacts,
        "note": (
            "run.log, completion.json and PSNR_COMPLETE are produced around/after this "
            "hash snapshot; run.log is captured separately in the docs run log."
        ),
    }
    write_json(PSNR_OUT / "hashes.json", hashes)

    elapsed = time.time() - t0
    completion = {
        "protocol": "learned_basis_hybrid_psnr_20261114",
        "completed": True,
        "created": now(),
        "elapsed_sec": elapsed,
        "counts": {
            "manifest": len(manifest),
            "scored": len(per_video),
            "failures": 0,
            "per_group": dict(sorted(group_counts.items())),
            "per_scene_per_group": 4,
            "frames": FRAMES,
            "reference_scenes": len(scenes),
            "aux_reference_scenes": len(aux_ref),
        },
        "checks": {
            "unique_entries_256": completeness["unique_video_ids"] == 256,
            "all_videos_exist_and_decode": True,
            "all_manifest_hashes_match": True,
            "all_shapes_17x256x256x3": completeness["shapes"] == [(17, 256, 256, 3)],
            "no_missing_or_failed": True,
            "main_reference_frame0_vs_ref_png_recorded": True,
            "main_reference_bbox_vs_sidecar_recorded": True,
        },
        "summary_json": str(PSNR_OUT / "summary.json"),
        "hashes_json": str(PSNR_OUT / "hashes.json"),
        "per_video_csv": str(PSNR_OUT / "per_video.csv"),
        "per_frame_csv": str(PSNR_OUT / "per_frame.csv"),
        "per_scene_csv": str(PSNR_OUT / "per_scene.csv"),
        "report": str(_PSNR_DOC),
    }
    write_json(PSNR_OUT / "completion.json", completion)
    (PSNR_OUT / "PSNR_COMPLETE").write_text(
        f"completed {now()} elapsed_sec={elapsed:.3f} scored={len(per_video)} failures=0\n",
        encoding="utf-8",
    )

    print(f"[{now()}] DONE scored={len(per_video)} elapsed={elapsed:.1f}s", flush=True)
    print("primary scene16 mean PSNR (all17 dB):", flush=True)
    for g in GROUPS:
        print(f"  {g}: {summary['groups'][g]['scene16_mean_psnr_all17']:.6f}", flush=True)
    print("primary contrasts (dB, a-b, 95% CI):", flush=True)
    for name, c in summary["contrasts_primary_all17"].items():
        print(
            f"  {name}: {c['mean_diff']:.6f} "
            f"[{c['ci95'][0]:.6f}, {c['ci95'][1]:.6f}] "
            f"scenes+={c['improved_scenes']}/16",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
