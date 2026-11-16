from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .util_eval import (
    BASE_1_3B,
    EVAL_DS,
    FRAMES,
    OUT,
    VIDEOX,
    WAN_CFG,
    WAN_FPS,
    WAN_LORA,
    WAN_MODEL,
    WAN_STEPS,
    WAN_SHIFT,
    append_eval_log,
    decode_video,
    ensure_eval_dirs,
    file_ok,
    read_json,
    sha256_file,
    update_status_eval,
    utcnow,
    write_json,
)

NEGATIVE_PROMPT = "blurry, distorted, deformed, static, low quality, text, watermark."


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    mse = float(np.mean((af - bf) ** 2))
    if mse == 0.0:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def build_wan_pipeline(device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16):
    """Exact pipeline builder from run_learned_basis_wan_20261113.py."""
    if str(VIDEOX) not in sys.path:
        sys.path.insert(0, str(VIDEOX))
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from videox_fun.models import (
        AutoencoderKLWan,
        AutoTokenizer,
        CLIPModel,
        WanT5EncoderModel,
        WanTransformer3DModel,
    )
    from videox_fun.pipeline import WanFunControlPipeline
    from videox_fun.utils import apply_gpu_memory_mode, filter_kwargs, merge_lora

    if device is None:
        device = torch.device("cuda")
    config = OmegaConf.load(VIDEOX / "config/wan2.1/wan_civitai.yaml")
    transformer = WanTransformer3DModel.from_pretrained(
        str(WAN_MODEL),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )
    vae = AutoencoderKLWan.from_pretrained(
        str(WAN_MODEL / config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        str(WAN_MODEL / config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        str(WAN_MODEL / config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).eval()
    clip_image_encoder = CLIPModel.from_pretrained(
        str(WAN_MODEL / config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"))
    ).to(dtype).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
    )
    pipe = WanFunControlPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )
    apply_gpu_memory_mode(pipe, "model_full_load", device, dtype)
    pipe = merge_lora(pipe, str(WAN_LORA), 1.0, device=device, dtype=dtype)
    return pipe, device


def run_one_wan(
    pipeline,
    control_path: str | Path,
    ref_path: str | Path,
    out_path: str | Path,
    seed: int,
    device: torch.device,
) -> None:
    """Exact single-video generator from run_learned_basis_wan_20261113.py."""
    if str(VIDEOX) not in sys.path:
        sys.path.insert(0, str(VIDEOX))
    from PIL import Image
    from cnf_motion.config import PROMPT as CONFIG_PROMPT
    from videox_fun.utils import get_image_latent, get_video_to_video_latent, save_videos_grid

    sample_size = [256, 256]
    control_latent, _, _, _ = get_video_to_video_latent(
        str(control_path), video_length=FRAMES, sample_size=sample_size, fps=WAN_FPS
    )
    start_latent = get_image_latent(str(ref_path), sample_size=sample_size)
    ref_latent = get_image_latent(str(ref_path), sample_size=sample_size)
    clip_image = Image.open(str(ref_path)).convert("RGB")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        pred = pipeline(
            CONFIG_PROMPT,
            num_frames=FRAMES,
            negative_prompt=NEGATIVE_PROMPT,
            height=256,
            width=256,
            generator=generator,
            guidance_scale=WAN_CFG,
            num_inference_steps=WAN_STEPS,
            control_video=control_latent,
            ref_image=ref_latent,
            start_image=start_latent,
            clip_image=clip_image,
            shift=WAN_SHIFT,
        ).videos
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_videos_grid(pred, str(out_path), fps=WAN_FPS)
    del pred, control_latent, start_latent, ref_latent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_gt_replay_verification(pipe, device: torch.device) -> dict[str, Any]:
    from cnf_motion.config import PROMPT as CONFIG_PROMPT

    replay_dir = OUT / "video" / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_rows = []
    for i in range(8):
        control = BASE_1_3B / "control" / f"{i}_gt.mp4"
        ref = EVAL_DS / "ref" / f"{i:04d}.png"
        old_video = BASE_1_3B / "videos" / f"{i}_gt.mp4"
        outv = replay_dir / f"{i}_gt_seed43.mp4"
        update_status_eval("wan_gt_replay", done=i, total=8, sample=i)
        t0 = time.time()
        if not (file_ok(outv) and decode_video(outv).shape == (FRAMES, 256, 256, 3)):
            run_one_wan(pipe, control, ref, outv, 43, device)
        a = decode_video(outv)
        b = decode_video(old_video)
        diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
        mae = float(np.mean(diff))
        mx = float(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))
        exact_pixel_frac = float(np.mean(a == b))
        row = {
            "sample": i,
            "control": str(control),
            "ref": str(ref),
            "old_video": str(old_video),
            "new_video": str(outv),
            "max_abs": mx,
            "mae": mae,
            "exact_pixel_fraction": exact_pixel_frac,
            "psnr": _psnr(a, b),
            "shape_new": list(a.shape),
            "shape_old": list(b.shape),
            "tolerance_pass": bool(mx <= 1.0 and mae <= 0.01),
            "strict_bitexact": bool(mx == 0.0 and mae == 0.0),
            "pass": bool(mx <= 1.0 and mae <= 0.01),
            "elapsed_sec": time.time() - t0,
        }
        replay_rows.append(row)
        write_json(replay_dir / "replay_rows.json", replay_rows)
        append_eval_log(
            f"GT replay [{i+1}/8] max_abs={mx} mae={mae:.6f} tolerance_pass={row['tolerance_pass']} strict_bitexact={row['strict_bitexact']}"
        )

    tol_pass = bool(all(r["tolerance_pass"] for r in replay_rows))
    strict_bitexact = bool(all(r["strict_bitexact"] for r in replay_rows))
    summary = {
        "rows": replay_rows,
        "replay_tolerance_pass": tol_pass,
        "strict_bitexact_ok": strict_bitexact,
        "bitexact_ok": strict_bitexact,
        "max_of_max_abs": float(max(r["max_abs"] for r in replay_rows)),
        "max_of_mae": float(max(r["mae"] for r in replay_rows)),
        "mean_mae": float(np.mean([r["mae"] for r in replay_rows])),
        "tolerance": {"max_abs": 1.0, "mae": 0.01},
        "time": utcnow(),
        "lora": str(WAN_LORA),
        "lora_sha256": sha256_file(WAN_LORA),
        "model": str(WAN_MODEL),
        "prompt": CONFIG_PROMPT,
        "negative": NEGATIVE_PROMPT,
        "steps": WAN_STEPS,
        "cfg": WAN_CFG,
        "shift": WAN_SHIFT,
        "wan_seed": 43,
        "note": "replay_tolerance_pass requires max_abs<=1 and MAE<=0.01 across all 8 GT replay samples; strict_bitexact_ok requires all pixels identical (max_abs==0 and MAE==0). No old A/C videos are reused.",
    }
    write_json(replay_dir / "replay_summary.json", summary)
    if not tol_pass:
        fail_path = replay_dir / "replay_failure_evidence.json"
        write_json(fail_path, summary)
        update_status_eval("failed_gt_replay", replay_summary_path=str(fail_path))
        raise RuntimeError(
            f"Wan GT replay failed tolerance check (max_abs={summary['max_of_max_abs']}, max_mae={summary['max_of_mae']:.6f}); "
            f"blocking 256-video generation. Evidence saved to {fail_path}."
        )
    return summary


def _compute_manifest_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    planned = len(items)
    generated = sum(1 for r in items if r.get("count") == "generated")
    reused = sum(1 for r in items if r.get("count") == "reused")
    failed = sum(1 for r in items if r.get("count") == "failed")
    pending = sum(1 for r in items if r.get("count") == "pending")
    assert generated + reused + failed + pending == planned, (
        generated,
        reused,
        failed,
        pending,
        planned,
    )
    return {
        "planned": planned,
        "generated": generated,
        "reused": reused,
        "failed": failed,
        "pending": pending,
    }


def run_all_wan_videos(video_manifest: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    from .sampling import build_video_manifest

    ensure_eval_dirs()
    manifest_path = OUT / "manifests" / "video_manifest.json"
    motion_manifest_path = OUT / "manifests" / "motion_manifest.json"
    if video_manifest is None:
        if file_ok(motion_manifest_path):
            items = build_video_manifest(read_json(motion_manifest_path))
        elif file_ok(manifest_path):
            items = read_json(manifest_path)
        else:
            raise FileNotFoundError(f"Neither motion_manifest nor video_manifest found at {manifest_path}")
    else:
        items = [dict(x) for x in video_manifest]

    if len(items) != 256:
        raise ValueError(f"Expected 256 video manifest rows, got {len(items)}")

    lora_sha = sha256_file(WAN_LORA)
    update_status_eval("wan_loading_pipeline", total=len(items))
    append_eval_log("Loading Wan 1.3B + 1k LoRA pipeline (single load)...")
    pipe, device = build_wan_pipeline()
    append_eval_log("Wan pipeline loaded.")

    replay_summary = run_gt_replay_verification(pipe, device)

    for idx, it in enumerate(items):
        vid = it["video_id"]
        vpath = Path(it["video_path"])
        vmeta_path = Path(str(vpath) + ".meta.json")
        cpath = Path(it["control_path"])
        qpath = Path(it["q_path"])
        rpath = Path(it["ref_path"])
        wan_seed = int(it["wan_seed"])

        # Check motion failure first
        if not it.get("motion_ok", False) or it.get("motion_status") == "failed":
            it["status"] = "motion_failed"
            it["count"] = "failed"
            it["failure_type"] = "motion_failed"
            continue

        # Verify control & q integrity by manifest fields (never guessing from filename)
        if not (file_ok(cpath) and file_ok(qpath) and file_ok(rpath)):
            it["status"] = "missing_control_or_q"
            it["count"] = "failed"
            it["failure_type"] = "missing_control_or_q"
            continue

        try:
            ctrl_arr = decode_video(cpath)
            if ctrl_arr.shape != (FRAMES, 256, 256, 3):
                raise ValueError(f"Control video bad shape: {ctrl_arr.shape}")
            ctrl_sha = sha256_file(cpath)
        except Exception as e:
            it["status"] = "corrupt_control"
            it["count"] = "failed"
            it["failure_type"] = "corrupt_control"
            it["error"] = str(e)
            continue

        # Check if already generated in THIS output directory with matching control_sha256 & lora_sha256
        if file_ok(vpath) and file_ok(vmeta_path):
            try:
                vm = read_json(vmeta_path)
                if (
                    vm.get("control_sha256") == ctrl_sha
                    and vm.get("lora_sha256") == lora_sha
                    and int(vm.get("wan_seed", -1)) == wan_seed
                ):
                    dec = decode_video(vpath)
                    if dec.shape == (FRAMES, 256, 256, 3):
                        it.pop("error", None)
                        it.pop("traceback", None)
                        it["status"] = "generated"
                        it["count"] = "generated"
                        it["failure_type"] = None
                        it["video_shape"] = list(dec.shape)
                        it["video_sha256"] = vm.get("video_sha256") or sha256_file(vpath)
                        it["control_sha256"] = ctrl_sha
                        continue
            except Exception:
                pass

        counts_now = _compute_manifest_counts(items)
        update_status_eval(
            "wan_generation",
            current_index=idx,
            current_video_id=vid,
            **counts_now,
        )
        t0 = time.time()
        try:
            run_one_wan(pipe, cpath, rpath, vpath, wan_seed, device)
            dec = decode_video(vpath)
            if dec.shape != (FRAMES, 256, 256, 3):
                raise RuntimeError(f"Decoded Wan video shape mismatch: {dec.shape}")
            v_sha = sha256_file(vpath)
            elapsed = time.time() - t0
            meta_obj = {
                "video_id": vid,
                "motion_id": it["motion_id"],
                "group": it["group"],
                "scene": int(it["scene"]),
                "motion_seed": int(it["motion_seed"]),
                "wan_seed": wan_seed,
                "control_path": str(cpath),
                "control_sha256": ctrl_sha,
                "lora_sha256": lora_sha,
                "video_sha256": v_sha,
                "shape": list(dec.shape),
                "elapsed_sec": elapsed,
                "time": utcnow(),
            }
            write_json(vmeta_path, meta_obj)
            it.pop("error", None)
            it.pop("traceback", None)
            it["status"] = "generated"
            it["count"] = "generated"
            it["failure_type"] = None
            it["video_shape"] = list(dec.shape)
            it["video_sha256"] = v_sha
            it["control_sha256"] = ctrl_sha
            it["elapsed_sec"] = elapsed
            append_eval_log(f"wan [{idx+1}/256] {vid} generated ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            it["status"] = "video_failed"
            it["count"] = "failed"
            it["failure_type"] = "video_generation_failed"
            it["error"] = str(e)
            it["traceback"] = traceback.format_exc()
            it["elapsed_sec"] = elapsed
            write_json(OUT / "logs" / f"wan_error_{vid}.json", it)
            append_eval_log(f"wan [{idx+1}/256] {vid} FAILED: {e}")

        counts_after = _compute_manifest_counts(items)
        write_json(manifest_path, items)
        write_json(
            OUT / "video" / "video_counts.json",
            {
                "counts": counts_after,
                "gt_replay_tolerance_pass": replay_summary["replay_tolerance_pass"],
                "gt_replay_strict_bitexact_ok": replay_summary["strict_bitexact_ok"],
                "time": utcnow(),
            },
        )
        update_status_eval(
            "wan_generation",
            current_index=idx + 1,
            last_completed_video_id=vid,
            **counts_after,
        )

    final_counts = _compute_manifest_counts(items)
    assert final_counts["pending"] == 0, f"Pending videos remain: {final_counts}"
    assert (
        final_counts["generated"] + final_counts["reused"] + final_counts["failed"] == 256
    ), final_counts

    summary = {
        "counts": final_counts,
        "gt_replay": replay_summary,
        "time": utcnow(),
    }
    write_json(manifest_path, items)
    write_json(OUT / "video" / "video_counts.json", summary)
    update_status_eval("wan_generation_done", **final_counts)
    return {"manifest": items, "summary": summary}
