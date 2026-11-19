#!/usr/bin/env python3
"""CLI for the SDFusion + CNF shape-completion application.

Subcommands
-----------
``preflight``  validation-only one-step forward/backward for plain/soft/cnf
               against a manifest; NEVER calls optimizer.step() or writes a
               checkpoint.  Writes a JSON verification log.
``train``      the real training entry point.  It only constructs an optimizer
               and steps when ``--run`` is given explicitly; by default it
               prints the planned configuration and exits without any update.

The external SDFusion checkout and official checkpoints are configured with
``--sdfusion-root`` / ``--weights-dir`` or the ``GCNF_SDFUSION_ROOT`` /
``GCNF_SDFUSION_WEIGHTS`` environment variables.  No path from a prior
workspace is assumed.

Thin wrappers::

    python scripts/train.py preflight --manifest <list.json> --arms plain soft cnf
    python scripts/train.py train --manifest <list.json> --arm cnf --run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict

import torch

from . import axes
from .conditioning import build_condition
from .data import build_loader, first_batch
from .runtime import ENV_UPSTREAM, ENV_WEIGHTS, RuntimePaths, load_sdfusion
from .train import ARMS, TrainHarness, make_query_points


def _state_checksum(state: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _load_runtime(args):
    paths = RuntimePaths.resolve(args.sdfusion_root, args.weights_dir)
    return load_sdfusion(device=args.device, renderer=False, verify_checksums=True, paths=paths)


def _grad_norm(params, *, finite: bool = True) -> float:
    total = 0.0
    n = 0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        if finite and not bool(torch.isfinite(g).all()):
            raise RuntimeError("non-finite gradient encountered")
        total += float(g.float().norm()) ** 2
        n += 1
    return total**0.5, n


def _reset_unet(unet, state: Dict[str, torch.Tensor]) -> None:
    unet.load_state_dict(state, strict=True)
    unet.zero_grad(set_to_none=True)


def make_harness(runtime, args, dev: torch.device) -> TrainHarness:
    """Single config factory shared by preflight and the real train path."""
    return TrainHarness(
        runtime,
        mask_reduce=args.mask_reduce,
        mask_erode=args.mask_erode,
        t_weight_mode=args.t_weight_mode,
        sigma_scale=args.sigma_scale,
        device=dev,
    )


def select_arm(args) -> str:
    """Resolve one unambiguous arm for the real train CLI (never takes [0])."""
    if getattr(args, "arm", None) is not None:
        return args.arm
    arms = list(args.arms)
    if len(arms) == 1:
        return arms[0]
    raise SystemExit(
        "train requires a single arm: pass --arm {plain,soft,cnf} "
        "(or --arms with exactly one value); got " + repr(arms)
    )


# --------------------------------------------------------------------------- preflight
def run_preflight(args) -> dict:
    t0 = time.time()
    device = args.device
    use_cuda = torch.device(device).type == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dev = torch.device(device)

    axis_report = axes.assert_axis_convention(res=args.resolution)
    runtime = _load_runtime(args)
    # Deterministic forwards: no stochastic dropout, identical across arms and
    # across the combined / geometry-only fresh passes.
    runtime.unet.eval()
    unet_eval = not runtime.unet.training
    harness = make_harness(runtime, args, dev)

    loader = build_loader(
        args.manifest,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=0,
        resolution=args.resolution,
        strict=True,
    )
    batch = first_batch(loader).to(dev)
    query = make_query_points(
        batch.anchors, args.resolution, n_grid=args.n_query_grid, seed=args.seed
    ).to(dev)
    z_shape = tuple(runtime.z_shape)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    noise_obs = torch.randn(args.batch_size, *z_shape, generator=gen, dtype=torch.float32).to(dev)
    noise_hidden = torch.randn(args.batch_size, *z_shape, generator=gen, dtype=torch.float32).to(
        dev
    )
    t = torch.full((args.batch_size,), args.timestep, dtype=torch.long, device=dev)

    # Structural GT-leak check: the observation-condition must be unchanged
    # when target_sdf is replaced with garbage.  Only partial_sdf + known_mask
    # may feed the condition path.
    cond_real = build_condition(
        harness, batch, mask_reduce=args.mask_reduce, mask_erode=args.mask_erode
    )
    cond_scrambled = build_condition(
        harness,
        replace(batch, target_sdf=torch.zeros_like(batch.target_sdf)),
        mask_reduce=args.mask_reduce,
        mask_erode=args.mask_erode,
    )
    condition_target_independent = bool(
        torch.equal(cond_real.z_cond, cond_scrambled.z_cond)
        and torch.equal(cond_real.latent_mask, cond_scrambled.latent_mask)
    )
    if not condition_target_independent:
        raise RuntimeError("condition path depends on target_sdf: GT leakage")

    # Never accept an empty condition silently: fail the preflight loudly.
    condition_nonempty = (
        int(cond_real.stats["latent_observed_count"]) > 0
        and int(cond_real.stats["n_empty_masks"]) == 0
    )
    if not condition_nonempty:
        raise RuntimeError(
            "condition mask is empty for the preflight batch "
            f"(mask_reduce={args.mask_reduce}, mask_erode={args.mask_erode}, "
            f"latent_observed_count={cond_real.stats['latent_observed_count']}, "
            f"n_empty_masks={cond_real.stats['n_empty_masks']})"
        )

    unet = runtime.unet
    init_state = {k: v.detach().clone() for k, v in unet.state_dict().items()}
    init_checksum = _state_checksum(init_state)
    vq_frozen = all(not p.requires_grad for p in harness.vqvae_params())
    if not vq_frozen:
        raise RuntimeError("VQ-VAE is not frozen")

    report = {
        "status": "running",
        "kind": "validation-only preflight (no optimizer step, no checkpoint)",
        "manifest": str(args.manifest),
        "object_ids": batch.object_ids,
        "data_strict": True,
        "data_contract_violations": list(loader.dataset.contract_violations),
        "device": str(dev),
        "arms_requested": list(args.arms),
        "seed": args.seed,
        "timestep": args.timestep,
        "z_shape": list(z_shape),
        "dataset": "as supplied by --manifest (official data status not asserted here)",
        "checkpoint_report": runtime.checkpoint_report,
        "axis_convention": axis_report,
        "vqvae_frozen": vq_frozen,
        "condition_target_independent": condition_target_independent,
        "condition_nonempty": condition_nonempty,
        "condition_latent_observed_count": int(cond_real.stats["latent_observed_count"]),
        "unet_eval_mode": unet_eval,
        "mask_reduce": args.mask_reduce,
        "mask_erode": args.mask_erode,
        "initial_param_checksum": init_checksum,
        "arms": {},
    }

    for arm in args.arms:
        _reset_unet(unet, init_state)
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)
        before = _state_checksum({k: v.detach() for k, v in unet.state_dict().items()})
        res = harness.train_step(batch, arm, t, noise_obs, noise_hidden, query)
        tensors = res.pop("_tensors")
        total = tensors["loss_total"]
        unet_params, vq_params = tensors["unet_params"], tensors["vqvae_params"]

        def finite_and_nonzero(x: float) -> bool:
            return x == x and abs(x) != float("inf") and x > 0.0

        total.backward()  # gradient checkpointing: one backward per graph
        total_grad_norm, n_grad = _grad_norm(unet_params)
        res["total_grad_norm_unet"] = total_grad_norm
        res["n_trainable_tensors_with_grad"] = n_grad

        # A FRESH forward is required: the official U-Net's checkpoint function
        # consumes its saved tensors on the first backward.  Same weights/noise/t/
        # query -> exact matched comparison.
        unet.zero_grad(set_to_none=True)
        geom_only = harness.train_step(batch, arm, t, noise_obs, noise_hidden, query)
        res["geometry_fresh_forward_match"] = bool(
            abs(geom_only["loss_geometry"] - res["loss_geometry"])
            <= 1e-6 * max(1.0, abs(res["loss_geometry"]))
        )
        geom_only["_tensors"]["loss_geometry"].backward()
        geom_grad_norm, n_geom = _grad_norm(unet_params)
        res["geometry_grad_norm_unet"] = geom_grad_norm
        res["n_geom_grad_tensors"] = n_geom

        vq_grads = [p.grad for p in vq_params if p.grad is not None]
        res["vqvae_grad_count"] = len(vq_grads)
        res["vqvae_frozen_after_backward"] = len(vq_grads) == 0
        res["param_checksum_before_backward"] = before
        res["loss_finite"] = all(
            isinstance(res[k], (int, float)) and res[k] == res[k] and abs(res[k]) != float("inf")
            for k in ("loss_total", "loss_diffusion", "loss_geometry")
        )
        res["trainable_grads_finite_nonzero"] = finite_and_nonzero(total_grad_norm)
        res["geometry_grads_finite_nonzero"] = finite_and_nonzero(geom_grad_norm)

        if use_cuda:
            torch.cuda.synchronize(dev)
            res["peak_gpu_mem_allocated_mib"] = torch.cuda.max_memory_allocated(dev) / 1024**2
            res["peak_gpu_mem_reserved_mib"] = torch.cuda.max_memory_reserved(dev) / 1024**2
        else:
            res["peak_gpu_mem_allocated_mib"] = None

        if arm == "cnf":
            d = res.get("cnf_solve", {})
            res["cnf_precise"] = bool(
                d.get("ok")
                and res.get("anchor_value_abs_err_max", 1e9) < 1e-6
                and res.get("anchor_grad_abs_err_max", 1e9) < 1e-5
            )

        ok = (
            res["loss_finite"]
            and res["trainable_grads_finite_nonzero"]
            and res["geometry_grads_finite_nonzero"]
            and res["vqvae_frozen_after_backward"]
            and res["z0_pred_requires_grad"]
            and res["sdf_requires_grad"]
            and res["geometry_grad_norm_sdf"] > 0.0
            and res["geometry_fresh_forward_match"]
        )
        if arm == "cnf":
            ok = ok and res["cnf_precise"]
        res["arm_ok"] = bool(ok)
        report["arms"][arm] = res
        print(
            f"[preflight] {arm:5s} loss={res['loss_total']:.4f} "
            f"diff={res['loss_diffusion']:.6f} geom={res['loss_geometry']:.4f} "
            f"|g_total|={total_grad_norm:.3e} |g_geom|={geom_grad_norm:.3e} "
            f"peak={res.get('peak_gpu_mem_allocated_mib')} MiB ok={ok}"
        )

    diff_losses = [report["arms"][a]["loss_diffusion"] for a in args.arms]
    max_diff = max(diff_losses) - min(diff_losses)
    report["shared_diffusion_loss"] = diff_losses[0] if diff_losses else None
    report["shared_diffusion_loss_max_abs_diff"] = max_diff
    report["shared_diffusion_loss_equal"] = bool(
        max_diff <= 1e-6 * max(1.0, max((abs(d) for d in diff_losses), default=1.0))
    )

    # Check final weights BEFORE any restore: restoring first would make the
    # no-optimizer-update assertion tautological.
    final_checksum = _state_checksum({k: v.detach() for k, v in unet.state_dict().items()})
    report["final_param_checksum"] = final_checksum
    report["no_optimizer_update"] = final_checksum == init_checksum
    report["all_arms_ok"] = all(a["arm_ok"] for a in report["arms"].values())
    report["status"] = (
        "passed"
        if (
            report["all_arms_ok"]
            and report["no_optimizer_update"]
            and report["shared_diffusion_loss_equal"]
            and condition_nonempty
            and unet_eval
        )
        else "failed"
    )
    _reset_unet(unet, init_state)
    report["elapsed_s"] = round(time.time() - t0, 2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        f"[preflight] status={report['status']} no_update={report['no_optimizer_update']} log={out}"
    )
    if report["status"] != "passed":
        raise SystemExit(2)
    return report


# --------------------------------------------------------------------------- train
def _checkpoint_name(args, arm: str) -> str:
    """Reproducible, settings-scoped checkpoint name."""
    return (
        f"unet_{arm}_s{args.steps}_seed{args.seed}"
        f"_mr{args.mask_reduce}_me{args.mask_erode}"
        f"_tw{args.t_weight_mode}_ss{args.sigma_scale:g}.pt"
    )


def _atomic_torch_save(obj, path: Path) -> None:
    """Write a weight file through a temp file + os.replace (never a partial)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run_train(args) -> dict:
    """Real training entry point.  Steps only with the explicit ``--run`` flag."""
    arm = select_arm(args)
    out_dir = Path(args.out_dir)
    settings = {
        "mask_reduce": args.mask_reduce,
        "mask_erode": args.mask_erode,
        "t_weight_mode": args.t_weight_mode,
        "sigma_scale": args.sigma_scale,
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    checkpoint = out_dir / _checkpoint_name(args, arm)
    if not args.run:
        plan = {
            "status": "dry-start (no optimizer, no step)",
            "manifest": str(args.manifest),
            "arms": [arm],
            "arm": arm,
            "steps": args.steps,
            "lr": args.lr,
            "device": args.device,
            "checkpoint_every": args.ckpt_every,
            "settings": settings,
            "checkpoint": str(checkpoint),
            "step_policy": "iterate epochs until exactly --steps optimizer steps",
            "hint": "pass --run to actually train",
        }
        print(json.dumps(plan, indent=2))
        return plan

    dev = torch.device(args.device)
    runtime = _load_runtime(args)
    harness = make_harness(runtime, args, dev)  # same settings as preflight
    if any(p.requires_grad for p in harness.vqvae_params()):
        raise RuntimeError("VQ-VAE must stay frozen in train")
    loader = build_loader(
        args.manifest,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=0,
        resolution=args.resolution,
        strict=True,
    )
    optimizer = torch.optim.AdamW(harness.unet_params(), lr=args.lr)
    out_dir.mkdir(parents=True, exist_ok=True)
    z_shape = tuple(runtime.z_shape)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    step = 0
    periodic: list[str] = []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    while step < args.steps:
        progressed = False
        for batch in loader:
            if step >= args.steps:
                break
            progressed = True
            batch = batch.to(dev)
            bs = batch.batch_size
            query = make_query_points(batch.anchors, args.resolution, seed=args.seed).to(dev)
            noise_obs = torch.randn(bs, *z_shape, generator=gen, dtype=torch.float32).to(dev)
            noise_hidden = torch.randn(bs, *z_shape, generator=gen, dtype=torch.float32).to(dev)
            t = torch.randint(0, runtime.num_timesteps, (bs,), generator=gen).to(dev)
            optimizer.zero_grad(set_to_none=True)
            res = harness.train_step(batch, arm, t, noise_obs, noise_hidden, query)
            loss_total = res["_tensors"]["loss_total"]
            loss_value = float(loss_total.detach())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"non-finite loss at arm={arm} step={step + 1}: {loss_value}")
            loss_total.backward()
            if step % args.log_every == 0:
                grad_norm, _ = _grad_norm(harness.unet_params(), finite=True)
                if not math.isfinite(grad_norm) or grad_norm <= 0.0:
                    raise RuntimeError(
                        f"non-finite/degenerate gradient at arm={arm} step={step + 1}: {grad_norm}"
                    )
            optimizer.step()
            step += 1
            if step % args.log_every == 0:
                metric = {
                    "event": "train_metric",
                    "arm": arm,
                    "step": step,
                    "steps": args.steps,
                    "lr": args.lr,
                    "loss_total": res["loss_total"],
                    "loss_diffusion": res["loss_diffusion"],
                    "loss_geometry": res["loss_geometry"],
                    "grad_norm_unet": grad_norm,
                    "peak_gpu_mem_allocated_mib": (
                        torch.cuda.max_memory_allocated(dev) / 1024**2
                        if dev.type == "cuda"
                        else None
                    ),
                    "known_voxel_count": res.get("known_voxel_count"),
                    "latent_observed_count": res.get("latent_observed_count"),
                }
                for key in ("anchor_value_abs_err_max", "anchor_grad_abs_err_max", "cnf_solve"):
                    if key in res:
                        metric[key] = res[key]
                print("[train-metric] " + json.dumps(metric, default=str), flush=True)
                print(f"[train] step {step}: {res['loss_total']:.4f}", flush=True)
            if args.ckpt_every > 0 and step % args.ckpt_every == 0:
                pname = checkpoint.name.replace(f"_s{args.steps}_", f"_s{step}_")
                pname = pname.replace(".pt", "_periodic.pt")
                ppath = out_dir / pname
                _atomic_torch_save(runtime.unet.state_dict(), ppath)
                periodic.append(str(ppath))
                olds = sorted(
                    out_dir.glob(f"unet_{arm}_s*_seed{args.seed}_*_periodic.pt"),
                    key=lambda q: q.stat().st_mtime,
                )
                for old in olds[:-2]:
                    old.unlink()
        if not progressed:
            raise RuntimeError(f"dataset exhausted after {step} steps before budget {args.steps}")
    _atomic_torch_save(runtime.unet.state_dict(), checkpoint)
    return {
        "status": "trained",
        "arm": arm,
        "steps": step,
        "checkpoint": str(checkpoint),
        "settings": settings,
        "final_loss": res["loss_total"],
        "periodic_checkpoints": periodic,
    }


# --------------------------------------------------------------------------- argparse
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--manifest", type=Path, required=True)
        sp.add_argument(
            "--sdfusion-root",
            default=os.environ.get(ENV_UPSTREAM),
            help=f"official SDFusion checkout (default: ${ENV_UPSTREAM})",
        )
        sp.add_argument(
            "--weights-dir",
            default=os.environ.get(ENV_WEIGHTS),
            help=f"directory with official .pth checkpoints (default: ${ENV_WEIGHTS})",
        )
        sp.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--seed", type=int, default=111)
        sp.add_argument("--resolution", type=int, default=64)
        sp.add_argument("--batch-size", type=int, default=1)
        sp.add_argument("--mask-reduce", default="half", choices=["half", "all", "any"])
        sp.add_argument("--mask-erode", type=int, default=0)
        sp.add_argument(
            "--t-weight-mode",
            default="alphas_cumprod",
            choices=["alphas_cumprod", "one"],
        )
        sp.add_argument("--sigma-scale", type=float, default=2.0)

    pf = sub.add_parser("preflight", help="validation-only forward/backward, no step")
    add_common(pf)
    pf.add_argument("--timestep", type=int, default=500)
    pf.add_argument("--n-query-grid", type=int, default=224)
    pf.add_argument("--out", type=Path, default=Path("preflight-report.json"))
    pf.set_defaults(func=run_preflight)

    tr = sub.add_parser("train", help="training entry point (steps only with --run)")
    add_common(tr)
    tr.add_argument(
        "--arm",
        choices=ARMS,
        default=None,
        help="single arm to train (preferred over a multi-value --arms)",
    )
    tr.add_argument("--steps", type=int, default=1000)
    tr.add_argument("--lr", type=float, default=1e-5)
    tr.add_argument("--log-every", type=int, default=50)
    tr.add_argument(
        "--ckpt-every",
        type=int,
        default=200,
        help="write atomic recoverable U-Net snapshots every N steps (0 disables)",
    )
    tr.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    tr.add_argument("--run", action="store_true", help="actually perform optimizer steps")
    tr.set_defaults(func=run_train)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
