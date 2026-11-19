"""Configurable loader for the external official SDFusion code and checkpoints.

This module is *our* compatibility layer; it never re-implements or stubs an
official numerical operation.  The official source tree and the two official
weight files live **outside** this repository and are discovered exclusively
through configuration:

* ``GCNF_SDFUSION_ROOT`` (or ``--sdfusion-root``): the official SDFusion
  checkout, pinned to :data:`PINNED_COMMIT`.
* ``GCNF_SDFUSION_WEIGHTS`` (or ``--weights-dir``): the directory holding
  ``vqvae-snet-all.pth`` and ``sdfusion-snet-all.pth``.

No workspace-relative or machine-absolute path is baked into the package.  If
``weights_dir`` is omitted and ``<root>/weights`` exists, that directory is
used as a convenience; otherwise both paths must be configured explicitly.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional, Tuple

import torch

# Keep the pinned source checkout pristine: importing the official modules
# must not create __pycache__ directories inside the external checkout.
sys.dont_write_bytecode = True

ENV_UPSTREAM = "GCNF_SDFUSION_ROOT"
ENV_WEIGHTS = "GCNF_SDFUSION_WEIGHTS"

VQVAE_CKPT_NAME = "vqvae-snet-all.pth"
DIFFUSION_CKPT_NAME = "sdfusion-snet-all.pth"

# The exact upstream revision this adapter was validated against.
PINNED_COMMIT = "09801f6d04c04816e5827af981e2ae77be77ce5b"

# SHA-256 of the two official Box downloads, measured on 2026-11-18.  The
# loader refuses to start on a mismatch.
EXPECTED_VQVAE_SHA256 = "e90257e0d0e1338b61e744a266fa7981f0f462ee7c87bff3b409e4fec68dc31e"
EXPECTED_DIFFUSION_SHA256 = "515ea6a5608bbcd43cc324019f51915fb88723c402d8773afbd690c856a056d4"
EXPECTED_SHA256 = {
    VQVAE_CKPT_NAME: EXPECTED_VQVAE_SHA256,
    DIFFUSION_CKPT_NAME: EXPECTED_DIFFUSION_SHA256,
}

__all__ = [
    "DIFFUSION_CKPT_NAME",
    "ENV_UPSTREAM",
    "ENV_WEIGHTS",
    "EXPECTED_DIFFUSION_SHA256",
    "EXPECTED_SHA256",
    "EXPECTED_VQVAE_SHA256",
    "PINNED_COMMIT",
    "VQVAE_CKPT_NAME",
    "RevisionError",
    "RuntimePaths",
    "SDFusionRuntime",
    "load_sdfusion",
    "load_vqvae",
    "sha256_file",
    "upstream_commit",
    "verify_checkpoint",
    "verify_revision",
]


class RevisionError(RuntimeError):
    """The configured external checkout does not match the pinned revision."""


def _configured(raw: Optional[str | os.PathLike], env: str, label: str) -> Optional[Path]:
    value = raw if raw is not None else os.environ.get(env)
    return Path(value).expanduser() if value else None


@dataclass(frozen=True)
class RuntimePaths:
    """All paths needed to reproduce one official-model load."""

    upstream: Path
    weights_dir: Path

    @property
    def vqvae_ckpt(self) -> Path:
        return self.weights_dir / VQVAE_CKPT_NAME

    @property
    def diffusion_ckpt(self) -> Path:
        return self.weights_dir / DIFFUSION_CKPT_NAME

    @property
    def df_config(self) -> Path:
        return self.upstream / "configs" / "sdfusion_snet.yaml"

    @property
    def vq_config(self) -> Path:
        return self.upstream / "configs" / "vqvae_snet.yaml"

    @classmethod
    def resolve(
        cls,
        sdfusion_root: Optional[str | os.PathLike] = None,
        weights_dir: Optional[str | os.PathLike] = None,
    ) -> "RuntimePaths":
        """Resolve paths from CLI arguments first, then environment variables."""
        root = _configured(sdfusion_root, ENV_UPSTREAM, ENV_UPSTREAM)
        if root is None:
            raise FileNotFoundError(
                f"official SDFusion checkout is not configured; set {ENV_UPSTREAM} "
                "or pass --sdfusion-root"
            )
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"official SDFusion checkout is not a directory: {root}")

        weights = _configured(weights_dir, ENV_WEIGHTS, ENV_WEIGHTS)
        if weights is None:
            inside_root = root / "weights"
            if not inside_root.is_dir():
                raise FileNotFoundError(
                    f"official checkpoints are not configured; set {ENV_WEIGHTS} "
                    "or pass --weights-dir (expected files: "
                    f"{VQVAE_CKPT_NAME}, {DIFFUSION_CKPT_NAME})"
                )
            weights = inside_root
        return cls(upstream=root, weights_dir=weights.resolve())

    def as_dict(self) -> Dict[str, str]:
        return {
            "upstream": str(self.upstream),
            "weights_dir": str(self.weights_dir),
            "vqvae_ckpt": str(self.vqvae_ckpt),
            "diffusion_ckpt": str(self.diffusion_ckpt),
            "df_config": str(self.df_config),
            "vq_config": str(self.vq_config),
        }


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Return the SHA-256 of a file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upstream_commit(paths: RuntimePaths) -> Optional[str]:
    """Commit currently checked out in the configured external tree (metadata only)."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(paths.upstream), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def verify_revision(paths: RuntimePaths, *, strict: bool = True) -> Dict[str, Any]:
    """Compare the external checkout HEAD with :data:`PINNED_COMMIT`.

    A git-derived revision is required to be exact when it is available.  A
    source snapshot without ``.git`` cannot be checked; that is reported as
    ``status: unverifiable`` (and rejected when ``strict`` is True).
    """
    actual = upstream_commit(paths)
    report: Dict[str, Any] = {
        "pinned_commit": PINNED_COMMIT,
        "actual_commit": actual,
        "status": "ok" if actual == PINNED_COMMIT else "mismatch" if actual else "unverifiable",
    }
    if report["status"] != "ok" and strict:
        raise RevisionError(
            "external SDFusion revision could not be confirmed as "
            f"{PINNED_COMMIT}: status={report['status']}, actual={actual}. "
            f"Check out the pinned commit under {ENV_UPSTREAM}."
        )
    return report


def _ensure_upstream_importable(paths: RuntimePaths) -> None:
    root = str(paths.upstream)
    if root not in sys.path:
        sys.path.insert(0, root)


@contextmanager
def _weights_only_guard() -> Iterator[None]:
    """Force ``torch.load(..., weights_only=True)`` while official code loads.

    The two official checkpoints contain only tensors/OrderedDicts, and their
    SHA-256 is verified before this guard is entered.  The guard removes the
    pickle-capable fallback without editing the pinned upstream source.
    """
    original = torch.load

    def safe_load(*args, **kwargs):
        kwargs.setdefault("weights_only", True)
        return original(*args, **kwargs)

    torch.load = safe_load
    try:
        yield
    finally:
        torch.load = original


def verify_checkpoint(path: Path, expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Hash a checkpoint and compare it with the recorded official digest."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    expected = expected_sha256 or EXPECTED_SHA256.get(path.name)
    actual = sha256_file(path)
    report = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
        "expected_sha256": expected,
        "match": expected is None or actual == expected,
    }
    if expected is None:
        raise RuntimeError(f"No expected checksum recorded for {path.name}")
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {path}: got {actual}, expected {expected}")
    return report


def _make_opt(
    paths: RuntimePaths,
    device: str,
    vqvae_ckpt: Path,
    diffusion_ckpt: Path,
) -> SimpleNamespace:
    """Minimal option object accepted by the official SDFusionModel."""
    return SimpleNamespace(
        isTrain=False,
        gpu_ids=[0],
        device=device,
        distributed=False,
        debug="0",
        model="sdfusion",
        df_cfg=str(paths.df_config),
        vq_cfg=str(paths.vq_config),
        ckpt=str(diffusion_ckpt),
        vq_ckpt=str(vqvae_ckpt),
        dataset_mode="snet",
        batch_size=1,
        nThreads=0,
        phase="test",
        serial_batches=False,
        max_dataset_size=10_000_000,
        trunc_thres=0.2,
        ratio=1.0,
        res=64,
        cat="all",
        cached_dir=None,
    )


def load_vqvae(
    device: str = "cuda",
    paths: Optional[RuntimePaths] = None,
    ckpt: Optional[Path] = None,
    verify: bool = True,
) -> torch.nn.Module:
    """Load the official VQ-VAE alone (strict state-dict load, frozen)."""
    paths = paths or RuntimePaths.resolve()
    _ensure_upstream_importable(paths)
    ckpt = Path(ckpt) if ckpt is not None else paths.vqvae_ckpt
    if verify:
        verify_checkpoint(ckpt, EXPECTED_VQVAE_SHA256)

    from models.model_utils import load_vqvae as official_load_vqvae
    from omegaconf import OmegaConf

    vq_conf = OmegaConf.load(paths.vq_config)
    opt = SimpleNamespace(device=device)
    with _weights_only_guard():
        vqvae = official_load_vqvae(vq_conf, vq_ckpt=str(ckpt), opt=opt)
    vqvae.eval()
    for parameter in vqvae.parameters():
        parameter.requires_grad_(False)
    return vqvae


def _load_official_model(
    paths: RuntimePaths,
    device: str,
    vqvae_ckpt: Path,
    diffusion_ckpt: Path,
    renderer: bool,
) -> Tuple[torch.nn.Module, Any]:
    """Instantiate the official SDFusionModel, optionally without a renderer."""
    _ensure_upstream_importable(paths)
    import models.sdfusion_model as official_module
    from models.sdfusion_model import SDFusionModel

    opt = _make_opt(paths, device, vqvae_ckpt, diffusion_ckpt)

    original_init = official_module.init_mesh_renderer
    if not renderer:
        official_module.init_mesh_renderer = lambda *args, **kwargs: None
    try:
        model = SDFusionModel()
        with _weights_only_guard():
            model.initialize(opt)
    finally:
        official_module.init_mesh_renderer = original_init

    model.df.eval()
    model.vqvae.eval()
    for parameter in model.vqvae.parameters():
        parameter.requires_grad_(False)
    return model, opt


@dataclass
class SDFusionRuntime:
    """Loaded official unconditioned SDFusion model plus small helpers."""

    model: torch.nn.Module
    paths: RuntimePaths
    device: str
    checkpoint_report: Dict[str, Any] = field(default_factory=dict)
    has_renderer: bool = False

    @property
    def vqvae(self) -> torch.nn.Module:
        return self.model.vqvae

    @property
    def unet(self) -> torch.nn.Module:
        return self.model.df

    @property
    def z_shape(self) -> Tuple[int, int, int, int]:
        return tuple(self.model.z_shape)

    @property
    def num_timesteps(self) -> int:
        return int(self.model.num_timesteps)

    @property
    def scale(self) -> float:
        return float(self.model.scale)

    def eval(self) -> "SDFusionRuntime":
        self.model.eval()
        self.vqvae.eval()
        self.unet.eval()
        return self

    def noise_schedule(self) -> Dict[str, torch.Tensor]:
        """Official DDPM schedule tensors (same objects used by the sampler)."""
        names = (
            "betas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "log_one_minus_alphas_cumprod",
            "sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod",
            "posterior_variance",
            "posterior_log_variance_clipped",
            "posterior_mean_coef1",
            "posterior_mean_coef2",
            "lvlb_weights",
        )
        schedule = {name: getattr(self.model, name) for name in names}
        schedule["num_timesteps"] = torch.tensor(self.num_timesteps, device=self.device)
        return schedule

    @torch.no_grad()
    def encode_latent(self, sdf: torch.Tensor) -> torch.Tensor:
        """Encode a (B, 1, 64, 64, 64) T-SDF grid to an unquantised latent."""
        if sdf.dim() != 5 or sdf.shape[1] != 1:
            raise ValueError(f"expected (B,1,64,64,64), got {tuple(sdf.shape)}")
        return self.vqvae.encode_no_quant(sdf.to(self.device))

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        """Decode a (B, 3, 16, 16, 16) latent to a 64^3 T-SDF grid."""
        return self.vqvae.decode_no_quant(z.to(self.device), force_not_quantize=not quantize)

    @torch.no_grad()
    def denoise(self, x_t: torch.Tensor, t: torch.Tensor, cond: Any = None) -> torch.Tensor:
        """Run the official 3D U-Net denoiser and return predicted noise."""
        x_t = x_t.to(self.device)
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=self.device, dtype=torch.long)
        else:
            t = t.to(device=self.device, dtype=torch.long)
        if t.dim() == 0:
            t = t[None]
        return self.model.apply_model(x_t, t, cond)

    @torch.no_grad()
    def sample_latent(
        self,
        n: int = 1,
        ddim_steps: int = 100,
        eta: float = 0.0,
        x_T: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Official DDIM sampling in latent space (unconditioned)."""
        self.eval()
        if seed is not None:
            torch.manual_seed(seed)
        from models.networks.diffusion_networks.samplers.ddim import DDIMSampler

        sampler = DDIMSampler(self.model)
        samples, intermediates = sampler.sample(
            S=ddim_steps,
            batch_size=n,
            shape=self.z_shape,
            conditioning=None,
            verbose=False,
            unconditional_guidance_scale=self.scale,
            eta=eta,
            x_T=x_T,
        )
        return samples, intermediates

    @torch.no_grad()
    def sample_sdf(
        self, n: int = 1, ddim_steps: int = 100, eta: float = 0.0, seed: Optional[int] = None
    ) -> torch.Tensor:
        """Unconditioned DDIM generation followed by VQ-VAE decoding."""
        samples, _ = self.sample_latent(n=n, ddim_steps=ddim_steps, eta=eta, seed=seed)
        return self.decode_latent(samples, quantize=True)

    @torch.no_grad()
    def inpaint_latent(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor,
        n: int = 1,
        ddim_steps: int = 100,
        eta: float = 0.0,
        x_T: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Official masked DDIM inpainting; the complete target latent is never input."""
        self.eval()
        if mask.shape[1] != 1:
            raise ValueError(f"mask must have one channel, got {tuple(mask.shape)}")
        mask = mask.to(device=self.device, dtype=torch.float32)
        if mask.dim() == 4:
            mask = mask.unsqueeze(0)
        if seed is not None:
            torch.manual_seed(seed)
        from models.networks.diffusion_networks.samplers.ddim import DDIMSampler

        sampler = DDIMSampler(self.model)
        samples, intermediates = sampler.sample(
            S=ddim_steps,
            batch_size=n,
            shape=self.z_shape,
            conditioning=None,
            verbose=False,
            x0=x0.to(self.device),
            mask=mask,
            unconditional_guidance_scale=self.scale,
            eta=eta,
            x_T=x_T,
        )
        return samples, intermediates


def load_sdfusion(
    device: str = "cuda",
    renderer: bool = False,
    verify_checksums: bool = True,
    verify_revision_strict: bool = True,
    paths: Optional[RuntimePaths] = None,
) -> SDFusionRuntime:
    """Load the official unconditioned SDFusion checkpoint via configured paths.

    ``device`` may be ``"cuda"``, ``"cuda:0"`` or ``"cpu"``; the official DDIM
    sampler hard-codes ``torch.device("cuda")`` buffers, so GPU sampling must
    use the default device / index 0.  ``renderer=False`` suppresses only the
    PyTorch3D camera construction used for notebook visualisation.
    """
    paths = paths or RuntimePaths.resolve()
    requested = torch.device(device)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        if requested.index not in (None, 0):
            raise ValueError(
                "The official DDIM sampler hard-codes cuda:0 buffers; use device='cuda'."
            )
    if not paths.df_config.is_file() or not paths.vq_config.is_file():
        raise FileNotFoundError(
            f"Official SDFusion config YAML files are missing under {paths.upstream / 'configs'}"
        )

    report: Dict[str, Any] = {
        "revision": verify_revision(paths, strict=verify_revision_strict),
        "checkpoints": {},
    }
    if verify_checksums:
        report["checkpoints"][VQVAE_CKPT_NAME] = verify_checkpoint(
            paths.vqvae_ckpt, EXPECTED_VQVAE_SHA256
        )
        report["checkpoints"][DIFFUSION_CKPT_NAME] = verify_checkpoint(
            paths.diffusion_ckpt, EXPECTED_DIFFUSION_SHA256
        )

    device_str = "cuda" if requested.type == "cuda" else str(requested)
    model, _ = _load_official_model(
        paths,
        device_str,
        paths.vqvae_ckpt,
        paths.diffusion_ckpt,
        renderer=renderer,
    )
    report["checkpoints"]["strict_load"] = {
        "vqvae": "ok (official load_vqvae, strict=True)",
        "diffusion": "ok (official load_ckpt, strict=True)",
    }
    return SDFusionRuntime(
        model=model,
        paths=paths,
        device=device_str,
        checkpoint_report=report,
        has_renderer=bool(renderer and getattr(model, "renderer", None) is not None),
    )
