"""Small 3D U-Net video world model conditioned on control video and q.

This is the 12.88M-parameter latent-level testbed.  It predicts the whole RGB
clip from the first frame, the semantic control clip, and the joint trajectory.
The motion latent is the Legendre coefficient vector ``c``; hard projection is
applied to ``c`` before it is converted to ``q`` and fed to this model.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import FRAMES, JOINTS
from .dataset import load_video_split


def conv_block(in_channels: int, out_channels: int):
    return torch.nn.Sequential(
        torch.nn.Conv3d(in_channels, out_channels, 3, padding=1),
        torch.nn.GroupNorm(8, out_channels),
        torch.nn.SiLU(),
        torch.nn.Conv3d(out_channels, out_channels, 3, padding=1),
        torch.nn.GroupNorm(8, out_channels),
        torch.nn.SiLU(),
    )


class VideoWorldModel6D(torch.nn.Module):
    """Residual 3D U-Net: ``x = first_frame + tanh(decoder)``."""

    def __init__(self, hidden: int = 32, joints: int = JOINTS):
        super().__init__()
        self.joints = joints
        in_channels = 3 + 3 + 3 + joints * 3
        self.enc0 = conv_block(in_channels, hidden)
        self.down1 = torch.nn.Conv3d(hidden, 64, 3, stride=(1, 2, 2), padding=1)
        self.enc1 = conv_block(64, 64)
        self.down2 = torch.nn.Conv3d(64, 128, 3, stride=(1, 2, 2), padding=1)
        self.enc2 = conv_block(128, 128)
        self.down3 = torch.nn.Conv3d(128, 256, 3, stride=(2, 2, 2), padding=1)
        self.enc3 = conv_block(256, 256)
        self.bottleneck = conv_block(256, 256)
        self.up3 = torch.nn.ConvTranspose3d(
            256, 128, (3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)
        )
        self.dec2 = conv_block(256, 128)
        self.up2 = torch.nn.ConvTranspose3d(
            128, 64, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.dec1 = conv_block(128, 64)
        self.up1 = torch.nn.ConvTranspose3d(
            64, hidden, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.dec0 = conv_block(hidden * 2, hidden)
        self.out = torch.nn.Conv3d(hidden, 3, 1)

    def forward(self, first_frame, control, q):
        batch, frames, _ = q.shape
        height, width = first_frame.shape[-2:]
        first = first_frame.unsqueeze(2).expand(-1, -1, frames, -1, -1)
        control0 = control[:, :, :1].expand(-1, -1, frames, -1, -1)
        q_features = torch.cat([q, torch.sin(q), torch.cos(q)], dim=-1)
        q_features = q_features.permute(0, 2, 1)[:, :, :, None, None]
        q_features = q_features.expand(-1, -1, -1, height, width)
        condition = torch.cat([first, control, control0, q_features], dim=1)
        x0 = self.enc0(condition)
        x1 = self.enc1(self.down1(x0))
        x2 = self.enc2(self.down2(x1))
        x3 = self.enc3(self.down3(x2))
        x = self.bottleneck(x3)
        x = self.dec2(torch.cat([self.up3(x), x2], dim=1))
        x = self.dec1(torch.cat([self.up2(x), x1], dim=1))
        x = self.dec0(torch.cat([self.up1(x), x0], dim=1))
        residual = torch.tanh(self.out(x))
        return (first + residual).clamp(0.0, 1.0).permute(0, 2, 1, 3, 4)


def foreground_weight(control: torch.Tensor) -> torch.Tensor:
    """Pixel weight ``1 + 10 * foreground``; control is ``[B, 3, T, H, W]``."""
    foreground = control.float().amax(dim=1) > 0.02
    return 1.0 + 10.0 * foreground.float()


def train_video_wm(
    data: str | Path,
    output: str | Path,
    *,
    count: int = 128,
    steps: int = 4000,
    batch: int = 2,
    lr: float = 2e-4,
    hidden: int = 32,
    size: int = 128,
    seed: int = 0,
    device: str = "cuda",
    log_every: int = 400,
) -> dict:
    """Train the action-conditioned video world model.

    Loss: ``weighted MSE + 0.1 * L1``, with foreground pixels weighted 11x.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rgb, control, q = load_video_split(data, count=count, size=size)
    print("data", tuple(rgb.shape), tuple(control.shape), tuple(q.shape), flush=True)
    rgb = rgb.to(device)
    control = control.to(device)
    q = q.to(device)

    model = VideoWorldModel6D(hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.05
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    model.train()
    start = time.time()
    for step in range(steps):
        index = torch.randint(
            0, rgb.shape[0], (batch,), device=device, generator=generator
        )
        target = rgb[index].float().permute(0, 1, 4, 2, 3) / 255.0
        control_batch = control[index].float().permute(0, 4, 1, 2, 3) / 255.0
        q_batch = q[index]
        first = target[:, 0]
        prediction = model(first, control_batch, q_batch)
        weight = foreground_weight(control_batch)
        loss = (
            weight[:, :, None] * (prediction - target).square()
        ).mean() / weight.mean()
        loss = loss + 0.1 * F.l1_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % max(1, log_every) == 0:
            print(
                step + 1,
                "loss",
                round(float(loss), 6),
                "elapsed",
                round(time.time() - start, 1),
                flush=True,
            )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "hidden": hidden,
        "size": size,
        "frames": FRAMES,
        "steps": steps,
    }
    torch.save(checkpoint, output)
    print("saved", output)
    return checkpoint


def load_video_wm(path: str | Path, device: str | torch.device = "cpu"):
    """Load the video world model; returns ``(model, checkpoint)``."""
    device = torch.device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = VideoWorldModel6D(hidden=int(checkpoint.get("hidden", 32))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def predict_video(model, first_frame, control, q):
    """Run the model; inputs are ``[B,3,H,W]``, ``[B,3,T,H,W]``, ``[B,T,J]``."""
    return model(first_frame, control, q)


__all__ = [
    "VideoWorldModel6D",
    "conv_block",
    "foreground_weight",
    "train_video_wm",
    "load_video_wm",
    "predict_video",
]
