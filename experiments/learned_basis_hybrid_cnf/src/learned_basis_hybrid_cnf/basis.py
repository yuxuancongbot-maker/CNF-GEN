from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn as nn

FRAMES = 17
DEGREE = 10
JOINTS = 7
ANCHOR_FRAMES = [0, 4, 8, 12, 13, 16]
NONANCHOR_FRAMES = [i for i in range(FRAMES) if i not in ANCHOR_FRAMES]
S_CENTERS = torch.tensor(
    [0.0, 2.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 8.0 / 16.0, 10.0 / 16.0, 12.0 / 16.0, 13.0 / 16.0, 15.0 / 16.0, 16.0 / 16.0],
    dtype=torch.float64,
)

BRAW_MIN_SV_BOUND = 1e-6
HTIME_MIN_SV_BOUND = 1e-8
MAX_COND_BOUND = 1e6


def legendre_raw(x: torch.Tensor, degree: int = DEGREE) -> torch.Tensor:
    if degree == 1:
        return torch.ones(*x.shape, 1, dtype=x.dtype, device=x.device)
    vals = [torch.ones_like(x), x]
    for n in range(2, degree):
        vals.append(((2 * n - 1) * x * vals[-1] - (n - 1) * vals[-2]) / n)
    return torch.stack(vals[:degree], dim=-1)


def t_to_x(t: torch.Tensor) -> torch.Tensor:
    return 2.0 * t - 1.0


class HNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 8),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t.reshape(-1, 1)
        return self.net(x)


class NeuralKernelBasis(nn.Module):
    """Neural-kernel residual basis with Legendre zero-residual initialization and fixed-grid thin QR."""

    def __init__(self, ell: float = 0.2):
        super().__init__()
        self.degree = DEGREE
        self.frames = FRAMES
        self.joints = JOINTS
        self.ell = float(ell)
        self.register_buffer("ref_t", torch.linspace(0.0, 1.0, FRAMES, dtype=torch.float64))
        self.register_buffer("S", S_CENTERS.clone().to(torch.float64))
        self.hnet = HNet()
        self.W = nn.Parameter(torch.zeros(DEGREE, DEGREE, dtype=torch.float64))
        self.register_buffer("Bleg_norm_ref", torch.zeros(FRAMES, DEGREE, dtype=torch.float64))
        self._init_Bleg_norm_ref()

    def _init_Bleg_norm_ref(self) -> None:
        t = self.ref_t
        x = t_to_x(t)
        Braw = legendre_raw(x, self.degree).to(torch.float64)
        coln = Braw.norm(dim=0).clamp_min(1e-12)
        self.Bleg_norm_ref.copy_(Braw / coln)

    def phi(self, t: torch.Tensor) -> torch.Tensor:
        t64 = t.to(dtype=torch.float64, device=self.ref_t.device).reshape(-1)
        param_dtype = next(self.hnet.parameters()).dtype
        h = self.hnet(t64.to(dtype=param_dtype)).to(dtype=torch.float64)
        tcol = t64.unsqueeze(1)
        return torch.cat([tcol, 0.1 * h], dim=1)

    def K(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        ph_t = self.phi(t)
        ph_s = self.phi(s)
        d2 = ((ph_t[:, None, :] - ph_s[None, :, :]) ** 2).sum(dim=2)
        return torch.exp(-d2 / (2.0 * self.ell * self.ell))

    def _Bleg_norm_at(self, t: torch.Tensor) -> torch.Tensor:
        t64 = t.to(dtype=torch.float64, device=self.ref_t.device).reshape(-1)
        x = t_to_x(t64)
        Bleg_raw = legendre_raw(x, self.degree).to(dtype=torch.float64, device=self.ref_t.device)
        Bleg_ref_raw = legendre_raw(t_to_x(self.ref_t), self.degree).to(dtype=torch.float64, device=self.ref_t.device)
        coln = Bleg_ref_raw.norm(dim=0).clamp_min(1e-12)
        return Bleg_raw / coln

    def Braw_at(self, t: torch.Tensor) -> torch.Tensor:
        t64 = t.to(dtype=torch.float64, device=self.ref_t.device).reshape(-1)
        Kts = self.K(t64, self.S)
        Bleg_t = self._Bleg_norm_at(t64)
        return Bleg_t + Kts @ self.W

    def Braw_ref(self) -> torch.Tensor:
        return self.Braw_at(self.ref_t)

    def qr_normalize_ref(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable thin QR on fixed 17-point reference grid with positive R diagonal."""
        Braw = self.Braw_ref()  # [17, 10], float64
        Q, R = torch.linalg.qr(Braw, mode="reduced")  # Q: [17, 10], R: [10, 10]
        diagR = torch.diag(R)
        s = torch.where(diagR >= 0, torch.ones_like(diagR), -torch.ones_like(diagR))
        Qnew = Q * s.unsqueeze(0)  # scale columns of Q
        Rnew = s.unsqueeze(1) * R  # scale rows of R so Braw = Qnew @ Rnew
        Btheta_ref = Qnew
        return Btheta_ref, Rnew

    def Btheta_from_R(self, t: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Evaluate Btheta(t) = Braw(t) @ R^{-1} via triangular solve (R upper -> R.T lower)."""
        Braw_t = self.Braw_at(t)
        X = torch.linalg.solve_triangular(R.T, Braw_t.T, upper=False).T
        return X

    def forward_ref(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.qr_normalize_ref()

    def forward_B_at(self, t: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        return self.Btheta_from_R(t, R)

    def compute_r_num(
        self,
        Braw_ref: torch.Tensor | None = None,
        Btheta_ref: torch.Tensor | None = None,
        anchor_frames: list[int] = ANCHOR_FRAMES,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Differentiable R_num on Braw [17,10] and Htime [6,10] plus numerical health check."""
        if Braw_ref is None or Btheta_ref is None:
            Braw_ref = self.Braw_ref()
            Btheta_ref, _ = self.qr_normalize_ref()
        Htime = Btheta_ref[anchor_frames, :]  # [6, 10]
        sv_Braw = torch.linalg.svdvals(Braw_ref)
        sv_Htime = torch.linalg.svdvals(Htime)
        min_sv_Braw = float(sv_Braw[-1].detach().cpu().item()) if sv_Braw.numel() else 0.0
        max_sv_Braw = float(sv_Braw[0].detach().cpu().item()) if sv_Braw.numel() else 0.0
        min_sv_Htime = float(sv_Htime[-1].detach().cpu().item()) if sv_Htime.numel() else 0.0
        max_sv_Htime = float(sv_Htime[0].detach().cpu().item()) if sv_Htime.numel() else 0.0

        cond_Braw = sv_Braw[0] / sv_Braw[-1].clamp_min(1e-30)
        cond_Htime = sv_Htime[0] / sv_Htime[-1].clamp_min(1e-30)
        cond_Braw_val = float(cond_Braw.detach().cpu().item())
        cond_Htime_val = float(cond_Htime.detach().cpu().item())

        rank_Braw = int((sv_Braw.detach() > BRAW_MIN_SV_BOUND).sum().item())
        rank_Htime = int((sv_Htime.detach() > HTIME_MIN_SV_BOUND).sum().item())

        finite_ok = bool(
            torch.isfinite(Braw_ref).all().item()
            and torch.isfinite(Btheta_ref).all().item()
            and torch.isfinite(sv_Braw).all().item()
            and torch.isfinite(sv_Htime).all().item()
            and np.isfinite(cond_Braw_val)
            and np.isfinite(cond_Htime_val)
        )
        failed = (
            (not finite_ok)
            or (rank_Braw < self.degree)
            or (rank_Htime < len(anchor_frames))
            or (min_sv_Braw <= BRAW_MIN_SV_BOUND)
            or (min_sv_Htime <= HTIME_MIN_SV_BOUND)
            or (cond_Braw_val > MAX_COND_BOUND)
            or (cond_Htime_val > MAX_COND_BOUND)
        )
        reasons = []
        if not finite_ok:
            reasons.append("non_finite_basis_or_svd")
        if rank_Braw < self.degree or min_sv_Braw <= BRAW_MIN_SV_BOUND:
            reasons.append(f"Braw_degenerate(rank={rank_Braw},min_sv={min_sv_Braw:.3e})")
        if rank_Htime < len(anchor_frames) or min_sv_Htime <= HTIME_MIN_SV_BOUND:
            reasons.append(f"Htime_degenerate(rank={rank_Htime},min_sv={min_sv_Htime:.3e})")
        if cond_Braw_val > MAX_COND_BOUND:
            reasons.append(f"Braw_cond_exceeded({cond_Braw_val:.3e})")
        if cond_Htime_val > MAX_COND_BOUND:
            reasons.append(f"Htime_cond_exceeded({cond_Htime_val:.3e})")

        r_num = (torch.log(cond_Braw) ** 2) + (torch.log(cond_Htime) ** 2)
        w_frob_sq = torch.sum(self.W ** 2)
        stats = {
            "ok": not failed,
            "reasons": reasons,
            "Braw": {
                "shape": list(Braw_ref.shape),
                "rank": rank_Braw,
                "min_sv": min_sv_Braw,
                "max_sv": max_sv_Braw,
                "cond": cond_Braw_val,
                "singular_values": [float(x) for x in sv_Braw.detach().cpu().tolist()],
            },
            "Htime": {
                "shape": list(Htime.shape),
                "rank": rank_Htime,
                "min_sv": min_sv_Htime,
                "max_sv": max_sv_Htime,
                "cond": cond_Htime_val,
                "singular_values": [float(x) for x in sv_Htime.detach().cpu().tolist()],
            },
            "r_num": float(r_num.detach().cpu().item()),
            "w_frob_sq": float(w_frob_sq.detach().cpu().item()),
        }
        return r_num, w_frob_sq, stats

    @torch.no_grad()
    def diagnostics(self, anchor_frames: list[int] = ANCHOR_FRAMES) -> dict[str, Any]:
        Braw = self.Braw_ref()
        Btheta, R = self.qr_normalize_ref()
        _, _, stats = self.compute_r_num(Braw, Btheta, anchor_frames=anchor_frames)
        sv_B = torch.linalg.svdvals(Btheta)
        ortho_err = float((Btheta.T @ Btheta - torch.eye(self.degree, dtype=torch.float64, device=Btheta.device)).abs().max().cpu().item())
        qr_recon_err = float((Btheta @ R - Braw).abs().max().cpu().item())
        stats["Btheta"] = {
            "shape": list(Btheta.shape),
            "rank": int((sv_B > 1e-8).sum().item()),
            "min_sv": float(sv_B[-1].cpu().item()),
            "max_sv": float(sv_B[0].cpu().item()),
            "cond": float((sv_B[0] / sv_B[-1]).cpu().item()),
            "orthonormality_max_abs": ortho_err,
            "qr_reconstruction_max_abs": qr_recon_err,
            "R_diag_min": float(torch.diag(R).min().cpu().item()),
            "R_diag_max": float(torch.diag(R).max().cpu().item()),
        }
        return stats
