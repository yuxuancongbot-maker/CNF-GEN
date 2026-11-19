import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H5 = "datasets/pde/burgers.nu0.01.h5"
NU = 0.01
KX = 32
NX = 1 + 2 * KX
KT = 20
DIM = NX * KT
N_T = 20
N_XS = 50
N_COND = N_T * N_XS
STEPS = 100000
BATCH = 128
HIDDEN = 2048
SOFT_WEIGHT = 20.0
SEED = 0


def fourier_basis_1d(x, K):
    out = [np.ones_like(x)]
    for k in range(1, K + 1):
        out += [np.cos(2 * np.pi * k * x), np.sin(2 * np.pi * k * x)]
    return np.stack(out, 1)


def legendre_basis(t, K):
    from numpy.polynomial import legendre
    u = 2 * t - 1
    return legendre.legvander(u, K - 1)


def sinusoidal_embedding(t, dim=32):
    half = dim // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


class BurgersFlowModel(nn.Module):
    def __init__(self, dim=DIM, cond_dim=N_COND, hidden=HIDDEN):
        super().__init__()
        self.cond_net = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden), nn.SiLU())
        self.time_net = nn.Sequential(nn.Linear(32, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden), nn.SiLU())
        self.net = nn.Sequential(nn.Linear(dim + hidden * 2, hidden * 2), nn.SiLU(),
                                 nn.Linear(hidden * 2, hidden * 2), nn.SiLU(),
                                 nn.Linear(hidden * 2, dim))

    def forward(self, coefficients, time, cond):
        te = self.time_net(sinusoidal_embedding(time, 32))
        c = torch.cat([self.cond_net(cond), te], dim=-1)
        return self.net(torch.cat([coefficients, c], dim=-1))


def build_constraint(Phi, Psi, t_idx, x_idx):
    H_rows = []
    for tj in t_idx:
        for xi in x_idx:
            H_rows.append((Phi[xi, :, None] * Psi[tj, None, :]).reshape(DIM))
    return np.stack(H_rows)


def gather_values(u, t_idx, x_idx):
    out = np.zeros((u.shape[0], len(t_idx) * len(x_idx)), dtype=np.float32)
    r = 0
    for tj in t_idx:
        for xi in x_idx:
            out[:, r] = u[:, tj, xi]
            r += 1
    return out


def main():
    dev = "cuda"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Path("outputs").mkdir(parents=True, exist_ok=True)

    with h5py.File(H5, "r") as f:
        u = f["tensor"][...].astype(np.float32)
    N, T, X = u.shape
    n_train = 9000
    train = u[:n_train]
    print(f"burgers {u.shape}, train {train.shape}", flush=True)

    xg = np.arange(X) / X
    tg = np.arange(T) / T
    Phi = fourier_basis_1d(xg, KX)
    Psi = legendre_basis(tg, KT)
    norms_x = (Phi ** 2).sum(0)

    t0 = time.time()
    u_T = train.transpose(0, 2, 1)
    A = (Phi.T @ u_T) / norms_x[:, None]
    pinv_PsiT = np.linalg.pinv(Psi.T)
    C = A @ pinv_PsiT
    coeffs = C.reshape(n_train, DIM)
    print(f"fit coefficients {time.time()-t0:.1f}s", flush=True)
    rec = (Phi @ C[-1] @ Psi.T).T
    print(f"fit rel mse (last train) = {np.mean((rec-train[-1])**2)/np.mean(train[-1]**2):.2e}", flush=True)

    t_idx = np.linspace(0, T - 1, N_T).astype(int)
    x_idx = np.sort(np.random.default_rng(SEED).choice(X, N_XS, replace=False))
    H = build_constraint(Phi, Psi, t_idx, x_idx)
    b = gather_values(train, t_idx, x_idx)
    print(f"constraint {N_COND} points, cond(H)={np.linalg.cond(H):.2f}", flush=True)

    mean = coeffs.mean(0)
    std = coeffs.std(0).clip(1e-6)
    Xn = ((coeffs - mean) / std).astype(np.float32)
    print(f"coeff std range [{std.min():.2e}, {std.max():.2e}]", flush=True)

    Xt = torch.tensor(Xn, device=dev)
    Bt = torch.tensor(b, device=dev)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=dev)
    std_t = torch.tensor(std, dtype=torch.float32, device=dev)
    H_t = torch.tensor(H, dtype=torch.float32, device=dev)

    model = BurgersFlowModel().to(dev)
    print(f"params {sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-5)
    gen = torch.Generator(device="cpu").manual_seed(SEED + 1)
    gen_dev = torch.Generator(device=dev).manual_seed(SEED + 2)

    model.train()
    t0 = time.time()
    for step in range(STEPS + 1):
        idx = torch.randint(0, n_train, (BATCH,), generator=gen).to(dev)
        target = Xt[idx]
        noise = torch.randn(target.shape, generator=gen_dev, device=dev)
        tt = torch.rand(BATCH, generator=gen_dev, device=dev)
        interp = (1 - tt[:, None]) * noise + tt[:, None] * target
        velocity = target - noise
        pred = model(interp, tt, Bt[idx])
        flow_loss = F.mse_loss(pred, velocity)
        clean = interp + (1 - tt[:, None]) * pred
        physical = clean * std_t + mean_t
        p_obs = physical @ H_t.T
        soft_loss = F.mse_loss(p_obs, Bt[idx])
        loss = flow_loss + SOFT_WEIGHT * soft_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 5000 == 0:
            print(f"step {step:5d}  flow {flow_loss.item():.4f}  soft {soft_loss.item():.4f}  {time.time()-t0:.1f}s", flush=True)
    print(f"trained {time.time()-t0:.1f}s", flush=True)

    Hn = H * std.astype(np.float64)
    Hn_d = torch.tensor(Hn, dtype=torch.float64, device=dev)
    Hn_pinv = torch.linalg.pinv(Hn_d, rcond=1e-3)

    def project(x, bn_d):
        r = bn_d - x.double() @ Hn_d.T
        return (x.double() + r @ Hn_pinv.T).float()

    test = u[n_train:n_train + 100]
    b_test = gather_values(test, t_idx, x_idx)

    acc = {"soft": {"res": 0, "mse": 0, "burg": 0}, "hybrid": {"res": 0, "mse": 0, "burg": 0}}
    NSAMP = 20
    dx = 1.0 / X
    dt = 1.0 / T
    with torch.no_grad():
        for i in range(100):
            p_gt = test[i]
            bb = b_test[i]
            cond = torch.tensor(bb, device=dev).unsqueeze(0).repeat(NSAMP, 1)
            bn_np = bb.astype(np.float64) - (H @ mean.astype(np.float64))
            bn_d = torch.tensor(bn_np, dtype=torch.float64, device=dev)
            for mode in ("soft", "hybrid"):
                gen_eval = torch.Generator(device=dev).manual_seed(SEED + 3 + i)
                x = torch.randn(NSAMP, DIM, generator=gen_eval, device=dev)
                for s in range(50):
                    tt = torch.full((NSAMP,), s / 50, device=dev)
                    x = x + model(x, tt, cond) / 50
                    if mode == "hybrid":
                        x = project(x, bn_d)
                if mode == "hybrid":
                    x = project(x, bn_d)
                c_phys = (x * std_t + mean_t).cpu().numpy()
                Cmat = c_phys.reshape(NSAMP, NX, KT)
                tmp = Cmat @ Psi.T
                fields = tmp.transpose(0, 2, 1) @ Phi.T
                obs_vals = c_phys @ H.T
                res = np.max(np.abs(obs_vals - bb[None, :]), axis=1).mean()
                mse = np.mean((fields - p_gt) ** 2) / np.mean(p_gt ** 2)
                du_dx = np.gradient(fields, dx, axis=2)
                d2u = np.gradient(du_dx, dx, axis=2)
                du_dt = np.gradient(fields, dt, axis=1)
                R = du_dt + fields * du_dx - NU * d2u
                burg = np.mean(R ** 2)
                acc[mode]["res"] += res
                acc[mode]["mse"] += mse
                acc[mode]["burg"] += burg
    du_dx = np.gradient(p_gt, dx, axis=1)
    R_gt = np.gradient(p_gt, dt, axis=0) + p_gt * du_dx - NU * np.gradient(du_dx, dx, axis=1)
    burg_gt = np.mean(R_gt ** 2)

    print("\n=== results (100 test fields x 20 samples) ===")
    print(f"{'metric':24s} {'Soft':>12s} {'Hybrid':>12s}")
    for k, lab in [("res", "constraint residual"), ("mse", "rel MSE vs GT"), ("burg", "Burgers residual")]:
        print(f"{lab:24s} {acc['soft'][k]/100:12.4e} {acc['hybrid'][k]/100:12.4e}")
    print(f"{'GT Burgers residual':24s} {burg_gt:12.4e}")

    torch.save({"model": model.state_dict(), "mean": mean.astype(np.float32), "std": std.astype(np.float32),
                "KX": KX, "KT": KT, "Phi": Phi.astype(np.float32), "Psi": Psi.astype(np.float32),
                "t_idx": t_idx, "x_idx": x_idx, "H": H.astype(np.float32), "NU": NU},
               "outputs/burgers_fm.pt")
    print("saved outputs/burgers_fm.pt", flush=True)


if __name__ == "__main__":
    main()
