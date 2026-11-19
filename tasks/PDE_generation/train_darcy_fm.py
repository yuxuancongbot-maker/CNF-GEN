import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H5 = "datasets/pde/darcy.h5"
K = 8
N_BASIS = 1 + 4 * K + 4 * K * K
N_BOUNDARY = 20
N_INTERIOR = 180
N_COND = N_BOUNDARY + N_INTERIOR
STEPS = 50000
BATCH = 128
HIDDEN = 512
SOFT_WEIGHT = 20.0
LR = 1e-3
SEED = 0
OUT = Path("outputs/darcy_fm")


def phi_matrix(x, y, K):
    out = [np.ones_like(x)]
    for k in range(1, K + 1):
        out += [np.cos(2 * np.pi * k * x), np.sin(2 * np.pi * k * x)]
        out += [np.cos(2 * np.pi * k * y), np.sin(2 * np.pi * k * y)]
    for k in range(1, K + 1):
        for l in range(1, K + 1):
            kx = 2 * np.pi * k * x
            ly = 2 * np.pi * l * y
            out += [np.cos(kx) * np.cos(ly), np.cos(kx) * np.sin(ly),
                    np.sin(kx) * np.cos(ly), np.sin(kx) * np.sin(ly)]
    return np.stack(out, 0)


def boundary_points_uniform(n=N_BOUNDARY, size=128):
    m = n // 4
    xs = np.linspace(0, size - 1, m).astype(int)
    ys = np.linspace(0, size - 1, m + 2).astype(int)[1:-1]
    top = [(x, 0) for x in xs]
    bottom = [(x, size - 1) for x in xs]
    left = [(0, y) for y in ys]
    right = [(size - 1, y) for y in ys]
    return top + bottom + left + right


def interior_points_uniform(n=N_INTERIOR, size=128):
    a = int(np.round(np.sqrt(n)))
    while a > 0 and n % a != 0:
        a -= 1
    b = n // a
    xs = np.linspace(0, size - 1, a + 2).astype(int)[1:-1]
    ys = np.linspace(0, size - 1, b + 2).astype(int)[1:-1]
    return [(x, y) for y in ys for x in xs]


def sinusoidal_embedding(t, dim=32):
    half = dim // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


class DarcyFlowModel(nn.Module):
    def __init__(self, dim=N_BASIS, cond_dim=N_COND, hidden=HIDDEN):
        super().__init__()
        self.dim = dim
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.time_net = nn.Sequential(
            nn.Linear(32, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + hidden * 2, hidden * 2), nn.SiLU(),
            nn.Linear(hidden * 2, hidden * 2), nn.SiLU(),
            nn.Linear(hidden * 2, dim),
        )

    def forward(self, coefficients, time, cond):
        te = self.time_net(sinusoidal_embedding(time, 32))
        c = torch.cat([self.cond_net(cond), te], dim=-1)
        return self.net(torch.cat([coefficients, c], dim=-1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with h5py.File(H5, "r") as f:
        u = f["tensor"][:, 0].astype(np.float32)
    n = u.shape[1]
    n_train = 9000
    train = u[:n_train]
    print(f"data {u.shape}, train {train.shape}, grid {n}", flush=True)

    gx = np.arange(n) / n
    Xf, Yf = np.meshgrid(gx, gx)
    P = phi_matrix(Xf.ravel(), Yf.ravel(), K)
    norms = (P**2).sum(axis=1)
    t0 = time.time()
    C = (train.reshape(n_train, -1) @ P.T) / norms[None, :]
    print(f"fit coefficients {time.time()-t0:.1f}s", flush=True)
    rec = (P.T @ C[-1]).reshape(n, n)
    print(f"fit rel mse (last train) = {np.mean((rec-train[-1])**2)/np.mean(train[-1]**2):.2e}", flush=True)

    bpts = boundary_points_uniform()
    ipts = interior_points_uniform()
    pts = bpts + ipts
    ix = np.array([p[0] for p in pts])
    iy = np.array([p[1] for p in pts])
    H = phi_matrix(ix / n, iy / n, K).T
    b = train[:, iy, ix]
    print(f"constraints {len(pts)}, cond(H)={np.linalg.cond(H):.2f}", flush=True)

    mean = C.mean(0)
    std = C.std(0).clip(1e-6)
    Xn = ((C - mean) / std).astype(np.float32)
    b_np = b.astype(np.float32)
    print(f"coeff std range [{std.min():.2e}, {std.max():.2e}]", flush=True)

    device = "cuda"
    X = torch.tensor(Xn, device=device)
    B = torch.tensor(b_np, device=device)
    H_t = torch.tensor(H, dtype=torch.float32, device=device)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)

    model = DarcyFlowModel().to(device)
    print(f"params {sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=LR * 0.05)
    gen = torch.Generator(device="cpu").manual_seed(SEED + 1)
    gen_dev = torch.Generator(device=device).manual_seed(SEED + 2)

    model.train()
    t0 = time.time()
    for step in range(STEPS + 1):
        idx = torch.randint(0, n_train, (BATCH,), generator=gen).to(device)
        target = X[idx]
        noise = torch.randn(target.shape, generator=gen_dev, device=device)
        time_ = torch.rand(BATCH, generator=gen_dev, device=device)
        interpolated = (1 - time_[:, None]) * noise + time_[:, None] * target
        velocity = target - noise
        pred = model(interpolated, time_, B[idx])
        flow_loss = F.mse_loss(pred, velocity)
        clean = interpolated + (1 - time_[:, None]) * pred
        physical = clean * std_t + mean_t
        p_constraint = physical @ H_t.T
        soft_loss = F.mse_loss(p_constraint, B[idx])
        loss = flow_loss + SOFT_WEIGHT * soft_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 5000 == 0:
            print(f"step {step:5d}  flow {flow_loss.item():.4f}  soft {soft_loss.item():.4f}  "
                  f"loss {loss.item():.4f}  {time.time()-t0:.1f}s", flush=True)

    print(f"trained {time.time()-t0:.1f}s", flush=True)

    ckpt = {
        "model": model.state_dict(),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "K": K,
        "N_BASIS": N_BASIS,
        "hidden": HIDDEN,
        "cond_dim": N_COND,
        "constraint_ix": ix,
        "constraint_iy": iy,
        "H": H.astype(np.float32),
        "n_boundary": N_BOUNDARY,
        "n_interior": N_INTERIOR,
        "soft_weight": SOFT_WEIGHT,
        "steps": STEPS,
        "n_train": n_train,
        "seed": SEED,
    }
    torch.save(ckpt, OUT / "darcy_fm.pt")
    print("saved", OUT / "darcy_fm.pt", flush=True)


if __name__ == "__main__":
    main()
