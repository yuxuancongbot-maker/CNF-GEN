import time

import h5py
import numpy as np
import torch

K = 8
N_BASIS = 1 + 4 * K + 4 * K * K
NSAMP = 20
STEPS = 50
N_TEST = 1000


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


def boundary_points_uniform(n=20, size=128):
    m = n // 4
    xs = np.linspace(0, size - 1, m).astype(int)
    ys = np.linspace(0, size - 1, m + 2).astype(int)[1:-1]
    top = [(x, 0) for x in xs]
    bottom = [(x, size - 1) for x in xs]
    left = [(0, y) for y in ys]
    right = [(size - 1, y) for y in ys]
    return top + bottom + left + right


def interior_points_uniform(n=180, size=128):
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


class DarcyFlowModel(torch.nn.Module):
    def __init__(self, dim=N_BASIS, cond_dim=200, hidden=512):
        super().__init__()
        self.cond_net = torch.nn.Sequential(
            torch.nn.Linear(cond_dim, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
        )
        self.time_net = torch.nn.Sequential(
            torch.nn.Linear(32, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + hidden * 2, hidden * 2), torch.nn.SiLU(),
            torch.nn.Linear(hidden * 2, hidden * 2), torch.nn.SiLU(),
            torch.nn.Linear(hidden * 2, dim),
        )

    def forward(self, coefficients, time, cond):
        te = self.time_net(sinusoidal_embedding(time, 32))
        c = torch.cat([self.cond_net(cond), te], dim=-1)
        return self.net(torch.cat([coefficients, c], dim=-1))


def darcy_div(p, Kk):
    dp_dy, dp_dx = np.gradient(p)
    fx = Kk * dp_dx
    fy = Kk * dp_dy
    return np.gradient(fx, axis=1) + np.gradient(fy, axis=0)


def main():
    dev = "cuda"
    ckpt = torch.load("outputs/darcy_fm/darcy_fm.pt", map_location="cpu", weights_only=False)
    model = DarcyFlowModel(dim=ckpt["N_BASIS"], cond_dim=ckpt["cond_dim"], hidden=ckpt["hidden"]).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean = torch.tensor(ckpt["mean"], dtype=torch.float32, device=dev)
    std = torch.tensor(ckpt["std"], dtype=torch.float32, device=dev)
    H = ckpt["H"].astype(np.float64)
    ix = ckpt["constraint_ix"]
    iy = ckpt["constraint_iy"]

    Hn = H * std.cpu().numpy().astype(np.float64)
    Hn_d = torch.tensor(Hn, dtype=torch.float64, device=dev)
    Q_h, R_h = torch.linalg.qr(Hn_d.T)

    def project(x, b_norm_d):
        r = b_norm_d - x.double() @ Hn_d.T
        y = torch.linalg.solve_triangular(R_h.T, r.T, upper=False)
        return (x.double().T + Q_h @ y).T.float()

    n = 128
    gx = np.arange(n) / n
    Xf, Yf = np.meshgrid(gx, gx)
    P = phi_matrix(Xf.ravel(), Yf.ravel(), K)
    P_t = torch.tensor(P, dtype=torch.float32, device=dev)

    with h5py.File("datasets/pde/darcy.h5", "r") as f:
        p_test = f["tensor"][9000:10000, 0].astype(np.float32)
        K_test = f["nu"][9000:10000].astype(np.float32)

    acc = {"soft": {"res": 0.0, "mse": 0.0, "darcy": 0.0},
           "hybrid": {"res": 0.0, "mse": 0.0, "darcy": 0.0}}
    t0 = time.time()

    for i in range(N_TEST):
        p_gt = p_test[i]
        Kk = K_test[i]
        b = p_gt[iy, ix].astype(np.float32)
        cond = torch.tensor(b, device=dev).unsqueeze(0).repeat(NSAMP, 1)
        bn_np = b.astype(np.float64) - (H @ mean.cpu().numpy())
        bn_d = torch.tensor(bn_np, dtype=torch.float64, device=dev)

        for mode in ("soft", "hybrid"):
            gen_eval = torch.Generator(device=dev).manual_seed(123 + i)
            x = torch.randn(NSAMP, N_BASIS, generator=gen_eval, device=dev)
            for s in range(STEPS):
                t = torch.full((NSAMP,), s / STEPS, device=dev)
                with torch.no_grad():
                    v = model(x, t, cond)
                x = x + v / STEPS
                if mode == "hybrid":
                    x = project(x, bn_d)
            if mode == "hybrid":
                x = project(x, bn_d)
            c_phys = (x * std + mean).cpu().numpy()
            fields = c_phys @ P
            fields = fields.reshape(NSAMP, n, n)

            res = np.max(np.abs(H @ c_phys.T - b[:, None]), axis=0)
            mse = np.mean((fields - p_gt) ** 2, axis=(1, 2)) / np.mean(p_gt ** 2)
            R_gt = darcy_div(p_gt, Kk)
            denom = np.linalg.norm(R_gt)
            darcy = np.array([
                np.linalg.norm(darcy_div(fields[j], Kk) - R_gt) / denom
                for j in range(NSAMP)
            ])
            acc[mode]["res"] += res.mean()
            acc[mode]["mse"] += mse.mean()
            acc[mode]["darcy"] += darcy.mean()

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N_TEST}  {time.time()-t0:.0f}s", flush=True)

    print("\n=== results (mean over 1000 test fields x 20 samples) ===")
    print(f"{'metric':24s} {'Soft':>12s} {'Hybrid':>12s}")
    for key, label in [("res", "constraint residual"), ("mse", "rel MSE vs GT"), ("darcy", "Darcy residual")]:
        print(f"{label:24s} {acc['soft'][key]/N_TEST:12.4e} {acc['hybrid'][key]/N_TEST:12.4e}")

    np.save("outputs/darcy_fm/eval_summary.npy", {
        "soft": {k: v / N_TEST for k, v in acc["soft"].items()},
        "hybrid": {k: v / N_TEST for k, v in acc["hybrid"].items()},
    }, allow_pickle=True)
    print("saved outputs/darcy_fm/eval_summary.npy", flush=True)


if __name__ == "__main__":
    main()
