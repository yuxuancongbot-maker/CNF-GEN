# 05 · 复现指南

> **一句话**：装依赖 → 跑测试 → `scripts/reproduce_pickplace.sh` 一条命令
> 生成 128+16 条数据、训练 motion prior / soft baseline / 12.88M video WM、
> 生成 dense/sparse controls、跑 latent-level 评测。1.3B / PF_Wan 需要
> 外部仓库和权重，不在本 release 内。
>
> 本文是“分”文档：命令、种子、产物、运行时间、失败排查都写到可直接执行。

---

## 1. 环境

已验证环境：

| 项 | 值 |
|---|---|
| Python | 3.11.18 |
| PyTorch | 2.7.1+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA A100-SXM4-40GB |
| NumPy | 2.4.6 |
| PyBullet | 202010061 |
| OS | Linux |

CPU-only 可以跑单元测试、motion prior、soft baseline；video WM 和
latent-level 评测建议 CUDA。

### 1.1 安装

```bash
cd release
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

或者不安装、直接用 `PYTHONPATH=src`：

```bash
export PYTHONPATH=src
```

### 1.2 依赖

`pyproject.toml` / `requirements.txt`：

- numpy
- torch
- pybullet
- opencv-python
- imageio
- imageio-ffmpeg
- pytest（dev）

1.3B / PF_Wan 的额外依赖（VideoX-Fun、PhysisForcing、CoTracker、T5、VAE）
不在这里，见 [`04_wan_1_3b.md`](04_wan_1_3b.md)。

---

## 2. 冒烟测试

```bash
cd release
pytest
RUN_PYBULLET=1 pytest tests/test_scene_smoke.py
```

覆盖：

- Legendre 基 / 系数往返 / 导数模式；
- 位置 / 速度投影精确性、幂等性；
- 稀疏约束矩阵；
- PSNR / cube 检测 / paired delta；
- 3D U-Net forward / foreground weight；
- PyBullet 场景、IK、渲染、clearance。

---

## 3. 一键复现 6-DOF

```bash
cd release
PYTHON=python3 bash scripts/reproduce_pickplace.sh
```

脚本内容（与 `scripts/reproduce_pickplace.sh` 一致）：

```bash
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/generate_dataset.py --output runs/pickplace256 --count 128 --seed 0
python scripts/generate_dataset.py --output runs/pickplace256_eval --count 16 --seed 1
python scripts/train_motion_prior.py --data runs/pickplace256 --output runs/motion_model.pt
python scripts/train_soft_prior.py \
  --data runs/pickplace256 --output runs/soft_motion_model_w20.pt --soft-weight 20
python scripts/train_video_wm.py --data runs/pickplace256 --output runs/video_wm.pt
python scripts/generate_controls.py \
  --eval runs/pickplace256_eval --motion-prior runs/motion_model.pt \
  --output runs/controls_sparse --mode sparse --samples 16
python scripts/generate_controls.py \
  --eval runs/pickplace256_eval --motion-prior runs/motion_model.pt \
  --output runs/controls_dense --mode dense --samples 16
python scripts/eval_action_wm.py \
  --eval runs/pickplace256_eval --motion-prior runs/motion_model.pt \
  --soft-prior runs/soft_motion_model_w20.pt --video-wm runs/video_wm.pt \
  --output runs/eval_action_wm --samples 16 --device cuda
```

### 3.1 运行时间（A100 40GB / 本机实测）

| 步骤 | 时间 |
|---|---:|
| 数据生成 128+16 | 约 1–2 分钟 |
| motion prior 6000 步 | 约 43 秒（CPU） |
| soft prior 6000 步 | 约 46 秒（CPU） |
| video WM 4000 步 | 约 4 分钟（A100，deterministic） |
| dense/sparse controls | 约数分钟（CPU + PyBullet） |
| latent-level 评测 16×5 | 约数分钟（A100 + PyBullet） |

总时长约 15–25 分钟，取决于 CPU/PyBullet 速度。`runs/reproduce.log`
会记录每一步。

### 3.2 种子

| 项 | 值 |
|---|---:|
| 训练数据 | `seed=0, count=128` |
| held-out | `seed=1, count=16` |
| 每样本 RNG | `seed * 10000 + index` |
| motion prior | `seed=0` |
| soft prior | `seed=1` |
| video WM | `seed=0` |
| dense control | `1000 + index + attempt*100` |
| sparse control | `2000 + index + attempt*100` |
| unconstrained | `4000 + index` |
| soft sample | `5000 + index` |
| posthoc / every-step | `6000 + index` |

### 3.3 产物

```text
runs/
  pickplace256/                 # 128 训练样本
    train/ control/ q/ metadata.json
  pickplace256_eval/            # 16 held-out
  motion_model.pt               # 0.74M hard prior
  soft_motion_model_w20.pt      # 0.95M soft baseline
  video_wm.pt                   # 12.88M 3D U-Net
  controls_dense/               # 7 anchor + 第 8 帧关节速度
    metrics.json
    rgb/ control/ q/
  controls_sparse/              # 6 anchor + 第 8 帧任务空间速度
    metrics.json
    rgb/ control/ q/
  eval_action_wm/
    metrics.json
    videos/{sample}_{method}.mp4
```

`runs/` 已在 `.gitignore` 中排除。

### 3.4 预期数值

完整复现快照见 [`08_reproduction_numbers.md`](08_reproduction_numbers.md)。
关键值：

| 指标 | 值 |
|---|---:|
| dense waypoint joint RMSE | `0.05087` |
| dense velocity joint RMSE | `0.06681` |
| sparse waypoint joint RMSE | `0.06180` |
| sparse velocity joint RMSE | `0.05592` |
| sparse velocity vs waypoint 关节更好 | `16/16` |
| latent every_step PSNR | `23.99` |
| latent posthoc PSNR | `23.58` |
| every_step vs posthoc PSNR | `+0.40 dB` |
| every_step vs posthoc jerk 更好 | `15/16` |

数据和运动层是确定性的；latent PSNR 在固定 seed 后应稳定在
`±0.02 dB` 以内。不同 GPU / cuDNN 版本可能有 `<0.2 dB` 的差异。

---

## 4. 单独跑某一阶段

### 4.1 只生成数据

```bash
PYTHONPATH=src python scripts/generate_dataset.py \
  --output runs/pickplace256 --count 128 --seed 0
```

### 4.2 只训练 motion prior

```bash
PYTHONPATH=src python scripts/train_motion_prior.py \
  --data runs/pickplace256 --output runs/motion_model.pt
```

### 4.3 只训练 soft baseline

```bash
PYTHONPATH=src python scripts/train_soft_prior.py \
  --data runs/pickplace256 --output runs/soft_motion_model_w20.pt \
  --soft-weight 20
```

### 4.4 只训练 video WM

```bash
PYTHONPATH=src python scripts/train_video_wm.py \
  --data runs/pickplace256 --output runs/video_wm.pt \
  --steps 4000 --batch 2 --size 128 --device cuda
```

### 4.5 只生成 controls

```bash
PYTHONPATH=src python scripts/generate_controls.py \
  --eval runs/pickplace256_eval --motion-prior runs/motion_model.pt \
  --output runs/controls_sparse --mode sparse --samples 16
```

`--methods` 可选 `sparse_waypoint,sparse_velocity,sparse_posthoc`；
dense 模式默认 `waypoint,velocity,posthoc`。

### 4.6 只跑 latent 评测

```bash
PYTHONPATH=src python scripts/eval_action_wm.py \
  --eval runs/pickplace256_eval \
  --motion-prior runs/motion_model.pt \
  --soft-prior runs/soft_motion_model_w20.pt \
  --video-wm runs/video_wm.pt \
  --output runs/eval_action_wm --samples 16 --device cuda
```

### 4.7 汇总指标

```bash
PYTHONPATH=src python scripts/summarize_metrics.py \
  --metrics runs/eval_action_wm/metrics.json \
  --reference posthoc --methods every_step,soft,unconstrained \
  --keys psnr,anchor_error,joint_rmse,ee_rmse,ee_velocity_error,jerk
```

---

## 5. Ablation：导数基

默认 `--derivative-mode discrete_autograd`，与旧实验一致。
可切换：

```bash
--derivative-mode continuous
--derivative-mode finite_difference
```

例如 continuous 模式的 sparse 复跑：

```bash
PYTHONPATH=src python scripts/generate_controls.py \
  --eval runs/pickplace256_eval --motion-prior runs/motion_model.pt \
  --output runs/controls_sparse_continuous --mode sparse \
  --methods sparse_waypoint,sparse_velocity --samples 16 \
  --derivative-mode continuous
```

在第 8 帧，`discrete_autograd` 与 `continuous` 一致到 `5e-7`，所以
速度约束结论不依赖归一化项的微分方式；`finite_difference` 有离散误差。

---

## 6. 1.3B / PF_Wan（可选，不在 release 内）

需要外部仓库和权重：

1. `VideoX-Fun`：Wan2.1-Fun-1.3B-Control 训练 / 推理；
2. `PhysisForcing/pf_wan`：PF_Wan 1.3B / A14B、low-level head、CNF；
3. CoTracker、T5、VAE 权重；
4. 单卡 A100 40GB。

步骤：

1. 按外部仓库 README 安装；
2. 用 `experiments/wan22_fun_control/` 的双摆数据/control 训练 Control LoRA
   （rank 64 / alpha 32 / 1500 步）；
3. 用 `experiments/wan22_fun_control/run_fun_control_1_3b.py` 推理；
4. 用 `experiments/wan22_fun_control/eval_1_3b.py` 评测；
5. PF_Wan 1.3B：`PhysisForcing/pf_wan/configs/train/small_1_3b/train_cnf_1_3b.jsonc`；
6. 官方 A14B：`experiments/pf_wan_official_1gpu/`；
7. 具体数字和命令见 [`04_wan_1_3b.md`](04_wan_1_3b.md)。

> 这些实验的代码/权重不随 release 打包；release 只保证 6-DOF 合成数据
> 端到端可复现。不要在没有外部权重时声称 1.3B 结果可复现。

---

## 7. 排查

### 7.1 PyBullet / 渲染

- 必须使用 `p.connect(p.DIRECT)`；本 release 已封装；
- `imageio` 写 mp4 需要 `imageio-ffmpeg`；报 `ffmpeg` 错误时
  `pip install imageio-ffmpeg`；
- 无显示环境不需要 X11，PyBullet DIRECT 可跑；
- 如果 IK 失败，检查 `pybullet_data` 是否包含 `franka_panda/panda.urdf`。

### 7.2 CUDA / 显存

- video WM 训练 128×128 约几 GB；A100 40GB 足够；
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 可减少碎片；
- 显存不足时减小 `--batch` 或 `--size`，但会改变结果；
- 1.3B / A14B 的显存需求见 `04_wan_1_3b.md`。

### 7.3 数值不一致

- 数据生成是确定性的；如果 `q` 不一致，检查 NumPy 版本和
  `np.random.default_rng` 调用顺序；
- motion prior / soft prior 是 CPU 确定性；如果 loss 不一致，检查
  `torch.manual_seed` 和 `torch.Generator`；
- video WM 在不同 GPU / cuDNN 上可能有 `<0.2 dB` 差异；
- 不要把 `discrete_autograd` 和 `continuous` 混用于同一组对比。

### 7.4 速度约束没有收益

检查：

1. anchor 是否过密（dense 会变差）；
2. 速度是否任务空间（7 个关节速度不如 3D 末端速度）；
3. 速度估计是否可靠（含噪差分会被硬约束放大）；
4. 是否做了安全 IK 非穿透；
5. 是否在 `enforce_clearance` 后重新检查 anchor。

---

## 8. 最小复现路径

```bash
cd release
pip install -e ".[dev]"
pytest
RUN_PYBULLET=1 pytest tests/test_scene_smoke.py
PYTHONPATH=src python scripts/generate_dataset.py \
  --output runs/smoke --count 2 --seed 0
PYTHONPATH=src python scripts/generate_controls.py \
  --eval runs/smoke --motion-prior runs/motion_model.pt \
  --output runs/smoke_controls --mode sparse --samples 1
```

完整复现再跑 `scripts/reproduce_pickplace.sh`。
