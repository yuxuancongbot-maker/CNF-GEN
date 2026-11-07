# CNF Motion

**视频世界模型的测试时硬运动约束**

一句话：在测试时用一个 CNF 运动场，把稀疏 waypoint / 速度变成满足硬约束的关节轨迹 `q*(t)`，再通过 **input-level control video** 或 **latent-level motion latent** 两条接口交给视频世界模型；不重训视频模型主体（input-level 只训练轻量 Control LoRA）。

> **术语澄清**
>
> - **CNF = Constrained Neural Fields**（NeurIPS 2023，任意阶微分硬约束），**不是** Continuous Normalizing Flow。
> - 本仓库的 **latent-level** 指 **70 维 Legendre 系数向量 `c`**（7 关节 × 10 阶），约束直接作用在 `c` 上，再经 `q(t)=B(t)c` 变成轨迹；**不是** Wan / 3D U-Net 内部的 video latent。

## 方法（5 条）

1. **轨迹表示**：`q(t) = B(t)c`。17 帧、7 关节、10 阶归一化 Legendre 基，`c ∈ R^70`；速度用 `B'(t)c`。
2. **运动先验**：`p(c | q0)` 用 flow matching 训练（MLP，6000 步）；推理 50 步 Euler 采样，初始噪声尺度 0.5。
3. **精确投影**：把位置/速度约束堆成 `A c = b`，做最小范数硬投影 `c ← c − Aᵀ(AAᵀ + εI)⁻¹(Ac − b)`，`ε = 1e-8`；位置/速度残差到求解精度，不依赖 loss 权重。
4. **every-step**：在每个采样步都投影，而不是只在最后投影一次（posthoc）；最终投影始终保留，便于对照。
5. **视频接口**：input-level 把 `q*(t)` 转成 skeleton / segmentation control video，交给 Wan2.1-Fun-1.3B-Control（主体冻结 + Control LoRA）；latent-level 直接把约束加在 70 维运动系数 `c` 上，再送小型 action-conditioned video WM。

## 已验证什么（带精确数字）

- **6-DOF 位置硬约束**：16 条 held-out 轨迹上，dense/sparse 的**纯投影** anchor 误差 `< 1e-6`；桌面穿透 0，方块最终位置误差约 0。6-DOF latent-level 评测（含安全 IK 后处理）的端到端 anchor error 约 `0.00131 rad`（见 hard vs soft 一条）。
- **every-step vs posthoc（6-DOF latent-level）**：小型 3D U-Net action WM（128×128、17 帧、16 held-out）上，PSNR `23.99 vs 23.58`，配对 `+0.40 ± 0.20 dB`（10/16 更好）；jerk `0.1638 vs 0.2061`，**15/16** 更好。原始报告为 `23.94 vs 23.53`（`+0.41 dB`，11/16），差异 `<0.2 dB`。
- **hard vs soft**：anchor error `0.00131 vs 0.1320 rad`（约 100 倍），hard 的 PSNR `23.99 vs 23.48`、jerk 16/16 更好；但 **soft 的关节 RMSE 更低**（`0.0411 vs 0.0587`）。原始报告为 soft anchor `0.1315 rad`、joint `0.0410`；这是“精确约束 vs 关节平滑”的 trade-off，不是全面碾压。
- **稀疏 waypoint + 任务空间速度（6-DOF 轨迹级）**：相对 sparse waypoint，关节 RMSE `0.0559 vs 0.0618`（**16/16**）、末端 RMSE `0.0334 vs 0.0376`（**14/16**）、末端速度误差 `0.116 vs 0.533`（**16/16**）；代价是 jerk 略高（`0.1597 vs 0.1442`，1/16 更好）。
- **1.3B input-level Control LoRA（双摆）**：同一 GT control 下，waypoint EPE（control 跟随）`8.87 → 6.56 px`，PSNR `31.06 → 34.98 dB`（4 held-out）。6-DOF 上 base→LoRA 为 control EPE `24.85 → 6.65 px`、PSNR `22.99 → 24.32 dB`（8 held-out）。
- **PF_Wan 1.3B 100-step 视频负结果**：PhysisForcing + Wan2.1-T2V-1.3B 在 1 个 held-out clip 上把 low-level objective 从 soft 的 `0.01104` 降到 `0.00259`，但生成视频没有复现参考运动：动态轨迹 EPE `66.27 vs 73.29 px`，运动位移 `3.44 vs 2.32 px`（参考 `80.89 px`）。单 clip、单 seed，且 T2V backbone 不是真正的 I2V，不能据此否定 CNF 投影。

## 结果总表

| 测试台 | 设置 | n | 关键指标 | 结果 | 结论 |
|---|---|---:|---|---|---|
| 6-DOF 轨迹（dense） | 7 anchors + 第 8 帧关节速度；flow prior + 投影 | 16 | anchor / joint RMSE / jerk | anchor `<1e-6`；waypoint `0.0509 / 0.1559`，velocity `0.0668 / 0.1988`，posthoc `0.0632 / 0.1861` | 硬约束成立；dense velocity 反而更差 |
| 6-DOF 轨迹（sparse） | 6 anchors + 第 8 帧末端 3D 速度 | 16 | joint / EE / velocity / jerk | velocity vs waypoint：`0.0559 vs 0.0618`（16/16）、`0.0334 vs 0.0376`（14/16）、`0.116 vs 0.533`（16/16）、jerk `0.1597 vs 0.1442`（1/16） | 轨迹级正；jerk 略差 |
| 6-DOF latent-level | 12.88M 3D U-Net，128²，17 帧 | 16 | PSNR / anchor / jerk | every-step vs posthoc `23.99 vs 23.58 dB`（+0.40，10/16）、jerk `0.1638 vs 0.2061`（15/16）；hard vs soft anchor `0.00131 vs 0.1320`，soft joint RMSE `0.0411 vs 0.0587` | every-step 成立；hard/soft trade-off |
| 2-link latent-level（历史） | 64²，16 帧，12 维系数，3D U-Net | 128 视频 / 1024 轨迹 | pixel EPE / jerk / PSNR | every-step vs posthoc：`0.586 vs 0.689 px`、`2.07 vs 6.05`、`38.94 vs 28.35 dB` | every-step 强正 |
| 1.3B input-level（双摆） | Wan2.1-Fun-1.3B-Control + LoRA，256² | 4 | waypoint EPE / PSNR | base→LoRA `8.87→6.56 px`、`31.06→34.98 dB`；velocity vs waypoint `6.42 vs 6.65 px`、`34.31 vs 34.16 dB`（4/4） | LoRA 有效；velocity 小幅正 |
| 1.3B noisy（双摆） | 4 px waypoint 噪声；oracle vs estimated velocity | 8 | track EPE / PSNR / joint RMSE | oracle `5.26 / 32.96 / 0.091` vs waypoint `6.45 / 32.55 / 0.105`；estimated `6.82 / 32.11 / 0.147` | oracle 正；estimated 负 |
| PF_Wan 1.3B 100-step | PhysisForcing + Wan2.1-T2V-1.3B | 1 clip | low-level obj / video EPE | low-level `0.00259 vs 0.01104`；dynamic EPE `66.27 vs 73.29 px`；位移 `3.44 vs 2.32 px`（参考 `80.89`） | 视频负结果 |

表中 6-DOF latent-level 为 release 复跑值；dense/sparse 与 1.3B 为原始报告值。原始报告的 latent-level 值已在上面 bullet 中标注（PSNR `23.94 / 23.53`、`+0.41 dB`、11/16；soft anchor `0.1315 rad`、soft joint RMSE `0.0410`）。完整复跑与逐项对照见 [`docs/08_reproduction_numbers.md`](docs/08_reproduction_numbers.md)。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `src/cnf_motion/` | 核心实现：Legendre 基、flow prior、精确投影、稀疏约束、PyBullet 场景、video WM、指标 |
| `scripts/` | 数据生成、运动先验 / soft baseline / video WM 训练、control 生成、评测、端到端复现 |
| `tests/` | 单元测试：投影精确性、basis、metrics、video WM、scene smoke |
| `docs/` | 总览、方法、6-DOF/latent/1.3B 实验、复现、指标、限制、复现快照 |
| `configs/` | 报告实验的 documentation-only 配置参考（`pickplace.json`） |

`runs/`、`datasets/`、`outputs/`、checkpoints 和视频均由脚本生成，已在 `.gitignore` 中排除。

## 快速开始

在 `release/` 目录下：

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
PYTHON=python3 bash scripts/reproduce_pickplace.sh
```

预期产物（均在 `runs/` 下）：

- `pickplace256/`：128 条训练数据（RGB + control + q）
- `pickplace256_eval/`：16 条 held-out 数据
- `motion_model.pt`、`soft_motion_model_w20.pt`、`video_wm.pt`
- `controls_sparse/`、`controls_dense/`：`metrics.json` + `rgb/`、`control/`、`q/`
- `eval_action_wm/metrics.json` 和 `eval_action_wm/videos/`

端到端脚本只覆盖 6-DOF 合成数据；1.3B / PF_Wan 实验需要外部权重和仓库（见下）。

## 未包含

- **数据集 / checkpoint / 生成视频**：`datasets/`、`runs/`、`outputs/`、`*.pt`、`*.mp4` 不在仓库中；6-DOF 数据可由脚本重新生成。
- **外部 1.3B 权重与仓库**：Wan2.1-Fun、VideoX-Fun、PhysisForcing / PF_Wan 等不随 release 提供；1.3B 双摆、1.3B 6-DOF 和 PF_Wan 实验需要外部代码与权重，不能只靠本 release 复现。
- **真实机器人数据**：没有 BridgeData / DROID、真实接触力或摩擦；全部是 PyBullet 合成 pick-and-place。
- **外部 baseline**：ProjFlow、PDM、Wan-Move、IOI 等没有代码 / 权重 / license，未做定量对比。

## 文档

- [docs/00_overview.md](docs/00_overview.md)：总览、实验矩阵、claim 边界。
- [docs/01_method.md](docs/01_method.md)：Legendre 基、flow matching、硬投影、安全 IK。
- [docs/02_pickplace.md](docs/02_pickplace.md)：6-DOF 数据、dense/sparse 结果。
- [docs/03_latent_wm.md](docs/03_latent_wm.md)：12.88M action WM、soft baseline、latent 评测。
- [docs/04_wan_1_3b.md](docs/04_wan_1_3b.md)：1.3B input-level、含噪速度、PF_Wan。
- [docs/05_reproduction.md](docs/05_reproduction.md)：安装、命令、种子、产物、排查。
- [docs/06_metrics.md](docs/06_metrics.md)：指标公式与统计。
- [docs/07_limitations.md](docs/07_limitations.md)：负结果、限制、falsification criteria。
- [docs/08_reproduction_numbers.md](docs/08_reproduction_numbers.md)：本 release 的复现数值快照。
- [docs/README.md](docs/README.md)：文档索引和推荐阅读顺序。
- [configs/pickplace.json](configs/pickplace.json)：报告实验的配置参考。
- [scripts/reproduce_pickplace.sh](scripts/reproduce_pickplace.sh)：端到端复现脚本。
- [src/cnf_motion/](src/cnf_motion/)：核心实现代码。

> 逐样本报告和日志保留在开发仓库的 `experiments/` 下，不随 release 打包；本 README 与 `00_overview.md` 中的数字均来自这些报告。
