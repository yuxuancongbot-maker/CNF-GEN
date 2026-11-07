# 复现结果快照

> 本文件记录 `release/` 中 `scripts/reproduce_pickplace.sh` 的一次完整运行。
> 运行环境：Python 3.11.18，PyTorch 2.7.1+cu128，NVIDIA A100-SXM4-40GB，
> 2026-11-07。
>
> 数据生成是确定性的：新生成的 128+16 条轨迹与原始数据集逐元素一致
> （`max |Δq| = 0`）。因此下表与旧报告的小差异只可能来自视频 WM 初始化，
> 不来自数据或运动层。

## 1. 复现命令

```bash
cd release
PYTHONPATH=src python scripts/reproduce_pickplace.sh
```

产物：

- `runs/pickplace256/`、`runs/pickplace256_eval/`
- `runs/motion_model.pt`、`runs/soft_motion_model_w20.pt`
- `runs/video_wm.pt`
- `runs/controls_dense/`、`runs/controls_sparse/`
- `runs/eval_action_wm/metrics.json`

## 2. Dense waypoint（7 个 anchor）

约束帧：`[0, 3, 4, 6, 10, 13, 16]`；可选第 8 帧关节速度。
16 个 held-out 样本。

| 方法 | 关节 RMSE | 末端 RMSE | Jerk | 第 8 帧关节速度误差 |
|---|---:|---:|---:|---:|
| waypoint | **0.05087** | **0.03311** | **0.15588** | 0.19803 |
| velocity | 0.06681 | 0.04076 | 0.19902 | 0.21847 |
| posthoc | 0.06323 | 0.04896 | 0.18633 | 0.19718 |

配对差（velocity − waypoint）：

- 关节 RMSE `+0.01594 ± 0.00030`，0/16 更好；
- 末端 RMSE `+0.00765 ± 0.00020`，0/16 更好；
- jerk `+0.04314 ± 0.00375`，0/16 更好。

结论：**密集 anchor 下加第 8 帧速度约束反而全面变差**。位置约束已经
几乎决定整条轨迹，额外速度约束与低阶基函数冲突，产生 ringing。

## 3. Sparse waypoint + task-space velocity（6 个 anchor）

约束帧：`[0, 4, 8, 12, 13, 16]`；第 8 帧末端 3D 速度；
安全 IK 抬升在采样后执行。16 个 held-out 样本。

| 方法 | 关节 RMSE | 末端 RMSE | 末端速度误差 | Jerk | 第 8 帧关节速度误差 |
|---|---:|---:|---:|---:|---:|
| sparse waypoint | 0.06180 | 0.03760 | 0.53302 | **0.14422** | 0.25947 |
| **sparse velocity** | **0.05592** | **0.03339** | **0.11630** | 0.15966 | **0.22028** |
| sparse posthoc | 0.06238 | 0.04670 | 0.62247 | 0.19969 | 0.20884 |

配对差（sparse velocity − sparse waypoint）：

- 关节 RMSE `−0.00588 ± 0.00093`，**16/16 更好**；
- 末端 RMSE `−0.00421 ± 0.00099`，**14/16 更好**；
- 末端速度误差 `−0.41671 ± 0.03438`，**16/16 更好**；
- jerk `+0.01544 ± 0.00264`，1/16 更好（速度约束略增 jerk）。

结论：**稀疏 anchor + 任务空间速度是目前最干净的 velocity 正结果**；
代价是 jerk 略高。

## 4. Latent-level 6-DOF action WM

16 个 held-out 样本；视频 128×128 / 17 帧；PSNR 对 GT 视频。

| 方法 | PSNR | Anchor error | 关节 RMSE | 末端 RMSE | 速度误差 | Jerk |
|---|---:|---:|---:|---:|---:|---:|
| gt（oracle） | **34.58** | **0** | **0** | **0** | **0** | 0.2665 |
| unconstrained | 20.30 | 0.3495 | 0.1123 | 0.0973 | 0.8238 | 0.1898 |
| soft | 23.48 | 0.1320 | **0.0411** | 0.0396 | 0.2885 | 0.2048 |
| posthoc | 23.58 | 0.00131 | 0.0662 | 0.0422 | 0.1156 | 0.2061 |
| **every_step** | **23.99** | **0.00131** | 0.0587 | **0.0338** | **0.1156** | **0.1638** |

配对差（every_step − posthoc）：

- PSNR `+0.4045 ± 0.1989 dB`，10/16 更好；
- 关节 RMSE `−0.00744 ± 0.00466`，8/16 更好；
- 末端 RMSE `−0.00840 ± 0.00355`，12/16 更好；
- jerk `−0.04227 ± 0.00868`，**15/16 更好**。

配对差（every_step − soft）：

- PSNR `+0.5066 ± 0.1322 dB`，12/16 更好；
- Anchor error `−0.13074 ± 0.00885 rad`，**16/16 更好**；
- 关节 RMSE `+0.01758 ± 0.00203`，**0/16**（soft 更低）；
- 末端 RMSE `−0.00577 ± 0.00189`，12/16 更好；
- 速度误差 `−0.17290 ± 0.03395`，14/16 更好；
- jerk `−0.04097 ± 0.00665`，**16/16 更好**。

结论：

> **every-step 比 posthoc 更自然**（PSNR +0.40 dB、jerk 15/16、末端 RMSE 12/16）。
> **hard 的约束精度约 soft 的 100 倍**，但 soft 的关节 RMSE 更低；
> 这是“精确约束 vs 关节平滑”的 trade-off，不是全面碾压。

## 5. 与旧报告的差异

| 指标 | 旧报告 | 本次复现 |
|---|---:|---:|
| gt PSNR | 34.47 | 34.58 |
| every_step PSNR | 23.94 | 23.99 |
| posthoc PSNR | 23.53 | 23.58 |
| soft PSNR | 23.46 | 23.48 |
| every_step − posthoc PSNR | +0.41 | +0.40 |
| every_step − posthoc jerk 更好 | 15/16 | 15/16 |

差异均 `<0.2 dB`。视频 WM 的初始化在旧代码中没有固定全局 seed；
本发布版在 `train_video_wm()` 中显式设置 `torch.manual_seed(seed)` 和
cuDNN deterministic，因此后续复现会稳定在同一组数。

## 6. 已知的数值细节

1. **投影本身**满足 anchor 等式到 `<1e-6`；表中的 `anchor_error = 0.00131`
   是经过 `enforce_clearance()` 的安全 IK 抬升之后的值。
2. 第 8 帧上 `discrete_autograd` 与 `continuous` 的导数基一致到 `5e-7`，
   所以速度约束结论不依赖归一化项的微分方式。
3. `runs/controls_dense` 与 `runs/controls_sparse` 的轨迹指标与旧报告
   在 4 位小数上一致；latent-level 的 PSNR 有 `<0.2 dB` 的初始化差异。
