# 02 · 6-DOF Panda Pick-and-Place 实验

> **一句话**：在 PyBullet 里生成 Franka Panda 抓取-搬运-放置数据，用
> 10 阶 Legendre 运动场 + 硬投影生成满足稀疏 waypoint / 速度的 `q(t)`，
> 再交给 1.3B Control LoRA 渲染。位置约束精确；稀疏速度在轨迹级稳定
> 有益；但像素跟踪没有完全复现轨迹收益。
>
> 本文是“分”文档：场景、数据、约束、协议、结果、失败模式都写到
> 常数和命令级别。总览见 [`00_overview.md`](00_overview.md)。

---

## 1. 场景

| 项 | 值 |
|---|---|
| 机器人 | Franka Panda：7 个手臂关节（0–6）+ 2 个手指（9/10）；IK 末端 link 11 |
| 物体 | 5 cm 橙色方块，半边长 `0.025 m` |
| 桌面 | 高 `0.30 m`，中心 `(0.55, 0)`，半尺寸 `(0.55, 0.45)` |
| 相机 | target `[0.42,0.02,0.33]`，distance `0.92`，yaw `30°`，pitch `-24°` |
| 视频 | 256×256，17 帧，8 fps |
| 任务 | 抓取 → 抬起 → 搬运 → 放置 → 撤离 |

代码：`src/cnf_motion/scene.py::build_scene`、`config.py`。

---

## 2. 数据生成

### 2.1 随机化

每条轨迹：

| 变量 | 范围 |
|---|---|
| 方块初始 `x` | `0.42–0.50 m` |
| 方块初始 `y` | `-0.14–0.14 m` |
| 目标 `x` | `0.56–0.68 m` |
| 目标 `y` | `-0.18–0.18 m` |
| 抓取 yaw | `0` 或 `π/2` |
| 拒绝条件 | `||target - cube|| < 0.10 m` |

每条样本的 RNG：

```text
rng = np.random.default_rng(seed * 10000 + index)
```

训练：`seed=0, count=128`；held-out：`seed=1, count=16`。
生成是确定性的：新生成的 128+16 条轨迹与旧数据逐元素一致
（`max |Δq| = 0`）。

### 2.2 笛卡尔关键帧

`make_keys` 产生 8 个关键帧：

```text
0  : above(cube)
3  : grasp(cube)
4  : grasp(cube)
6  : above(cube)
10 : above(target)
12 : place(target)
13 : place(target)
16 : above(target)
```

`above` 的 `z = 0.50 m`；`grasp/place` 的 `z = TABLE_TOP + CUBE_HALF + 0.040`。

### 2.3 Minimum-jerk 插值

两帧之间：

```text
s = (frame - f0) / (f1 - f0)
smooth = s³ (10 - 15 s + 6 s²)
p(s) = p0 + (p1 - p0) * smooth
```

`smooth` 在两端一阶、二阶导数为 0，避免速度/加速度跳变。
代码：`scene.py::minimum_jerk`、`interpolate`。

### 2.4 IK

每一帧：

- 目标位置 `p_k`；
- 姿态 `[π, 0, yaw]`（夹爪朝下）；
- 初始 `prev_q = [0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.7]`；
- `calculateInverseKinematics`：`maxNumIterations=500`，
  `residualThreshold=1e-6`，`jointDamping=[0.05]*9`；
- rest pose 是上一帧 `prev_q + [0.0, 0.0]`，保证连续分支；
- IK 期间手指 `0.040`（张开）；
- 接受条件：位置残差 `<= 1e-3 m`，相邻帧最大关节变化 `<= 0.6 rad`；
- 失败则整条样本重采样（最多 20 次）。

代码：`scene.py::solve_ik`、`generate_one`。

### 2.5 抓取与渲染

- 抓取帧 `4–13`：手指闭合 `0.025`，方块刚性跟随手爪：
  `cube_pos = hand_pos + [0,0,-0.040]`，`cube_orn = hand_orn`；
- 其他帧：手指张开 `0.040`，方块停在桌面；
- RGB 与 segmentation control 同时渲染。

Control 颜色：

| 对象 | RGB |
|---|---|
| 桌面 | `(80, 80, 80)` |
| 方块 | `(255, 120, 20)` |
| 机械臂 | `(30, 80, 220)` |
| 夹爪（link 8–11） | `(30, 200, 120)` |

代码：`scene.py::render`、`render_trajectory`。

### 2.6 数据产物

```text
datasets/pickplace256/
  train/0000.mp4 ... 0127.mp4     # RGB
  control/0000.mp4 ... 0127.mp4   # segmentation
  q/0000.npy ... 0127.npy         # [17,7] float32
  metadata.json
```

生成命令：

```bash
PYTHONPATH=src python scripts/generate_dataset.py \
  --output runs/pickplace256 --count 128 --seed 0
PYTHONPATH=src python scripts/generate_dataset.py \
  --output runs/pickplace256_eval --count 16 --seed 1
```

---

## 3. 运动层设置

### 3.1 表示

- `T=17`，`J=7`，`D=10`；
- `c ∈ R^{70}`；
- 训练 128 条轨迹，flow prior 6000 步，CPU 约 43 秒；
- 推理 50 步 Euler，初始噪声 `N(0,0.5²)`。

### 3.2 Dense 约束

| 项 | 值 |
|---|---|
| anchor 帧 | `[0,3,4,6,10,13,16]` |
| 速度 | 第 8 帧，7 个关节速度 |
| every-step | 每个采样步投影 |
| posthoc | 只在最后投影 |
| 采样 seed | `1000 + index + attempt*100` |
| 重采样 | 最多 12 次 |
| 约束矩阵 | `A` 形状 `[56,70]` = 49 个位置行 + 7 个关节速度行 |

### 3.3 Sparse 约束

| 项 | 值 |
|---|---|
| anchor 帧 | `[0,4,8,12,13,16]` |
| 速度 | 第 8 帧，末端 3D 速度 |
| 任务空间速度 | `(p(q9)-p(q7))/2*16` |
| Jacobian 迭代 | 3 次 |
| 安全 IK | 抓取帧 `0.3655 m`，其他 `0.332 m` |
| 采样 seed | `2000 + index + attempt*100` |
| 约束矩阵 | 每次迭代 `A` 形状 `[45,70]` = 42 个位置行 + 3 个任务空间速度行 |
| Jacobian | 中心差分，`eps=1e-4` |
| clearance 行 | `CLEARANCE_FRAMES` 在代码中定义，但报告实验**没有**把它传入投影；非穿透由采样后的 `enforce_clearance` 保证 |

### 3.4 验证

- 关节限位（容差 `1e-4 rad`）；
- 最大关节步长 `< 0.8 rad`；
- 夹爪/桌面 clearance `> -0.001 m`；
- 不满足则重采样。

---

## 4. 结果：轨迹级

### 4.1 Dense

16 held-out，关节 RMSE / 末端 RMSE / jerk / 第 8 帧关节速度误差：

| 方法 | 关节 RMSE | 末端 RMSE | Jerk | 速度误差 | Anchor error | 桌面穿透 | 方块最终误差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| waypoint | **0.05087** | **0.03311** | **0.15588** | 0.19803 | 0.00131 | 0 | ~0 |
| velocity | 0.06681 | 0.04076 | 0.19902 | 0.21847 | 0.00131 | 0 | ~0 |
| posthoc | 0.06323 | 0.04896 | 0.18633 | 0.19718 | 0.00131 | 0 | ~0 |

> 纯投影的 anchor 残差 `<1e-6`；表中的 `0.00131 rad` 是经过
> `enforce_clearance()` 后的实际值。

配对（velocity − waypoint）：

- 关节 RMSE `+0.01594 ± 0.00030`，**0/16 更好**；
- 末端 RMSE `+0.00765 ± 0.00020`，**0/16 更好**；
- jerk `+0.04314 ± 0.00375`，**0/16 更好**。

结论：密集 anchor 下，位置约束已经几乎决定整条轨迹；额外强制
第 8 帧速度会与全局多项式基冲突，产生 ringing，整条轨迹反而变差。

### 4.2 Sparse

| 方法 | 关节 RMSE | 末端 RMSE | 末端速度误差 | Jerk | Anchor error | 桌面穿透 | 方块最终误差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse waypoint | 0.06180 | 0.03760 | 0.53302 | **0.14422** | 0.00131 | 0 | ~0 |
| **sparse velocity** | **0.05592** | **0.03339** | **0.11630** | 0.15966 | 0.00131 | 0 | ~0 |
| sparse posthoc | 0.06238 | 0.04670 | 0.62247 | 0.19969 | 0.00131 | 0 | ~0 |

配对（sparse velocity − sparse waypoint）：

- 关节 RMSE `−0.00588 ± 0.00093`，**16/16 更好**；
- 末端 RMSE `−0.00421 ± 0.00099`，**14/16 更好**；
- 末端速度误差 `−0.41671 ± 0.03438`，**16/16 更好**；
- jerk `+0.01544 ± 0.00264`，1/16 更好（略差）。

结论：**稀疏 anchor + 任务空间速度是目前最干净的 velocity 正结果**。
速度约束需要三个条件同时满足：anchor 稀疏、任务空间表示、安全 IK 非穿透。

### 4.3 every-step vs posthoc

Dense：

- every-step（waypoint）关节 RMSE `0.05087` vs posthoc `0.06323`；
- jerk `0.15588` vs `0.18633`；
- 15/16 样本 RMSE 更好、15/16 jerk 更低。

Sparse：

- sparse velocity `0.05592` vs sparse posthoc `0.06238`；
- 末端 RMSE `0.03339` vs `0.04670`；
- jerk `0.15966` vs `0.19969`。

---

## 5. 结果：像素级（1.3B Control LoRA）

8 held-out，256×256，Wan2.1-Fun-1.3B-Control + LoRA：

| 方法 | PSNR | cube track EPE | control EPE | final error | cube jerk | n |
|---|---:|---:|---:|---:|---:|---:|
| base（无 LoRA） | 22.99 | 24.80 | 24.85 | 14.74 | 71.81 | 4 |
| gt + LoRA | **24.32** | **6.28** | **6.65** | **2.92** | **28.50** | 8 |
| waypoint | 23.08 | 8.07 | 8.56 | 5.48 | 29.62 | 8 |
| velocity | **23.32** | **6.94** | **7.32** | 3.62 | 30.66 | 8 |
| posthoc | 22.53 | 7.97 | 8.43 | 3.20 | 30.38 | 8 |

配对（velocity − waypoint）：

- PSNR `+0.24 ± 0.05 dB`，**8/8 更好**；
- cube track `−1.13 ± 0.37 px`，7/8 更好；
- control EPE `−1.24 ± 0.41 px`，7/8 更好；
- final cube error `−1.86 ± 1.94 px`，4/8（不显著）。

Dense 像素的其他 pair（相对 waypoint）：

- posthoc：PSNR `−0.55 ± 0.22 dB`（2/8）、cube track `−0.10 ± 0.13 px`（5/8）、
  control `−0.13 ± 0.13 px`（4/8）、final `−2.27 ± 1.86 px`（6/8）；
- gt + LoRA：PSNR `+1.24 ± 0.20 dB`（8/8）、cube track `−1.79 ± 0.30 px`（8/8）、
  control `−1.91 ± 0.27 px`（8/8）、final `−2.55 ± 1.73 px`（6/8）。

GT control ceiling 是 `24.32 dB / 6.65 px`；velocity 与它的差距是
`1.00 dB / 0.67 px`，waypoint 与它的差距是 `1.24 dB / 1.91 px`。

Sparse 像素级（8 held-out）：

| 方法 | PSNR | cube track EPE | control EPE | final error | cube jerk |
|---|---:|---:|---:|---:|---:|
| sparse waypoint | 22.22 | **5.29** | **7.07** | 3.39 | 26.35 |
| **sparse velocity** | **22.38** | 5.49 | 7.18 | **3.19** | **24.15** |
| sparse posthoc | 21.86 | 6.93 | 7.59 | 3.38 | 26.94 |

配对（sparse velocity − sparse waypoint）：

- PSNR `+0.16 ± 0.07 dB`，**8/8 更好**；
- cube track `+0.20 ± 0.36 px`，3/8 更好（略差，不显著）；
- control EPE `+0.11 ± 0.19 px`，3/8 更好（略差，不显著）；
- final error `−0.20 ± 0.54 px`，4/8（不显著）。

结论：

> **轨迹级的 16/16 正收益没有干净地转化成像素级跟踪收益。**
> velocity 在 PSNR 上稳定更好，但 cube tracking / control EPE 未复现；
> 渲染器仍是瓶颈。

---

## 6. 关键解释

### 6.1 Base vs LoRA

- base 完全不跟 segmentation control：control EPE `24.85 px`；
- LoRA 把它降到 `6.65 px`，PSNR `22.99 → 24.32 dB`；
- 结论：**1.3B Control 对 6-DOF segmentation control 仍是零样本失败，
  LoRA 必要且有效。**

### 6.2 为什么 dense velocity 是负结果

7 个 anchor × 7 关节 = 49 个位置等式，已经接近 70 维系数的自由度；
再加强制第 8 帧速度，会与低阶全局基冲突。最小范数投影只能在
零空间里调整，产生 ringing，整条轨迹 RMSE 和 jerk 都变差。

### 6.3 为什么 sparse velocity 是正结果

- anchor 从 49 个等式降到 42 个，留出更多零空间；
- 任务空间速度只约束 3 个方向，而不是 7 个关节；
- 安全 IK 保证非穿透；
- 速度约束把整条轨迹拉回 GT 的动力学，而不是只修一帧。

### 6.4 为什么像素跟踪没有复现

- 渲染器在 256×256、17 帧、合成数据上仍有明显误差；
- GT control 自身的 track EPE 就有 `2.35 px`（双摆）或 `6.28 px`（6-DOF）；
- 像素检测用颜色阈值，受遮挡、光照、关节帽影响；
- 更准的轨迹不一定被视频模型更准确地渲染。

---

## 7. 复现命令

```bash
cd release
PYTHONPATH=src python scripts/generate_dataset.py \
  --output runs/pickplace256 --count 128 --seed 0
PYTHONPATH=src python scripts/generate_dataset.py \
  --output runs/pickplace256_eval --count 16 --seed 1
PYTHONPATH=src python scripts/train_motion_prior.py \
  --data runs/pickplace256 --output runs/motion_model.pt
PYTHONPATH=src python scripts/train_soft_prior.py \
  --data runs/pickplace256 --output runs/soft_motion_model_w20.pt --soft-weight 20
PYTHONPATH=src python scripts/generate_controls.py \
  --eval runs/pickplace256_eval --motion-prior runs/motion_model.pt \
  --output runs/controls_sparse --mode sparse --samples 16
PYTHONPATH=src python scripts/generate_controls.py \
  --eval runs/pickplace256_eval --motion-prior runs/motion_model.pt \
  --output runs/controls_dense --mode dense --samples 16
```

完整脚本：`scripts/reproduce_pickplace.sh`。
逐样本复现快照：`docs/08_reproduction_numbers.md`。

原始实验产物（开发仓库，不随 release 打包）：

- `experiments/cnf_pickplace/make_dataset.py`
- `experiments/cnf_pickplace/generate_controls.py`
- `experiments/cnf_pickplace/generate_sparse_controls.py`
- `experiments/cnf_pickplace/report_pickplace.md`
- `experiments/cnf_pickplace/review/`
