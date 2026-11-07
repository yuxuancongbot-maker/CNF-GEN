# 00 · 总览：CNF 运动场 + 视频世界模型

> 本文件是 release 的总览文档。项目入口见 [`../README.md`](../README.md)，文档索引见 [`README.md`](README.md)。

## 0. 一句话

在测试时，把一个 **CNF（Constrained Neural Fields）运动场**约束到稀疏 waypoint / 速度上，得到满足硬约束的关节轨迹 `q*(t)`；再通过 **input-level control video** 或 **latent-level motion latent** 两条接口驱动视频世界模型，不重训主体。

CNF 不是 Continuous Normalizing Flow。本项目的 latent-level 也不是视频模型的内部 latent，而是 **70 维 Legendre 系数 `c`**（7 关节 × 10 阶）。

---

## 1. 问题设定：为什么是稀疏约束

### 1.1 标准 action-conditioned WM 假设 dense action

常见的 action-conditioned 视频世界模型按下面的方式工作：

```text
训练：视频 + dense action 序列 → p(video | action)
推理：首帧 + dense action 序列 → 视频
```

这里的 “dense” 是相对 sparse waypoint 而言，指逐帧或高频采样的动作序列；真实 action 仍然是离散采样，不是连续时间函数。模型靠数据学习“怎么跟随 action”，没有硬约束保证；换动作空间通常要重训或加 adapter。

### 1.2 测试时通常只有稀疏 waypoint

真实的高层接口往往不会给出 5–50 Hz 的整段关节动作，而是给：

- 抓取点 / 放置点；
- 一个 8–16 步的 VLA action chunk；
- 人点的 3D 目标点；
- 工业机械臂的 moveJ / moveL 关键点；
- 动画师的 keyframe。

这些命令在时间上是**稀疏、低频、不规则**的。中间的运动需要低层运动先验或规划器补全。

### 1.3 本项目的 setting

```text
输入：首帧 x0 / 初始关节角 q0 + 稀疏 waypoint（+ 可选速度）
输出：整条关节轨迹 q(t) + 视频
不要求：dense action
不重训：视频模型主体
```

这不是要替代 action-conditioned WM，而是补在“高层稀疏接口”和“视频世界模型”之间的运动层。

### 1.4 稀疏约束为什么难

- **欠定**：6 个 anchor × 7 关节 = 42 个位置等式，再加第 8 帧 3 个末端速度等式，仍少于 70 维系数；满足约束的轨迹有很多条。
- **多解**：dense GT 只是其中一条。用 GT 算 PSNR / joint RMSE 会惩罚其他合法、甚至更自然的解，所以 GT 是 **oracle reference / rendering ceiling**，不是方法的理论上限。
- **需要先验**：硬约束只保证“命中 waypoint”，不保证轨迹自然；自然性由 flow prior 提供。
- **约束来源**：`q*_k` 可以由关节 waypoint 直接给出；也可以由 Cartesian waypoint 经 IK 得到；实验里还用 GT 关节角作为 oracle anchor，以隔离投影误差与感知/IK 误差。三种情况最后都是线性约束 `B(t_k)c = q*_k`。
- **速度约束有条件**：可靠速度 + 稀疏 waypoint 时收益稳定；密集 anchor 或从含噪低频 action 差分估计速度时，硬约束会把误差放大。

---

## 2. 两条接口

项目里有两条接入视频世界模型的路径。它们在“约束在哪里生效”和“视频模型看到什么”上不同，不能混为一谈。

### 2.1 input-level：CNF → control video → 视频模型

```text
首帧 x0 + q0 + 稀疏 waypoint / 速度
        │
        ▼
CNF 运动场：q(t) = B(t)c，flow prior p(c | q0)
        │  every-step 硬投影 A c = b
        ▼
q*(t) ──► skeleton / segmentation control video
        │
        ▼
Wan2.1-Fun-1.3B-Control（主体冻结 + Control LoRA）
        │
        ▼
生成视频
```

- 硬约束在**运动空间**（`q` / `c`）成立；
- 像素空间**不是**硬约束，control video 只是接口，视频模型可能不跟随；
- 1.3B 双摆 / 6-DOF 实验走这条路径；base 零样本可能忽略 control，因此训练轻量 Control LoRA。

### 2.2 latent-level：CNF → motion latent `c` → 视频模型

```text
首帧 x0 + q0 + 稀疏 waypoint / 速度
        │
        ▼
CNF 运动场 + every-step 硬投影
        │
        ▼
c*（70 维 Legendre 系数）──► q*(t)
        │
        ▼
小型 action-conditioned 3D U-Net（12.88M）
        │
        ▼
生成视频
```

- 硬约束直接作用在**运动 latent `c`** 上；
- 视频模型仍然以 `q(t)`（以及 `sin q`、`cos q`）为条件，不是把 `c` 喂进 Wan 的内部 latent；
- 2-link 历史实验和 6-DOF 小型 action WM 走这条路径，用来验证“投影能否转化为像素收益”。

### 2.3 两条接口的区别

| 维度 | input-level | latent-level |
|---|---|---|
| 投影位置 | 运动空间；生成 control video 之前 | 直接作用在 `c`（70 维） |
| 视频模型看到 | 首帧 + control video | 首帧 + control + `q` 特征 |
| 像素空间保证 | 无；control 只是接口 | 无；`q` 只是条件 |
| 主要证据 | 1.3B 双摆 / 6-DOF + Control LoRA | 2-link / 6-DOF 小型 3D U-Net |
| 适用场景 | 大预训练视频模型，冻结主体 | 小型可控 action WM，验证机制 |

---

## 3. 方法

### 3.1 运动层：`q(t) = B(t)c`

每条关节轨迹用归一化 Legendre 基展开：

```text
q(t) = B(t)c,        q(t) ∈ R^J,   c ∈ R^{D·J}
```

- 6-DOF 实验：`J = 7`，`D = 10`，`c` 为 70 维；17 个均匀采样帧，`t ∈ [0,1]`；
- 2-link 历史实验：6 阶 Legendre，每个关节 6 个系数，共 12 维；16 帧；
- 速度：`q'(t) = B'(t)c`；报告中的导数使用 `discrete_autograd` 模式，即对采样归一化基做 autograd；`continuous` 是数学上更干净的算子，但不是报告结果所用的配置；
- 系数布局：`c[d * J + j]`；
- flow prior：`p(c | q0)` 用 flow matching 训练，MLP hidden 256，6000 步；推理 50 步 Euler，初始噪声 `N(0, 0.5²)`；系数按全局 mean/std 归一化。

### 3.2 约束层：`A c = b`

位置约束：

```text
B(t_k) c = q*_k
```

关节速度约束：

```text
B'(t_k) c = v*_k
```

任务空间速度约束（第 8 帧末端 3D 速度）：

```text
J(q_k) B'(t_k) c = v*_ee,k
```

其中 `J(q_k)` 是末端平移 Jacobian；因为它依赖当前轨迹，`project_sparse` 会重新线性化 3 次。把所有行堆起来得到 `A c = b`，再做最小范数硬投影：

```text
c* = c − Aᵀ (A Aᵀ + ε I)⁻¹ (A c − b),        ε = 1e-8
```

性质：

- 这是“改动最小”的投影：只修正违反约束的分量，不引入 loss 权重；
- 位置/速度残差可到求解精度；6-DOF anchor 误差 `< 1e-6`；
- `ε` 只用于数值稳定，不改变约束含义；
- 投影在**物理系数空间**做，再变换回归一化空间。

### 3.3 非穿透与可行性后处理

线性投影只保证 waypoint / 速度，不保证桌面非穿透、关节限位和相邻帧跳变。因此还有一层后处理：

- `enforce_clearance`：对低于安全高度的帧做安全 IK 抬升（抓取帧安全高度 `0.3655 m`，其他帧 `0.332 m`）；
- `validate_trajectory`：检查关节限位、最大关节步长 `0.8 rad`、夹爪/桌面 clearance；
- control 生成时最多重采样 12 次，直到轨迹通过检查。

安全 IK 会扰动 anchor，所以“纯投影残差 `< 1e-6`”和“经过安全 IK 后的实际 anchor error”是两个数。6-DOF latent-level 评测中后者为 `0.00131 rad`，soft baseline 为 `0.1320 rad`（原始报告为 `0.1315 rad`）。

### 3.4 视频层

**input-level（1.3B）**

- 视频模型：Wan2.1-Fun-1.3B-Control；
- Control：skeleton / segmentation video；
- 主体冻结，只训练 Control LoRA：rank 64，alpha 32，`q,k,v,ffn.0,ffn.2`，1500 步，单卡 A100 约 25 分钟；
- 1.3B base 已能跟随 skeleton，但 6-DOF segmentation control 上仍需要 LoRA；base 在 6-DOF 上 control EPE `24.85 px`、PSNR `22.99 dB`，LoRA 后 `6.65 px` / `24.32 dB`。

**latent-level（小型 action WM）**

- 3D U-Net，12.88M 参数；
- 128×128，17 帧；输入首帧 RGB、segmentation control、首帧 control、`q / sin(q) / cos(q)`；
- 残差预测 `x_t = x_0 + residual`；
- 128 条训练视频，16 条 held-out；训练 4000 步，batch 2，约 3.5 分钟（A100）。

### 3.5 every-step vs posthoc

两种做法的区别只在投影频率：

- **posthoc**：整条轨迹采样完后，最后投影一次；
- **every-step**：每个采样步都投影；
- 两者都会在最后再做一次投影，保证最终 `A c = b`。

直觉：posthoc 是一次性大修正，会把系数拉离先验的自然流形，视频模型看到 OOD 的 `q(t)`，容易产生抖动、重影；every-step 让采样过程始终停留在可行集附近，轨迹更接近自然流形。

证据：

| 测试台 | 指标 | every-step | posthoc | 结论 |
|---|---|---:|---:|---|
| 2-link 小 WM | pixel EPE / jerk / PSNR | `0.586 px` / `2.07` / `38.94 dB` | `0.689 px` / `6.05` / `28.35 dB` | every-step PSNR 高约 10.6 dB，jerk 约为 posthoc 的 1/2.9 |
| 6-DOF latent-level | PSNR / jerk | `23.99 dB` / `0.1638` | `23.58 dB` / `0.2061` | PSNR `+0.40 dB`（10/16），jerk 15/16 更好 |
| 6-DOF dense 轨迹 | joint RMSE / jerk | `0.0509` / `0.1559` | `0.0632` / `0.1861` | every-step 在 15/16 样本上 RMSE 更好、15/16 jerk 更低 |

边界：every-step 不是所有指标全面赢。6-DOF dense 像素上，posthoc 的 final cube error 有时更好（`3.20 vs 5.48 px`，6/8）；every-step 的优势主要在过程自然性和 PSNR。

---

## 4. 实验矩阵

| # | 测试台 | 视频模型 / 渲染器 | 约束 | n | 指标 | 结果状态 |
|---:|---|---|---|---:|---|---|
| 1 | 6-DOF dense | Wan2.1-Fun-1.3B-Control + LoRA（input-level） | 7 anchors `[0,3,4,6,10,13,16]` + 第 8 帧关节速度 | 16 traj / 8 pixel | anchor, joint/EE RMSE, jerk, PSNR, EPE | 硬约束成立；dense velocity 反而更差 |
| 2 | 6-DOF sparse | 同上 | 6 anchors `[0,4,8,12,13,16]` + 第 8 帧末端 3D 速度 | 16 / 8 | 同上 | 轨迹级 velocity 正；像素仅 PSNR 稳定正 |
| 3 | 6-DOF latent-level | 12.88M 3D U-Net（128²、17 帧） | 6 anchors + 第 8 帧末端 3D 速度 | 16 | PSNR, anchor, joint/EE RMSE, jerk | every-step > posthoc；hard/soft trade-off |
| 4 | 1.3B 双摆 clean | Wan2.1-Fun-1.3B-Control + LoRA（input-level） | 3 waypoints `[0,8,16]` + 第 7 帧速度 | 4 | waypoint/track EPE, PSNR, jerk | LoRA 有效；velocity 4/4 小幅正 |
| 5 | 1.3B 双摆 noisy | 同上 | 第 8/16 帧 4 px 噪声；estimated vs oracle velocity | 8 | track EPE, PSNR, joint RMSE | oracle 正；estimated 负 |
| 6 | 2-link 历史 | 3D U-Net（64²、16 帧） | 6 阶 Legendre，12 维系数；waypoint / 速度 | 1024 测试轨迹 | pixel EPE, jerk, PSNR | every-step 强正；action-conditioned soft 已足够 |
| 7 | PF_Wan 官方 A14B | 官方 PF_Wan（Wan2.2-A14B-I2V） | text-only + oracle 轨迹 / warp | 1 clip（17/81 帧） | CoTracker 2D EPE | 原生 I2V 基线；CNF oracle/warp 无实质优势 |
| 8 | PF_Wan 1.3B 100-step | PhysisForcing + Wan2.1-T2V-1.3B | hard vs soft low-level | 1 held-out clip | low-level obj, video EPE | low-level 正；视频负 |

---

## 5. 主要结果

### 5.1 6-DOF dense：位置硬约束成立，dense velocity 是负结果

设置：6-DOF 任务（7 关节 Franka Panda），17 帧、10 阶 Legendre、70 维系数；128 条训练轨迹 / 16 条 held-out；256×256、8 fps；anchors `[0,3,4,6,10,13,16]`，第 8 帧关节速度。

轨迹级（16 held-out）：

| 方法 | joint RMSE | jerk | 桌面穿透 | 方块最终误差 |
|---|---:|---:|---:|---:|
| waypoint（every-step） | **0.0509** | **0.1559** | 0 | ~0 |
| velocity（every-step） | 0.0668 | 0.1988 | 0 | ~0 |
| posthoc | 0.0632 | 0.1861 | 0 | ~0 |

关键数字：

- 位置 anchor 误差 `< 1e-6`；桌面穿透 0，方块最终位置误差约 0；
- 第 8 帧速度在基函数意义上 16/16 精确满足，但整条轨迹关节 RMSE 比 waypoint 差 `+0.016`（0/16 更好），jerk 高 `+0.043`（0/16 更好）；
- 解释：密集 anchor 已经几乎决定整条轨迹，额外强制第 8 帧 oracle 速度会与全局多项式基冲突，产生 ringing；
- every-step 在 15/16 样本上 joint RMSE 更好、15/16 jerk 更低。

像素级（8 held-out，1.3B Control LoRA）：

| 方法 | PSNR | cube track EPE | control EPE |
|---|---:|---:|---:|
| base（无 LoRA） | 22.99 | 24.80 px | 24.85 px |
| gt + LoRA | **24.32** | **6.28** | **6.65** |
| waypoint | 23.08 | 8.07 | 8.56 |
| velocity | **23.32** | **6.94** | **7.32** |
| posthoc | 22.53 | 7.97 | 8.43 |

关键分裂：

- 像素上 velocity 一致更好：PSNR `+0.24 ± 0.05 dB`（8/8）、cube track `−1.13 ± 0.37 px`（7/8）、control EPE `−1.24 ± 0.41 px`（7/8）；
- 但同一批轨迹上 velocity 的关节 RMSE `0.0668 vs 0.0509`（0/16 更好）、末端 Cartesian 误差 `0.0256 vs 0.0225 m`（0/8 更好）、jerk 更高；
- 说明像素收益不能解释为“运动更准”；在 7-DOF 冗余臂 + 密集 anchor 下，关节空间最小范数投影会改变零空间姿态，像素指标和轨迹指标可能背离。

### 5.2 6-DOF sparse：轨迹级速度正结果，像素跟踪未复现

设置：6 个 anchor `[0,4,8,12,13,16]`；第 8 帧约束**末端 3D 速度**（不是 7 个关节速度）；低于桌面的帧用安全 IK 抬升，保证零穿透。

轨迹级（16 held-out）：

| 方法 | joint RMSE | EE RMSE | EE 速度误差 | jerk |
|---|---:|---:|---:|---:|
| sparse waypoint | 0.0618 | 0.0376 | 0.533 | **0.1442** |
| **sparse velocity** | **0.0559** | **0.0334** | **0.116** | 0.1597 |
| sparse posthoc | 0.0624 | 0.0467 | 0.6225 | 0.1997 |

配对（sparse velocity − sparse waypoint）：

- 关节 RMSE：`−0.0059 ± 0.0009`，**16/16 更好**；
- 末端 RMSE：`−0.0042 ± 0.0010`，**14/16 更好**；
- 末端速度误差：`−0.417 ± 0.034`，**16/16 更好**；
- jerk：`+0.015 ± 0.003`，1/16 更好（略差）。

every-step 对 posthoc：joint RMSE `0.0559 vs 0.0624`、EE RMSE `0.0334 vs 0.0467`、jerk `0.1597 vs 0.1997`。

像素级（8 held-out）：

| 方法 | PSNR | cube track EPE | control EPE | final error |
|---|---:|---:|---:|---:|
| sparse waypoint | 22.22 | **5.29** | **7.07** | 3.39 |
| **sparse velocity** | **22.38** | 5.49 | 7.18 | **3.19** |
| sparse posthoc | 21.86 | 6.93 | 7.59 | 3.38 |

配对（sparse velocity − sparse waypoint）：

- PSNR：`+0.16 ± 0.07 dB`，**8/8 更好**；
- cube track EPE：`+0.20 ± 0.36 px`，3/8 更好（略差，不显著）；
- control EPE：`+0.11 ± 0.19 px`，3/8 更好（略差，不显著）；
- final error：`−0.20 ± 0.54 px`，4/8 更好（不显著）。

结论：**轨迹级 16/16 的收益没有干净地转化成像素级跟踪收益**。velocity 在 PSNR 上稳定更好，但 cube tracking / control EPE 未复现；渲染器仍是瓶颈。velocity 的稳定正收益需要同时满足：稀疏 waypoint、任务空间速度表示、安全 IK 非穿透。

### 5.3 6-DOF latent-level：every-step 成立，hard vs soft 是 trade-off

设置：小型 3D U-Net action WM，128×128、17 帧、12.88M 参数；128 条训练视频、16 条 held-out；直接在 motion latent `c`（7×10=70 维）上投影。

| 方法 | PSNR | anchor error | joint RMSE | EE RMSE | 速度误差 | jerk |
|---|---:|---:|---:|---:|---:|---:|
| gt | **34.58** | **0** | **0** | **0** | **0** | 0.2665 |
| unconstrained | 20.31 | 0.3495 | 0.1123 | 0.0973 | 0.8238 | 0.1898 |
| soft | 23.48 | 0.1320 | **0.0411** | 0.0396 | 0.2885 | 0.2048 |
| posthoc | 23.58 | 0.00131 | 0.0662 | 0.0422 | 0.1156 | 0.2061 |
| **every_step** | **23.99** | **0.00131** | 0.0587 | **0.0338** | **0.1156** | **0.1638** |

**every-step vs posthoc**：

- PSNR：`+0.40 ± 0.20 dB`，10/16 更好；
- jerk：`−0.0423 ± 0.0087`，**15/16 更好**；
- EE RMSE：`−0.0084 ± 0.0036`，12/16 更好；
- joint RMSE：`−0.0074 ± 0.0047`，8/16 更好（不一致）。

**hard vs soft**：

- anchor error：`0.00131 vs 0.1320 rad`，hard 约 **100 倍**更准；16/16 更好；
- PSNR：`+0.51 ± 0.13 dB`，12/16 更好；
- jerk：`−0.0406 ± 0.0066`，**16/16 更好**；
- 速度误差：`−0.172 ± 0.034`，14/16 更好；
- EE RMSE：`−0.0056 ± 0.0019`，12/16 更好；
- **但 soft 的 joint RMSE 更低**：`0.0411 vs 0.0587`，hard 在这一项 0/16 更好。

解释：soft 看到 anchor 作为输入，可以“贴着 anchor 附近”生成平滑轨迹，但无法保证精确命中；hard 用投影保证精确，代价是最小范数修正会改变零空间姿态，导致关节 RMSE 略高。对任务而言，约束精度、末端精度、jerk 更重要，hard 更合适；但如果只看 joint RMSE，soft 会显得更好。这是指标选择问题，不是全面碾压。

补充：表中的 anchor error `0.00131 rad` 是经过安全 IK 后处理后的实际误差；纯投影残差仍 `< 1e-6`。GT 的 PSNR 34.58 是 rendering ceiling，不是方法上界；every-step 与 GT 差约 10.6 dB，差距主要来自稀疏约束、小型视频 WM 的渲染能力和 control 量化。原始报告为 gt/unconstrained/soft/posthoc/every_step = `34.47 / 20.31 / 23.46 / 23.53 / 23.94 dB`，every-step vs posthoc `+0.41 dB`（11/16），soft anchor `0.1315 rad`、soft joint RMSE `0.0410`；release 复跑为 `34.58 / 20.30 / 23.48 / 23.58 / 23.99 dB`，`+0.40 dB`（10/16），soft anchor `0.1320 rad`、soft joint RMSE `0.0411`。因视频 WM 初始化不同，PSNR 有约 0.1–0.5 dB 波动，结论不变。完整见 [`08_reproduction_numbers.md`](08_reproduction_numbers.md)。

### 5.4 1.3B 双摆：干净数据小幅正，含噪数据看速度估计

设置：Wan2.1-Fun-1.3B-Control + LoRA，256×256、17 帧；4 个 held-out 样本；waypoint 在 0/8/16 帧，速度在第 7 帧。

干净数据：

| Control | waypoint EPE | track EPE | PSNR | endpoint jerk |
|---|---:|---:|---:|---:|
| waypoint only | 6.65 | 3.41 | 34.16 | 9.56 |
| **velocity only** | **6.42** | **3.22** | **34.31** | **9.20** |
| ground truth | 6.56 | 2.35 | 34.98 | 10.69 |

Base vs LoRA（同一 GT control）：

| 指标 | Base | LoRA（gt） |
|---|---:|---:|
| waypoint EPE | 8.87 px | **6.56 px** |
| PSNR | 31.06 dB | **34.98 dB** |

配对（velocity − waypoint，4 样本）：

- waypoint EPE：`−0.22 ± 0.09 px`，4/4 更好；
- track EPE：`−0.20 ± 0.10 px`，4/4 更好；
- PSNR：`+0.16 ± 0.07 dB`，4/4 更好；
- endpoint jerk：`−0.36 ± 0.23`，3/4 更低；
- 轨迹关节 RMSE 降低 `43% / 8% / 21% / 39%`。

注意：白色关节帽在 256 分辨率下造成约 `6.5 px` 的 waypoint EPE 系统偏差，绝对值只适合同条件相对比较；GT control 的 endpoint jerk `10.69` 高于 velocity 的 `9.20`，所以“jerk 更低”不等于“更自然”。

含噪数据（8 样本，第 8/16 帧加 `4 px` 高斯噪声，目标始终是干净 GT）：

| 方法 | track EPE | PSNR | joint RMSE |
|---|---:|---:|---:|
| noisy waypoint | 6.45 | 32.55 | 0.105 |
| noisy velocity（估计） | 6.82 | 32.11 | 0.147 |
| **noisy oracle velocity** | **5.26** | **32.96** | **0.091** |

配对（相对 noisy waypoint）：

- oracle velocity：track `−1.18 ± 0.64 px`（6/8）、PSNR `+0.41 ± 0.16 dB`（7/8）、joint RMSE `−0.013 ± 0.004`（8/8）；
- estimated velocity：track `+0.37 ± 1.01 px`（2/8）、PSNR `−0.45 ± 0.32 dB`（2/8）、joint RMSE `+0.042 ± 0.018`（2/8）。

结论：**速度硬约束本身有效，但要求速度估计可靠**。用干净 action 做同样的中心差分，速度误差很小，轨迹 RMSE 在 7/8 样本上优于 waypoint-only；用 9 个含噪帧做局部二次拟合仍然救不回来（6/8 更差）。瓶颈是速度估计器，不是 CNF 投影。论文里必须把 velocity 写成条件性收益，并报告 oracle / estimated 的分解。

### 5.5 PF_Wan：2D 跟踪与视频级负结果

#### 5.5.1 官方 PF_Wan A14B

官方 `DAGroup-PKU/PF_Wan` 权重在单张 A100 40GB 上完成 17 帧 / 256×256 和 81 帧 / 256×256 生成。17 帧、20 步 Euler、guidance 5.0、seed 0；采样约 11 秒，未 OOM。

基于 8×8 CoTracker 网格的 17 帧结果：

| 指标 | 官方 PF_Wan | 参考 |
|---|---:|---:|
| 动态轨迹 EPE 均值 | 58.65 px | 目标轨迹 |
| 动态轨迹 EPE 中位数 | 50.84 px | 目标轨迹 |
| 全轨迹 EPE 中位数 | 0.59 px | 目标轨迹 |
| 静态区域 EPE 均值 | 0.86 px | 目标轨迹 |
| 动态轨迹位移 | 68.64 px | 80.89 px |
| 前后向 EPE 中位数 | 0.35 px | 0.27 px |
| 前后向 EPE p95 | 4.32 px | 1.15 px |

81 帧结果：动态 EPE `58.39 px`、静态 EPE `1.26 px`、动态位移 `52.32 px`，参考动态位移 `80.89 px`。生成机械臂会穿过场景，但没有复现参考扫动；夹爪几何发生变化，胡椒被推开但没有形成最终集中堆积。

结论：官方 PF_Wan 是有效的原生 I2V 基线，运动幅度和静态稳定性明显好于旧 1.3B 实验；但仅凭文本 prompt 仍无法精确复现参考动作。这支持“需要动作条件几何约束”的动机，但不证明 CNF 在视频层面有效。

#### 5.5.2 Oracle latent 修正与 warp

第一个 oracle 实验冻结官方 PF_Wan，从生成视频开始，编码到 VAE latent，只优化一个加性 latent correction；参考 CoTracker 轨迹作为答案。Soft MSE 和 CNF 使用相同的 16 条动态轨迹、query grid、图像/latent 正则项和固定梯度预算；每条 CNF 轨迹最多用 4 个均匀约束帧，避免高次多项式外推。

内部优化指标：

| 指标 | Soft | CNF |
|---|---:|---:|
| 初始选定轨迹 EPE | 50.01 px | 50.01 px |
| 最终更新前 EPE | **45.94 px** | 49.53 px |
| CNF 修正场残差 | — | 约 `1e-15` |

独立 8×8 CoTracker 评估：

| 指标 | Roundtrip baseline | Soft | CNF |
|---|---:|---:|---:|
| 动态 EPE 均值 | 54.84 px | **49.04 px** | 49.07 px |
| 动态 EPE 中位数 | 43.34 px | **40.11 px** | 38.72 px |
| 静态 EPE 均值 | 0.94 px | **0.74 px** | 0.89 px |
| 前后向 EPE p95 | **3.63 px** | 62.49 px | 75.12 px |

CNF 的动态中位数略好，但静态误差更差、时序一致性明显恶化；视觉上只有轻微运动变化。两者都没有让官方生成遵循参考扫动。CNF 线性求解只在构造出的修正场上精确，不代表生成视频具有零轨迹残差；修正仍需通过冻结 VAE 和 latent 优化传递到视频，而该映射是主要瓶颈。

Warp 验证：用 8×8 CoTracker 控制网格对生成视频执行 warp，控制点轨迹从每条选定轨迹最多 4 个锚点插值得到。

- 全局 RBF 变形：动态 EPE 从 `54.84 px` 降到 soft `39.19 px` / CNF `50.06 px`，但产生严重流体状拖影；
- 固定拓扑分片仿射三角网格：降到 soft `22.60 px` / CNF `26.71 px`，但出现明显三角形撕裂和错误局部几何；
- 每帧使用精确参考位移的 dense target warp 也产生不可接受的视觉伪影。

结论：主要问题不是 CNF 能否求解轨迹方程，而是如何把稀疏轨迹变化转换为稠密、遮挡感知且光度一致的视频变形。简单几何插值器被否定；认真实现需要可学习 dense motion field 和 inpainting/refinement renderer。

#### 5.5.3 PF_Wan 1.3B 100-step：low-level 正，视频负

设置：PhysisForcing + Wan2.1-T2V-1.3B（1.419B 参数）；4 个 BridgeData V2 示例 clip 中 3 个训练、1 个 held-out（`episode_047007`）；CNF 与 soft 从相同初始化、相同 seed、相同数据顺序训练 100 步；在 held-out clip 上以零学习率评估 10 个时间步。

held-out 匹配时间步均值：

| 指标 | CNF | Soft |
|---|---:|---:|
| Total objective | **0.26873** | 0.27649 |
| Diffusion loss | 0.26613 | **0.26542** |
| Low-level objective | **0.00259** | 0.01104 |

CNF 的 held-out low-level objective 约为 soft 的 4.3 倍下降，diffusion loss 仅高 0.27%；每个样本约 33 条轨迹通过 hard-constraint gate，平均条件数 2.93，最大等式残差低于 `2.3e-16`，单卡 A100 无 NaN/Inf 或 OOM。

但生成视频检查是负结果：两个 100 步 checkpoint 在 `episode_047007` 上用相同首帧、prompt、seed 和 sampler 解码；detailed caption 下 CNF 动态轨迹 EPE `66.27 px`，soft `73.29 px`，参考运动位移 `80.89 px`；但 CNF 平均只移动 `3.44 px`，soft `2.32 px`，两者都未复现真实扫动。

这是当前 100 步设置的初步负结果，不是对 CNF idea 的反证：只使用 1 个 clip、1 个 seed，且 1.3B T2V backbone 不是真正的 image-to-video backbone。它说明“low-level 目标改善”不等于“视频生成改善”。

### 5.6 2-link 历史实验：every-step 的强证据

设置：2-link 机械臂，64×64、16 帧；每个关节 6 阶 Legendre，共 12 维；小型 3D U-Net；1024 条测试轨迹；1 个 seed。

| 方法 | 像素 EPE（detector） | 像素 jerk | PSNR |
|---|---:|---:|---:|
| blind unconstrained | 17.09 px | 2.34 | 26.76 dB |
| blind posthoc | 0.689 px | 6.05 | 28.35 dB |
| **blind every-step** | **0.586 px** | **2.07** | **38.94 dB** |
| ground truth | 0.589 px | 2.16 | 46.46 dB |

关键：

- posthoc 满足约束，但像素 jerk 约为 every-step 的 `2.92×`，PSNR 低约 `10.6 dB`；
- every-step 的像素 EPE 与 GT 几乎相同（`0.586 vs 0.589 px`），但 PSNR 仍比 GT 低约 `7.5 dB`，说明渲染层仍有差距；
- action-conditioned 模型（看到 waypoint）的 unconstrained latent EPE 已经只有 `0.427 px`，posthoc / every-step 视觉几乎一样；soft 的 PSNR 为 `39.76 dB`。

结论：CNF 最合适的场景是“世界模型不知道新动作、约束在推理时外部给出”；如果 WM 已经 action-conditioned，soft conditioning 可能已经足够。这个历史实验只覆盖 2-link、64×64、16 帧，不能直接外推到 6/7-DOF 或真实数据。

---

## 6. Claims：哪些成立，哪些不成立

### 6.1 成立

1. **位置硬约束精确**：6-DOF 纯投影 anchor 误差 `< 1e-6`；桌面穿透 0，方块最终位置误差约 0。6-DOF latent-level 评测（含安全 IK 后处理）的端到端 anchor error 为 `0.00131 rad`，仍比 soft 的 `0.1320 rad` 约小 100 倍（原始报告 soft `0.1315 rad`）。
2. **every-step 比 posthoc 更自然**：2-link 上 PSNR `+10.6 dB`、jerk 约为 `1/2.9`；6-DOF latent-level 上 PSNR `+0.40 dB`（release 复跑；原始报告 `+0.41 dB`，11/16）、jerk 15/16 更好；6-DOF dense 轨迹上 RMSE `0.0509 vs 0.0632`、jerk `0.1559 vs 0.1861`。
3. **hard 的约束精度远高于 soft**：anchor `0.00131 vs 0.1320 rad`（原始 `0.1315`）；hard 的 PSNR `+0.51 dB`（12/16；原始 `+0.48 dB`，13/16）、jerk 16/16、速度误差 14/16、EE RMSE 12/16 更好。**但 soft 的 joint RMSE 更低**（`0.0411 vs 0.0587`；原始 `0.0410`），所以不能 claim hard 在所有指标上全面更好。
4. **稀疏 waypoint + 任务空间速度在轨迹级稳定有益**：joint 16/16、EE 14/16、速度 16/16 更好；条件是稀疏 anchor、任务空间速度、安全 IK 非穿透三者同时满足。代价是 jerk 略高（1/16 更好）。
5. **1.3B input-level Control LoRA 有效**：双摆上 waypoint EPE `8.87 → 6.56 px`、PSNR `31.06 → 34.98 dB`；6-DOF 上 control EPE `24.85 → 6.65 px`、PSNR `22.99 → 24.32 dB`。base 零样本在 6-DOF segmentation control 上失败，LoRA 是必要的轻量接口。
6. **速度可靠时，velocity 优于 waypoint**：1.3B 干净数据 4/4 样本小幅更好；含噪 waypoint + oracle 速度下 PSNR 7/8、joint RMSE 8/8 更好。
7. **CNF 投影可以直接作用在 action WM 的 motion latent 上**：6-DOF 小型 3D U-Net 上 every-step 的像素/轨迹收益成立，不只是 control-video 接口。

### 6.2 不成立 / 不能 claim

1. **velocity 无条件优于 waypoint**：密集 anchor 下 velocity 的 joint RMSE `0.0668 vs 0.0509`（0/16 更好）、EE 误差 `0.0256 vs 0.0225 m`（0/8 更好）；含噪 estimated velocity 在 6/8 样本上比 waypoint 更差。只有“稀疏 waypoint + 任务空间速度 + 安全 IK”给出干净正结果。
2. **像素跟踪能复现轨迹收益**：6-DOF sparse velocity 的 PSNR 8/8 更好，但 cube track 3/8、control EPE 3/8（略差，不显著）；密集 velocity 像素 8/8 PSNR、7/8 track 更好，但轨迹上 velocity 0/16 更好。像素指标不能单独证明运动更准，渲染器是瓶颈。
3. **hard 在所有指标上优于 soft**：soft 的 joint RMSE 更低（`0.0411 vs 0.0587`；原始 `0.0410`）；hard 的优势在 anchor 精度、PSNR、jerk、速度误差、EE RMSE。正确说法是 trade-off，不是全面碾压。
4. **1.3B 的 7-DOF latent-level 已做**：1.3B 只做 input-level（control video → Wan）；7-DOF latent-level 的证据来自 12.88M 小型 3D U-Net。没有证据表明在 Wan 内部 latent 上投影也能工作。
5. **严格“冻结 1.3B”**：视频主体冻结，但训练了 Control LoRA（rank 64，1500 步）；base 零样本在 6-DOF 上 control EPE `24.85 px`、PSNR `22.99 dB`，并不够用。准确说法是“冻结主体 + 轻量 Control LoRA”。
6. **dense GT 是理论上限**：GT 是 oracle reference / rendering ceiling；稀疏约束下运动多解，PSNR / joint RMSE 会惩罚其他合法解。
7. **真实机器人 / 真实接触**：全部是 PyBullet 合成 pick-and-place；方块在抓取期间刚性跟随夹爪，没有真实接触力、摩擦、变形；没有 BridgeData / DROID 或真机数据。
8. **与 ProjFlow / PDM / Wan-Move / IOI 的定量对比**：工作区没有这些方法的代码、权重或 license，未做定量对比。
9. **PF_Wan 视频级正结果**：PF_Wan 1.3B 100-step 的视频是负结果（动态 EPE 66.27 vs 73.29 px，位移 3.44 vs 2.32 px，参考 80.89 px）；官方 A14B 的 oracle 实验中 CNF 相对 soft 没有实质优势（动态 EPE 均值 49.07 vs 49.04 px，时序一致性更差），warp 也产生明显伪影。
10. **every-step 所有指标全面赢**：6-DOF dense 像素上 posthoc 的 final cube error 有时更好（`3.20 vs 5.48 px`，6/8）；every-step 的优势主要在过程自然性和 PSNR。

### 6.3 推荐的 claim 版本

> **位置硬约束 + every-step 投影 + 接触/非穿透一致性**是稳健的主贡献；6-DOF latent-level 已证明 every-step > posthoc，hard 的约束精度远高于 soft，但 soft 的 joint RMSE 略低（精确约束 vs 关节平滑的 trade-off）。**速度硬约束**只在 **稀疏 waypoint + 任务空间速度 + 安全 IK 投影** 时给出稳定轨迹级正收益：joint/EE/速度跟随 14–16/16 更好，PSNR 8/8 更好，但像素跟踪未复现；密集 anchor 下仍为负。不要写成“velocity 永远优于 waypoint”。

---

## 7. 阅读指南

| 想了解 | 建议阅读 | 说明 |
|---|---|---|
| 项目入口、结果表、快速开始 | [`../README.md`](../README.md) | 最短路径；先看能跑什么、验证了什么 |
| 问题、方法、实验矩阵、claim 边界 | 本文件 | 总览；适合写 related work / rebuttal 前建立边界感 |
| 方法公式、张量形状、代码入口 | [`01_method.md`](01_method.md) | Legendre 表示、flow matching、硬投影、采样循环 |
| 6-DOF Panda 场景 / 数据 / 约束 / 结果 | [`02_pickplace.md`](02_pickplace.md) | dense/sparse 设置、失败模式、逐项结果 |
| latent-level 小型 action WM | [`03_latent_wm.md`](03_latent_wm.md) | 70 维 `c`、12.88M 3D U-Net、every-step / hard / soft |
| Wan 1.3B / PF_Wan / A14B oracle | [`04_wan_1_3b.md`](04_wan_1_3b.md) | input-level、2D 轨迹头、oracle/warp；明确 1.3B 7-DOF latent-level 未做 |
| 复现环境、命令、预期产物 | [`05_reproduction.md`](05_reproduction.md) | 6-DOF 一键入口与可选 1.3B 外部复现 |
| 指标定义与统计口径 | [`06_metrics.md`](06_metrics.md) | 轨迹级 / 像素级指标、配对 delta、PSNR 尺度 |
| 结论边界、负结果、不能 claim | [`07_limitations.md`](07_limitations.md) | soft joint RMSE、像素跟踪、1.3B 7-DOF、真实数据 |
| release 完整复跑快照 | [`08_reproduction_numbers.md`](08_reproduction_numbers.md) | dense / sparse / latent 数字与原始报告的差异 |
| 代码实现 | [`../src/cnf_motion/`](../src/cnf_motion/) | 按 `basis.py` → `motion_field.py` → `constraints.py` → `video_wm.py` 读 |
| 原始实验报告（不在 release 内） | 开发仓库 `experiments/cnf_pickplace/report_pickplace.md`、`report_action_wm.md`；`experiments/wan22_fun_control/report_1_3b.md`、`report_noisy.md`；`experiments/cnf_wan_1_3b/report.md`；`experiments/pf_wan_official_1gpu/report.md` | 逐样本表格、日志和失败分析 |

---

## 8. 结论

这个 release 的定位是“**测试时硬运动约束 + 轻量视频接口**”，而不是“替代 action-conditioned WM”。在受控的合成场景里，核心机制成立：位置硬约束可以做到机器精度；every-step 投影比 posthoc 更自然，2-link 小模型、6-DOF 轨迹级和 6-DOF latent-level 都支持这一点；hard 的 anchor 精度约是 soft 的 100 倍，但 soft 的 joint RMSE 更低，这是“精确约束 vs 关节平滑”的 trade-off。最强的轨迹级正结果是“稀疏 waypoint + 任务空间速度”：joint/EE/速度跟随 14–16/16 更好，代价是 jerk 略高；但像素跟踪没有复现轨迹收益，1.3B 的 7-DOF latent-level 投影和真实机器人数据仍未验证，PF_Wan 的 100-step 视频结果也是负的。因此，当前最适合的主 claim 是“位置硬约束 + every-step + 接触/非穿透一致性”，velocity 作为有条件增益，视频级收益和真实数据是下一步要补的证据。
