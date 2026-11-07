# 06 指标：轨迹级、像素级与配对统计

## 一句话 / 结论

- 轨迹级指标分四组：约束满足（anchor）、关节/末端精度（joint RMSE、EE RMSE）、速度跟随（EE velocity、frame-8 joint velocity）、自然性/接触（jerk、table clearance、cube target error）。
- 像素级 6-DOF 用颜色阈值 + 8 连通域检测橙色方块；PSNR 的 `data_range=255`。
- 1.3B 双摆用另一套橙色连杆端点检测（PCA + 10/90 分位端点 + 蓝色连杆中心选端）；PSNR 的 `data_range=1`。
- 两套 PSNR 不能直接比：255 与 1 的尺度差是 \(20\log_{10}255\approx48.13\) dB。
- 配对统计统一是 \(\Delta=\text{method}-\text{baseline}\)；\(\mathrm{SE}=\mathrm{std}(\Delta,\text{ddof}=1)/\sqrt n\)；`better` 按指标方向数符号。
- release 快照 `release/runs/controls_dense|controls_sparse|eval_action_wm/metrics.json` 与原始报告的 PSNR 差 \(<0.5\) dB（视频 WM 初始化/训练非确定性）；本文优先使用 release 快照。
- GT 是 oracle reference，不是理论上限；稀疏约束下运动多解，jerk 低也不等于更自然。

---

## 1. 记号与报告口径

| 符号 | 含义 | 形状 / 单位 |
|---|---|---|
| \(q,q_{\text{gt}}\) | 预测 / GT 关节轨迹 | \([17,7]\)，rad |
| \(p_t,p_t^{\text{gt}}\) | 第 \(t\) 帧末端位置，\(p_t=\mathrm{FK}(q_t)\) | \([3]\)，m |
| \(T,J\) | 帧数 / 关节数 | 17 / 7 |
| \(\mathcal{K}_{\text{anchor}}\) | 参与 anchor 指标的帧 | dense `[0,3,4,6,10,13,16]`；sparse `[0,4,8,12,13,16]` |
| \(\Delta_i\) | 第 \(i\) 个配对样本的 method − baseline | 与指标同单位 |
| \(n\) | 有效配对数 | 整数 |

报告约定：

1. 每条轨迹先算一个标量指标，再对样本取均值；`mean_metrics` 用 `np.nanmean`，`n` 是该方法的行数。
2. 配对统计只在同一条 `sample` 同时存在 method 和 baseline、且两个值都有限时计入；缺失对直接跳过，不补零。
3. `release/runs/eval_action_wm/metrics.json` 的 `anchor_error=0.00131` 是 **clearance 后**的端到端值；`tests/test_projection.py` 的投影残差是 **clearance 前**的单元测试值。两个口径不能混用。

---

## 2. 轨迹级指标

### 2.1 精确公式

**Anchor 最大绝对误差**（所有 anchor 帧、所有关节取最大）：

\[
E_{\text{anchor}}
=\max_{k\in\mathcal{K}_{\text{anchor}}}\max_{j=1,\dots,7}
\left|q_{k,j}-q^{\text{gt}}_{k,j}\right|.
\]

代码：`np.abs(q[list(anchor_frames)] - q_gt[list(anchor_frames)]).max()`。单位 rad。

**关节 RMSE**：

\[
E_{\text{joint}}
=\sqrt{\frac{1}{TJ}\sum_{t=0}^{T-1}\sum_{j=1}^{J}
\left(q_{t,j}-q^{\text{gt}}_{t,j}\right)^2}.
\]

代码：`np.sqrt(((q - q_gt) ** 2).mean())`。单位 rad。

**末端 RMSE**：

\[
E_{\text{EE}}
=\sqrt{\frac{1}{T}\sum_{t=0}^{T-1}
\left\|p_t-p^{\text{gt}}_t\right\|_2^2}.
\]

代码：`np.sqrt((((ee - ee_gt) ** 2).sum(-1)).mean())`；先对 3 个轴求和，再对 17 帧平均，最后开方。单位 m。

**末端速度误差（第 8 帧中心差分）**：

\[
E_{\text{EEvel}}
=\left\|
\frac{p_9-p_7}{2}\cdot(T-1)-v_8^\star
\right\|_2,
\qquad T-1=16.
\]

其中 \(v_8^\star\) 由 `task_space_velocity` 对 GT 轨迹算：

\[
v_8^\star=\frac{p^{\text{gt}}_9-p^{\text{gt}}_7}{2}\cdot16.
\]

代码：`frame_velocity(ee, frame=8, frames=17)`，即 `(positions[9]-positions[7])/2*16`。单位 m / 归一化时间。

**Jerk（三阶有限差分范数）**：

\[
E_{\text{jerk}}
=\frac{1}{T-3}\sum_{t=0}^{T-4}
\left\|
q_{t+3}-3q_{t+2}+3q_{t+1}-q_t
\right\|_2.
\]

代码：`np.linalg.norm(np.diff(q, n=3, axis=0), axis=-1).mean()`。\(T=17\) 时有 14 个三阶差分项。**没有除以 \(dt^3\)，也没有乘 \((T-1)^3\)**；单位是 rad / frame³。跨数据集比较 jerk 时必须确认帧率和差分定义一致。

**桌面间隙**：

\[
C_{\text{table}}
=\min_{t=0,\dots,T-1}
\left(z_{\text{cube}}(t)-0.025-0.30\right).
\]

其中 `CUBE_HALF=0.025` m，`TABLE_TOP=0.30` m。单位 m；负值表示方块底部穿进桌面，0 表示刚好接触，正值表示离桌面有间隙。

**方块最终目标误差**：

\[
E_{\text{cube target}}
=\left\|
\left(x_{\text{cube}}(T-1),\,y_{\text{cube}}(T-1)\right)
-\left(x_{\text{target}},\,y_{\text{target}}\right)
\right\|_2.
\]

只取最后一帧的 XY，单位 m；不包含 Z。

**第 8 帧关节速度误差**：

\[
E_{\text{jvel8}}
=\left\|
\left(q_9-q_8\right)
-\left(q^{\text{gt}}_9-q^{\text{gt}}_8\right)
\right\|_2.
\]

代码：`np.diff(q, axis=0)[8] - np.diff(q_gt, axis=0)[8]`。**没有乘 16**；单位 rad / frame。dense 脚本把它命名为 `velocity_error`，sparse 脚本命名为 `joint_velocity_at_8_error`。

### 2.2 代码中的键名

| 报告里的名字 | `trajectory_metrics` 的键 | 代码 / 说明 |
|---|---|---|
| anchor 误差 | `anchor_error` | 最大绝对误差，rad |
| 关节 RMSE | `joint_rmse` | 全轨迹，rad |
| 末端 RMSE | `ee_rmse` | 全轨迹，m |
| 末端速度误差 | `ee_velocity_error` | 仅 sparse / latent-level；第 8 帧，m / normalized time |
| jerk | `jerk` | 三阶差分，rad / frame³ |
| 桌面间隙 | `table_clearance` | `min(z_cube-0.025-0.30)`，m |
| 方块最终误差 | `cube_target_error` | 最后一帧 XY，m |
| 第 8 帧关节速度误差 | dense: `velocity_error`；sparse: `joint_velocity_at_8_error` | 逐帧差分，rad / frame |

`generate_controls.py` 在 `trajectory_metrics` 之后又把 `table_clearance` 覆盖成 `table_clearance(cube_positions)`，公式相同，值也相同；这是为了避免重复计算口径不一致。

### 2.3 Release 快照：dense（16 held-out）

| 方法 | anchor | joint RMSE | EE RMSE | jerk | `velocity_error` | table | cube target |
|---|---:|---:|---:|---:|---:|---:|---:|
| waypoint | 0.00131 | **0.05087** | **0.03311** | **0.15588** | **0.19803** | 0.0 | 0.0 |
| velocity | 0.00131 | 0.06681 | 0.04076 | 0.19902 | 0.21847 | 0.0 | 0.0 |
| posthoc | 0.00131 | 0.06323 | 0.04896 | 0.18633 | 0.19718 | 0.0 | 0.0 |

要点：dense 下 velocity 的 `velocity_error` 反而比 waypoint 差 `+0.02044`（0/16 更好）。这不是约束失败：约束是在 \(V_8\) 基函数意义上精确，`velocity_error` 是逐帧差分；两者不是同一算子（见 `01_method.md` 3.3）。

### 2.4 Release 快照：sparse（16 held-out）

| 方法 | anchor | joint RMSE | EE RMSE | EE velocity | jerk | `joint_velocity_at_8_error` | table | cube target |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_waypoint | 0.00131 | 0.06180 | 0.03760 | 0.53302 | **0.14422** | 0.25947 | 0.0 | 0.0 |
| **sparse_velocity** | 0.00131 | **0.05592** | **0.03339** | **0.11630** | 0.15966 | 0.22028 | 0.0 | 0.0 |
| sparse_posthoc | 0.00131 | 0.06238 | 0.04670 | 0.62247 | 0.19969 | **0.20884** | 0.0 | 0.0 |

要点：sparse velocity 在 joint RMSE、EE RMSE、EE velocity 上更好；jerk 略高。`joint_velocity_at_8_error` 也更好，但它仍是逐帧差分，不是任务空间约束本身。

### 2.5 Release 快照：6-DOF latent-level（16 held-out）

| 方法 | PSNR (dB) | anchor | joint RMSE | EE RMSE | EE velocity | jerk | table | cube target |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gt | **34.579** | 0 | 0 | 0 | 0 | 0.26645 | 0.0 | 0.0 |
| unconstrained | 20.302 | 0.34954 | 0.11233 | 0.09728 | 0.82383 | 0.18977 | 0.0 | 0.12749 |
| soft | 23.481 | 0.13205 | **0.04113** | 0.03960 | 0.28846 | 0.20479 | 0.0 | 0.02212 |
| posthoc | 23.583 | 0.00131 | 0.06615 | 0.04223 | 0.11556 | 0.20609 | 0.0 | 0.0 |
| **every_step** | **23.988** | **0.00131** | 0.05871 | **0.03383** | **0.11556** | **0.16382** | 0.0 | 0.0 |

注意：

- `gt` 的 jerk `0.26645` **高于** every_step 的 `0.16382`；所以“jerk 更低”不能直接解释为“更自然”。
- `anchor=0.00131` 是 clearance 后的值；投影本身的单元测试残差 `\(<10^{-5}\)`。
- 原始报告 `report_action_wm.md` 的 PSNR 为 `34.47 / 20.31 / 23.46 / 23.53 / 23.94`，与 release 快照差 \(<0.5\) dB；个别 better count 差 1（例如 every_step vs posthoc PSNR 11/16 vs 10/16）。以 release 快照为准，方向不变。

---

## 3. 像素级指标（6-DOF）

### 3.1 PSNR

\[
\mathrm{PSNR}
=10\log_{10}\frac{R^2}{\mathrm{MSE}},
\qquad
\mathrm{MSE}
=\frac{1}{T\,H\,W\,3}
\sum_{t,h,w,c}
\left(I_{t,h,w,c}-\hat I_{t,h,w,c}\right)^2.
\]

| 实验 | 输入范围 | \(R\) | 代码 |
|---|---|---:|---|
| 6-DOF `pixel_metrics` | uint8 \(0,\dots,255\) | 255 | `release/src/cnf_motion/metrics.py:psnr(..., data_range=255.0)` |
| 1.3B 双摆 `eval_1_3b.py` | float \(0,\dots,1\) | 1 | `psnr=10*np.log10(1/mse)` |
| 2-link 小视频 WM `run_world.py` | float \(0,\dots,1\) | 1 | `10*log10(1/mse.clamp_min(1e-8))` |

`psnr` 在 `MSE <= 1e-12` 时返回 `inf`。不要把 255 尺度和 1 尺度的 PSNR 直接比较；两者相差 \(20\log_{10}255\approx48.13\) dB。

### 3.2 橙色方块检测器

`metrics.py:detect_cube(frame)`：

1. RGB uint8 转 float 并除以 255；
2. 阈值：

\[
(R>0.55)\land(0.10<G<0.60)\land(B<0.32)\land(R>1.25G);
\]

3. 用 `cv2.connectedComponentsWithStats(mask, 8)` 做 8 连通域；
4. 保留面积 \(20\le a\le1500\) 的连通域；
5. 取面积最大的连通域的质心 \((x,y)\)；没有候选则返回 `None`。

代码里的实际条件：

```python
mask = (
    (red > 0.55)
    & (green > 0.10)
    & (green < 0.60)
    & (blue < 0.32)
    & (red > green * 1.25)
).astype(np.uint8)
```

这个检测器是 **6-DOF 合成数据专用**：它依赖方块橙色、桌面灰色、机械臂蓝色/绿色；对光照、遮挡、颜色漂移和生成视频的色偏都敏感。它不是 CoTracker，也不是学习式检测器。

### 3.3 Cube track / control / final / jerk

对每一帧 \(t\)，记预测视频检测到的方块质心为 \(\hat p_t\)，GT 视频为 \(p_t^{\text{gt}}\)，control 视频为 \(p_t^{\text{ctrl}}\)。只有两帧都检测到才计入。

**Cube track EPE**：

\[
E_{\text{cube track}}
=\frac{1}{|\mathcal{T}_{\text{valid}}|}
\sum_{t\in\mathcal{T}_{\text{valid}}}
\left\|\hat p_t-p_t^{\text{gt}}\right\|_2.
\]

单位 px。

**Control EPE**：

\[
E_{\text{control}}
=\frac{1}{|\mathcal{T}_{\text{ctrl}}|}
\sum_{t\in\mathcal{T}_{\text{ctrl}}}
\left\|\hat p_t-p_t^{\text{ctrl}}\right\|_2.
\]

单位 px；control 视频是 PyBullet 语义分割，橙色方块颜色 `(255,120,20)`，检测器同样适用。

**Final cube error**：

\[
E_{\text{final}}
=\left\|\hat p_{T-1}-p_{T-1}^{\text{gt}}\right\|_2
\]

（仅当最后一帧两边都检测到；否则 `NaN`）。

**Cube jerk（像素域）**：

\[
E_{\text{cube jerk}}
=\frac{1}{|\mathcal{J}|}
\sum_{t\in\mathcal{J}}
\left\|
\hat p_{t+3}-3\hat p_{t+2}+3\hat p_{t+1}-\hat p_t
\right\|_2,
\]

其中 \(\mathcal{J}\) 是预测视频中所有连续 4 帧都检测到的起点集合；少于 4 个有效点则 `NaN`。**没有时间缩放**，单位 px / frame³。

`pixel_metrics` 还返回 `detected_frames`（预测视频有效检测帧数）。检测率低时，track/control/final 的均值只覆盖少数帧，必须同时看 `detected_frames`。

### 3.4 1.3B 6-DOF 像素结果（原始报告，8 held-out）

| 方法 | PSNR (dB) | cube track (px) | control (px) | final cube (px) | cube jerk |
|---|---:|---:|---:|---:|---:|
| base（无 LoRA） | 22.99 | 24.80 | 24.85 | 14.74 | 71.81 |
| gt + LoRA | **24.32** | **6.28** | **6.65** | **2.92** | **28.50** |
| waypoint | 23.08 | 8.07 | 8.56 | 5.48 | 29.62 |
| velocity | **23.32** | **6.94** | **7.32** | **3.62** | 30.66 |
| posthoc | 22.53 | 7.97 | 8.43 | 3.20 | 30.38 |

这张表来自 `experiments/cnf_pickplace/report_pickplace.md`，不是 release/runs 快照；它对应 1.3B Control LoRA 的像素评测，PSNR 是 255 尺度。原始报告同时给出配对差：velocity vs waypoint PSNR `+0.24 ± 0.05 dB`（8/8），cube track `−1.13 ± 0.37 px`（7/8），control `−1.24 ± 0.41 px`（7/8）。这些像素收益与同批轨迹的 joint/EE 变差并存，必须分开报告。

---

## 4. 1.3B 双摆端点检测器

### 4.1 `segmentation_end_effector` 的实际逻辑

代码：`experiments/cnf_action_wm/run_world.py:305-357`。输入是 `[B,T,3,H,W]`、值域 \([0,1]\) 的视频张量。逐 batch、逐帧：

1. CHW → HWC，取 R/G/B；
2. **橙色连杆 mask**：

\[
(R>0.50)\land(G>0.12)\land(G<0.72)\land(B<0.48)
\land(R>1.12G)\land(G>1.10B);
\]

3. 如果橙色像素少于 5 个，该帧 `valid=False`；
4. 对橙色像素坐标 \((x,y)\) 去中心化，算 \(2\times2\) 协方差，`np.linalg.eigh` 取最大特征值对应的特征向量作为主轴方向 \(u\)；
5. 把像素投影到 \(u\)：\(s_i=(x_i-\bar x,y_i-\bar y)\cdot u\)；
6. `low` = 投影 \(\le\) 第 10 百分位的像素均值；`high` = 投影 \(\ge\) 第 90 百分位的像素均值；
7. **蓝色连杆 mask**：

\[
(B>0.50)\land(R<0.48)\land(B>1.25R);
\]

8. 蓝连杆中心 = 所有蓝色像素的均值；如果没有蓝色像素，用图像中心 \(((W-1)/2,(H-1)/2)\)；
9. 选择离蓝连杆中心更远的端点：

\[
\text{endpoint}=
\begin{cases}
\text{low}, & \|\text{low}-c_{\text{blue}}\|>\|\text{high}-c_{\text{blue}}\|,\\
\text{high}, & \text{otherwise}.
\end{cases}
\]

10. `valid=True`，写入 `points[b,t]`。

要点：

- 它 **不是** 6-DOF 的连通域方块检测器；它对整帧橙色像素做 PCA，不要求连通；
- 端点由 10/90 百分位均值给出，不是单个极值点，因而对少量离群像素有一定鲁棒性；
- 蓝色连杆中心只用于选“远端”，不参与端点坐标计算；
- 白色关节帽会遮住橙色连杆的一部分，造成约 **6.5 px** 的系统偏差（256 分辨率）。因此 1.3B 的绝对 EPE 不能直接和 128 分辨率实验比较。

### 4.2 `eval_1_3b.py` 的指标

`experiments/wan22_fun_control/eval_1_3b.py`：

- `ANCHOR_FRAMES=[0,8,16]`；
- `anchors_px=(np.array(meta[i]["anchors"])+1.0)*127.5`，因为 anchors 是 \([-1,1]\) 归一化坐标，256 分辨率下乘 127.5 再平移到 \([0,255]\)；
- `waypoint_epe` = 在 anchor 帧上，预测端点与 anchor 像素的欧氏距离均值；
- `track_epe` = 每一帧预测端点与 GT target 端点的欧氏距离均值，仅两边 `valid`；
- `psnr = 10*np.log10(1/mse)`，`mse=((video-target)**2).mean()`，值域 \([0,1]\)，即 `data_range=1`；
- `endpoint_speed` = `np.linalg.norm(np.diff(points,axis=0),axis=-1).mean()`；
- `endpoint_acc` = `np.diff(points,n=2,axis=0)` 的范数均值；
- `endpoint_jerk` = `np.diff(points,n=3,axis=0)` 的范数均值；
- `valid` = 有效帧比例。

这些端点速度/加速度/jerk 都是 **像素域逐帧差分**，没有除以 fps 或乘 16；只在同一实验内部比较。

1.3B 干净数据（4 held-out）：

| Control | waypoint EPE (px) | track EPE (px) | PSNR (dB) | endpoint jerk |
|---|---:|---:|---:|---:|
| waypoint only | 6.65 | 3.41 | 34.16 | 9.56 |
| **velocity only** | **6.42** | **3.22** | **34.31** | **9.20** |
| ground truth | 6.56 | 2.35 | **34.98** | 10.69 |

注意 GT 的 endpoint jerk `10.69` 高于 velocity 的 `9.20`：jerk 低不等于更自然。

1.3B 含噪 waypoint（8 samples，4 px 噪声）：

| 方法 | track EPE (px) | PSNR (dB) | joint RMSE |
|---|---:|---:|---:|
| noisy waypoint | 6.45 | 32.55 | 0.105 |
| noisy velocity（估计） | 6.82 | 32.11 | 0.147 |
| noisy oracle velocity | **5.26** | **32.96** | **0.091** |

结论：oracle 速度稳定更好（track `−1.18 ± 0.64 px`，6/8；PSNR `+0.41 ± 0.16 dB`，7/8；joint RMSE 8/8 更好）；从含噪低频 action 差分估计的速度反而更差（track `+0.37 px`，PSNR `−0.45 dB`，joint RMSE 6/8 更差）。瓶颈是速度估计器，不是投影本身。

### 4.3 它不是 CoTracker

- 6-DOF `pixel_metrics`：颜色阈值 + 连通域，见 3.2；
- 1.3B 双摆：颜色分割 + PCA 端点，见 4.1；
- CoTracker：用于 `experiments/cnf_wan_1_3b/video_eval` 的 PF_Wan 100-step 视频评测：首帧 8×8 网格，64 条轨迹，最大位移 \(\ge2\) px 视为动态（17/64），再算 dynamic EPE、full EPE、forward-backward cycle error。它与上面的颜色检测器不是同一套指标，不能混用。

---

## 5. 配对统计

### 5.1 定义

`metrics.py:paired_delta`：

\[
\Delta_i=m_i-b_i,
\]

\[
\bar\Delta=\frac{1}{n}\sum_{i=1}^{n}\Delta_i,
\qquad
\mathrm{SE}=\frac{\mathrm{std}(\Delta,\text{ddof}=1)}{\sqrt n}.
\]

- \(n\) 是同时存在 method 和 baseline、且两个值都有限的样本数；
- \(n\le1\) 时 `SE=NaN`；
- `better`：
  - `higher_is_better=True`：\(\sum_i \mathbf{1}[\Delta_i>0]\)；
  - `higher_is_better=False`：\(\sum_i \mathbf{1}[\Delta_i<0]\)；
- 没有做置信区间、正态性检验或配对 t 检验；只有均值和标准误。

`summarize_metrics.py` 的默认方向：

| 指标 | 方向 | 是否在默认 paired keys |
|---|---|---|
| `psnr` | 越高越好 | 是 |
| `anchor_error` | 越低越好 | 是 |
| `joint_rmse` | 越低越好 | 是 |
| `ee_rmse` | 越低越好 | 是 |
| `ee_velocity_error` | 越低越好 | 是 |
| `jerk` | 越低越好（但不等于更自然） | 是 |
| `table_clearance` | 越高越好（负值=穿透） | 否；默认脚本未配对 |
| `cube_target_error` | 越低越好 | 否；默认脚本未配对 |

`summarize_metrics.py --higher psnr` 只把 `psnr` 设为 higher-is-better；其余默认 lower-is-better。若以后把 `table_clearance` 加入配对，必须显式设 higher-is-better，否则“更负的穿透”会被误判为更好。

### 5.2 Release 快照的关键配对结果

**dense：velocity − waypoint**

| 指标 | 均值 ± SE | better |
|---|---:|---:|
| joint RMSE | \(+0.015942\pm0.000304\) | 0/16 |
| EE RMSE | \(+0.007654\pm0.000196\) | 0/16 |
| jerk | \(+0.043138\pm0.003750\) | 0/16 |
| `velocity_error` | \(+0.020438\pm0.000426\) | 0/16 |

**sparse：sparse_velocity − sparse_waypoint**

| 指标 | 均值 ± SE | better |
|---|---:|---:|
| joint RMSE | \(-0.005878\pm0.000927\) | 16/16 |
| EE RMSE | \(-0.004210\pm0.000991\) | 14/16 |
| EE velocity | \(-0.416713\pm0.034379\) | 16/16 |
| jerk | \(+0.015439\pm0.002640\) | 1/16 |
| `joint_velocity_at_8_error` | \(-0.039193\pm0.005200\) | 16/16 |

**latent-level：every_step − posthoc**

| 指标 | 均值 ± SE | better |
|---|---:|---:|
| PSNR | \(+0.404517\pm0.198863\) | 10/16 |
| joint RMSE | \(-0.007443\pm0.004663\) | 8/16 |
| EE RMSE | \(-0.008403\pm0.003554\) | 12/16 |
| jerk | \(-0.042274\pm0.008679\) | 15/16 |

**latent-level：every_step − soft**

| 指标 | 均值 ± SE | better |
|---|---:|---:|
| PSNR | \(+0.506596\pm0.132198\) | 12/16 |
| anchor | \(-0.130739\pm0.008852\) | 16/16 |
| joint RMSE | \(+0.017579\pm0.002032\) | 0/16 |
| EE RMSE | \(-0.005768\pm0.001890\) | 12/16 |
| EE velocity | \(-0.172899\pm0.033949\) | 14/16 |
| jerk | \(-0.040972\pm0.006645\) | 16/16 |

原始 `report_action_wm.md` 的 every_step − posthoc PSNR 是 `+0.41 ± 0.19 dB`、11/16；release 快照是 `+0.4045 ± 0.1989`、10/16。差 \(<0.01\) dB，better count 差 1，属于视频 WM 初始化/训练非确定性；结论方向不变。

---

## 6. 哪个指标用在哪类实验

| 实验 / 脚本 | 轨迹级 | 像素级 | 统计 / 样本 |
|---|---|---|---|
| 2-link 小视频 WM (`experiments/cnf_action_wm/run_world.py`) | latent EPE、anchor EPE/max、acceleration、jerk、valid rate | detector EPE、seg EPE、pixel jerk、PSNR (\(R=1\))、arm PSNR | 128 test videos，均值 |
| 1.3B 双摆 (`experiments/wan22_fun_control/eval_1_3b.py`) | waypoint EPE、track EPE、endpoint speed/acc/jerk、valid | PSNR (\(R=1\))、waypoint EPE、track EPE | 4 clean / 8 noisy，配对 delta |
| 6-DOF 轨迹生成 (`release/scripts/generate_controls.py`) | anchor、joint RMSE、EE RMSE、EE velocity、jerk、table、cube target、frame-8 joint velocity | —（只渲染视频，不算像素指标） | 16 held-out，逐样本 JSON |
| 6-DOF 1.3B 像素 (`experiments/cnf_pickplace/report_pickplace.md`) | 同一批 \(q\) 的轨迹指标 | PSNR (\(R=255\))、cube track、control EPE、final cube、cube jerk | 8 held-out，配对 delta |
| 6-DOF latent-level (`release/scripts/eval_action_wm.py`) | anchor、joint RMSE、EE RMSE、EE velocity、jerk、table、cube target | 代码支持 PSNR (\(R=255\))、cube track、final、cube jerk；release 快照实际报告 PSNR | 16 held-out，配对 delta |
| 1.3B PF_Wan 100-step (`experiments/cnf_wan_1_3b/video_eval`) | — | CoTracker dynamic/full EPE、forward-backward EPE、displacement | 1 clip / 1 seed；负结果 |

补充：

- `release/runs/eval_action_wm/metrics.json` 的键是 `psnr, anchor_error, joint_rmse, ee_rmse, ee_velocity_error, jerk, table_clearance, cube_target_error`；没有 `cube_track_epe`/`control_epe`。6-DOF 的 cube/control EPE 看 `report_pickplace.md`，不要把它当成 release 快照。
- 2-link 小视频 WM 的 PSNR 是 \(R=1\)；1.3B 双摆也是 \(R=1\)；6-DOF 是 \(R=255\)。跨 testbed 的 PSNR 不可直接比较。

---

## 7. 评测 caveats

1. **GT 是 oracle reference，不是上界。** 6-DOF 的 GT 是 minimum-jerk + IK + Legendre 拟合出来的人工连续轨迹；稀疏约束下运动多解，PSNR/joint RMSE vs GT 会惩罚其他合法、甚至更自然的解。`gt` 行衡量的是“给视频模型完整可行运动时它能渲染到什么程度”，是渲染层的 ceiling，不是运动层的上界。
2. **多解性。** 7-DOF 冗余臂 + 6 个 anchor 时，零空间姿态不唯一；关节 RMSE 低不等于任务空间更准，反之亦然。必须同时看 joint、EE、velocity、table、cube target。
3. **像素检测器敏感。** 6-DOF 方块检测依赖橙色阈值和连通域；生成视频色偏、遮挡、运动模糊都会导致漏检。1.3B 端点检测依赖橙色/蓝色分割和 PCA；白色关节帽遮挡约 6.5 px。检测率低时 EPE 只覆盖少数帧，不能只看均值。
4. **Jerk 低不等于自然。** GT 的 jerk 可能高于方法：6-DOF latent-level 的 gt `0.26645` vs every_step `0.16382`；1.3B 双摆的 gt endpoint jerk `10.69` vs velocity `9.20`。低 jerk 可能只是更保守/更平滑，不等于更像专家动作。
5. **PSNR 尺度不同。** 255 与 1 的 PSNR 差约 48.13 dB；6-DOF 的 23 dB 和 1.3B 的 35 dB 不是同一把尺子。
6. **Anchor 误差包含 clearance。** release 快照的 `anchor_error=0.00131` 是 `enforce_clearance` 之后的值；投影单元测试是 `<1e-5`。报告时要写清楚是“投影残差”还是“端到端 anchor 误差”。
7. **速度算子不匹配。** dense 的 `velocity_error` 是 `np.diff`，约束是 \(V_8\)；sparse 的任务空间目标来自 FK 中心差分，约束行用 \(V_8\)。不要用“约束精确”去推断“逐帧差分精确”。
8. **样本量。** 6-DOF 主结果 n=16，1.3B 像素 n=8，1.3B 双摆干净 n=4、含噪 n=8；SE 只反映样本间方差，不反映训练 seed 方差。当前没有多 seed 置信区间。

---

## 8. 代码与命令

| 功能 | 代码 |
|---|---|
| PSNR / 检测 / 轨迹指标 / 配对 | `release/src/cnf_motion/metrics.py` |
| 6-DOF 像素评测 | `release/scripts/eval_action_wm.py` |
| 6-DOF 轨迹生成与指标 | `release/scripts/generate_controls.py` |
| 均值 + 配对 delta | `release/scripts/summarize_metrics.py` |
| 1.3B 双摆端点检测 | `experiments/cnf_action_wm/run_world.py:305-357` |
| 1.3B 指标 | `experiments/wan22_fun_control/eval_1_3b.py` |
| PF_Wan 100-step CoTracker 评测 | `experiments/cnf_wan_1_3b/video_eval/run_video_eval.py` |

示例：

```bash
PYTHONPATH=src python scripts/summarize_metrics.py \
  --metrics runs/eval_action_wm/metrics.json \
  --reference every_step \
  --keys psnr,anchor_error,joint_rmse,ee_rmse,ee_velocity_error,jerk \
  --higher psnr
```

这条命令输出每个方法的 `mean_metrics`，以及 `method - every_step` 的 mean / SE / better / n。若要看 dense 或 sparse 的轨迹指标，把 `--metrics` 换成 `release/runs/controls_dense/metrics.json` 或 `release/runs/controls_sparse/metrics.json`，并把 `--reference` 换成 `waypoint` 或 `sparse_waypoint`。
