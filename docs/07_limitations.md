# 07 限制与边界：哪些结论成立，哪些不成立

## 一句话 / 结论

- **成立**：位置硬投影在投影口径下精确；`every-step` 在受控实验里稳定优于 `posthoc`；稀疏 waypoint + 任务空间速度在轨迹级带来 14–16/16 的改善；1.3B Control LoRA 能把 control EPE 从 `24.85 px` 降到 `6.65 px`。
- **条件成立**：速度约束只在“稀疏 waypoint + 任务空间速度 + 安全 IK 投影”时给出稳定轨迹收益；像素 PSNR 8/8 更好，但 cube tracking 没有复现。
- **不成立**：velocity 无条件优于 waypoint；hard 在所有指标上都优于 soft；稀疏 velocity 改善像素跟踪；dense GT 是理论上限；当前场景有真实接触/摩擦；外部 1.3B/5B baseline 已做定量对比；1.3B PF_Wan 100-step 已产生有意义的视频提升。
- **负结果**：密集 waypoint + velocity 伤害轨迹（joint RMSE 0/16 更好）；hard 的 joint RMSE 差于 soft（0/16 更好），但 anchor 精度约 100 倍；2-link 无分支固定时 42.97% 翻转。
- **最大实现风险**：`enforce_clearance` 在投影后改动 anchor；release 端到端 `anchor_error=0.00131`，而投影单元测试 `<1e-5`。
- **评测风险**：6-DOF 和 1.3B 都用颜色检测器；白色关节帽约 6.5 px 偏差；jerk 低不等于自然。
- 以下给出支撑/不支撑的精确数字、会 falsify 这些 claim 的实验，以及未决问题。

---

## 1. 结论分级

### 1.1 支撑较强的结论

| 结论 | 证据（release / 报告） | 状态 |
|---|---|---|
| 位置硬投影精确 | `tests/test_projection.py` 位置/速度残差 `<1e-5`；原始报告投影口径 `<1e-6`；端到端 `anchor_error=0.00131`（clearance 后） | 成立，但必须写口径 |
| `every-step` 优于 `posthoc` | 2-link：PSNR `38.94` vs `28.35`，pixel jerk `2.07` vs `6.05`；6-DOF latent-level：PSNR `+0.4045 ± 0.1989`（10/16），jerk `−0.042274 ± 0.008679`（15/16）；dense 轨迹：joint RMSE `0.05087` vs `0.06323`，jerk `0.15588` vs `0.18633`（15/16） | 成立（受控实验） |
| 稀疏 + 任务空间速度改善轨迹 | joint RMSE `−0.005878 ± 0.000927`（16/16）；EE RMSE `−0.004210 ± 0.000991`（14/16）；EE velocity `−0.416713 ± 0.034379`（16/16）；jerk `+0.015439 ± 0.002640`（1/16 更好，略差） | 条件成立 |
| 1.3B Control LoRA 能跟随 control | base：PSNR `22.99`、cube track `24.80 px`、control `24.85 px`；LoRA：`24.32`、`6.28 px`、`6.65 px` | 成立（但不是严格冻结） |
| hard 的 anchor 精度远高于 soft | anchor `0.00131` vs `0.13205 rad`（16/16，约 100 倍）；PSNR `+0.5066 ± 0.1322 dB`（12/16）；jerk `−0.040972 ± 0.006645`（16/16）；EE velocity `−0.172899 ± 0.033949`（14/16） | 成立，但 joint RMSE 反向 |
| hard 的 joint RMSE 差于 soft | joint RMSE `0.05871` vs `0.04113`（hard 0/16 更好；差值 `+0.017579 ± 0.002032`） | 成立；这是 trade-off，不是全面碾压 |

### 1.2 条件性结论

| 结论 | 条件 | 证据 |
|---|---|---|
| 速度约束有益 | 稀疏 waypoint（6 个）+ 任务空间 3D 速度 + 安全 IK 投影 | 轨迹级 14–16/16；像素 tracking 未复现 |
| `every-step` 更自然 | 视频 WM 足够小、可控；评测能分辨采样路径差异 | 2-link 和 6-DOF latent-level 都成立；1.3B 大模型上尚未稳定复现 |
| 像素 PSNR 更好 | 同一组 control、同一评测协议 | 稀疏 velocity PSNR `+0.16 ± 0.07 dB`（8/8）；dense velocity `+0.24 ± 0.05 dB`（8/8） |
| hard 更适合任务 | 任务更看重约束精度、末端精度、jerk，而不是纯关节 RMSE | hard anchor/EE/jerk 更好；soft joint RMSE 更好 |

### 1.3 不支撑的结论

| 不支撑的 claim | 反例 / 边界 |
|---|---|
| velocity 无条件优于 waypoint | dense + velocity：joint RMSE `0.06681` vs `0.05087`（0/16 更好）；jerk `0.19902` vs `0.15588`（0/16 更好） |
| 稀疏 velocity 改善像素跟踪 | cube track `5.49` vs `5.29 px`（`+0.20 ± 0.36`，3/8 更好）；control `7.18` vs `7.07 px`（`+0.11 ± 0.19`，3/8 更好） |
| hard 在所有指标上优于 soft | soft joint RMSE `0.04113` vs hard `0.05871`；hard 0/16 更好 |
| dense GT 是方法的理论上限 | GT 是 minimum-jerk + IK + Legendre 拟合的人工连续轨迹；稀疏约束多解；PSNR/joint RMSE vs GT 会惩罚其他合法解 |
| 当前场景有真实接触/摩擦 | 抓取期间方块刚性跟随夹爪；没有接触力、摩擦、形变、滑移 |
| 已和 ProjFlow/PDM/Wan-Move/IOI 做定量对比 | 工作区没有这些方法的代码/权重/license；只有内部 row（gt、waypoint、posthoc、soft、every-step、unconstrained） |
| branch-aware 是核心贡献 | 2-link 的 42.97% 翻转在 6-DOF warm-start 下为 0 次；更像 naive 选解 artifact |
| 1.3B PF_Wan 100-step 已证明视频收益 | 动态 track `66.27` vs `73.29 px`，CNF 运动 `3.44 px` vs soft `2.32 px`，参考位移 `80.89 px`；两者都没复现扫动 |

---

## 2. 负结果

### 2.1 密集 waypoint + velocity 伤害轨迹

Release 快照（16 held-out，`controls_dense/metrics.json`）：

| 方法 | joint RMSE | EE RMSE | jerk | `velocity_error` |
|---|---:|---:|---:|---:|
| waypoint | **0.05087** | **0.03311** | **0.15588** | **0.19803** |
| velocity | 0.06681 | 0.04076 | 0.19902 | 0.21847 |
| posthoc | 0.06323 | 0.04896 | 0.18633 | 0.19718 |

配对 velocity − waypoint：

- joint RMSE `+0.015942 ± 0.000304`，0/16 更好；
- EE RMSE `+0.007654 ± 0.000196`，0/16 更好；
- jerk `+0.043138 ± 0.003750`，0/16 更好；
- `velocity_error` `+0.020438 ± 0.000426`，0/16 更好。

原始报告还给出同批 8 样本的末端指标：末端 Cartesian 误差 `0.0256` vs `0.0225 m`（0/8 更好），第 8 帧末端误差 `0.103` vs `0.074 m`（0/8 更好）。解释：密集位置 anchor 已经几乎决定整条轨迹；额外强制第 8 帧 oracle 速度会与全局多项式基冲突，产生 ringing；关节空间最小范数投影还会改变 7-DOF 零空间姿态。**这是“velocity 无条件更好”的直接反例。**

### 2.2 稀疏 velocity 改善轨迹，但像素跟踪未复现

轨迹级（release 快照，16 held-out）：

| 方法 | joint RMSE | EE RMSE | EE velocity | jerk |
|---|---:|---:|---:|---:|
| sparse_waypoint | 0.06180 | 0.03760 | 0.53302 | **0.14422** |
| **sparse_velocity** | **0.05592** | **0.03339** | **0.11630** | 0.15966 |
| sparse_posthoc | 0.06238 | 0.04670 | 0.62247 | 0.19969 |

配对 sparse_velocity − sparse_waypoint：

- joint RMSE `−0.005878 ± 0.000927`，**16/16 更好**；
- EE RMSE `−0.004210 ± 0.000991`，**14/16 更好**；
- EE velocity `−0.416713 ± 0.034379`，**16/16 更好**；
- jerk `+0.015439 ± 0.002640`，1/16 更好（略差）。

像素级（原始报告，8 held-out；1.3B Control LoRA）：

| 方法 | PSNR (dB) | cube track (px) | control (px) | final (px) | cube jerk |
|---|---:|---:|---:|---:|---:|
| sparse_waypoint | 22.22 | **5.29** | **7.07** | 3.39 | 26.35 |
| **sparse_velocity** | **22.38** | 5.49 | 7.18 | **3.19** | **24.15** |
| sparse_posthoc | 21.86 | 6.93 | 7.59 | 3.38 | 26.94 |

配对 sparse_velocity − sparse_waypoint：

- PSNR `+0.16 ± 0.07 dB`，**8/8 更好**；
- cube track `+0.20 ± 0.36 px`，3/8 更好（略差，不显著）；
- control `+0.11 ± 0.19 px`，3/8 更好（略差，不显著）；
- final `−0.20 ± 0.54 px`，4/8 更好（不显著）；
- cube jerk `−2.19 ± 3.33`，4/8 更好（不显著）。

结论：轨迹级的 16/16 收益没有干净地转化成像素跟踪收益。PSNR 更好，但 cube/control EPE 略差且不显著；渲染器仍是瓶颈。论文里只能写“轨迹级稳定正收益 + 像素 PSNR 正收益”，不能写“像素跟踪全面更好”。

### 2.3 hard 的 joint RMSE 差于 soft，但 anchor 精度高约 100 倍

Release 快照（16 held-out，`eval_action_wm/metrics.json`）：

| 方法 | PSNR | anchor | joint RMSE | EE RMSE | EE velocity | jerk |
|---|---:|---:|---:|---:|---:|---:|
| soft | 23.481 | 0.13205 | **0.04113** | 0.03960 | 0.28846 | 0.20479 |
| posthoc | 23.583 | 0.00131 | 0.06615 | 0.04223 | 0.11556 | 0.20609 |
| **every_step** | **23.988** | **0.00131** | 0.05871 | **0.03383** | **0.11556** | **0.16382** |

配对 every_step − soft：

- PSNR `+0.5066 ± 0.1322 dB`，12/16 更好；
- anchor `−0.130739 ± 0.008852 rad`，**16/16 更好**（约 100 倍）；
- joint RMSE `+0.017579 ± 0.002032`，**0/16 更好**（soft 更低）；
- EE RMSE `−0.005768 ± 0.001890`，12/16 更好；
- EE velocity `−0.172899 ± 0.033949`，14/16 更好；
- jerk `−0.040972 ± 0.006645`，**16/16 更好**。

解释：soft 看到 anchor 作为输入，可以生成一条“贴着 anchor 附近”的平滑轨迹，但不能保证精确命中；hard 用最小范数投影保证精确，代价是零空间姿态被改动，关节 RMSE 略高。**这是“精确约束 vs 关节平滑”的 trade-off，不是 hard 全面胜出。** 如果任务只看 joint RMSE，soft 会显得更好；如果看 anchor、EE、jerk、速度跟随，hard 更合适。

### 2.4 1.3B PF_Wan 100-step 负结果

设置：Wan2.1-T2V-1.3B + PhysisForcing；3 个训练 clip、1 个 held-out clip；100 optimizer steps；17 帧、256×256；单 seed；解码 20 步。held-out 时间步：

| 指标 | CNF | Soft |
|---|---:|---:|
| total objective | **0.26873** | 0.27649 |
| diffusion loss | 0.26613 | **0.26542** |
| low-level objective | **0.00259** | 0.01104 |

硬约束把 low-level objective 降低到 soft 的约 1/4.3，diffusion loss 只高 0.27%。但生成视频：

| 指标 | CNF | Soft | Reference |
|---|---:|---:|---:|
| dynamic track EPE | **66.27 px** | 73.29 px | 目标轨迹 |
| full track EPE | **8.37 px** | 11.82 px | 目标轨迹 |
| forward-backward EPE median | **1.02 px** | 1.32 px | 0.27 px |
| forward-backward EPE p95 | 18.80 px | **15.43 px** | 1.15 px |
| dynamic displacement | **3.44 px** | 2.32 px | 80.89 px |

CNF 和 soft 都只移动了参考位移的约 3%–4%，都没有复现真实扫动；CNF 的数值小幅领先不足以支撑视频质量 claim。边界：单 clip、单 seed、100 步、T2V backbone 不是真正的 image-to-video backbone。**这是一个初步负结果，不是对 hard constraint 的反证；但它说明“低层目标下降”不等于“生成视频变好”。**

### 2.5 2-link branch flips 42.97% 与 6-DOF warm-start 0 次

2-link 玩具（`docs/cnf_action_world_model.md`）：

| 每步投影方式 | 关节加速度 | Jerk | 分支翻转率 |
|---|---:|---:|---:|
| 固定首帧 IK 分支 | **0.0765** | **0.0350** | **0.00%** |
| 每步自适应选择分支 | 0.1660 | 0.0812 | **42.97%** |

6-DOF Puma560（`experiments/cnf_branch_6dof/report.md`）：

- 小/中等扰动 `0.05/0.15/0.30/0.8 rad`：naive / warm / branch 都是 **0 次翻转**；
- 大扰动 `1.5 rad`：naive 2 次翻转，jerk `0.588`，最大关节跳变 `4.50`；warm/branch 0 次，jerk `0.00074`，最大跳变 `0.081`；
- 大扰动 `2.5 rad`：naive 2–4 次翻转，jerk `0.588–1.026`，最大跳变 `4.50`；warm/branch 0 次，`0.00074`，`0.081`。

结论：分支翻转只在非常大的扰动 + naive 每帧独立选解时出现；标准 warm-start 已经能避免。2-link 的 42.97% 更可能是 naive 投影/选解策略的 artifact，不能直接推广到 6/7-DOF。**branch-aware 不应作为主创新点。**

---

## 3. 已知实现限制

### 3.1 数据与场景

- **合成数据**：PyBullet 生成，128 条训练 / 16 条 held-out；256×256、17 帧、8 fps；没有真实机器人数据。
- **刚性抓取**：抓取期间（帧 4–13）方块直接跟随 link 11，偏移 `0.040 m`；没有接触力、摩擦、形变、滑移、夹爪力控。
- **场景简单**：一个 5 cm 方块、平面桌面、固定相机；没有杂乱物体、遮挡、可变形物体、多物体交互。
- **IK 是启发式**：`solve_ik` 用 `calculateInverseKinematics`，damping `0.05`、500 次迭代、residual `1e-6`；数据生成时还要求 IK 残差 `<=1e-3 m`、关节跳变 `<=0.6 rad`。这不是全局最优或碰撞感知的 IK。
- **安全间隙不是碰撞规划**：`enforce_clearance` 只检查 link 11 高度，不检查完整夹爪几何、自碰撞、其他障碍物；它是逐帧 IK 后处理，不是 certified planner。

### 3.2 运动表示与导数基

- **17 帧 / degree 10**：10 个基函数、17 个采样点；采样 Gram 矩阵不是单位阵（max off-diagonal `0.6787`，`cond(B)=3.476`）。虽然 `B^+` 在基张成空间上精确，但外推/高频运动受基函数限制。
- **导数基**：默认 `discrete_autograd` 是为与原始实验对齐；在第 8 帧它与 `continuous` 一致到 `4.77e-7`，`finite_difference` 差 `0.6487`。release 的 continuous 复跑与默认复跑在轨迹指标上最大差 `<=1.3e-6`，所以**报告的速度结论不受归一化导数项影响**。但非中点帧差异不小（frame 0 的 disc-cont 最大差 `26.07`，frame 4/12 `0.1925`）；如果以后把速度帧移到非中点，必须显式做模式消融。这是“已核验的边界”，不是当前结果的主要风险。
- **算子不匹配**：dense 的 `velocity_error` 用 `np.diff`，约束用 \(V_8\)；sparse 的任务空间目标用 FK 中心差分，约束行用 \(V_8\)。因此“基函数速度精确”不等于“逐帧差分/任务空间中心差分精确”；sparse `ee_velocity_error=0.11630` 不是 0。
- **批次限制**：`project_sparse` 只支持 batch size 1；16 个样本串行投影，不能像 dense 那样 batch。

### 3.3 后处理与 anchor

- `enforce_clearance` 在投影之后执行，且不会重新投影；它可以改动 anchor 帧。
- release 端到端 `anchor_error=0.00131 rad`（dense/sparse 所有方法），而 `tests/test_projection.py` 的投影残差 `<1e-5`；原始 `report_pickplace.md` 的 `<1e-6` 是投影/未扰动口径。
- `validate_trajectory` 只检查关节限位、最大关节步长 `<0.8 rad`、手部 clearance `> -0.001 m`；不检查 anchor 是否仍精确，也不检查 cube 最终目标。
- 需要区分三个口径：投影残差、clearance 后 anchor 误差、视频/任务成功率。当前 release 没有把三者完全分开记录。

### 3.4 视频世界模型

- **小 3D U-Net**：128×128、17 帧、12.88M 参数、确定性残差预测；不是 diffusion/flow，不是大模型。训练 4000 步、batch 2、A100 约 3.5 分钟；PSNR 20–24 dB，渲染能力是主要瓶颈。
- **1.3B Control LoRA**：主体冻结但加了 LoRA（rank 64、alpha 32、1500 步、约 25 分钟 A100）；所以“冻结”不是严格冻结。base 对 segmentation control 零样本失败（control EPE `24.85 px`），LoRA 是必要接口。
- **像素空间不是硬约束**：CNF 只在运动空间硬；Wan/3D U-Net 的输出不保证满足 waypoint/速度。像素指标是评测目标，不是约束满足。
- **1.3B PF_Wan 100-step**：单 clip、单 seed、100 步、T2V backbone 不是真正 I2V；负结果只能说明该设置下没有视频收益。

### 3.5 评测

- **颜色检测器**：6-DOF 方块检测阈值 `R>0.55, 0.10<G<0.60, B<0.32, R>1.25G`，连通域面积 `20–1500`；对色偏、遮挡、运动模糊敏感。1.3B 端点检测用另一套橙/蓝阈值和 PCA；白色关节帽约 `6.5 px` 系统偏差。
- **不是 CoTracker**：6-DOF 和 1.3B 的像素指标都是颜色分割，不是学习式跟踪；CoTracker 只用于 PF_Wan 100-step 的视频评测。
- **PSNR 尺度不同**：6-DOF `R=255`，1.3B `R=1`；相差约 `48.13 dB`，不能跨 testbed 比较。
- **Jerk 不是自然性**：6-DOF latent-level 的 gt jerk `0.26645` 高于 every_step `0.16382`；1.3B 双摆 gt endpoint jerk `10.69` 高于 velocity `9.20`。低 jerk 可能是保守/平滑，不是更像专家。
- **单 seed / 小样本**：6-DOF n=16，1.3B 像素 n=8，1.3B 干净 n=4、含噪 n=8；SE 只反映样本间方差，不反映训练 seed 方差。没有多 seed 置信区间。

### 3.6 基线可跑性

- 外部 1.3B/5B baseline（ProjFlow、PDM、Wan-Move、Motion Prompting、Boximator、IOI）在当前工作区没有代码、权重或 license；不能声称定量对比。
- 官方 PF_Wan 权重可以在单卡 A100 40GB 上完成生成（81 帧、256×256、5 fps，约 47 秒），但它不是同一控制接口下的受控 baseline；其运动过度/轨迹错误，不能直接和 CNF 比较。
- 当前可用的内部对照只有：`gt`（dense oracle / rendering ceiling）、`unconstrained`、`soft`、`posthoc`、`every_step`、`waypoint`/`sparse_waypoint`、`velocity`/`sparse_velocity`。

---

## 4. 什么会 falsify 这些 claim

| Claim | Falsification 条件 |
|---|---|
| 位置硬投影精确 | 在 held-out anchor 上，投影后（clearance 前）\(\max|Ac'-b|>10^{-5}\) rad 的样本比例不可忽略；或 clearance 后 anchor 误差系统性 `>10^{-3}` rad 且任务成功率下降 |
| `every-step` 优于 `posthoc` | 多 seed（≥3）× \(n\ge32\) 的配对 95% CI 包含 0，或 PSNR/jerk 的符号反转 |
| 稀疏 + 任务空间速度改善轨迹 | 新 seed/数据上 joint RMSE、EE velocity 的配对 CI 包含 0；或改善只出现在单 seed |
| hard 的 anchor 精度约 100 倍于 soft | 重新评测后 hard/soft anchor 比值降到个位数，或 soft 通过后处理也达到同等 anchor 精度 |
| hard 更适合任务 | soft 的 joint RMSE 优势转化为更高的任务成功率/约束满足率；或 hard 的 EE/jerk 优势消失 |
| 稀疏 velocity 改善像素跟踪 | CoTracker/学习式检测器显示 cube/control EPE 没有改善，或当前 PSNR 优势消失；则像素 claim 是检测器 artifact |
| branch-aware 必要 | 7-DOF 或更大扰动下 warm-start 出现翻转；当前 6-DOF 大扰动 0 次翻转不支持 branch-aware 为主贡献 |
| 1.3B PF_Wan 硬约束改善视频 | 更多 clip、更多步数、真正 I2V backbone 下仍然没有 dynamic/full track 改善；当前 100-step 负结果已经削弱该 claim |
| 真实接触/摩擦可迁移 | 真实机器人或带接触的仿真中，刚性抓取假设导致滑移/掉落/失败；当前没有接触力模型 |
| 像素指标能证明运动更准 | 像素 PSNR 与轨迹 joint/EE 指标持续背离（当前 dense 已经出现：像素 velocity 更好，轨迹 0/16 更好） |

---

## 5. Claim 建议

### 可以写

- 测试时稀疏 waypoint 的位置硬约束可以做到投影口径精确（`<1e-5`，原始报告 `<1e-6`），端到端 anchor 误差受 clearance 影响（`0.00131 rad`）。
- 在受控 2-link / 6-DOF latent-level 实验中，`every-step` 比 `posthoc` 更自然：PSNR 更高、jerk 更低。
- 稀疏 waypoint + 任务空间速度在轨迹级改善 joint/EE/速度跟随（14–16/16），代价是 jerk 略高。
- hard 的 anchor 精度约 soft 的 100 倍；hard 的 EE/jerk/速度跟随更好；soft 的 joint RMSE 更低。
- 1.3B Control LoRA 能把 segmentation control 的跟随从零样本失败降到 `6.65 px` control EPE；这是轻量接口，不是严格冻结。

### 不要写

- “velocity 永远优于 waypoint”——dense + velocity 是 0/16 的反例。
- “稀疏 velocity 改善像素跟踪”——cube/control EPE 3/8，略差且不显著。
- “hard 在所有指标上优于 soft”——soft joint RMSE `0.04113` vs hard `0.05871`。
- “dense GT 是理论上限”——它只是 oracle reference / rendering ceiling。
- “当前方法有真实接触/摩擦”——没有；方块刚性跟随夹爪。
- “已和 ProjFlow/PDM/Wan-Move/IOI 定量对比”——没有可运行代码/权重/license。
- “branch-aware 是核心贡献”——6-DOF warm-start 0 次翻转；2-link 42.97% 更像 artifact。
- “1.3B PF_Wan 100-step 已证明视频收益”——没有；动态位移只有参考的约 3%–4%。

---

## 6. 未决问题

1. **多 seed / 更大样本**：当前 6-DOF n=16、1.3B 像素 n=8、1.3B 干净 n=4；需要 ≥3 seeds × n≥32，报告 seed 间方差和配对 CI。
2. **Anchor 口径拆分**：在 `enforce_clearance` 前后分别记录 anchor_error，区分投影残差和 clearance 扰动；或在 clearance 后补一次 anchor-only 投影。
3. **导数模式消融**：当前第 8 帧已核验 `discrete_autograd`≈`continuous`；若改用非中点速度帧，需要 3 种模式的消融和有限差分对照。
4. **速度算子对齐**：把 sparse 任务空间目标也改成 \(V_8\) 算子，或把约束行改成真正的逐帧中心差分，量化“算子不匹配”对 `ee_velocity_error=0.11630` 的贡献。
5. **像素检测器升级**：用 CoTracker/学习式检测器复评 6-DOF 和 1.3B，确认当前 PSNR/cube track 的分裂不是颜色阈值 artifact。
6. **真实接触/摩擦**：用夹爪力/接触模型替代刚性跟随，评估滑移、掉落、接触一致性；再到真实机器人数据（BridgeData/DROID）。
7. **外部 baseline**：等 ProjFlow/PDM/Wan-Move/Motion Prompting/Boximator/IOI 的代码、权重、license 可用后，做同协议定量对比。
8. **1.3B PF_Wan 扩大**：更多 clip、更多步数、真正 image-to-video backbone；加入 motion success 指标，避免静态/保守视频靠 cycle consistency 取巧。
9. **分支一致性**：在 7-DOF 和更大扰动下复测 warm-start 的翻转率；如果 0 次仍成立，branch-aware 只保留为 ablation。
10. **多解评测**：对同一组稀疏约束采样多条轨迹，报告约束满足 + 多样性 + 任务成功率，而不是只对一条 GT 算 PSNR/joint RMSE。
