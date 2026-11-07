# 03 Latent-level 动作世界模型与 CNF 投影

> 代码：`release/src/cnf_motion/{video_wm,soft_field,motion_field,constraints,metrics}.py`、
> `release/scripts/{train_video_wm,train_soft_prior,eval_action_wm}.py`
> 报告来源：`experiments/cnf_pickplace/report_action_wm.md`
> 历史证据：`docs/cnf_action_world_model.md`（2-link，代码不在 release）

## 摘要

- 这里的 “latent-level” 指动作侧的 **70 维 Legendre 运动系数 `c`**：先投影 `c`，再 `q=B c`，最后把 `q` 送进视频世界模型；**不是**视频模型内部的 feature/latent。
- 视频模型是 12.88M 参数的 3D U-Net：输入 30 通道 = 首帧 RGB 3 + control 视频 3 + 首帧 control 3 + `q/sin(q)/cos(q)` 21；输出 `[B,T,3,H,W]`。
- 数据：128 条训练视频 + 16 条 held-out；128×128、17 帧；确定性回归，不是 latent diffusion。
- 评测：`gt / unconstrained / soft / posthoc / every_step`；posthoc 与 every_step 使用同一组稀疏 anchor + 任务空间速度，只差投影时机。
- release 复跑：PSNR `34.58 / 20.30 / 23.48 / 23.58 / 23.99`；every-step 比 posthoc `+0.40 dB`，jerk 15/16 更好，末端 RMSE 12/16 更好。
- hard anchor `0.00131 rad` vs soft `0.13205 rad`（约 100 倍）；但 soft 的关节 RMSE 更低（`0.0411 vs 0.0587`），是“约束精度 vs 关节平滑”的交换。
- 历史 2-link 证据：固定分支 every-step 同时满足 waypoint 且保持低加速度/jerk；不固定分支时 branch flip 达 `42.97%`。
- 限制：确定性视频 WM、12.88M 很小、单 seed、16 样本、合成数据；不是生成式 latent diffusion。

## 1. “Latent-level” 在这里指什么

### 1.1 投影对象是动作系数，不是视频 latent

```text
q0 / anchors / task velocity
        │
        ▼
hard prior 或 soft prior
        │
        ▼
c ∈ R^70  ──[硬投影 A c = b]──►  c'
        │
        ▼
q = B c' ∈ R^{17×7}
        │
        ├─ enforce_clearance
        ▼
渲染 control 视频（256×256）→ nearest resize 128×128
        │
        ▼
12.88M 3D U-Net（首帧 + control + q）→ RGB 视频
```

关键点：

- 投影发生在 **70 维动作系数 `c`** 上，`c` 是 10 阶 Legendre × 7 关节的系数；
- `q = B c` 是投影后的关节轨迹；视频模型只看到 `q`（以及 control 视频和首帧）；
- 视频模型内部的 32/64/128/256 通道 feature map **没有**被投影或约束；
- 视频模型没有 VAE latent，也没有扩散/去噪采样；它是单次前向的回归网络。

因此更严格的称呼是 **action-latent-level**，而不是 video-latent-level。它比 02 的 1.3B 输入级控制
更靠近动作，但仍没有进入视频模型的内部表示。

### 1.2 与 02 的输入级控制对比

| 项目 | 02：1.3B Control LoRA | 03：12.88M action WM |
|---|---|---|
| 投影对象 | `c` → `q`，再渲染 control 视频 | `c` → `q`，直接作为视频模型条件 |
| 视频模型 | Wan2.1-Fun-V1.1-1.3B-Control | 12.88M 3D U-Net |
| 分辨率 / 帧数 | 256×256 / 17 | 128×128 / 17 |
| 视频模型内部是否投影 | 否 | 否 |
| 是否生成式 | 是（diffusion/flow 视频模型） | 否（确定性回归） |
| 评测 | PSNR、cube track、control EPE | PSNR + 轨迹指标 |

## 2. 12.88M 3D U-Net 架构

### 2.1 输入张量

| 输入 | 形状（训练） | 说明 |
|---|---|---|
| 首帧 RGB | `[B,3,H,W]` | 来自 GT 视频第 0 帧；沿时间展开到 `[B,3,T,H,W]` |
| control 视频 | `[B,3,T,H,W]` | 语义 control，`/255` 到 `[0,1]` |
| 首帧 control | `[B,3,1,H,W]` | 取 control 第 0 帧，沿时间展开到 `[B,3,T,H,W]` |
| 关节特征 | `[B,T,7]` | `q` |
| `q` 特征拼接 | `[B,T,21]` | `[q, sin(q), cos(q)]` |
| 空间展开 | `[B,21,T,H,W]` | `permute(0,2,1)[:, :, :, None, None]` 后 expand |
| 总条件 | `[B,30,T,H,W]` | `3 + 3 + 3 + 21 = 30` |

输入通道顺序严格为 `torch.cat([first, control, control0, q_features], dim=1)`。

### 2.2 张量形状（B=1，T=17，H=W=128）

| 阶段 | 模块 | 输出形状 |
|---:|---|---|
| 输入 | condition | `[1,30,17,128,128]` |
| enc0 | 2×Conv3d(3×3×3)+GN(8)+SiLU | `[1,32,17,128,128]` |
| down1 | Conv3d 32→64，stride `(1,2,2)` | `[1,64,17,64,64]` |
| enc1 | conv_block 64→64 | `[1,64,17,64,64]` |
| down2 | Conv3d 64→128，stride `(1,2,2)` | `[1,128,17,32,32]` |
| enc2 | conv_block 128→128 | `[1,128,17,32,32]` |
| down3 | Conv3d 128→256，stride `(2,2,2)` | `[1,256,9,16,16]` |
| enc3 | conv_block 256→256 | `[1,256,9,16,16]` |
| bottleneck | conv_block 256→256 | `[1,256,9,16,16]` |
| up3 | ConvTranspose3d 256→128，k `(3,4,4)`，stride 2，pad 1 | `[1,128,17,32,32]` |
| dec2 | concat x2 → conv_block 256→128 | `[1,128,17,32,32]` |
| up2 | ConvTranspose3d 128→64，k `(1,4,4)`，stride 2，pad 1 | `[1,64,17,64,64]` |
| dec1 | concat x1 → conv_block 128→64 | `[1,64,17,64,64]` |
| up1 | ConvTranspose3d 64→32，k `(1,4,4)`，stride 2，pad 1 | `[1,32,17,128,128]` |
| dec0 | concat x0 → conv_block 64→32 | `[1,32,17,128,128]` |
| out | Conv3d 32→3，1×1×1 | `[1,3,17,128,128]` |
| residual | `first + tanh(out)`，clamp `[0,1]` | `[1,17,3,128,128]` |

时间维只在 `down3` 下采样一次（17→9），`up3` 恢复到 17；`down1/down2` 的时间 stride 为 1。
空间维依次 128→64→32→16，再恢复到 128。

### 2.3 模块细节

| 项目 | 值 |
|---|---|
| `conv_block` | `Conv3d(3×3×3, padding=1)` → `GroupNorm(8,C)` → `SiLU` → 重复一次 |
| 归一化 | 所有 block 均为 `GroupNorm(8, C)`；C=32/64/128/256 均可被 8 整除 |
| 激活 | `SiLU` |
| 下采样 | 3 个 `Conv3d`，stride 分别为 `(1,2,2)`, `(1,2,2)`, `(2,2,2)` |
| 上采样 | 3 个 `ConvTranspose3d`，k 分别为 `(3,4,4)`, `(1,4,4)`, `(1,4,4)` |
| skip | `up3↔x2`、`up2↔x1`、`up1↔x0`，通道 concat |
| 输出 | `tanh` 残差，`first + residual`，再 clamp |
| 参数量 | `12,883,651`（约 12.88M） |

参数分组（精确计数）：

| 部分 | 参数量 |
|---|---:|
| 编码器（enc0/down1/enc1/down2/enc2/down3/enc3） | 5,862,976 |
| bottleneck | 3,540,480 |
| 解码器（up3/dec2/up2/dec1/up1/dec0/out） | 3,480,195 |
| **总计** | **12,883,651** |

## 3. 训练

### 3.1 数据与批处理

| 项目 | 值 |
|---|---|
| 训练视频 | 128 条（`runs/pickplace256`） |
| 分辨率 | 128×128 |
| 帧数 | 17 |
| RGB 缩放 | `cv2.INTER_AREA` |
| control 缩放 | `cv2.INTER_NEAREST` |
| 张量 | RGB `uint8 [128,17,128,128,3]`；control 同形；q `float32 [128,17,7]` |
| batch | 2 |
| 训练步数 | 4000 |
| 随机种子 | 0 |
| 数据增强 | 无；每步用 `torch.randint` 有放回采样 batch |
| 训练耗时 | 报告记录 A100 约 3.5 分钟 |
| 最终 loss | 约 `7.6e-4`（原始报告） |

### 3.2 损失函数

```text
foreground = max_c control[b,c,t,h,w] > 0.02
weight     = 1 + 10 * foreground            # 1 或 11
loss_mse   = mean(weight * (pred - target)^2) / mean(weight)
loss       = loss_mse + 0.1 * L1(pred, target)
```

精确形状：control 为 `[B,3,T,H,W]`，`amax(dim=1)` 得 `[B,T,H,W]`；`weight[:, :, None]`
广播到 3 个 RGB 通道。`target` 和 `pred` 都是 `[B,T,3,H,W]`，值域 `[0,1]`。

注意：`foreground` 的定义是“control 任一通道 > 0.02”，而桌面 control 灰度
`80/255≈0.314` 也大于 0.02，所以桌面像素同样获得 11× 权重；它不是只对物体前景加权。

### 3.3 优化器与调度

| 项目 | 值 |
|---|---|
| 优化器 | AdamW |
| 学习率 | `2e-4` |
| weight decay | `1e-5` |
| 调度 | cosine，`T_max=4000`，`eta_min=lr*0.05=1e-5` |
| 梯度裁剪 | `1.0` |
| 精度 | 代码默认 float32；报告训练使用 A100 |
| 保存 | `{"model": state_dict, "hidden": 32, "size": 128, "frames": 17, "steps": 4000}` |

### 3.4 输出

模型内部输出 `[B,3,T,H,W]`；`permute(0,2,1,3,4)` 后为 `[B,T,3,H,W]`。首帧被加到所有帧上，
因此第 0 帧严格等于输入首帧（残差为 `tanh`，加到首帧后再 clamp；第 0 帧不是恒等保证，
但训练目标使残差接近 0）。

## 4. Soft baseline

### 4.1 架构

`PandaSoftFlowModel` 与 hard prior 同为 coefficient-space flow，但额外看到 6 个 anchor。

| 项目 | 值 |
|---|---|
| 输入 | `c`（70）、`q0`（7）、6 个 anchor（6×7=42）、时间 `t` |
| `q0` 特征 | `[q0, sin(q0), cos(q0)]`，21 |
| anchor 特征 | 42 维直接 flatten |
| 时间特征 | 32 维正弦嵌入 |
| 分支 | state 21→256→256；anchors 42→256→256；time 32→256→256 |
| 主网络 | `(70+3×256)=838 → 512 → 512 → 70`，SiLU |
| 参数量 | `950,598`（约 0.95M） |
| 采样 | 50 步 Euler，无投影；seed `5000+index` |

### 4.2 训练

| 项目 | 值 |
|---|---|
| 步数 | 6000 |
| batch | 64 |
| 优化器 | AdamW，`lr=1e-3`，`wd=1e-4` |
| 调度 | cosine，`eta_min=5e-5` |
| 梯度裁剪 | 1.0 |
| seed | 1 |
| 数据 | 128 条轨迹，17×7 |
| soft weight | 20（在 20/50/200 中选 20；报告称 jerk/joint 最合理） |
| anchor 误差 | `0.13205 rad`（release 复跑；报告 0.1315） |

soft loss 的精确定义：

```text
flow_loss = MSE(v_theta(c_t,t,q0,anchors), c - eps)
clean     = c_t + (1-t) * v_theta(...)
physical  = (clean * std + mean).reshape(-1, 10, 7)
q_anchor  = einsum("kd,bdj->bkj", B[anchors], physical)
soft_loss = MSE(q_anchor, anchor_targets)
loss      = flow_loss + soft_weight * soft_loss
```

它不是“在采样后加惩罚”，而是训练时对一步去噪估计的 anchor 值做可微 MSE；推理时没有投影，
所以 anchor 误差非零是结构性的。

### 4.3 为什么 soft 的关节 RMSE 更低

- soft 把 6 个 anchor 作为输入，可以在 anchor 附近生成一条平滑轨迹；
- 但它不能保证精确命中，anchor 误差 `0.132 rad`；
- hard 用最小范数投影精确满足约束，代价是零空间姿态被修正，关节 RMSE 略高；
- 这是“约束精度/末端精度/jerk”与“关节空间平均误差”之间的 trade-off。

## 5. 评测协议

### 5.1 五种方法

| 方法 | q 来源 | anchor | 第 8 帧速度 | 投影 | seed |
|---|---|---|---|---|---|
| `gt` | GT `q` | — | — | 无 | — |
| `unconstrained` | hard prior | 无 | 无 | 无 | `4000+index` |
| `soft` | soft prior | 6 个（输入） | 无 | 无 | `5000+index` |
| `posthoc` | hard prior | 6 个 | 任务空间 EE | 仅最后 | `6000+index` |
| `every_step` | hard prior | 6 个 | 任务空间 EE | 每步 | `6000+index` |

- `posthoc` 与 `every_step` 使用完全相同的 anchor、速度目标和 seed，唯一差别是
  `project_every_step`；
- `unconstrained` / `soft` 不使用速度约束；
- `gt` 不调用 `enforce_clearance`；其余四种方法都在采样后调用；
- 评测没有重试/拒绝循环：每个样本、每个方法只采样一次。

### 5.2 安全 IK

`enforce_clearance` 对每个帧：

```text
safe_z = 0.3655   if 4 <= frame <= 13
         0.332    otherwise
if ee_z < safe_z:
    target = [ee_x, ee_y, safe_z]
    orn    = Euler[π, 0, yaw]
    q[frame] = IK(target, orn, rest=q[frame]+[open,open],
                  damping=0.05, max_iter=500, residual_threshold=1e-6)
```

它会把低于 safe height 的帧抬高，因此会轻微扰动 anchor。`gt` 不做这一步，所以 gt 的 anchor error
严格为 0；hard/soft 的 anchor error 是 `enforce_clearance` 之后的值。

### 5.3 视频生成与 PSNR

对每个方法：

1. 用 `q` 渲染 256×256 的 control 视频；
2. control 用 `cv2.INTER_NEAREST` 缩到 128×128；
3. 首帧来自 GT 视频（128×128，`INTER_AREA`）；
4. 视频模型输入 `first [1,3,128,128]`、`control [1,3,17,128,128]`、`q [1,17,7]`；
5. 输出 `[1,17,3,128,128]`，转 uint8；
6. PSNR 对 17×128×128×3 与 GT 视频逐像素计算：`10 log10(255²/MSE)`。

视频模型是确定性的：相同输入、相同权重得到相同输出；没有采样温度、没有噪声。

### 5.4 指标定义

| 指标 | 定义 |
|---|---|
| `psnr` | 全视频 PSNR，128×128，单位 dB |
| `anchor_error` | `max |q[anchors]-q_gt[anchors]|`，6 个 anchor，单位 rad |
| `joint_rmse` | `sqrt(mean((q-q_gt)²))`，17×7 |
| `ee_rmse` | `sqrt(mean(‖p_t-p_t^gt‖²))`，17 帧，单位 m |
| `ee_velocity_error` | `‖v_8-v_8^gt‖`，`v_8=(p_9-p_7)/2*16`，单位 m/归一化时间 |
| `jerk` | `mean_t ‖Δ³ q_t‖₂` |
| `table_clearance` | `min(cube_bottom_z - 0.30)`；0 表示无穿透 |
| `cube_target_error` | `‖cube_final_xy-target_xy‖` |

## 6. 结果

### 6.1 主表（release/runs 复跑，16 held-out）

| 方法 | PSNR | Anchor error | 关节 RMSE | 末端 RMSE | 速度误差 | Jerk |
|---|---:|---:|---:|---:|---:|---:|
| gt | **34.58** | **0** | **0** | **0** | **0** | 0.2665 |
| unconstrained | 20.30 | 0.3495 | 0.1123 | 0.0973 | 0.8238 | 0.1898 |
| soft | 23.48 | 0.13205 | **0.0411** | 0.0396 | 0.2885 | 0.2048 |
| posthoc | 23.58 | 0.00131 | 0.0662 | 0.0422 | 0.1156 | 0.2061 |
| **every_step** | **23.99** | **0.00131** | 0.0587 | **0.0338** | **0.1156** | **0.1638** |

原始报告值（供对照）：PSNR `34.47 / 20.31 / 23.46 / 23.53 / 23.94`，anchor error 同为
`0 / 0.3495 / 0.1315 / 0.00131 / 0.00131`。release 复跑重训了视频 WM，PSNR 有约 0.1 dB 的差异；
主表以 `release/runs/eval_action_wm/metrics.json` 为准。

### 6.2 配对差（every_step − baseline）

| 指标 | vs posthoc | vs soft | vs unconstrained |
|---|---:|---:|---:|
| PSNR | `+0.405 ± 0.199 dB`（10/16） | `+0.507 ± 0.132 dB`（12/16） | `+3.685 ± 0.504 dB`（14/16） |
| Anchor error | `0`（相同） | `−0.1307 ± 0.0089 rad`（**16/16**） | `−0.3482 ± 0.0347`（**16/16**） |
| 关节 RMSE | `−0.0074 ± 0.0047`（8/16） | `+0.0176 ± 0.0020`（**0/16**） | `−0.0536 ± 0.0089`（15/16） |
| 末端 RMSE | `−0.0084 ± 0.0036`（12/16） | `−0.0058 ± 0.0019`（12/16） | `−0.0634 ± 0.0110`（15/16） |
| 速度误差 | `≈0`（两者均 0.1156） | `−0.1729 ± 0.0339`（14/16） | `−0.7083 ± 0.1016`（**16/16**） |
| Jerk | `−0.0423 ± 0.0087`（**15/16**） | `−0.0410 ± 0.0066`（**16/16**） | `−0.0260 ± 0.0084`（13/16） |

“better/n”中，PSNR 按越高越好计数，其余按越低越好计数。`posthoc` 与 `every_step` 的 anchor
与速度目标相同，所以 anchor 差为 0、速度误差差小于 `1e-5`。

### 6.3 every-step vs posthoc

- PSNR：`23.99 vs 23.58`，`+0.405 dB`，10/16 更好；
- Jerk：`0.1638 vs 0.2061`，`15/16` 更好；
- 末端 RMSE：`0.0338 vs 0.0422`，12/16 更好；
- 关节 RMSE：`0.0587 vs 0.0662`，均值更低，但只有 8/16 更好。

结论：every-step 的主要收益是**更平滑的轨迹**（jerk 15/16）和更好的 PSNR，而不是所有样本的
关节位置都更准。PSNR 的 +0.4 dB 不是“全面碾压”，但方向与 02 的 dense every-step 一致。

### 6.4 hard vs soft：约束精度 vs 关节平滑

- hard anchor error `0.00131 rad`；soft `0.13205 rad`；hard 约 **100 倍**更准；
- hard PSNR `+0.507 dB`（12/16）、jerk `−0.0410`（16/16）、速度误差 `−0.1729`（14/16）、
  末端 RMSE `−0.0058`（12/16）均更好；
- 但 soft 的关节 RMSE 更低：`0.0411 vs 0.0587`（hard 0/16 更好）。

解释：soft 看到 anchor 作为输入，能生成“贴着 anchor 附近”的平滑轨迹，因此关节空间平均误差更低；
hard 用最小范数投影精确命中 anchor，会改变 7-DOF 冗余臂的零空间姿态，牺牲部分关节空间精度。
对任务而言，anchor 精度、末端精度、速度跟随和 jerk 更重要，所以 hard 更合适；如果只看关节 RMSE，
soft 会显得更好，这是指标选择问题。

### 6.5 与 GT 的差距

- GT PSNR `34.58`，every-step `23.99`，差约 `10.6 dB`；
- 差距来源：稀疏约束 vs dense GT 运动、小型视频 WM 的渲染能力、control 接口量化、合成数据分布；
- GT 是 rendering ceiling / oracle reference，不是方法上界；不能把 `10.6 dB` 全部归因于投影层。

### 6.6 与 02 的关系

- 02 的 1.3B 控制只把 CNF 用在 control video 输入级，Wan 内部 latent 完全不受约束；
- 03 把投影放到 `q` 之前的 70 维动作系数 `c`，视频模型直接消费投影后的 `q`；
- 两者都说明：**动作层的硬约束可以精确满足，但视频层的收益取决于渲染器和视频模型容量**；
- 03 的 every-step vs posthoc 证据比 02 更干净，因为两者的 anchor、速度目标、seed 完全相同。

## 7. 历史 2-link latent-level 证据

以下结果来自 `docs/cnf_action_world_model.md`，是 2-link 玩具实验，**代码不在 release 中**，
只作为历史证据记录，不能由本仓库脚本直接复现。

设置：双关节机械臂，64×64，16 帧，1024 条测试轨迹，blind world model（模型看不到未来 waypoint）。

| 方法 | Waypoint EPE | 加速度 | Jerk | 分支翻转 | kNN10 |
|---|---:|---:|---:|---:|---:|
| unconstrained | 17.804 px | 0.0773 | 0.0364 | 0.49% | 1.056 |
| soft | 0.580 px | 0.0794 | 0.0374 | 0.00% | 1.060 |
| posthoc | 3.35e-6 px | 0.2436 | 0.1435 | 0.20% | 14.389 |
| **every-step（固定分支）** | **2.21e-6 px** | **0.0765** | **0.0350** | **0.00%** | **1.154** |
| late50 | 2.27e-6 px | 0.0768 | 0.0351 | 0.00% | 1.253 |
| late60 | 2.27e-6 px | 0.0773 | 0.0355 | 0.00% | 1.312 |
| ground truth | 2.04e-6 px | 0.0795 | 0.0375 | 0.00% | 1.063 |

要点：

- posthoc 满足 waypoint，但加速度是 every-step 的 **3.18×**，jerk 是 **4.09×**；
- every-step（固定分支）的 kNN10 `1.154`，远低于 posthoc 的 `14.389`，接近 GT 的 `1.063`；
- 不固定分支时，every-step 的 branch flip 达到 **42.97%**；固定分支后为 `0.00%`；
- soft 的 waypoint EPE `0.580 px`，加速度/jerk 接近 GT，但不能精确命中 waypoint；
- `late50/late60` 表示从采样进度 50%/60% 开始投影（固定分支），不是 6-DOF release 的默认方法；
- kNN10 不能单独作为约束指标：unconstrained 的 kNN10 `1.056` 甚至低于 GT，但 waypoint EPE 17.804 px。

6-DOF 实验没有实现 branch-aware 投影；它的 anchor 直接给 GT 关节角，离散 IK 分支由数据隐含固定。
因此 2-link 的 `42.97%` 不能直接外推到 6-DOF，但它说明“naive every-step 不固定分支”是一个真实风险。

## 8. 结论与限制

### 8.1 结论

> 在 6-DOF 任务上，CNF 硬投影可以直接作用在 action WM 的 70 维运动系数上；
> every-step 比 posthoc 更平滑（jerk 15/16、PSNR `+0.405 dB`）；
> hard 的 anchor 精度比 soft 高约 100 倍，代价是关节 RMSE 略高（`0.0587 vs 0.0411`）。

这条结论补上了“latent-level 只在 2-link 上验证”的缺口，但仍是 12.88M 小模型、128×128、
16 个 held-out 的合成数据证据。

### 8.2 限制

- **确定性视频 WM**：单次前向回归，不是生成式 latent diffusion；没有 VAE latent、没有扩散采样、
  没有随机性。不要把它称为“视频 latent diffusion”。
- **投影不在视频内部**：只投影动作系数 `c`，视频模型的 feature map 不受约束；
  “latent-level” 是 action-latent-level。
- **模型很小**：12.88M 参数，128×128、17 帧；渲染能力有限，GT 与 every-step 差约 10.6 dB。
- **单 seed、16 样本**：没有多 seed 方差；表中的 `±` 是配对标准误，不是跨训练 seed 的置信区间。
- **合成数据**：方块在抓取期间刚性跟随夹爪，没有真实接触/摩擦；场景是准静态的。
- **soft baseline 只调了 soft weight**：20/50/200 中选 20，没有做更强的时间平滑/正则；
  报告没有给出 50/200 的完整指标表。
- **评测指标有限**：03 只报告 PSNR + 轨迹指标，没有 FVD/SSIM/CoTracker；PSNR 不能完全代表视频质量。
- **1.3B 结果不可直接比较**：02 的 1.3B PSNR 是 256×256、不同视频模型；03 的 34.58 是 128×128
  小模型 GT，两者不是同一评测。

## 9. 复现

### 9.1 一键复现

```bash
cd release
pip install -e .
bash scripts/reproduce_pickplace.sh
```

`reproduce_pickplace.sh` 会依次生成数据、训练 hard prior / soft prior / 12.88M 视频 WM、
生成 dense/sparse controls，最后运行 `eval_action_wm.py`。

### 9.2 分步命令

```bash
cd release
export PYTHONPATH="$PWD/src"

python scripts/generate_dataset.py --output runs/pickplace256 --count 128 --seed 0
python scripts/generate_dataset.py --output runs/pickplace256_eval --count 16 --seed 1

python scripts/train_motion_prior.py --data runs/pickplace256 --output runs/motion_model.pt
python scripts/train_soft_prior.py --data runs/pickplace256 \
  --output runs/soft_motion_model_w20.pt --soft-weight 20
python scripts/train_video_wm.py --data runs/pickplace256 --output runs/video_wm.pt

python scripts/eval_action_wm.py \
  --eval runs/pickplace256_eval --motion-prior runs/motion_model.pt \
  --soft-prior runs/soft_motion_model_w20.pt --video-wm runs/video_wm.pt \
  --output runs/eval_action_wm --samples 16 --device cuda
```

### 9.3 输出

| 路径 | 内容 |
|---|---|
| `runs/video_wm.pt` | 12.88M 视频 WM checkpoint |
| `runs/soft_motion_model_w20.pt` | soft baseline（0.95M） |
| `runs/motion_model.pt` | hard prior（0.74M） |
| `runs/eval_action_wm/metrics.json` | 5 方法 × 16 样本指标 |
| `runs/eval_action_wm/videos/` | 预测视频 |
| `runs/reproduce.log` | 完整日志 |

汇总：

```bash
python scripts/summarize_metrics.py \
  --metrics runs/eval_action_wm/metrics.json --reference every_step
```

### 9.4 数值口径

主表使用 `release/runs/eval_action_wm/metrics.json`（16 held-out，release 复跑）。原始报告
`experiments/cnf_pickplace/report_action_wm.md` 的 PSNR 为
`34.47 / 20.31 / 23.46 / 23.53 / 23.94`，与 release 复跑相差约 0.1 dB；这是重新训练视频 WM
带来的随机性。anchor error、关节 RMSE、末端 RMSE、速度误差、jerk 在两次运行中基本一致。
不要要求逐位对齐报告或复跑日志；以同配置重跑得到的 `metrics.json` 为准。
