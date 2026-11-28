# 3D Shape Completion via Tangential-Free Normal-Only Matérn-5/2 RKHS CNF Coordinate Warp on Hunyuan3D-2mini

本目录提供在冻结预训练 3D 基础模型 **Hunyuan3D-2mini**（0.6B ShapeVAE + DiT，128³）上施加连续硬表面约束的 **切向自由法向 Matérn-5/2 RKHS 坐标变换 CNF（Normal-Only CNF Coordinate Warp）** 完整可复现代码与评测工具。

与在标量场中直接加性叠加修正（$f = f_0 + \Delta f$，容易在薄壁椅子上产生 32 颗表面凸起麻点、穿孔或锯齿腐蚀）不同，本方法在**坐标空间**构造最小 Sobolev $H^3$ 能量微分同胚形变：

$$f_{\mathrm{CNF}}(x) = f_0\big(x + u(x)\big), \qquad u(x) = \sum_{j=1}^{M} k(x, p_j)\, w_j$$

其中 $k(x, y)$ 采用严格正定 $C^2$ 的 **Matérn-5/2 + 弱惩罚仿射核**：

$$k(x, y) = \left(1 + s + \frac{s^2}{3}\right) e^{-s} + \gamma_{\mathrm{tr}} + \gamma_{\mathrm{lin}} (x^\top y), \qquad s = \frac{\sqrt{5}\,\|x - y\|}{\ell}$$

默认超参数：$\ell = 0.65,\ \gamma_{\mathrm{tr}} = 1.0,\ \gamma_{\mathrm{lin}} = 0.15$。

## 核心设计（三点关键改进）

1. **仅 32 条标量法向方程（释放切向滑移，零凹坑/零尖嘴）**：
   - 点对点坐标 Warp 锁死了全部 $32 \times 3 = 96$ 个 XYZ 分量，会把平面和拐角横向拉扯出凹坑与尖嘴。
   - 本方法在每个可见表面锚点 $p_i$ 上仅施加 1 条标量法向方程 $g_i^\top u(p_i) = \mathrm{rhs}_i$，由表示定理得到极小的 $32 \times 32$ 严格正定线性系统 $(K \odot (G G^\top))\alpha = \mathrm{rhs}$、$w_j = \alpha_j g_j$，受力纯沿表面法向、切向完全自由滑移。
2. **消除普通 TPS 的零代价仿射拉伸 + 锚点 $C^2$ 光滑**：
   - 普通 3D TPS 对 12 维仿射矩阵零惩罚，在仅观测到正面局部的椅子上容易把隐藏背面拉宽。Matérn-5/2 + 弱惩罚仿射核严格正定、无仿射零空间，且在锚点 $x = p_j$ 处仍保持解析 $C^2$ 光滑，可直接计算精确链式法则梯度 $\nabla f_{\mathrm{CNF}}(p_i) = (I + J_u(p_i))\nabla f_0(p_i + u(p_i))$。
3. **两阶段连续 FP64 精确求解 + 双向正立可见对齐**：
   - **Stage 1**：全表面法向一致匹配（防止薄壁吸到背面）+ 边缘弦向拉回线性区；
   - **Stage 2**：直接在冻结的 Hunyuan3D-2mini 连续 FP64 神经场上做增量最小 RKHS 范数牛顿微调（Armijo 回溯），将全部 32 锚点连续残差压至 $10^{-14}\text{–}10^{-13}$；
   - **双向对称正立可见对齐（`align_visible_upright_sym`）**：仅使用可见深度首击点与 32 个可见锚点法向（零隐藏几何泄漏），限制纯绕 $+Y$ 偏航角旋转并采用双向配对消除单向 ICP 30% 缩水。

## 文件说明

```text
hunyuan3d_cnf_warp/
├── warp_core.py          # Matérn-5/2 核、解析 C^2 雅可比、折叠审计、双向正立对齐、Normal-Only/P2P 求解器、Gaussian Patch、向量化光滑度与 bootstrap
├── hy3d_field.py         # 冻结 Hunyuan3D-2mini ShapeVAE + DiT 封装（FP32 128^3 网格 + FP64 连续锚点 autograd + canonical box 越界掩码）
├── run_eval.py           # 4-Arm 配对评测 CLI 入口（支持复用缓存 tokens 或从 RGBA 现算）
├── render_comparisons.py # Open3D 光追渲染器（正面 + 180° 绕 +Y 正立背面 + 全集总览图）
├── verify_results.py     # 独立验收门限脚本（FP64 残差、零折叠、无静默回退、零泄漏核查）
├── test_warp_core.py     # 离线确定性 CPU 单元测试（无需 GPU/权重）
└── requirements.txt      # 依赖清单（Hunyuan3D-2 上游源码与模型权重不入库）
```

## 快速运行与测试

### 1. 离线 CPU 单元测试（无需 GPU 或权重）

```bash
python -m unittest tasks/shape_completion/hunyuan3d_cnf_warp/test_warp_core.py -v
```

### 2. 64 把 ShapeNet 椅子全量 4-Arm 评测（需外部 Hunyuan3D-2 代码与 `tencent/Hunyuan3D-2mini` 权重）

```bash
python tasks/shape_completion/hunyuan3d_cnf_warp/run_eval.py \
  --manifest /path/to/shapenet_chairs/test.json \
  --data-root /path/to/shapenet_chairs \
  --hy3d-repo /path/to/Hunyuan3D-2 \
  --weights-dir /path/to/hunyuan3d-2mini-weights \
  --rgba-dir /path/to/upright_rgba_inputs \
  --out-dir outputs/hy3d_matern_normal_warp

# 渲染正背面对比图与全集总览图
python tasks/shape_completion/hunyuan3d_cnf_warp/render_comparisons.py \
  --run-dir outputs/hy3d_matern_normal_warp \
  --manifest /path/to/shapenet_chairs/test.json \
  --data-root /path/to/shapenet_chairs

# 独立验证全部门限
python tasks/shape_completion/hunyuan3d_cnf_warp/verify_results.py \
  --run-dir outputs/hy3d_matern_normal_warp
```

> 若已有确定性缓存的 ShapeVAE `tokens`（`.npz`），可传入 `--token-cache-dir /path/to/inference` 跳过 DiT 扩散采样，直接复用相同生成潜变量做公平 4-Arm 对比。

## 已验证实验结果（64 把留出 ShapeNet 椅子，128³）

所有 4 个 Arm 共享同一组 Hunyuan3D-2mini 生成 `tokens`、同一套双向正立可见对齐与同一套 `|q| > 1.01 -> +1.0` canonical-box 掩码（零隐藏泄漏、零微调、零静默回退）：

| 方法 | Full Chamfer（均值 / 中位）↓ | Hidden Chamfer（均值 / 中位）↓ | FP64 $\max\|f(p_i)\|$（中位 / 最大）↓ | 表面 $\min\det J$（中位 / 最小）↑ | 折叠椅数 ↓ | 二面角 Mean / p95 (°) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| *SDFusion original（64³ 历史基线）* | `0.04753` / — | `0.09569` / — | — | — | — | — |
| *SDFusion learned-basis CNF（64³ 基线）* | `0.01915` / — | `0.05702` / — | — | — | — | — |
| *V2 `hy3d_raw`（旧 24 轴对齐参考）* | `0.02495` / `0.00279` | `0.04886` / `0.02066` | — | — | — | — |
| `hy3d_raw`（双向正立可见对齐） | `0.00425` / `0.00231` | `0.02540` / `0.01782` | `1.00` / `1.04` | — | — | `6.46` / `31.25` |
| `hy3d_cnf_gaussian_patch`（128 方程局部补丁） | `0.00424` / `0.00228` | `0.02605` / `0.01919` | `2.22e-16` / `4.44e-16` | — | — | `7.63` / `33.90` |
| `hy3d_cnf_warp_p2p`（96 方程点对点 Matérn 消融） | `0.00240` / `0.00145` | `0.01671` / `0.01189` | `2.85e-13` / `1.57e-12` | `0.656` / `-0.020` | `1 / 64` | `7.13` / `31.13` |
| **`hy3d_cnf_normal_warp`（32 方程法向 Matérn）** | **`0.00186` / `0.00100`** | **`0.01348` / `0.01051`** | **`2.25e-14` / `1.53e-13`** | **`0.872` / `+0.080`** | **`0 / 64`** | **`6.59` / `31.07`** |

- **10,000 次 Paired Bootstrap（seed=111）**：
  - 对比同对齐 `hy3d_raw`：Full Chamfer **`-56.2%`**（95% CI `[-0.00340, -0.00145]`，**53/64 胜**），Hidden Chamfer **`-46.9%`**（95% CI `[-0.01651, -0.00775]`，**52/64 胜**）。
  - 对比 96 方程点对点消融 `hy3d_cnf_warp_p2p`（相同 Matérn-5/2 核）：Full Chamfer **`-22.3%`**（53/64 胜），Hidden Chamfer **`-19.3%`**（43/64 胜），$\min\det J$ **`+35.1%`**（**64/64 全胜**、0 折叠 vs 1 折叠），表面二面角更光滑 **`-7.6%`**（55/64 胜）。
  - 对比 `hy3d_cnf_gaussian_patch`：Full Chamfer **`-56.1%`**（54/64 胜），Hidden Chamfer **`-48.2%`**（54/64 胜），二面角更光滑 **`-13.7%`**（57/64 胜）。
  - 单把 Normal-Only Warp 求解均值仅 **`0.91s`**（零网络微调）。

## 已知边界

- 在近正方形极对称立方体沙发上（64 把中的 1 例 `59fd3d1409`），仅靠单视角可见点对齐可能选中旋转 90° 的朝向，强行形变会导致局部扭曲（`min_detJ = 0.080`，仍保持无折叠并将 Full Chamfer 从 `0.0280 → 0.0133` 减半）。
- 本方法精确强制 32 个锚点处连续隐式场 $f_{\mathrm{CNF}}(p_i) = 0$（FP64 残差 $\le 1.53 \times 10^{-13}$），未额外强制一阶法向等于粗糙 64³ GT 有限差分法向（以免引入加性偶极波纹）。
