# Darcy Neural-Basis CNF (`MLP-289-PCA` + `std_proj` Flow Matching)

本目录提供 **Darcy 2D 压力场神经基 CNF 直接生成** 的完整可复现实现。与原有的固定 2D Fourier 基相比，该方法在**完全相同的 289 维基预算、200 个观测点和 Flow Matching 生成器结构**下，仅增加 **0.41M 参数（+12.3%）**，将 1,000 个留出场的生成相对 MSE 降低 **10.2 倍**（1,000/1,000 场全胜），同时保持 FP64 机器精度硬约束（`~1e-15`）和同等水平的同条件多样性。

## 核心方法（三步）

1. **Stage 1 — CNF 约束感知 MLP 神经基（`train_mlp_basis.py`）**：
   - 使用 4 层残差坐标 MLP（隐层 256，多尺度位置编码 + 原始线性坐标，**411,619 参数**）直接输出 289 个连续基函数 $\Phi_\theta(x,y)$。
   - **关键点**：训练目标直接穿过 **FP64 精确 KKT 约束求解层**，最小化“在 200 个观测点被硬钉死后的全场相对重建误差”，同时加入轻量观测矩阵条件数正则与 Gram 正则，使观测矩阵 $H$ 满秩 `200/200` 且条件数仅 **`1.61–1.95`**（对比 Fourier `85.4`、POD `579.0`）。
2. **训练集 PCA 正交坐标对齐（内置于 `train_mlp_basis.py`）**：
   - 冻结 MLP 后，在 9,000 个训练场上对正交化基做一次正交 PCA 旋转 $P_{\mathrm{pca}} = U_{\mathrm{pca}}^\top P_{\mathrm{orth}}$。
   - 该旋转 **100% 保持原 289 维函数空间与理论约束下界不变**（实测不变性误差 `5.4e-20`），同时去除系数相关性并按主干到细节排序；如果不做 PCA 对齐，任意 QR 坐标会导致各向同性 Flow Matching 失败（误差 `0.198`）。
3. **Stage 2 & 3 — 系数空间 CNF Flow Matching 与评测（`train_fm.py` / `eval_fm.py`）**：
   - 在对齐后的神经基系数空间中，复用完全相同的 `DarcyFlowModel`（3.34M 参数），以 `flow + 20*soft + 1.0*std_proj_geom` 训练 30,000 步；
   - 推理使用 50 步 Euler + 标准化空间 FP64 QR 硬投影（`std_proj`）。

## 运行方式

在仓库根目录（或已准备 `datasets/pde/darcy.h5` 的目录）下运行：

```bash
# Stage 1: 训练 MLP-289 神经基 + 训练集 PCA 对齐（A100 约 1 分钟）
python tasks/PDE_generation/neural_basis_darcy/train_mlp_basis.py \
  --data datasets/pde/darcy.h5 \
  --out outputs/darcy_neural_basis

# Stage 2: 训练匹配 CNF Flow Matching 生成器（可选 --basis-type mlp_pca / fourier / pod，A100 约 3 分钟）
python tasks/PDE_generation/neural_basis_darcy/train_fm.py \
  --data datasets/pde/darcy.h5 \
  --basis-type mlp_pca \
  --basis-npz outputs/darcy_neural_basis/mlp_289_pca.npz \
  --out outputs/darcy_neural_basis

# Stage 3: 1,000 留出场 × 20 配对采样全量评测
python tasks/PDE_generation/neural_basis_darcy/eval_fm.py \
  --data datasets/pde/darcy.h5 \
  --ckpt outputs/darcy_neural_basis/mlp_pca_std_cnf_fm.pt \
  --out outputs/darcy_neural_basis/eval_results.json
```

## 已验证实验结果（9,000 训练 / 1,000 留出测试 × 20 采样）

| 基 / 方法（289 维） | $\kappa(H)$ | 真实约束下界 (`full / unobs`) | 生成误差 (`full / unobs`) ↓ | 锚点最大残差 (`FP64 / f32`) ↓ | 归一化多样性 (`unobs`) | Darcy 散度误差 ↓ | 留出 PDE 相对 RMS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Fourier-289`（锁定基线 `e551b35 cnf_loss`） | 85.42 | `2.000e-3 / 2.025e-3` | `5.343e-3 / 5.412e-3` | `1.39e-15 / 3.19e-8` | `1.605%` | `9.350` | **`59.54`** |
| `Fourier-289`（同口径 30k `std_proj`） | 85.42 | `2.000e-3 / 2.025e-3` | `5.453e-3 / 5.523e-3` | `1.50e-15 / 4.73e-8` | `1.730%` | `9.390` | `59.89` |
| `POD-289`（训练集 SVD，同口径 30k `std_proj`） | 578.97 | `3.006e-3 / 3.044e-3` | `4.693e-3 / 4.753e-3` | `1.61e-15 / 6.30e-8` | `1.223%` | `11.715` | `72.71` |
| **`MLP-289-PCA`（同口径 30k `std_proj`）** | **1.95** | **`2.083e-4 / 2.110e-4`** | **`5.252e-4 / 5.319e-4`** | `1.61e-15 / 4.94e-8` | **`1.643%`** | **`4.927`** | `92.19` |
| `MLP-289-PCA`（确定性零先验 `z=0` 对照） | 1.95 | `2.083e-4 / 2.110e-4` | `4.114e-4 / 4.167e-4` | `1.39e-15 / 4.06e-8` | `0` | `4.627` | `89.22` |

- **配对 Bootstrap（1,000 场，10,000 次重采样）**：`MLP-289-PCA` 相对同口径 `Fourier-289` 未观测 rel MSE 差值 `-4.991e-3`（95% CI `[-5.433e-3, -4.589e-3]`，**1000/1000 场胜出**）；相对锁定 `e551b35 CNF` 差值 `-4.880e-3`（**1000/1000 场胜出**）。
- **参数量**：MLP-289 神经基 `411,619`（0.41M）+ `DarcyFlowModel` `3,336,481`（3.34M）= 总计 `3.75M`（比原 Fourier 基线仅增加 +12.3%）。

## 已知边界与下一步改进方向

1. **生成器与零先验差距**：`MLP-289-PCA` 的 10× 精度跃升主要来自“约束感知神经基 + PCA 先验加权投影”，30k 步 MLP 生成器（`5.319e-4`，多样性 `1.64%`）目前略高于确定性 `z=0` 投影（`4.167e-4`，多样性 `0`）。下一步可将生成器升级为带空间坐标注意力的 Transformer，或仅在 89 维零空间建模。
2. **二阶 PDE 残差**：Stage 1 未加入导数光滑或 PDE 物理正则，因此二阶稳态 PDE 残差（`92.2`）高于光滑解析的 Fourier 基（`59.5`），可在 Stage 1 中引入轻量梯度光滑/PDE 正则进一步优化。
