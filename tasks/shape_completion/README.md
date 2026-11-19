# 3D 形状补全（SDFusion + 硬约束 CNF）

优先级 P0。已通过离线 CPU 回归和真实椅子数据上的 A100 预检。本模块提供三组微调和六组评测的运行入口。

本模块给出给定部分观测下的多样完整隐式形状生成：用官方 SDFusion 作生成先验，用 Hermite-Gaussian CNF 在 32 个观测表面点上施加 `f(p_i)=0, ∇f(p_i)=n_i`（共 128 条等式），并在连续校正场上比较六组方法。

## 目录与边界

```text
src/gcnf_shape_completion/   可导入实现（场、CNF、数据、条件、训练、CLI、评测）
configs/                 配置模板与固定数据 ID 清单
scripts/                 薄入口（train / evaluate）+ 数据准备脚本
tests/                   离线 CPU 回归
```

领域算子留在本任务：三次 Hermite 样条场及其解析导数、Hermite 高斯 CNF 及其泛函导数、xyz↔zyx 轴约定、掩码潜变量条件。core 未被改动，也没有复制别处的通用投影器。官方 SDFusion 源码与权重**不入库**；通过配置指向外部只读资源。

## 外部资源与固定版本

| 资源 | 配置 | 说明 |
|---|---|---|
| 官方 SDFusion checkout | `GCNF_SDFUSION_ROOT` 或 `--sdfusion-root` | 固定 revision `09801f6d04c04816e5827af981e2ae77be77ce5b`，加载时校验 |
| 官方 checkpoint 目录 | `GCNF_SDFUSION_WEIGHTS` 或 `--weights-dir` | `vqvae-snet-all.pth`、`sdfusion-snet-all.pth`，加载时校验 SHA-256 |
| 数据 manifest | `--manifest` | JSON list `[{"object_id": ..., "path": ...}]`，NPZ 契约见 [接口契约](../../docs/contracts.md) |

数据下载、准备和固定 ID 清单见 [数据入口](scripts/README.md)。模型与数据从外部发布来源取得，使用上表配置运行。

## 安装

已验证的组合使用 Linux x86-64、Python 3.10 和 CUDA 12.1 版 PyTorch。

```bash
python3.10 -m venv .venv-shape
source .venv-shape/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.4.1+cu121' 'torchvision==0.19.1+cu121'
python -m pip install -e . -e tasks/shape_completion \
  'numpy==1.26.4' 'trimesh==5.2.1' 'scipy==1.15.3' \
  'scikit-image==0.25.2' 'open3d==0.20.0' 'setuptools<81' \
  omegaconf einops termcolor tqdm h5py imageio PyMCubes matplotlib \
  opencv-python-headless joblib tensorboard kornia ftfy regex tbb plyfile
python -m pip install \
  https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl
```

获取两个 ShapeNet 权重后放在同一个目录。用上表的 Git revision 检出上游代码。 [SDFusion 权重下载说明](https://github.com/yccyenchicheng/SDFusion#download-the-pretrained-weight)

## 命令

从仓库根目录执行；`scripts/train.py` 与 `scripts/evaluate.py` 会自行定位本任务包。

```bash
# 1) no-update preflight（不写 checkpoint、不更新参数），三分支前向/反向与 FP64 校正检查
export GCNF_SDFUSION_ROOT=/path/to/SDFusion
export GCNF_SDFUSION_WEIGHTS=/path/to/official-weights

python tasks/shape_completion/scripts/train.py preflight \
    --manifest /path/to/test.json --arms plain soft cnf \
    --device cuda --out preflight-report.json

# 2) 训练入口：默认 dry-start，只打印配置；显式 --run 才步进
python tasks/shape_completion/scripts/train.py train \
    --manifest /path/to/train.json --arm cnf --steps 1000 --lr 1e-5 --run \
    --out-dir checkpoints

# 3) 六方法留出评测
python tasks/shape_completion/scripts/evaluate.py \
    --manifest /path/to/test.json \
    --checkpoint-dir /path/to/checkpoints \
    --out heldout.json --count 64 --ddim-steps 50 --seed 111
```

训练保留原有 CLI 与 mask 选项：`--mask-reduce half|all|any`（默认 `half`）、`--mask-erode`（默认 0）、`--t-weight-mode alphas_cumprod|one`、`--sigma-scale`（默认 2.0）、`--seed`、`--resolution`、`--batch-size`、`--n-query-grid`。评测接受 `--manifest`、`--checkpoint-dir`、`--out`、`--count`、`--ddim-steps`、`--seed`（另有 `--device`、`--surface-samples` 与外部路径选项）。

`configs/full.example.json` 提供训练和评估参数示例。评估命令会将逐样本误差、约束残差和配对统计写入指定输出文件。

## 三训练分支与六评测方法

训练分支只在约束执行上不同，共享数据、mask、噪声、时间步、查询点与初始权重：

- `plain`：掩码潜变量去噪 + 解码干净场的重建；
- `soft`：`plain` + 锚点值/法线软惩罚；
- `cnf`：在 FP64 精确校正场 `f = decode + Hermite 校正` 上计算重建，无锚点惩罚，记录残差。

VQ-VAE 冻结（`requires_grad=False`），解码使用直连量化使梯度回传到 3D U-Net。CNF 在 FP64 下直接求解、无 ridge/jitter，带秩/残差闸门；锚点处直接以独立求值检查值与导数残差。

评测固定六方法：`original`（官方预训练）、`plain`、`soft`、`post`（Soft + 事后 CNF）、`plain_post`（Plain + 事后 CNF）、`cnf`（CNF 训练权重 + 同一输出校正）。同一对象使用同一 DDIM seed（eta=0）；表面采样 seed 显式传入，CNF 与 plain_post 的事后校正共享同一采样偏移以配对比较。全局 RNG 不被读写。残差、网格 Chamfer 与未观测 Chamfer 分别报告，失败逐样本记录。

## 保证边界

128 条等式约束**连续校正场在该 32 个位置**的点值与一阶梯度。Marching Cubes 网格质量、全域 SDF 性质与拓扑由各自的几何指标衡量。

## 测试

默认测试离线、确定性、无 GPU、无需权重或数据：

```bash
python -m unittest discover -s tasks/shape_completion/tests -v
```

覆盖 xyz↔zyx 各向异性轴约定、样条解析导数与 autograd、掩码与噪声、严格数据契约、FP64 CNF 精确值/梯度求解与梯度路径、以及固定表面采样 RNG 的可复现性。

提交前检查：`python scripts/check_repository.py`、`ruff check .`、`ruff format --check .`、`python -m build`。
