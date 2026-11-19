# 3D 形状补全工作项

以下是形状补全应用的实现和验收清单。 [运行说明](README.md)

| ID | 交付 | 依赖 | 验收与状态 |
|---|---|---|---|
| SHP-001 | SDFusion 连续场先验 + 观测数据适配：外部 checkpoint 经 `GCNF_SDFUSION_ROOT` / `GCNF_SDFUSION_WEIGHTS` 配置并校验固定 revision 09801f6d…；manifest/NPZ 严格契约加载；各向异性轴检查 | 无 | ✅ 已实现；离线合成/真实 manifest 由严格校验路径加载，权重缺失时给出明确错误 |
| SHP-002 | 点值与梯度 CNF 约束：三次 Hermite 场解析导数、128 条等式的 FP64 精确求解（值与导数残差闸门、秩/条件数/残差诊断）、三训练分支 | SHP-001, CORE-002 | ✅ 已实现；CPU 回归覆盖精确值/梯度、batched 列右端项、梯度路径；A100 preflight 验证锚点残差 ~1e-15，字段约束与网格误差分别记录 |
| SHP-003 | 六方法评测矩阵，共享 DDIM seed、显式表面 RNG、配对 bootstrap 和逐样本失败记录 | SHP-002, CORE-003 | ✅ 已实现，并用于固定留出清单的 64 个样本 |

## 交付物索引

- `src/gcnf_shape_completion/`：场、CNF、数据、条件、训练、CLI、评测实现（core 无改动）。
- `scripts/train.py`：preflight（no-update）与 train（显式 `--run` 才步进）入口。
- `scripts/evaluate.py`：六方法留出评测入口。
- `tests/`：离线 CPU 回归（18 项）。
- `configs/onet_chair_split.json`、`scripts/download_onet_chairs.py` 和 `scripts/prepare_onet_chairs.py` 提供固定数据 ID 与数据准备流程。

## 后续（不属于本工作项）

多视角、跨类别与真实扫描观测；同一条件下的多样性与成本评测；训练时 CNF 相对纯输出校正的收益需更多 seed/对象才能定论。新增共享算子应先提 core PR，不在本任务复制通用投影器。
