# 3D 形状补全

**目标：** 给定部分观测，生成多个合理的完整三维形状，同时保持给定的表面条件。

**约束：** 指定表面点的隐式场值为零，并在可靠法线已知时约束对应的场梯度（32 锚点、128 条标量等式）。评估约束残差、网格几何误差、未观测区域质量与条件多样性。

已实现六组评测和三组微调入口。CPU 回归和真实数据上的 A100 预检均已通过。 [运行说明](README.md)

**从这里开始：** 阅读 [README](README.md)、[TASKS](TASKS.md)、[共享契约](../../docs/contracts.md)，再按需查看 `src/gcnf_shape_completion/`。

## 实现边界

- `src/gcnf_shape_completion/field.py`、`cnf.py`、`axes.py` 保存领域算子与专用 Hermite 泛函导数；core 未改动，不复制通用投影器。
- 外部官方 SDFusion 与权重不入库，只经 `GCNF_SDFUSION_ROOT` / `GCNF_SDFUSION_WEIGHTS`（或 CLI）定位并校验 revision 09801f6d… 与 SHA-256。
- 数据交换使用标准 `[{"object_id": ..., "path": ...}]` manifest 与 NPZ 契约；数据、checkpoint、大输出保持未跟踪。
- 不修改原始实验或其他任务代码；训练/评测只走本模块入口。

指定点的约束不自动保证全域 SDF、固定拓扑或网格顶点逐点满足要求。开发协作遵循[仓库开发者注意事项](../../AGENT.md)。
