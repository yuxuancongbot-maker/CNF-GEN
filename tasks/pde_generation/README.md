# PDE 场生成

优先级：P3。状态：scaffold；尚无可运行的模型 pipeline。负责人未认领。

## 任务定义

生成满足初边值或线性守恒要求的连续物理场，单列控制方程残差。

核心指标：IC/BC/积分约束残差、PDE residual、分布质量和推理成本。

保证边界：有限 collocation 不等于全域 PDE 保证；完整确定性条件可能唯一决定解。

## 负责范围

本模块拥有数据适配、生成先验、领域约束目标、领域指标及导出逻辑。通用算子和 CNF 求解器通过 core 接入；不导入其他任务内部代码。

```text
src/gcnf_pde_generation/    本任务实现
configs/                 配置模板与后续正式实验配置
scripts/                 薄入口/数据准备
tests/                  离线领域/契约测试
TASKS.md                 可认领工作项与验收
module.json              状态/依赖/归属元数据
```

目录中的 `tests/` 与 `scripts/` 以实际目录名为准；代码尚待认领实施。

## 开始认领

阅读 [TASKS.md](TASKS.md)、[共享契约](../../docs/contracts.md)、[协作规则](../../CONTRIBUTING.md)。配置 `configs/full.example.json` 是未执行模板，不表示结果。选定数据、模型及运行依赖后，在本模块登记来源与锁定环境，并补真实 setup/train/sample/evaluate 命令。

安装包骨架（从仓库根目录，已进入独立任务环境）：

```bash
python -m pip install -e '.[dev]' -e tasks/pde_generation
```

拥有实际测试后，任务 CPU 检查入口约定为：

```bash
python -m unittest discover -s tasks/pde_generation/tests -v
```

当前只有目录说明，没有任务测试；默认仓库 CI 不等于本任务实验已经通过。
