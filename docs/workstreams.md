# 开发任务与分工

采用一个 monorepo、一个 main 分支、按工作项开短期分支。认领表中的角色不是已经分配的人；所有负责人初始为空，由团队填真实成员。

各应用的当前实验完成情况、具体交付与验收标准见 [Applications 进展与协作认领](applications_status_collaboration.md)。

| 工作流 | 优先级 | 负责人角色 | 本轮交付 | 待办入口 |
|---|---|---|---|---|
| Core | P0 | 方法/数值开发者 | 场、线性算子、求解器与生成器适配接口 | [TASKS](../TASKS.md) |
| Shape | P0 | 3D 开发者 | 点值/梯度约束的多样形状补全 | [TASKS](../tasks/shape_completion/TASKS.md) |
| Action | P0 | 机器人/动作开发者 | 迁入现有 flow 与公平基线，统一误差口径 | [TASKS](../tasks/action_generation/TASKS.md) |
| Video | P1 | 视频生成开发者 | 统一可行任务，从运动产物驱动 Wan 1.3B input-level，做信息匹配和独立跟随评测 | [TASKS](../tasks/video_generation/TASKS.md) |
| Deformation | P2 | 几何开发者 | handle 位移/Jacobian 约束的形变任务 | [TASKS](../tasks/deformation/TASKS.md) |
| Fluid | P2 | 物理场开发者 | 结构无散度与测试时点观测组合 | [TASKS](../tasks/fluid_generation/TASKS.md) |
| PDE | P3 | 科学计算开发者 | 初边值硬约束与 PDE 残差分开评测 | [TASKS](../tasks/pde_generation/TASKS.md) |
| Infra/Evaluation | P0 | 复现/评测开发者 | CI、产物契约、聚合工具与版本记录 | [TASKS](../TASKS.md) |

## 建议的人员安排

4 人：Core/Infra 一人、Shape 一人、Action 一人、Video/Evaluation 一人。形变、流体、PDE 暂留接口和待办，核心论文线完成后再认领。

3 人：Core/Infra 一人、Shape 一人、Action/Video 一人。5 人以上可拆独立 Evaluation，或认领一个 P2 任务。无需为每个目录配一名成员。

## 依赖与可同时开展的工作

首先完成 CORE-001 的接口审阅。Shape 和 Action 随后可独立实现数据/生成器适配；不必等待对方。Video 使用 [运动产物契约](contracts.md)，开发时可自行构造明确标注的合成测试输入。

任务模块的代码依赖只有 core。视频实验在数据上可能依赖动作输出，但不在 Python 包依赖上绑定 action；这两个依赖分别记录。

2026-11-13 视频路线决定：当前只推进 input-level；latent（Cosmos 2B、scratch U-Net）为 deferred/exploratory，历史代码/模型/结果保留，不进入当前主贡献或待办。动作生成和其他应用继续按上表安排。视频后续优先级为任务定义/可行性 → 同 prior/noise/条件信息投影对照 → anchor/nonanchor、video→control/video→GT、失败率与独立跟踪/pose → scene/Wan seed 扩展和配对统计；详见 [Applications 视频入口](applications_status_collaboration.md#3-application-a视频世界建模)。

里程碑：M0 接口与基线口径；M1 Shape/Action 各自完成完整 pipeline；M2 Video 与组合约束集成；M3 多 seed、完整失败统计与论文表。每个里程碑必须有可复现产物，不用“代码写完”代替验收。

## 任务完成标准

一个任务能按明确配置，从输入到最终指标完整运行；与必要基线同条件比较；有单元/契约测试；保存失败与重试；字段单位、约束作用范围、数据/模型来源明确。`module.json` 仅在实际迁移/验证后更新状态。
