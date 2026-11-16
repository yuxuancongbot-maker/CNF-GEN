# Application 目录

每个目录对应一项应用，合作者可在对应目录查看 `README.md` 和 `TASKS.md`，认领数据、模型、约束接入和评测工作。共享方法位于 [`src/gcnf/`](../src/gcnf/)。

| 应用 | 目录 | 研究内容 |
|---|---|---|
| 视频／世界模型 | [video_generation](video_generation/README.md) | 用受约束动作场构造视频生成的输入条件 |
| 动作生成／具身 Agent | [action_generation](action_generation/README.md) | 从稀疏约束生成连续动作；后续接入 Agent 的 action chunk |
| 3D 形状补全 | [shape_completion](shape_completion/README.md) | 满足表面点、法线等条件并生成完整形状 |
| 形变场生成 | [deformation](deformation/README.md) | 满足固定 handle 或指定局部位移条件 |
| 流体场生成 | [fluid_generation](fluid_generation/README.md) | 结合结构化流场与给定的速度观测 |
| PDE 场生成 | [pde_generation](pde_generation/README.md) | 研究初始条件、边界条件等约束下的场生成 |

各应用目录现为协作骨架：已有动作和视频实验代码仍在原研究工作区，尚未迁入；其余应用待开发。方向介绍与当前完成情况详见 [Applications 协作文档](../docs/applications_status_collaboration.md)。
