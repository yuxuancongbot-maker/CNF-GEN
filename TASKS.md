# Core 与基础设施待办

所有条目初始未认领；工作项 ID 用于规划，建仓后关联实际 Issue。

| ID | 优先级 | 交付 | 依赖 | 验收 |
|---|---|---|---|---|
| CORE-001 | P0 | 实现并冻结 v0.1 Field/Operator/Constraint/GeneratorAdapter 契约 | 无 | Shape 与 Action 各有离线合成 adapter 测试；单位、batch、可微阶数明确 |
| CORE-002 | P0 | 迁入线性 CNF 修正与稳定求解诊断 | CORE-001 | 相容、冗余、病态、不可行系统测试；独立验证残差与梯度 |
| CORE-003 | P0 | 输出修正与采样反馈两个独立接入点 | CORE-002 | 同一先验/种子/约束比较；不强制非线性 decoder 的 latent 线性 |
| CORE-004 | P1 | 保持结构的修正基与组合约束 | CORE-002 | 至少一个结构性质与测试时观测同时满足；说明有限点/全域边界 |
| INFRA-001 | P0 | 填写真实模块 owner、建立 GitHub Issue/Project、确定许可证 | 维护者建仓 | 真实账号有效，CODEOWNERS 与 review 设置实际生效 |
| INFRA-002 | P0 | 结果 schema writer、聚合器、历史证据登记 | CORE-001 | 拒绝不同单位/阶段/数据版本的误聚合；失败样本计入 |
| INFRA-003 | P1 | 按模块增加适配器 CPU 测试与显式 GPU 集成入口 | 各模块实现 | 默认 CI 离线无 GPU；完整集成有配置、环境和产物索引 |

现阶段已提供包/目录骨架、元数据/schema 检查、默认 CPU CI 模板。以上算法和平台操作仍待实际开发。

视频当前仅推进 Wan 1.3B input-level，latent 作为 deferred/exploratory 保留；动作生成与其他应用的工作项继续推进。视频的可行性检查、信息匹配对照、独立评测和 scene/Wan seed 扩展按 [视频 TASKS](tasks/video_generation/TASKS.md) 执行。此范围决定不改变 core 的通用接口或迁移状态。
