# 动作生成待办

所有工作项未认领；下列 ID 尚未创建为远程 Issue。

| ID | 交付 | 依赖 | 验收 |
|---|---|---|---|
| ACT-001 | 迁入现有 Panda motion flow 与最小数据生成 | CORE-001 | 独立于旧工作区运行；数据/模型/种子一致；迁移不混入算法改进 |
| ACT-002 | 迁入 waypoint/速度与 IK/Jacobian 适配 | ACT-001, CORE-002 | 统一真实时间单位；投影与 clearance 后误差分开；不复制 core solver |
| ACT-003 | 公平比较、多样动作与版本化输出 | ACT-002, CORE-003 | 同 prior/约束的 soft/posthoc/stepwise 比较；保留 fair 2x2/hybrid 口径；输出供视频读取 |

首次认领时在 GitHub Issue 指定负责人和本模块路径。需要 core 改动先提共享接口 PR；完整实验一次运行并一起报告指标、产物、成本与失败。
