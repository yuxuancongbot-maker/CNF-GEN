# 共享接口契约 v0.1（设计，待 CORE-001 实现）

接口先在小范围内审阅和实现；下面是语义约定，不代表包中已有相应类。

## Core 负责什么

| 目录 | 拟接口 | 职责 |
|---|---|---|
| `src/gcnf/fields/` | `Field.evaluate(coordinates)` | 连续场求值；标明坐标/输出单位与可微阶数 |
| `src/gcnf/operators/` | `LinearOperator.apply(field)` | 点值、导数、积分等线性函数算子，不承担领域 IK |
| `src/gcnf/constraints/` | `ConstraintSet`、编译器 | 聚合算子、目标、容差与来源，输出有限系统 |
| `src/gcnf/solvers/` | `ConstraintSolver.solve(system)` | 秩、条件数、相容性、修正与实际残差 |
| `src/gcnf/sampling/` | `GeneratorAdapter`、约束反馈钩子 | 对接现有生成器；不训练领域 backbone |
| `src/gcnf/evaluation/` | 通用残差/成本记录 | 单位、阶段、作用范围；不计算领域感知质量 |

首个数值后端优先迁入现有 PyTorch 实现，但先保持任务依赖隔离。引入 JAX 等后端需要明确适配，不把不同框架 tensor 静默混用。空骨架阶段不预装大型运行依赖。

## 张量和算子约定

- 点查询坐标 `[batch, points, spatial_dim]`；输出 `[batch, points, channels]`。动作时间也是坐标，真实时间与归一化时间都必须记录。
- 线性系数系统 `A [batch, constraints, coefficients]`、`b [batch, constraints]`。支持的广播规则需由 CORE-001 定义并测试；不要隐式 squeeze batch=1。
- 任意神经 decoder 的 latent 不假设约束线性。一般场采用函数层 CNF 修正；线性系数表示使用矩阵投影。
- 解返回 corrected field/coefficients 和诊断：rank、condition estimate、最大残差、容差、feasible/failed、失败原因。
- 正则化求解、迭代非线性近似和精确线性求解分别标注。不同约束单位的行缩放必须可追踪。

## 任务适配器

每个任务负责数据、生成先验、领域表示与约束目标。通用投影/求解逻辑从 core 调用。任务适配器至少提供：输入到场/约束的转换、生成调用、独立领域评测、产物导出。

按噪声或可信度把观测划为精确要求、容差要求或软目标；不把所有测量值无条件作为硬等式。

## 动作到视频：产物连接 v0.1

一个动作产物由 metadata JSON 加 NPY/NPZ 数组组成。大数组不进入 Git；测试仅用小型合成 fixture。

metadata 至少含 `schema_version`、`coordinate_frame`、`robot_model`、`joint_names`、`angle_unit`、`time_unit`、数组路径与 SHA256。时间为秒且严格递增；关节角为 rad；`q [samples, frames, joints]`。记录 FK/相机标定来源、constraints、是否后处理、已知失败与有效样本 mask。

由视频任务的 adapter 将动作产物转成 renderer 所需 control。core 不包含 PyBullet、Wan 或相机渲染代码。VID-001 将这份设计实现为独立 schema 与合成契约测试；当前尚无可用交换实现。

## 结果连接

所有任务的公共结果包遵循 [`run.schema.json`](../schemas/run.schema.json)。领域的逐样本数组和额外指标作为 artifact，公共 JSON 保留来源与链接。主统计表只聚合相同 task、数据版本/split、约束布局、单位和评测口径，不将不同实验配置混成一个均值。

schema/API 不兼容修改先提独立 PR：版本升级、迁移说明、至少一个调用方契约测试同步更新。
