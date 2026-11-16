# 实验交付规范

## 一个完整实验包

每次运行使用独立 `runs/<task>/<run_id>/`，至少包含：解析后的配置、代码版本与 dirty 状态、环境版本、数据/模型版本、结果 JSON、逐样本结果、失败记录及 artifact 索引。

`schemas/examples/run.json` 是未执行模板，不是历史或新实验结果。任务 pipeline 必须在真实运行后生成自己的 run ID 和字段。

## 结果口径

每个指标记录 `name/value/unit/stage/scope/count`。stage 区分 raw、constrained、postprocess、rendered、executed；scope 区分有限约束、连续域、中间场与最终输出。

例如解析场点值残差、提取网格误差、图像 tracker 误差分别记录；不将前者的精确保证转移给后两者。未计算指标不填零；保留缺失原因。示例 schema 允许空指标列表表示 incomplete。

失败和重试必须保留；报告首次成功率和预算内成功率。已有 checkpoint 复评和完整重训分别标注；训练 seed 与采样 seed 分别记录。

## 公平比较

固定任务输入、split、先验、采样 seed 和约束布局，比较无约束、软引导、最终修正、逐步修正及适用的外部强基线。若改变 prior conditioning、训练成本或约束集，必须单独列出。

多样性只在有效且满足要求的样本中计算，同时报告有效率；同一条件下的多个 latent 不作为独立场景统计。完整矩阵一次端到端执行并共同交付产物与失败，避免只挑成功视频。

## 计算环境

数据和 GPU 实验显式运行，不在普通 PR CI 自动下载权重/数据。每个任务将选定的运行依赖和精确环境快照记录在自己目录；共享元数据 CI 只使用 dev 环境。任务入口实施后，README 必须列出真实可运行的 setup/train/sample/evaluate 命令。
