# 视频生成与世界模型

**目标：** 使用满足稀疏动作约束的连续动作场，构造视频模型的输入条件，研究生成视频是否更好地跟随目标动作。

**方法示例：** 生成动作场 → 对指定动作锚点施加 CNF 约束 → 渲染 control → 输入 Wan 1.3B 生成视频；与 Soft 条件进行匹配对照。

**当前状态：** 原工作区已有 input-level 的视频实验和评测；本目录仍是骨架，代码尚未迁入。latent-level 保留为探索，不是当前主线。最新结果和实验口径见 [Applications 进展](../../docs/applications_status_collaboration.md#3-application-a视频世界建模)。

**从这里开始：** 阅读 [README](README.md) 与 [任务清单](TASKS.md)，迁入 control 转换、视频生成和评测代码，记录模型和数据版本。

硬约束作用于动作场的指定条件；视频是否跟随需要单独测量。开发协作遵循[仓库开发者注意事项](../../AGENT.md)。
