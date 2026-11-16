# 视频与 world model

优先级：P1。状态：scaffold；本模块尚无可运行的模型 pipeline，旧工作区已有 Wan 1.3B input-level 审计结果。负责人未认领。

## 任务定义

2026-11-13 决定：视频/world-model 应用收敛到 **input-level**，以 `Wan2.1-Fun-V1.1-1.3B-Control` 为主证据。生成 motion/neural-field representation → CNF 在该表示上施加有限维线性约束 → render control → 以 control 条件驱动预训练视频模型。

最新因果/公平对照、可行性限制及证据路径见 [Applications 视频入口](../../docs/applications_status_collaboration.md#3-application-a视频世界建模)。Cosmos 2B、scratch U-Net 等 latent 探索保留历史，标记 deferred/exploratory，不进入当前主贡献或待办。动作生成和其他应用按原分工推进。

核心指标：anchor/nonanchor 的 video→control 与 video→GT 位置/pose 误差、漏检/失败率、运动幅度及图像质量。当前 HSV cube 位置仅是未经独立校准的二维代理。

保证边界：仅在适用的相容条件下保证实际约束表示/算子的性质，RGB 跟随属于测得效果。当前 GT 锚点与 clearance 在 8/8 scene 冲突，clearance 后约 `0.0013 rad` 的锚点误差不能称为最终轨迹严格可行；也不宣称 RGB 硬约束或全面质量优势。

后续顺序（尚未执行）：先统一锚点/clearance 任务并做可行性检查；再固定同 prior、同 noise、同条件信息比较投影，去掉独享 GT 速度；随后补独立跟踪/pose 与分层失败统计；最后扩展 scene/Wan 随机 seed 并保存配对统计。具体交付见 [TASKS.md](TASKS.md)。

## 负责范围

本模块拥有数据适配、生成先验、领域约束目标、领域指标及导出逻辑。通用算子和 CNF 求解器通过 core 接入；不导入其他任务内部代码。

```text
src/gcnf_video_generation/    本任务实现
configs/                 配置模板与后续正式实验配置
scripts/                 薄入口/数据准备
tests/                  离线领域/契约测试
TASKS.md                 可认领工作项与验收
module.json              状态/依赖/归属元数据
```

目录中的 `tests/` 与 `scripts/` 以实际目录名为准；代码尚待认领实施。

## 开始认领

阅读 [TASKS.md](TASKS.md)、[共享契约](../../docs/contracts.md)、[协作规则](../../CONTRIBUTING.md)。配置 `configs/full.example.json` 是未执行的通用模板，尚未编码本次 input-level protocol，其中 seed/样本数不表示已有结果。后续任务负责人应先完成可行性与信息匹配协议，再登记数据、模型及锁定环境，并补真实 setup/train/sample/evaluate 命令。

安装包骨架（从仓库根目录，已进入独立任务环境）：

```bash
python -m pip install -e '.[dev]' -e tasks/video_generation
```

拥有实际测试后，任务 CPU 检查入口约定为：

```bash
python -m unittest discover -s tasks/video_generation/tests -v
```

当前只有目录说明，没有任务测试；默认仓库 CI 不等于本任务实验已经通过。
