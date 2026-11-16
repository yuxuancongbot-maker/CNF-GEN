# 视频与 world model 待办

所有工作项未认领；下列 ID 尚未创建为远程 Issue。

当前仅推进 Wan 1.3B **input-level**。Cosmos 2B 与 scratch U-Net 等 latent 探索为 deferred/exploratory，保留既有代码/模型/结果，不进入当前待办或主贡献；动作生成及其他应用的工作继续。范围与证据见 [Applications 视频入口](../../docs/applications_status_collaboration.md#3-application-a视频世界建模)。下面均为后续交付，未在本次范围更新中执行。

| ID | 交付 | 依赖 | 验收 |
|---|---|---|---|
| VID-001 | 统一锚点/clearance 定义、可行性前置检查与运动产物/control 接口 | CORE-001 | 与动作方对齐 grasp/时序/几何；不相容任务显式拒绝；检查后处理后的全部要求；独立 schema/合成 fixture，不导入 action 内部代码 |
| VID-002 | 迁入 Wan 1.3B-Control input-level 适配并固定信息匹配协议 | VID-001 | 依赖/权重版本独立登记；同 prior、同 noise、同条件信息及固定 renderer/LoRA/输入比较无投影、最终投影、逐步投影；无 ours 独享 GT 速度 |
| VID-003 | 独立视频评测与扩展配对矩阵 | VID-002, ACT-003 | 分报 anchor/nonanchor、video→control/video→GT、漏检/失败率；独立跟踪/pose 校验；扩展 scene/Wan seed，保留逐样本配对统计、质量和失败案例 |

优先级依次为：① 可行性与任务定义；② 同 prior/noise/条件信息的投影对照；③ 分层指标与独立跟踪/pose；④ scene/Wan seed 扩展与配对统计。现有 8 scene、单 Wan seed43、未独立校准 HSV 代理不是扩展矩阵完成证据。既有 `configs/full.example.json` 仍为通用未执行模板，不能直接当作本次 protocol。

首次认领时在 GitHub Issue 指定负责人和本模块路径。需要 core 改动先提共享接口 PR；完整实验一次运行并一起报告指标、产物、成本与失败。

## 迁移进展（2026-11-16）

- VID-002 的 faithful 代码已迁入 `experiments/learned_basis_hybrid_cnf/` 与 `src/cnf_motion/`（含数据准备、固定/学习基训练、软/Hybrid 采样、约束与安全修复、control 渲染、Wan 1.3B + LoRA、评测与 PSNR 全链路），入口 `scripts/run_learned_basis_hybrid_cnf.py` 已通过 CPU 数值测试与 import 检查。这是“代码迁移”，不等于 VID-002 的信息匹配协议已完成；通用 adapter 与 core 契约拆分仍未做。
