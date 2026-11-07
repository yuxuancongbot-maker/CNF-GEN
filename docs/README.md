# 文档索引

本目录按编号组织 release 文档。第一次访问先读 [`../README.md`](../README.md)，再读 [`00_overview.md`](00_overview.md)。

## 文档

| 顺序 | 文档 | 一句话 |
|---:|---|---|
| 0 | [`../README.md`](../README.md) | 项目入口：一句话、CNF 澄清、方法 5 条、结果总表、快速开始、未包含内容。 |
| 1 | [`00_overview.md`](00_overview.md) | 总览：问题设定、两条接口、方法、实验矩阵、主要结果、claim 边界、阅读指南。 |
| 2 | [`01_method.md`](01_method.md) | 方法细节：Legendre 表示、flow matching、硬投影公式、张量形状与代码入口。 |
| 3 | [`02_pickplace.md`](02_pickplace.md) | 6-DOF Panda 抓取-放置：场景、数据、dense/sparse 约束、结果与失败模式。 |
| 4 | [`03_latent_wm.md`](03_latent_wm.md) | latent-level 小型 action WM：70 维运动系数、12.88M 3D U-Net、every-step / hard / soft 结果。 |
| 5 | [`04_wan_1_3b.md`](04_wan_1_3b.md) | Wan 1.3B 三类实验：input-level Control LoRA、PF_Wan 1.3B 2D 轨迹头、A14B oracle；明确 1.3B 7-DOF latent-level 未做。 |
| 6 | [`05_reproduction.md`](05_reproduction.md) | 复现指南：环境、一键命令、预期产物、可选 1.3B 外部复现。 |
| 7 | [`06_metrics.md`](06_metrics.md) | 指标定义：轨迹级 / 像素级指标、配对统计、PSNR 尺度与检测器口径。 |
| 8 | [`07_limitations.md`](07_limitations.md) | 结论边界、负结果与限制：soft joint RMSE、像素跟踪、1.3B 7-DOF、真实数据。 |
| 9 | [`08_reproduction_numbers.md`](08_reproduction_numbers.md) | release 完整复跑快照：dense / sparse / latent-level 数字与差异说明。 |

## 代码与配置入口

| 入口 | 一句话 |
|---|---|
| [`../configs/pickplace.json`](../configs/pickplace.json) | 6-DOF 报告实验的 documentation-only 配置参考。 |
| [`../scripts/reproduce_pickplace.sh`](../scripts/reproduce_pickplace.sh) | 端到端复现脚本：数据 → 运动先验 → video WM → control → 评测。 |
| [`../src/cnf_motion/`](../src/cnf_motion/) | 核心实现：`basis.py` → `motion_field.py` → `constraints.py` → `video_wm.py`。 |
| [`../tests/`](../tests/) | 单元测试入口：投影精确性、basis、metrics、video WM、scene smoke。 |

## 推荐阅读顺序

1. **`../README.md`** — 先看结论、结果表和能跑的命令。
2. **`00_overview.md`** — 建立全貌：问题、两条接口、方法、实验矩阵、claim 边界。
3. **`07_limitations.md`** — 先知道哪些不能 claim，再读细节。
4. **`01_method.md` → `02_pickplace.md` → `03_latent_wm.md`** — 方法与两个 testbed 的细节。
5. **`04_wan_1_3b.md`** — 1.3B / PF_Wan / A14B 的真实模型实验与边界。
6. **`05_reproduction.md` → `08_reproduction_numbers.md`** — 复现命令、预期产物和 release 复跑数字。
7. **`06_metrics.md`** — 需要解释指标或复核数字时再读。

## 说明

开发仓库中的原始实验报告（`experiments/cnf_pickplace/`、`experiments/wan22_fun_control/`、`experiments/cnf_wan_1_3b/`、`experiments/pf_wan_official_1gpu/`）包含逐样本表格和日志，不随 release 打包；本索引只列 release 内可访问的文档和入口。`08_reproduction_numbers.md` 记录 release 复跑；`README.md` 与 `00_overview.md` 在 latent-level 处同时保留原始报告值和复跑值，并标注差异（PSNR 约 0.1–0.5 dB）。
