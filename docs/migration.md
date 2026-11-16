# 现有代码迁移清单

此文档中的旧路径相对于原工作区 `cvpr2027/`；迁移后的仓库必须独立运行，不通过父目录 `sys.path` 导入旧代码，不提交机器绝对路径。下列迁移均尚未实施。

| 旧来源 | 目标 | 负责工作项 | 注意事项 |
|---|---|---|---|
| `release/src/cnf_motion/basis.py` | core 通用场/基函数与 action 专用表示 | CORE-002 / ACT-001 | 归一化常数、真实时间导数与非中点行为 |
| `release/src/cnf_motion/motion_field.py` 投影部分 | `src/gcnf/solvers/` | CORE-002 | QR/SVD、相容性、正则误差诊断 |
| 同文件 flow prior 部分 | action task | ACT-001 | 保留模型设置，迁移先不改算法 |
| `constraints.py` | core 通用编译 + action IK/Jacobian adapter | CORE-001 / ACT-002 | 不把机器人依赖放进 core |
| `scene.py`、`dataset.py`、`soft_field.py` | action task | ACT-001 | 刚性跟随不称为真实接触动力学 |
| `video_wm.py`、Wan 相关适配脚本 | video task | VID-002 | 小模型与大型 renderer 分开；权重不复制 |
| `scripts/eval_fair_2x2.py` 与相关报告 | action/video 基线 | ACT-003 / VID-003 | prior conditioning、velocity、projection 分开 |
| `experiments/*/results*.json`、报告 | 实验登记与证据索引 | INFRA-002 | 保留原始版本/口径；不冒称重新运行 |

操作顺序：迁入最小可复用部分 → 适配 core 接口 → 单元/契约测试 → 一次完整旧配置复现 → 对齐指标 → 单独提交新方法。先迁入 action 的 CPU 运动部分，视频与权重适配另行处理。

不迁入旧虚拟环境、完整第三方 checkout、会话记录、checkpoint、数据或已有视频。对外部仓库记录版本与许可证，通过下载/准备脚本获取。另一实例的插拔任务等有完整代码、数据说明和结果后，再在 action 下增加任务配置。

## 已实施迁移（2026-11-16，待维护者审核）

Wan 1.3B input-level v2 实验（learned-basis Hybrid-CNF，16-scene，2026-11-14）已以
保持实验口径为目标迁入本仓库，并增加资源路径配置、运行入口与覆盖保护。共享运动代码与独立实验模块分别放置，避免虚构通用 API：

| 旧来源（原工作区） | 新位置（本仓库） | 说明 |
|---|---|---|
| `release/src/cnf_motion/` | `src/cnf_motion/` | 共享运动/动作/视频代码，逐字迁入；随根包 `pip install -e .` 安装，依赖见根 `pyproject.toml` 的 `motion`/`wan` extras |
| `experiments/cnf_pickplace/learned_basis_hybrid_cnf/` | `experiments/learned_basis_hybrid_cnf/src/learned_basis_hybrid_cnf/` | v2 实验包，唯一重构是把硬编码路径收敛到 `config.py`（`CNF_V2_*` 环境变量 / JSON 配置） |
| `run_learned_basis_hybrid_cnf_20261114.py` | `experiments/learned_basis_hybrid_cnf/scripts/run_learned_basis_hybrid_cnf.py` | 入口，新增 `--verify-config` |
| `finalize_learned_basis_hybrid_cnf_20261114.py` | `experiments/learned_basis_hybrid_cnf/scripts/finalize_learned_basis_hybrid_cnf.py` | CPU-only 统计/报告终结器 |
| `psnr_eval.py` | `experiments/learned_basis_hybrid_cnf/scripts/psnr_eval.py` | CPU PSNR 补测 |
| `video_task_validation_20261114.py`（v2 参考 + 评测器部分） | `experiments/learned_basis_hybrid_cnf/scripts/prepare_v2_references.py` + `train_evaluator.py` | 数据准备/评测器训练，仅抽取 v2 相关函数 |

未把 `cnf_motion` 拆成 `gcnf` core 与各任务包：为保证数值结果口径（float64 QR 无 ridge、
SLSQP 安全修复、scene 配对 bootstrap、MSE-first PSNR）先保留独立包。后续按 CORE-001/ACT/VID
接口拆分时以本次迁移的复现为基准对齐。迁移实验代码在根 `pyproject.toml` 的 `[tool.ruff]
extend-exclude` 中排除于脚手架 lint/format，以保持与原运行逐字一致；脚手架自身代码仍全量检查。

复现命令、缺失资源（Wan/LoRA/评测器/数据集权重不入 Git）与验证范围见
`experiments/learned_basis_hybrid_cnf/docs/reproduction.md`。
