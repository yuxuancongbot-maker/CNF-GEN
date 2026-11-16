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
