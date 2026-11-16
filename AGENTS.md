# Repository Guidelines

完整开发者约定见 [AGENT.md](AGENT.md)；应用进展与认领见 [Applications 协作文档](docs/applications_status_collaboration.md)。本文件保留编码助手的快速入口。

这是多任务研究 monorepo；当前仅有协作骨架、包占位和可执行仓库检查，算法尚未迁入。先运行 `git status --short` 和 `rg --files`，阅读认领模块的 README、TASKS 与 `docs/contracts.md`。

- 用户要求普通实验由 OpenCode 执行，Codex 负责规划、协调与审阅；不得用 Codex subagent 静默代替 OpenCode。任务说明、执行会话、日志与完成证据按 [AGENT.md](AGENT.md#代理分工codex-规划opencode-执行实验) 保存。
- 共享代码放 `src/gcnf/`；任务实现放 `tasks/<task>/src/gcnf_<task>/`。新模块遵循任务模板。
- 任务可导入 `gcnf`；core 禁止导入 `gcnf_<task>`；任务禁止直接互相导入，数据交换走版本化产物契约。
- 不在任务内部复制投影器。需要新的通用算子或求解器时，单独提出 core Issue/PR。
- Python 3.10+，四空格，类型注解，`snake_case` / `PascalCase`。Ruff 负责 lint/format；默认 CPU 测试采用标准库 unittest。
- 检查命令：`python scripts/check_repository.py`、`python -m unittest discover -s tests -v`、`ruff check .`、`ruff format --check .`、`python -m build`。安装：`python -m pip install -e '.[dev]'`。
- 算法单元/回归测试必须离线且不依赖 GPU。GPU 与数据集实验走显式的任务 pipeline，不进入默认 CI。
- 按完整端到端矩阵一次验证，统一报告最终指标、配置、产物和失败；只有具体 bug 才做最小调试检查。
- 明确区分场约束残差、离散化/后处理误差、像素/执行误差。不得把模板、未运行配置或历史结果写成新结果。
- 视频当前收敛到 Wan 1.3B input-level；latent 为 deferred/exploratory，不进入当前待办或主贡献。动作与其他应用继续推进；依据与可行性前置条件见 [Applications 视频入口](docs/applications_status_collaboration.md#3-application-a视频世界建模)。
- 不提交数据、模型、视频、凭证或机器绝对路径。配置用相对路径和环境变量；记录外部模型、数据与代码版本及授权。
- 不改动当前工作项以外的实验或他人未提交修改。接口和 schema 的兼容变更需要维护者与使用方共同审阅。
- 包/API 尚未实现时，文档必须明确标记 planned；不写虚假的运行命令或生成结果。
