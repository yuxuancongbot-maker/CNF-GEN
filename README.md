# Generative CNF

为生成神经场提供测试时点值与微分硬约束；在满足要求的同时，评估生成质量和可行多样性。

**当前状态：协作仓库骨架。算法尚未迁入，未执行新的模型实验，当前仅为协作骨架。** 各任务的状态与待办以其 `module.json`、`TASKS.md` 为准。

## 仓库分工

```text
src/gcnf/                 共享方法：场接口、算子、求解器、采样接入、通用评测
tasks/
  shape_completion/      3D 点/法线约束下的多样形状补全
  action_generation/     waypoint/导数约束下的动作生成
  video_generation/      受约束中间场驱动的视频/world model
  deformation/           handle/Jacobian 约束下的形变生成
  fluid_generation/      无散度结构与测试时观测的组合
  pde_generation/        初边值约束下的物理场生成
schemas/                 任务元数据和实验结果格式
tests/                   共享核心与仓库契约测试
scripts/                 仓库检查、任务状态等轻量工具
docs/                    接口、分工、实验规范、历史代码迁移
assets/                  小型架构图和示意图
data/                    数据获取说明；实际数据置于忽略目录
external/                第三方仓库/权重的来源与版本记录
.github/                 Issue、PR、CODEOWNERS 模板与 CPU CI
```

每个任务拥有独立的 `src/`、`tests/`、`configs/`、`scripts/`、`pyproject.toml`、`README.md`、`TASKS.md`、`module.json`。不把六个任务的模型依赖装进同一个环境。

## 开始工作

先看 [协作任务与认领](docs/workstreams.md)，再看 [接口契约](docs/contracts.md) 和自己任务的 `TASKS.md`。

面向合作者的六个应用进展、已有证据与验收要求汇总在 [Applications 进展与协作认领](docs/applications_status_collaboration.md)，可直接复制到飞书文档。

仓库元数据检查仅需 Python 3.10+：

```bash
python3 scripts/check_repository.py
```

开发环境与默认检查：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/check_repository.py
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
python -m build
```

任务开发使用自己的环境，例如：

```bash
python3 -m venv .venv-action
source .venv-action/bin/activate
python -m pip install -e '.[dev]' -e tasks/action_generation
```

任务包当前只含包骨架，模型依赖由对应迁移 Issue 添加；上面的安装不会下载 PyTorch、数据或权重。训练/评测入口尚未实现，不能将任务配置模板当成已经可运行的 pipeline。

## 开发规则

开始开发前阅读 [AGENT.md：开发者注意事项](AGENT.md)，其中集中说明模块边界、实验口径、多人共用服务器与交付要求。

- 任务代码可以依赖 `gcnf`；共享核心不依赖任务；任务之间不导入对方内部实现。
- 动作到视频通过版本化产物格式连接，视频开发者不必运行动作训练器。
- 先认领一个工作项，再开一个范围明确的 PR；共享接口变更单独提 PR。
- 每次完整实验同时交付配置、版本、种子、结果、失败与产物索引。

详见 [CONTRIBUTING.md](CONTRIBUTING.md)、[实验规范](docs/experiments.md)、[GitHub 配置](docs/github_setup.md)。

现有工作区的 `release/` 与 `experiments/` 按 [迁移清单](docs/migration.md) 逐项接入。本仓库不复制历史会话、第三方整个仓库、数据和 checkpoint。发布前由维护者确定代码许可证，各外部资源保留自己的授权要求。
