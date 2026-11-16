# GitHub 建仓与协作配置

当前是协作骨架；团队权限、Issue 和项目管理设置需由维护者配置。

远程仓库为 `CNF-GEN`。原工作区包含大量实验和第三方文件，不应整体推送到仓库。

## 团队设置

由维护者创建远程并确定可见性、许可证及有写权限的团队。填写 `.github/CODEOWNERS` 的真实账号后，按 core、shape、action、video 等路径分配 review；模板注释当前不生效。

建议 main 使用 PR 合并，并要求 `CPU / repository` 检查和一次审阅。核心接口/schema PR 额外邀请一个实际使用方 reviewer。CODEOWNERS 不会自动启用强制审批，需要管理员按实际 GitHub 方案配置。[GitHub 文档](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)

## Issues / Projects

各模块 TASKS 中的工作项转成真实 Issue，使用 Task Issue 模板。建议标签：`scope:core/shape/action/video/deformation/fluid/pde/infra`，`priority:P0/P1/P2/P3`，`type:method/baseline/data/eval/docs`。

Project 字段：负责人、scope、优先级、里程碑、依赖 Issue、状态。状态：Backlog → Ready → In progress → In review → Done；Blocked 明确列依赖。不把规划中的任务标成完成。

数据和模型 URL、结果存储和算力预约由团队选定，不写入未知账号或虚假路径。每次主要实验占用 GPU 前在对应 Issue 记录配置与预计资源，避免多人覆盖相同运行目录。

CI checkout/setup-python 使用已核实的官方 release commit SHA；升级时更新固定 SHA 与注释。[checkout 官方仓库](https://github.com/actions/checkout)
