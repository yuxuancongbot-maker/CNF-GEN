# Figure 1 版式草图

目标：用“给定要求 → 共享的约束生成接口 → 应用输出”说明论文研究问题。具体矩阵、基函数和采样更新留给 Method 图。

当前草图：`figure1_concept_draft.svg`、`figure1_concept_draft.pdf`、`figure1_concept_draft.png`。

## 面板与证据

- **(a) Inputs**：点值、导数等要求，以及随机输入/上下文和生成先验。
- **(b) Shared field interface**：CNF 接入与满足相同锚点/一阶导数的曲线。曲线是解析示意，明确标为非模型采样；不能作为已验证多样性的证据。
- **(c) Existing application examples**：现有动作轨迹与 1.3B input-level 视频输出。动作选取变化最大的第 4 个关节，虚线为参考，橙点为给定 anchor。Control 与 RGB 展示同一样本第 0、8、16 帧。
- **底部扩展栏**：3D、形变、流体和 PDE 尚未形成实验结果，草图只作计划标注。Latent-level 视频仍在验证。

实验素材来自原工作区 `experiments/cnf_pickplace/outputs/input_baselines_1_3b_1k/` 的 sample 0，参考来自对应 held-out 数据。它们不是新运行，也不是已验证的多 seed 多样性展示；后处理后的实际约束残差以原始指标为准。

生成脚本位于原工作区 `presentations/build_figure1_concept.py`，使用既有 presentations 的 Python/Matplotlib 环境。脚本相对工作区定位数据；这份草图不为共享 core 增加依赖。

## 论文定稿时

1. 选择已经完成验证、视觉上可辨认的代表性应用，不必将六个方向平均挤进一张图。
2. 将解析曲线替换为真正的同条件多 seed 结果后，才能把它们标作生成样本。
3. 新增应用只在实际完成后加入结果面板；最终论文不使用“planned domains”充当实验结果。
4. 明确约束保证作用于场；视频的像素跟随、网格或执行误差另行评测。
5. 本图是叙事/版式草图，字体、尺寸与最终样本须按论文栏宽重新检查。

图注草稿：

> Overview of our approach and current application interfaces. Prescribed functional requirements and a generative prior are combined through a CNF-based field interface. The motion field can be used directly as an action trajectory or converted into input controls for a video world model. Curves in the concept panel are schematic; application panels use existing experimental outputs. Field-level constraint enforcement and downstream rendering fidelity are evaluated separately.
