# Method 写作草稿：Hard-Constrained Generative Neural Fields

状态：拟议方法结构与数学草稿，不代表新仓库已实现或实验已验证。有限线性函数约束的解析构造继承 CNF；生成接入、结构保持与先验质量需要进一步实现和比较。

## 方法总述

> We formulate constrained generation as learning and sampling a distribution over continuous fields whose admissible outputs satisfy a set of prescribed functional constraints. Our framework separates a constraint-determined component from a generative component that lies in the null space of the constraint operators. We instantiate this construction with different field representations and generative priors, while distinguishing exact field-level guarantees from downstream rendering or execution accuracy.

论文主 Method 定义统一接口和保证，Applications 再给出动作、形状和视频的实例。不同领域允许使用不同权重和先验。

## 3.1 Problem Formulation

令 `x` 为场坐标，`h` 为上下文，`z` 为随机变量。一个生成先验给出足够光滑的连续场：

\[
g_z(x)=D_\theta(x,G_\eta(z,h)),\qquad z\sim p_0(z).
\]

`G` 可以是 flow、diffusion、VAE 或权重生成器。只要能查询场及所需导数，不要求所有模型共享相同 latent 表示。

测试时给定有限个约束：

\[
\mathcal C=\{(\mathcal L_i,y_i)\}_{i=1}^{m},\qquad
\mathcal L_i[f]=y_i.
\]

每个 `L_i` 是对函数线性的评价算子，例如指定位置的函数值、偏导数、分量组合或适当定义的积分。多通道目标展开为标量约束后堆叠成：

\[
C[f]=y,\qquad C:\mathcal F\rightarrow\mathbb R^m.
\]

目标是在可行集合 `F_C={f:C[f]=y}` 上生成质量良好、具有多样性的样本。严格可行性与原始先验的精确条件分布是不同要求：解析修正只直接提供前者。

对一般神经 decoder，`C[D_theta(·,z)]` 通常对 `z` 非线性；不能直接套 latent 空间线性投影。

## 3.2 Constraint Compilation

把应用提供的要求转成 `C,y`，再在修正基上评价。选取 `K` 个足够光滑的向量值基函数，写作：

\[
\Psi(x)=[\psi_1(x),\ldots,\psi_K(x)],\qquad
M_{ij}=\mathcal L_i[\psi_j].
\]

这里的 `Psi` 可以是固定基或带参数的神经基，但必须满足对应阶数的可微性和数值稳定条件。约束改变时重新构造 `M`。

不同应用的编译例子：

| 应用要求 | 函数约束 |
|---|---|
| 指定表面点 | `f(p_i)=0` |
| 指定表面梯度 | `∇f(p_i)=n_i`；方向与尺度都被指定 |
| 动作 waypoint | `q(t_i)=q_i` |
| 动作速度 | `dq/dt(t_i)=v_i`；时间单位明确 |
| 形变 handle | `u(p_i)=d_i` |
| 连接两段场 | 对对应边界点/导数作线性差分约束 |

对冗余等式采用秩揭示分解；对冲突目标或表示能力不足显式返回不可行/失败。噪声观测不默认当作精确要求。

## 3.3 Hard-Constrained Field Parameterization

对于 `M` 行满秩的情况，定义一个满足要求的特解与一个齐次修正算子：

\[
f_{\mathcal C}^{p}(x)=\Psi(x)M^\dagger y,
\qquad
\bigl(\mathcal Q_{\mathcal C}[g]\bigr)(x)=g(x)-\Psi(x)M^\dagger C[g].
\]

最终生成场为：

\[
\boxed{f_z^{\mathcal C}(x)=f_{\mathcal C}^{p}(x)+\bigl(\mathcal Q_{\mathcal C}[g_z]\bigr)(x)}.
\]

等价地：

\[
f_z^{\mathcal C}(x)=g_z(x)+\Psi(x)M^\dagger(y-C[g_z]).
\]

约束负责确定特解，随机生成器负责剩余变化；齐次算子确保这些变化不改变指定要求。

**可行性命题。** 若算子线性、场和基在算子定义域内、`M` 行满秩，则精确算术下对每个合法 `z`：

\[
C[f_z^{\mathcal C}]=MM^\dagger y+(I-MM^\dagger)C[g_z]=y.
\]

对于秩不足但相容的系统，可在独立约束子空间求解；更一般地需要 `y-C[g_z]` 属于 `M` 的值域。实际保证取决于求解容差、条件数与算子数值误差，不能一律写为机器精度。

这部分是 [CNF](https://arxiv.org/abs/2306.08943) 思想的生成式使用，不将该解析公式或零空间分解本身当作新发明。除非明确给出函数空间度量与相应基，否则称其为 constraint correction，不默认称作正交投影。

## 3.4 Coupling with Generative Priors

统一约束层有两种接入位置，但共享同一个 `C,y,Psi` 机制。

### Output-field coupling

从现有先验采样 `g_z`，输出 `f_z^C`。构建约束系统本身不要求更新生成模型权重，适用于一般 neural-field decoder。

这是最直接的基础版本，也应作为必要 baseline。它可能改变全局语义或生成分布，不能仅凭约束精确就声称质量、多样性保持。

### Sampling-time coupling

若生成模型直接生成线性场系数：

\[
f_c(x)=\Phi(x)c,\qquad Ac=y,\qquad A_{ij}=\mathcal L_i[\phi_j],
\]

则可在运动 flow 采样中调用同一线性修正：

\[
\tilde c_{s+\Delta s}=c_s+\Delta s\,v_\eta(c_s,s,h),
\qquad
c_{s+\Delta s}=\Pi_{A,y}(\tilde c_{s+\Delta s}),
\]

\[
\Pi_{A,y}(c)=c+A^\dagger(y-Ac).
\]

`s` 是生成时间，与动作的物理时间 `t` 分开。现有动作代码具备该形式的实现基础。

一般 diffusion/flow 的中间噪声状态不天然是最终干净场；不要求所有中间状态都满足最终约束。针对这些模型，需要明确定义干净样本估计、约束修正及下一步反馈，而不是无条件把 latent 投到最终场可行集。

[ECI](https://arxiv.org/abs/2412.01786) 与 [ProjFlow](https://arxiv.org/abs/2602.22742) 已研究相关采样修正机制，必须作为比较对象。Method 中只写实际实施并验证过的接入，不宣称每步投影或 clean-endpoint 修正是独有发明。

如果额外训练一个显式读取约束的生成器，单独定义其训练分布、损失和成本。不能将这种路线与固定先验的“换约束无需重训”混为一谈。

## 3.5 Preserving Structural Constraints

如果原生成场已经满足一个齐次线性结构条件 `S[g_z]=0`，选择同样满足 `S[psi_j]=0` 的修正基，则：

\[
S[f_z^{\mathcal C}]=S[g_z]+S[\Psi]M^\dagger(y-C[g_z])=0.
\]

这允许研究“已有结构性质 + 测试时新观测”的组合，例如：对称场加点值/梯度，周期场加观测，流函数构造的无散度场加速度要求。目标与结构必须相容，且基在结构子空间内具有足够能力。

非齐次结构可先构造 lifting，再在齐次空间中生成/修正。全域非线性 PDE、拓扑、不翻折和完整碰撞保证不由本命题得到。

这部分可作为方法增量候选，但需要核查已有结构化生成研究，并比较普通修正与结构保持修正的实际差异。

## 3.6 Application Instantiations

主方法之后用表格说明应用变化，避免每个应用重新定义一套算法：

| 应用 | 生成场 | 测试时要求 | 最终使用的输出 |
|---|---|---|---|
| 3D 补全 | 空间隐式场 `f_z(x)` | 表面点、梯度 | 连续场及提取网格 |
| 动作生成 | 时间场 `q_z(t)` | waypoint、可靠导数 | 目标轨迹及可能的控制器执行 |
| 形变 | 位移场 `u_z(x)` | handle 位移、Jacobian 分量 | 形变场及变形网格 |
| 视频/world model（input-level） | 连续中间运动场 | 相容的 waypoint/可靠速度 | Wan 1.3B control 条件下的视频 |

硬保证只属于实际施加约束的场。网格提取、后处理、渲染与真实执行均单独评测；视频应用不自动提供像素硬约束。

**视频范围（2026-11-13）**：当前聚焦 input-level：生成 motion/neural-field representation → CNF 在该表示上施加有限维线性约束 → render control → 以 control 条件驱动预训练 Wan 1.3B。动作生成与其他应用继续按原计划推进；Cosmos 2B、scratch U-Net 的 latent 探索为 deferred/exploratory，不进入当前主贡献或待办。八组 scratch 矩阵虽有有限图像运动改善，仍未验证 CNF→视频闭环。

Input-level 审计支持 control 的因果作用及同 prior/noise、无额外 GT 速度时稀疏 anchor 二维位置代理改善；不支持 RGB 硬约束或全面质量优势。当前 GT 锚点与 clearance 在 8/8 scene 不相容，clearance 又破坏 anchor，因此本节表示层命题不能用于宣称最终整条轨迹严格可行。后续先统一任务定义并检查可行性，再做信息匹配投影比较、独立跟踪/pose 和 scene/Wan seed 扩展；详细证据与优先级见 [Applications 视频入口](applications_status_collaboration.md#3-application-a视频世界建模)。

## 论文定稿前必须补上的技术决策

这份结构说明“方法怎么定义”，不意味着新颖性已经成立。至少需要明确并验证一项实质增量：

1. 面向变化位置/阶数/数量的约束，如何稳定选择修正基并保持生成质量；
2. CNF 场修正如何反馈给生成先验，相对直接输出修正与适用强基线有何收益；
3. 如何组合已有结构与新约束，并在多个领域复用。

若最终只有通用 CNF 公式与多个 demo，应收紧贡献表述。不要用未实现的基设计、训练损失或采样策略填充 Method。方法—任务实现分工见 [接口契约](contracts.md) 和 [工作流](workstreams.md)。
