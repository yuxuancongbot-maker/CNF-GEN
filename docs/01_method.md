# 01 方法：Legendre 运动场、硬投影与视频世界模型接口

## 一句话 / 结论

- 本文把 17 帧、7 关节的轨迹写成 10 个列归一化 Legendre 基函数的线性组合，系数维度 \(D=10\times7=70\)。
- 测试时只给首帧 \(q_0\)、稀疏关节 waypoint，以及可选的第 8 帧速度；不要求 dense action。
- 运动先验是在系数空间做 flow matching 的 MLP；硬约束通过在系数空间解最小范数投影 \(A c=b\) 实现，而不是 soft loss。
- 采样用 50 步 Euler；`every-step` 在每个采样步后投影，`posthoc` 只在最后投影一次；两者最后都会再投影一次。
- 两条视频接口：input-level 是 \(q(t)\rightarrow\) PyBullet 语义 control video \(\rightarrow\) Wan-Control LoRA；latent-level 是 \(c\rightarrow q(t)\rightarrow\) 12.88M 3D U-Net。
- 这里的 “latent” 指 Legendre 运动系数向量 \(c\)，不是视频模型内部的 feature latent；3D U-Net 实际接收的是 \(q\) 及其 \(\sin/\cos\) 特征。
- 报告中的速度约束都在第 8 帧（\(t=0.5\)）。在这个帧上 `discrete_autograd` 与 `continuous` 的导数基一致到 \(4.77\times10^{-7}\)，所以速度结论不依赖归一化导数项；`finite_difference` 仍有 \(0.6487\) 的离散差。
- 安全间隙 `enforce_clearance` 是投影之后的 IK 后处理，会改动 anchor；因此端到端 anchor 误差要区分“投影残差”和“clearance 后的误差”。

---

## 1. 问题设定

### 1.1 输入、输出与约束

设一条轨迹有 \(T=17\) 帧、\(J=7\) 个关节，第 \(k\) 帧对应归一化时间

\[
t_k=\frac{k}{T-1}=\frac{k}{16},\qquad k=0,\dots,16.
\]

轨迹记为 \(q\in\mathbb{R}^{T\times J}\)，其中 \(q_{k,j}\) 是第 \(k\) 帧第 \(j\) 个关节角，单位为 rad。测试时的输入是：

| 输入 | 维度 / 取值 | 说明 |
|---|---|---|
| 首帧关节角 \(q_0\) | \(\mathbb{R}^{7}\) | 运动先验的条件；也是第 0 帧 anchor |
| 稀疏 waypoint | \(K_{\text{pos}}\times 7\) | 报告实验用 6 个或 7 个 anchor 帧的关节角 |
| 可选速度 | 7 维关节速度或 3 维末端速度 | 报告实验固定在第 8 帧 |
| dense action | 不使用 | 这是与标准 action-conditioned WM 的关键区别 |

输出是整条 \(q(t)\) 和对应视频。约束统一写成线性等式

\[
A c=b,
\]

其中 \(c\in\mathbb{R}^{D}\) 是运动系数，\(A\in\mathbb{R}^{K\times D}\) 是约束矩阵，\(b\in\mathbb{R}^{K}\) 是目标值。报告实验中的 \(A\)、\(b\) 由 waypoint 和速度构造；求解方式是最小范数硬投影，详见第 5 节。

### 1.2 符号表

| 符号 | 含义 | 形状 / 取值 |
|---|---|---|
| \(T\) | 帧数 | 17 |
| \(J\) | 关节数 | 7 |
| \(D\) | 基函数个数 | 10（`DEGREE=10`，即 \(P_0,\dots,P_9\)） |
| \(t_k\) | 第 \(k\) 帧归一化时间 | \(k/(T-1)=k/16\) |
| \(q\) | 关节轨迹 | \([T,J]=[17,7]\)，rad |
| \(q_0\) | 首帧关节角 | \([J]=[7]\) |
| \(c\) | 运动系数 | \([D]=[70]\)，布局 \(c_{dJ+j}\) |
| \(B\) | 采样 Legendre 基矩阵 | \([T,D]=[17,10]\) |
| \(B^{+}\) | \(B\) 的 Moore–Penrose 伪逆 | \([D,T]=[10,17]\) |
| \(A\) | 约束矩阵 | \([K,D]\)，\(K\) 见表 5 |
| \(b\) | 约束目标 | \([K]\) |
| \(V\) | 导数基矩阵 | \([T,D]=[17,10]\) |
| \(\mu,\sigma\) | 系数的标量均值 / 标准差 | 标量 |
| \(\varepsilon\) | 投影正则 | \(10^{-8}\) |
| \(K\) | 约束行数 | 42–56，见 5.1 |

### 1.3 两条接口

| 接口 | 运动层做什么 | 视频模型看到什么 | 约束在哪里是硬的 |
|---|---|---|---|
| input-level（control video） | \(c\rightarrow q(t)\)，再用 PyBullet 渲染语义分割 control | 首帧 RGB + control video；Wan 不看 \(c\) 也不看 \(q\) | 只在运动空间 \(q(t)\) 硬；像素空间只是评测目标 |
| latent-level（motion latent） | 直接在 \(c\) 上投影，再转成 \(q(t)\) | 首帧 + control + \(q,\sin q,\cos q\)；3D U-Net 不看 \(c\) | 在运动系数 \(c\) 上硬；像素空间由小 3D U-Net 渲染 |

“latent-level” 这个说法容易误解。这里的 latent 是 **Legendre 运动系数向量 \(c\in\mathbb{R}^{70}\)**，不是视频模型内部的 VAE latent 或 feature map。`video_wm.py` 的模块注释也明确写了：motion latent 是 Legendre coefficient vector \(c\)，硬投影先作用在 \(c\) 上，再转成 \(q\) 喂给视频模型。

---

## 2. Legendre 表示

### 2.1 三项递推

`basis.py:legendre_values` 先做变量替换

\[
x=2t-1\in[-1,1],
\]

再用标准三项递推生成 \(P_0,\dots,P_{\text{degree}-1}\)：

\[
P_0(x)=1,\qquad P_1(x)=x,
\]

\[
nP_n(x)=(2n-1)xP_{n-1}(x)-(n-1)P_{n-2}(x),\qquad n=2,\dots,9.
\]

`DEGREE=10` 返回 10 个基函数 \(P_0,\dots,P_9\)，最高多项式次数是 9。代码中 `degree=1` 单独处理，返回全 1；`degree>=2` 时先放 \(P_0,P_1\)，再循环到 `range(2, degree)`。

### 2.2 采样归一化与 Gram 矩阵

在 17 个均匀采样点 \(t_k=k/16\) 上，先得到原始矩阵

\[
R_{k,d}=P_d(t_k),\qquad R\in\mathbb{R}^{17\times10}.
\]

每一列用 **采样 L2 范数** 归一化：

\[
n_d=\left(\sum_{k=0}^{16}P_d(t_k)^2\right)^{1/2},
\qquad
B_{k,d}=\frac{P_d(t_k)}{\max(n_d,10^{-6})}.
\]

`basis.py:legendre_basis` 的实现就是 `basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-6)`。注意这里的范数是 17 个采样点上的离散范数，不是连续 L2 范数 \(\sqrt{\int_0^1 P_d(t)^2dt}\)。因此列范数严格为 1（float32 下误差约 \(10^{-7}\)），但不同列之间并不严格正交。

按 release 代码在 float32 下计算：

| 量 | 数值 |
|---|---:|
| \(\max_{i\ne j}|(B^\top B)_{ij}|\) | \(0.6786885857582092\) |
| \(\max_{i,j}|(B^\top B-I)_{ij}|\) | \(0.6786885857582092\) |
| \(\operatorname{cond}(B)\) | \(3.476057529449463\) |
| \(\|B B^{+}B-B\|_{\max}\) | \(5.960464477539062\times10^{-7}\) |
| 随机系数 roundtrip 最大误差 | \(1.6689300537109375\times10^{-6}\) |

结论是：**采样 Gram 矩阵在 degree=10、17 帧下不是单位阵**，但 \(B\) 仍然列满秩（rank=10），\(B^{+}\) 能在基张成空间上精确重建系数。对于任意 \(c\)，\(B^{+}(Bc)=c\)；对于不在基张成空间内的 \(q\)，\(B^{+}q\) 是最小二乘系数，重建 \(B B^{+}q\) 会有残差。`tests/test_basis.py` 用 `atol=1e-5` 检查列范数和 roundtrip。

### 2.3 系数布局与转换

系数采用 **关节维最快** 的布局：

\[
c[dJ+j]\quad\Longleftrightarrow\quad c_{d,j},\qquad d=0,\dots,9,\ j=0,\dots,6.
\]

因此 \(c\in\mathbb{R}^{70}\) 可以 reshape 成 \([10,7]\)。正向和反向转换是：

\[
q(t)=B(t)c,\qquad q\in\mathbb{R}^{T\times J},
\]

\[
c=B^{+}q,\qquad c\in\mathbb{R}^{10\times J}.
\]

代码对应关系：

| 操作 | 代码 | 形状 |
|---|---|---|
| \(B,B^{+}\) | `basis.py:make_basis(17,10)` | \(B[17,10]\), \(B^{+}[10,17]\) |
| \(c=B^{+}q\) | `coefficients_from_trajectory`：`einsum("dt,btj->bdj")` | \([B,10,7]\) |
| \(q=Bc\) | `trajectory_from_coefficients`：`einsum("td,bdj->btj")` | \([B,17,7]\) |

`make_basis` 的伪逆是 `torch.linalg.pinv(basis)`；在 \(B\) 列满秩时等价于 \((B^\top B)^{-1}B^\top\)。`trajectory_from_coefficients` 也接受扁平化的 \([B,70]\)，内部会 reshape 成 \([B,10,7]\)。

### 2.4 标量均值 / 标准差归一化

运动先验不是逐维归一化，而是对所有系数取一个 **标量** 均值和一个 **标量** 标准差：

\[
\mu=\operatorname{mean}(c),\qquad
\sigma=\operatorname{std}(c)\ \text{（clamp 到}\ge10^{-6}\text{）},
\]

\[
c_{\text{norm}}=\frac{c-\mu}{\sigma}.
\]

`fit_motion_prior` 里是 `coefficients.mean()` 和 `coefficients.std()`，没有 `dim` 参数，所以是标量。采样时先在归一化空间积分，投影时再回到物理空间：

\[
c_{\text{phys}}=\sigma c_{\text{norm}}+\mu,
\]

投影后再做

\[
c_{\text{norm}}=\frac{c_{\text{phys}}-\mu}{\sigma}.
\]

因为 \(\sigma\) 是标量，这一步是逐元素的；但投影必须在物理空间做，否则约束目标 \(b\) 的 rad 单位对不上。`motion_field.py:_apply_projection` 和 `constraints.py:sample_sparse` 都是这个顺序。

---

## 3. 导数基与速度语义

### 3.1 三种模式

`basis.py:make_derivative_basis(frames=17, degree=10, mode=...)` 返回

\[
(B,\ \dot B,\ \ddot B),
\]

三者形状都是 \([17,10]\)。导数是对归一化时间 \(t\in[0,1]\) 求的；要和逐帧差分比较，需要乘以 \(T-1=16\)。

| 模式 | 代码做法 | 数学含义 | 用途 |
|---|---|---|---|
| `discrete_autograd` | 对采样归一化后的 `basis` 做 autograd；`source=basis` | 对采样归一化矩阵的求和做 autograd，包含采样范数对 \(t\) 的依赖 | **默认；报告中的速度行使用此模式** |
| `continuous` | 对原始 \(P_d(t)\) 做 autograd，再除以固定的采样范数 `norm.detach()` | \(P'_d(t_m)/n_d\)，即“连续多项式 / 固定采样范数”的点式导数 | 数学上更干净的对照 / ablation |
| `finite_difference` | `torch.gradient(basis, spacing=1/16, dim=0)` | 采样基行的二阶中心差分（边界一阶） | 离散差分基线 / ablation |

`make_derivative_basis` 的默认参数是 `mode="discrete_autograd"`。`scripts/generate_controls.py` 和 `scripts/eval_action_wm.py` 都调用 `make_derivative_basis()`，所以 **dense 关节速度约束和 sparse 任务空间速度约束用的都是 `discrete_autograd` 的第 8 行**。另外两种模式只用于对照，不用于报告结果。

### 3.2 第 8 帧的关键事实：三种模式的关系

报告的两种情况都把速度约束放在第 8 帧：

- dense：`velocity_frames=(VELOCITY_FRAME,)`，`VELOCITY_FRAME=8`；
- sparse：`velocity_basis[VELOCITY_FRAME]`，同样是第 8 帧。

第 8 帧对应 \(t_8=8/16=0.5\)，正好是 17 帧的中点。按 release 代码在 float32 下计算：

| 比较 | 第 8 帧基行最大绝对差 |
|---|---:|
| `discrete_autograd` vs `continuous` | \(4.76837158203125\times10^{-7}\) |
| `discrete_autograd` vs `finite_difference` | \(0.6487436294555664\) |
| `continuous` vs `finite_difference` | \(0.6487441062927246\) |

原因是中点处 Legendre 多项式的奇偶性：

- 对偶数 \(d\)，\(P_d(x)\) 是偶函数，\(P'_d(0)=0\)，所以 \(P'_d(0.5)=0\)；
- 对奇数 \(d\)，\(P_d(x)\) 是奇函数，\(P_d(0)=0\)，所以 \(P_d(0.5)=0\)。

`discrete_autograd` 与 `continuous` 的差来自采样范数对 \(t\) 的耦合项，该项含有 \(P_d(0.5)P'_d(0.5)\)，在中点必为零。因此 **在第 8 帧，`discrete_autograd` 和 `continuous` 是同一个算子（差 \(4.77\times10^{-7}\)，float32 舍入级别）；`finite_difference` 仍有 \(0.6487\) 的离散差分误差**。

这不是只靠推导：release 里附带了 continuous 模式的稀疏复跑 `release/runs/controls_sparse_continuous/metrics.json`。把它和 `release/runs/controls_sparse/metrics.json` 逐样本比较，以下指标的最大绝对差为：

| 指标 | 最大绝对差 |
|---|---:|
| `anchor_error` | \(1.19\times10^{-7}\) |
| `joint_rmse` | \(4.10\times10^{-8}\) |
| `ee_rmse` | \(7.40\times10^{-8}\) |
| `jerk` | \(3.28\times10^{-7}\) |
| `ee_velocity_error` | \(1.22\times10^{-6}\) |
| `joint_velocity_at_8_error` | \(3.28\times10^{-7}\) |

最大差不超过 \(1.3\times10^{-6}\)。因此可以明确说：**报告的 dense 关节速度结论和 sparse 任务空间速度结论不受 `discrete_autograd` / `continuous` 归一化导数项的影响；它们在第 8 帧等价。** 默认保留 `discrete_autograd` 是为了和原始实验逐位对齐，而不是因为它在第 8 帧有实质不同的结果。

在非中点帧，两种模式会有差异。例如：

| 帧 | `discrete_autograd` vs `continuous` 最大差 | `discrete_autograd` vs `finite_difference` 最大差 |
|---:|---:|---:|
| 0 | \(26.0668\) | \(42.8647\) |
| 4 | \(0.1925\) | \(0.8403\) |
| 8 | \(4.77\times10^{-7}\) | \(0.6487\) |
| 12 | \(0.1925\) | \(0.8403\) |
| 16 | \(26.0668\) | \(42.8647\) |

这张表只用于说明模式差异，不是实验指标。如果以后把速度帧移到非中点（例如第 6 帧或第 10 帧），必须明确写出用的是哪种导数基，并做消融；当前报告不需要这个消融。

`discrete_autograd` 的精确形式可以写成：令

\[
S_d=\sum_{k=0}^{16}P_d(t_k),\qquad
n_d=\left(\sum_{k=0}^{16}P_d(t_k)^2\right)^{1/2},
\]

则代码计算的第 \(m\) 行是

\[
V^{(\text{disc})}_{m,d}
=\frac{\partial}{\partial t_m}\sum_{k=0}^{16}\frac{P_d(t_k)}{n_d}
=\frac{P'_d(t_m)}{n_d}
-\frac{S_d\,P_d(t_m)P'_d(t_m)}{n_d^3}.
\]

而 `continuous` 是

\[
V^{(\text{cont})}_{m,d}=\frac{P'_d(t_m)}{n_d}.
\]

两者只差第二项；在中点该项为零。这个形式也解释了为什么 `discrete_autograd` 不是逐点的 \(P'_d(t_m)\)：它是对“整列求和”求导，范数依赖所有采样点。

### 3.3 速度约束的“算子”与评测的“差分”不是同一件事

这点对读结果很重要：

- **dense 关节速度**：目标 `gt_joint_velocity = einsum("td,bdj->btj", velocity_basis, coefficients_gt)`，约束行也用同一个 `velocity_basis[8]`。所以“基函数速度残差”可以精确到 0（报告里 16/16 精确满足）。
- **评测里的 `velocity_error`**：`np.diff(q, axis=0)[8] - np.diff(q_gt, axis=0)[8]`，是逐帧有限差分，不是 \(V_8\) 算子。release 快照里 dense `waypoint` 的 `velocity_error=0.19803`，`velocity` 是 `0.21847`，`posthoc` 是 `0.19718`；velocity 在这个有限差分指标上反而差 `+0.02044`（0/16 更好）。这不是矛盾：硬约束保证的是 \(V_8c=V_8c_{\text{gt}}\)，不是 \(\Delta q_8=\Delta q_{\text{gt},8}\)。
- **sparse 任务空间速度**：目标 \(v^\star\) 是 FK 中心差分（见 6.2），约束行用的是 \(V_8\)。两者不是同一算子，加上 3 次再线性化和 clearance 后处理，最终 `ee_velocity_error` 是 `0.11630`，不是 0；但它从 `0.53302` 降下来（16/16 更好）。详见 6.6。

因此，文档里说“速度硬约束”时，必须区分：约束是在 \(V_8\) 基函数意义上精确，还是在逐帧有限差分 / 任务空间中心差分意义上精确。报告只对前者给出“精确满足”，对后者给出误差数值。

---

## 4. 运动先验

### 4.1 网络结构

`motion_field.py:PandaFlowModel` 是一个 3 个 MLP 块的 velocity field：

\[
v_\theta(c_t,t,q_0)\in\mathbb{R}^{70}.
\]

| 块 | 层 | 输入 → 输出 |
|---|---|---|
| state | `Linear(21,256) → SiLU → Linear(256,256) → SiLU` | \([q_0,\sin q_0,\cos q_0]\in\mathbb{R}^{21}\rightarrow\mathbb{R}^{256}\) |
| time | `Linear(32,256) → SiLU → Linear(256,256) → SiLU` | 32 维正弦时间嵌入 \(\rightarrow\mathbb{R}^{256}\) |
| net | `Linear(582,512) → SiLU → Linear(512,512) → SiLU → Linear(512,70)` | \([c_t;\text{state};\text{time}]\in\mathbb{R}^{70+256+256=582}\rightarrow\mathbb{R}^{70}\) |

参数量按 release 代码为 **742,726**（hidden=256, joints=7, degree=10）。注意 state 块的输入是 21 维：7 个关节的 \(q_0\)、\(\sin q_0\)、\(\cos q_0\) 拼接。

### 4.2 正弦时间嵌入

`sinusoidal_embedding(time, 32)` 的实现是：

\[
\text{half}=16,\qquad
\omega_i=\exp\left(-\log(10000)\cdot\frac{i}{\max(\text{half}-1,1)}\right),\quad i=0,\dots,15,
\]

\[
\text{emb}(t)=\left[\sin(t\omega_0),\dots,\sin(t\omega_{15}),\cos(t\omega_0),\dots,\cos(t\omega_{15})\right]\in\mathbb{R}^{32}.
\]

`time` 是 \([B]\) 的标量，`frequencies` 是 16 维，`angles=time[:,None]*frequencies[None,:]`，最后 `cat([sin,cos],dim=-1)`。

### 4.3 Flow-matching 目标

训练数据是归一化系数 \(c_{\text{norm}}\)。每个训练步：

\[
\epsilon\sim\mathcal{N}(0,0.5^2),\qquad t\sim\mathcal{U}[0,1],
\]

\[
c_t=(1-t)\epsilon+t\,c_{\text{norm}},
\]

\[
\mathcal{L}(\theta)=\mathbb{E}_{c,\epsilon,t}\left\|v_\theta(c_t,t,q_0)-(c_{\text{norm}}-\epsilon)\right\|_2^2.
\]

代码对应 `motion_field.py:fit_motion_prior`：

```python
noise = torch.randn(target.shape, generator=generator) * INIT_SCALE
time = torch.rand(batch, generator=generator)
interpolated = (1 - time[:, None]) * noise + time[:, None] * target
velocity = target - noise
prediction = model(interpolated, time, q0[index])
loss = F.mse_loss(prediction, velocity)
```

其中 `INIT_SCALE=0.5`。

训练超参数：

| 项目 | 值 |
|---|---|
| steps | 6000 |
| batch | 64 |
| optimizer | AdamW |
| lr | \(10^{-3}\) |
| weight decay | \(10^{-4}\) |
| schedule | CosineAnnealingLR, `T_max=6000`, `eta_min=lr*0.05=5e-5` |
| grad clip | 1.0 |
| noise scale | 0.5 |
| 归一化 | 标量 mean/std |
| seed | 0 |
| 训练设备 / 时间 | CPU，报告约 1 分钟 |

`fit_motion_prior` 用 `torch.manual_seed(seed)`，数据采样 generator 用 `seed+1`。`q0 = q[:,0]`，即首帧关节角；hard prior 不看未来 anchor，只有 soft baseline 看 anchor。

### 4.4 Euler 采样

`sample_with_projection` 和 `sample_unconstrained` 的采样过程相同，只是后者不投影：

\[
c_0\sim\mathcal{N}(0,0.5^2),
\]

\[
t_n=\frac{n}{50},\qquad
c_{n+1}=c_n+\frac{1}{50}v_\theta(c_n,t_n,q_0),\qquad n=0,\dots,49.
\]

注意时间网格是 \(0,0.02,\dots,0.98\)，**不会到 1.0**；这是 50 步 Euler 的固定网格。采样输出是物理系数：

\[
c_{\text{phys}}=\sigma c_{50}+\mu.
\]

如果 `project_every_step=True`，每个 Euler 步后做一次物理空间投影；无论是否为 every-step，最后都会再做一次最终投影。`posthoc` 就是 `project_every_step=False`，只在最后投影一次。

### 4.5 Soft baseline

`soft_field.py:PandaSoftFlowModel` 是 hard prior 的对照。它多了一个 anchor 条件分支：

| 块 | 输入 → 输出 |
|---|---|
| state | \(q_0,\sin q_0,\cos q_0\in\mathbb{R}^{21}\rightarrow\mathbb{R}^{256}\) |
| anchors | 6 个 anchor × 7 关节 = 42 维 \(\rightarrow\mathbb{R}^{256}\) |
| time | 32 维时间嵌入 \(\rightarrow\mathbb{R}^{256}\) |
| net | \([c_t;\text{state};\text{anchors};\text{time}]\in\mathbb{R}^{70+256\times3=838}\rightarrow\mathbb{R}^{70}\) |

训练损失是

\[
\mathcal{L}=\mathcal{L}_{\text{flow}}+20\cdot\mathcal{L}_{\text{anchor}},
\]

其中 anchor 损失是把去噪后的系数转成物理系数、在 anchor 帧上算 MSE。soft baseline 不执行任何硬投影，所以它的 anchor 残差按构造非零。soft weight 默认 20，6000 步，batch 64，lr \(10^{-3}\)，seed 1。

---

## 5. 硬投影

### 5.1 约束矩阵

`motion_field.py:constraint_matrix` 按“帧优先、关节次优先”的顺序构造 \(A\)：

- 位置行：对每个 anchor 帧 \(k\)、每个关节 \(j\)，

\[
A_{\text{pos}}[k,j;d,j']=B_{k,d}\,\delta_{j,j'}.
\]

代码写法是 `row[joint::joints] = basis[frame]`，即行内第 \(dJ+j\) 个元素是 \(B_{k,d}\)。

- 关节速度行：对每个速度帧 \(k\)、每个关节 \(j\)，

\[
A_{\text{vel}}[k,j;d,j']=V_{k,d}\,\delta_{j,j'}.
\]

- 任务空间速度行：对每个末端轴 \(a\in\{x,y,z\}\)、每个关节 \(j\)，

\[
A_{\text{task}}[a;d,j]=J_{a,j}(q_8)\,V_{8,d}.
\]

其中 \(J(q)=\partial p/\partial q\in\mathbb{R}^{3\times7}\) 是末端位置对关节的平移 Jacobian，\(V_8\) 是导数基第 8 行。代码写法是 `row[joint::joints] += J[axis,joint] * velocity_basis[8]`。

目标 \(b\) 的顺序必须和 \(A\) 一致：`flatten_targets` 先把 `position_targets` reshape 成 \([B,K_{\text{pos}}\cdot J]\)，再拼上速度目标。位置目标在报告实验中是 GT 关节角 \(q_{\text{gt}}(t_k)\)；一般场景下也可以是 Cartesian waypoint 经 IK 得到的 \(q_k^\star\)。

不同配置的行数：

| 配置 | 位置行 | 速度行 | \(K\) | 备注 |
|---|---:|---:|---:|---|
| dense 位置 | \(7\times7=49\) | 0 | 49 | `ANCHOR_FRAMES_DENSE` |
| dense + 7 维关节速度 | 49 | 7 | 56 | dense `velocity` 方法；唯一 \(K>49\) 的配置 |
| sparse 位置 | \(6\times7=42\) | 0 | 42 | `ANCHOR_FRAMES_SPARSE` |
| sparse + 3 维任务空间速度 | 42 | 3 | 45 | sparse `velocity` 方法 |
| sparse + 7 维关节速度（未作主实验） | 42 | 7 | 49 | 仅用于说明 |

因此主对比的 \(K\) 最大为 49（dense 位置 49、sparse 任务空间 45）；dense + 关节速度的 56 是唯一例外，写复杂度时不能把它忽略。

### 5.2 最小范数修正

`motion_field.py:project_coefficients` 实现：

\[
c'=c-A^\top\left(AA^\top+\varepsilon I_K\right)^{-1}(Ac-b),
\qquad \varepsilon=10^{-8}.
\]

这是一个 \(K\times K\) 线性求解：

```python
residual = coefficients @ constraint.T - targets
normal = constraint @ constraint.T + eps * torch.eye(K)
dual = torch.linalg.solve(normal, residual.T).T
return coefficients - dual @ constraint
```

它的含义是：在所有满足 \(A(c+\delta)=b\) 的修正 \(\delta\) 中，取 \(\delta\) 在 \(A\) 的行空间内、且 \(\|\delta\|_2\) 最小的那个。\(\varepsilon\) 只用于数值正则，防止 \(AA^\top\) 奇异或病态。

数值验收：

- `tests/test_projection.py` 对位置投影和速度投影都断言 \(\max|Ac'-b|<10^{-5}\)；
- 同一测试还断言投影幂等：`project(project(c))` 与 `project(c)` 的差 `<1e-6`；
- `project_coefficients` 的 docstring 说返回系数满足 \(Ac'=b\) 到线性求解容差。

固定 \(A\) 的条件数（按 release 代码在 17×10 基上计算）：

| 配置 | \(K\) | rank | \(\operatorname{cond}(AA^\top)\) |
|---|---:|---:|---:|
| dense 位置 | 49 | 49 | \(60.5342\) |
| dense + 7 维关节速度 | 56 | 56 | \(2863.1997\) |
| sparse 位置 | 42 | 42 | \(43.6040\) |
| sparse + 7 维关节速度 | 49 | 49 | \(345.1896\) |

sparse 任务空间速度的 \(A\) 含 \(J(q_8)\)，条件数随样本的 \(q_8\) 变化，不能给一个固定常数；它的 \(K=45\)、rank=45（在 Jacobian 行满秩的样本上）。这也是为什么 3 次再线性化后仍要用数值残差来报告，而不是假设机器精度。

### 5.3 Every-step 与 posthoc

`sample_with_projection` 的控制流：

```python
for step in range(steps):
    time = step / steps
    v = model(c, time, q0)
    c = c + v / steps
    if project_every_step:
        c = _apply_projection(c, mean, std, A, b, eps)
c = _apply_projection(c, mean, std, A, b, eps)  # 最终一定投影
```

- `every-step`：`project_every_step=True`，每个 Euler 步后投影，最后再投影一次。
- `posthoc`：`project_every_step=False`，只在最后投影一次。

`sample_sparse` 的逻辑相同，只是把 `_apply_projection` 换成 `project_sparse`（含任务空间速度的 3 次再线性化）。`posthoc` 不是“完全不投影”，而是“只在最后投影”；两种方法最终都满足（或近似满足）同一组约束，差别在于采样路径是否被每步拉回可行集。

### 5.4 归一化与投影顺序

`_apply_projection`：

```python
physical = coefficients * std + mean
physical = project_coefficients(physical, A, b, eps)
return (physical - mean) / std
```

`sample_sparse`：

```python
physical = coefficients * std + mean
physical = project_sparse(robot, physical, ...)
coefficients = (physical - mean) / std
```

因为 \(\mu,\sigma\) 是标量，物理空间投影和归一化空间投影只差一个缩放；但约束目标 \(b\) 是 rad 或 m，必须在物理空间做。不要在归一化空间直接解 \(A c_{\text{norm}}=b\)，否则 \(b\) 的单位和尺度都不对。

### 5.5 Batch 限制

`project_sparse` 明确检查 `coefficients.shape[0] != 1` 并报错，所以 sparse + 任务空间速度当前只支持 batch size 1。`sample_sparse` 也是按单样本循环使用。`sample_with_projection` 支持 batch，因为它的约束矩阵与样本无关；dense 实验可以 batch 采样。这个差异在写复杂度时要说明：sparse 的 16 个样本是串行投影的。

---

## 6. 稀疏 waypoint + 任务空间速度

### 6.1 Anchor 与速度帧

报告配置：

\[
\mathcal{K}_{\text{anchor}}=[0,4,8,12,13,16],
\qquad
k_{\text{vel}}=8.
\]

`ANCHOR_FRAMES_SPARSE=(0,4,8,12,13,16)`、`VELOCITY_FRAME=8` 都在 `config.py`。dense 配置则是 `ANCHOR_FRAMES_DENSE=(0,3,4,6,10,13,16)`，速度帧同样是 8。

### 6.2 任务空间速度目标

`constraints.py:task_space_velocity`：

\[
p_k=\operatorname{FK}(q_k)\in\mathbb{R}^3,
\]

\[
v_8^\star=\frac{p_9-p_7}{2}\cdot(T-1)
=\frac{p_9-p_7}{2}\cdot16.
\]

单位是“米 / 归一化时间”。乘以 16 是把“每帧中心差分”换算成“每单位归一化时间”的速度；因为 \(t\) 的步长是 \(1/16\)。

目标由 GT 轨迹 \(q_{\text{gt}}\) 计算：

```python
task_velocity = task_space_velocity(robot, q_gt)
```

它是 3 维向量，作为 `project_sparse(..., task_velocity=...)` 的目标。

### 6.3 Jacobian

`scene.py:ee_jacobian` 用中心有限差分：

\[
J_{a,j}(q)=\frac{p_a(q+\epsilon e_j)-p_a(q-\epsilon e_j)}{2\epsilon},
\qquad \epsilon=10^{-4}.
\]

- \(J\in\mathbb{R}^{3\times7}\)；
- 每个关节需要 2 次 FK，7 个关节共 14 次 FK；
- `ee_position` 取 link 11（gripper base）的位置；
- 这是平移 Jacobian，不包含姿态。

代码：

```python
for joint in range(7):
    plus = q.copy(); plus[joint] += eps
    minus = q.copy(); minus[joint] -= eps
    jacobian[:, joint] = (ee_position(robot, plus) - ee_position(robot, minus)) / (2 * eps)
```

### 6.4 行构造

对每个末端轴 \(a\in\{x,y,z\}\)，构造一行：

\[
A_{\text{task}}[a;d,j]=J_{a,j}(q_8)\,V_{8,d}.
\]

代码：

```python
for axis in range(3):
    row = torch.zeros(dim)
    for joint in range(7):
        row[joint::joints] += float(jacobian[axis, joint]) * velocity_basis[8]
    rows.append(row)
```

把 42 行 anchor 和 3 行任务空间速度拼起来：

\[
A=\begin{bmatrix}A_{\text{anchor}}\\A_{\text{task}}\end{bmatrix}\in\mathbb{R}^{45\times70},
\qquad
b=\begin{bmatrix}q_{\text{anchor}}\\v_8^\star\end{bmatrix}\in\mathbb{R}^{45}.
\]

### 6.5 三次再线性化

`project_sparse` 的默认 `iterations=3`。算法是：

```text
c^0 = c
for i = 1..3:
    q8 = B(8) c^{i-1}
    J = ee_jacobian(robot, q8)
    A_task = rows from J and V8
    A = [A_anchor; A_task]
    b = [q_anchor; v8_star]
    c^i = project_rows(c^{i-1}, A, b)
return c^3
```

为什么需要再线性化：任务空间速度约束 \(J(q_8)V_8c=v_8^\star\) 对 \(c\) 不是线性的，因为 \(J\) 依赖 \(q_8=B(8)c\)。每次投影改变 \(c\)，也就改变 \(q_8\) 和 \(J\)，所以要在新的 \(q_8\) 上重建速度行。anchor 行是线性的，每轮都一起投影；速度行每轮重建。

3 是代码默认的固定次数，没有收敛判据；报告没有给出 1/2/3/5 次迭代的消融（未做消融）。因此“3 次”是工程默认值，不应说成已经调到最优。

### 6.6 再线性化后的残差与算子不匹配

`project_sparse` 最后返回的 \(c\) 满足的是 **最后一轮线性化系统** \(A c=b\) 到求解容差；它不保证原始非线性任务空间速度 \(J(q_8(c))V_8c=v_8^\star\) 精确成立，也不保证逐帧中心差分 \(\frac{p_9-p_7}{2}\cdot16=v_8^\star\) 精确成立。原因有三层：

1. **再线性化**：只做 3 次固定迭代，没有收敛到 \(J(q_8(c))\) 的不动点；
2. **算子不匹配**：目标 \(v_8^\star\) 是 FK 中心差分，约束行用的是 \(V_8\)（`discrete_autograd` 的第 8 行）；两者在第 8 帧虽然与 `continuous` 一致到 \(4.77\times10^{-7}\)，但 \(V_8\) 本身不是“逐帧中心差分”算子；
3. **clearance 后处理**：采样后又调用 `enforce_clearance`，会改动 \(q\)，包括 anchor 帧。

release 快照的结果与这个解释一致：`sparse_velocity` 的 `ee_velocity_error=0.11630`，不是 0；但它相对 `sparse_waypoint` 的 `0.53302` 降低了 `0.41671`（16/16 更好）。所以报告写“速度跟随改善”是准确的，但不能写“任务空间速度精确满足”。

### 6.7 报告实验的 anchor 是 oracle

在报告实验里，anchor 目标直接取 GT 轨迹的关节角：

```python
position_targets = qt[:, anchor_frames]
# 或
anchors = qt[:, list(ANCHOR_FRAMES_SPARSE)]
```

这不是从测试时的 Cartesian waypoint 现场做 IK，而是用 GT 关节角当 oracle anchor，目的是隔离投影误差、避免 IK 误差混入。一般方法里如果上层给的是 Cartesian waypoint，需要先做 IK 得到 \(q_k^\star\)，再走同一个 \(A c=b\) 形式；这一步在 release 的 6-DOF 实验中是由数据生成器 `scene.py:solve_ik` 离线完成的，不是测试时在线 IK。

---

## 7. 安全间隙

### 7.1 `enforce_clearance`

`scene.py:enforce_clearance(robot,q,yaw)` 是投影之后的逐帧 IK 后处理。对每一帧 \(k=0,\dots,16\)：

\[
z_{\text{safe}}(k)=
\begin{cases}
0.3655\ \text{m}, & 4\le k\le 13,\\
0.332\ \text{m}, & \text{otherwise}.
\end{cases}
\]

如果当前末端高度 \(p_z(q_k)<z_{\text{safe}}(k)\)，就把目标改成

\[
p_k^{\text{new}}=[p_x(q_k),\,p_y(q_k),\,z_{\text{safe}}(k)],
\]

姿态用欧拉角

\[
[\pi,0,\text{yaw}]
\]

转成四元数；调用 PyBullet IK：

| 参数 | 值 |
|---|---|
| end effector link | 11 |
| orientation | `getQuaternionFromEuler([pi, 0, yaw])` |
| rest poses | `list(q[frame]) + [FINGER_OPEN, FINGER_OPEN]` |
| joint damping | `[0.05] * 9` |
| max iterations | 500 |
| residual threshold | \(10^{-6}\) |

其中 `FINGER_OPEN=0.040`，`GRASP_FRAMES=(4,13)` 是闭区间。`enforce_clearance` 只检查 link 11 的高度，不检查完整夹爪几何、自碰撞或桌面以外的障碍物。

### 7.2 验证

`constraints.py:validate_trajectory` 检查：

| 检查 | 阈值 |
|---|---|
| 关节限位 | `check_limits`，容差 \(10^{-4}\) rad |
| 最大关节步长 | \(\max_t\|q_{t+1}-q_t\|_\infty<0.8\) rad |
| 手部 clearance | `hand_clearance > -0.001` m |

`hand_clearance` 的定义是

\[
\min_{k=4,\dots,13} z_{\text{hand}}(q_k)
-\left(0.30+0.025+0.040\right)
=
\min_{k=4,\dots,13} z_{\text{hand}}(q_k)-0.365.
\]

`table_clearance(cube_positions)` 则是

\[
\min_t\left(z_{\text{cube}}(t)-0.025-0.30\right),
\]

即方块底部离桌面的最小高度。`validate_trajectory` 不检查 anchor 是否仍然精确，也不检查 cube 的最终目标误差；这些由评测脚本单独算。

### 7.3 后处理会扰动 anchor

执行顺序（`scripts/generate_controls.py`）：

```text
sample_with_projection / sample_sparse
    -> q = coefficients_to_trajectory(c)
    -> q = enforce_clearance(robot, q, yaw)
    -> validate_trajectory(robot, q)
    -> render_trajectory(...)
    -> trajectory_metrics(...)
```

`enforce_clearance` 在投影之后，且不会重新投影。因此：

- 投影本身可以把 anchor 做到 \(<10^{-5}\)（单元测试口径）；
- 但 release 快照的端到端 `anchor_error` 是 **0.00131 rad**（dense 和 sparse 的所有方法都是这个值）；
- 这个 0.00131 包含 clearance IK 对 anchor 帧的扰动，不能直接当作投影残差。原始 `report_pickplace.md` 里的 `<1e-6` 应理解为投影/未扰动口径；两者不矛盾，但报告时必须写清楚口径。

如果要严格保证“投影后 anchor 不变”，应在 clearance 之后再做一次 anchor-only 投影，或把 clearance 写成一个带约束的 IK 问题。当前 release 没有这么做，这是已知限制（见 `07_limitations.md`）。

---

## 8. 视频接口

### 8.1 Input-level：语义 control video → Wan-Control LoRA

运动层输出 \(q(t)\in\mathbb{R}^{17\times7}\)。`scene.py:render_trajectory` 用 PyBullet 渲染 RGB 和 3 通道语义 control：

| 语义 | RGB |
|---|---|
| 桌面 | \((80,80,80)\) |
| 橙色方块 | \((255,120,20)\) |
| 机械臂 | \((30,80,220)\) |
| 夹爪 link 8–11 | \((30,200,120)\) |

相机与渲染参数：

| 参数 | 值 |
|---|---|
| camera target | `[0.42, 0.02, 0.33]` |
| distance | 0.92 |
| yaw / pitch | 30 / −24 |
| FOV | 50 |
| near / far | 0.05 / 5 |
| resolution | 256×256 |
| FPS | 8 |
| renderer | `ER_TINY_RENDERER` |
| segmentation flags | `ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX` |

渲染时 `object_id=seg & 0xFFFFFF`，`link_index=seg >> 24`；先按 object 上色，再用 link 8–11 覆盖夹爪颜色。抓取帧 \(4\le k\le13\) 手指闭合到 `FINGER_CLOSED=0.025`，其余帧张开到 `FINGER_OPEN=0.040`；抓取期间方块刚性跟随 link 11，偏移 `GRASP_OFFSET=0.040` m。

视频模型是 Wan2.1-Fun-V1.1-1.3B-Control + Control LoRA：

| 项目 | 值 |
|---|---|
| 分辨率 / 帧数 | 256×256 / 17 |
| LoRA rank / alpha | 64 / 32 |
| LoRA 模块 | `q,k,v,ffn.0,ffn.2` |
| 训练步数 | 1500 |
| batch | 1 |
| 精度 | bf16 |
| gradient checkpointing | 是 |
| 硬件 / 时间 | 单卡 A100 40GB，约 25 分钟 |
| 输出 | 17 帧 RGB 视频 |

关键点：Wan 只看到首帧和 control video，**不看 \(c\) 也不看 \(q\)**。所以运动空间的硬约束不会自动变成像素空间的硬约束；像素跟踪只是评测目标。base 模型对 segmentation control 是零样本失败（control EPE `24.85 px`），LoRA 把它降到 `6.65 px`，但这不是“严格冻结”，而是“主体冻结 + 轻量 control adapter”。

### 8.2 Latent-level：运动系数 \(c\) → 12.88M 3D U-Net

这条接口的“latent”是 \(c\)，不是视频模型的内部 latent。流程是：

```text
c [B,70] --硬投影--> c' --B--> q [B,17,7] --sin/cos--> 3D U-Net
```

3D U-Net 的输入是：

- 首帧 RGB，重复到 17 帧：3 通道；
- 完整 control video：3 通道；
- 首帧 control，重复到 17 帧：3 通道；
- \(q,\sin q,\cos q\)：\(7\times3=21\) 通道；
- 总输入通道 \(3+3+3+21=30\)。

`VideoWorldModel6D(hidden=32)` 的逐层形状（输入 17×128×128，batch 维省略）：

| stage | 模块 | 输出 \([C,T,H,W]\) |
|---|---|---|
| input | first + control + control0 + qfeat | \(30\times17\times128\times128\) |
| enc0 | `conv_block(30,32)` | \(32\times17\times128\times128\) |
| down1 | `Conv3d(32,64,3,stride=(1,2,2),padding=1)` | \(64\times17\times64\times64\) |
| enc1 | `conv_block(64,64)` | \(64\times17\times64\times64\) |
| down2 | `Conv3d(64,128,3,stride=(1,2,2),padding=1)` | \(128\times17\times32\times32\) |
| enc2 | `conv_block(128,128)` | \(128\times17\times32\times32\) |
| down3 | `Conv3d(128,256,3,stride=(2,2,2),padding=1)` | \(256\times9\times16\times16\) |
| enc3 | `conv_block(256,256)` | \(256\times9\times16\times16\) |
| bottleneck | `conv_block(256,256)` | \(256\times9\times16\times16\) |
| up3 | `ConvTranspose3d(256,128,(3,4,4),stride=2,padding=1)` | \(128\times17\times32\times32\) |
| dec2 | `conv_block(256,128)`，concat \(x_2\) | \(128\times17\times32\times32\) |
| up2 | `ConvTranspose3d(128,64,(1,4,4),stride=(1,2,2),padding=(0,1,1))` | \(64\times17\times64\times64\) |
| dec1 | `conv_block(128,64)`，concat \(x_1\) | \(64\times17\times64\times64\) |
| up1 | `ConvTranspose3d(64,32,(1,4,4),stride=(1,2,2),padding=(0,1,1))` | \(32\times17\times128\times128\) |
| dec0 | `conv_block(64,32)`，concat \(x_0\) | \(32\times17\times128\times128\) |
| out | `Conv3d(32,3,1)` + `tanh` | \(3\times17\times128\times128\) |

`conv_block` 的组成是：

```text
Conv3d(3x3x3, padding=1) -> GroupNorm(8) -> SiLU
-> Conv3d(3x3x3, padding=1) -> GroupNorm(8) -> SiLU
```

残差预测是

\[
\hat x = \operatorname{clamp}\left(x_0+\tanh(\text{decoder}(x_0,\text{control},q)),0,1\right),
\]

其中 \(x_0\) 是首帧重复到 17 帧。输出从 \([B,3,T,H,W]\) permute 成 \([B,T,3,H,W]\)。hidden=32 时参数量为 **12,883,651**（约 12.88M）。

训练参数（`video_wm.py:train_video_wm`）：

| 项目 | 值 |
|---|---|
| 数据 | 128 条 pick-and-place 视频 |
| 分辨率 / 帧数 | 128×128 / 17 |
| batch | 2 |
| steps | 4000 |
| optimizer | AdamW |
| lr | \(2\times10^{-4}\) |
| weight decay | \(10^{-5}\) |
| schedule | CosineAnnealingLR, `eta_min=1e-5` |
| grad clip | 1.0 |
| loss | weighted MSE + \(0.1\times\) L1 |
| foreground weight | \(1+10\cdot\mathbf{1}[\max_c \text{control}_c>0.02]\)，即前景 11、背景 1 |
| 硬件 / 时间 | 单卡 A100，约 3.5 分钟 |
| 训练 loss | 约 \(7.6\times10^{-4}\) |

同样要强调：这个 3D U-Net 接收的是 \(q\) 和 control，不是 \(c\)。硬投影发生在 \(c\) 上，发生在进入 3D U-Net 之前；“latent-level”描述的是运动表示层，不是视频模型的内部特征层。

---

## 9. 复杂度与计算

| 项目 | 数值 / 说明 |
|---|---|
| 运动系数维度 | \(D=10\times7=70\) |
| 约束行数 | 主对比 \(K=45\)（sparse + task velocity）或 49（dense position）；dense + joint velocity 为 56 |
| 投影求解 | \(K\times K\) 线性 solve；\(K\le56\)，单次可忽略 |
| 采样 | 50 步 Euler；每步 3 层 MLP（70→582→512→512→70） |
| 运动先验训练 | 6000 步，batch 64，CPU 约 1 分钟 |
| 6-DOF 视频 WM 训练 | 4000 步，batch 2，A100 约 3.5 分钟 |
| 1.3B Control LoRA | 1500 步，batch 1，A100 40GB 约 25 分钟 |
| sparse 投影 batch | 1（`project_sparse` 限制），16 个样本串行 |

`K` 的准确说法是：主对比最大 49；如果 dense 位置再加 7 维关节速度，则是 56。不要写成“永远 \(K\le49\)”。

---

## 10. 代码地图

| 功能 | 代码 |
|---|---|
| Legendre 递推 / 采样归一化 | `release/src/cnf_motion/basis.py:legendre_values`, `legendre_basis` |
| \(B,B^{+}\) | `release/src/cnf_motion/basis.py:make_basis` |
| 三种导数基 | `release/src/cnf_motion/basis.py:make_derivative_basis` |
| 系数 ↔ 轨迹 | `release/src/cnf_motion/basis.py:coefficients_from_trajectory`, `trajectory_from_coefficients` |
| 运动先验 | `release/src/cnf_motion/motion_field.py:PandaFlowModel`, `fit_motion_prior` |
| 约束矩阵 / 投影 | `release/src/cnf_motion/motion_field.py:constraint_matrix`, `project_coefficients`, `sample_with_projection` |
| 稀疏 + 任务空间速度 | `release/src/cnf_motion/constraints.py:project_sparse`, `sample_sparse`, `task_space_velocity` |
| 安全间隙 / IK / Jacobian | `release/src/cnf_motion/scene.py:solve_ik`, `ee_jacobian`, `enforce_clearance` |
| 视频世界模型 | `release/src/cnf_motion/video_wm.py:VideoWorldModel6D`, `train_video_wm` |
| 轨迹 / 像素指标 | `release/src/cnf_motion/metrics.py` |
| 端到端脚本 | `release/scripts/generate_controls.py`, `eval_action_wm.py`, `reproduce_pickplace.sh` |
| 单元测试 | `release/tests/test_basis.py`, `test_projection.py`, `test_metrics.py`, `test_video_wm.py` |
