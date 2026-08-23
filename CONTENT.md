# Counterfactual Evaluation of Temporal Observation Protocols

# 时间观测协议的反事实评估


## 网站结构（三层）

| 页面 | 路径 | 阅读时间 | 内容来源 |
|---|---|---|---|
| 影响力首页 | `index.html` | 30–60 秒（含 60 秒反例） | 文案直接写在 `index.html`（`.en` / `.zh` 成对元素）；交互在 `js/landing.js` |
| 四分钟导览 | `explainer/index.html` | 约 4 分钟，7 章 21 个 beat | `narration/explainer.json` |
| 技术交互导览 | `technical/index.html` | 约 10 分钟，10 章 43 个 beat | `narration/technical.json` |
| 核心图 / 分享图 | `figures/`、`assets/` | — | `tools/render.mjs` 生成 PNG / MP4 / GIF |

链接、引用、DOI 等集中在 `js/config.js`。

网站内容文稿 · Xizhe Zhang · Nanjing Medical University · 解说音色：我的声音_讲解（VoxCPM2 克隆）


本文件由 `tools/export_content.py` 从网站的真实来源生成：解说词与“Read”段落来自 `narration/script.json`，屏幕文字来自 `js/scenes/ch*.js`，时长来自 `js/narration.js`。

**怎么改**

- 解说词 / 字幕 / Read 段落：直接改 `narration/script.json`（`en`、`zh`；若朗读文本需与字幕不同，用 `en_tts` / `zh_tts`；Read 段落是 HTML 片段，`<span class='f'>` 表示公式）。改完：删掉对应的 `.m4a`，运行 `python3 tools/make_audio.py && python3 tools/check_audio.py && python3 tools/build_narration.py`。
- 屏幕文字（画面里的标题、标签、注释）：在 `js/scenes/ch*.js` 里搜索对应的 `S.t("英文", "中文")`。
- 章节顺序与标题：`narration/script.json` 的 `chapters`（`id`、`title`、`kicker`）；动画场景按 `id` 对应 `js/scenes/`。
- 论文数字：不要手改，`python3 tools/extract_data.py` 从 `paper/numbers.tex` 与 `results/` 重新生成 `js/data.js`。

## 技术导览的结构总览（narration/technical.json）

| # | 章 Chapter | 引题 Kicker | beat 数 | 英文 / 中文时长 | 对应论文 | 画面 |
|---|---|---|---|---|---|---|
| 0 | 问题 · The question | 序 · Prologue | 3 | 0.7 / 0.8 分 | Abstract / §1 | WebGL：260 条潜在轨迹的发光带；A 的蓝色观测线、B 的品红虚线；片头标题卡 |
| 1 | 一个基准固定了什么 · What a benchmark fixes | 第 1 节 · Section 1 | 5 | 1.4 / 1.3 分 | §1 | 整夜睡眠分期图（toy hypnogram）；REM 比例读数；A 的单个片段、B 的四个片段；n → ∞ 与“No.”；两个时刻之间的弧线 |
| 2 | 协议价值 · Protocol value | 第 2 节 · Section 2 | 4 | 1.4 / 1.3 分 | §2 · Definition 1, Proposition 2 | 定义 1 的公式与价值表；K 与 Q_S 热力图、ω 权重条、A_S 行；可拖动的三个测量时刻，均值/占用时间两个目标实时计算 |
| 3 | 两个世界，同一基准 · Two worlds, one benchmark | 第 3 节 · 定理 3 · Section 3 · Theorem 3 | 6 | 2.0 / 1.9 分 | §3 · Theorem 3, Figure 1 | 四点网格；相关函数 ρ₀、ρ₊、ρ₋ 折线；三个可观测量的两列数字与最大差异；I(B;K±) 两个价值表；可拖动的 ε 滑块；Θ–Z₁–Z₂ 重叠示意 |
| 4 | 不可见方向 · Invisible directions | 第 3 节 · 定理 5、命题 6 · Section 3 · Theorem 5, Proposition 6 | 4 | 1.3 / 1.3 分 | §3 · Definition 4, Theorem 5, Proposition 6, Example 7 | 16×16 热力图只亮 4×4 子块；三块公式与不可见条件；120 vs 15 的计数；定理 5 的导数；Z₁↔Z₂ 置换动画与 K_a / K_a′ 热力图 |
| 5 | 恢复可识别性 · Restoring identification | 第 3.5–4 节 · 命题 8、定理 10–11 · Sections 3.5–4 · Proposition 8, Theorems 10–11 | 5 | 1.6 / 1.7 分 | §3.5–4 · Proposition 8, Theorems 10–11, Figure 2a | 四点补测与不可见维数 4 → 2 → 0；60 条密集校准轨迹（WebGL）；四步估计流水线；定理 10 的 β 表；图 2a 的对数–对数误差曲线（真实结果） |
| 6 | 分辨率 · Resolution | 第 4 节 · 推论 12 · Section 4 · Corollary 12 | 4 | 1.2 / 1.2 分 | §4 · Corollary 12, Figure 2b–d | “≤ 2ε”的三行论证与误差条；四层嵌套候选类与 S*；图 2c,d 的损失与一致误差曲线；结论卡 |
| 7 | 针对目标的观测设计 · Target-aware design | 第 5 节 · 命题 13、引理 14 · Section 5 · Proposition 13, Lemma 14 | 5 | 1.4 / 1.5 分 | §5 · Proposition 13, Lemma 14, Table 1 | 后验轨迹带（WebGL）随贪心加点塌缩；候选目录与动作五元组；秩一收益公式；各候选的边际收益条；两种目标选出的时刻；表 1 的效率条 |
| 8 | 睡眠与房颤 · Sleep and atrial fibrillation | 第 6.4 节 · 图 3–4 · Section 6.4 · Figures 3–4 | 4 | 1.4 / 1.5 分 | §6.4 · Figures 3–4 | 两张数据卡（Sleep-EDF / LTAF）；REM 分散 vs 连续柱图；房颤负担折线；图 4 的支持稳定性曲线与重抽样区间 |
| 9 | 先识别，再优化 · Identification before optimisation | 第 8 节 · Section 8 | 3 | 0.8 / 0.8 分 | §8 | 识别 / 校准 / 设计三列；一句话原则；发布信息卡 |

合计：43 个 beat，英文 13.4 分钟，中文 13.2 分钟。


叙事线：识别（ch3–ch4）→ 校准（ch5–ch6）→ 设计（ch7）→ 真实数据（ch8）→ 原则（ch9），前面 ch0–ch2 建立问题与“协议价值”这个被评估的对象。每个 beat = 一句解说 + 一次画面变化；解说读完才推进到下一个 beat。


---

## 0 · 问题 / The question

- 引题：序 / Prologue
- 对应论文：Abstract / §1
- 画面：WebGL：260 条潜在轨迹的发光带；A 的蓝色观测线、B 的品红虚线；片头标题卡
- Visual: WebGL field of 260 latent trajectories; blue ticks for protocol A, dashed magenta for B; title card

### 解说词 Narration

**ch0-b0**  (EN 13 s · ZH 13 s)

- EN: Counterfactual evaluation of temporal observation protocols. Over the next ten minutes or so, I'd like to walk you through what this paper asks, what it proves, and what it found in real data.
- ZH: 《时间观测协议的反事实评估》。接下来的十分钟左右，我想带你走一遍这篇论文：它问了什么、证明了什么，又在真实数据里看到了什么。

**ch0-b1**  (EN 18 s · ZH 19 s)

- EN: Each of these glowing lines is a latent trajectory, unfolding over a long horizon. In most studies we never record the whole thing. What we record is decided by an observation protocol: when to look, for how long, how often, and how precisely.
- ZH: 你看到的每一条发光的曲线，都是一条在长时间跨度上展开的潜在轨迹。在大多数研究里，我们从来不会把它完整记录下来。记录下来的是什么，由观测协议决定：什么时候看、看多久、看几次、看得多准。

**ch0-b2**  (EN 14 s · ZH 14 s)

- EN: So here is the question the paper asks. Suppose the data were collected under one protocol. Can those data tell us how well a different protocol, one that was never run, would predict the target?
- ZH: 于是论文要问的问题是这样的：假设数据是在某一种协议下采集的，这些数据能不能告诉我们，另一种从来没有实施过的协议，会把目标预测到多准？

### 屏幕文字 On-screen text

| EN | ZH |
|---|---|
| horizon of one unit (a night, a day, a week) | 一个对象的时间跨度（一夜、一天、一周） |
| Counterfactual Evaluation of | 时间观测协议的 |
| Temporal Observation Protocols | 反事实评估 |
| Xizhe Zhang · Nanjing Medical University | 张锡哲 · 南京医科大学 |
| latent trajectory Z(t), one unit per line | 潜在轨迹 Z(t)，每条线是一个对象 |
| target Θ = aggregate over the whole horizon | 目标 Θ = 整个时间跨度上的聚合量 |
| protocol A: where the benchmark looked | 协议 A：基准数据实际观测的位置 |
| protocol B: never run | 协议 B：从未实施 |
| Do data collected under A determine how well B would predict Θ? | 在 A 下采集的数据，能否确定 B 会把 Θ 预测到多准？ |

### Read 段落（章下方的论文摘要）

**EN**

Predictive performance in temporal learning depends on the learner *and* on the observation protocol that produced the data. Benchmarks hold the protocol fixed. This paper asks whether the joint measurement–target law under the **realised** protocol determines the predictive value of an **undeployed** alternative — counterfactual protocol evaluation. The answer is no in general (Section 3), but it can be restored by targeted augmentation or dense calibration (Sections 3.5–4), after which cost-constrained, target-aware design becomes well posed (Section 5).

**ZH**

时间序列学习的预测性能既取决于模型，也取决于产生数据的观测协议。基准数据集把协议固定住了。这篇论文问：**已实施**协议下测量与目标的联合分布，能否确定一个**从未实施**的替代协议的预测价值——即反事实协议评估。一般情况下答案是否定的（第 3 节），但通过定向补测或密集校准可以恢复可识别性（第 3.5–4 节），此后在预算约束下针对目标做观测设计才是适定的（第 5 节）。


---

## 1 · 一个基准固定了什么 / What a benchmark fixes

- 引题：第 1 节 / Section 1
- 对应论文：§1
- 画面：整夜睡眠分期图（toy hypnogram）；REM 比例读数；A 的单个片段、B 的四个片段；n → ∞ 与“No.”；两个时刻之间的弧线
- Visual: Toy hypnogram; REM-fraction read-out; A's single epoch vs B's four; the n → ∞ chain and “No.”; the arc between two times

### 解说词 Narration

**ch1-b0**  (EN 18 s · ZH 15 s)

- EN: Let's make this concrete with a night of sleep. The night is scored every thirty seconds, so roughly nine hundred and sixty epochs. The thing we want to predict is the fraction of the whole night spent in REM sleep. One number, summarising the entire trajectory.
- ZH: 我们用一夜睡眠把它说具体。整夜每三十秒评一个分期，大约九百六十个片段。我们想预测的，是整夜处于快速眼动睡眠的时间比例——一个数，概括整条轨迹。

**ch1-b1**  (EN 17 s · ZH 18 s)

- EN: Now, scoring is expensive. So imagine a benchmark that scores just one epoch per subject, thirty minutes after sleep onset. Let's call that protocol A. The benchmark is a large collection of pairs: one measurement, and one target, for each person.
- ZH: 问题是，人工分期很贵。所以想象这样一个基准数据集：每个人只评一个片段，就是入睡后三十分钟那一段。我们把它叫做协议 A。这个基准数据，就是一大批配对样本：每个人一个测量值，一个目标值。

**ch1-b2**  (EN 17 s · ZH 13 s)

- EN: Then a study planner comes along and asks: what if, instead, we scored one epoch at hours one, three, five and seven? How much better would the prediction be? That's protocol B. And the crucial thing is, nobody has ever run it.
- ZH: 这时候，一位做研究设计的人来问：如果我们改成在第一、三、五、七小时各评一个片段，预测能好多少？这就是协议 B。关键在于，它从来没有人实施过。

**ch1-b3**  (EN 18 s · ZH 18 s)

- EN: The natural intuition is that if we just collect enough subjects under A, the answer will emerge. A hundred. Ten thousand. A million. The first result of the paper is that it need not. Even infinite data under protocol A can leave protocol B's value undetermined.
- ZH: 很自然的直觉是：只要在协议 A 下多收集一些人，答案总会浮现出来。一百人，一万人，一百万人。而论文的第一个结果是：未必。哪怕协议 A 下的数据无穷多，协议 B 的价值仍然可能无法确定。

**ch1-b4**  (EN 16 s · ZH 16 s)

- EN: And the reason is actually simple. Protocol A never observes two different times of the same night. So it does not determine the joint dependence among the times it never looked at. And protocol B's value depends on exactly that.
- ZH: 原因其实很简单。协议 A 从来没有在同一夜里观测过两个不同的时刻，所以它确定不了那些没被看过的时刻之间是怎么联合依赖的。而协议 B 的价值，恰恰取决于这一点。

### 屏幕文字 On-screen text

| EN | ZH |
|---|---|
| A night of sleep, fully scored | 整夜睡眠：完整轨迹与聚合目标 |
| = REM epochs / all epochs  = | = REM 片段数 / 全部片段数  = |
| The target is not the label of one moment; it is an aggregate of the whole trajectory. | 目标不是某一瞬间的标签，而是整条轨迹的聚合量。 |
| A: one epoch, 30 min after onset | A：入睡后 30 分钟的一个片段 |
| one measurement per subject · one trajectory-level target per subject | 每人一个测量值 · 每人一个轨迹级目标 |
| B: one epoch at hours 1, 3, 5, 7 — never deployed | B：第 1、3、5、7 小时各一个片段——从未实施 |
| How well would protocol B predict Θ?  Only A's data exist. | 协议 B 能把 Θ 预测到多好？手上只有 A 的数据。 |
| Intuition: with enough subjects under A, the answer must emerge. | 直觉：协议 A 下样本够多，总能算出来吧？ |
| No. | 不行。 |
| Even infinite data under A need not determine B's value. | 即使 A 下的数据无穷多，也未必能确定 B 的价值。 |
| how are two times related? | 两个时刻之间如何相关？ |
| Protocol A never observes two different times of the same night, | 协议 A 从未在同一夜里观测过两个不同时刻， |
| so it does not determine the joint dependence among the unobserved times. | 所以它无法确定未观测时刻之间的联合依赖。 |
| Protocol B's value depends on exactly that. | 而协议 B 的价值恰恰取决于这一点。 |
| The first question is whether the benchmark contains enough information to evaluate a new sampling scheme. | 第一个问题是：基准数据是否包含足够的信息来评估一种新的采样方案。 |
| latent trajectory Z(t): one sleep stage every 30 s, about 960 epochs per night | 潜在轨迹 Z(t)：每 30 秒一个睡眠分期，整夜约 960 个片段 |

### Read 段落（章下方的论文摘要）

**EN**

The target is a *temporal aggregate* of a latent trajectory, `Θ = \int ω(t)\,g(Z(t))\,\text{d}t`: with `g(z)=z` an average level, with `g(z) = \mathbf{1}\{z > c\}` an occupation time such as REM proportion or AF burden. A protocol is a finite set of noisy linear acquisition actions under a cost budget. Weak benchmark performance may reflect the learner, the sample size, or target-relevant information the protocol never admitted. Only the third is addressed by changing what is observed — and whether the benchmark can even evaluate that change is the identification question of Section 3.

**ZH**

目标是潜在轨迹的*时间聚合量*：`Θ = \int ω(t)\,g(Z(t))\,\text{d}t`。取 `g(z)=z` 是平均水平；取 `g(z) = \mathbf{1}\{z > c\}` 是占用时间，例如 REM 比例或房颤负担。协议是预算约束下的一组带噪声线性测量动作。基准上表现差，可能源于模型、样本量，或协议根本没有采到的目标相关信息。只有第三种需要改变观测本身——而基准数据能否评估这种改变，正是第 3 节的识别问题。


---

## 2 · 协议价值 / Protocol value

- 引题：第 2 节 / Section 2
- 对应论文：§2 · Definition 1, Proposition 2
- 画面：定义 1 的公式与价值表；K 与 Q_S 热力图、ω 权重条、A_S 行；可拖动的三个测量时刻，均值/占用时间两个目标实时计算
- Visual: Definition 1 with a gauge; K and Q_S heat maps, ω strip, A_S rows; three draggable measurement times with live mean / occupation values

### 解说词 Narration

**ch2-b0**  (EN 24 s · ZH 20 s)

- EN: Before going further, let's be precise about what is being evaluated. The value of a protocol is the population R squared of the best possible predictor of the target, built from that protocol's measurements. It's a number between zero and one, and it belongs to the protocol itself. Not to any particular learner, and not to any sample size.
- ZH: 在往下走之前，先把评估的对象说清楚。一个协议的价值，是用这个协议的测量去预测目标时，最优预测器能达到的总体 R 方。它是零到一之间的一个数，属于协议本身——跟具体用什么模型、有多少样本都没有关系。

**ch2-b1**  (EN 23 s · ZH 20 s)

- EN: In the Gaussian model this becomes very tangible. The latent trajectory, on a grid of time points, has a correlation matrix K. The target is a weighted average of some transformation of the state, with weights omega. And a protocol observes just a few noisy linear measurements, described by its rows A and its noise R.
- ZH: 在高斯模型里，这件事变得很具体。潜在轨迹在一组时间格点上有一个相关矩阵 K。目标是对状态做某种变换之后的加权平均，权重是 omega。而一个协议，只观测少数几个带噪声的线性测量，由它的测量矩阵 A 和噪声 R 来描述。
  - ZH 朗读文本: 在高斯模型里，这件事变得很具体。潜在轨迹在一组时间格点上有一个相关矩阵 K。目标是对状态做某种变换之后的加权平均，权重是 欧米伽。而一个协议，只观测少数几个带噪声的线性测量，由它的测量矩阵 A 和噪声 R 来描述。

**ch2-b2**  (EN 22 s · ZH 22 s)

- EN: When we condition on those measurements, we recover part of the trajectory. That recovered part has a covariance, Q. And here's a nice fact: however nonlinear the target is, it enters the calculation through a single scalar transform. The value is then simply the explained target variance, F, divided by the total target variance, V.
- ZH: 以这些测量为条件，我们能恢复轨迹的一部分。恢复出来的这部分有一个协方差，叫 Q。这里有一个很漂亮的事实：不管目标多么非线性，它都只通过一个标量变换进入计算。于是协议价值，就是被解释的目标方差 F，除以目标的总方差 V。

**ch2-b3**  (EN 17 s · ZH 15 s)

- EN: Try it yourself. Drag the measurement times along the horizon and watch the value respond. It changes with where you look, and it also changes with what you're trying to predict. In this example, the two targets end up selecting different measurement times.
- ZH: 你可以自己试一试。沿着时间轴拖动测量点，看价值怎么变。它随观测位置变化，也随你想预测的目标变化。在这个例子里，两种目标最终会选出不同的测量时刻。

### 屏幕文字 On-screen text

| EN | ZH |
|---|---|
| Protocol value: the population R² a protocol can support | 协议价值：一个协议所能支持的总体 R² |
| population R² of the best predictor of Θ from the protocol's measurements | 用该协议的测量预测 Θ 的最优预测器的总体 R² |
| a property of the protocol — not of a learner, not of a sample size | 是协议的性质——与模型、样本量无关 |
| the part of the trajectory recovered from Y_S | 从 Y_S 中恢复出的那部分轨迹 |
| every nonlinearity of the target enters through C_g alone | 目标的全部非线性只通过 C_g 进入 |
| \\text{mean target: } C_g(r) = r | \\text{均值目标：} C_g(r) = r |
| \\text{occupation above 0: } C_g(r) = \\frac{arcsin(r)}{2π} | \\text{零阈值占用时间：} C_g(r) = \\frac{arcsin(r)}{2π} |
| Try it: drag the three measurement times | 试一试：拖动三个测量时刻 |
| mean target | 均值目标 |
| occupation target | 占用时间目标 |
| same K, same three measurements — different targets, different values | 同一个 K、同样三次测量——目标不同，价值不同 |

### Read 段落（章下方的论文摘要）

**EN**

**Definition 1.** The Bayes value of protocol `S` is `I(S) = \frac{Var(E[Θ \mid Y_S])}{Var(Θ)}`, the population R² of the optimal predictor. Under the discrete Gaussian model `Z \sim N(0, K)`, `diag(K)=1`, with measurements `Y_S = A_SZ + ε`, **Proposition 2** gives the closed form `I_g(S;K) = \frac{F_g(S;K)}{V_g(K)}`, where `Q_S(K) = K A_S^\top (A_S K A_S^\top + R_S)^{-1} A_S K` is the covariance of the recovered trajectory and `C_g(r) = Cov\{g(U), g(V_r)\}` is the Gaussian covariance transform of the target function (for the mean target `C_g(r)=r`; for occupation above 0, `\frac{arcsin(r)}{2π}`). Every nonlinearity of the target enters through `C_g` alone.

**ZH**

**定义 1.** 协议 `S` 的贝叶斯价值是 `I(S) = \frac{Var(E[Θ \mid Y_S])}{Var(Θ)}`，即最优预测器的总体 R²。在离散高斯模型 `Z \sim N(0, K)`、`diag(K)=1`、测量 `Y_S = A_SZ + ε` 下，**命题 2** 给出闭式 `I_g(S;K) = \frac{F_g(S;K)}{V_g(K)}`：其中 `Q_S(K) = K A_S^\top (A_S K A_S^\top + R_S)^{-1} A_S K` 是从测量恢复出的那部分轨迹的协方差，`C_g(r) = Cov\{g(U), g(V_r)\}` 是目标函数的高斯协方差变换（均值目标 `C_g(r)=r`；零阈值占用时间为 `\frac{arcsin(r)}{2π}`）。目标的全部非线性只通过 `C_g` 进入。


---

## 3 · 两个世界，同一基准 / Two worlds, one benchmark

- 引题：第 3 节 · 定理 3 / Section 3 · Theorem 3
- 对应论文：§3 · Theorem 3, Figure 1
- 画面：四点网格；相关函数 ρ₀、ρ₊、ρ₋ 折线；三个可观测量的两列数字与最大差异；I(B;K±) 两个价值表；可拖动的 ε 滑块；Θ–Z₁–Z₂ 重叠示意
- Visual: Four-point grid; ρ₀/ρ₊/ρ₋ profiles; the three observables in both worlds and their discrepancy; gauges for I(B;K±); draggable ε slider; the Θ–Z₁–Z₂ overlap diagram

### 解说词 Narration

**ch3-b0**  (EN 22 s · ZH 21 s)

- EN: Now let's look at the smallest case where evaluation fails. Four time points, a stationary process, and the target is simply their mean. Protocol A observes only the first point. Everything the benchmark can ever learn comes down to three numbers: the variance of the measurement, its covariance with the target, and the variance of the target.
- ZH: 现在来看评估失效的最小例子。四个时间点，一个平稳过程，目标就是这四个点的平均。协议 A 只观测第一个点。这个基准数据能学到的全部内容，归结起来只有三个数：测量值的方差、它与目标的协方差、还有目标本身的方差。

**ch3-b1**  (EN 21 s · ZH 17 s)

- EN: Start from the correlation profile e to the minus lag. Then push it along the direction zero, one, minus two, one, by plus epsilon in one direction and minus epsilon in the other. That gives us two worlds, K plus and K minus, and as you can see, their temporal dependence looks quite different.
- ZH: 我们从相关函数 e 的负滞后次方出发，沿着零、一、负二、一这个方向，一边加上 epsilon，一边减去 epsilon。这样就得到两个世界：K 加和 K 减。你可以看到，它们的时间依赖结构明显不一样。
  - ZH 朗读文本: 我们从相关函数 e 的负滞后次方出发，沿着零、一、负二、一这个方向，一边加上 艾普西隆，一边减去 艾普西隆。这样就得到两个世界：K 加和 K 减。你可以看到，它们的时间依赖结构明显不一样。

**ch3-b2**  (EN 16 s · ZH 16 s)

- EN: And yet, every number the benchmark can measure is identical in the two worlds. Identical to sixteen decimal places. Because the benchmark laws are the same, no procedure that only uses protocol A's data can ever distinguish these two worlds.
- ZH: 然而，基准数据能测到的每一个数，在两个世界里都完全一样，精确到小数点后十六位。既然基准分布相同，那么任何只用协议 A 数据的方法，都不可能把这两个世界区分开。

**ch3-b3**  (EN 20 s · ZH 20 s)

- EN: Now let's evaluate protocol B, which observes points one and two. In world K plus, its value is zero point six eight. In world K minus, it's zero point eight three. The same data support two different answers. So no estimator built on A's data can be right in both worlds.
- ZH: 现在我们来评估协议 B，它观测第一和第二个点。在 K 加的世界里，它的价值是零点六八；在 K 减的世界里，是零点八三。同样的数据，支持两个不同的答案。所以任何只建立在 A 数据上的估计器，都不可能在两个世界里同时正确。

**ch3-b4**  (EN 17 s · ZH 19 s)

- EN: Go ahead and move epsilon yourself. The three observed numbers stay frozen, while B's value slides. This is Theorem 3. And four points is sharp: with three points or fewer, the same benchmark would pin down the whole stationary correlation profile.
- ZH: 你可以自己拖一下 epsilon。三个可观测的数纹丝不动，而协议 B 的价值在滑动。这就是定理三。而且四个点是一个精确的门槛：只有三个点或者更少的时候，同样的基准数据就足以确定整个平稳相关函数。
  - ZH 朗读文本: 你可以自己拖一下 艾普西隆。三个可观测的数纹丝不动，而协议 B 的价值在滑动。这就是定理三。而且四个点是一个精确的门槛：只有三个点或者更少的时候，同样的基准数据就足以确定整个平稳相关函数。

**ch3-b5**  (EN 23 s · ZH 23 s)

- EN: So why does B's value move? Look at what the perturbation does. It keeps each new measurement's link to the target exactly fixed. What it changes is only how redundant the two new measurements are with each other. A protocol's value depends on the overlap between its measurements, and that overlap is precisely what A's data never see.
- ZH: 那么，B 的价值为什么会动？看看这个扰动做了什么。它让每个新测量与目标之间的联系保持不变，改变的只是两个新测量彼此之间的冗余程度。一个协议的价值，取决于它的测量之间重叠了多少；而这个重叠，恰恰是协议 A 的数据永远看不到的。

### 屏幕文字 On-screen text

| EN | ZH |
|---|---|
| Theorem 3 — the smallest case where evaluation fails | 定理 3——评估失效的最小例子 |
| \\text{stationary, standardised: } Corr(Z_j, Z_k) = ρ(\|j − k\|) | \\text{平稳、标准化：} Corr(Z_j, Z_k) = ρ(\|j − k\|) |
| correlation profile ρ(lag) | 相关函数 ρ(滞后) |
| Everything the benchmark can ever learn | 基准数据能学到的全部内容 |
| \\text{world } K_+ | \\text{世界 } K_+ |
| \\text{world } K_− | \\text{世界 } K_− |
| max discrepancy   | 最大差异   |
| = floating-point noise: the two benchmark laws coincide | = 浮点误差量级：两个基准分布完全相同 |
| \\text{value of protocol } B = \\{Z_1, Z_2\\} | \\text{协议 } B = \\{Z_1, Z_2\\} \\text{ 的价值} |
| difference  | 相差  |
| same data, two answers → no A-data estimator is consistent in both worlds | 同样的数据，两个答案 → 没有任何只用 A 数据的估计器能在两个世界都相合 |
| paper's ε | 论文取值 |
| \\text{ (last admissible)} | \\text{（可容许的上限）} |
| drag ε: the three observed numbers stay frozen, B's value slides | 拖动 ε：三个可观测的数纹丝不动，B 的价值在滑动 |
| fixed | 不变 |
| the only thing that moves: the overlap between B's two measurements | 唯一在变的：B 的两个测量之间的重叠 |
| A protocol's value depends on how much its measurements overlap — and A's data never saw two times together. | 协议的价值取决于它的测量之间重叠多少——而 A 的数据从未同时看到两个时刻。 |

### Read 段落（章下方的论文摘要）

**EN**

**Theorem 3 (sharp minimal stationary counterexample).** For a stationary standardised process on `p` points, a uniform mean target and a one-point benchmark, the stationary invisible space has dimension `p − 3`: identification holds for `p ≤ 3` and fails from `p = 4`. At `p = 4` the kernel is spanned by the lag perturbation `δ = (0, 1, −2, 1)`, and for `B = \{Z_1, Z_2\}` the two values are `I(B;K_±) = \frac{2b^2}{(1 + ν_B^2 + ρ(1) ± ε)\,Var(Θ)}`, `b = \frac{1 + 2ρ(1) + ρ(2)}{4}`. Figure 1 of the paper uses `ρ_0(u) = e^{−u}` and `ε = 0.1321`: both worlds give `Var(Y_A) = 1`, `Cov(Y_A, Θ) = 0.388250`, `Var(Θ) = 0.428012` to `10^{-16}`, yet `I(B;K_{+}) = 0.6817` and `I(B;K_{−}) = 0.8274`. Along `δ` the target's covariance with `Z_1` and `Z_2` is fixed; only `Cov(Z_1, Z_2)` moves — the redundancy between B's two measurements.

**ZH**

**定理 3（精确的最小平稳反例）.** 对 `p` 个格点上的平稳标准化过程、均匀均值目标和单点基准，平稳不可见空间的维数是 `p − 3`：`p ≤ 3` 时可识别，`p = 4` 起失效。`p = 4` 时核由滞后扰动 `δ = (0, 1, −2, 1)` 张成；对 `B = \{Z_1, Z_2\}`，两个价值为 `I(B;K_±) = \frac{2b^2}{(1 + ν_B^2 + ρ(1) ± ε)\,Var(Θ)}`，`b = \frac{1 + 2ρ(1) + ρ(2)}{4}`。论文图 1 取 `ρ_0(u) = e^{−u}`、`ε = 0.1321`：两个世界的 `Var(Y_A) = 1`、`Cov(Y_A, Θ) = 0.388250`、`Var(Θ) = 0.428012` 一致到 `10^{-16}`，而 `I(B;K_{+}) = 0.6817`、`I(B;K_{−}) = 0.8274`。沿 `δ` 移动时，目标与 `Z_1`、`Z_2` 的协方差不变，只有 `Cov(Z_1, Z_2)`——B 两个测量之间的冗余——在变。


---

## 4 · 不可见方向 / Invisible directions

- 引题：第 3 节 · 定理 5、命题 6 / Section 3 · Theorem 5, Proposition 6
- 对应论文：§3 · Definition 4, Theorem 5, Proposition 6, Example 7
- 画面：16×16 热力图只亮 4×4 子块；三块公式与不可见条件；120 vs 15 的计数；定理 5 的导数；Z₁↔Z₂ 置换动画与 K_a / K_a′ 热力图
- Visual: 16×16 heat map with the 4×4 visible block lit; the three blocks and the invisibility conditions; the 120 vs 15 count; Theorem 5's derivative; the Z₁↔Z₂ swap with K_a / K_a′

### 解说词 Narration

**ch4-b0**  (EN 18 s · ZH 19 s)

- EN: Let's step back to the general picture. The benchmark law depends on K only through three blocks: the covariance among the realised measurements, their covariance with the target, and the target variance. Any symmetric perturbation that leaves all three blocks unchanged is, in effect, invisible.
- ZH: 退一步看一般的情形。基准数据的分布只通过三块依赖于 K：已实施测量之间的协方差、它们与目标的协方差、以及目标的方差。任何一个保持这三块不变的对称扰动，对基准数据来说都是不可见的。

**ch4-b1**  (EN 24 s · ZH 20 s)

- EN: It helps to count degrees of freedom. A standardised K on p grid points has p times p minus one over two free entries. A benchmark with d measurements supplies at most d times d plus one over two, plus d, plus one constraints. With sixteen points and four measurements, that's one hundred and twenty unknowns against fifteen constraints.
- ZH: 数一数自由度会很有帮助。p 个格点上的标准化 K，有 p 乘 p 减一除以二个自由参数。而有 d 个测量的基准数据，最多提供 d 乘 d 加一除以二，再加 d 加一个约束。十六个格点、四个测量的话，就是一百二十个未知数对十五个约束。

**ch4-b2**  (EN 20 s · ZH 20 s)

- EN: Theorem 5 turns this into a test you can actually run. If the value of protocol B has a non-zero derivative along an invisible direction, then that value is not identified. The two worlds, K zero plus and minus epsilon delta, generate the same benchmark law, and they disagree about B.
- ZH: 定理五把这件事变成一个真正可以执行的检验。如果协议 B 的价值沿着某个不可见方向的导数不为零，那么这个价值就不可识别：K 零加减 epsilon delta 这两个世界，产生同样的基准分布，却对协议 B 给出不同的答案。
  - ZH 朗读文本: 定理五把这件事变成一个真正可以执行的检验。如果协议 B 的价值沿着某个不可见方向的导数不为零，那么这个价值就不可识别：K 零加减 艾普西隆 德尔塔 这两个世界，产生同样的基准分布，却对协议 B 给出不同的答案。

**ch4-b3**  (EN 18 s · ZH 17 s)

- EN: What about nonlinear targets, like occupation time? There, matching moments is not enough, so the paper uses a permutation argument instead. Swapping two unobserved time points leaves the benchmark law exactly unchanged, and yet it can change B's value. And three grid points already suffice.
- ZH: 那么非线性目标呢，比如占用时间？在那种情况下，只比较矩是不够的，所以论文改用了置换构造。把两个没被观测的时间点交换一下，基准分布完全不变，但协议 B 的价值却可以改变。而且，三个格点就足够了。

### 屏幕文字 On-screen text

| EN | ZH |
|---|---|
| Nonlinear targets: an exact permutation argument (Proposition 6, Example 7) | 非线性目标：精确的置换构造（命题 6、例 7） |
| Which directions of K can the benchmark see? | 基准数据能看到 K 的哪些方向？ |
| protocol A observes d = 4 of the 16 grid points | 协议 A 观测 16 个格点中的 d = 4 个 |
| lit: what AKAᵀ pins down · dimmed: unconstrained by the benchmark | 亮：AKAᵀ 固定的部分 · 暗：基准数据不约束的部分 |
| The benchmark law depends on K through three blocks | 基准分布只通过三块依赖于 K |
| covariance among realised measurements | 已实施测量之间的协方差 |
| their covariance with the target | 它们与目标的协方差 |
| the target variance | 目标的方差 |
| Δ \\text{ is invisible } \\Leftrightarrow A Δ A^\\top = 0, \\; A Δ h = 0, \\; h^\\top Δ h = 0, \\; diag(Δ) = 0 | Δ \\text{ 不可见 } \\Leftrightarrow A Δ A^\\top = 0, \\; A Δ h = 0, \\; h^\\top Δ h = 0, \\; diag(Δ) = 0 |
| free entries of K | K 的自由参数 |
| constraints from the benchmark | 基准数据给出的约束 |
| invisible directions | 不可见方向 |
| O(d²) constraints against O(p²) parameters: long horizons with sparse protocols leave most of K invisible | O(d²) 个约束对 O(p²) 个参数：长跨度 + 稀疏采样，K 的大部分方向都看不见 |
| Theorem 5 — value-changing invisible directions imply non-identification | 定理 5——改变价值的不可见方向意味着不可识别 |
| D_B ≠ 0  ⇒  K₀ ± εΔ share the benchmark law and value B differently; more units under A cannot help. | D_B ≠ 0  ⇒  K₀ ± εΔ 有相同的基准分布，却给 B 不同的价值；在 A 下再多收集对象也无济于事。 |
| Ξ\\text{: swap coordinates 1 and 2;} \\quad A Ξ = A, \\quad Ξ^\\top ω = ω | Ξ\\text{：交换坐标 1 与 2；} \\quad A Ξ = A, \\quad Ξ^\\top ω = ω |
| the permutation leaves every realised measurement and the aggregate weights unchanged | 置换不改变任何已实施的测量，也不改变聚合权重 |
| joint law of (Y_A, Θ_g):  identical for every g ∈ L²(φ) | (Y_A, Θ_g) 的联合分布：对每个 g ∈ L²(φ) 都完全相同 |
| for every nonconstant g, every a ∈ (0,1), every noise level | 对每个非常数 g、每个 a ∈ (0,1)、每个噪声水平都成立 |
| Exact, not asymptotic; covers occupation-time targets; needs no stationarity — three grid points suffice. | 精确而非渐近；覆盖占用时间目标；不需要平稳性——三个格点就够。 |

### Read 段落（章下方的论文摘要）

**EN**

**Definition 4 / Theorem 5.** A symmetric zero-diagonal `Δ` is *invisible* to `(A, h)` if `AΔA^\top = 0`, `AΔh = 0`, `h^\topΔh = 0`. For every small `ε`, `K_0 ± εΔ` are correlation matrices with the same benchmark law. If the directional derivative D_B(Δ;K_0) = \frac{2h^\top Δ B^\top W B K_0 h − h^\top K_0 B^\top W (B Δ B^\top) W B K_0 h}{h^\top K_0 h} (`W = (B K_0 B^\top + R_B)^{-1}`) is non-zero, the value of `B` is not locally identified. Rank–nullity gives `dim\, D(A, h) ≥ p(p − 1)/2 − d(d + 1)/2 − d − 1`: `O(d^2)` constraints against `O(p^2)` parameters. **Proposition 6 / Example 7.** For arbitrary `g ∈ L^2(φ)`, a permutation `Ξ` with `AΞ = A` and `Ξ^\topω = ω` makes the laws of `(Y_A, Θ_g)` under `K` and `ΞKΞ^\top` coincide exactly while `I_g(B; ΞKΞ^\top) = I_g(BΞ; K)`; three points suffice under uniform aggregation.

**ZH**

**定义 4 / 定理 5.** 对称、零对角的 `Δ` 对 `(A, h)` *不可见*，若 `AΔA^\top = 0`、`AΔh = 0`、`h^\topΔh = 0`。对足够小的 `ε`，`K_0 ± εΔ` 都是相关矩阵且基准分布相同。若方向导数 D_B(Δ;K_0) = \frac{2h^\top Δ B^\top W B K_0 h − h^\top K_0 B^\top W (B Δ B^\top) W B K_0 h}{h^\top K_0 h}（`W = (B K_0 B^\top + R_B)^{-1}`）非零，则 `B` 的价值局部不可识别。秩–零化度给出 `dim\, D(A, h) ≥ p(p − 1)/2 − d(d + 1)/2 − d − 1`：`O(d^2)` 个约束对 `O(p^2)` 个参数。**命题 6 / 例 7.** 对任意 `g ∈ L^2(φ)`，满足 `AΞ = A`、`Ξ^\topω = ω` 的置换 `Ξ` 使 `K` 与 `ΞKΞ^\top` 下 `(Y_A, Θ_g)` 的分布完全相同，而 `I_g(B; ΞKΞ^\top) = I_g(BΞ; K)`；均匀聚合下三个格点即可。


---

## 5 · 恢复可识别性 / Restoring identification

- 引题：第 3.5–4 节 · 命题 8、定理 10–11 / Sections 3.5–4 · Proposition 8, Theorems 10–11
- 对应论文：§3.5–4 · Proposition 8, Theorems 10–11, Figure 2a
- 画面：四点补测与不可见维数 4 → 2 → 0；60 条密集校准轨迹（WebGL）；四步估计流水线；定理 10 的 β 表；图 2a 的对数–对数误差曲线（真实结果）
- Visual: Augmentation with invisible dimension 4 → 2 → 0; 60 dense calibration paths (WebGL); the four-step estimator; Theorem 10's β table; Figure 2a log–log error curves from the results files

### 解说词 Narration

**ch5-b0**  (EN 23 s · ZH 23 s)

- EN: So what does restore identification? Not more of the same data. The criterion is value-specific: we only need to remove the ambiguity that changes the value we're evaluating. In the four-point example, without assuming stationarity, adding one measurement at point one shrinks the invisible space from four dimensions to two, and adding point two removes it entirely.
- ZH: 那么，什么才能恢复可识别性？不是更多同样的数据。识别的准则是“针对价值”的：我们只需要消除会改变所评估价值的那部分歧义。在不假设平稳的四点例子里，增加对第一个点的测量，把不可见空间从四维缩到二维；再加上第二个点，就完全消除了。

**ch5-b1**  (EN 20 s · ZH 20 s)

- EN: If we want to evaluate a whole family of candidate protocols, there's a more direct route: dense calibration. Take a small number of units, m, and record their entire trajectories, noise and all. The law of those data identifies the latent covariance, and with it, the value of every candidate.
- ZH: 如果我们想评估整个候选协议族，还有一条更直接的路：密集校准。取少量 m 个对象，把它们的整条轨迹完整记录下来，带噪声也没关系。这些数据的分布就能识别潜在协方差，进而识别每一个候选协议的价值。

**ch5-b2**  (EN 16 s · ZH 18 s)

- EN: The estimator itself has four steps. Form the sample covariance. Subtract the calibration noise. Floor the eigenvalues, so that what's left is a valid covariance. And rescale to a correlation matrix. Then you plug that estimate into every candidate's value.
- ZH: 这个估计量本身分四步：先算样本协方差；再扣掉校准噪声；然后给特征值设一个下限，保证结果是合法的协方差；最后重新标准化成相关矩阵。之后，把这个估计代入每个候选协议的价值公式就可以了。

**ch5-b3**  (EN 20 s · ZH 22 s)

- EN: Theorem 10 then tells us how covariance error turns into value error, uniformly across the whole candidate family. For smooth targets, the value error is linear in the covariance error. For threshold targets, the worst case is a square root. But at any fixed interior model, it's linear again.
- ZH: 接着，定理十告诉我们协方差误差是怎么变成价值误差的，而且是在整个候选族上一致成立的。对平滑目标，价值误差和协方差误差是线性关系；对阈值型目标，最坏情形是平方根关系；但在任何固定的内部模型上，它又回到线性。

**ch5-b4**  (EN 19 s · ZH 18 s)

- EN: And in simulation this is exactly what we see. The largest value error across four hundred and ninety-five protocols falls with the number of calibration units, at a log-log slope of about minus zero point four one, approaching the root-m rate. The mean target and the occupation target behave alike.
- ZH: 模拟里看到的正是这样。四百九十五个协议上的最大价值误差，随着校准对象的数目下降，对数坐标下的斜率大约是负零点四一，逼近根号 m 的速率。均值目标和占用时间目标的表现是一样的。

### 屏幕文字 On-screen text

| EN | ZH |
|---|---|
| Simulation: family-uniform error over 495 protocols vs calibration size | 模拟：495 个协议上的一致误差随校准规模的变化 |
| Theorem 10 — covariance error becomes value error, uniformly over the family | 定理 10——协方差误差一致地转化为整个候选族上的价值误差 |
| Dense calibration: a small set of units observed over the whole horizon | 密集校准：少量对象，记录整条轨迹 |
| Proposition 8 — identification by augmentation is value-specific | 命题 8——通过补测恢复识别，且只需针对所评估的价值 |
| add Z₁ | 补测 Z₁ |
| add Z₂ | 补测 Z₂ |
| dimension of the invisible space (no stationarity assumed) | 不可见空间的维数（不假设平稳） |
| under stationarity, Z₁ alone already suffices (1 → 0); identification targets the value of B, not all of K | 若假设平稳，仅补测 Z₁ 已足够（1 → 0）；识别的对象是 B 的价值，而不是整个 K |
| m = 60 calibration units, W = Z + η | m = 60 个校准对象，W = Z + η |
| Their population law identifies K, hence the value of every candidate protocol. The question becomes statistical: how accurately, with finite m? | 它们的总体分布识别 K，从而识别每一个候选协议的价值。问题于是变成统计问题：有限的 m 下能估多准？ |
| sample covariance | 样本协方差 |
| subtract calibration noise | 扣除校准噪声 |
| eigenvalue floor (valid covariance) | 特征值下限（合法协方差） |
| rescale to a correlation matrix | 重新标准化为相关矩阵 |
| plug in, for every candidate S | 代入每一个候选 S |
| C depends on the target modulus, the target-variance floor, bounds on K, and the family's conditioning — not on S | C 只依赖目标的连续模、目标方差下界、K 的界和候选族的条件数——与 S 无关 |
| \\text{smooth target } g ∈ W^{1,2}(φ) | \\text{平滑目标 } g ∈ W^{1,2}(φ) |
| \\text{threshold target, worst case over all correlation matrices} | \\text{阈值目标，所有相关矩阵上的最坏情形} |
| \\text{threshold target at any fixed interior model } (\|r\| ≤ r_0 < 1) | \\text{阈值目标，任何固定的内部模型 }(\|r\| ≤ r_0 < 1) |
| Proposition 9: the square-root envelope is sharp — C_gc(1) − C_gc(1−δ) ~ e^{−c²/2}√(2δ)/2π.  Theorem 11: at a fixed model the plug-in values converge at the root-m rate. | 命题 9：平方根包络不可改进——C_gc(1) − C_gc(1−δ) ~ e^{−c²/2}√(2δ)/2π。定理 11：固定模型下插入估计以根号 m 速率收敛。 |
| Regret and resolution follow from one uniform error number ε_m — next chapter. | 选择损失与分辨率都由同一个一致误差 ε_m 决定——见下一章。 |
| calibration trajectories m | 校准轨迹数 m |
| temporal mean | 均值目标 |
| occupation above zero | 零阈值占用时间 |
| log–log slopes | 对数–对数斜率 |
| full range  | 全程  |
|    largest three m:  |    最大的三个 m： |
| fixed-model exponent: −1/2 | 固定模型下的理论指数：−1/2 |

### Read 段落（章下方的论文摘要）

**EN**

**Proposition 8 (augmentation).** If the row space of `B` lies in the row space of the augmented benchmark together with `h^\top`, the value of `B` is identified exactly; a non-zero derivative along an augmented-invisible direction certifies failure. **Calibration (Section 4).** With `m` densely observed units `W = Z + η`, the estimator is `\hat K = R_τ(\hat Σ − \hat R_0)` — deconvolution, eigenvalue floor, standardisation — and `\hat I_g(S) = F_g(S;\hat K)/V_g(\hat K)`. **Theorem 10.** Uniformly over a conditioned family, `\sup_S |\hat I_g(S) − I_g(S)| ≤ C\,\norm{\hat K − K}^{β}`, with `β = 1` for `g ∈ W^{1,2}(φ)`, `β = 1/2` for threshold targets globally (sharp, Proposition 9), and `β = 1` at any fixed interior model. **Theorem 11.** Hence the plug-in values converge at the root-`m` rate at a fixed model. Figure 2(a): uniform error over 495 four-action protocols, slopes −0.413 (mean) and −0.414 (occupation at 0); over the three largest `m`, −0.462 and −0.464.

**ZH**

**命题 8（补测）.** 若 `B` 的行空间落在补测后的基准与 `h^\top` 所张成的行空间内，则 `B` 的价值被精确识别；沿补测后仍不可见的方向导数非零，则证明失败。**校准（第 4 节）.** 用 `m` 个密集观测对象 `W = Z + η`，估计量为 `\hat K = R_τ(\hat Σ − \hat R_0)`——去噪、特征值下限、标准化——再取 `\hat I_g(S) = F_g(S;\hat K)/V_g(\hat K)`。**定理 10.** 在条件数受控的候选族上一致地有 `\sup_S |\hat I_g(S) − I_g(S)| ≤ C\,\norm{\hat K − K}^{β}`：`g ∈ W^{1,2}(φ)` 时 `β = 1`；阈值目标全局 `β = 1/2`（命题 9 证明其不可改进）；在任何固定内部模型上 `β = 1`。**定理 11.** 因而固定模型下的插入估计以根号 `m` 速率收敛。图 2(a)：495 个四动作协议上的一致误差，斜率 −0.413（均值）与 −0.414（零阈值占用时间）；最大的三个 `m` 上为 −0.462 与 −0.464。


---

## 6 · 分辨率 / Resolution

- 引题：第 4 节 · 推论 12 / Section 4 · Corollary 12
- 对应论文：§4 · Corollary 12, Figure 2b–d
- 画面：“≤ 2ε”的三行论证与误差条；四层嵌套候选类与 S*；图 2c,d 的损失与一致误差曲线；结论卡
- Visual: The three-line “≤ 2ε” argument with error brackets; four nested candidate classes and S*; Figure 2c,d regret and uniform-error curves; takeaway card

### 解说词 Narration

**ch6-b0**  (EN 23 s · ZH 24 s)

- EN: Here's where uniform error becomes a decision guarantee, and it takes only three lines. The true best protocol might be underestimated, by at most epsilon. The one we select might be overestimated, by at most epsilon. So selecting from estimated values costs us at most two epsilon. Value gaps smaller than that simply cannot be resolved.
- ZH: 接下来，一致误差就变成了一个决策保证，而且只需要三行。真正最优的协议可能被低估，但最多低估 epsilon；我们选中的那个可能被高估，也最多高估 epsilon。所以按估计值来选，损失最多是二倍 epsilon。比这个尺度更小的价值差距，根本分辨不出来。
  - ZH 朗读文本: 接下来，一致误差就变成了一个决策保证，而且只需要三行。真正最优的协议可能被低估，但最多低估 艾普西隆；我们选中的那个可能被高估，也最多高估 艾普西隆。所以按估计值来选，损失最多是二倍 艾普西隆。比这个尺度更小的价值差距，根本分辨不出来。

**ch6-b1**  (EN 18 s · ZH 18 s)

- EN: Candidate classes can be coarse, say contiguous versus dispersed, or fine, down to exact anchor sets. A coarse class may simply not contain the optimum. A fine class contains it, but is harder to resolve. The total loss is the class restriction plus two epsilon.
- ZH: 候选类可以很粗，比如只分“连续还是分散”；也可以很细，细到精确的锚点集合。粗的类可能根本不包含最优解；细的类包含它，却更难分辨。总的损失，就是类别限制的损失，再加上二倍 epsilon。
  - ZH 朗读文本: 候选类可以很粗，比如只分“连续还是分散”；也可以很细，细到精确的锚点集合。粗的类可能根本不包含最优解；细的类包含它，却更难分辨。总的损失，就是类别限制的损失，再加上二倍 艾普西隆。

**ch6-b2**  (EN 22 s · ZH 22 s)

- EN: In the experiment, the finest class already wins with just twenty-five calibration units, even though its uniform error is the largest. Ordering the leading candidates turns out to be easier than estimating every value. By a thousand units, the coarsest class is stuck at its restriction regret, because the better protocols lie outside it.
- ZH: 实验里，最细的类在只有二十五个校准对象的时候就已经领先了，尽管它的一致误差是最大的。把前几名排好，原来比估准每一个值要容易。到一千个对象的时候，最粗的类卡在了它的限制损失上，因为更好的协议根本不在这个类里。

**ch6-b3**  (EN 8 s · ZH 8 s)

- EN: So the granularity at which protocols can be optimised is set by the calibration data, not by the optimiser.
- ZH: 所以，协议能优化到多细的粒度，是由校准数据决定的，而不是由优化算法决定的。

### 屏幕文字 On-screen text

| EN | ZH |
|---|---|
| Simulation: regret and uniform error by candidate class (Figure 2c,d) | 模拟：各候选类的选择损失与一致误差（图 2c,d） |
| Coarse or fine candidate classes? | 候选类该粗还是细？ |
| Corollary 12 — from uniform error to a decision guarantee | 推论 12——从一致误差到决策保证 |
| true optimum | 真正最优 |
| true value of the selected | 被选中者的真实价值 |
| the true optimum may be underestimated by at most ε | 真正最优的协议最多被低估 ε |
| the selected protocol may be overestimated by at most ε | 被选中的协议最多被高估 ε |
| the selected one won on estimated values, so | 它是按估计值胜出的，所以 |
| \\text{with an approximate maximiser: } ≤ 2ε + η \\text{ (Corollary 12)} | \\text{若只是近似极大化：} ≤ 2ε + η \\text{（推论 12）} |
| No probability is used — it is an inequality. An estimated gap larger than 2ε certifies the same ordering in the population; gaps below 2ε cannot be resolved from the calibration data. | 这里没有用到概率——它只是一个不等式。估计出的差距若大于 2ε，则总体上的排序相同；小于 2ε 的差距无法从校准数据中分辨。 |
| layouts | 布局 |
| phase | 相位 |
| coarse bins | 粗分箱 |
| fine supports | 精确位置 |
| regret of the best estimated protocol in class ℓ | 第 ℓ 类中最优估计协议的损失 |
| class restriction | 类别限制 |
| calibration error | 校准误差 |
| A coarse class may not contain S* — no amount of calibration fixes that. A fine class contains it, but its many close values need more data to resolve: the restriction term falls and the error term rises as the class is refined. | 粗的类可能不包含 S*——再多校准也弥补不了。细的类包含它，但众多相近的值需要更多数据才能分辨：随着类别变细，限制项下降而误差项上升。 |
| true selection regret | 真实选择损失 |
| uniform error ε_ℓ | 一致误差 ε_ℓ |
| ordering the leading candidates is easier than estimating every value uniformly | 排好前几名，比一致地估准每一个值容易 |
| The granularity at which protocols can be optimised is set by the calibration data, not by the optimiser. | 协议能优化到多细的粒度，由校准数据决定，而不是由优化算法决定。 |
| — the reading rule for the real-data results in Chapter 8 | ——这也是第 8 章真实数据结果的解读规则 |

### Read 段落（章下方的论文摘要）

**EN**

**Corollary 12.** With `ε_m = \sup_S |\hat I_g(S) − I_g(S)|`, an empirical maximiser `\tilde S` with optimisation gap `η_m` satisfies `I_g(S^*) − I_g(\tilde S) ≤ 2ε_m + η_m`. For nested classes `Π^{(1)} ⊆ … ⊆ Π^{(L)}`, the regret of the best estimated protocol in class `ℓ` is at most the *class restriction* `I(S^*) − \max_{Π^{(ℓ)}} I` plus `2ε_ℓ`. Figure 2(b–d): across 7200 checks the realised regret never exceeds `2ε_m` (median ratio 0.024, maximum 0.37); regret falls at slope −0.94 while the envelope falls at −0.49; in four nested classes of sizes 2, 5, 75 and 568, the finest class has regret 0.037 at `m = 25` against 0.051 for the coarsest, and at `m = 1000` the coarsest is still at 0.037 while the finest reaches 0.006.

**ZH**

**推论 12.** 记 `ε_m = \sup_S |\hat I_g(S) − I_g(S)|`，则优化间隙为 `η_m` 的经验极大化解 `\tilde S` 满足 `I_g(S^*) − I_g(\tilde S) ≤ 2ε_m + η_m`。对嵌套类 `Π^{(1)} ⊆ … ⊆ Π^{(L)}`，第 `ℓ` 类中最优估计协议的损失不超过*类别限制* `I(S^*) − \max_{Π^{(ℓ)}} I` 加 `2ε_ℓ`。图 2(b–d)：7200 次检验中实际损失从未超过 `2ε_m`（比值中位数 0.024，最大 0.37）；损失以 −0.94 的斜率下降而包络以 −0.49 下降；在大小为 2、5、75、568 的四个嵌套类中，`m = 25` 时最细类的损失为 0.037，最粗类为 0.051；`m = 1000` 时最粗类仍停在 0.037，而最细类降到 0.006。


---

## 7 · 针对目标的观测设计 / Target-aware design

- 引题：第 5 节 · 命题 13、引理 14 / Section 5 · Proposition 13, Lemma 14
- 对应论文：§5 · Proposition 13, Lemma 14, Table 1
- 画面：后验轨迹带（WebGL）随贪心加点塌缩；候选目录与动作五元组；秩一收益公式；各候选的边际收益条；两种目标选出的时刻；表 1 的效率条
- Visual: Posterior band (WebGL) collapsing as greedy adds measurements; the catalogue and the action tuple; rank-one gain formula; per-candidate gain bars; times chosen by two targets; Table 1 efficiency bars

### 解说词 Narration

**ch7-b0**  (EN 13 s · ZH 15 s)

- EN: Once we have an estimated covariance, design becomes an optimisation problem under a cost budget. Each candidate action fixes a time, a window length, a repetition count, a noise level, and a cost.
- ZH: 有了协方差的估计，设计就变成了预算约束下的优化问题。每个候选动作规定了一个时刻、一个窗口长度、一个重复次数、一个噪声水平，还有一个成本。

**ch7-b1**  (EN 17 s · ZH 17 s)

- EN: Adding an action has an exact, rank-one marginal gain. The residual covariance shrinks by an outer product: the residual's covariance with the new measurement, divided by that measurement's conditional variance. For the mean target, the whole gain is a single fraction.
- ZH: 增加一个动作的边际收益，有一个精确的秩一形式。残差协方差减去一个外积：残差与新测量的协方差，除以新测量的条件方差。对均值目标来说，整个收益就是一个简单的分式。

**ch7-b2**  (EN 19 s · ZH 19 s)

- EN: Watch greedy selection at work. At each step it picks the largest gain per unit cost, and the posterior band collapses wherever we look. A time that is already well predicted earns only a small gain, automatically. There's no need for an extra rule against redundancy.
- ZH: 来看贪心选择是怎么运作的。每一步，它选单位成本收益最大的动作，后验带就在被观测的地方塌缩下去。一个已经被预测得很好的时刻，自动只能得到很小的收益，不需要额外加一条去冗余的规则。

**ch7-b3**  (EN 13 s · ZH 13 s)

- EN: Different targets choose different times, on the same process. The mean target and the occupation target disagree. The right sampling time is a property of the process and the target together.
- ZH: 同一个过程上，不同的目标会选出不同的时刻。均值目标和占用时间目标并不一致。合适的采样时间，是过程和目标共同决定的性质。

**ch7-b4**  (EN 25 s · ZH 24 s)

- EN: One caution before we move on. The objective is monotone, but it is not submodular: an action can become more valuable after another one is added. So the paper claims no approximation ratio for greedy. Instead it compares against exhaustive search over twenty-five instances, where greedy with one swap reaches at least ninety-eight point seven percent of the optimum.
- ZH: 往下走之前，提醒一点。这个目标函数是单调的，但不是次模的：一个动作在加入另一个动作之后，可能反而变得更有价值。所以论文不对贪心声称任何近似比，而是在二十五个实例上跟穷举搜索对照——贪心加一轮交换，至少能达到最优值的百分之九十八点七。

### 屏幕文字 On-screen text

| EN | ZH |
|---|---|
| Monotone, but not submodular — and how close greedy gets | 单调但不次模——以及贪心离最优有多近 |
| Target-aware observation design under a cost budget | 预算约束下针对目标的观测设计 |
| posterior sample paths given the measurements so far | 给定已有测量的后验样本轨迹 |
| one true trajectory | 一条真实轨迹 |
| catalogue: 16 candidate point actions, noise r = 0.3, cost 1 each, budget 4 · target weights ω rise linearly over the horizon (recency-weighted) | 候选目录：16 个点测量动作，噪声 r = 0.3，成本各 1，预算 4 · 目标权重 ω 随时间线性增大（偏重近期） |
| \\text{an action } a = (ℓ_a, r_a, c_a) | \\text{一个动作 } a = (ℓ_a, r_a, c_a) |
| timing & support | 时刻与支撑 |
| repetition & noise | 重复次数与噪声 |
| cost | 成本 |
| feasible set | 可行集 |
| exact rank-one gain (Proposition 13) | 精确的秩一边际收益（命题 13） |
| P_S = K − Q_S is the residual covariance: what is still unexplained | P_S = K − Q_S 是残差协方差：还没被解释的部分 |
| marginal gain of each candidate, given the measurements so far | 给定已有测量，每个候选的边际收益 |
| \\text{mean target} \\quad g(z) = z | \\text{均值目标} \\quad g(z) = z |
| \\text{occupation target} \\quad g(z) = \\mathbf{1}\\{z > 1\\} | \\text{占用时间目标} \\quad g(z) = \\mathbf{1}\\{z > 1\\} |
| same K, same catalogue, same budget — the two targets select different times | 同一个 K、同一个候选目录、同样的预算——两种目标选出不同的时刻 |
| latent-state mutual information (target-free) | 潜在状态互信息（与目标无关） |
| integrated posterior variance (target-free) | 积分后验方差（与目标无关） |
| actual-noise linear target | 带真实噪声的线性目标 |
| noiseless kernel quadrature | 无噪声核求积 |
| target-aware greedy | 针对目标的贪心 |
| target-aware greedy + one swap | 针对目标的贪心 + 一轮交换 |
| relative efficiency I(S)/I(S*) over 25 enumerated instances — minimum (bar), mean, median | 25 个穷举实例上的相对效率 I(S)/I(S*)——最小值（条形）、均值、中位数 |
| mean  | 均值  |
|   median  |   中位数  |
| exhaustive optimum | 穷举最优 |

### Read 段落（章下方的论文摘要）

**EN**

**Proposition 13 (rank-one marginal gains).** With residual covariance `P_S = K − Q_S = Cov(Z \mid Y_S)`, adding action `a = (ℓ_a, r_a, c_a)` gives `v = P_Sℓ_a`, `s = ℓ_a^\top P_S ℓ_a + r_a`, `Q_{S ∪ a} = Q_S + v v^\top / s`, and the exact gain Δ_g(a \mid S) = \sum_{j,k} ω_j ω_k \,[\, C_g(Q_{jk} + v_j v_k / s) − C_g(Q_{jk}) \,]; for the mean target `Δ = \frac{(ω^\top P_S ℓ_a)^2}{ℓ_a^\top P_S ℓ_a + r_a}`. **Lemma 14.** `F_g` is monotone under nested protocols, but not submodular (R² subset selection is not). Table 1: over 25 enumerated instances, relative efficiency against the exhaustive target-specific optimum has minimum 0.601 (latent mutual information), 0.802 (integrated posterior variance), 0.767 (linear target with actual noise), 0.231 (noiseless kernel quadrature), 0.912 (target-aware greedy) and 0.987 (greedy + one swap), the last while evaluating 9–23% of the enumerated sets.

**ZH**

**命题 13（秩一边际收益）.** 记残差协方差 `P_S = K − Q_S = Cov(Z \mid Y_S)`，加入动作 `a = (ℓ_a, r_a, c_a)` 时 `v = P_Sℓ_a`、`s = ℓ_a^\top P_S ℓ_a + r_a`、`Q_{S ∪ a} = Q_S + v v^\top / s`，精确收益为 Δ_g(a \mid S) = \sum_{j,k} ω_j ω_k \,[\, C_g(Q_{jk} + v_j v_k / s) − C_g(Q_{jk}) \,]；均值目标下 `Δ = \frac{(ω^\top P_S ℓ_a)^2}{ℓ_a^\top P_S ℓ_a + r_a}`。**引理 14.** `F_g` 对嵌套协议单调，但不是次模的（R² 子集选择本身就不是）。表 1：25 个穷举实例上，相对穷举最优的效率最小值依次为 0.601（潜在互信息）、0.802（积分后验方差）、0.767（带真实噪声的线性目标）、0.231（无噪声核求积）、0.912（针对目标的贪心）、0.987（贪心加一轮交换），最后一项只评估了穷举集合的 9–23%。


---

## 8 · 睡眠与房颤 / Sleep and atrial fibrillation

- 引题：第 6.4 节 · 图 3–4 / Section 6.4 · Figures 3–4
- 对应论文：§6.4 · Figures 3–4
- 画面：两张数据卡（Sleep-EDF / LTAF）；REM 分散 vs 连续柱图；房颤负担折线；图 4 的支持稳定性曲线与重抽样区间
- Visual: Two dataset cards; REM dispersed-vs-contiguous bars; AF-burden lines; Figure 4 support-stability curves and resampling ranges

### 解说词 Narration

**ch8-b0**  (EN 20 s · ZH 22 s)

- EN: Now to real data. Two fully annotated data sets let us reconstruct any protocol from the complete record and test it on held-out units. Sleep-EDF, with one hundred and ninety-seven hypnograms from one hundred subjects. And Long-Term AF, with eighty-four rhythm records of roughly twenty-four hours each.
- ZH: 现在来看真实数据。有两个带完整标注的数据集，让我们可以从完整记录里重建任何一种协议，再在留出的对象上检验：一个是 Sleep-EDF，一百名受试者的一百九十七段整夜分期；另一个是长程房颤数据库，八十四段大约二十四小时的心律记录。

**ch8-b1**  (EN 24 s · ZH 22 s)

- EN: Take sleep first, predicting the REM fraction. At matched budgets, dispersing the scored epochs across the night beats a contiguous block: zero point six eight versus zero point six five at four epochs, and zero point seven four versus zero point six six at sixteen. I should add that the pooled contrast is heterogeneous across the two source studies.
- ZH: 先看睡眠，预测快速眼动的比例。在相同预算下，把评分片段分散到整夜，要好过连续的一块：四个片段时是零点六八对零点六五，十六个片段时是零点七四对零点六六。不过要补一句，这个合并后的对比，在两个来源研究之间并不均匀。

**ch8-b2**  (EN 19 s · ZH 18 s)

- EN: Then atrial-fibrillation burden. In the Long-Term AF analysis, four dispersed fifteen-minute windows, one hour in total, reached an R squared of zero point nine seven. That outperforms every contiguous block that was evaluated, up to eight hours. And four dispersed hours reached zero point nine nine eight.
- ZH: 再看房颤负担。在长程房颤分析里，四个分散的十五分钟窗口，总共一小时，就达到了零点九七的 R 方。这超过了所评估的每一个连续块，最长到八小时的那个。而分散的四小时，达到了零点九九八。

**ch8-b3**  (EN 24 s · ZH 30 s)

- EN: The learned Sleep supports, however, tell a more cautious story. They did not show stable held-out advantages across subsamples. The target-aware support's advantage over fixed dispersion is minus zero point zero four on the original sample, and across a thousand subject resamples its range spans zero. In this analysis, about eighty training subjects supported clearer broad-layout comparisons than exact anchor selection.
- ZH: 不过，学得的睡眠位置讲的是一个更谨慎的故事。它们在重抽样之间并没有显示出稳定的留出优势。针对目标学出的位置，相对固定分散方案的优势，在原样本上是负零点零四；在一千次受试者重抽样里，范围跨过了零。在这项分析里，大约八十个训练受试者，支持的是对粗布局的清楚比较，而不是对精确锚点的选择。

### 屏幕文字 On-screen text

| EN | ZH |
|---|---|
| Learned REM supports at N = 16: no stable held-out advantage (Figure 4) | N = 16 时学得的 REM 位置：没有稳定的留出优势（图 4） |
| Long-Term AF burden: dispersed windows vs a contiguous block (Figure 3b) | 长程房颤负担：分散窗口 vs 连续块（图 3b） |
| Sleep-EDF, REM fraction: dispersed vs contiguous at matched budgets (Figure 3a) | Sleep-EDF，REM 比例：相同预算下分散 vs 连续（图 3a） |
| Two fully annotated data sets: any protocol can be reconstructed from the complete record | 两个完整标注的数据集：任何协议都能从完整记录里重建 |
| hypnograms | 整夜分期记录 |
| subjects | 受试者 |
| valid annotated hours | 有效标注小时数 |
| relative-time anchors p | 相对时间锚点 p |
| targets | 目标 |
| budgets N (30-s epochs) | 预算 N（30 秒片段） |
| evaluation | 评估 |
| 5-fold, subject-disjoint, pooled R² | 五折、受试者不交叠、汇总 R² |
| one night, scored every 30 s (schematic) | 一夜，每 30 秒评分一次（示意） |
| records | 记录数 |
| reviewed rhythm hours | 审阅过的心律小时数 |
| median record length | 记录长度中位数 |
| relative-time bins p | 相对时间分箱 p |
| target | 目标 |
| AF burden (fraction of time in AF) | 房颤负担（房颤时间占比） |
| budgets N (15-min windows) | 预算 N（15 分钟窗口） |
| 5-fold by record, pooled R² | 按记录五折、汇总 R² |
| 24 h of rhythm annotation: AF (red) and other rhythm (schematic) | 24 小时心律标注：房颤（红）与其他心律（示意） |
| only expert annotation files are used — no raw PSG or ECG signal; the question is which segments to annotate, not end-to-end signal decoding | 只使用专家标注文件——不涉及原始 PSG 或 ECG 信号；问题是“该标注哪些片段”，而不是端到端信号解码 |
| cross-fitted R² | 交叉拟合 R² |
| centred contiguous block | 居中连续块 |
| uniformly dispersed | 均匀分散 |
| matched budget, different layout | 相同预算，不同布局 |
| target-aware and learned kernel-quadrature supports: next beat but one | 针对目标与学得核求积的位置：见本章末 |
| total observed time (N windows of 15 min; 24-h record) | 总观测时长（N 个 15 分钟窗口；24 小时记录） |
| contiguous | 连续 |
| dispersed | 分散 |
| four dispersed 15-min windows outperformed every contiguous block evaluated | 四个分散的 15 分钟窗口优于所评估的每一个连续块 |
| AF episodes cluster within the day: one block may land inside or outside a cluster. Both layouts need the same 24-h wear time; what is saved is the signal that must be reviewed. | 房颤发作在一天内成簇：一整块可能恰好落在簇内或簇外。两种布局的佩戴时间相同（24 小时）；省下的是需要人工审阅的信号时长。 |
| training subjects used to select the support | 用于选择位置的训练受试者数 |
| held-out R² advantage of the target-aware support | 针对目标位置的留出 R² 优势 |
| vs learned kernel quadrature | 对比学得核求积 |
| vs fixed dispersion | 对比固定分散 |
| vs kernel quadrature | 对比核求积 |
| median  | 中位数  |
| tick: original sample · dot: subsample median · bar: 2.5–97.5 percentile range | 竖线：原样本 · 圆点：重抽样中位数 · 横条：2.5–97.5 分位区间 |

### Read 段落（章下方的论文摘要）

**EN**

Both analyses use only expert annotation files — 197 Sleep-EDF hypnograms (100 subjects, 3818 valid hours, `p = 128` relative-time anchors) and 84 Long-Term AF records (1934 hours, `p = 96` bins) — with subject-disjoint five-fold cross-fitting and pooled held-out R². **Sleep (REM):** centred-contiguous 0.648 / 0.659 / 0.863 versus uniformly dispersed 0.682 / 0.738 / 0.881 at `N = 4, 16, 64`; 9 of 15 target–budget cells have a dispersed-minus-contiguous percentile range above zero, but SC and ST source studies disagree (10 above, 5 below, 15 spanning zero of 30 cells). **AF burden:** contiguous versus dispersed 0.696 vs 0.971 at one hour (`N = 4` windows) and 0.851 vs 0.998 at four hours; the paired range is positive at every multi-window budget. **Learned supports:** the target-aware REM support at `N = 16` scores 0.694 against 0.650 (learned kernel quadrature) and 0.738 (fixed dispersion); over 1000 study-stratified 80% subsamples the differences have medians −0.014 and −0.012 with ranges [−0.102, +0.078] and [−0.126, +0.097]. Coarse layout contrasts were clearer than exact learned anchors — the resolution pattern of Figure 2 seen in real data.

**ZH**

两项分析只使用专家标注文件——197 段 Sleep-EDF 分期（100 名受试者、3818 有效小时、`p = 128` 个相对时间锚点）与 84 段长程房颤记录（1934 小时、`p = 96` 个分箱）——采用受试者不交叠的五折交叉拟合与汇总留出 R²。**睡眠（REM）：** `N = 4, 16, 64` 时居中连续块为 0.648 / 0.659 / 0.863，均匀分散为 0.682 / 0.738 / 0.881；15 个目标–预算格中有 9 个“分散减连续”的分位数区间高于零，但 SC 与 ST 两个来源研究并不一致（30 格中 10 个高于零、5 个低于零、15 个跨零）。**房颤负担：** 一小时（`N = 4` 个窗口）连续对分散为 0.696 对 0.971，四小时为 0.851 对 0.998；每个多窗口预算上配对区间都为正。**学得的位置：** `N = 16` 时针对目标的 REM 位置得分 0.694，对比学得核求积 0.650 与固定分散 0.738；在 1000 次按研究分层的 80% 重抽样中，差值中位数为 −0.014 与 −0.012，区间为 [−0.102, +0.078] 与 [−0.126, +0.097]。粗布局的对比比精确学得的锚点更清楚——这正是图 2 的分辨率规律在真实数据上的体现。


---

## 9 · 先识别，再优化 / Identification before optimisation

- 引题：第 8 节 / Section 8
- 对应论文：§8
- 画面：识别 / 校准 / 设计三列；一句话原则；发布信息卡
- Visual: Three columns identification / calibration / design; the one-sentence principle; release card

### 解说词 Narration

**ch9-b0**  (EN 19 s · ZH 15 s)

- EN: Let me close with three questions, in the order they have to be answered. Is the value in the data at all? That's identification. If it is, how accurately can it be estimated? That's calibration. And at that accuracy, which candidates can actually be told apart? That's design.
- ZH: 最后用三个问题来收尾，而且要按顺序问。答案到底在不在数据里？这是识别。如果在，能估多准？这是校准。在这个精度下，哪些候选真的能分得开？这是设计。

**ch9-b1**  (EN 15 s · ZH 19 s)

- EN: A benchmark fixes a statistical experiment, not just a sample. So put identification before optimisation. Evaluate undeployed protocols only once the data can actually determine their value, and optimise only at the granularity the calibration data support.
- ZH: 一个基准数据集固定的，是一个统计实验，而不只是一个样本。所以，先识别，再优化。只有当数据真的能确定一个协议的价值时，才去评估那些没有实施过的协议；也只在校准数据支持的粒度上做优化。

**ch9-b2**  (EN 13 s · ZH 12 s)

- EN: The manuscript, the code and experiment package, and all the experiment scripts are released together. And every number you heard in this tour is read from the same files that generate the paper.
- ZH: 论文、代码与实验软件包、以及全部实验脚本，是一起发布的。你在这次导览里听到的每一个数字，都读取自生成论文的同一批文件。

### 屏幕文字 On-screen text

| EN | ZH |
|---|---|
| Three questions, in order | 三个问题，按顺序问 |
| Identification | 识别 |
| Is the value in the data? | 答案在数据里吗？ |
| Theorems 3 & 5, Proposition 6: a single-protocol benchmark need not determine an undeployed protocol's value — even with infinite data. Proposition 8: targeted augmentation restores it, value by value. | 定理 3、5，命题 6：单一协议的基准数据未必能确定未实施协议的价值——即使数据无穷多。命题 8：定向补测可以逐个价值地恢复它。 |
| Section 3 | 第 3 节 |
| Calibration | 校准 |
| How accurately can it be estimated? | 能估多准？ |
| Theorems 10–11: dense calibration gives uniform value error C‖K̂−K‖^β, root-m at a fixed model. Corollary 12: regret ≤ 2ε — a resolution below which protocols cannot be told apart. | 定理 10–11：密集校准给出一致的价值误差 C‖K̂−K‖^β，固定模型下为根号 m 速率。推论 12：损失 ≤ 2ε——低于这个分辨率的协议分不开。 |
| Section 4 | 第 4 节 |
| Design | 设计 |
| Which candidates can be told apart? | 哪些候选分得开？ |
| Proposition 13: exact rank-one gains for cost-constrained, target-aware search; monotone but not submodular. Real data: coarse layout contrasts were clearer than exact learned anchors. | 命题 13：预算约束下针对目标搜索的精确秩一收益；单调但不次模。真实数据：粗布局的差异比精确学得的锚点更清楚。 |
| Sections 5–6 | 第 5–6 节 |
| the order cannot be reversed: if the answer is not in the data, no optimiser helps; if it is but ε is large, fine optimisation overfits | 顺序不能颠倒：答案不在数据里，优化算法无济于事；在数据里但 ε 很大，精细优化就会过拟合 |
| A benchmark fixes a statistical experiment, not just a sample. | 一个基准数据集固定的是一个统计实验，而不只是一个样本。 |
| Identification before optimisation: evaluate undeployed protocols only once the data can determine their value, and optimise only at the granularity the calibration data support. | 先识别，再优化：只有当数据能够确定协议的价值时才评估未实施的协议；也只在校准数据支持的粒度上做优化。 |
| practical pattern: a large routine cohort + a small, intensively observed calibration subset chosen with future protocol decisions in mind | 实践模式：大规模常规队列 + 为未来协议决策而设计的小规模密集校准子集 |
| Xizhe Zhang · manuscript, code and experiment package released together | 张锡哲 · 论文、代码与实验软件包一起发布 |
| every number in this tour is read from paper/numbers.tex and results/ — the same files that generate the paper | 本导览中的每一个数字都读取自 paper/numbers.tex 与 results/——生成论文的同一批文件 |

### Read 段落（章下方的论文摘要）

**EN**

The information required to evaluate a protocol is not the information required to recover the full latent dependence; it is the information needed to eliminate the ambiguity that changes the values under consideration. Finite calibration then sets the resolution at which values can be compared, and the design problem should be posed at that granularity. Practically: pair a large routine cohort with a small, intensively observed calibration subset chosen with future protocol decisions in mind; compare coarse layouts first; refine to exact placements only when the calibration data can resolve the differences. Limitations stated in the paper: static (non-adaptive) protocols, a scalar Gaussian latent process, annotated states rather than raw signals, and no approximation ratio for greedy search.

**ZH**

评估一个协议所需的信息，不等于恢复完整潜在依赖结构所需的信息；它只是消除那些会改变所考虑价值的歧义所需的信息。有限的校准数据随后决定了价值可以比较到的分辨率，设计问题应当在这个粒度上提出。实践上：用一个大规模常规队列，配上一个为未来协议决策而设计的小规模密集校准子集；先比较粗的布局；只有当校准数据能分辨差异时，才细化到精确位置。论文自述的限制：静态（非自适应）协议、标量高斯潜在过程、使用已标注状态而非原始信号、不对贪心声称近似比。


---

# 四分钟导览的解说词（narration/explainer.json）


## E0 · 问题 / The question

- 引题：序 / Prologue

**e0-b0**  (EN 15 s · ZH 17 s)

- EN: Normally, a benchmark compares learning algorithms while holding the observation protocol fixed. This paper changes the question. Can data collected under one protocol tell us how well another protocol, one that was never used, would predict the target?
- ZH: 通常，一个基准数据集是固定观测协议，然后比较不同的学习算法。这篇论文把问题换了一层：一种协议采集的数据，能不能告诉我们，另一种从来没用过的协议，会把目标预测到多好？

**e0-b1**  (EN 12 s · ZH 13 s)

- EN: And the answer is: not always. Even with infinitely many units, the benchmark may leave some temporal dependence unresolved, and that dependence can be exactly what decides the alternative protocol's value.
- ZH: 答案是：并不总是可以。哪怕样本趋于无穷，基准数据仍然可能留下某些没有确定的时间依赖，而恰恰是这些依赖，决定了替代协议的价值。

**Technical detail (EN)**

Predictive performance in temporal learning depends on the learner *and* on the observation protocol that produced the data. The paper asks whether the joint measurement–target law under the **realised** protocol determines the predictive value of an **undeployed** alternative — counterfactual protocol evaluation (Section 1).

**技术细节（ZH）**

时间序列学习的预测性能既取决于模型，也取决于产生数据的观测协议。论文问：**已实施**协议下测量与目标的联合分布，能否确定一个**从未实施**的替代协议的预测价值——即反事实协议评估（第 1 节）。


## E1 · 同一基准，两个答案 / Same benchmark, two answers

- 引题：定理 3 · 最小数学例子 / Theorem 3 · minimal mathematical example

**e1-b0**  (EN 18 s · ZH 14 s)

- EN: Let's see the smallest case. Four time points. Protocol A observes only the first one, and the target is the average over all four. The benchmark law is then determined by just three quantities: the measurement variance, its covariance with the target, and the target variance.
- ZH: 来看最小的例子。四个时间点，协议 A 只观测第一个，目标是四个点的平均。这时基准分布只由三个量决定：测量方差、测量与目标的协方差，以及目标方差。

**e1-b1**  (EN 10 s · ZH 10 s)

- EN: Now we construct two latent temporal models that agree on all three quantities. The benchmark data are therefore exactly the same in the two worlds.
- ZH: 现在，我们构造两种潜在时间模型，让这三个量完全一致。于是，两个世界里的基准数据分布就完全相同。

**e1-b2**  (EN 13 s · ZH 13 s)

- EN: Protocol B observes two other points. Its value is 0.682 in one world and 0.827 in the other. The same benchmark supports two different answers.
  - EN 朗读文本: Protocol B observes two other points. Its value is zero point six eight two in one world, and zero point eight two seven in the other. The same benchmark supports two different answers.
- ZH: 协议 B 改为观测另外两个点。它的价值在一个世界里是 0.682，在另一个世界里是 0.827。同一份基准数据，支持两个不同的答案。
  - ZH 朗读文本: 协议 B 改为观测另外两个点。它的价值在一个世界里是零点六八二，在另一个世界里是零点八二七。同一份基准数据，支持两个不同的答案。

**e1-b3**  (EN 10 s · ZH 12 s)

- EN: What actually changes is the redundancy between B's two measurements. Protocol A never observes that relation, so collecting more units under A cannot recover it.
- ZH: 真正在变的，是协议 B 两个测量之间的冗余程度。协议 A 从来没有观测过这种关系，所以继续在 A 下扩大样本，也无法把它恢复出来。

**Technical detail (EN)**

**Theorem 3.** For a stationary standardised process on `p` points, a uniform mean target and a one-point benchmark, identification holds for `p ≤ 3` and fails from `p = 4`, where the invisible direction is the lag perturbation `δ = (0, 1, −2, 1)`. With `ρ_0(u) = e^{−u}` and `ε = 0.1321`, both worlds give `Var(Y_A) = 1`, `Cov(Y_A, Θ) = 0.388250`, `Var(Θ) = 0.428012`, yet `I(B;K_{+}) = 0.6817` and `I(B;K_{−}) = 0.8274` (Figure 1). Only `Cov(Z_1, Z_2)` moves along `δ`.

**技术细节（ZH）**

**定理 3.** 对 `p` 个格点上的平稳标准化过程、均匀均值目标和单点基准，`p ≤ 3` 时可识别，`p = 4` 起失效，不可见方向是滞后扰动 `δ = (0, 1, −2, 1)`。取 `ρ_0(u) = e^{−u}`、`ε = 0.1321`：两个世界的 `Var(Y_A) = 1`、`Cov(Y_A, Θ) = 0.388250`、`Var(Θ) = 0.428012` 完全一致，而 `I(B;K_{+}) = 0.6817`、`I(B;K_{−}) = 0.8274`（图 1）。沿 `δ` 移动时只有 `Cov(Z_1, Z_2)` 在变。


## E2 · 价值特定的识别 / Value-specific identification

- 引题：定义 1 · 定理 5 / Definition 1 · Theorem 5

**e2-b0**  (EN 13 s · ZH 11 s)

- EN: So what is a protocol's value, precisely? It is the population R squared of the best possible predictor of the fixed trajectory-level target, using that protocol's measurements.
- ZH: 那么，协议价值到底是什么？它是理想预测器只利用该协议的测量时，对固定的轨迹级目标所能达到的总体 R 方。

**e2-b1**  (EN 8 s · ZH 10 s)

- EN: And it is identified only when every latent model compatible with the realised benchmark assigns the alternative the same value.
- ZH: 而只有当所有与已实施基准相容的潜在模型，都赋予替代协议同一个价值时，这个价值才算被识别。

**e2-b2**  (EN 11 s · ZH 12 s)

- EN: Notice that this is value-specific. We don't need to recover the full latent covariance. We only need to remove the ambiguity that changes the value being evaluated.
- ZH: 注意，这种识别是针对价值本身的。我们不需要恢复完整的潜在协方差，只需要消除会改变当前所评估价值的那部分歧义。

**Technical detail (EN)**

**Definition 1.** `I(S) = \frac{Var(E[Θ \mid Y_S])}{Var(Θ)}`, the population R² of the optimal predictor; under the Gaussian model `I_g(S;K) = \frac{F_g(S;K)}{V_g(K)}` (Proposition 2). **Criterion.** `I_g(B; ·)` is identified from the benchmark law iff it is constant over the observational-equivalence class `K_A(P_A)`. **Theorem 5.** A symmetric zero-diagonal `Δ` with `AΔA^\top = 0`, `AΔh = 0`, `h^\topΔh = 0` is invisible; a non-zero directional derivative `D_B(Δ;K_0)` certifies non-identification. Rank–nullity: `O(d^2)` constraints against `O(p^2)` parameters.

**技术细节（ZH）**

**定义 1.** `I(S) = \frac{Var(E[Θ \mid Y_S])}{Var(Θ)}`，最优预测器的总体 R²；高斯模型下 `I_g(S;K) = \frac{F_g(S;K)}{V_g(K)}`（命题 2）。**识别准则.** `I_g(B; ·)` 可从基准分布识别，当且仅当它在观测等价类 `K_A(P_A)` 上为常数。**定理 5.** 满足 `AΔA^\top = 0`、`AΔh = 0`、`h^\topΔh = 0` 的对称零对角 `Δ` 不可见；方向导数 `D_B(Δ;K_0)` 非零即证明不可识别。秩–零化度：`O(d^2)` 个约束对 `O(p^2)` 个参数。


## E3 · 哪些补充数据能解决问题 / What additional data resolve

- 引题：命题 8 · 定理 10–11 / Proposition 8 · Theorems 10–11

**e3-b0**  (EN 9 s · ZH 12 s)

- EN: More observations under the same protocol simply repeat the same information. What helps is a measurement that cuts through the invisible, value-changing directions.
- ZH: 在同一协议下继续增加观测，只是在重复同一组信息。真正有帮助的，是能够切断那些不可见、而且会改变价值的方向的测量。

**e3-b1**  (EN 10 s · ZH 13 s)

- EN: For one specified alternative, targeted augmentation may already be enough. For a whole family of alternatives, a small, densely observed calibration subset provides a broader route.
- ZH: 如果只评估一个指定的替代协议，定向补测可能就已经够了。如果要评估整个候选族，一个小规模、但观测密集的校准子集，提供了更宽的路径。

**e3-b2**  (EN 11 s · ZH 13 s)

- EN: The paper shows how covariance-calibration error propagates into protocol-value error. At a fixed, well-conditioned model, the error falls at the usual inverse-square-root rate as the calibration sample grows.
- ZH: 论文给出了协方差的校准误差，是怎么传递成协议价值误差的。在固定、条件良好的模型下，随着校准样本增加，误差按通常的平方根速率下降。

**Technical detail (EN)**

**Proposition 8.** If the row space of `B` lies in that of the augmented benchmark with `h^\top`, the value is identified exactly; in the four-point example without stationarity, adding `Z_1` shrinks the invisible space from 4 to 2 dimensions and adding `Z_2` removes it. **Calibration.** With `m` dense units `W = Z + η`, `\hat K = R_τ(\hat Σ − \hat R_0)` and `\hat I_g(S) = F_g(S;\hat K)/V_g(\hat K)`. **Theorem 10.** `\sup_S |\hat I_g(S) − I_g(S)| ≤ C\,\norm{\hat K − K}^{β}`, `β = 1` for smooth targets and at fixed interior threshold models, `β = 1/2` for threshold targets globally. **Theorem 11.** Root-`m` at a fixed model; Figure 2(a): slopes −0.413 / −0.414 over 495 protocols.

**技术细节（ZH）**

**命题 8.** 若 `B` 的行空间落在补测后的基准与 `h^\top` 张成的行空间内，价值被精确识别；不假设平稳的四点例子中，补测 `Z_1` 把不可见空间从 4 维缩到 2 维，再补 `Z_2` 则完全消除。**校准.** 用 `m` 个密集对象 `W = Z + η`，`\hat K = R_τ(\hat Σ − \hat R_0)`，`\hat I_g(S) = F_g(S;\hat K)/V_g(\hat K)`。**定理 10.** `\sup_S |\hat I_g(S) − I_g(S)| ≤ C\,\norm{\hat K − K}^{β}`：平滑目标与固定内部阈值模型 `β = 1`，阈值目标全局 `β = 1/2`。**定理 11.** 固定模型下根号 `m` 速率；图 2(a)：495 个协议上斜率 −0.413 / −0.414。


## E4 · 先有分辨率，再谈优化 / Resolution before optimisation

- 引题：推论 12 / Corollary 12

**e4-b0**  (EN 12 s · ZH 16 s)

- EN: Calibration error sets a comparison scale. If every estimated value is within epsilon of its truth, then selecting the empirical best loses at most about two epsilon, apart from optimisation error.
- ZH: 校准误差给出了协议比较的尺度。如果每个估计价值与真实价值的偏差都不超过 epsilon，那么按估计值选出的最优协议，损失至多约为二倍 epsilon，再加上优化误差。
  - ZH 朗读文本: 校准误差给出了协议比较的尺度。如果每个估计价值与真实价值的偏差都不超过艾普西隆，那么按估计值选出的最优协议，损失至多约为二倍艾普西隆，再加上优化误差。

**e4-b1**  (EN 11 s · ZH 12 s)

- EN: A fine candidate class may well contain better protocols. But exact locations are meaningful only when the data can resolve the small gaps among the leading candidates.
- ZH: 更细的候选类，很可能包含更好的协议。但只有当数据能够分辨领先候选之间的微小差距时，精确的位置才有可信的意义。

**e4-b2**  (EN 8 s · ZH 8 s)

- EN: So the useful granularity of optimisation is set by the calibration data, not by how sophisticated the optimiser is.
- ZH: 所以，优化能够细化到什么程度，是由校准数据决定的，而不是由优化器有多复杂决定的。

**Technical detail (EN)**

**Corollary 12.** With `ε_m = \sup_S |\hat I_g(S) − I_g(S)|`, an empirical maximiser with optimisation gap `η_m` satisfies `I_g(S^*) − I_g(\tilde S) ≤ 2ε_m + η_m`; for nested classes the regret splits into class restriction plus `2ε_ℓ`. Figure 2(b–d): over 7200 checks the realised regret never exceeds `2ε_m`; among classes of sizes 2, 5, 75 and 568 the finest has regret 0.037 at `m = 25` versus 0.051 for the coarsest, and at `m = 1000` the coarsest is still at 0.037 while the finest reaches 0.006.

**技术细节（ZH）**

**推论 12.** 记 `ε_m = \sup_S |\hat I_g(S) − I_g(S)|`，优化间隙为 `η_m` 的经验极大化解满足 `I_g(S^*) − I_g(\tilde S) ≤ 2ε_m + η_m`；嵌套类下损失分解为类别限制加 `2ε_ℓ`。图 2(b–d)：7200 次检验中实际损失从未超过 `2ε_m`；大小为 2、5、75、568 的四类中，`m = 25` 时最细类损失 0.037、最粗类 0.051；`m = 1000` 时最粗类仍为 0.037，最细类降到 0.006。


## E5 · 设计与证据 / Design and evidence

- 引题：命题 13 · Sleep-EDF · 长程房颤 / Proposition 13 · Sleep-EDF · Long-Term AF

**e5-b0**  (EN 10 s · ZH 12 s)

- EN: Once values are identified and estimable, the paper derives the exact gain from adding a measurement, under constraints on timing, support, repetition, noise, and cost.
- ZH: 当候选价值已经可识别、也可以估计之后，论文推导了在时刻、窗口、重复、噪声与成本的约束下，新增一次测量的精确收益。

**e5-b1**  (EN 11 s · ZH 11 s)

- EN: The best schedule depends on the target. A protocol that covers the trajectory well need not be the best one for predicting a particular aggregate of that trajectory.
- ZH: 最佳的观测方案取决于目标。把整条轨迹覆盖得很好的协议，不一定最适合预测这条轨迹的某个特定聚合量。

**e5-b2**  (EN 13 s · ZH 15 s)

- EN: In Sleep-EDF, dispersed scoring often outperformed a contiguous block in the pooled analyses, although the two source studies were heterogeneous. In Long-Term AF, dispersed multi-window observation outperformed the contiguous layouts that were evaluated.
- ZH: 在 Sleep-EDF 里，分散评分在合并分析中常常优于连续块，不过两个来源研究之间存在异质性。在长程房颤数据里，分散的多窗口观测，优于所评估的那些连续布局。

**e5-b3**  (EN 12 s · ZH 14 s)

- EN: Exact learned Sleep locations, however, were less stable across subsamples. The data distinguished broad layouts more clearly than fine anchor positions. That is the same resolution pattern the theory predicts.
- ZH: 不过，精确学习出来的睡眠位置，在重抽样之间稳定性较弱。数据对粗粒度布局的区分，比对精确锚点清楚得多。这正是理论所刻画的那种分辨率规律。

**Technical detail (EN)**

**Proposition 13.** Adding action `a` gives `Q_{S ∪ a} = Q_S + v v^\top / s` with `v = P_Sℓ_a`, `s = ℓ_a^\top P_S ℓ_a + r_a`; for the mean target `Δ = \frac{(ω^\top P_S ℓ_a)^2}{s}`. **Lemma 14:** monotone, not submodular; Table 1: greedy + one swap reaches ≥ 0.987 of the exhaustive optimum over 25 instances. **Real data.** Sleep (REM): contiguous 0.648 / 0.659 / 0.863 vs dispersed 0.682 / 0.738 / 0.881 at `N = 4, 16, 64`, heterogeneous across SC/ST. AF burden: 0.696 vs 0.971 at one hour (four 15-min windows), 0.851 vs 0.998 at four hours. Learned REM support at `N = 16`: 0.694 vs 0.738 (fixed dispersion); over 1000 subsamples the difference has median −0.012 and range [−0.126, +0.097].

**技术细节（ZH）**

**命题 13.** 加入动作 `a` 时 `Q_{S ∪ a} = Q_S + v v^\top / s`，其中 `v = P_Sℓ_a`、`s = ℓ_a^\top P_S ℓ_a + r_a`；均值目标下 `Δ = \frac{(ω^\top P_S ℓ_a)^2}{s}`。**引理 14：**单调但不次模；表 1：25 个实例上贪心加一轮交换达到穷举最优的 ≥ 0.987。**真实数据.** 睡眠（REM）：`N = 4, 16, 64` 时连续 0.648 / 0.659 / 0.863，分散 0.682 / 0.738 / 0.881，在 SC/ST 之间有异质性。房颤负担：一小时（四个 15 分钟窗口）0.696 对 0.971，四小时 0.851 对 0.998。`N = 16` 时学得的 REM 位置 0.694，固定分散 0.738；1000 次重抽样的差值中位数 −0.012，区间 [−0.126, +0.097]。


## E6 · 结论 / Takeaway

- 引题：第 8 节 / Section 8

**e6-b0**  (EN 13 s · ZH 13 s)

- EN: Existing data do not automatically determine the value of measurements that were never made. So before optimising an observation system, first determine whether the proposed values are in the data at all.
- ZH: 现有的数据，不会自动确定那些从未实施过的测量的价值。所以在优化一个观测系统之前，先判断拟议协议的价值到底在不在数据里。

**e6-b1**  (EN 6 s · ZH 5 s)

- EN: Identification first. Calibration second. And design only at the resolution the calibration data support.
- ZH: 先识别，再校准；只在校准数据支持的分辨率上做设计。

**Technical detail (EN)**

The information required to evaluate a protocol is not the information required to recover the full latent dependence; it is the information needed to eliminate the ambiguity that changes the values under consideration. Finite calibration then sets the resolution at which values can be compared. Practically: pair a large routine cohort with a small, intensively observed calibration subset chosen with future protocol decisions in mind.

**技术细节（ZH）**

评估一个协议所需的信息，不等于恢复完整潜在依赖结构所需的信息；它只是消除那些会改变所考虑价值的歧义所需的信息。有限的校准数据随后决定了价值可以比较到的分辨率。实践上：用一个大规模常规队列，配上一个为未来协议决策而设计的小规模密集校准子集。


---

## 页面固定文字（technical/index.html、explainer/index.html）

- 页眉眉题：Paper explainer · identification · calibration · observation design
- 标题：Counterfactual Evaluation of Temporal Observation Protocols
- 作者行：Xizhe Zhang · School of Biomedical Engineering and Informatics, Nanjing Medical University
- 控件：English / 中文；Auto-advance chapters；Manuscript (PDF)
- 舞台提示：Press play to start the narration
- 键位说明：space play / pause · ← → beats · PgUp PgDn chapters · L language
- Read 段落标题：Read: this chapter in the paper / 读一读：这一章在论文里
- 页脚：Colour legend（latent trajectory Z · target Θ · realised protocol A · alternative protocol B · explained variance / value · non-identification）；Provenance（每个数字来自 paper/numbers.tex 与 results/）；Data（Sleep-EDF Expanded 与 Long-Term AF Database，仅用标注文件）
