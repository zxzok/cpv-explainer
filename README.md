# Counterfactual Evaluation of Temporal Observation Protocols

**Xizhe Zhang** (张锡哲)

Manuscript: [`paper/main.pdf`](paper/main.pdf) · Reproduction code and data: [`code/`](code/) · Interactive explainer: <https://cpv.xizhe.net>

## Abstract

We study *counterfactual protocol evaluation*: whether data collected under a realised observation protocol determine
the predictive value of alternatives that were never deployed. Protocol value is the population R² of the
Bayes-optimal predictor of a fixed trajectory-level target from the measurements an alternative would collect. We show
that even infinite benchmark data need not determine this value: distinct latent covariance structures can induce the
same benchmark measurement–target law while assigning different values to the same alternative. We develop a
value-specific identification theory in which only latent ambiguity that changes the alternative's value matters. For
linear targets, invisible covariance directions certify non-identification, while targeted measurements can restore
identification without recovering the full latent covariance; an exact permutation construction extends the result to
nonlinear aggregate targets. With finite dense calibration data, uniform error bounds control protocol-selection regret
and distinguishable value gaps. Exact marginal gains then support cost-constrained, target-aware observation design.
Simulations and retrospective analyses of Sleep-EDF and Long-Term AF show that broad temporal-layout differences can be
more reliably distinguished than fine placements selected from finite data. Together, these results connect
identification, calibration resolution and observation design for undeployed protocols.

## The problem in one picture

A latent trajectory Z(t) (a night of sleep, a day of heart rhythm) is observed by a **protocol**: when to look, how
often, how long, how precisely. A benchmark pairs the measurements of one protocol A with a trajectory-level target Θ
(e.g. the fraction of the night spent in REM). Someone proposes a different protocol B that has never been run.
Does the benchmark tell us how well B would predict Θ?

```
      Z₀        Z₁        Z₂        Z₃        stationary, standardised latent process, Θ = mean(Z₀…Z₃)
      ▲                                        protocol A observes Z₀ only
                ┆         ┆                    protocol B would observe Z₁ and Z₂
```

Two correlation profiles ρ₀ ± ε·(0, 1, −2, 1) with ε = 0.1321 give **identical** benchmark laws
(Var Y_A = 1, Cov(Y_A, Θ) = 0.388250, Var Θ = 0.428012 in both worlds) but assign protocol B the values
**0.682** and **0.827**. No estimator built on A's data can be right in both worlds — and more subjects do not help.
What A never sees is the dependence *between* the two new measurement times.

## Main results

| | Result | Where |
|---|---|---|
| **Protocol value** | I(S) = Var(E[Θ \| Y_S]) / Var(Θ); for Gaussian latent processes I_g = F_g / V_g with an explicit covariance transform C_g for linear and threshold (occupation-time) targets | Def. 1, Prop. 2 |
| **Non-identification** | Sharp minimal stationary counterexample: a one-point benchmark on p points identifies the value of every alternative for p ≤ 3 and fails from p = 4 | Thm. 3 |
| **Value-specific identification** | "Invisible" covariance directions Δ (AΔAᵀ = 0, AΔh = 0, hᵀΔh = 0) certify non-identification exactly when the directional derivative of the alternative's value is non-zero; rank–nullity bounds the invisible space by p(p−1)/2 − d(d+1)/2 − d − 1 | Def. 4, Thm. 5 |
| **Nonlinear targets** | An exact permutation construction (swap two unobserved times) gives non-identification for every non-constant aggregate target on as few as three grid points | Prop. 6, Ex. 7 |
| **Restoring identification** | Targeted augmentation removes only the ambiguity that changes the evaluated value (4 → 2 → 0 invisible dimensions in the running example) without recovering the full latent covariance | Prop. 8 |
| **Calibration** | From a small densely observed subset, the plug-in value is uniformly accurate over a candidate family: sup_S \|Î_g(S) − I_g(S)\| ≤ C‖K̂ − K‖^β with β = 1 for smooth targets and at any fixed interior model, β = 1/2 for threshold targets in the worst case (sharp); root-m rates at a fixed model | Thms. 10–11, Prop. 9 |
| **Resolution** | Selecting the empirical best protocol loses at most 2ε (+ optimisation error) — value gaps below that scale cannot be resolved, and the useful granularity of optimisation is set by the calibration data, not by the optimiser | Cor. 12 |
| **Design** | Exact rank-one marginal gains for adding a measurement under a cost budget; the objective is monotone but not submodular; greedy + one swap attains ≥ 0.987 of the exhaustive optimum across 25 enumerated settings | Prop. 13, Lemma 14, Table 1 |
| **Real data** | Sleep-EDF (REM fraction): dispersed epochs beat a contiguous block at matched budgets (0.648 → 0.682 at N = 4; 0.659 → 0.738 at N = 16). Long-Term AF (AF burden): four dispersed 15-minute windows reach R² 0.971 vs 0.696 for the best contiguous hour; 0.998 vs 0.851 at N = 16. Learned exact anchor positions show no stable held-out advantage (−0.044 on the original sample; resampled range [−0.126, +0.097]) | Section 6 |

## Reproduction code (`code/`)

`code/` is the complete reproduction package for the final manuscript — methods library, every simulation and
real-data experiment, unit tests, the LaTeX sources, cached open annotation data, archived results and reference
outputs. Its own [`README.md`](code/README.md) (Chinese) and [`REPRODUCIBILITY_REPORT.md`](code/REPRODUCIBILITY_REPORT.md)
give the full details; the essentials:

```bash
cd code
make setup            # Python 3.11–3.14 virtual environment + pinned dependencies
make verify-quick     # unit tests → regenerate figures and paper/numbers.tex from the archived results →
                      # compile the manuscript → check that all 587 numeric macros resolve → compare with reference/
make all              # complete re-run from the cached data (≈ 60–90 min on a 24-core Apple Silicon machine; no -j)
make retained-regressions   # historical regressions kept for audit (not cited by the final manuscript)
```

Requirements: Python 3.11–3.14, GNU Make, and TeX Live / `latexmk` for compiling the paper. Every file in the package
is listed in `code/SHA256SUMS` (`cd code && shasum -a 256 -c SHA256SUMS`).

```
code/
├── protocol_ceiling/        methods library ("protocol-limited learning of temporal aggregates")
│   ├── values.py            protocol values: Bayes, best-linear and achieved (Def. 1, Prop. 2)
│   ├── transforms.py        target functionals and their Gaussian covariance transforms C_g
│   ├── covariance.py        latent-process covariances, observation actions and protocol matrices
│   ├── identifiability.py   counterfactual identifiability: invisible-direction certificates, the four-point,
│   │                        permutation and augmentation constructions (Thms. 3, 5; Prop. 6, 8)
│   ├── estimation.py        calibration-based estimation of protocol values and its error analysis (Thms. 10–11)
│   ├── resolution.py        resolution-adaptive protocol selection (Cor. 12, nested protocol classes)
│   ├── risk.py              protocol-conditioned Bayes risk, value ceilings and rank-one design updates (Prop. 13)
│   ├── design.py            target-aware observation design under temporal and measurement budgets (greedy, swap)
│   ├── uncertainty.py       subject-level bootstrap uncertainty for estimated values
│   ├── adaptive.py          subject-adaptive acquisition and the boundary of its usefulness
│   ├── diagnostics.py       risk decomposition, ceiling utilisation, simple learners
│   └── continuous.py        continuous-time trait–state theory (not used by the final manuscript)
├── experiments/
│   ├── synthetic/           s1_s2_regression_identifiability, s3_ceiling_estimation, s4_selection_regret, s5_design,
│   │                        s6_misspecification, s8_resolution, s9_proof_checks (+ s3b, s5b, s7 retained for audit)
│   ├── sleep_edf/run_sleep.py, ltaf/run_ltaf.py   retrospective real-data analyses (REM fraction; AF burden)
│   ├── crossfit_real.py     subject-disjoint five-fold cross-fitting with pooled held-out R²
│   ├── calibration_sweep.py, sensitivity_checks.py
│   ├── make_fig*.py, make_numbers.py   paper figures and the 587 numeric macros in paper/numbers.tex
│   └── fetch_data.py        re-downloads the pinned PhysioNet annotation files if the cache is missing
├── tests/                   unit tests (run by make verify-quick)
├── data/                    cached PhysioNet annotation files and processed arrays (no raw PSG/ECG waveforms)
├── results/                 archived outputs of the final runs      reference/   sealed outputs for comparison
├── figures/                 paper figures as produced by the scripts
├── paper/                   LaTeX sources of the final manuscript (main.tex, section files, numbers.tex, references.bib)
├── validation/              logs of the final verification and full-reproduction runs
├── Makefile, pyproject.toml, requirements.txt, config/, scripts/
└── CITATION.cff, DATA_PROVENANCE.md, LICENSE-NOTICE.md, REPRODUCIBILITY_REPORT.md, SHA256SUMS
```

### Data

Sleep-EDF Expanded (197 whole-night hypnograms, 100 subjects; target = REM fraction, also N3 and wake) and the
Long-Term AF Database (84 records of about 24 h; target = AF burden), both from PhysioNet. Only the expert annotation
files are used — no raw polysomnography or ECG waveforms — and they are cached in `code/data/` so that a full re-run
works offline. See [`code/DATA_PROVENANCE.md`](code/DATA_PROVENANCE.md) for sources, versions and terms.

### License

The code is released for review and reproduction; see [`code/LICENSE-NOTICE.md`](code/LICENSE-NOTICE.md)
(a specific open-source license will be added on publication). The PhysioNet data remain under their own terms.

## Interactive explainer

<https://cpv.xizhe.net> — a landing page built around the four-point counterexample, a four-minute narrated story and
a ten-minute interactive technical tour (English / Chinese). Every number shown or spoken is read from
`code/paper/numbers.tex` and `code/results/`, and the interactive scenes recompute protocol values in the browser with
the paper's formulas. The site's source is the rest of this repository; see [`WEBSITE.md`](WEBSITE.md) for how it is
built and edited.

## Citation

```bibtex
@unpublished{zhang2026counterfactual,
  title  = {Counterfactual Evaluation of Temporal Observation Protocols},
  author = {Zhang, Xizhe},
  year   = {2026},
  note   = {Manuscript. Code: https://github.com/zxzok/cpv-explainer/tree/main/code}
}
```

---

### 中文摘要

本文研究**反事实协议评估**：在某一观测协议下采集的数据，能否确定另一种从未实施过的协议的预测价值。协议价值定义为用该协议
所采集测量对固定轨迹级目标做贝叶斯最优预测的总体 R²。我们证明即使基准数据无穷多也未必能确定这一价值：不同的潜在协方差结构
可以诱导出完全相同的基准“测量–目标”分布，却给同一替代协议赋予不同的价值。我们建立了“价值特定”的可识别性理论——只有会改变
替代协议价值的潜在歧义才重要：对线性目标，不可见协方差方向刻画不可识别性，而定向补测可以在不恢复完整潜在协方差的情况下恢复
可识别性；一个精确的置换构造把结论推广到非线性聚合目标。在有限的密集校准数据下，一致误差界控制协议选择的损失与可区分的价值差。
精确的边际收益进而支持预算约束下的目标感知观测设计。模拟与 Sleep-EDF、Long-Term AF 的回溯分析表明：粗粒度的时间布局差异，
比从有限数据中选出的精确位置更容易被可靠区分。

**代码在 `code/` 目录**（方法库 `protocol_ceiling/`、实验 `experiments/`、测试、论文源文件、缓存数据与结果）：
`cd code && make setup && make verify-quick` 快速验证，`make all` 完整重跑。网站源码见仓库根目录与 `WEBSITE.md`。
