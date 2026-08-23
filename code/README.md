# Counterfactual Evaluation of Temporal Observation Protocols — Code

这是论文最终稿的完整复现包。它包含方法库、全部模拟与真实数据实验、单元测试、
最终论文源文件、已缓存的开放注释数据、原始结果、参考图，以及从结果生成论文中
587 个数值宏的脚本。

最终论文源文件保持原样放在 `paper/`。实验脚本输出到 `results/` 和 `figures/`，
随后由 Makefile 自动同步论文使用的 4 张图并重新编译 35 页 PDF。

## 最快验证

需要 Python 3.11–3.14、GNU Make，以及编译论文时使用的 TeX Live/`latexmk`。

```bash
make setup
make verify-quick
```

`make verify-quick` 不重新执行耗时的 Monte Carlo 和 1000 次真实数据重采样；它会：

1. 运行全部单元测试；
2. 从包内封存结果重新生成论文图和 `paper/numbers.tex`；
3. 编译最终论文；
4. 检查所有论文数值宏均有定义；
5. 将结果文件与 `reference/` 中的封存输出逐项比较。

如果已经有可用环境，也可以指定 Python：

```bash
make verify-quick PY=/absolute/path/to/python
```

## 从数据完整重跑

```bash
make setup
make all
```

不要使用 `make -j`。多个步骤会按顺序写入相同的结果文件。

完整重跑包括数据解析、最终稿使用的 S1/S2、S3、S4、S5、S6、S8、S9、
Sleep-EDF、Long-Term AF、交叉拟合、1000 次支持稳定性重采样、敏感性分析、图、
数值宏和论文编译。在 24 核 Apple Silicon 验证机上通常约 60–90 分钟，其中校准
估计、错设实验、嵌套协议类和支持稳定性分析占主要时间。
实际时间依赖 BLAS、CPU 和 Python 版本。

最终稿未引用但为审计保留的 S3b、S5b、S7、旧 record-64 sensitivity 和
`fig_framework` 可另行运行：

```bash
make retained-regressions
```

这些历史回归不属于最终论文复现门禁；其中包含随机学习器，其末位结果可能随
BLAS/Python 构建而改变。

`data/` 已包含本稿使用的 PhysioNet 注释文件与处理后数组，因此完整重跑可以离线
完成。若缓存缺失，`experiments/fetch_data.py` 会从固定版本的 PhysioNet 路径重新
下载；本项目不下载 PSG 或 ECG 原始波形。

## 论文结果与代码对应

| 论文产物 | 主要生成脚本 | 原始输出 |
|---|---|---|
| 识别性示意图 | `experiments/synthetic/s1_s2_regression_identifiability.py` | `results/s1_s2_*` |
| 有限校准与协议类图 | S3、S4、S8；`experiments/make_fig_calibration.py` | `results/s3_*`, `s4_*`, `s8_*` |
| 目标感知设计表 | `experiments/synthetic/s5_design.py` | `results/s5_design.*` |
| 错设稳健性表 | `experiments/synthetic/s6_misspecification.py` | `results/s6_misspecification.*` |
| Sleep/AF 真实数据图 | `run_sleep.py`, `run_ltaf.py`, `crossfit_real.py` | `results/sleep_edf.*`, `ltaf.*`, `crossfit_real.json` |
| Sleep 支持稳定性图 | `experiments/calibration_sweep.py` | `results/calibration_sweep.json` |
| 附录敏感性结果 | `sensitivity_checks.py`, `record64_sensitivity.py` | 对应 JSON/CSV |
| 论文全部数值 | `experiments/make_numbers.py` | `paper/numbers.tex` |

校准图脚本使用最终稿所需的修订版：panel (b) 先在同一校准重复内聚合共享误差的
三个目标，再按 covariance strata 合并标准误；panel (c,d) 使用“protocol class”
术语。该脚本与最终稿中的图相对应。

## 目录

```text
protocol_ceiling/   核心 Python 方法库
experiments/        数据、模拟、真实数据和制图脚本
tests/              独立数值与实现回归检查
config/             错设实验的固定配置
data/               开放注释缓存和处理后数组
results/            当前可重建的结果文件
figures/            当前可重建的图
paper/              用户提供的最终论文源文件和已编译 PDF
reference/          封存的参考结果、数值宏和最终图
scripts/            环境记录与复现比较工具
```

## 关键复现约束

- 全局实验种子是 `20260802`；少数稳定性分析在脚本中固定了派生种子。
- Sleep 的拆分以 subject 为单位，同一人的两晚不跨 outer fold，并按 SC/ST 分层。
- AF 以 record 为单位拆分，因为公开数据库未提供可用的重复受试者标识。
- 支持选择、标准化、ridge 调参和预测器拟合均限制在相应 outer-training fold。
- 真实数据的 second-moment 指标是 best-linear value，不应改称 Bayes ceiling。
- `reference/numbers.tex` 是本最终稿的数值封印；最终稿实际引用的宏必须逐项一致。
  未引用的历史 S7 宏不作为论文复现失败条件。
- PDF 二进制哈希可能因 Matplotlib/TeX 的时间戳、字体和后端版本不同而变化；
  数值文件、宏、页面数、引用与图的数据映射是复现判据。

## 数据与依赖

数据来源、固定版本路径和使用边界见 `DATA_PROVENANCE.md`。验证环境的直接依赖
已固定在 `requirements.txt`；运行 `make environment` 可打印当前机器的版本记录。

## 完整性文件

- `REPRODUCIBILITY_REPORT.md`：本次交付前的实际运行与对照结果。
- `SHA256SUMS`：压缩前文件级 SHA-256 清单。
- `CITATION.cff`：论文与代码的引用信息。
- `LICENSE-NOTICE.md`：发布前仍需作者决定的代码许可事项。
