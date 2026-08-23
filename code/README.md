# Counterfactual Evaluation of Temporal Observation Protocols — Code

This is the complete reproduction package for the final manuscript. It contains the methods library, every
simulation and real-data experiment, the unit tests, the final LaTeX sources, cached open annotation data, the raw
results, the reference figures, and the script that generates the 587 numeric macros used in the paper from those
results.

The final manuscript sources are kept unchanged in `paper/`. The experiment scripts write to `results/` and
`figures/`; the Makefile then syncs the four figures used by the paper and recompiles the 35-page PDF.

## Quickest verification

Requires Python 3.11–3.14, GNU Make, and the TeX Live / `latexmk` installation used to compile the paper.

```bash
make setup
make verify-quick
```

`make verify-quick` does not repeat the long Monte Carlo runs or the 1000 real-data resamples; it

1. runs all unit tests;
2. regenerates the paper figures and `paper/numbers.tex` from the sealed results shipped in the package;
3. compiles the final manuscript;
4. checks that every numeric macro used by the paper is defined;
5. compares the result files, item by item, with the sealed outputs in `reference/`.

An existing Python environment can be used instead of the one created by `make setup`:

```bash
make verify-quick PY=/absolute/path/to/python
```

## Full re-run from the data

```bash
make setup
make all
```

Do not use `make -j`: several stages write the same result files in sequence.

The full re-run covers data parsing, the simulations used by the final manuscript (S1/S2, S3, S4, S5, S6, S8, S9),
Sleep-EDF, Long-Term AF, cross-fitting, the 1000 support-stability resamples, the sensitivity analyses, the figures,
the numeric macros, and the compilation of the paper. On a 24-core Apple Silicon validation machine it typically
takes 60–90 minutes; calibration estimation, the misspecification experiment, the nested protocol classes and the
support-stability analysis account for most of that time. Actual times depend on BLAS, CPU and Python version.

The historical regressions that the final manuscript does not cite but that are retained for audit — S3b, S5b, S7,
the old record-64 sensitivity analysis and `fig_framework` — can be run separately:

```bash
make retained-regressions
```

These retained regressions are not part of the reproduction gate for the final paper; they include randomised
learners whose last digits may change with the BLAS / Python build.

`data/` already contains the PhysioNet annotation files and processed arrays used by this manuscript, so a full
re-run works offline. If the cache is missing, `experiments/fetch_data.py` downloads the files again from pinned
PhysioNet paths; the project never downloads raw PSG or ECG waveforms.

## Mapping from paper artefacts to code

| Paper artefact | Main generating script | Raw output |
|---|---|---|
| Identifiability illustration | `experiments/synthetic/s1_s2_regression_identifiability.py` | `results/s1_s2_*` |
| Finite-calibration and protocol-class figure | S3, S4, S8; `experiments/make_fig_calibration.py` | `results/s3_*`, `s4_*`, `s8_*` |
| Target-aware design table | `experiments/synthetic/s5_design.py` | `results/s5_design.*` |
| Misspecification robustness table | `experiments/synthetic/s6_misspecification.py` | `results/s6_misspecification.*` |
| Sleep / AF real-data figure | `run_sleep.py`, `run_ltaf.py`, `crossfit_real.py` | `results/sleep_edf.*`, `ltaf.*`, `crossfit_real.json` |
| Sleep support-stability figure | `experiments/calibration_sweep.py` | `results/calibration_sweep.json` |
| Appendix sensitivity results | `sensitivity_checks.py`, `record64_sensitivity.py` | corresponding JSON / CSV |
| All numbers in the paper | `experiments/make_numbers.py` | `paper/numbers.tex` |

The calibration-figure script is the revised version required by the final manuscript: panel (b) first pools the
three targets that share each calibration draw within a replicate and then combines standard errors across
covariance strata; panels (c, d) use the "protocol class" terminology. The script corresponds to the figure in the
final manuscript.

## Layout

```text
protocol_ceiling/   core Python methods library
experiments/        data, simulation, real-data and figure scripts
tests/              independent numerical and implementation regression checks
config/             fixed configuration of the misspecification experiment
data/               cached open annotation files and processed arrays
results/            the currently rebuildable result files
figures/            the currently rebuildable figures
paper/              the final manuscript sources and compiled PDF
reference/          sealed reference results, numeric macros and final figures
scripts/            environment recording and reproduction-comparison tools
validation/         logs of the final verification and full reproduction runs
```

## Key reproduction constraints

- The global experiment seed is `20260802`; a few stability analyses fix derived seeds inside their scripts.
- Sleep-EDF is split by subject — both nights of a subject stay within one outer fold — and stratified by SC/ST.
- Long-Term AF is split by record, because the public database provides no usable repeated-subject identifiers.
- Support selection, standardisation, ridge tuning and predictor fitting are all confined to the corresponding
  outer-training fold.
- The second-moment metric on real data is the best-linear value and must not be relabelled a Bayes ceiling.
- `reference/numbers.tex` is the numeric seal of this final manuscript; every macro the final manuscript actually
  cites must agree item by item. Uncited historical S7 macros are not a reproduction-failure criterion.
- The binary hash of the PDF may change with Matplotlib / TeX timestamps, fonts and backend versions; the numeric
  files, the macros, the page count, the citations and the figure-to-data mapping are the reproduction criteria.

## Data and dependencies

Data sources, pinned version paths and usage boundaries are described in `DATA_PROVENANCE.md`. The direct
dependencies of the validation environment are pinned in `requirements.txt`; `make environment` prints the version
record of the current machine.

## Integrity files

- `REPRODUCIBILITY_REPORT.md`: the actual runs and comparisons performed before delivery.
- `SHA256SUMS`: file-level SHA-256 manifest (computed before compression).
- `CITATION.cff`: citation information for the paper and the code.
- `LICENSE-NOTICE.md`: the code-licensing decision that remains with the author before public release.
