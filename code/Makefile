# Reproducibility entry points for the final manuscript.
# Do not run experiment targets with -j: several stages write shared files.

PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
PAPER := paper

.PHONY: all setup environment test data synthetic real retained-regressions figures numbers \
		sync-paper-artifacts paper compare-reference verify-quick verify \
		clean distclean

# Full from-data reproduction. Cached open annotation files are included, so
# this target does not need network access unless data/ has been removed.
all: test data synthetic real figures paper compare-reference

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt

environment:
	"$(PY)" scripts/record_environment.py

test:
	PYTHONDONTWRITEBYTECODE=1 "$(PY)" tests/test_core.py
	PYTHONDONTWRITEBYTECODE=1 "$(PY)" tests/test_identifiability_design.py
	PYTHONDONTWRITEBYTECODE=1 "$(PY)" tests/test_real_protocols.py

data:
	"$(PY)" experiments/fetch_data.py

synthetic:
	"$(PY)" experiments/synthetic/s1_s2_regression_identifiability.py
	"$(PY)" experiments/synthetic/s3_ceiling_estimation.py
	"$(PY)" experiments/synthetic/s4_selection_regret.py
	"$(PY)" experiments/synthetic/s5_design.py
	"$(PY)" experiments/synthetic/s6_misspecification.py
	"$(PY)" experiments/synthetic/s8_resolution.py
	"$(PY)" experiments/synthetic/s9_proof_checks.py

real:
	"$(PY)" experiments/sleep_edf/run_sleep.py
	"$(PY)" experiments/ltaf/run_ltaf.py
	"$(PY)" experiments/crossfit_real.py
	"$(PY)" experiments/calibration_sweep.py
	"$(PY)" experiments/sensitivity_checks.py

# Historical regression studies retained for auditability but not referenced
# by the final manuscript. Their stochastic learner outputs can vary slightly
# across BLAS/Python builds and are deliberately outside the paper gate.
retained-regressions:
	"$(PY)" experiments/synthetic/s3b_modulus_regimes.py
	"$(PY)" experiments/synthetic/s5b_submodularity.py
	"$(PY)" experiments/synthetic/s7_protocol_vs_architecture.py
	"$(PY)" experiments/ltaf/record64_sensitivity.py
	"$(PY)" experiments/make_fig1.py

figures:
	"$(PY)" experiments/make_fig_calibration.py
	"$(PY)" experiments/make_fig_real.py
	"$(PY)" experiments/make_fig_sweep.py

numbers:
	"$(PY)" experiments/make_numbers.py

sync-paper-artifacts: figures numbers
	cp figures/fig_identifiability.pdf $(PAPER)/figures/fig_identifiability.pdf
	cp figures/fig_calibration.pdf $(PAPER)/figures/fig_calibration.pdf
	cp figures/fig_real.pdf $(PAPER)/figures/fig_real.pdf
	cp figures/fig_sweep.pdf $(PAPER)/figures/fig_sweep.pdf

paper: sync-paper-artifacts
	cd $(PAPER) && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cd $(PAPER) && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

compare-reference:
	"$(PY)" scripts/verify_artifacts.py

# Fast, offline validation from the included reference results. This runs all
# unit checks, rebuilds every manuscript figure and number, compiles the
# 35-page PDF, and compares outputs with the sealed reference artefacts.
verify-quick: test figures paper compare-reference

verify: verify-quick

clean:
	cd $(PAPER) && latexmk -C

distclean: clean
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
