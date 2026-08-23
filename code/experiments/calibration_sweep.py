"""How many calibration objects are needed to resolve a fine placement?

The paper's resolution decomposition says a calibration sample may order coarse
protocol classes reliably while being unable to separate exact placements
within a class.  Reporting only "target-aware did not win" leaves open whether
the comparison is hopeless or merely under-powered, so this script estimates
the calibration size at which the fine distinction becomes resolvable.

Two quantities, both re-selecting the protocol inside every replicate and both
using the same selector as the main analysis (`select_protocol_greedy` with the
one-swap pass, i.e. Algorithm 1):

1.  Calibration-size sweep.  Within each outer fold the protocol is selected
    from a calibration subset holding a fraction q of the training subjects,
    while the ridge predictor is fitted on the *complete* outer-training fold.
    That separation is deliberate: it isolates protocol-calibration error from
    predictor-sample-size error, so a null result cannot be blamed on having
    fewer objects with which to fit the predictor.

2.  Selection-aware repeated subsampling.  Each replicate draws 80% of the
    subjects without replacement separately within SC and ST, then repeats
    standardisation, covariance estimation, protocol selection, inner tuning,
    predictor fitting and held-out evaluation.  Unique subjects remain the
    independent calibration clusters, so duplicate bootstrap copies cannot
    leak across outer or inner folds.

Writes results/calibration_sweep.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from experiments.common import save_json  # noqa: E402
from crossfit_real import FOLD_SEED, LAMBDAS, N_INNER  # noqa: E402
from protocol_ceiling.diagnostics import r2_score, ridge_fit_predict  # noqa: E402
from protocol_ceiling.values import best_linear_greedy  # noqa: E402
from sleep_edf.run_sleep import (BUDGETS, MAX_SWAP_ROUNDS, P_GRID,  # noqa: E402
                                 agnostic_order, anchor_matrix, build_labels,
                                 consecutive_cols, load_corpus, moments,
                                 sleep_baseline_covariates,
                                 stratified_subject_folds, subject_base_weights,
                                 uniform_cols)

SEED = 0
# The outer split must match the one `sec:sleep` evaluates the stored designs on,
# otherwise the two analyses of the same budget cannot be reconciled.
N_OUTER = 5
FRACTIONS = (0.25, 0.50, 0.75, 1.00)
N_SUBSAMPLE = 40          # calibration subsets per fraction
N_REPEATED_SUBSAMPLES = 1000
SELECTION_SUBSAMPLE_FRACTION = 0.80
BUDGET = 16               # scored epochs; N=64 costs ~50x more in the swap pass
LABEL = "REM"
METHODS = ("target_aware", "kernel_quadrature", "uniform")


def ridge_cv_predict(Xtr, ytr, Xte, groups, sample_weight, fold_id,
                     n_unpenalized=0):
    return ridge_fit_predict(
        Xtr, ytr, Xte, alphas=tuple(LAMBDAS), n_folds=N_INNER,
        rng=np.random.default_rng(SEED + 1009 * fold_id), groups=groups,
        sample_weight=sample_weight, n_unpenalized=n_unpenalized)


def r2(y, pred, sample_weight):
    ok = ~np.isnan(pred)
    return r2_score(y[ok], pred[ok], sample_weight=sample_weight[ok])


def select(Y, theta, baseline, idx, method, subject_ids, budget=BUDGET):
    """Protocol chosen from calibration objects `idx` (indices into the corpus).

    Same selector as the headline analysis: the target-aware criterion is
    forward selection on the best-linear value of the acquired anchors, capped
    at MAX_SWAP_ROUNDS accepted one-swap refinements; kernel quadrature is the
    target-agnostic baseline run on the same estimated covariance; uniform is
    the fixed dispersed schedule, which depends only on the pre-specified full
    anchor grid and the budget.  Learned arms exclude calibration-constant
    anchors; the fixed comparator deliberately does not adapt to those moments.
    This analysis reproduces ``run_sleep.py`` at ``cal_frac = 1``.
    """
    mom = moments(Y, theta, np.asarray(idx), subject_ids, baseline)
    if method == "uniform":
        return uniform_cols(budget)
    if method == "kernel_quadrature":
        return sorted(agnostic_order(mom, "kernel_quadrature", max(BUDGETS))[:budget])
    return best_linear_greedy(mom.Sigma, mom.c, mom.v, budget,
                              forbidden=mom.degenerate,
                              max_swap_rounds=MAX_SWAP_ROUNDS)


def evaluate(Yg, thg, baseline, X, y, groups, strata, order, folds, method,
             cal_frac, rng):
    """Cross-fitted R^2; `order` maps fold positions to corpus indices."""
    pred = np.full(len(order), np.nan)
    weights = subject_base_weights(groups)
    for fold_id, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(order)), te)
        if len(tr) < 5 or len(te) == 0:
            continue
        cal_pos = tr
        if cal_frac < 1.0:
            keep_parts = []
            for level in np.unique(strata[tr]):
                g = np.unique(groups[tr][strata[tr] == level])
                keep_parts.append(rng.choice(
                    g, size=max(1, int(round(cal_frac * len(g)))), replace=False))
            keep = np.concatenate(keep_parts)
            cal_pos = tr[np.isin(groups[tr], keep)]
        cols = select(Yg[order], thg[order], baseline[order], cal_pos, method,
                      groups)
        if not cols:
            continue
        B = baseline[order]
        Xtr = np.column_stack([B[tr], X[np.ix_(order[tr], cols)]])
        Xte = np.column_stack([B[te], X[np.ix_(order[te], cols)]])
        pred[te] = ridge_cv_predict(
            Xtr, y[order[tr]], Xte, groups[tr], weights[tr], fold_id,
            n_unpenalized=B.shape[1])
    return r2(y[order], pred, weights)


def stratified_subject_subsample(groups, strata, fraction, rng):
    """Row indices for a study-stratified subject sample without replacement."""
    keep = []
    for level in np.unique(strata):
        subs = np.unique(groups[strata == level])
        n_keep = max(2, int(round(fraction * subs.size)))
        keep.extend(rng.choice(subs, size=n_keep, replace=False).tolist())
    return np.flatnonzero(np.isin(groups, keep))


def _distribution_summary(v):
    a = np.array(v, float)
    return {"median": float(np.nanmedian(a)),
            "p025": float(np.nanpercentile(a, 2.5)),
            "p975": float(np.nanpercentile(a, 97.5)),
            "frac_positive": float(np.nanmean(a > 0)),
            "n": int(np.sum(~np.isnan(a)))}


def main() -> None:
    t0 = time.perf_counter()
    corp = load_corpus()
    spec = {s.key: s for s in build_labels(corp)}[LABEL]
    y = spec.theta_exact
    X = anchor_matrix(corp, LABEL)
    baseline, baseline_names = sleep_baseline_covariates(corp)
    groups = corp.subject_ids
    n = len(y)
    order = np.arange(n)
    folds = stratified_subject_folds(
        groups, corp.cohorts, N_OUTER, np.random.default_rng(FOLD_SEED))
    n_subj = len(np.unique(groups))

    out: dict = {
        "note": ("calibration-size sweep and study-stratified selection-aware "
                 "repeated subject subsampling without replacement; protocol "
                 "re-selected in every replicate"),
        "seed": SEED, "n_outer": N_OUTER, "fractions": list(FRACTIONS),
        "n_subsample": N_SUBSAMPLE,
        "n_repeated_subsamples": N_REPEATED_SUBSAMPLES,
        "selection_subsample_fraction": SELECTION_SUBSAMPLE_FRACTION,
        "budget": BUDGET, "label": LABEL, "n_subjects": int(n_subj),
        "weighting": "subject-balanced throughout selection, ridge and R2",
        "selection_conditioning": {
            "baseline_covariates": baseline_names,
            "operation": "weighted training-fold residualisation of anchors and target",
        },
        "outer_folds": "subject-disjoint and stratified by SC/ST study",
        "selection_resampling": {
            "replacement": False,
            "stratified_by": "SC/ST study",
            "independent_cluster": "unique original subject",
            "duplicate_subject_copies": False,
            "moments_weights_floor_recomputed": True,
        },
    }

    # ---- 1. calibration-size sweep -------------------------------------
    sweep = {}
    for q in FRACTIONS:
        m_eff = int(round(q * n_subj * (N_OUTER - 1) / N_OUTER))
        reps = 1 if q >= 1.0 else N_SUBSAMPLE
        vals = {mth: [] for mth in METHODS}
        for r in range(reps):
            for mth in METHODS:
                rr = np.random.default_rng(SEED + 7919 * r)
                vals[mth].append(evaluate(X, y, baseline, X, y, groups, corp.cohorts,
                                          order, folds, mth, q, rr))
        rec = {"m_train_subjects": m_eff, "n_reps": reps}
        for mth in METHODS:
            a = np.array(vals[mth], float)
            rec[mth] = {"mean": float(np.nanmean(a)), "sd": float(np.nanstd(a))}
        # paired differences: store the standard error of the plotted mean so the
        # figure can draw a band, not just the point
        _dkq = np.array(vals["target_aware"], float) - np.array(vals["kernel_quadrature"], float)
        _dun = np.array(vals["target_aware"], float) - np.array(vals["uniform"], float)
        rec["delta_vs_kq"] = float(np.nanmean(_dkq))
        rec["delta_vs_uniform"] = float(np.nanmean(_dun))
        rec["delta_vs_kq_se"] = float(np.nanstd(_dkq, ddof=1) / np.sqrt(len(_dkq))) if len(_dkq) > 1 else 0.0
        rec["delta_vs_uniform_se"] = float(np.nanstd(_dun, ddof=1) / np.sqrt(len(_dun))) if len(_dun) > 1 else 0.0
        sweep[f"q{q:g}"] = rec
        print(f"  [sweep] q={q:g} m~{m_eff:3d}  aware {rec['target_aware']['mean']:+.3f}"
              f"  kq {rec['kernel_quadrature']['mean']:+.3f}"
              f"  delta {rec['delta_vs_kq']:+.3f}", flush=True)
    out["sweep"] = sweep

    out["original_sample"] = {
        "delta_vs_kq": sweep["q1"]["delta_vs_kq"],
        "delta_vs_uniform": sweep["q1"]["delta_vs_uniform"],
        "target_aware_r2": sweep["q1"]["target_aware"]["mean"],
    }

    # ---- 2. study-stratified repeated subsampling without replacement --
    d_kq, d_uni, abs_aware = [], [], []
    for b in range(N_REPEATED_SUBSAMPLES):
        rr = np.random.default_rng(SEED + 104729 * b)
        idx = stratified_subject_subsample(
            groups, corp.cohorts, SELECTION_SUBSAMPLE_FRACTION, rr)
        gb = groups[idx]
        fb = stratified_subject_folds(
            gb, corp.cohorts[idx], N_OUTER,
            np.random.default_rng(FOLD_SEED + b))
        a = evaluate(X, y, baseline, X, y, gb, corp.cohorts[idx], idx, fb,
                     "target_aware", 1.0, rr)
        k = evaluate(X, y, baseline, X, y, gb, corp.cohorts[idx], idx, fb,
                     "kernel_quadrature", 1.0, rr)
        u = evaluate(X, y, baseline, X, y, gb, corp.cohorts[idx], idx, fb,
                     "uniform", 1.0, rr)
        abs_aware.append(a); d_kq.append(a - k); d_uni.append(a - u)
        if (b + 1) % 50 == 0:
            print(f"  [subsample] {b+1}/{N_REPEATED_SUBSAMPLES}  delta_kq median "
                  f"{np.nanmedian(d_kq):+.3f}", flush=True)
    out["repeated_subsampling"] = {
        "target_aware_r2": _distribution_summary(abs_aware),
        "delta_vs_kq": _distribution_summary(d_kq),
        "delta_vs_uniform": _distribution_summary(d_uni),
    }

    out["runtime_s"] = round(time.perf_counter() - t0, 1)
    save_json(out, "calibration_sweep")
    print(f"  ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
