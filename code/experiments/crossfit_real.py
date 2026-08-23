"""Honest out-of-sample evaluation of the real-data protocols.

The protocol-value estimate reported previously recomputed $\\widehat c$,
$\\widehat\\Sigma$ and $\\widehat\\Var(\\Theta)$ on the held-out fold, which makes it
the in-fold multiple correlation

    y' X_S (X_S' X_S)^+ X_S' y / y'y ,

i.e. the test labels are used to choose the regression coefficients as well as
to score them.  It can therefore saturate as the protocol size approaches the
fold size and is not an out-of-sample performance measure.

This script instead fits the linear predictor on the training folds --- ridge
with the penalty chosen by an inner subject-level cross-validation --- predicts
the held-out objects, pools the predictions across outer folds, and reports

    R^2_cf(S) = 1 - sum_i (Theta_i - Theta_hat_i^{-f(i)})^2 / sum_i (Theta_i - mean)^2 .

Sleep protocols are the ones already selected on the training folds and stored
in results/sleep_edf_designs.csv, so no selection information leaks.  AF uses
the two pre-specified fixed templates.

Writes results/crossfit_real.json.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from experiments.common import SEED as DESIGN_SEED, save_json  # noqa: E402
from protocol_ceiling.diagnostics import r2_score, ridge_fit_predict  # noqa: E402
from sleep_edf.run_sleep import (P_GRID, anchor_matrix,  # noqa: E402
                                 build_labels, consecutive_cols, load_corpus,
                                 sleep_baseline_covariates,
                                 stratified_subject_folds, subject_folds,
                                 subject_base_weights, uniform_cols)

LAMBDAS = np.logspace(-6.0, 2.0, 25)
N_OUTER = 5
N_INNER = 3
SEED = 0
# The stored Sleep-EDF designs were selected fold-by-fold by run_sleep.py, which
# splits subjects with experiments.common.SEED.  Evaluating them on a split
# drawn from a different seed would score each protocol on subjects that helped
# choose it -- 84% of records, as it happens.  The outer split here must be the
# one the designs were selected against.
FOLD_SEED = DESIGN_SEED
# Atrial-fibrillation grid.  The protocols below are described as fifteen-minute
# windows, so the bin has to *be* fifteen minutes: 24 h / 96 = 15 min exactly.
# At p = 128 a bin is 11.25 min and `round(15 / 11.25) = 1`, so a "15 min" window
# was really 11.25 min while the reported duration still charged 0.25 h -- every
# duration on the AF axis was then 4/3 of the time actually analysed.
AF_P_GRID = 96
AF_HORIZON_H = 24.0
AF_WINDOW_MIN = 15.0
N_BOOTSTRAP = 2000


single_epoch_matrix = anchor_matrix   # the acquired variable, defined in run_sleep


def cross_fitted_predictions(
        X: np.ndarray, y: np.ndarray, cols_by_fold: dict[int, list[int]],
        folds: list[np.ndarray], subject_ids: np.ndarray,
        sample_weight: np.ndarray | None = None,
        n_unpenalized: int = 0) -> np.ndarray:
    """Pool held-out predictions; the protocol may differ per outer fold."""
    pred = np.full(len(y), np.nan)
    for f, te in enumerate(folds):
        cols = cols_by_fold.get(f)
        if not cols:
            continue
        tr = np.setdiff1d(np.arange(len(y)), te)
        pred[te] = ridge_fit_predict(
            X[np.ix_(tr, cols)], y[tr], X[np.ix_(te, cols)],
            alphas=tuple(LAMBDAS), n_folds=N_INNER,
            rng=np.random.default_rng(SEED + 1009 * f), groups=subject_ids[tr],
            sample_weight=None if sample_weight is None else sample_weight[tr],
            n_unpenalized=n_unpenalized)
    return pred


def cross_fitted_r2(X: np.ndarray, y: np.ndarray, cols_by_fold: dict[int, list[int]],
                    folds: list[np.ndarray], subject_ids: np.ndarray,
                    sample_weight: np.ndarray | None = None,
                    n_unpenalized: int = 0) -> float:
    """Cross-fitted pooled squared-loss R2."""
    pred = cross_fitted_predictions(X, y, cols_by_fold, folds, subject_ids,
                                    sample_weight=sample_weight,
                                    n_unpenalized=n_unpenalized)
    ok = ~np.isnan(pred)
    weight = None if sample_weight is None else sample_weight[ok]
    return float(r2_score(y[ok], pred[ok], sample_weight=weight))


def paired_cluster_percentile_range(
        y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
        clusters: np.ndarray, seed: int, n_boot: int = N_BOOTSTRAP,
        strata: np.ndarray | None = None) -> dict:
    """Conditional paired range from realised held-out prediction pairs.

    This resamples clusters of the already-computed cross-fitted pairs. It does
    not refit the predictor, retune ridge or recreate the outer folds, and is
    therefore deliberately not labelled a confidence interval for the fitted
    pipeline.
    """
    y, pred_a, pred_b, clusters = map(np.asarray, (y, pred_a, pred_b, clusters))
    uniq = np.unique(clusters)
    members = {u: np.flatnonzero(clusters == u) for u in uniq}
    strata_levels: dict[object, object] = {}
    if strata is not None:
        strata = np.asarray(strata)
        if strata.shape != clusters.shape:
            raise ValueError("strata must have one entry per row")
        for u in uniq:
            levels = np.unique(strata[members[u]])
            if levels.size != 1:
                raise ValueError("each resampling cluster must belong to one stratum")
            strata_levels[u] = levels[0]
    rng = np.random.default_rng(seed)
    d_r2 = np.empty(n_boot)
    d_mse = np.empty(n_boot)
    point_weight = subject_base_weights(clusters)

    def weighted_mse(yy, pp, ww):
        return float(np.sum(ww * (yy - pp) ** 2) / np.sum(ww))

    for b in range(n_boot):
        if strata is None:
            sampled = rng.choice(uniq, size=uniq.size, replace=True)
        else:
            sampled = np.concatenate([
                rng.choice([u for u in uniq if strata_levels[u] == level],
                           size=sum(strata_levels[u] == level for u in uniq),
                           replace=True)
                for level in np.unique(list(strata_levels.values()))])
        idx = np.concatenate([members[u] for u in sampled])
        # Each sampled cluster contributes total weight one, including when it
        # appears more than once in the conditional resampling draw.
        w = np.concatenate([np.full(members[u].size, 1.0 / members[u].size)
                            for u in sampled])
        d_r2[b] = (r2_score(y[idx], pred_b[idx], sample_weight=w)
                   - r2_score(y[idx], pred_a[idx], sample_weight=w))
        d_mse[b] = (weighted_mse(y[idx], pred_b[idx], w)
                    - weighted_mse(y[idx], pred_a[idx], w))
    return {
        "n_resamples": n_boot,
        "n_clusters": int(uniq.size),
        "range": ("conditional 2.5--97.5 percentile range of realised "
                  "cross-fitted prediction pairs"),
        "stratified_by": None if strata is None else "study cohort",
        "refitting": False,
        "multiplicity_adjustment": None,
        "r2_difference_b_minus_a": float(
            r2_score(y, pred_b, sample_weight=point_weight)
            - r2_score(y, pred_a, sample_weight=point_weight)),
        "r2_difference_range": [float(x) for x in np.quantile(d_r2, [0.025, 0.975])],
        "mse_difference_b_minus_a": float(
            weighted_mse(y, pred_b, point_weight)
            - weighted_mse(y, pred_a, point_weight)),
        "mse_difference_range": [float(x) for x in np.quantile(d_mse, [0.025, 0.975])],
    }


def designs_by_fold(label: str) -> dict[tuple[str, int], dict[int, list[int]]]:
    """Protocol bins selected on the training folds, from the stored designs."""
    out: dict[tuple[str, int], dict[int, list[int]]] = defaultdict(dict)
    for r in csv.DictReader(open(ROOT / "results" / "sleep_edf_designs.csv")):
        if r["label"] != label:
            continue
        # run_sleep stores the selected anchor indices directly, so there is no
        # time-to-bin round trip to get wrong here
        cols = sorted({int(c) for c in r["cols"].split(";") if c.strip()})
        out[(r["method"], int(r["budget"]))][int(r["fold"])] = cols
    return out


def sleep_cohort_fixed_results(corp, labels: dict) -> dict:
    """Baseline-adjusted fixed-template descriptions within each Sleep study."""
    pooled_base, _ = sleep_baseline_covariates(corp)
    out = {}
    for ci, cohort in enumerate(("SC", "ST")):
        idx = np.flatnonzero(corp.cohorts == cohort)
        sid = corp.subject_ids[idx]
        weights = subject_base_weights(sid)
        folds = subject_folds(sid, N_OUTER,
                              np.random.default_rng(FOLD_SEED + 100 + ci))
        if cohort == "SC":
            base = pooled_base[idx, 1:2]
            baseline_names = ["analysed valid-stage duration (hours)"]
        else:
            base = pooled_base[idx][:, [1, 2]]
            baseline_names = ["analysed valid-stage duration (hours)",
                              "temazepam-night indicator"]
        n_base = base.shape[1]
        cohort_out = {
            "n_records": int(idx.size), "n_subjects": int(np.unique(sid).size),
            "baseline_covariates": baseline_names,
            "baseline_covariates_unpenalized": True,
            "median_hours": float(np.median([len(corp.stages[i])
                                              * corp.epoch_seconds / 3600.0
                                              for i in idx])),
            "labels": {},
        }
        for li, label in enumerate(("REM", "N3", "wake")):
            Xp = single_epoch_matrix(corp, label)[idx]
            X = np.column_stack([base, Xp])
            y = labels[label].theta_exact[idx]
            cells = {}
            for budget in (4, 8, 16, 32, 64):
                cc = list(range(n_base)) + [n_base + c for c in consecutive_cols(budget)]
                dd = list(range(n_base)) + [n_base + c for c in uniform_cols(budget)]
                pred_c = cross_fitted_predictions(
                    X, y, {f: cc for f in range(N_OUTER)}, folds, sid, weights,
                    n_unpenalized=n_base)
                pred_d = cross_fitted_predictions(
                    X, y, {f: dd for f in range(N_OUTER)}, folds, sid, weights,
                    n_unpenalized=n_base)
                cells[f"N={budget}"] = paired_cluster_percentile_range(
                    y, pred_c, pred_d, sid,
                    seed=SEED + 9001 + 1000 * ci + 100 * li + budget)
            cohort_out["labels"][label] = cells
        out[cohort] = cohort_out
    return out


def sleep_baseline_adjusted_increment(corp, labels: dict, folds: list[np.ndarray],
                                      weights: np.ndarray) -> dict:
    """Protocol value beyond unpenalised study, duration and treatment baselines."""
    base, baseline_names = sleep_baseline_covariates(corp)
    n_base = base.shape[1]
    out = {}
    for label in ("REM", "N3", "wake"):
        y = labels[label].theta_exact
        Xp = single_epoch_matrix(corp, label)
        pred0 = cross_fitted_predictions(
            base, y, {f: list(range(n_base)) for f in range(N_OUTER)}, folds,
            corp.subject_ids, weights, n_unpenalized=n_base)
        r20 = r2_score(y, pred0, sample_weight=weights)
        cells = {"baseline_r2": float(r20), "budgets": {}}
        X = np.column_stack([base, Xp])
        for budget in (4, 8, 16, 32, 64):
            row = {}
            predictions = {}
            for method, cols in (("contiguous", consecutive_cols(budget)),
                                 ("dispersed", uniform_cols(budget))):
                use = list(range(n_base)) + [n_base + c for c in cols]
                pred = cross_fitted_predictions(
                    X, y, {f: use for f in range(N_OUTER)}, folds,
                    corp.subject_ids, weights, n_unpenalized=n_base)
                r2 = r2_score(y, pred, sample_weight=weights)
                predictions[method] = pred
                row[method] = {"r2": float(r2), "increment_over_baseline":
                               float(r2 - r20)}
            row["dispersed_minus_contiguous"] = paired_cluster_percentile_range(
                y, predictions["contiguous"], predictions["dispersed"],
                corp.subject_ids, seed=SEED + 12001 + 101 * budget
                + 17 * ("REM", "N3", "wake").index(label),
                strata=corp.cohorts)
            cells["budgets"][f"N={budget}"] = row
        out[label] = cells
    return {"baseline_covariates": baseline_names,
            "baseline_covariates_unpenalized": True,
            "labels": out}



# ==========================================================================
#  Atrial fibrillation
# ==========================================================================
def af_matrix(p_grid: int = AF_P_GRID):
    """Records on a common normalised grid, and the burden target."""
    from ltaf.run_ltaf import cohort_index, design_matrix, load_af
    af, ids, _ = load_af()
    burden = np.array([x.mean() for x in af])
    idx = cohort_index(burden, "all")
    X = design_matrix(af, idx, p_grid)
    return X, burden[idx], ids[idx]


def af_protocol_bins(method: str, n_win: int, win_min: float,
                     p_grid: int = AF_P_GRID,
                     horizon_h: float = AF_HORIZON_H) -> list[int]:
    bin_min = horizon_h * 60.0 / p_grid
    if abs(win_min / bin_min - round(win_min / bin_min)) > 1e-9:
        raise ValueError(
            f"a {win_min:g} min window is not a whole number of {bin_min:g} min "
            f"bins; the reported duration would not be the duration analysed")
    k = max(1, int(round(win_min / bin_min)))
    if method == "contiguous":
        start = max(0, (p_grid - n_win * k) // 2)
        return list(range(start, min(p_grid, start + n_win * k)))
    centres = np.linspace(0, p_grid - k, n_win).astype(int)
    cols: list[int] = []
    for c in centres:
        cols.extend(range(c, min(p_grid, c + k)))
    return sorted(set(cols))


def af_matrix_cohort(cohort: str, p_grid: int = AF_P_GRID):
    """Records on a common normalised grid, for a named cohort."""
    from ltaf.run_ltaf import cohort_index, design_matrix, load_af
    af, ids, _ = load_af()
    burden = np.array([x.mean() for x in af])
    idx = cohort_index(burden, cohort)
    return design_matrix(af, idx, p_grid), burden[idx], ids[idx]


def af_cohort_table(rng_seed: int = SEED + 2) -> dict:
    """The same contiguous/dispersed comparison on each cohort.

    The primary ("all") row is produced by exactly the code path that writes
    the headline numbers, so the table and the text can never disagree.
    """
    out = {}
    for cohort in ("all", "strict", "mixed"):
        X, y, ids = af_matrix_cohort(cohort)
        rng = np.random.default_rng(rng_seed)
        folds = [np.array(f) for f in np.array_split(rng.permutation(len(y)), N_OUTER)]
        rows = []
        for n_win in (4, 16):
            r2 = {}
            for method in ("contiguous", "dispersed"):
                cols = af_protocol_bins(method, n_win, AF_WINDOW_MIN)
                cbf = {f: cols for f in range(len(folds))}
                r2[method] = cross_fitted_r2(X, y, cbf, folds, ids)
            # per-fold gaps: pooled point estimates cannot show fold-level
            # reproducibility, so record the single-fold held-out differences too
            per_fold = []
            for te in folds:
                one = [np.array(te)]
                gc = cross_fitted_r2(X, y, {0: af_protocol_bins("contiguous", n_win, AF_WINDOW_MIN)}, one, ids)
                gd = cross_fitted_r2(X, y, {0: af_protocol_bins("dispersed", n_win, AF_WINDOW_MIN)}, one, ids)
                per_fold.append(float(gd - gc))
            rows.append({"n_windows": n_win,
                         "contiguous": float(r2["contiguous"]),
                         "dispersed": float(r2["dispersed"]),
                         "dispersed_minus_contiguous":
                             float(r2["dispersed"] - r2["contiguous"]),
                         "per_fold_gap": per_fold,
                         "min_fold_gap": float(min(per_fold))})
        out[cohort] = {"n_records": int(len(y)), "budgets": rows}
    return out


def af_cross_fitted(rng) -> dict:
    X, y, ids = af_matrix()
    folds = [np.array(f) for f in np.array_split(rng.permutation(len(y)), N_OUTER)]
    out = {}
    boot = {}
    bin_hours = AF_HORIZON_H / AF_P_GRID
    for n_win in (1, 2, 4, 8, 16, 32):
        preds = {}
        for method in ("contiguous", "dispersed"):
            cols = af_protocol_bins(method, n_win, AF_WINDOW_MIN)
            cbf = {f: cols for f in range(len(folds))}
            key = f"{method}|n={n_win}"
            pred = cross_fitted_predictions(X, y, cbf, folds, ids)
            preds[method] = pred
            out[key] = {
                # derived from the bins actually observed, never from the
                # nominal window length, so the axis cannot drift from the data
                "total_hours": len(cols) * bin_hours,
                "observed_fraction": len(cols) / AF_P_GRID,
                "equivalent_hours_at_24h": len(cols) * bin_hours,
                "n_bins": len(cols),
                "window_minutes": AF_WINDOW_MIN,
                "cross_fitted_r2": float(r2_score(y, pred)),
            }
        boot[f"N={n_win}"] = paired_cluster_percentile_range(
            y, preds["contiguous"], preds["dispersed"], ids,
            seed=SEED + 7001 + n_win)
    return out, boot


def main() -> None:
    rng = np.random.default_rng(SEED)
    corp = load_corpus()
    specs = {s.key: s for s in build_labels(corp)}
    sleep_weight = subject_base_weights(corp.subject_ids)
    sleep_base, sleep_baseline_names = sleep_baseline_covariates(corp)
    n_sleep_base = sleep_base.shape[1]
    folds = stratified_subject_folds(
        corp.subject_ids, corp.cohorts, N_OUTER, np.random.default_rng(FOLD_SEED))

    out: dict = {
        "note": ("cross-fitted ridge R2; fixed-template conditional ranges "
                 "resample realised paired held-out predictions without refitting"),
        "n_outer_folds": N_OUTER, "n_inner_folds": N_INNER,
        "ridge": {"intercept": True, "feature_scaling": "inner-training fold",
                  "loss": "squared error", "alphas": LAMBDAS.tolist(),
                  "prediction_clipping": False, "tie_rule": "smallest alpha",
                  "zero_variance_feature": "retained with scale 1 after centering",
                  "baseline_penalty_factor": 0.0,
                  "protocol_feature_penalty_factor": 1.0},
        "sleep_weighting": ("subject-balanced for training standardisation, ridge fitting, "
                            "inner-CV loss, pooled R2 and conditional pair "
                            "resampling statistics"),
        "sleep_evaluation_base_weights": "w_i^(0)=1/n_subject over the full corpus",
        "sleep_selection_conditioning": {
            "baseline_covariates": sleep_baseline_names,
            "operation": "weighted training-fold residualisation of anchors and target",
        },
        "sleep_outer_folds": "subject-disjoint and stratified by SC/ST study",
        "sleep_alignment_primary": corp.alignment,
        "sleep": {}, "sleep_fixed_template_conditional_ranges": {}}

    for label in ("REM", "N3", "wake"):
        spec = specs[label]
        y = spec.theta_exact
        Xp = single_epoch_matrix(corp, label)      # one scored epoch per anchor
        X = np.column_stack([sleep_base, Xp])
        des = designs_by_fold(label)
        rows = {}
        for (method, budget), cbf in sorted(des.items()):
            adjusted_cbf = {f: list(range(n_sleep_base))
                            + [n_sleep_base + c for c in cols]
                            for f, cols in cbf.items()}
            rows[f"{method}|N={budget}"] = {
                "cross_fitted_r2": cross_fitted_r2(
                    X, y, adjusted_cbf, folds, corp.subject_ids, sleep_weight,
                    n_unpenalized=n_sleep_base),
                "baseline_adjusted": True,
            }
        out["sleep"][label] = rows
        out["sleep_fixed_template_conditional_ranges"][label] = {}
        for budget in (4, 8, 16, 32, 64):
            cbf_c = des[("consecutive", budget)]
            cbf_d = des[("uniform", budget)]
            cbf_c = {f: list(range(n_sleep_base))
                     + [n_sleep_base + c for c in cols]
                     for f, cols in cbf_c.items()}
            cbf_d = {f: list(range(n_sleep_base))
                     + [n_sleep_base + c for c in cols]
                     for f, cols in cbf_d.items()}
            pred_c = cross_fitted_predictions(
                X, y, cbf_c, folds, corp.subject_ids, sleep_weight,
                n_unpenalized=n_sleep_base)
            pred_d = cross_fitted_predictions(
                X, y, cbf_d, folds, corp.subject_ids, sleep_weight,
                n_unpenalized=n_sleep_base)
            out["sleep_fixed_template_conditional_ranges"][label][f"N={budget}"] = \
                paired_cluster_percentile_range(
                    y, pred_c, pred_d, corp.subject_ids,
                    seed=SEED + 5003 + 101 * budget + 17 * ("REM", "N3", "wake").index(label),
                    strata=corp.cohorts)
        print(f"\n[{label}]  (1 scored epoch per anchor)")
        for k, v in rows.items():
            print(f"  {k:26s} R2cf {v['cross_fitted_r2']:+.3f}")

    # The previous first/last-non-Wake alignment is retained only as a
    # sensitivity analysis of the two fixed templates.
    oracle = load_corpus("oracle")
    if not np.array_equal(oracle.record_ids, corp.record_ids):
        raise RuntimeError("full-record and oracle-aligned corpora are not record-aligned")
    oracle_specs = {s.key: s for s in build_labels(oracle)}
    oracle_weight = subject_base_weights(oracle.subject_ids)
    oracle_base, _ = sleep_baseline_covariates(oracle)
    n_oracle_base = oracle_base.shape[1]
    oracle_folds = stratified_subject_folds(
        oracle.subject_ids, oracle.cohorts, N_OUTER,
        np.random.default_rng(FOLD_SEED))
    out["sleep_oracle_alignment_sensitivity"] = {}
    for label in ("REM", "N3", "wake"):
        y = oracle_specs[label].theta_exact
        Xp = single_epoch_matrix(oracle, label)
        X = np.column_stack([oracle_base, Xp])
        cells = {}
        for budget in (4, 8, 16, 32, 64):
            cc = list(range(n_oracle_base)) + [n_oracle_base + c
                                                for c in consecutive_cols(budget)]
            dd = list(range(n_oracle_base)) + [n_oracle_base + c
                                                for c in uniform_cols(budget)]
            cbf_c = {f: cc for f in range(N_OUTER)}
            cbf_d = {f: dd for f in range(N_OUTER)}
            pred_c = cross_fitted_predictions(
                X, y, cbf_c, oracle_folds, oracle.subject_ids, oracle_weight,
                n_unpenalized=n_oracle_base)
            pred_d = cross_fitted_predictions(
                X, y, cbf_d, oracle_folds, oracle.subject_ids, oracle_weight,
                n_unpenalized=n_oracle_base)
            cells[f"N={budget}"] = {
                "contiguous": float(r2_score(y, pred_c, sample_weight=oracle_weight)),
                "dispersed": float(r2_score(y, pred_d, sample_weight=oracle_weight)),
                "dispersed_minus_contiguous": float(
                    r2_score(y, pred_d, sample_weight=oracle_weight)
                    - r2_score(y, pred_c, sample_weight=oracle_weight)),
            }
        out["sleep_oracle_alignment_sensitivity"][label] = cells

    out["sleep_cohort_fixed_templates"] = sleep_cohort_fixed_results(corp, specs)
    out["sleep_baseline_adjusted_increment"] = sleep_baseline_adjusted_increment(
        corp, specs, folds, sleep_weight)

    out["af"], out["af_fixed_template_conditional_ranges"] = af_cross_fitted(
        np.random.default_rng(SEED + 2))
    out["af_primary_cohort"] = "all"
    out["af_cohorts"] = af_cohort_table(SEED + 2)
    print("\n[AF burden]")
    for k, v in out["af"].items():
        print(f"  {k:18s} {v['observed_fraction']:.4f} of record   "
              f"R2cf {v['cross_fitted_r2']:+.3f}")

    save_json(out, "crossfit_real")


if __name__ == "__main__":
    main()
