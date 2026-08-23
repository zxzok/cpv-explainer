"""Sensitivity and diagnostic checks requested in the final revision round.

Four blocks, all read-only with respect to the other experiments:

(A) Covariance diagnostics.  For every outer training fold of both datasets,
    the algebraic rank of the raw sample covariance, its smallest eigenvalue,
    and the number of eigenvalues below the flooring threshold.  Effective rank
    is NOT evidence of algebraic rank deficiency, so the manuscript quotes these
    instead.

(B) Recording-weighted Sleep-EDF sensitivity.  Subject-balanced moments are the
    primary analysis; here the legacy one-row-per-recording weighting is compared.

(C) Eigenvalue-floor sensitivity.  tau = c * m^{-1} * mean-diagonal for
    c in {0.5, 1, 2}: does the coarse contiguous-versus-dispersed ordering move,
    and does the fine placement move?

(The AF cohort table lives in crossfit_real.py, so that its primary row is
    produced by the same code path as the headline AF numbers.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from experiments.common import SEED, save_json                      # noqa: E402
from experiments.sleep_edf.run_sleep import (                       # noqa: E402
    DEGENERATE_TOL, P_GRID, anchor_matrix, best_linear_greedy, build_labels,
    consecutive_cols, load_corpus, sleep_baseline_covariates,
    stratified_subject_folds, uniform_cols, weighted_residuals)
from protocol_ceiling.estimation import effective_rank              # noqa: E402
from protocol_ceiling.covariance import project_psd                  # noqa: E402

N_OUTER = 5
BUDGET = 16
FLOOR_MULTIPLIERS = (0.5, 1.0, 2.0)


def _floor_tau(Sigma: np.ndarray, m: int, c: float = 1.0) -> float:
    return c * max(float(np.mean(np.diag(Sigma))), 1e-12) / max(m, 1)


def _diagnostics(Ys: np.ndarray, baseline: np.ndarray,
                 w: np.ndarray, n_clusters: int) -> dict:
    m, p = Ys.shape
    residual, _ = weighted_residuals(Ys, np.zeros(m), baseline, w)
    w = np.asarray(w, float)
    w = w / w.sum()
    residual -= w @ residual
    denom = max(1.0 - float(w @ w), 1e-12)
    S = (residual * w[:, None]).T @ residual / denom
    ev = np.linalg.eigvalsh(S)
    tau = _floor_tau(S, n_clusters)
    return {"m": int(m), "p": int(p),
            "rank": int(np.linalg.matrix_rank(S)),
            "lambda_min": float(ev.min()),
            "tau": float(tau),
            "n_floored": int((ev < tau).sum()),
            "effective_rank": float(effective_rank(S))}


def _weighted_moments(Ys, th, baseline, w, n_clusters, tau_mult=1.0):
    """Weighted second moments plus the floored covariance and its selection score."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    Ys, th = weighted_residuals(Ys, th, baseline, w)
    Xc = Ys - w @ Ys
    Sigma = (Xc * w[:, None]).T @ Xc / (1.0 - np.sum(w ** 2))
    thc = th - w @ th
    c = (Xc * w[:, None]).T @ thc / (1.0 - np.sum(w ** 2))
    v = float((w * thc ** 2).sum() / (1.0 - np.sum(w ** 2)))
    Sigma = project_psd(Sigma, floor=_floor_tau(Sigma, n_clusters, tau_mult))
    sd = np.sqrt(np.maximum(
        np.sum(w[:, None] * Xc * Xc, axis=0) / (1.0 - np.sum(w ** 2)), 0.0))
    degenerate = [int(j) for j in np.nonzero(sd < DEGENERATE_TOL)[0]]
    c[degenerate] = 0.0
    return Sigma, c, v, degenerate


def main() -> None:
    corpus = load_corpus()
    sid = corpus.subject_ids
    baseline, baseline_names = sleep_baseline_covariates(corpus)
    out: dict = {"note": "baseline-conditioned sensitivity and covariance diagnostics",
                 "seed": SEED, "budget": BUDGET,
                 "baseline_covariates": baseline_names,
                 "floor_multipliers": list(FLOOR_MULTIPLIERS)}
    exact_target = {s.key: s.theta_exact for s in build_labels(corpus)}
    n = len(sid)
    folds = stratified_subject_folds(
        sid, corpus.cohorts, N_OUTER, np.random.default_rng(SEED))

    # ---- (A) + (B) + (C) on Sleep-EDF ---------------------------------
    diag: dict = {}
    balance: dict = {}
    floor: dict = {}
    for key in ("REM", "N3", "wake"):
        Y = anchor_matrix(corpus, key)
        theta = exact_target[key]
        per_fold, bal_rows, floor_rows = [], [], []
        for f, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(n), te)
            uniq, inv = np.unique(sid[tr], return_inverse=True)
            counts = np.bincount(inv)
            w_rec = np.ones(len(tr))
            w_sub = 1.0 / counts[inv]
            per_fold.append(_diagnostics(
                Y[tr], baseline[tr], w_sub, n_clusters=len(uniq)))

            # subject-balanced primary versus one-row-per-recording sensitivity
            sel = {}
            for name, w in (("recording", w_rec), ("subject", w_sub)):
                S, c, v, deg = _weighted_moments(
                    Y[tr], theta[tr], baseline[tr], w, len(uniq))
                sel[name] = best_linear_greedy(S, c, v, BUDGET, forbidden=deg)
            a, b = set(sel["recording"]), set(sel["subject"])
            bal_rows.append({"fold": f, "jaccard": len(a & b) / len(a | b)})

            # eigenvalue-floor multiplier sweep
            row = {"fold": f}
            base = None
            for cmult in FLOOR_MULTIPLIERS:
                S, c, v, deg = _weighted_moments(
                    Y[tr], theta[tr], baseline[tr], w_sub, len(uniq), cmult)
                cols = set(best_linear_greedy(S, c, v, BUDGET, forbidden=deg))
                if base is None:
                    base = cols
                row[f"c{cmult}_jaccard_vs_c1"] = None
                row[f"c{cmult}_cols"] = sorted(cols)
            one = set(row["c1.0_cols"])
            for cmult in FLOOR_MULTIPLIERS:
                cols = set(row[f"c{cmult}_cols"])
                row[f"c{cmult}_jaccard_vs_c1"] = len(cols & one) / len(cols | one)
            floor_rows.append(row)

        diag[key] = per_fold
        balance[key] = bal_rows
        floor[key] = floor_rows

    n_ok = []
    for key in ("REM", "N3", "wake"):
        Y = anchor_matrix(corpus, key)
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            uniq, inv = np.unique(sid[tr], return_inverse=True)
            w = 1.0 / np.bincount(inv)[inv]
            residual, _ = weighted_residuals(
                Y[tr], np.zeros(len(tr)), baseline[tr], w)
            n_ok.append(int((residual.std(axis=0, ddof=1)
                             >= DEGENERATE_TOL).sum()))
    out["sleep_min_informative_anchors"] = int(min(n_ok))
    out["sleep_covariance_diagnostics"] = diag
    out["sleep_subject_balanced"] = balance
    out["sleep_floor_sensitivity"] = floor

    # coarse-class ordering is a property of the fixed comparators, so it is
    # evaluated once per floor multiplier on the training-fold score itself
    coarse = {}
    for key in ("REM", "N3", "wake"):
        Y = anchor_matrix(corpus, key)
        theta = exact_target[key]
        rows = []
        for cmult in FLOOR_MULTIPLIERS:
            gaps = []
            for f, te in enumerate(folds):
                tr = np.setdiff1d(np.arange(n), te)
                uniq, inv = np.unique(sid[tr], return_inverse=True)
                counts = np.bincount(inv)
                S, c, v, deg = _weighted_moments(
                    Y[tr], theta[tr], baseline[tr], 1.0 / counts[inv],
                    len(uniq), cmult)
                def score(cols):
                    cols = list(cols)
                    return float(c[cols] @ np.linalg.solve(S[np.ix_(cols, cols)], c[cols]) / v)
                gaps.append(score(uniform_cols(BUDGET))
                            - score(consecutive_cols(BUDGET)))
            rows.append({"c": cmult, "mean_dispersed_minus_contiguous": float(np.mean(gaps)),
                         "min_dispersed_minus_contiguous": float(np.min(gaps)),
                         "all_positive": bool(all(g > 0 for g in gaps))})
        coarse[key] = rows
    out["sleep_coarse_under_floor"] = coarse

    # ---- (B) coarse ordering under subject balancing --------------------
    coarse_bal = {}
    for key in ("REM", "N3", "wake"):
        Y = anchor_matrix(corpus, key)
        theta = exact_target[key]
        rows = []
        for name in ("recording", "subject"):
            gaps = []
            for f, te in enumerate(folds):
                tr = np.setdiff1d(np.arange(n), te)
                uniq, inv = np.unique(sid[tr], return_inverse=True)
                counts = np.bincount(inv)
                w = np.ones(len(tr)) if name == "recording" else 1.0 / counts[inv]
                S, c, v, deg = _weighted_moments(
                    Y[tr], theta[tr], baseline[tr], w, len(uniq))

                def score(cols):
                    cols = list(cols)
                    return float(c[cols] @ np.linalg.solve(S[np.ix_(cols, cols)], c[cols]) / v)
                gaps.append(score(uniform_cols(BUDGET))
                            - score(consecutive_cols(BUDGET)))
            rows.append({"weighting": name,
                         "mean_dispersed_minus_contiguous": float(np.mean(gaps)),
                         "min_dispersed_minus_contiguous": float(np.min(gaps)),
                         "all_positive": bool(all(g > 0 for g in gaps))})
        coarse_bal[key] = rows
    out["sleep_coarse_under_balancing"] = coarse_bal

    # ---- (A) + (D) on Long-Term AF -------------------------------------
    from crossfit_real import AF_P_GRID, af_matrix_cohort

    af_diag = []
    X, y, _ = af_matrix_cohort("all", AF_P_GRID)
    rng = np.random.default_rng(SEED + 2)
    fds = [np.array(f) for f in np.array_split(rng.permutation(len(y)), N_OUTER)]
    for f, te in enumerate(fds):
        tr = np.setdiff1d(np.arange(len(y)), te)
        af_diag.append({"fold": f, **_diagnostics(
            X[tr], np.empty((len(tr), 0)), np.ones(len(tr)), len(tr))})
    out["af_covariance_diagnostics"] = af_diag

    save_json(out, "sensitivity_checks")
    print("[sensitivity] wrote results/sensitivity_checks.json")
    for key, rows in diag.items():
        r = rows[0]
        print(f"  {key:5} fold0: m={r['m']} p={r['p']} rank={r['rank']} "
              f"lam_min={r['lambda_min']:.2e} floored={r['n_floored']}")


if __name__ == "__main__":
    main()
