"""Regression tests for the fixed real-data comparators and AF estimand."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from sleep_edf.run_sleep import (anchor_indices, consecutive_cols,
                                 load_corpus, moments, stratified_subject_folds,
                                 subject_base_weights, subject_row_weights,
                                 uniform_cols, weighted_residuals)
from ltaf.run_ltaf import HORIZON_H, Model, contiguous_windows
from protocol_ceiling.covariance import protocol_matrices
from protocol_ceiling.diagnostics import r2_score, ridge_fit_predict

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def test_fixed_comparators() -> None:
    check("centred contiguous comparator", consecutive_cols(4, p=10) == [3, 4, 5, 6])
    check("dispersed comparator spans fixed full domain",
          uniform_cols(4, p=10) == [0, 3, 6, 9])
    for budget in (1, 4, 8, 16, 32, 64, 128):
        cols = uniform_cols(budget, p=128)
        check(f"dispersed comparator has {budget} distinct anchors",
              len(cols) == budget and len(set(cols)) == budget
              and min(cols) >= 0 and max(cols) < 128)


def test_sleep_estimand_mechanics() -> None:
    for n in (128, 129, 197, 2720, 10000):
        idx = anchor_indices(n, 128)
        check(f"{n}-epoch record maps anchors one-to-one",
              idx.size == 128 and np.unique(idx).size == 128
              and np.all(np.diff(idx) > 0))

    sid = np.array(["SC1", "SC1", "SC2", "ST1", "ST1", "ST1", "ST2"])
    strata = np.array(["SC", "SC", "SC", "ST", "ST", "ST", "ST"])
    wb = subject_base_weights(sid)
    check("Sleep base weights are exactly inverse recording counts",
          np.allclose(wb, np.array([1 / 2, 1 / 2, 1, 1 / 3, 1 / 3, 1 / 3, 1])),
          detail=str(wb.tolist()))
    w = subject_row_weights(sid)
    totals = np.array([w[sid == s].sum() for s in np.unique(sid)])
    check("subject-balanced rows give equal subject totals",
          np.allclose(totals, totals[0], atol=1e-14))
    folds = stratified_subject_folds(sid, strata, 2, np.random.default_rng(7))
    fold_subjects = [set(sid[f].tolist()) for f in folds]
    check("stratified Sleep folds are subject-disjoint",
          fold_subjects[0].isdisjoint(fold_subjects[1])
          and sum(len(f) for f in folds) == len(sid))

    y = np.array([0.0, 1.0, 3.0])
    pred = np.array([0.5, 0.5, 2.5])
    ww = np.array([1.0, 1.0, 2.0])
    ybar = float(ww @ y / ww.sum())
    manual = 1.0 - float(ww @ ((y - pred) ** 2)) / float(ww @ ((y - ybar) ** 2))
    check("weighted R2 matches its defining formula",
          np.isclose(r2_score(y, pred, sample_weight=ww), manual, atol=1e-14))
    check("pooled weighted R2 is invariant to common weight scaling",
          np.isclose(r2_score(y, pred, sample_weight=17.0 * ww), manual,
                     atol=1e-14))

    # Training-fold residualisation must remove all weighted linear baseline
    # signal from both the anchor matrix and target before protocol selection.
    C = np.column_stack([np.linspace(-1.0, 1.0, 7),
                         np.array([0, 0, 0, 1, 1, 1, 1], dtype=float)])
    X = np.column_stack([2.0 + 3.0 * C[:, 0] - C[:, 1],
                         np.sin(np.arange(7.0)) + 0.5 * C[:, 0]])
    theta = 1.0 - 2.0 * C[:, 0] + 4.0 * C[:, 1] + np.cos(np.arange(7.0))
    Xp, tp = weighted_residuals(X, theta, C, wb)
    D = np.column_stack([np.ones(7), C])
    check("baseline-residualised anchors are weighted-orthogonal to baseline",
          np.max(np.abs(D.T @ (wb[:, None] * Xp))) < 1e-11)
    check("baseline-residualised target is weighted-orthogonal to baseline",
          np.max(np.abs(D.T @ (wb * tp))) < 1e-11)
    conditioned = moments(X, theta, np.arange(7), sid, C)
    check("Sleep moments record baseline-conditioned selection",
          conditioned.baseline_conditioned and np.isfinite(conditioned.v))

    Xtr = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 5.0], [1.0, 7.0]])
    ytr = np.array([0.0, 1.0, 2.0, 4.0])
    p = ridge_fit_predict(Xtr, ytr, Xtr[:2], alphas=(0.1,), n_folds=2,
                          rng=np.random.default_rng(1),
                          groups=np.arange(4), sample_weight=np.array([2., 2., 1., 1.]))
    check("weighted ridge protects a zero-variance feature", np.all(np.isfinite(p)))

    # A large ridge penalty must leave the shared baseline slope untouched.
    x = np.linspace(-2.0, 2.0, 30)
    X = np.column_stack([x, np.sin(7.0 * x)])
    yy = 1.5 + 2.25 * x
    pu = ridge_fit_predict(
        X, yy, X, alphas=(1e6,), n_folds=3, rng=np.random.default_rng(2),
        groups=np.arange(x.size), n_unpenalized=1)
    check("ridge leaves declared baseline covariates unpenalised",
          np.max(np.abs(pu - yy)) < 1e-10,
          detail=f"max error {np.max(np.abs(pu - yy)):.2e}")

    corpus = load_corpus()
    st = corpus.cohorts == "ST"
    check("official ST metadata maps one placebo and one temazepam night per subject",
          all(sorted(corpus.treatments[corpus.subject_ids == s].tolist())
              == ["placebo", "temazepam"]
              for s in np.unique(corpus.subject_ids[st])))


def test_af_model_representations() -> None:
    # One constant coordinate checks the stated zero-variance convention.
    W = np.array([
        [0.25, 0.00, 0.20, 0.90],
        [0.25, 0.20, 0.40, 0.70],
        [0.25, 0.60, 0.80, 0.10],
        [0.25, 0.80, 1.00, 0.30],
        [0.25, 1.00, 0.60, 0.50],
    ])
    affine = Model(W, "indicator_linear")
    scale = np.where(affine.sd > 0.0, affine.sd, 1.0)
    Z = (W - affine.mu) / scale
    a_p = affine.raw_weights.sum() / W.shape[1]
    reconstructed = W.mean(axis=0).mean() + a_p * (
        Z @ (affine.raw_weights / affine.raw_weights.sum()))
    check("AF affine representation exactly reconstructs discretised burden",
          np.allclose(reconstructed, W.mean(axis=1), atol=1e-14))
    check("zero-variance AF bins receive zero target weight",
          affine.raw_weights[0] == 0.0)

    actions = contiguous_windows(2, HORIZON_H / W.shape[1])
    A, _ = protocol_matrices(actions, affine.grid)
    D = np.diag(affine.sd)
    M_raw = A @ affine.raw_cov @ A.T
    c_raw = A @ affine.raw_c
    direct = float(c_raw @ np.linalg.pinv(M_raw) @ c_raw / affine.raw_v)
    Zc = Z - Z.mean(axis=0)
    Sc = np.cov(Z, rowvar=False)
    cz = np.array([np.cov(Z[:, j], W.mean(axis=1))[0, 1]
                   for j in range(W.shape[1])])
    Az = A @ D
    transformed = float((Az @ cz) @ np.linalg.pinv(Az @ Sc @ Az.T)
                        @ (Az @ cz) / affine.raw_v)
    check("AF raw-window value equals transformed-row standardised value",
          np.isclose(direct, transformed, atol=1e-12),
          detail=f"raw={direct:.12f}, transformed={transformed:.12f}")
    check("AF Model.ceiling evaluates the raw protocol",
          np.isclose(affine.ceiling(actions), direct, atol=1e-12))


if __name__ == "__main__":
    test_fixed_comparators()
    test_sleep_estimand_mechanics()
    test_af_model_representations()
    print("\n" + "=" * 68)
    if failures:
        print(f"{len(failures)} FAILURE(S): " + ", ".join(failures))
        raise SystemExit(1)
    print("all real-protocol tests passed")
