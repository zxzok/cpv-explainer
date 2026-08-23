"""Real-data experiment R1 -- Sleep-EDF Expanded whole-night stage proportions.

The human-scored hypnogram *is* the local state: no sensor model, no classifier.
The only thing that varies across the arms is the observation protocol -- which
30 s epochs a technician is allowed to score.

What a protocol acquires
------------------------
The ``p = 128`` grid supplies relative-time *anchors*.  For each selected anchor
the protocol reads the single scored epoch nearest it, so a budget of ``N``
anchors costs exactly ``N`` scored epochs.  The acquired variable is therefore
the binary ``Y_j``, and the design is built on its second moments directly:

    Sigma = Var(Y^perp),   c = Cov(Y^perp, Theta^perp),
    v = Var(Theta^perp),

with ``Theta`` the exact stage proportion over all valid epochs of the record
and ``perp`` denoting weighted training-fold residualisation on the shared
study, valid-duration and treatment baselines.
No bin proportion, no measurement-error model and no latent transform are
involved.  Selecting ``S`` to maximise

    I_L(S) = c_S' Sigma_SS^+ c_S / v

is the best-linear protocol value of Definition 1 evaluated on the variable the
protocol actually reads, so the selection criterion and the held-out evaluation
concern the same statistical experiment.  Adding an anchor has the exact
Schur-complement increment

    Delta_L(a | S) = c_{a|S}^2 / (v s_{a|S}),

which is what makes forward selection cheap.

Methods compared at each budget: a contiguous block of adjacent anchors, uniform
spacing, random spacing, integrated posterior variance,
kernel quadrature, and the target-aware criterion above with a one-swap pass.
The two covariance-based baselines use the same estimated covariance but not
the target, which is what makes them the right comparison.

Subject-wise 5-fold cross-validation throughout: both nights of a subject stay in
the same fold, moments and protocol come from the training objects only, and a
ridge predictor fitted on the training objects is scored on the held-out objects.

Outputs
-------
``results/sleep_edf.csv``          one row per (label, budget, method, fold)
``results/sleep_edf_summary.csv``  aggregated over folds
``results/sleep_edf_designs.csv``  the selected anchor indices
``results/sleep_edf.json``         headline numbers, settings, environment
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import (SEED, Timer, environment_record, save_csv,
                                save_json)
from protocol_ceiling import (Action, MeanLabel, TimeGrid, bin_midpoints,
                              effective_rank,
                              design_imse, design_kernel_quadrature,
                              project_psd, to_correlation)
from protocol_ceiling.diagnostics import r2_score, ridge_fit_predict
from protocol_ceiling.values import (best_linear_greedy,
                                     best_linear_value_from_moments)


Array = np.ndarray

# --------------------------------------------------------------------------
# Fixed experiment constants -- every replication count is stated here
# --------------------------------------------------------------------------
DATA_NPZ = Path(__file__).resolve().parents[2] / "data" / "sleep_edf" / "hypnograms.npz"
P_GRID = 128                 # relative-time anchors on the normalised night
HORIZON = 1.0                # normalised night length
BUDGETS = (4, 8, 16, 32, 64)  # scored 30 s epochs = number of anchors
N_FOLDS = 5                  # subject-level folds
N_RANDOM = 20                # random-spacing replicates per (fold, budget)
MAX_SWAP_ROUNDS = 3          # accepted one-swap refinements after greedy
MIN_EPOCHS = P_GRID          # a recording must have at least one epoch per anchor
DEGENERATE_TOL = 1e-9        # column sd below which an anchor carries no information
FIGURE_BUDGET = 16           # budget used for the cross-label transfer matrix

STAGE_CODE = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4,
              "MOVEMENT": 5, "UNKNOWN": 6}
SLEEP_CODES = (1, 2, 3, 4)
STAGE_ORDER = ("W", "N1", "N2", "N3", "REM")

METHODS = ["consecutive", "uniform", "random", "imse",
           "kernel_quadrature", "label_aware"]
PRIMARY_LABELS = ("REM", "N3", "wake")
LABEL_SEED = {"REM": 11, "N3": 23, "wake": 37,
              "REM_over_TST": 53, "N3_over_TST": 71}


@dataclass
class Corpus:
    stages: list[Array]          # raw 30 s epoch code sequences
    subject_ids: Array
    record_ids: Array
    cohorts: Array
    treatments: Array
    epoch_seconds: int
    alignment: str = "full_record"


def load_corpus(alignment: str = "full") -> Corpus:
    if not DATA_NPZ.exists():
        raise FileNotFoundError(
            f"{DATA_NPZ} is missing; regenerate it with experiments/fetch_data.py")
    d = np.load(DATA_NPZ, allow_pickle=True)
    needed = {"stages", "subject_ids", "record_ids", "cohorts", "treatments",
              "epoch_seconds"}
    if not needed.issubset(set(d.files)):
        raise ValueError(f"{DATA_NPZ} is malformed: keys {d.files}")
    if alignment == "full":
        key, alignment_name = ("stages_full" if "stages_full" in d.files else "stages"), "full_record"
    elif alignment == "oracle":
        if "stages_oracle_aligned" not in d.files:
            raise ValueError("oracle-aligned sequences are missing; rerun experiments/fetch_data.py")
        key, alignment_name = "stages_oracle_aligned", "first_last_nonwake_oracle"
    else:
        raise ValueError(f"unknown alignment {alignment!r}")
    stages = [np.asarray(s, dtype=int) for s in d[key]]
    return Corpus(stages=stages, subject_ids=np.asarray(d["subject_ids"]),
                  record_ids=np.asarray(d["record_ids"]),
                  cohorts=np.asarray(d["cohorts"]),
                  treatments=np.asarray(d["treatments"]),
                  epoch_seconds=int(d["epoch_seconds"]), alignment=alignment_name)


def _bin_edges(n: int, p: int = P_GRID) -> Array:
    """Proportional bin edges; bin sizes differ by at most one epoch."""
    return np.round(np.linspace(0.0, float(n), p + 1)).astype(int)


def resample_series(values: Array, p: int = P_GRID) -> tuple[Array, Array]:
    """Return ``(bin means, bin population variances)`` of a per-epoch series."""
    e = _bin_edges(len(values), p)
    mean = np.empty(p)
    var = np.empty(p)
    for j in range(p):
        seg = values[e[j]:e[j + 1]]
        mean[j] = seg.mean()
        var[j] = seg.var()          # population variance = Var of one random epoch
    return mean, var


@dataclass
class LabelSpec:
    key: str
    kind: str                    # "indicator" or "linearised_ratio"
    X: Array = field(repr=False, default=None)    # (n_rec, p) bin means
    WV: Array = field(repr=False, default=None)   # (n_rec, p) within-bin variances
    theta_grid: Array = field(repr=False, default=None)   # (n_rec,) grid label
    theta_exact: Array = field(repr=False, default=None)  # (n_rec,) target quantity
    description: str = ""


def build_labels(corpus: Corpus) -> list[LabelSpec]:
    stages = corpus.stages
    n = len(stages)

    # linearisation point for the ratio labels: population means of the two
    # aggregates.  This is a *definitional* constant of the label (it fixes the
    # point at which A/B is linearised), not a quantity used to select any
    # protocol, so it is computed once on the whole corpus and held fixed.
    Bbar = float(np.mean([np.isin(s, SLEEP_CODES).mean() for s in stages]))
    Abar = {c: float(np.mean([(s == c).mean() for s in stages])) for c in (3, 4)}

    specs: list[LabelSpec] = []
    for key, code in (("REM", 4), ("N3", 3), ("wake", 0)):
        X = np.empty((n, P_GRID))
        WV = np.empty((n, P_GRID))
        exact = np.empty(n)
        for i, s in enumerate(stages):
            v = (s == code).astype(float)
            X[i], WV[i] = resample_series(v)
            exact[i] = v.mean()
        specs.append(LabelSpec(
            key=key, kind="indicator", X=X, WV=WV,
            theta_grid=X.mean(axis=1), theta_exact=exact,
            description=f"1{{stage = {key}}} averaged over the recording"))

    for key, code in (("REM_over_TST", 4), ("N3_over_TST", 3)):
        A = Abar[code]
        X = np.empty((n, P_GRID))
        WV = np.empty((n, P_GRID))
        exact = np.empty(n)
        for i, s in enumerate(stages):
            v = ((s == code).astype(float) / Bbar
                 - (A / Bbar**2) * np.isin(s, SLEEP_CODES).astype(float))
            X[i], WV[i] = resample_series(v)
            exact[i] = (s == code).sum() / max(int(np.isin(s, SLEEP_CODES).sum()), 1)
        specs.append(LabelSpec(
            key=key, kind="linearised_ratio",
            X=X, WV=WV, theta_grid=X.mean(axis=1), theta_exact=exact,
            description=(f"delta-method linearisation of {key.replace('_over_', '/')}"
                         f" at Abar={A:.4f}, Bbar={Bbar:.4f}")))
    return specs


def anchor_matrix(corpus: Corpus, key: str, p: int = P_GRID) -> Array:
    """The variable a protocol actually acquires: one scored epoch per anchor.

    The grid supplies ``p`` relative-time anchors; the protocol reads the single
    scored epoch nearest each selected anchor, so a budget of ``N`` anchors
    costs exactly ``N`` scored epochs.  This is the acquired variable itself --
    not a bin proportion observed through measurement error -- and the design
    below is built on its second moments directly.
    """
    code = {"REM": 4, "N3": 3, "wake": 0}[key]
    out = np.empty((len(corpus.stages), p))
    for i, sq in enumerate(corpus.stages):
        mid = anchor_indices(len(sq), p)
        out[i] = (sq[mid] == code).astype(float)
    return out


def anchor_indices(n_epochs: int, p: int = P_GRID) -> Array:
    """Distinct epoch indices read by the relative-time anchor grid.

    Records shorter than ``p`` are excluded before analysis. Consequently each
    proportional bin contains at least one epoch, and choosing its midpoint
    gives a strictly increasing one-to-one map from anchors to scored epochs.
    """
    if n_epochs < p:
        raise ValueError("a record needs at least one epoch per anchor")
    e = _bin_edges(n_epochs, p)
    mid = np.minimum((e[:-1] + e[1:]) // 2, n_epochs - 1)
    if np.unique(mid).size != p or np.any(np.diff(mid) <= 0):
        raise RuntimeError("relative-time anchors did not map to distinct epochs")
    return mid


def subject_row_weights(subject_ids: Array) -> Array:
    """Mean-one row weights giving every subject equal total weight."""
    raw = subject_base_weights(subject_ids)
    return raw * (raw.size / raw.sum())


def subject_base_weights(subject_ids: Array) -> Array:
    """Fixed base weights ``1 / n_subject`` for the subject-balanced estimand."""
    subject_ids = np.asarray(subject_ids)
    _, inv, counts = np.unique(subject_ids, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv]


def sleep_baseline_covariates(corpus: Corpus) -> tuple[Array, list[str]]:
    """Zero-cost covariates shared by every Sleep protocol."""
    study = (corpus.cohorts == "ST").astype(float)
    duration_h = np.array([len(s) * corpus.epoch_seconds / 3600.0
                           for s in corpus.stages])
    temazepam = (corpus.treatments == "temazepam").astype(float)
    return (np.column_stack([study, duration_h, temazepam]),
            ["ST study indicator", "analysed valid-stage duration (hours)",
             "ST temazepam-night indicator"])


def weighted_residuals(Y: Array, theta: Array, baseline: Array,
                       weights: Array) -> tuple[Array, Array]:
    """Residualise every anchor and the target on an intercept plus baseline.

    The weighted least-squares projection is fitted only to the rows supplied by
    the caller. A Moore--Penrose solve tolerates redundant columns, as occurs
    when a study indicator is constant in a study-specific analysis.
    """
    Y = np.asarray(Y, dtype=float)
    theta = np.asarray(theta, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if baseline.ndim != 2 or baseline.shape[0] != Y.shape[0]:
        raise ValueError("baseline must have one row per observation")
    if weights.shape != (Y.shape[0],):
        raise ValueError("weights must have one value per observation")
    D = np.column_stack([np.ones(Y.shape[0]), baseline])
    sw = np.sqrt(weights / weights.sum())
    Dw = D * sw[:, None]
    beta_y = np.linalg.lstsq(Dw, Y * sw[:, None], rcond=1e-12)[0]
    beta_t = np.linalg.lstsq(Dw, theta * sw, rcond=1e-12)[0]
    return Y - D @ beta_y, theta - D @ beta_t


@dataclass
class Moments:
    """``(Sigma, c, v)`` of the acquired anchors and the exact target."""

    Sigma: Array
    c: Array
    v: float
    n_records: int
    n_subjects: int
    effective_rank: float
    degenerate: list[int]
    baseline_conditioned: bool

    def value(self, cols: Sequence[int]) -> float:
        cols = list(cols)
        if not cols:
            return 0.0
        return best_linear_value_from_moments(
            self.Sigma[np.ix_(cols, cols)], self.c[cols], self.v)


def moments(Y: Array, theta: Array, idx: Array,
            subject_ids: Array | None = None,
            baseline: Array | None = None) -> Moments:
    r"""Second moments of the acquired variable and the target on ``idx``.

    No latent process and no measurement-error model: ``Sigma`` is the sample
    covariance of the anchors, ``c`` their covariance with the exact target and
    ``v`` the target variance.  The only regularisation is the eigenvalue floor
    of \eqref{eq:plugin}, applied because the anchor covariance is rank
    deficient: some anchors are constant across the fold and the effective rank
    is an order of magnitude below ``p``.
    """
    Ys, th = Y[idx], theta[idx]
    m = Ys.shape[0]
    if subject_ids is None:
        weights = np.full(m, 1.0 / m)
        n_subjects = m
    else:
        sid = np.asarray(subject_ids)[idx]
        _, inv, counts = np.unique(sid, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inv]
        weights = weights / weights.sum()
        n_subjects = int(counts.size)
    if baseline is not None:
        Ys, th = weighted_residuals(Ys, th, np.asarray(baseline)[idx], weights)
    mu_y = weights @ Ys
    mu_t = float(weights @ th)
    Yc, tc = Ys - mu_y, th - mu_t
    denom = max(1.0 - float(weights @ weights), 1e-12)
    residual_var = np.sum(weights[:, None] * Yc * Yc, axis=0) / denom
    degenerate = [int(j) for j in np.nonzero(
        np.sqrt(np.maximum(residual_var, 0.0)) < DEGENERATE_TOL)[0]]
    Sigma = (Yc.T * weights) @ Yc / denom
    scale = max(float(np.mean(np.diag(Sigma))), 1e-12)
    # The independent resampling unit is the subject, not the recording.  Use
    # the number of subject clusters in the m^{-1} floor when moments are
    # subject-balanced; this matches the cluster-level asymptotic unit.
    Sigma = project_psd(Sigma, floor=scale / max(n_subjects, 1))
    c = (Yc.T @ (weights * tc)) / denom
    c[degenerate] = 0.0
    v = float(np.sum(weights * tc * tc) / denom)
    # the canonical r_eff of protocol_ceiling.estimation, tr(K)/||K||_op, which
    # is the quantity the concentration rate and every other r_eff in the paper
    # use; the participation ratio tr(K)^2/tr(K^2) is a different number
    r_eff = effective_rank(Sigma)
    return Moments(Sigma=Sigma, c=c, v=v, n_records=m, n_subjects=n_subjects,
                   effective_rank=r_eff, degenerate=degenerate,
                   baseline_conditioned=baseline is not None)


def heldout_r2(Y: Array, theta: Array, tr_idx: Array, te_idx: Array,
               cols: Sequence[int], rng, subject_ids: Array | None = None,
               baseline: Array | None = None) -> float:
    r"""Out-of-sample ``R^2`` of a ridge predictor fitted on the training objects.

    The selection criterion \eqref{eq:real-objective} is a plug-in on the
    training moments and is the right thing to *optimise*; it is the wrong thing
    to *report*, because a plug-in multiple correlation recomputed on a held-out
    fold of 39 objects with up to 64 regressors can saturate as the protocol
    dimension approaches the fold size.  What is reported is therefore the
    achieved out-of-sample ``R^2`` of a predictor that never sees the held-out
    objects.
    """
    cols = list(cols)
    if not cols:
        return float("nan")
    base_weight = (None if subject_ids is None else
                   subject_base_weights(np.asarray(subject_ids)))
    train_weight = None if base_weight is None else base_weight[tr_idx]
    test_weight = None if base_weight is None else base_weight[te_idx]
    if baseline is None:
        Xtr, Xte, n_unpenalized = (Y[np.ix_(tr_idx, cols)],
                                   Y[np.ix_(te_idx, cols)], 0)
    else:
        baseline = np.asarray(baseline)
        n_unpenalized = baseline.shape[1]
        Xtr = np.column_stack([baseline[tr_idx], Y[np.ix_(tr_idx, cols)]])
        Xte = np.column_stack([baseline[te_idx], Y[np.ix_(te_idx, cols)]])
    pred = ridge_fit_predict(
        Xtr, theta[tr_idx], Xte, rng=rng,
        groups=None if subject_ids is None else subject_ids[tr_idx],
        sample_weight=train_weight, n_unpenalized=n_unpenalized)
    return float(r2_score(theta[te_idx], pred, sample_weight=test_weight))


def consecutive_cols(budget: int, p: int = P_GRID) -> list[int]:
    """``budget`` adjacent anchors centred on the fixed full grid."""
    if not 1 <= budget <= p:
        raise ValueError("budget must lie between 1 and p")
    start = (p - budget) // 2
    return list(range(start, start + budget))


def uniform_cols(budget: int, p: int = P_GRID) -> list[int]:
    """Nearest distinct anchors to equally spaced points on the fixed grid.

    This is a protocol-class comparator, not a learned design: it depends only
    on ``budget`` and the pre-specified domain ``{0, ..., p - 1}``.  In
    particular, it never reads labels, moments, or fold-specific degeneracy.
    """
    if not 1 <= budget <= p:
        raise ValueError("budget must lie between 1 and p")
    if budget == 1:
        return [(p - 1) // 2]
    cols = np.rint(np.linspace(0, p - 1, budget)).astype(int).tolist()
    if len(set(cols)) != budget:
        raise RuntimeError("equally spaced rounding did not produce distinct anchors")
    return cols


def agnostic_order(mom: Moments, method: str, n_max: int) -> list[int]:
    """Target-agnostic baselines, run on the same estimated covariance.

    These criteria do not use ``c``: integrated posterior variance is a
    property of the covariance alone, and kernel
    quadrature targets the uniform integral rather than the record's target.
    That is exactly what makes them the right comparison for a target-aware
    criterion built from the same information.
    """
    grid = TimeGrid(times=bin_midpoints(HORIZON, P_GRID),
                    weights=np.full(P_GRID, 1.0 / P_GRID), horizon=HORIZON)
    K = to_correlation(mom.Sigma)
    cands = [Action(time=float(t), width=0.0, n_segments=1, noise=0.0, cost=1.0)
             for t in grid.times]
    fn = {"imse": design_imse,
          "kernel_quadrature": design_kernel_quadrature}[method]
    res = fn(MeanLabel(), K, grid, cands, n_max)
    cols = [min(int(a.time / HORIZON * P_GRID), P_GRID - 1) for a in res.actions]
    # Budget parity.  The eigenvalue floor gives a train-constant anchor a
    # variance of scale/m, which to_correlation then rescales to one, so these
    # criteria see pure regularisation noise as a maximally unexplained
    # coordinate and spend real scored epochs on it.  The target-aware arm is
    # barred from them because c is exactly zero there; barring every arm is
    # what makes the budgets comparable.
    return [c for c in cols if c not in set(mom.degenerate)]


def subject_folds(subject_ids: Array, n_folds: int, rng: np.random.Generator
                  ) -> list[Array]:
    subs = np.array(sorted(set(subject_ids.tolist())))
    order = rng.permutation(len(subs))
    groups = np.array_split(subs[order], n_folds)
    return [np.nonzero(np.isin(subject_ids, g))[0] for g in groups]


def stratified_subject_folds(subject_ids: Array, strata: Array, n_folds: int,
                             rng: np.random.Generator) -> list[Array]:
    """Subject-disjoint folds balanced separately within each study stratum."""
    subject_ids, strata = np.asarray(subject_ids), np.asarray(strata)
    if subject_ids.shape != strata.shape:
        raise ValueError("subject_ids and strata must have the same shape")
    fold_subjects: list[list[object]] = [[] for _ in range(n_folds)]
    for level in np.unique(strata):
        idx = np.flatnonzero(strata == level)
        subs = np.unique(subject_ids[idx])
        for s in subs:
            if np.unique(strata[subject_ids == s]).size != 1:
                raise ValueError("each subject must belong to one stratum")
        pieces = np.array_split(subs[rng.permutation(subs.size)], n_folds)
        for f, piece in enumerate(pieces):
            fold_subjects[f].extend(piece.tolist())
    return [np.flatnonzero(np.isin(subject_ids, fs)) for fs in fold_subjects]


def run_cv(specs: list[LabelSpec], folds: list[Array], corpus: Corpus
           ) -> tuple[list[dict], list[dict], dict]:
    """Select on training objects and report held-out ridge ``R^2``.

    The selection score is the regularised best-linear value computed from the
    training moments.  The reported quantity is instead the pooled out-of-sample
    performance of a ridge predictor fitted on training objects and evaluated on
    held-out objects.  Neither selection nor regression fitting sees test labels.
    """
    rows: list[dict] = []
    design_rows: list[dict] = []
    cross: dict = {}
    n_rec = len(corpus.stages)
    all_idx = np.arange(n_rec)
    baseline, _ = sleep_baseline_covariates(corpus)

    for spec in specs:
        if spec.kind != "indicator":
            continue
        Y = anchor_matrix(corpus, spec.key)
        theta = spec.theta_exact
        with Timer(f"R1 {spec.key}"):
            for f, test_idx in enumerate(folds):
                train_idx = np.setdiff1d(all_idx, test_idx)
                tr = moments(Y, theta, train_idx, corpus.subject_ids, baseline)
                te = moments(Y, theta, test_idx, corpus.subject_ids, baseline)
                rng = np.random.default_rng(SEED + 1000 * f + LABEL_SEED[spec.key])

                n_max = max(BUDGETS)
                orders: dict[str, list[int]] = {}
                runtimes: dict[str, float] = {}
                for meth in ("imse", "kernel_quadrature"):
                    t0 = time.perf_counter()
                    orders[meth] = agnostic_order(tr, meth, min(P_GRID, n_max + len(tr.degenerate)))
                    runtimes[meth] = time.perf_counter() - t0

                for N in BUDGETS:
                    t0 = time.perf_counter()
                    la = best_linear_greedy(tr.Sigma, tr.c, tr.v, N,
                                            forbidden=tr.degenerate,
                                            max_swap_rounds=MAX_SWAP_ROUNDS)
                    la_rt = time.perf_counter() - t0

                    designs = {
                        "consecutive": consecutive_cols(N),
                        "uniform": uniform_cols(N),
                        "label_aware": la,
                        **{m: orders[m][:N] for m in
                           ("imse", "kernel_quadrature")},
                    }
                    rand = [sorted(rng.choice(P_GRID, size=N, replace=False).tolist())
                            for _ in range(N_RANDOM)]

                    for meth in METHODS:
                        if meth == "random":
                            te_vals = [heldout_r2(Y, theta, train_idx, test_idx, cc, rng,
                                                 corpus.subject_ids, baseline)
                                       for cc in rand]
                            I_te = float(np.mean(te_vals))
                            I_tr = float(np.mean([tr.value(cc) for cc in rand]))
                            sd_te = float(np.std(te_vals, ddof=1))
                            cols, rt = rand[0], 0.0
                        else:
                            cols = designs[meth]
                            I_te = heldout_r2(Y, theta, train_idx, test_idx, cols, rng,
                                            corpus.subject_ids, baseline)
                            I_tr = tr.value(cols)
                            sd_te = float("nan")
                            rt = la_rt if meth == "label_aware" else runtimes.get(meth, 0.0)
                        rows.append({
                            "label": spec.key, "budget": N, "method": meth, "fold": f,
                            "r2_heldout": I_te, "selection_score_train": I_tr,
                            "r2_heldout_sd_over_draws": sd_te,
                            "n_test_records": len(test_idx),
                            "n_train_records": len(train_idx),
                            "cost_epochs": float(len(cols)),
                            "runtime_s": float(rt),
                        })
                        if meth != "random":
                            design_rows.append({
                                "label": spec.key, "budget": N, "method": meth,
                                "fold": f,
                                "cols": ";".join(str(int(c)) for c in cols),
                                "times": ";".join(f"{(c + 0.5) / P_GRID:.5f}"
                                                  for c in cols),
                            })

                    if N == FIGURE_BUDGET and spec.key in PRIMARY_LABELS:
                        cross.setdefault(f, {})[spec.key] = la

                if f == 0:
                    print(f"    fold 0: r_eff(train)={tr.effective_rank:.1f}, "
                          f"r_eff(test)={te.effective_rank:.1f}, "
                          f"{len(tr.degenerate)} constant anchors, v={tr.v:.5f}")
    return rows, design_rows, cross

# --------------------------------------------------------------------------
# Cross-target transfer
# --------------------------------------------------------------------------
def cross_label_matrix(corpus: Corpus, specs: dict[str, LabelSpec],
                       folds: list[Array], cross: dict) -> tuple[Array, Array]:
    """Held-out ridge-``R^2`` transfer ratios, averaged over folds."""
    keys = list(PRIMARY_LABELS)
    acc = np.zeros((len(keys), len(keys)))
    raw = np.zeros((len(keys), len(keys)))
    all_idx = np.arange(len(corpus.stages))
    baseline, _ = sleep_baseline_covariates(corpus)
    for j, ktarget in enumerate(keys):
        Y = anchor_matrix(corpus, ktarget)
        theta = specs[ktarget].theta_exact
        for f, test_idx in enumerate(folds):
            tr_idx = np.setdiff1d(all_idx, test_idx)
            # One rng per (fold, design), not one threaded through the loop: a
            # shared generator advances between the denominator and the i == j
            # entry, so the two would be different ridge fits of the same design
            # and the diagonal of a matrix that is 1 by construction would not be.
            vals = [heldout_r2(Y, theta, tr_idx, test_idx, cross[f][kd],
                               np.random.default_rng(SEED + 7 * f + 101 * i),
                               corpus.subject_ids, baseline=baseline)
                    for i, kd in enumerate(keys)]
            own = vals[j]
            for i in range(len(keys)):
                raw[i, j] += vals[i] / len(folds)
                acc[i, j] += (vals[i] / own if own > 0 else np.nan) / len(folds)
    return acc, raw


# --------------------------------------------------------------------------
def main() -> None:
    t_start = time.perf_counter()
    rng = np.random.default_rng(SEED)

    corpus = load_corpus()
    lengths = np.array([len(s) for s in corpus.stages])
    keep = np.nonzero(lengths >= MIN_EPOCHS)[0]
    n_excluded = int(len(lengths) - len(keep))
    if n_excluded:
        corpus = Corpus(stages=[corpus.stages[i] for i in keep],
                        subject_ids=corpus.subject_ids[keep],
                        record_ids=corpus.record_ids[keep],
                        cohorts=corpus.cohorts[keep],
                        treatments=corpus.treatments[keep],
                        epoch_seconds=corpus.epoch_seconds,
                        alignment=corpus.alignment)
    n_rec = len(corpus.stages)
    n_sub = len(set(corpus.subject_ids.tolist()))
    lengths = np.array([len(s) for s in corpus.stages])
    print(f"[R1] {n_rec} recordings / {n_sub} subjects, {int(lengths.sum())} epochs, "
          f"median {np.median(lengths):.0f} ({np.median(lengths) / P_GRID:.2f} per anchor)")

    specs = build_labels(corpus)
    spec_by_key = {s.key: s for s in specs}

    label_info = {}
    for s in specs:
        Y = anchor_matrix(corpus, s.key) if s.kind == "indicator" else None
        label_info[s.key] = {
            "kind": s.kind, "description": s.description,
            "mean_target": float(s.theta_exact.mean()),
            "sd_target": float(s.theta_exact.std(ddof=1)),
            "corr_grid_vs_exact_target": float(
                np.corrcoef(s.theta_grid, s.theta_exact)[0, 1]),
            "max_abs_grid_minus_exact": float(
                np.max(np.abs(s.theta_grid - s.theta_exact))),
            "max_abs_grid_minus_exact_in_sd": float(
                np.max(np.abs(s.theta_grid - s.theta_exact))
                / max(float(np.std(s.theta_exact, ddof=1)), 1e-12)),
            "n_constant_anchors": int((Y.std(0, ddof=1) < DEGENERATE_TOL).sum())
            if Y is not None else None,
        }

    folds = stratified_subject_folds(corpus.subject_ids, corpus.cohorts,
                                     N_FOLDS, rng)
    assert sum(len(ix) for ix in folds) == n_rec
    for a in range(N_FOLDS):
        for b in range(a + 1, N_FOLDS):
            assert not (set(corpus.subject_ids[folds[a]].tolist())
                        & set(corpus.subject_ids[folds[b]].tolist())), \
                "subject leaked across folds"
    fold_info = [{"fold": f, "n_records": int(len(ix)),
                  "n_subjects": int(len(set(corpus.subject_ids[ix].tolist()))),
                  "n_sc_records": int(np.sum(corpus.cohorts[ix] == "SC")),
                  "n_st_records": int(np.sum(corpus.cohorts[ix] == "ST"))}
                 for f, ix in enumerate(folds)]
    print(f"[R1] folds: {[fi['n_records'] for fi in fold_info]} records")

    rows, design_rows, cross = run_cv(specs, folds, corpus)
    with Timer("R1 cross-target transfer"):
        cross_eff, cross_raw = cross_label_matrix(corpus, spec_by_key, folds, cross)

    # full-sample designs, recorded for reference
    full_designs = {}
    all_idx = np.arange(n_rec)
    baseline, baseline_names = sleep_baseline_covariates(corpus)
    for key in PRIMARY_LABELS:
        Y = anchor_matrix(corpus, key)
        mom = moments(Y, spec_by_key[key].theta_exact, all_idx,
                      corpus.subject_ids, baseline)
        full_designs[key] = {
            "budget": FIGURE_BUDGET,
            "cols": best_linear_greedy(mom.Sigma, mom.c, mom.v, FIGURE_BUDGET,
                                       forbidden=mom.degenerate,
                                       max_swap_rounds=MAX_SWAP_ROUNDS),
            "effective_rank": mom.effective_rank,
            "n_constant_anchors": len(mom.degenerate),
            "baseline_covariates": baseline_names,
            "selection_score": "best-linear value conditional on shared baselines",
        }

    # ---------------------------------------------------------------- aggregate
    def agg(label, budget, method, field="r2_heldout"):
        sub = [r for r in rows if r["label"] == label and r["budget"] == budget
               and r["method"] == method]
        return float(np.mean([r[field] for r in sub])), float(np.std([r[field] for r in sub], ddof=1))

    summary_rows = []
    for label in PRIMARY_LABELS:
        for N in BUDGETS:
            means = {m: agg(label, N, m)[0] for m in METHODS}
            best = max(means.values())
            for m in METHODS:
                mu, sd = agg(label, N, m)
                tr_mu, _ = agg(label, N, m, "selection_score_train")
                summary_rows.append({
                    "label": label, "budget": N, "method": m,
                    "r2_mean": mu, "r2_sd": sd, "selection_score_train_mean": tr_mu,
                    "relative_efficiency": mu / best if best > 0 else np.nan,
                    "best_method_at_budget": max(means, key=means.get),
                })

    def sm(label, N, m, field="r2_mean"):
        for r in summary_rows:
            if r["label"] == label and r["budget"] == N and r["method"] == m:
                return r[field]
        raise KeyError((label, N, m))

    agnostic = ["imse", "kernel_quadrature", "uniform"]
    headline = {
        "n_recordings": n_rec, "n_subjects": n_sub, "n_folds": N_FOLDS,
        "p_grid": P_GRID, "budgets": list(BUDGETS),
        "median_epochs_per_anchor": float(np.median(lengths) / P_GRID),
        "n_excluded_short": n_excluded,
    }
    gains, wins = [], []
    for label in PRIMARY_LABELS:
        for N in BUDGETS:
            la = sm(label, N, "label_aware")
            bag = max(sm(label, N, m) for m in agnostic)
            gains.append(la - bag)
            wins.append(la >= bag - 1e-12)
    headline["label_aware_minus_best_agnostic_mean"] = float(np.mean(gains))
    headline["label_aware_minus_best_agnostic_max"] = float(np.max(gains))
    headline["label_aware_minus_best_agnostic_min"] = float(np.min(gains))
    headline["label_aware_win_fraction"] = float(np.mean(wins))


    cons_disp = {}
    for label in PRIMARY_LABELS:
        for N in BUDGETS:
            c_, u_ = sm(label, N, "consecutive"), sm(label, N, "uniform")
            cons_disp[f"{label}_N{N}"] = {
                "consecutive": c_, "uniform_dispersed": u_,
                "ratio_dispersed_over_consecutive": u_ / c_ if c_ > 0 else np.nan,
                "absolute_gain": u_ - c_}
    headline["consecutive_vs_dispersed"] = cons_disp
    ratios = [v["ratio_dispersed_over_consecutive"] for v in cons_disp.values()]
    headline["dispersed_over_consecutive_min"] = float(np.nanmin(ratios))
    headline["dispersed_over_consecutive_max"] = float(np.nanmax(ratios))
    # A ratio of out-of-sample R^2 is not a usable summary: the contiguous
    # denominator approaches zero, so report the two levels and their difference.
    for label in PRIMARY_LABELS:
        cell = cons_disp[f"{label}_N{FIGURE_BUDGET}"]
        headline[f"consec_{label}_N{FIGURE_BUDGET}"] = cell["consecutive"]
        headline[f"disp_{label}_N{FIGURE_BUDGET}"] = cell["uniform_dispersed"]
        headline[f"gain_{label}_N{FIGURE_BUDGET}"] = cell["absolute_gain"]
    gaps = [v["absolute_gain"] for v in cons_disp.values()]
    headline["dispersed_minus_consecutive_min"] = float(min(gaps))
    headline["dispersed_minus_consecutive_max"] = float(max(gaps))

    headline["cross_label_relative_efficiency"] = {
        PRIMARY_LABELS[i]: {PRIMARY_LABELS[j]: float(cross_eff[i, j])
                            for j in range(len(PRIMARY_LABELS))}
        for i in range(len(PRIMARY_LABELS))}
    off = [cross_eff[i, j] for i in range(3) for j in range(3) if i != j]
    headline["cross_label_min_off_diagonal"] = float(np.nanmin(off))
    headline["cross_label_mean_off_diagonal"] = float(np.nanmean(off))

    def jac(a, b):
        vals = []
        for f in cross:
            A, B = set(cross[f][a]), set(cross[f][b])
            vals.append(len(A & B) / len(A | B))
        return float(np.mean(vals))
    headline["design_jaccard_between_labels"] = {
        f"{a}_vs_{b}": jac(a, b)
        for a, b in (("REM", "N3"), ("REM", "wake"), ("N3", "wake"))}
    headline["effective_rank_full_sample"] = {
        k: full_designs[k]["effective_rank"] for k in PRIMARY_LABELS}
    headline["test_fold_rank_limit"] = int(min(fi["n_records"] for fi in fold_info) - 1)

    payload = {
        "seed": SEED,
        "estimand": ("selection by the incremental best-linear value of the acquired "
                     "anchor variable after weighted training-fold residualisation on "
                     "shared baselines; reporting by baseline-adjusted held-out ridge "
                     "R2, never a plug-in recomputed on the evaluation fold"),
        "settings": {"p_grid": P_GRID, "horizon": HORIZON, "budgets": list(BUDGETS),
                     "n_folds": N_FOLDS, "n_random_draws": N_RANDOM,
                     "max_swap_rounds": MAX_SWAP_ROUNDS,
                     "min_epochs_per_recording": MIN_EPOCHS,
                     "figure_budget": FIGURE_BUDGET, "methods": METHODS},
        "corpus": {"n_recordings": n_rec, "n_subjects": n_sub,
                   "n_epochs_total": int(lengths.sum()),
                   "median_epochs": float(np.median(lengths)),
                   "min_epochs": int(lengths.min()), "max_epochs": int(lengths.max()),
                   "n_excluded_short": n_excluded, "folds": fold_info,
                   "cohort_record_counts": {
                       str(k): int(v) for k, v in zip(*np.unique(
                           corpus.cohorts, return_counts=True))},
                   "anchor_epoch_mapping": "one distinct bin midpoint per anchor"},
        "labels": label_info,
        "full_sample_designs": full_designs,
        "cross_label_relative_efficiency_matrix": cross_eff.tolist(),
        "cross_label_raw_value_matrix": cross_raw.tolist(),
        "cross_label_order": list(PRIMARY_LABELS),
        "summary": summary_rows,
        "headline": headline,
        "environment": environment_record(),
        "runtime_seconds": time.perf_counter() - t_start,
    }

    save_csv(rows, "sleep_edf")
    save_csv(summary_rows, "sleep_edf_summary")
    save_csv(design_rows, "sleep_edf_designs")
    save_json(payload, "sleep_edf")
    print(f"\n[R1] total runtime {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
