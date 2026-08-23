r"""S7 -- protocol gap vs approximation gap vs estimation gap.

This is the experiment behind the paper's central practical claim: the excess
risk of a *learned* predictor splits as

    R(f_hat_S) - R_full*  =  [R_S* - R_full*]              protocol gap
                           + [inf_{f in F} R(f) - R_S*]    approximation gap
                           + [R(f_hat_S) - inf_F R(f)]     estimation gap,

and the three terms respond to three different interventions.  Only the last
two are what machine-learning practice usually optimises, so a benchmark that
reports total ``R^2`` alone cannot tell "a big model gap under an informative
protocol" from "a small model gap under an uninformative one".

Setup
-----
Trait--state model ``Z(t) = sqrt(alpha) M + sqrt(1-alpha) X(t)`` with
``X`` an OU process of integral time ``tau = 1`` on ``[0, T]``, ``T = 20``,
discretised on ``p = 128`` midpoints; ``alpha in {0, 0.35}``.
Labels: the occupation label ``g(z) = 1{z > 0}`` and the mean label ``g(z) = z``.
Two protocols at the *same* raw-segment budget ``N = 8``:

  (A) same-time replication  ``D = 1, M = 8`` -- one occasion at ``T/2``,
      eight repeated segments, i.e. measurement noise ``nu^2 / 8``;
  (B) dispersed              ``D = 8, M = 1`` -- eight distinct occasions at
      the bin midpoints ``(j + 1/2) T / 8``, each with noise ``nu^2``.

``R_full*`` is **exactly zero** here: ``Theta = sum_j omega_j g(Z_j)`` is a
deterministic functional of the trajectory, so a predictor with access to the
whole path incurs no risk.  The protocol gap therefore equals ``R_S*`` itself,
and in ``R^2`` units (everything divided by ``V_g = Var Theta``) the
decomposition becomes a partition of one:

    1 = R^2(f_hat)  +  [I_class - R^2(f_hat)]  +  [I(S) - I_class]  +  [1 - I(S)]
                        \____estimation____/     \__approximation__/  \_protocol_/

Outputs
-------
results/s7_protocol_vs_architecture.csv   per-seed learning-curve rows
results/s7_gap_decomposition.csv          per-arm three-way decomposition
results/s7_within_between.csv             between-/within-subject split
results/s7_protocol_vs_architecture.json  headline numbers for the manuscript
figures/fig_protocol_vs_architecture.pdf  (a) learning curves, (b) stacked gaps
"""

from __future__ import annotations

import os

# Small dense problems: single-threaded BLAS in each worker beats nested
# threading.  Must be set before numpy is imported (also inherited by spawned
# multiprocessing children, which re-import this module).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common import (PALETTE, SEED, environment_record, save_csv,
                                save_figure, save_json, setup_matplotlib)
from protocol_ceiling import (MeanLabel, ThresholdLabel, ceiling_utilization,
                              dispersed_protocol, evaluate_protocol,
                              make_kernel, same_time_protocol,
                              trait_state_correlation, uniform_grid,
                              within_between_r2)
from protocol_ceiling.covariance import protocol_matrices
from protocol_ceiling.diagnostics import (kernel_ridge_fit_predict,
                                          mlp_fit_predict, r2_score,
                                          ridge_fit_predict)
from protocol_ceiling.risk import trait_state_split

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
HORIZON = 20.0
P_GRID = 128
TAU = 1.0
BUDGET = 8                    # raw segment budget N = D * M
NOISE = 1.0                   # per-segment measurement-noise variance nu^2
ALPHAS = (0.0, 0.35)
LABEL_SPECS = ("occupation", "mean")
PROTOCOLS = ("same_time", "dispersed")
MODELS = ("ridge", "gp", "mlp")

N_TRAIN = (100, 250, 500, 1000, 2500, 5000)
N_REF = 20_000                # proxy for the population optimum inf_F R(f)
N_TEST = 20_000               # held-out objects (>= 5000 required)
N_SEEDS = 10
GP_MAX_TRAIN = 1500           # kernel ridge is O(n^3); subsample above this
MLP_WEIGHT_DECAY_GRID = (1e-3, 1e-2, 1e-1, 1.0)
VAL_FRACTION = 0.25

# arm shown in the figure
MAIN_ALPHA, MAIN_LABEL = 0.0, "occupation"

# within-/between-subject arm
WB_ALPHA = 0.35
WB_HORIZONS = 2
WB_N_SUBJ_TRAIN = 2500        # x 2 horizons = 5000 training rows
WB_N_SUBJ_TEST = 5000         # x 2 horizons = 10000 test rows

MODEL_MARKER = {"ridge": "o", "gp": "s", "mlp": "^"}
MODEL_LABEL = {"ridge": "Ridge", "gp": "GP / kernel ridge", "mlp": "MLP"}
MODEL_SHORT = {"ridge": "Ridge", "gp": "GP", "mlp": "MLP"}
PROTOCOL_COLOR = {"same_time": "0.45", "dispersed": PALETTE[0]}
PROTOCOL_LABEL = {"same_time": r"Same-time $D{=}1$, $M{=}8$",
                  "dispersed": r"Dispersed $D{=}8$, $M{=}1$"}


# --------------------------------------------------------------------------
# Model / protocol plumbing
# --------------------------------------------------------------------------
def make_label(spec: str):
    return ThresholdLabel(c=0.0) if spec == "occupation" else MeanLabel()


def first_hermite_coefficient(label) -> float:
    """``E g'(U)`` for ``U ~ N(0,1)``: ``phi(c)`` for a threshold, ``1`` for the mean.

    This is the coefficient that determines the best *linear* predictor of
    ``Theta_g`` from ``Y_S``, because for jointly Gaussian ``(U, W)``
    ``Cov{g(U), W} = E g'(U) Cov(U, W)``.
    """
    if label.name == "mean":
        return 1.0
    return float(np.exp(-0.5 * label.c ** 2) / np.sqrt(2.0 * np.pi))


def label_mean(label) -> float:
    """``E Theta_g = E g(U)``: ``0`` for the mean label, ``1 - Phi(c)`` for a threshold."""
    return 0.0 if label.name == "mean" else float(1.0 - ndtr(label.c))


def make_protocol(grid, name: str):
    if name == "same_time":
        return same_time_protocol(grid, BUDGET, noise=NOISE)
    return dispersed_protocol(grid, BUDGET, noise=NOISE)


def features(Y: np.ndarray, actions, protocol: str) -> np.ndarray:
    """Feature matrix seen by the learners.

    For the dispersed protocol the observation times are appended, as the task
    specifies.  Under a *fixed* protocol these columns are constant across
    objects and therefore carry no information; they are included so that the
    learner sees the same ``(t, y)`` representation a practitioner would build,
    and the learners standardise them to exact zeros.
    """
    if protocol != "dispersed":
        return Y
    times = np.array([a.time for a in actions], dtype=float)
    return np.hstack([Y, np.tile(times, (Y.shape[0], 1))])


def bayes_predict(label, K, omega, A, R, Y) -> np.ndarray:
    """Exact posterior mean ``E(Theta_g | Y_S)`` (the Bayes predictor)."""
    KA = K @ A.T                                   # p x d
    M = A @ KA + R
    M = 0.5 * (M + M.T) + 1e-10 * np.eye(M.shape[0])
    sol = np.linalg.solve(M, KA.T)                 # d x p
    mpost = Y @ sol                                # n x p
    qdiag = np.einsum("pd,dp->p", KA, sol)         # diag(Q_S)
    if label.name == "mean":
        return mpost @ omega
    sd = np.sqrt(np.maximum(1.0 - qdiag, 1e-12))
    return (1.0 - ndtr((label.c - mpost) / sd)) @ omega


def best_linear(label, K, omega, A, R, Y) -> np.ndarray:
    """Exact population-optimal *linear* predictor of ``Theta_g`` from ``Y_S``.

    ``f(y) = E Theta_g + Cov(Theta_g, Y)^T Var(Y)^{-1} y`` with
    ``Cov(Theta_g, Y) = E g'(U) A K omega`` and ``Var(Y) = A K A^T + R``.
    For the mean label this *is* the Bayes predictor, so its approximation gap
    is exactly zero -- a built-in check on the decomposition.
    """
    M = A @ K @ A.T + R
    c = first_hermite_coefficient(label) * (A @ K @ omega)
    return label_mean(label) + Y @ np.linalg.solve(0.5 * (M + M.T), c)


def gp_fit(Xtr, ytr, Xte, rng):
    """Kernel ridge / GP with the ridge parameter chosen on a validation split.

    ``kernel_ridge_fit_predict`` defaults to ``reg = 1e-3``, which interpolates
    the training labels in this low-signal regime and gives negative test
    ``R^2``; the regulariser has to be selected, not assumed.  Training is
    subsampled to ``GP_MAX_TRAIN = 1500`` objects for tractability.
    """
    grid_reg = (1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0)
    n = len(ytr)
    if n > GP_MAX_TRAIN:
        keep = rng.permutation(n)[:GP_MAX_TRAIN]
        Xtr, ytr = Xtr[keep], ytr[keep]
        n = GP_MAX_TRAIN
    n_val = max(10, int(round(VAL_FRACTION * n)))
    perm = rng.permutation(n)
    va, tr = perm[:n_val], perm[n_val:]
    best_reg, best_err = grid_reg[0], np.inf
    for reg in grid_reg:
        pv = kernel_ridge_fit_predict(Xtr[tr], ytr[tr], Xtr[va], reg=reg)
        err = float(np.mean((ytr[va] - pv) ** 2))
        if err < best_err:
            best_reg, best_err = reg, err
    return kernel_ridge_fit_predict(Xtr, ytr, Xte, reg=best_reg), n, f"reg={best_reg:g}"


def mlp_fit(Xtr, ytr, Xte, rng):
    """Two-layer tanh MLP with the weight decay chosen on a validation split.

    Weight decay dominates every other knob here (at ``n = 250`` the test
    ``R^2`` ranges from ``-0.73`` to ``+0.31`` across the grid), so leaving it
    at the default would report a broken learner rather than an estimation gap.
    """
    n = len(ytr)
    n_val = max(10, int(round(VAL_FRACTION * n)))
    perm = rng.permutation(n)
    va, tr = perm[:n_val], perm[n_val:]
    seed = int(rng.integers(1 << 31))
    best_wd, best_err = MLP_WEIGHT_DECAY_GRID[0], np.inf
    for wd in MLP_WEIGHT_DECAY_GRID:
        pv = mlp_fit_predict(Xtr[tr], ytr[tr], Xtr[va], weight_decay=wd,
                             rng=np.random.default_rng(seed))
        err = float(np.mean((ytr[va] - pv) ** 2))
        if err < best_err:
            best_wd, best_err = wd, err
    pred = mlp_fit_predict(Xtr, ytr, Xte, weight_decay=best_wd,
                           rng=np.random.default_rng(seed))
    return pred, n, f"weight_decay={best_wd:g}"


def fit_model(name, Xtr, ytr, Xte, rng):
    if name == "ridge":
        return ridge_fit_predict(Xtr, ytr, Xte, rng=rng), len(ytr), "cv-alpha"
    if name == "gp":
        return gp_fit(Xtr, ytr, Xte, rng)
    return mlp_fit(Xtr, ytr, Xte, rng)


# --------------------------------------------------------------------------
# Exact (population) quantities -- no Monte Carlo involved
# --------------------------------------------------------------------------
def exact_table(grid) -> dict:
    """``V_g``, ``I(S)``, the exact best-linear ceiling and the state ceiling."""
    rho = make_kernel("ou", tau=TAU)
    out = {}
    for alpha, spec in product(ALPHAS, LABEL_SPECS):
        K = trait_state_correlation(grid, alpha, rho)
        label = make_label(spec)
        a1 = first_hermite_coefficient(label)
        for pname in PROTOCOLS:
            actions = make_protocol(grid, pname)
            rep = evaluate_protocol(label, K, grid, actions)
            A, R = protocol_matrices(actions, grid)
            M0 = A @ K @ A.T + R
            M = 0.5 * (M0 + M0.T)
            c = a1 * (A @ K @ grid.weights)
            i_lin = float(c @ np.linalg.solve(M, c)) / rep.total
            split = trait_state_split(label, K, A, R, grid.weights, alpha)
            out[(alpha, spec, pname)] = {
                "alpha": alpha, "label": spec, "protocol": pname,
                "n_rows": int(A.shape[0]),
                "label_variance": float(rep.total),
                "ceiling": float(rep.ceiling),
                "bayes_risk": float(rep.risk),
                "full_trajectory_risk": 0.0,
                "protocol_gap_normalised": float(1.0 - rep.ceiling),
                "ceiling_best_linear": float(i_lin),
                "ceiling_state_channel": float(split["state"]),
            }
    return out


# --------------------------------------------------------------------------
# Simulation workers
# --------------------------------------------------------------------------
def _simulate(K, n, rng):
    L = np.linalg.cholesky(K + 1e-11 * np.eye(K.shape[0]))
    return rng.standard_normal((n, K.shape[0])) @ L.T


def within_squared_corr(y, pred, subject) -> float:
    """Squared within-subject correlation, i.e. within ``R^2`` after rescaling.

    ``within_between_r2`` reports the *unrescaled* within-subject ``R^2``, which
    goes negative whenever the predictor's within-subject variance exceeds twice
    its covariance with the truth -- the situation created by measurement noise.
    The squared correlation isolates how much within-subject signal is present
    at all, independently of that miscalibration.
    """
    y, pred, subject = np.asarray(y), np.asarray(pred), np.asarray(subject)
    _, inv = np.unique(subject, return_inverse=True)
    cnt = np.bincount(inv)
    ybar = np.bincount(inv, weights=y) / cnt
    pbar = np.bincount(inv, weights=pred) / cnt
    dy, dp = y - ybar[inv], pred - pbar[inv]
    denom = float(np.sqrt(np.sum(dy ** 2) * np.sum(dp ** 2)))
    if denom <= 0:
        return float("nan")
    r = float(np.sum(dy * dp)) / denom
    return float(r ** 2)


def run_main_job(args) -> list[dict]:
    """One ``(alpha, seed)`` replication: all labels, protocols, models, sizes."""
    alpha, seed_idx = args
    grid = uniform_grid(HORIZON, P_GRID)
    rho = make_kernel("ou", tau=TAU)
    K = trait_state_correlation(grid, alpha, rho)
    omega = grid.weights
    seed_key = [SEED, int(round(1000 * alpha)), seed_idx]
    rng = np.random.default_rng(seed_key)

    n_pool = N_REF + N_TEST
    Z = _simulate(K, n_pool, rng)
    rows: list[dict] = []

    for li, spec in enumerate(LABEL_SPECS):
        label = make_label(spec)
        theta = label.apply(Z) @ omega
        for pi, pname in enumerate(PROTOCOLS):
            actions = make_protocol(grid, pname)
            A, R = protocol_matrices(actions, grid)
            eps = rng.standard_normal((n_pool, A.shape[0])) * np.sqrt(np.diag(R))
            Y = Z @ A.T + eps
            X = features(Y, actions, pname)
            Xte, yte = X[N_REF:], theta[N_REF:]

            r2_bayes = r2_score(yte, bayes_predict(label, K, omega, A, R, Y[N_REF:]))
            r2_lin = r2_score(yte, best_linear(label, K, omega, A, R, Y[N_REF:]))

            for ni, n_tr in enumerate(tuple(N_TRAIN) + (N_REF,)):
                for mi, model in enumerate(MODELS):
                    # deterministic, index-based sub-stream (never string hashing)
                    frng = np.random.default_rng(seed_key + [li, pi, ni, mi])
                    pred, n_used, hyper = fit_model(model, X[:n_tr], theta[:n_tr], Xte, frng)
                    rows.append({
                        "alpha": alpha, "label": spec, "protocol": pname,
                        "model": model, "n_train": int(n_tr),
                        "n_train_used": int(n_used),
                        "is_reference": int(n_tr == N_REF),
                        "seed_index": seed_idx, "seed_key": str(seed_key),
                        "r2_test": float(r2_score(yte, pred)),
                        "r2_bayes_test": float(r2_bayes),
                        "r2_best_linear_test": float(r2_lin),
                        "hyperparameter": hyper,
                    })
    return rows


def run_wb_job(seed_idx: int) -> list[dict]:
    """Between-/within-subject split: one subject, ``WB_HORIZONS`` label horizons."""
    grid = uniform_grid(HORIZON, P_GRID)
    rho = make_kernel("ou", tau=TAU)
    K_state = trait_state_correlation(grid, 0.0, rho)
    omega = grid.weights
    seed_key = [SEED, 777, seed_idx]
    rng = np.random.default_rng(seed_key)

    n_subj = WB_N_SUBJ_TRAIN + WB_N_SUBJ_TEST
    trait = rng.standard_normal(n_subj)[:, None]
    Zs = [np.sqrt(WB_ALPHA) * trait + np.sqrt(1.0 - WB_ALPHA) * _simulate(K_state, n_subj, rng)
          for _ in range(WB_HORIZONS)]
    subject = np.repeat(np.arange(n_subj), WB_HORIZONS)
    Z = np.empty((n_subj * WB_HORIZONS, grid.p))
    for h, Zh in enumerate(Zs):
        Z[h::WB_HORIZONS] = Zh
    n_tr = WB_N_SUBJ_TRAIN * WB_HORIZONS

    K_total = trait_state_correlation(grid, WB_ALPHA, rho)
    rows: list[dict] = []
    for li, spec in enumerate(LABEL_SPECS):
        label = make_label(spec)
        theta = label.apply(Z) @ omega
        for pi, pname in enumerate(PROTOCOLS):
            actions = make_protocol(grid, pname)
            A, R = protocol_matrices(actions, grid)
            Y = Z @ A.T + rng.standard_normal((Z.shape[0], A.shape[0])) * np.sqrt(np.diag(R))
            X = features(Y, actions, pname)
            Xte, yte, ste = X[n_tr:], theta[n_tr:], subject[n_tr:]
            bayes = bayes_predict(label, K_total, omega, A, R, Y[n_tr:])
            wb_bayes = within_between_r2(yte, bayes, ste)
            for mi, model in enumerate(MODELS):
                frng = np.random.default_rng(seed_key + [li, pi, mi])
                pred, _, hyper = fit_model(model, X[:n_tr], theta[:n_tr], Xte, frng)
                wb = within_between_r2(yte, pred, ste)
                rows.append({
                    "alpha": WB_ALPHA, "label": spec, "protocol": pname,
                    "model": model, "seed_index": seed_idx, "seed_key": str(seed_key),
                    "n_subjects_train": WB_N_SUBJ_TRAIN,
                    "n_subjects_test": WB_N_SUBJ_TEST, "n_horizons": WB_HORIZONS,
                    "r2_total": float(wb["total"]), "r2_between": float(wb["between"]),
                    "r2_within": float(wb["within"]),
                    "within_squared_corr": within_squared_corr(yte, pred, ste),
                    "bayes_r2_total": float(wb_bayes["total"]),
                    "bayes_r2_between": float(wb_bayes["between"]),
                    "bayes_r2_within": float(wb_bayes["within"]),
                    "bayes_within_squared_corr": within_squared_corr(yte, bayes, ste),
                    "hyperparameter": hyper,
                })
    return rows


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def aggregate(rows, exact) -> tuple[list[dict], dict]:
    """Per-seed three-way decomposition, then average over seeds."""
    ref = {}
    for r in rows:
        if r["is_reference"]:
            ref[(r["alpha"], r["label"], r["protocol"], r["model"], r["seed_index"])] = r["r2_test"]

    for r in rows:
        key = (r["alpha"], r["label"], r["protocol"])
        ex = exact[key]
        r["ceiling"] = ex["ceiling"]
        r["label_variance"] = ex["label_variance"]
        r["ceiling_best_linear"] = ex["ceiling_best_linear"]
        r2c = ref[(r["alpha"], r["label"], r["protocol"], r["model"], r["seed_index"])]
        r["r2_class_reference"] = r2c
        # (i) exact-anchored: protocol gap is the exact 1 - I(S).  Normalised by
        #     V_g, the four terms R^2 + estimation + approximation + protocol sum
        #     to exactly 1.
        r["protocol_gap"] = 1.0 - ex["ceiling"]
        r["approximation_gap"] = ex["ceiling"] - r2c
        r["estimation_gap"] = r2c - r["r2_test"]
        # (ii) Bayes-anchored: the same partition with the *sample* ceiling, i.e.
        #      the exact Bayes predictor scored on this very test set.  It also
        #      sums to 1, and its approximation term is a paired difference, so
        #      the test-sample fluctuation common to both predictors cancels.
        r["protocol_gap_paired"] = 1.0 - r["r2_bayes_test"]
        r["approximation_gap_paired"] = r["r2_bayes_test"] - r2c
        # (iii) for the linear class the approximation gap is available in closed
        #       form, with no Monte Carlo at all: I(S) - I_lin(S).
        r["approximation_gap_exact_linear"] = (
            ex["ceiling"] - ex["ceiling_best_linear"] if r["model"] == "ridge" else float("nan"))
        u = ceiling_utilization(r["r2_test"], ex["ceiling"])
        r["ceiling_utilization"] = u["utilization"]
        r["absolute_gap"] = u["absolute_gap"]
        r["protocol_share_of_residual"] = (
            r["protocol_gap"] / (1.0 - r["r2_test"]) if r["r2_test"] < 1.0 else float("nan"))

    keys = ("alpha", "label", "protocol", "model", "n_train")
    fields = ("r2_test", "r2_class_reference", "r2_bayes_test", "r2_best_linear_test",
              "protocol_gap", "approximation_gap", "estimation_gap",
              "protocol_gap_paired", "approximation_gap_paired",
              "ceiling_utilization", "absolute_gap", "protocol_share_of_residual")
    buckets: dict = {}
    for r in rows:
        buckets.setdefault(tuple(r[k] for k in keys), []).append(r)
    summary = []
    for k in sorted(buckets, key=lambda t: (t[0], t[1], t[2], t[3], t[4])):
        grp = buckets[k]
        rec = dict(zip(keys, k))
        ex = exact[(rec["alpha"], rec["label"], rec["protocol"])]
        rec.update({"n_seeds": len(grp), "is_reference": grp[0]["is_reference"],
                    "ceiling": ex["ceiling"], "label_variance": ex["label_variance"],
                    "ceiling_best_linear": ex["ceiling_best_linear"],
                    "ceiling_state_channel": ex["ceiling_state_channel"],
                    "approximation_gap_exact_linear":
                        grp[0]["approximation_gap_exact_linear"]})
        for f in fields:
            v = np.array([g[f] for g in grp], dtype=float)
            rec[f"{f}_mean"] = float(np.mean(v))
            rec[f"{f}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        summary.append(rec)
    lookup = {(r["alpha"], r["label"], r["protocol"], r["model"], r["n_train"]): r
              for r in summary}
    return summary, lookup


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def make_figure(lookup, exact, plt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # ---- panel (a): learning curves against the exact ceilings ---------
    ax = axes[0]
    for pname in PROTOCOLS:
        ceil = exact[(MAIN_ALPHA, MAIN_LABEL, pname)]["ceiling"]
        ax.axhline(ceil, color=PROTOCOL_COLOR[pname], ls="--", lw=1.0, alpha=0.9, zorder=1)
        ax.annotate(rf"$\mathcal{{I}}(S)={ceil:.3f}$", xy=(N_TRAIN[-1], ceil),
                    xytext=(-2, 4), textcoords="offset points", fontsize=7,
                    ha="right", va="bottom",
                    color="0.25" if pname == "same_time" else PROTOCOL_COLOR[pname])
        for model in MODELS:
            recs = [lookup[(MAIN_ALPHA, MAIN_LABEL, pname, model, n)] for n in N_TRAIN]
            y = np.array([r["r2_test_mean"] for r in recs])
            e = np.array([r["r2_test_sd"] for r in recs]) / np.sqrt(N_SEEDS)
            ax.errorbar(N_TRAIN, y, yerr=e, color=PROTOCOL_COLOR[pname],
                        marker=MODEL_MARKER[model], ms=3.6, lw=1.2, elinewidth=0.7,
                        capsize=1.5, zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel(r"training objects $n$")
    ax.set_ylabel(r"test $R^2$ for $\Theta_g$")
    ax.set_xticks(list(N_TRAIN))
    ax.set_xticklabels([str(n) for n in N_TRAIN])
    ax.set_ylim(0.0, 0.385)
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, va="top", fontsize=9)

    handles = [plt.Line2D([], [], color=PROTOCOL_COLOR[p], ls="-", lw=1.6,
                          label=PROTOCOL_LABEL[p]) for p in PROTOCOLS]
    handles.append(plt.Line2D([], [], color="0.2", ls="--", lw=1.0,
                              label=r"exact ceiling $\mathcal{I}(S)$"))
    handles += [plt.Line2D([], [], color="0.2", marker=MODEL_MARKER[m], ls="-",
                           ms=3.6, lw=1.2, label=MODEL_SHORT[m]) for m in MODELS]
    ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.58, 0.46), ncol=2,
              fontsize=7, frameon=True, framealpha=0.92, edgecolor="none",
              handlelength=1.6, columnspacing=0.9, borderpad=0.4, labelspacing=0.35)

    # ---- panel (b): stacked gap decomposition at the largest n --------
    ax = axes[1]
    n_big = N_TRAIN[-1]
    seg = [("r2_test_mean", PALETTE[2], r"explained ($R^2$)"),
           ("estimation_gap_mean", PALETTE[4], "estimation gap"),
           ("approximation_gap_mean", PALETTE[1], "approximation gap"),
           ("protocol_gap_mean", "0.45", "protocol gap")]
    xs, ticks, labels_x = [], [], []
    x = 0.0
    for pname in PROTOCOLS:
        for model in MODELS:
            xs.append((x, pname, model))
            ticks.append(x)
            labels_x.append(MODEL_LABEL[model].split(" /")[0])
            x += 1.0
        x += 0.7

    def height(pname, model, field):
        # negative segments are Monte Carlo noise around zero; draw them as zero
        return max(lookup[(MAIN_ALPHA, MAIN_LABEL, pname, model, n_big)][field], 0.0)

    for si, (field, color, lab) in enumerate(seg):
        bottoms = [sum(height(p, m, f) for f, _, _ in seg[:si]) for _, p, m in xs]
        heights = [height(p, m, field) for _, p, m in xs]
        ax.bar([xi for xi, _, _ in xs], heights, bottom=bottoms, width=0.72,
               color=color, edgecolor="white", linewidth=0.5, label=lab, zorder=2)

    # exact ceiling per protocol group, and the numbers that are too thin to see
    for pname in PROTOCOLS:
        group = [xi for xi, p, _ in xs if p == pname]
        ceil = exact[(MAIN_ALPHA, MAIN_LABEL, pname)]["ceiling"]
        ax.plot([min(group) - 0.44, max(group) + 0.44], [ceil, ceil], ls="--", lw=1.0,
                color="k", zorder=4)
    for xi, pname, model in xs:
        rec = lookup[(MAIN_ALPHA, MAIN_LABEL, pname, model, n_big)]
        gaps = max(rec["approximation_gap_mean"], 0.0) + max(rec["estimation_gap_mean"], 0.0)
        ax.annotate(f"{rec['r2_test_mean']:.3f}", xy=(xi, rec["r2_test_mean"] + 0.045),
                    ha="center", va="center", fontsize=6.5, color="white", zorder=5)
        # the two model gaps are far too thin to read off the bar; print them
        ax.annotate(f"{gaps:.3f}", xy=(xi, 1.02), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=6.5, color=PALETTE[1],
                    zorder=5, annotation_clip=False)
    ax.annotate("model gaps (approximation + estimation)", xy=(0.0, 1.105),
                xycoords="axes fraction", ha="left", va="bottom", fontsize=6.5,
                color=PALETTE[1], annotation_clip=False)

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels_x)
    ax.set_xlim(-0.65, max(t for t, _, _ in xs) + 0.65)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel(r"share of $\mathrm{Var}(\Theta_g)$")
    ax.set_xlabel(rf"model  ($n = {n_big}$)")
    for pname in PROTOCOLS:
        centre = float(np.mean([xi for xi, p, _ in xs if p == pname]))
        ax.annotate(PROTOCOL_LABEL[pname].split(" (")[0], xy=(centre, -0.20),
                    xycoords=("data", "axes fraction"), ha="center", va="top",
                    fontsize=8, annotation_clip=False)
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.62), fontsize=7, frameon=True,
              framealpha=0.92, edgecolor="none", handlelength=1.3, borderpad=0.4,
              labelspacing=0.35)
    ax.text(0.02, 0.965, "(b)", transform=ax.transAxes, va="top", fontsize=9)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.235, top=0.885, wspace=0.26)
    save_figure(fig, "fig_protocol_vs_architecture")
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> None:
    t_start = time.perf_counter()
    plt = setup_matplotlib()
    grid = uniform_grid(HORIZON, P_GRID)
    exact = exact_table(grid)

    print("[S7] exact protocol ceilings (no Monte Carlo)")
    for k in sorted(exact):
        e = exact[k]
        print(f"      alpha={e['alpha']:<5} {e['label']:<11s} {e['protocol']:<10s}"
              f" V_g={e['label_variance']:.5f}  I(S)={e['ceiling']:.4f}"
              f"  I_lin={e['ceiling_best_linear']:.4f}"
              f"  I_state={e['ceiling_state_channel']:.4f}")

    jobs = [(a, s) for a in ALPHAS for s in range(N_SEEDS)]
    n_proc = max(1, min(12, (os.cpu_count() or 4) - 2))
    print(f"[S7] running {len(jobs)} (alpha, seed) jobs on {n_proc} processes")
    with Pool(n_proc) as pool:
        rows = [r for chunk in pool.map(run_main_job, jobs) for r in chunk]
        print(f"[S7] {len(rows)} learning-curve rows; within/between arm "
              f"(alpha={WB_ALPHA}, {WB_HORIZONS} horizons per subject)")
        wb_rows = [r for chunk in pool.map(run_wb_job, list(range(N_SEEDS))) for r in chunk]

    summary, lookup = aggregate(rows, exact)
    save_csv(rows, "s7_protocol_vs_architecture")
    save_csv(summary, "s7_gap_decomposition")

    # -- within / between ------------------------------------------------
    wb_keys = ("alpha", "label", "protocol", "model")
    wb_buckets: dict = {}
    for r in wb_rows:
        wb_buckets.setdefault(tuple(r[k] for k in wb_keys), []).append(r)
    wb_summary = []
    for k in sorted(wb_buckets):
        grp = wb_buckets[k]
        rec = dict(zip(wb_keys, k))
        ex = exact[(WB_ALPHA, rec["label"], rec["protocol"])]
        rec.update({"n_seeds": len(grp), "ceiling_total": ex["ceiling"],
                    "ceiling_state_channel": ex["ceiling_state_channel"]})
        for f in ("r2_total", "r2_between", "r2_within", "within_squared_corr",
                  "bayes_r2_total", "bayes_r2_between", "bayes_r2_within",
                  "bayes_within_squared_corr"):
            v = np.array([g[f] for g in grp], dtype=float)
            rec[f"{f}_mean"] = float(np.mean(v))
            rec[f"{f}_sd"] = float(np.std(v, ddof=1))
        wb_summary.append(rec)
    save_csv(wb_rows, "s7_within_between")

    # -- headline numbers -------------------------------------------------
    def best_model(alpha, spec, pname, n):
        cands = [lookup[(alpha, spec, pname, m, n)] for m in MODELS]
        return max(cands, key=lambda r: r["r2_test_mean"])

    n_small, n_big = N_TRAIN[0], N_TRAIN[-1]
    hi = best_model(MAIN_ALPHA, MAIN_LABEL, "dispersed", n_small)      # high ceiling, small n
    hi_big = best_model(MAIN_ALPHA, MAIN_LABEL, "dispersed", n_big)
    lo = best_model(MAIN_ALPHA, MAIN_LABEL, "same_time", n_big)        # low ceiling, large n
    lo_small = best_model(MAIN_ALPHA, MAIN_LABEL, "same_time", n_small)

    headline = {
        "arm": {"alpha": MAIN_ALPHA, "label": MAIN_LABEL, "horizon": HORIZON,
                "p": P_GRID, "tau": TAU, "budget_N": BUDGET, "noise": NOISE},
        "full_trajectory_risk": 0.0,
        "ceiling_same_time": exact[(MAIN_ALPHA, MAIN_LABEL, "same_time")]["ceiling"],
        "ceiling_dispersed": exact[(MAIN_ALPHA, MAIN_LABEL, "dispersed")]["ceiling"],
        "ceiling_ratio_dispersed_over_same_time":
            exact[(MAIN_ALPHA, MAIN_LABEL, "dispersed")]["ceiling"]
            / exact[(MAIN_ALPHA, MAIN_LABEL, "same_time")]["ceiling"],
        "same_time_large_n": {
            "model": lo["model"], "n_train": n_big, "r2": lo["r2_test_mean"],
            "r2_sd": lo["r2_test_sd"], "ceiling": lo["ceiling"],
            "ceiling_utilization": lo["ceiling_utilization_mean"],
            "absolute_gap": lo["absolute_gap_mean"],
            "protocol_gap": lo["protocol_gap_mean"],
            "approximation_gap": lo["approximation_gap_mean"],
            "estimation_gap": lo["estimation_gap_mean"],
            "protocol_share_of_residual": lo["protocol_share_of_residual_mean"],
            "model_gaps_total": lo["approximation_gap_mean"] + lo["estimation_gap_mean"],
        },
        "same_time_small_n": {
            "model": lo_small["model"], "n_train": n_small,
            "r2": lo_small["r2_test_mean"],
            "ceiling_utilization": lo_small["ceiling_utilization_mean"],
        },
        "dispersed_small_n": {
            "model": hi["model"], "n_train": n_small, "r2": hi["r2_test_mean"],
            "r2_sd": hi["r2_test_sd"], "ceiling": hi["ceiling"],
            "ceiling_utilization": hi["ceiling_utilization_mean"],
            "absolute_gap": hi["absolute_gap_mean"],
            "protocol_gap": hi["protocol_gap_mean"],
            "approximation_gap": hi["approximation_gap_mean"],
            "estimation_gap": hi["estimation_gap_mean"],
            "protocol_share_of_residual": hi["protocol_share_of_residual_mean"],
            "model_gaps_total": hi["approximation_gap_mean"] + hi["estimation_gap_mean"],
        },
        "dispersed_large_n": {
            "model": hi_big["model"], "n_train": n_big, "r2": hi_big["r2_test_mean"],
            "ceiling_utilization": hi_big["ceiling_utilization_mean"],
            "absolute_gap": hi_big["absolute_gap_mean"],
        },
        "r2_gain_from_protocol_at_large_n": hi_big["r2_test_mean"] - lo["r2_test_mean"],
        "r2_gain_from_data_same_time_100_to_5000":
            lo["r2_test_mean"] - lo_small["r2_test_mean"],
        "validation": {
            "bayes_r2_test_vs_exact_ceiling_same_time":
                [lo["r2_bayes_test_mean"], lo["ceiling"]],
            "bayes_r2_test_vs_exact_ceiling_dispersed":
                [hi["r2_bayes_test_mean"], hi["ceiling"]],
            "ridge_approximation_gap_exact_vs_estimated_dispersed":
                [lookup[(MAIN_ALPHA, MAIN_LABEL, "dispersed", "ridge", n_big)]
                 ["approximation_gap_exact_linear"],
                 lookup[(MAIN_ALPHA, MAIN_LABEL, "dispersed", "ridge", n_big)]
                 ["approximation_gap_paired_mean"]],
            "ridge_approximation_gap_exact_vs_estimated_same_time":
                [lookup[(MAIN_ALPHA, MAIN_LABEL, "same_time", "ridge", n_big)]
                 ["approximation_gap_exact_linear"],
                 lookup[(MAIN_ALPHA, MAIN_LABEL, "same_time", "ridge", n_big)]
                 ["approximation_gap_paired_mean"]],
        },
    }

    per_arm = {}
    for k in sorted(exact):
        alpha, spec, pname = k
        rec_big = best_model(alpha, spec, pname, n_big)
        per_arm[f"alpha={alpha}|{spec}|{pname}"] = {
            **exact[k],
            "best_model_at_n5000": rec_big["model"],
            "r2_at_n5000": rec_big["r2_test_mean"],
            "ceiling_utilization_at_n5000": rec_big["ceiling_utilization_mean"],
            "absolute_gap_at_n5000": rec_big["absolute_gap_mean"],
            "approximation_gap_at_n5000": rec_big["approximation_gap_mean"],
            "estimation_gap_at_n5000": rec_big["estimation_gap_mean"],
            "r2_bayes_test_at_n5000": rec_big["r2_bayes_test_mean"],
        }

    wb_head = {}
    for rec in wb_summary:
        wb_head[f"{rec['label']}|{rec['protocol']}|{rec['model']}"] = {
            "r2_total": rec["r2_total_mean"], "r2_between": rec["r2_between_mean"],
            "r2_within": rec["r2_within_mean"], "r2_within_sd": rec["r2_within_sd"],
            "within_squared_corr": rec["within_squared_corr_mean"],
            "bayes_r2_total": rec["bayes_r2_total_mean"],
            "bayes_r2_between": rec["bayes_r2_between_mean"],
            "bayes_r2_within": rec["bayes_r2_within_mean"],
            "bayes_within_squared_corr": rec["bayes_within_squared_corr_mean"],
            "ceiling_total": rec["ceiling_total"],
            "ceiling_state_channel": rec["ceiling_state_channel"],
        }

    runtime = time.perf_counter() - t_start
    payload = {
        "experiment": "S7 protocol vs approximation vs estimation gap",
        "seed": SEED, "n_seeds": N_SEEDS,
        "config": {
            "horizon": HORIZON, "p": P_GRID, "kernel": f"ou(tau={TAU})",
            "alphas": list(ALPHAS), "labels": list(LABEL_SPECS),
            "budget_N": BUDGET, "per_segment_noise": NOISE,
            "protocols": {"same_time": "D=1, M=8 at t=T/2",
                          "dispersed": "D=8, M=1 at bin midpoints"},
            "n_train": list(N_TRAIN), "n_reference": N_REF, "n_test": N_TEST,
            "models": list(MODELS), "gp_max_train": GP_MAX_TRAIN,
            "mlp_weight_decay_grid": list(MLP_WEIGHT_DECAY_GRID),
            "validation_fraction": VAL_FRACTION,
        },
        "notes": {
            "full_trajectory_risk":
                "R_full* = 0 exactly: Theta = sum_j omega_j g(Z_j) is a deterministic "
                "functional of the trajectory, so the protocol gap equals R_S* itself "
                "and, normalised by Var(Theta), equals 1 - I(S).",
            "class_risk_proxy":
                f"inf_F R(f) is approximated by the same learner trained on n = {N_REF} "
                "objects; for the GP this is capped at 1500 training points, so its "
                "approximation gap also absorbs the subsampling cap.",
            "dispersed_time_features":
                "Observation times are appended to the dispersed feature vector; under a "
                "fixed protocol they are constant across objects and carry no information.",
            "two_anchors":
                "The gap columns come in two internally exact variants. The reported "
                "'protocol_gap / approximation_gap / estimation_gap' are anchored at the "
                "exact ceiling I(S); the '*_paired' columns are anchored at the exact "
                "Bayes predictor scored on the same test sample, which makes the "
                "approximation term a paired difference and removes the shared "
                "test-sample fluctuation. Both partitions sum to 1 - R^2. Where the "
                "exact-anchored approximation gap is slightly negative it is zero to "
                "within Monte Carlo error; the closed-form value for the linear class, "
                "I(S) - I_lin(S), is given in 'approximation_gap_exact_linear'.",
        },
        "exact_ceilings": {f"alpha={a}|{l}|{p}": exact[(a, l, p)] for (a, l, p) in exact},
        "per_arm": per_arm,
        "headline": headline,
        "within_between": wb_head,
        "environment": environment_record(),
        "runtime_seconds": runtime,
    }
    save_json(payload, "s7_protocol_vs_architecture")
    make_figure(lookup, exact, plt)

    print("\n[S7] headline")
    print(f"      same-time  n={n_big}: R2={lo['r2_test_mean']:.4f} "
          f"ceiling={lo['ceiling']:.4f} utilisation={lo['ceiling_utilization_mean']:.3f} "
          f"gap={lo['absolute_gap_mean']:.4f}  ({lo['model']})")
    print(f"      dispersed  n={n_small}: R2={hi['r2_test_mean']:.4f} "
          f"ceiling={hi['ceiling']:.4f} utilisation={hi['ceiling_utilization_mean']:.3f} "
          f"gap={hi['absolute_gap_mean']:.4f}  ({hi['model']})")
    print(f"      protocol change buys {headline['r2_gain_from_protocol_at_large_n']:+.4f} R2 "
          f"at n={n_big}; 50x more data under the same-time protocol buys "
          f"{headline['r2_gain_from_data_same_time_100_to_5000']:+.4f}")
    print(f"[S7] total runtime {runtime:.1f}s")


if __name__ == "__main__":
    main()
