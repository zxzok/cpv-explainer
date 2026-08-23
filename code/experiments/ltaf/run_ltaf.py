"""Real-data experiment 2 -- atrial-fibrillation burden (Long-Term AF Database).

The label is
::

    Theta_AF = (time spent in AF) / (analysable recording time),

i.e. the *literal* occupation time of the rhythm process: an equally weighted
time average ``Theta = sum_j omega_j g(Z_j)`` of a 0/1 occupation indicator.
Nothing about it is a modelling convenience -- it is the quantity a clinician
reads off a Holter report, and every AF-screening protocol in use is an
observation protocol ``S`` for exactly this ``Theta``.

``indicator_linear``
    Take the per-bin AF fraction series itself as the process, standardise it
    across records (``Z_j = (X_j - mu_j)/sd_j``) and estimate ``K`` with
    :func:`fit_covariance`.  For the discretised burden,
    ``Theta_AF^(p) = mean_j X_j = mean_j mu_j + (sum_j sd_j / p) omega^T Z``
    with ``omega_j = sd_j / sum_k sd_k``.  Zero-variance bins have zero weight.
    Thus location and non-zero scale are the only changes to the target, and the
    raw protocol row becomes ``A diag(sd)`` in standardised coordinates. The
    reported discretisation diagnostic therefore evaluates the full-sample
    empirical best-linear value directly in the raw bin-fraction coordinates.
    The earlier homogeneous-threshold sensitivity construction is not reported:
    one pooled threshold cannot reproduce bin-specific marginal prevalences and
    therefore is not a fitted latent binary model.

Outputs
-------
``results/ltaf.json``              headline numbers cited by the manuscript
``results/ltaf.csv``               budget x width x method protocol table
``results/ltaf_autocorr.csv``      empirical rho(u) of the AF indicator
``results/ltaf_covariance.csv``    effective rank / eigenvalue shares
``results/ltaf_duration.csv``      equal-duration contiguous vs intermittent
``results/ltaf_headline.csv``      clinical protocols with bootstrap intervals
``results/ltaf_cv.csv``            record-level repeated 5-fold cross-validation
``results/ltaf_resolution.csv``    p = 64 / 128 / 256 discretisation check
``results/ltaf_noise.csv``         measurement-noise sensitivity
``figures/fig_ltaf.pdf/.png``      (a) rho(u) with OU fit, (b) value vs
                                   24-hour-equivalent observed fraction

Run with::

    .venv/bin/python experiments/ltaf/run_ltaf.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import (METHOD_STYLE, PALETTE, RESULTS, SEED, Timer,
                                environment_record, save_csv, save_figure,
                                save_json, setup_matplotlib)
from protocol_ceiling import (Action, MeanLabel, TimeGrid,
                              design_imse, design_kernel_quadrature,
                              design_mutual_information, design_random,
                              design_uniform, dispersed_protocol,
                              effective_rank, evaluate_protocol,
                              fit_covariance, select_protocol_greedy)
from protocol_ceiling.covariance import candidate_actions, protocol_matrices
from protocol_ceiling.diagnostics import r2_score, ridge_fit_predict

Array = np.ndarray

# --------------------------------------------------------------------------
# Fixed experiment constants (every replication count is declared here)
# --------------------------------------------------------------------------
HORIZON_H = 24.0            # reference horizon for relative-time coordinates
P_MAIN = 128                # latent grid points (primary discretisation)
P_CHECK = (64, 128, 256)    # discretisation check
NOISE = 0.01                # nu^2 in units of the standardised latent variance
NOISE_LEVELS = (0.0, 0.01, 0.1)
BUDGETS = (1, 2, 4, 8, 16, 32)
WIDTH_MIN = (1.0, 5.0, 15.0)          # window durations in minutes
DISPERSION_K = (1, 2, 4, 8, 16, 32)   # windows in the equal-duration family
EQUAL_DURATION_WIDTH_MIN = 15.0       # window length in the equal-duration family
N_CANDIDATE_TIMES = 48      # |V| for the design algorithms
N_RANDOM_RESTARTS = 20      # "random spacing" = best of 20 draws (as in S5)
N_BOOTSTRAP = 1000          # record-level bootstrap replicates
N_CV_REPEATS = 5            # repeated 5-fold CV
N_CV_FOLDS = 5
CV_WIDTH_MIN = 15.0         # window duration used in the CV protocol sweep
LABEL_MODELS = ("indicator_linear",)
COHORTS = ("strict", "all", "mixed")
PRIMARY_COHORT = "all"
PRIMARY_MODEL = "indicator_linear"
PSEUDOINVERSE_RCOND = 1e-10

DESIGN_METHODS = ("contiguous", "contiguous_block", "dispersed", "uniform",
                  "random", "mutual_information", "imse", "kernel_quadrature",
                  "label_aware")
NESTED_METHODS = ("mutual_information", "imse", "kernel_quadrature")

STYLE = dict(METHOD_STYLE)
STYLE["contiguous"] = dict(color=PALETTE[1], marker="s", ls="-",
                           label="Contiguous (1 block)")
STYLE["contiguous_block"] = dict(color=PALETTE[1], marker="s", ls="--",
                                 label="Contiguous (block average)")
STYLE["dispersed"] = dict(color=PALETTE[0], marker="o", ls="-",
                          label="Intermittent (dispersed)")


# ==========================================================================
# 1.  Data
# ==========================================================================
def load_af() -> tuple[list[Array], Array, int]:
    """Load the prepared 60-second AF series."""
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "ltaf" / "af_series.npz"
    if not path.exists():
        raise SystemExit(
            f"missing {path}; regenerate with experiments/fetch_data.py")
    d = np.load(path, allow_pickle=True)
    af = [np.asarray(x, dtype=float) for x in d["af"]]
    ids = np.asarray(d["record_ids"])
    epoch_seconds = int(d["epoch_seconds"])
    if len(af) == 0 or epoch_seconds != 60:
        raise SystemExit(f"malformed af_series.npz (n={len(af)}, "
                         f"epoch_seconds={epoch_seconds})")
    return af, ids, epoch_seconds


def cohort_index(burden: Array, name: str) -> Array:
    """Record subsets.

    ``all``     every record; the primary deployment cohort.
    ``strict``  0 < burden < 1 -- outcome-defined secondary sensitivity.
    ``mixed``   0.05 <= burden <= 0.95 -- records where a sparse protocol has
                to resolve genuine within-record structure rather than merely
                notice that the patient is permanently in (or out of) AF.
    """
    if name == "all":
        return np.arange(burden.size)
    if name == "strict":
        return np.where((burden > 0.0) & (burden < 1.0))[0]
    if name == "mixed":
        return np.where((burden >= 0.05) & (burden <= 0.95))[0]
    raise ValueError(name)


def resample_series(x: Array, p: int) -> Array:
    """Mean AF indicator per bin on a common normalised-time grid of ``p`` bins."""
    n = x.size
    idx = np.minimum((np.arange(n) * p) // n, p - 1)
    tot = np.zeros(p)
    cnt = np.zeros(p)
    np.add.at(tot, idx, x)
    np.add.at(cnt, idx, 1.0)
    if np.any(cnt == 0):
        raise ValueError(f"empty bin at p={p} for a series of length {n}")
    return tot / cnt


def design_matrix(af: list[Array], idx: Array, p: int) -> Array:
    return np.stack([resample_series(af[i], p) for i in idx])


# ==========================================================================
# 2.  Empirical affine model construction
# ==========================================================================
class Model:
    """The empirical affine AF-burden model on one set of records."""

    def __init__(self, W: Array, label_model: str, horizon: float = HORIZON_H,
                 ref: "Model | None" = None):
        self.label_model = label_model
        if label_model != "indicator_linear":
            raise ValueError("only the empirical affine AF-burden model is supported")
        self.horizon = float(horizon)
        m, p = W.shape
        self.m, self.p = m, p
        mu = W.mean(axis=0)
        sd = W.std(axis=0, ddof=1)
        scale = np.where(sd > 0.0, sd, 1.0)
        self.mu, self.sd = mu, sd
        times = (np.arange(p) + 0.5) * self.horizon / p
        # K is the correlation of the standardised bin-fraction series.
        # Constant bins are represented by a zero column and receive zero
        # target weight, so the affine identity remains exact.
        self.fit = fit_covariance((W - mu) / scale)
        self.raw_cov = np.cov(W, rowvar=False)
        theta = W.mean(axis=1)
        self.raw_c = np.array([np.cov(W[:, j], theta)[0, 1] for j in range(p)])
        self.raw_v = float(np.var(theta, ddof=1))
        self.label = MeanLabel()
        # Theta_AF = sum_j (1/p) X_j = const + sum_j (sd_j/p) Z_j, so the
        # label weights of the standardised process are proportional to sd.
        weights = (ref.raw_weights if ref is not None else sd)
        self.raw_weights = np.asarray(weights, dtype=float)
        if not np.any(self.raw_weights > 0.0):
            raise ValueError("AF fraction target has no non-constant bins")
        self.K = self.fit.K
        self.prevalence = float(W.mean())
        self.c = float("nan")
        w = self.raw_weights / self.raw_weights.sum()
        self.grid = TimeGrid(times=times, weights=w, horizon=self.horizon)
        self.bin_minutes = 60.0 * self.horizon / p

    # -- diagnostics ----------------------------------------------------
    def spectrum(self) -> dict:
        ev = np.linalg.eigvalsh(self.K)[::-1]
        tot = float(ev.sum())
        return {
            "effective_rank": float(effective_rank(self.K)),
            "leading_eigenvalue_share": float(ev[0] / tot),
            "top5_eigenvalue_share": float(ev[:5].sum() / tot),
            "mean_offdiag_correlation":
                float(self.K[np.triu_indices(self.p, 1)].mean()),
            "label_variance": float(evaluate_protocol(
                self.label, self.K, self.grid, []).total),
        }

    def ceiling(self, actions) -> float:
        # Exact best-linear value of the *raw* window averages. If A averages
        # F over a window, its row in standardised Z coordinates would be
        # A diag(sd), not A. Working in raw coordinates makes that transform
        # explicit and avoids changing the protocol during standardisation.
        A, _ = protocol_matrices(actions, self.grid)
        return self.raw_ceiling(A)

    def raw_ceiling(self, A: Array) -> float:
        """Empirical best-linear value for an explicitly supplied raw row matrix."""
        M = A @ self.raw_cov @ A.T
        c = A @ self.raw_c
        if self.raw_v <= 0 or c.size == 0:
            return 0.0
        return float(np.clip(c @ np.linalg.pinv(
            M, rcond=PSEUDOINVERSE_RCOND) @ c / self.raw_v,
                             0.0, 1.0))


# ==========================================================================
# 3.  Protocol constructors
# ==========================================================================
def _act(t: float, width_h: float, noise: float, tag: str,
         n_segments: int = 1) -> Action:
    return Action(time=float(t), width=float(width_h), n_segments=int(n_segments),
                  noise=float(noise), cost=1.0, tag=tag)


def contiguous_windows(n: int, width_h: float, noise: float = NOISE,
                       horizon: float = HORIZON_H) -> list[Action]:
    """``n`` *adjacent* windows forming one uninterrupted block of length ``n*w``.

    Matched comparator: same number of measurements, same observed fraction and
    same measurement noise as the dispersed protocol; only placement differs.
    """
    centre = 0.5 * horizon
    offs = (np.arange(n) - 0.5 * (n - 1)) * width_h
    return [_act(centre + o, width_h, noise, "contiguous") for o in offs]


def contiguous_block(total_h: float, n_segments: int = 1, noise: float = NOISE,
                     horizon: float = HORIZON_H) -> list[Action]:
    """One window of length ``total_h``: the clinically literal "monitor for D hours"."""
    return [_act(0.5 * horizon, min(total_h, horizon), noise, "contiguous_block",
                 n_segments=n_segments)]


def dispersed_windows(grid: TimeGrid, n: int, width_h: float,
                      noise: float = NOISE) -> list[Action]:
    return dispersed_protocol(grid, n, noise=noise, width=width_h)


def fractional_window_matrix(actions: list[Action], grid: TimeGrid) -> Array:
    """Window averages using exact overlap with piecewise-constant grid bins.

    This mapping is used by the discretisation diagnostic so the nominal
    one-hour, six-hour and 15-minute supports do not change duration when the
    grid changes. Boundary bins receive their fractional overlap weight.
    """
    edges = np.linspace(0.0, grid.horizon, grid.p + 1)
    rows = []
    for action in actions:
        if action.width <= 0.0:
            row = np.zeros(grid.p)
            row[int(np.argmin(np.abs(grid.times - action.time)))] = 1.0
        else:
            lo = max(0.0, action.time - 0.5 * action.width)
            hi = min(grid.horizon, action.time + 0.5 * action.width)
            overlap = np.maximum(0.0, np.minimum(edges[1:], hi)
                                 - np.maximum(edges[:-1], lo))
            if overlap.sum() <= 0.0:
                raise ValueError("window has no overlap with the AF grid")
            row = overlap / overlap.sum()
        rows.append(row)
    return np.stack(rows)


def candidates_for(grid: TimeGrid, width_h: float, noise: float = NOISE) -> list[Action]:
    return candidate_actions(grid, n_times=N_CANDIDATE_TIMES, widths=(width_h,),
                             segments=(1,), noise=noise, cost_fixed=1.0)


def label_aware_design(model: Model, cands, budget: int):
    """Algorithm 1: label-aware greedy followed by the 1-swap refinement."""
    return select_protocol_greedy(model.label, model.K, model.grid, cands,
                                  budget=float(budget), cost_aware=True,
                                  local_search=True)


def nested_designs(model: Model, cands, n_max: int) -> dict[str, list[Action]]:
    """Mutual-information / IMSE / kernel-quadrature selections at ``n_max``.

    All three are pure greedies, so the size-``N`` selection is the length-``N``
    prefix of the size-``n_max`` selection; running them once and slicing is
    exact, not an approximation.
    """
    out = {}
    out["mutual_information"] = design_mutual_information(
        model.label, model.K, model.grid, cands, n_max).actions
    out["imse"] = design_imse(model.label, model.K, model.grid, cands, n_max).actions
    out["kernel_quadrature"] = design_kernel_quadrature(
        model.label, model.K, model.grid, cands, n_max).actions
    return out


# ==========================================================================
# 4.  Empirical autocorrelation of the AF indicator (absolute time)
# ==========================================================================
def pooled_autocorrelation(series: list[Array], lags: Array,
                           within_record: bool = False) -> Array:
    """Pooled ``rho(u)`` of the 60-second AF indicator.

    With ``within_record=False`` the grand mean is removed, so ``rho(u)`` retains
    the between-record ("trait") component and plateaus at the between-record
    variance share; with ``within_record=True`` each record's own mean is removed
    and what is left is the state autocorrelation ``rho_state(u)``.
    """
    if within_record:
        xs = [x - x.mean() for x in series]
    else:
        g = float(np.concatenate(series).mean())
        xs = [x - g for x in series]
    var = float(np.concatenate([x * x for x in xs]).mean())
    out = np.empty(lags.size)
    for i, u in enumerate(lags):
        s, n = 0.0, 0
        for x in xs:
            if x.size <= u:
                continue
            a, b = x[:x.size - u], x[u:]
            s += float(a @ b)
            n += a.size
        out[i] = (s / n) / var if n else np.nan
    return out


def trait_ou_fit(lag_h: Array, rho: Array) -> dict:
    """Fit ``rho(u) = alpha + (1 - alpha) exp(-u / tau_1)`` and the pure OU form."""
    def trait_ou(u, alpha, tau):
        return alpha + (1.0 - alpha) * np.exp(-u / tau)

    def pure_ou(u, tau):
        return np.exp(-u / tau)

    ok = np.isfinite(rho)
    u, r = lag_h[ok], rho[ok]
    p1, _ = curve_fit(trait_ou, u, r, p0=[0.6, 3.0],
                      bounds=([0.0, 1e-3], [0.999, 1e3]), maxfev=20000)
    p2, _ = curve_fit(pure_ou, u, r, p0=[5.0], bounds=([1e-3], [1e4]), maxfev=20000)
    ss = float(np.sum((r - r.mean()) ** 2))
    return {
        "alpha": float(p1[0]),
        "tau1_hours": float(p1[1]),
        "trait_ou_r2": float(1.0 - np.sum((r - trait_ou(u, *p1)) ** 2) / ss),
        "pure_ou_tau_hours": float(p2[0]),
        "pure_ou_r2": float(1.0 - np.sum((r - pure_ou(u, *p2)) ** 2) / ss),
        "fit": lambda x: trait_ou(np.asarray(x, dtype=float), *p1),
        "pure_fit": lambda x: pure_ou(np.asarray(x, dtype=float), *p2),
    }


def between_record_share(series: list[Array]) -> float:
    """Between-record share of the total epoch-level indicator variance."""
    g = float(np.concatenate(series).mean())
    n_tot = sum(x.size for x in series)
    between = sum(x.size * (x.mean() - g) ** 2 for x in series) / n_tot
    total = float(np.concatenate([(x - g) ** 2 for x in series]).mean())
    return float(between / total)


# ==========================================================================
# 5.  Bootstrap over records
# ==========================================================================
def bootstrap_ceilings(W: Array, label_model: str, protocols: dict[str, list[Action]],
                       n_boot: int, rng: np.random.Generator) -> dict[str, Array]:
    """Record-level bootstrap of ``I_g(S)`` for a family of *fixed* protocols.

    Records are the independent replication unit, so every resample is taken at
    the record level; the full affine model is refitted inside each replicate.
    """
    m = W.shape[0]
    reps = {k: np.empty(n_boot) for k in protocols}
    for b in range(n_boot):
        idx = rng.integers(0, m, size=m)
        try:
            mb = Model(W[idx], label_model)
        except Exception:            # pragma: no cover - degenerate resample
            for k in reps:
                reps[k][b] = np.nan
            continue
        for k, acts in protocols.items():
            reps[k][b] = mb.ceiling(acts)
    return reps


def percentile_ci(x: Array, level: float = 0.95) -> tuple[float, float]:
    a = 0.5 * (1.0 - level)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, a)), float(np.quantile(x, 1.0 - a))


# ==========================================================================
# 6.  Empirical out-of-sample predictor (the ceiling is supposed to bound it)
# ==========================================================================
def empirical_oos_r2(W: Array, y: Array, grid: TimeGrid, actions: list[Action],
                     folds: Array, rng: np.random.Generator) -> float:
    """Cross-validated ``R^2`` of a ridge predictor of burden from the windows."""
    A, _ = protocol_matrices(actions, grid)
    X = W @ A.T
    pred = np.zeros(y.size)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        pred[te] = ridge_fit_predict(X[tr], y[tr], X[te], rng=rng)
    return float(r2_score(y, pred))


# ==========================================================================
# Main
# ==========================================================================
def main() -> None:
    t_start = time.perf_counter()
    plt = setup_matplotlib()
    rng = np.random.default_rng(SEED)

    af, ids, epoch_seconds = load_af()
    burden = np.array([float(x.mean()) for x in af])
    lengths = np.array([x.size for x in af])
    print(f"[data] {len(af)} records, {epoch_seconds}s epochs, "
          f"{lengths.sum() * epoch_seconds / 3600:.1f} h total")

    idx_by_cohort = {c: cohort_index(burden, c) for c in COHORTS}
    excluded = np.setdiff1d(idx_by_cohort["all"], idx_by_cohort["strict"])
    print(f"[cohorts] " + ", ".join(f"{c}={idx_by_cohort[c].size}" for c in COHORTS))
    print(f"[cohorts] excluded by 0<burden<1: {list(ids[excluded])} "
          f"(burden {np.round(burden[excluded], 4).tolist()})")

    W_by_cohort_p = {(c, p): design_matrix(af, idx_by_cohort[c], p)
                     for c in COHORTS for p in P_CHECK}
    y_by_cohort = {c: W_by_cohort_p[(c, P_MAIN)].mean(axis=1) for c in COHORTS}

    headline: dict = {
        "seed": SEED,
        "n_records_total": int(len(af)),
        "epoch_seconds": epoch_seconds,
        "horizon_hours": HORIZON_H,
        "p_main": P_MAIN,
        "bin_minutes_p128": 60.0 * HORIZON_H / P_MAIN,
        "noise_variance": NOISE,
        "n_bootstrap": N_BOOTSTRAP,
        "n_cv_repeats": N_CV_REPEATS,
        "n_cv_folds": N_CV_FOLDS,
        "n_candidate_times": N_CANDIDATE_TIMES,
        "cohort_sizes": {c: int(idx_by_cohort[c].size) for c in COHORTS},
        "excluded_record_ids": [str(s) for s in ids[excluded]],
        "excluded_record_burdens": [float(b) for b in burden[excluded]],
        "mean_burden_strict": float(burden[idx_by_cohort["strict"]].mean()),
        "median_burden_strict": float(np.median(burden[idx_by_cohort["strict"]])),
    }

    # ------------------------------------------------------------------
    # (A)  Empirical autocorrelation of the AF indicator
    # ------------------------------------------------------------------
    with Timer("autocorrelation"):
        series = [af[i] for i in idx_by_cohort[PRIMARY_COHORT]]
        lags = np.unique(np.concatenate([
            np.arange(0, 61),
            np.arange(60, 361, 5),
            np.arange(360, 1201, 15)]).astype(int))
        lag_h = lags * epoch_seconds / 3600.0
        rho_total = pooled_autocorrelation(series, lags, within_record=False)
        rho_state = pooled_autocorrelation(series, lags, within_record=True)
        fit_total = trait_ou_fit(lag_h, rho_total)
        fit_state = trait_ou_fit(lag_h, rho_state)
        alpha_emp = between_record_share(series)
        # state integral time from the within-record curve (pure OU fit)
        tau_state = fit_state["pure_ou_tau_hours"]
        print(f"  alpha_fit={fit_total['alpha']:.4f} (empirical between-record "
              f"share {alpha_emp:.4f}), tau_1={fit_total['tau1_hours']:.3f} h, "
              f"R2={fit_total['trait_ou_r2']:.4f}")

        save_csv([{
            "lag_minutes": int(u),
            "lag_hours": float(h),
            "rho_total": float(rt),
            "rho_total_fit": float(fit_total["fit"](h)),
            "rho_state_within_record": float(rs),
            "rho_state_fit": float(fit_state["pure_fit"](h)),
            "n_pairs": int(sum(max(x.size - u, 0) for x in series)),
        } for u, h, rt, rs in zip(lags, lag_h, rho_total, rho_state)],
            "ltaf_autocorr")

        headline.update({
            "rho_1min": float(rho_total[lags == 1][0]),
            "rho_15min": float(rho_total[lags == 15][0]),
            "rho_1h": float(rho_total[lags == 60][0]),
            "rho_6h": float(rho_total[lags == 360][0]),
            "rho_12h": float(rho_total[lags == 720][0]),
            "rho_state_1h": float(rho_state[lags == 60][0]),
            "trait_share_alpha_fit": fit_total["alpha"],
            "trait_share_alpha_empirical": alpha_emp,
            "tau1_hours": fit_total["tau1_hours"],
            "trait_ou_fit_r2": fit_total["trait_ou_r2"],
            "pure_ou_tau_hours": fit_total["pure_ou_tau_hours"],
            "pure_ou_fit_r2": fit_total["pure_ou_r2"],
            "state_tau_hours_within_record": tau_state,
        })

    # ------------------------------------------------------------------
    # (B)  Covariance diagnostics for every cohort / p
    # ------------------------------------------------------------------
    with Timer("covariance diagnostics"):
        models: dict[tuple, Model] = {}
        cov_rows = []
        for c in COHORTS:
            for p in P_CHECK:
                for lm in LABEL_MODELS:
                    mdl = Model(W_by_cohort_p[(c, p)], lm)
                    models[(c, p, lm)] = mdl
                    spec = mdl.spectrum()
                    cov_rows.append({"cohort": c, "n_records": mdl.m, "p": p,
                                     "label_model": lm,
                                     "bin_minutes": mdl.bin_minutes,
                                     "prevalence": mdl.prevalence,
                                     **spec})
        save_csv(cov_rows, "ltaf_covariance")
        for lm in LABEL_MODELS:
            s = models[(PRIMARY_COHORT, P_MAIN, lm)].spectrum()
            headline[f"effective_rank_{lm}"] = s["effective_rank"]
            headline[f"leading_eigenvalue_share_{lm}"] = s["leading_eigenvalue_share"]
            headline[f"top5_eigenvalue_share_{lm}"] = s["top5_eigenvalue_share"]
            headline[f"mean_offdiag_correlation_{lm}"] = s["mean_offdiag_correlation"]
        print("  " + ", ".join(
            f"{lm}: r_eff={headline[f'effective_rank_{lm}']:.3f}, "
            f"lead={headline[f'leading_eigenvalue_share_{lm}']:.3f}"
            for lm in LABEL_MODELS))

    # ------------------------------------------------------------------
    # (C)  Protocol table: budget x width x method (in-sample, primary cohort)
    # ------------------------------------------------------------------
    with Timer("protocol table"):
        table_rows = []
        for c in COHORTS:
            for lm in LABEL_MODELS:
                mdl = models[(c, P_MAIN, lm)]
                for w_min in WIDTH_MIN:
                    w_h = w_min / 60.0
                    cands = candidates_for(mdl.grid, w_h)
                    nested = nested_designs(mdl, cands, max(BUDGETS))
                    for N in BUDGETS:
                        protos = {
                            "contiguous": contiguous_windows(N, w_h),
                            "contiguous_block": contiguous_block(N * w_h,
                                                                 n_segments=N),
                            "dispersed": dispersed_windows(mdl.grid, N, w_h),
                            "uniform": design_uniform(
                                mdl.label, mdl.K, mdl.grid, N,
                                _act(0.0, w_h, NOISE, "uniform")).actions,
                            "random": design_random(
                                mdl.label, mdl.K, mdl.grid, cands, N,
                                np.random.default_rng(SEED + N),
                                n_restarts=N_RANDOM_RESTARTS).actions,
                            "label_aware": label_aware_design(mdl, cands, N).actions,
                        }
                        for k in NESTED_METHODS:
                            protos[k] = nested[k][:N]
                        for method, acts in protos.items():
                            table_rows.append({
                                "cohort": c, "label_model": lm, "p": P_MAIN,
                                "method": method, "n_windows": N,
                                "width_minutes": w_min,
                                "total_minutes": N * w_min,
                                "resolution_limited":
                                    int(w_min < mdl.bin_minutes - 1e-9),
                                "ceiling": mdl.ceiling(acts),
                            })
        save_csv(table_rows, "ltaf")
        print(f"  {len(table_rows)} rows")

    # ------------------------------------------------------------------
    # (D)  Equal-duration contiguous vs intermittent (+ label-aware)
    # ------------------------------------------------------------------
    with Timer("equal-duration comparison"):
        w_h = EQUAL_DURATION_WIDTH_MIN / 60.0
        duration_rows = []
        curve: dict[tuple, dict[str, float]] = {}
        boot_protocols: dict[str, list[Action]] = {}
        for lm in LABEL_MODELS:
            mdl = models[(PRIMARY_COHORT, P_MAIN, lm)]
            cands = candidates_for(mdl.grid, w_h)
            for k in DISPERSION_K:
                total_min = k * EQUAL_DURATION_WIDTH_MIN
                protos = {
                    "contiguous_block": contiguous_block(total_min / 60.0),
                    "contiguous": contiguous_windows(k, w_h),
                    "dispersed": dispersed_windows(mdl.grid, k, w_h),
                    "label_aware": label_aware_design(mdl, cands, k).actions,
                }
                for method, acts in protos.items():
                    val = mdl.ceiling(acts)
                    duration_rows.append({
                        "label_model": lm, "method": method, "n_windows": k,
                        "window_minutes": EQUAL_DURATION_WIDTH_MIN,
                        "total_minutes": total_min,
                        "total_hours": total_min / 60.0,
                        "ceiling": val,
                    })
                    curve[(lm, method, k)] = val
                    if lm == PRIMARY_MODEL:
                        boot_protocols[f"{method}|k={k}"] = acts
        save_csv(duration_rows, "ltaf_duration")

    # ------------------------------------------------------------------
    # (E)  Clinical headline protocols + bootstrap intervals
    # ------------------------------------------------------------------
    with Timer("headline protocols + bootstrap"):
        clinical: dict[str, list[Action]] = {
            "contiguous_1h": contiguous_block(1.0),
            "contiguous_6h": contiguous_block(6.0),
            "intermittent_4x15min": dispersed_windows(
                models[(PRIMARY_COHORT, P_MAIN, PRIMARY_MODEL)].grid, 4, 0.25),
            "contiguous_1h_as_4x15min": contiguous_windows(4, 0.25),
            "intermittent_2x30min": dispersed_windows(
                models[(PRIMARY_COHORT, P_MAIN, PRIMARY_MODEL)].grid, 2, 0.5),
            "intermittent_8x15min": dispersed_windows(
                models[(PRIMARY_COHORT, P_MAIN, PRIMARY_MODEL)].grid, 8, 0.25),
        }
        head_rows = []
        boot_cache: dict[tuple, dict[str, Array]] = {}
        for c in COHORTS:
            for lm in LABEL_MODELS:
                mdl = models[(c, P_MAIN, lm)]
                protos = dict(clinical)
                protos["label_aware_4x15min"] = label_aware_design(
                    mdl, candidates_for(mdl.grid, 0.25), 4).actions
                reps = bootstrap_ceilings(
                    W_by_cohort_p[(c, P_MAIN)], lm, protos, N_BOOTSTRAP,
                    np.random.default_rng(SEED + 17 * COHORTS.index(c)))
                boot_cache[(c, lm)] = reps
                for name, acts in protos.items():
                    lo, hi = percentile_ci(reps[name])
                    head_rows.append({
                        "cohort": c, "label_model": lm, "protocol": name,
                        "n_windows": len(acts),
                        "total_minutes": float(
                            sum(a.width for a in acts) * 60.0),
                        "ceiling": mdl.ceiling(acts),
                        "boot_mean": float(np.nanmean(reps[name])),
                        "boot_sd": float(np.nanstd(reps[name], ddof=1)),
                        "ci_lower": lo, "ci_upper": hi,
                    })
                # paired equal-duration contrast: 4x15 min vs 1 h contiguous
                d = reps["intermittent_4x15min"] - reps["contiguous_1h"]
                dlo, dhi = percentile_ci(d)
                head_rows.append({
                    "cohort": c, "label_model": lm,
                    "protocol": "contrast_4x15min_minus_1h_contiguous",
                    "n_windows": 4, "total_minutes": 60.0,
                    "ceiling": mdl.ceiling(clinical["intermittent_4x15min"])
                               - mdl.ceiling(clinical["contiguous_1h"]),
                    "boot_mean": float(np.nanmean(d)),
                    "boot_sd": float(np.nanstd(d, ddof=1)),
                    "ci_lower": dlo, "ci_upper": dhi,
                })
        save_csv(head_rows, "ltaf_headline")

        # how much *contiguous* monitoring matches 4 x 15 min?
        equiv, equiv_max = {}, {}
        for lm in LABEL_MODELS:
            mdl = models[(PRIMARY_COHORT, P_MAIN, lm)]
            target = mdl.ceiling(clinical["intermittent_4x15min"])
            durs = np.linspace(0.25, HORIZON_H, 96)
            vals = np.array([mdl.ceiling(contiguous_block(float(D))) for D in durs])
            hit = np.where(vals >= target)[0]
            # ``None`` means: not attainable by *any* contiguous block inside the
            # 24 h record, so the equal-information saving is unbounded.
            equiv[lm] = float(durs[hit[0]]) if hit.size else None
            equiv_max[lm] = float(vals.max())
        headline["contiguous_hours_matching_4x15min"] = equiv
        headline["contiguous_24h_max_ceiling"] = equiv_max
        headline["contiguous_time_saving_factor"] = {
            lm: (None if equiv[lm] is None else float(equiv[lm] / 1.0))
            for lm in LABEL_MODELS}
        print("  contiguous hours matching 4x15 min: " + ", ".join(
            f"{k}=" + ("not attainable within 24 h" if v is None else f"{v:.2f}")
            for k, v in equiv.items()))

    # ------------------------------------------------------------------
    # (F)  Empirical out-of-sample R^2 -- does the ceiling actually bound it?
    # ------------------------------------------------------------------
    with Timer("empirical out-of-sample R2"):
        emp_rows = []
        mdl0 = models[(PRIMARY_COHORT, P_MAIN, PRIMARY_MODEL)]
        Wp = W_by_cohort_p[(PRIMARY_COHORT, P_MAIN)]
        yp = y_by_cohort[PRIMARY_COHORT]
        emp_protocols = dict(clinical)
        for k in DISPERSION_K:
            emp_protocols[f"dispersed_{k}x15min"] = dispersed_windows(
                mdl0.grid, k, 0.25)
            emp_protocols[f"contiguous_{k*15}min"] = contiguous_block(
                k * 0.25)
        for name, acts in emp_protocols.items():
            r2s = []
            for rep in range(N_CV_REPEATS):
                folds = np.random.default_rng(SEED + rep).permutation(
                    yp.size) % N_CV_FOLDS
                r2s.append(empirical_oos_r2(Wp, yp, mdl0.grid, acts, folds,
                                            np.random.default_rng(SEED + rep)))
            row = {"protocol": name, "n_windows": len(acts),
                   "total_minutes": float(sum(a.width for a in acts) * 60.0),
                   "empirical_oos_r2_mean": float(np.mean(r2s)),
                   "empirical_oos_r2_sd": float(np.std(r2s, ddof=1))}
            for lm in LABEL_MODELS:
                row[f"ceiling_{lm}"] = models[
                    (PRIMARY_COHORT, P_MAIN, lm)].ceiling(acts)
            emp_rows.append(row)
        save_csv(emp_rows, "ltaf_empirical_r2")
        for lm in LABEL_MODELS:
            gaps = np.array([r[f"ceiling_{lm}"] - r["empirical_oos_r2_mean"]
                             for r in emp_rows])
            headline[f"empirical_vs_ceiling_max_abs_gap_{lm}"] = float(np.max(np.abs(gaps)))
            headline[f"empirical_vs_ceiling_mean_gap_{lm}"] = float(np.mean(gaps))
            headline[f"empirical_vs_ceiling_min_gap_{lm}"] = float(np.min(gaps))
            # A negative gap means the estimated ceiling was exceeded.  Report the
            # count at *zero* tolerance -- a threshold hidden inside a field named
            # "n_violations" turns a small, explainable exceedance into a claim of
            # none -- together with the largest exceedance, so the size can be
            # judged against the bootstrap width of the ceiling itself.
            headline[f"empirical_vs_ceiling_n_violations_{lm}"] = int(np.sum(gaps < 0.0))
            headline[f"empirical_vs_ceiling_n_comparisons_{lm}"] = int(gaps.size)
            headline[f"empirical_vs_ceiling_max_exceedance_{lm}"] = float(
                max(-gaps.min(), 0.0))
            headline[f"empirical_vs_ceiling_max_shortfall_{lm}"] = float(
                max(gaps.max(), 0.0))
        headline["empirical_r2_n_protocols"] = len(emp_rows)
        print("  " + ", ".join(
            f"{lm}: max|ceiling-R2|={headline[f'empirical_vs_ceiling_max_abs_gap_{lm}']:.4f}, "
            f"violations={headline[f'empirical_vs_ceiling_n_violations_{lm}']}"
            for lm in LABEL_MODELS))

    # ------------------------------------------------------------------
    # (G)  Record-level repeated 5-fold cross-validation
    # ------------------------------------------------------------------
    with Timer("cross-validation"):
        cv_rows = []
        w_h = CV_WIDTH_MIN / 60.0
        Wp = W_by_cohort_p[(PRIMARY_COHORT, P_MAIN)]
        yp = y_by_cohort[PRIMARY_COHORT]
        m = Wp.shape[0]
        for rep in range(N_CV_REPEATS):
            folds = np.random.default_rng(SEED + 101 * rep).permutation(m) % N_CV_FOLDS
            for f in range(N_CV_FOLDS):
                tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
                for lm in LABEL_MODELS:
                    m_tr = Model(Wp[tr], lm)
                    # target weights are fixed by the training-fold affine model
                    m_te = Model(Wp[te], lm, ref=m_tr)
                    cands = candidates_for(m_tr.grid, w_h)
                    nested = nested_designs(m_tr, cands, max(BUDGETS))
                    for N in BUDGETS:
                        protos = {
                            "contiguous": contiguous_windows(N, w_h),
                            "dispersed": dispersed_windows(m_tr.grid, N, w_h),
                            "uniform": design_uniform(
                                m_tr.label, m_tr.K, m_tr.grid, N,
                                _act(0.0, w_h, NOISE, "uniform")).actions,
                            "random": design_random(
                                m_tr.label, m_tr.K, m_tr.grid, cands, N,
                                np.random.default_rng(SEED + 7 * rep + f),
                                n_restarts=N_RANDOM_RESTARTS).actions,
                            "label_aware": label_aware_design(
                                m_tr, cands, N).actions,
                        }
                        for k in NESTED_METHODS:
                            protos[k] = nested[k][:N]
                        for method, acts in protos.items():
                            cv_rows.append({
                                "repeat": rep, "fold": f, "label_model": lm,
                                "method": method, "n_windows": N,
                                "width_minutes": CV_WIDTH_MIN,
                                "total_minutes": N * CV_WIDTH_MIN,
                                "n_train": int(tr.size), "n_test": int(te.size),
                                "ceiling_train": m_tr.ceiling(acts),
                                "ceiling_test": m_te.ceiling(acts),
                            })
        save_csv(cv_rows, "ltaf_cv")

        # aggregate
        cv_summary = {}
        for lm in LABEL_MODELS:
            for method in ("contiguous", "dispersed", "uniform", "random",
                           "mutual_information", "imse", "kernel_quadrature",
                           "label_aware"):
                for N in BUDGETS:
                    vals = np.array([r["ceiling_test"] for r in cv_rows
                                     if r["label_model"] == lm
                                     and r["method"] == method
                                     and r["n_windows"] == N])
                    tr_vals = np.array([r["ceiling_train"] for r in cv_rows
                                        if r["label_model"] == lm
                                        and r["method"] == method
                                        and r["n_windows"] == N])
                    cv_summary[f"{lm}|{method}|N={N}"] = {
                        "test_mean": float(vals.mean()),
                        "test_sd": float(vals.std(ddof=1)),
                        "train_mean": float(tr_vals.mean()),
                    }
        headline["cv_summary"] = cv_summary
        # Two different comparisons, and they do not agree on this dataset:
        #  * against the other *data-driven* design criteria (they all estimate K
        #    from the same training fold and then optimise a criterion), and
        #  * against the fixed, K-free "dispersed = bin midpoints" heuristic,
        #    which is optimal by symmetry whenever the process is stationary.
        LEARNED = ("random", "mutual_information", "imse", "kernel_quadrature")
        FIXED = ("contiguous", "dispersed", "uniform")
        for lm in LABEL_MODELS:
            for N in BUDGETS:
                la = cv_summary[f"{lm}|label_aware|N={N}"]
                best_learned = max(cv_summary[f"{lm}|{b}|N={N}"]["test_mean"]
                                   for b in LEARNED)
                best_fixed = max(cv_summary[f"{lm}|{b}|N={N}"]["test_mean"]
                                 for b in FIXED)
                headline[f"cv_gain_vs_learned_baselines_{lm}_N{N}"] = \
                    float(la["test_mean"] - best_learned)
                headline[f"cv_gain_vs_fixed_heuristics_{lm}_N{N}"] = \
                    float(la["test_mean"] - best_fixed)
                headline[f"cv_optimism_gap_label_aware_{lm}_N{N}"] = \
                    float(la["train_mean"] - la["test_mean"])
                headline[f"cv_optimism_gap_dispersed_{lm}_N{N}"] = float(
                    cv_summary[f"{lm}|dispersed|N={N}"]["train_mean"]
                    - cv_summary[f"{lm}|dispersed|N={N}"]["test_mean"])
        headline["cv_label_aware_wins_vs_learned"] = {
            lm: int(sum(headline[f"cv_gain_vs_learned_baselines_{lm}_N{N}"] > 0
                        for N in BUDGETS)) for lm in LABEL_MODELS}
        headline["cv_label_aware_wins_vs_fixed"] = {
            lm: int(sum(headline[f"cv_gain_vs_fixed_heuristics_{lm}_N{N}"] > 0
                        for N in BUDGETS)) for lm in LABEL_MODELS}
        print("  CV done: " + ", ".join(
            f"{lm} N=4 label-aware {cv_summary[f'{lm}|label_aware|N=4']['test_mean']:.4f}"
            for lm in LABEL_MODELS))
        print("  label-aware beats the learned baselines in "
              f"{headline['cv_label_aware_wins_vs_learned']} of "
              f"{len(BUDGETS)} budgets; beats the fixed heuristics in "
              f"{headline['cv_label_aware_wins_vs_fixed']}")

    # ------------------------------------------------------------------
    # (H)  Discretisation check and noise sensitivity
    # ------------------------------------------------------------------
    with Timer("discretisation + noise checks"):
        res_rows = []
        for p in P_CHECK:
            for lm in LABEL_MODELS:
                mdl = models[(PRIMARY_COHORT, p, lm)]
                protos = {
                    "contiguous_1h": contiguous_block(1.0),
                    "contiguous_6h": contiguous_block(6.0),
                    "intermittent_4x15min": dispersed_windows(mdl.grid, 4, 0.25),
                    "intermittent_8x15min": dispersed_windows(mdl.grid, 8, 0.25),
                }
                for name, acts in protos.items():
                    A_resolution = fractional_window_matrix(acts, mdl.grid)
                    res_rows.append({
                        "p": p, "bin_minutes": mdl.bin_minutes,
                        "label_model": lm, "protocol": name,
                        "ceiling": mdl.raw_ceiling(A_resolution),
                        "pseudoinverse_rcond": PSEUDOINVERSE_RCOND,
                        "window_mapping": "exact fractional overlap of continuous support with grid bins",
                        "nominal_total_minutes": float(sum(a.width for a in acts) * 60.0),
                        "effective_rank": mdl.spectrum()["effective_rank"],
                    })
        save_csv(res_rows, "ltaf_resolution")
        by = {(r["p"], r["label_model"], r["protocol"]): r["ceiling"] for r in res_rows}
        headline["resolution_max_spread"] = {
            f"{lm}|{prot}": float(
                max(by[(p, lm, prot)] for p in P_CHECK)
                - min(by[(p, lm, prot)] for p in P_CHECK))
            for lm in LABEL_MODELS
            for prot in ("contiguous_1h", "contiguous_6h", "intermittent_4x15min",
                         "intermittent_8x15min")}
        headline["resolution_pseudoinverse_rcond"] = PSEUDOINVERSE_RCOND
        headline["resolution_window_mapping"] = (
            "exact fractional overlap of each continuous window with grid bins")

        noise_rows = []
        for nu2 in NOISE_LEVELS:
            for lm in LABEL_MODELS:
                mdl = models[(PRIMARY_COHORT, P_MAIN, lm)]
                protos = {
                    "contiguous_1h": contiguous_block(1.0, noise=max(nu2, 1e-12)),
                    "contiguous_6h": contiguous_block(6.0, noise=max(nu2, 1e-12)),
                    "intermittent_4x15min": dispersed_windows(
                        mdl.grid, 4, 0.25, noise=max(nu2, 1e-12)),
                    "intermittent_8x15min": dispersed_windows(
                        mdl.grid, 8, 0.25, noise=max(nu2, 1e-12)),
                }
                for name, acts in protos.items():
                    noise_rows.append({"noise_variance": nu2, "label_model": lm,
                                       "protocol": name,
                                       "ceiling": mdl.ceiling(acts)})
        save_csv(noise_rows, "ltaf_noise")

    # ------------------------------------------------------------------
    # (I)  Bootstrap bands for the duration curve (primary model)
    # ------------------------------------------------------------------
    with Timer("duration-curve bootstrap"):
        band_reps = bootstrap_ceilings(
            W_by_cohort_p[(PRIMARY_COHORT, P_MAIN)], PRIMARY_MODEL,
            boot_protocols, N_BOOTSTRAP, np.random.default_rng(SEED + 991))
        bands = {k: percentile_ci(v) for k, v in band_reps.items()}

    # ------------------------------------------------------------------
    # Headline numbers
    # ------------------------------------------------------------------
    def head(cohort, lm, prot, field):
        for r in head_rows:
            if (r["cohort"] == cohort and r["label_model"] == lm
                    and r["protocol"] == prot):
                return r[field]
        raise KeyError(prot)

    for lm in LABEL_MODELS:
        for cohort in COHORTS:
            tag = f"{lm}_{cohort}"
            for prot, short in (("contiguous_1h", "ceiling_1h_contiguous"),
                                ("contiguous_6h", "ceiling_6h_contiguous"),
                                ("intermittent_4x15min", "ceiling_4x15min"),
                                ("contiguous_1h_as_4x15min",
                                 "ceiling_1h_contiguous_4windows"),
                                ("label_aware_4x15min", "ceiling_label_aware_4win"),
                                ("contrast_4x15min_minus_1h_contiguous",
                                 "equal_duration_gain_1h")):
                headline[f"{short}__{tag}"] = head(cohort, lm, prot, "ceiling")
                headline[f"{short}__{tag}__ci"] = [head(cohort, lm, prot, "ci_lower"),
                                                   head(cohort, lm, prot, "ci_upper")]
    headline["primary"] = {
        "cohort": PRIMARY_COHORT, "label_model": PRIMARY_MODEL,
        "ceiling_1h_contiguous": head(PRIMARY_COHORT, PRIMARY_MODEL,
                                      "contiguous_1h", "ceiling"),
        "ceiling_1h_contiguous_ci": [head(PRIMARY_COHORT, PRIMARY_MODEL,
                                          "contiguous_1h", "ci_lower"),
                                     head(PRIMARY_COHORT, PRIMARY_MODEL,
                                          "contiguous_1h", "ci_upper")],
        "ceiling_6h_contiguous": head(PRIMARY_COHORT, PRIMARY_MODEL,
                                      "contiguous_6h", "ceiling"),
        "ceiling_6h_contiguous_ci": [head(PRIMARY_COHORT, PRIMARY_MODEL,
                                          "contiguous_6h", "ci_lower"),
                                     head(PRIMARY_COHORT, PRIMARY_MODEL,
                                          "contiguous_6h", "ci_upper")],
        "ceiling_4x15min": head(PRIMARY_COHORT, PRIMARY_MODEL,
                                "intermittent_4x15min", "ceiling"),
        "ceiling_4x15min_ci": [head(PRIMARY_COHORT, PRIMARY_MODEL,
                                    "intermittent_4x15min", "ci_lower"),
                               head(PRIMARY_COHORT, PRIMARY_MODEL,
                                    "intermittent_4x15min", "ci_upper")],
        "equal_duration_gain_1h": head(PRIMARY_COHORT, PRIMARY_MODEL,
                                       "contrast_4x15min_minus_1h_contiguous",
                                       "ceiling"),
        "equal_duration_gain_1h_ci": [
            head(PRIMARY_COHORT, PRIMARY_MODEL,
                 "contrast_4x15min_minus_1h_contiguous", "ci_lower"),
            head(PRIMARY_COHORT, PRIMARY_MODEL,
                 "contrast_4x15min_minus_1h_contiguous", "ci_upper")],
    }
    headline["environment"] = environment_record()

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    with Timer("figure"):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.95))

        # -- (a) autocorrelation -------------------------------------
        # The within-record curve rho_state is *not* drawn: removing each
        # record's own mean forces the demeaned autocovariances to sum to zero,
        # so that curve turns negative at long lags for arithmetic rather than
        # physiological reasons.  It is kept in results/ltaf_autocorr.csv.
        ax = axes[0]
        uu = np.linspace(0, lag_h.max(), 400)
        ax.plot(lag_h, rho_total, color=PALETTE[0], lw=1.8,
                label=r"empirical $\rho(u)$")
        ax.plot(uu, fit_total["fit"](uu), color="k", ls="--", lw=1.1,
                label=rf"trait + OU fit ($R^2={fit_total['trait_ou_r2']:.3f}$)")
        ax.plot(uu, fit_total["pure_fit"](uu), color=PALETTE[2], ls="-.", lw=1.0,
                label=rf"OU only ($R^2={fit_total['pure_ou_r2']:.2f}$)")
        ax.axhline(fit_total["alpha"], color=PALETTE[1], ls=":", lw=1.0)
        ax.annotate(rf"trait share $\alpha={fit_total['alpha']:.2f}$",
                    xy=(lag_h.max() * 0.60, fit_total["alpha"] + 0.035),
                    color=PALETTE[1], fontsize=7.5)
        tau = fit_total["tau1_hours"]
        ax.annotate(rf"$\tau_1={tau:.1f}$ h",
                    xy=(tau, float(fit_total["fit"](tau))),
                    xytext=(tau + 1.6, 0.40),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="k"),
                    fontsize=8)
        ax.set_xlabel("lag $u$ (hours)")
        ax.set_ylabel("autocorrelation of the AF indicator")
        ax.set_xlim(0, lag_h.max())
        ax.set_ylim(0.0, 1.03)
        ax.legend(loc="upper right")
        ax.text(-0.17, 1.02, "(a)", transform=ax.transAxes, fontweight="bold")

        # -- (b) value vs 24-hour-equivalent observed fraction -------
        ax = axes[1]
        hours = np.array([k * EQUAL_DURATION_WIDTH_MIN / 60.0 for k in DISPERSION_K])
        series_spec = [
            ("contiguous_block", "Contiguous (one block)", PALETTE[1], "s", "-"),
            ("dispersed", r"Intermittent ($15$ min windows)", PALETTE[0], "o", "-"),
            ("label_aware", "Target-aware design (ours)", PALETTE[2], "^", "-"),
        ]
        for key, lab, col, mk, ls in series_spec:
            y = np.array([curve[(PRIMARY_MODEL, key, k)] for k in DISPERSION_K])
            ax.plot(hours, y, color=col, marker=mk, ls=ls, label=lab)
            lo = np.array([bands[f"{key}|k={k}"][0] for k in DISPERSION_K])
            hi = np.array([bands[f"{key}|k={k}"][1] for k in DISPERSION_K])
            ax.fill_between(hours, lo, hi, color=col, alpha=0.16, lw=0)
        # the label-aware curve above is in-sample; its cross-validated twin
        # shows how much of the apparent gain is selection variance
        ax.plot(hours, [cv_summary[f"{PRIMARY_MODEL}|label_aware|N={k}"]["test_mean"]
                        for k in DISPERSION_K],
                color=PALETTE[2], ls=":", marker="^", ms=3, lw=1.1, mfc="none",
                label="Target-aware, record-level CV")
        emp = {r["protocol"]: r["empirical_oos_r2_mean"] for r in emp_rows}
        ax.plot(hours, [emp[f"dispersed_{k}x15min"] for k in DISPERSION_K],
                ls="none", marker="x", ms=5, color="k",
                label=r"Achieved out-of-sample $R^2$")
        ax.plot(hours, [emp[f"contiguous_{int(k*15)}min"] for k in DISPERSION_K],
                ls="none", marker="x", ms=5, color="k")
        ax.axhline(1.0, color="0.6", ls="--", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(hours)
        ax.set_xticklabels([f"{h:g}" for h in hours])
        ax.set_xlabel("24-hour-equivalent observed duration (hours)")
        ax.set_ylabel(r"best-linear acquisition value $I_L(S)$")
        ax.set_ylim(0.50, 1.035)
        ax.legend(loc="lower right", ncol=1)
        ax.text(-0.17, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")

        fig.tight_layout()
        save_figure(fig, "fig_ltaf")
        plt.close(fig)

    headline["runtime_seconds"] = float(time.perf_counter() - t_start)
    save_json(headline, "ltaf")
    print(f"[total] {headline['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
