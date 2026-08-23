"""Experiment S3 -- finite-calibration-sample estimation of the protocol ceiling.

A benchmark identifies its own Bayes risk but not the ceiling of a
counterfactual protocol; what restores estimability is a densely observed
*calibration* sample of ``m`` objects.  This experiment quantifies how good the
resulting plug-in ceiling ``I_hat_g(S) = F_g(S; K_hat) / V_g(K_hat)`` is as a
function of ``m``, for a whole family of counterfactual protocols at once.

Design
------
* Truth ``K = trait_state_correlation(grid, alpha, rho)`` on ``T = 20`` with a
  ``p = 128`` grid, over the full 2 x 2 x 2 factorial
  ``kernel in {OU, Matern-3/2} (tau = 1) x alpha in {0, 0.3} x nu0^2 in {0, 0.25}``.
  ``p = 128`` is a runtime choice: it keeps the 8 x 6 x 200 replication grid
  inside a ~10 min budget while still resolving the correlation length,
  because ``dt = T / p = 0.156 << tau = 1``.
* Calibration data: ``W = Z + eta``, ``Z ~ N(0, K)`` on the FULL grid,
  ``eta ~ N(0, nu0^2 I)`` with ``nu0^2 in {0, 0.25}``.
  Arm "known": ``nu0^2`` is supplied to ``fit_covariance``.
  Arm "estimated": two independent measurement replicates per subject,
  ``nu0^2`` estimated by ``estimate_noise_from_replicates`` and then plugged in
  (run at the primary cell only; see ``CONFIGS``).
* Evaluation family ``Pi_B``: all ``C(12, 4) = 495`` size-4 subsets of the 12
  bin-midpoint candidate occasions, every action a point observation with
  per-action noise ``nu_a^2 = 0.5``.
  Reference protocol: candidates ``{1, 4, 7, 10}``, i.e. exactly the equal-budget
  dispersed protocol ``bin_midpoints(T, 4)``.
* Labels: two smooth (``mean``; ``sigmoid(slope 2)``) and two threshold
  (``occupation c = 0``; ``occupation c = 1``).  `thm:uniform-error` gives
  ``sup_S |I_hat - I| <= C ||K_hat - K||_op^beta`` with ``beta = 1`` (smooth)
  and ``beta = 1/2`` (threshold); since ``||K_hat - K||_op = O_P(m^{-1/2})``,
  the predicted uniform-error rates are ``m^{-1/2}`` and ``m^{-1/4}``.

Outputs
-------
results/s3_ceiling_estimation.csv        aggregated (config, arm, m, label)
results/s3_ceiling_estimation_reps.csv   per-replication raw numbers
results/s3_ceiling_estimation.json       headline numbers for the manuscript
figures/fig_ceiling_estimation.pdf/.png  2 x 2 summary figure
"""

from __future__ import annotations

import os

# Single-threaded BLAS inside every worker: the parallelism here is over
# replications, and nested threading only causes oversubscription.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import (PALETTE, RESULTS, SEED, Timer,
                                environment_record, save_csv, save_figure,
                                save_json, setup_matplotlib)
from protocol_ceiling import (Action, MeanLabel, ThresholdLabel, bin_midpoints,
                              bootstrap_covariances, bootstrap_protocol_ceiling,
                              effective_rank, estimate_protocol_ceiling,
                              fit_covariance, make_kernel, sample_paths,
                              sigmoid_label, trait_state_correlation,
                              uniform_error_bound, uniform_grid)
from protocol_ceiling.covariance import action_vector
from protocol_ceiling.estimation import estimate_noise_from_replicates

# --------------------------------------------------------------------------
# Fixed experimental constants
# --------------------------------------------------------------------------
NAME = "s3_ceiling_estimation"
HORIZON = 20.0
P_GRID = 128                     # runtime-driven; dt = 0.156 << tau = 1
TAU = 1.0
N_CANDIDATES = 12
BUDGET = 4
PROTO_NOISE = 0.5                # per-action measurement noise nu_a^2
REF_SUBSET = (1, 4, 7, 10)       # == bin_midpoints(T, 4)
M_GRID = (25, 50, 100, 250, 500, 1000)
N_REP = int(os.environ.get("S3_NREP", 200))
N_BOOT = int(os.environ.get("S3_NBOOT", 200))
LEVEL = 0.95
SIGMOID_KMAX = 30                # C_g truncation error 3.7e-8, L_g error 1e-6
BLOCK = 55                       # protocols per vectorised C_g call (495 = 9 x 55)

LABELS = {
    "mean": MeanLabel(),
    "sigmoid": sigmoid_label(slope=2.0, c=0.0, kmax=SIGMOID_KMAX),
    "occ_c0": ThresholdLabel(c=0.0),
    "occ_c1": ThresholdLabel(c=1.0),
}
LABEL_ORDER = ["mean", "sigmoid", "occ_c0", "occ_c1"]
LABEL_PRETTY = {
    "mean": r"mean ($\beta=1$)",
    "sigmoid": r"sigmoid ($\beta=1$)",
    "occ_c0": r"occupation $c{=}0$ ($\beta=1/2$)",
    "occ_c1": r"occupation $c{=}1$ ($\beta=1/2$)",
}
LABEL_STYLE = {
    "mean": dict(color=PALETTE[0], marker="o", ls="-"),
    "sigmoid": dict(color=PALETTE[1], marker="s", ls="-"),
    "occ_c0": dict(color=PALETTE[2], marker="^", ls="--"),
    "occ_c1": dict(color=PALETTE[3], marker="v", ls="--"),
}
BETA = {"mean": 1.0, "sigmoid": 1.0, "occ_c0": 0.5, "occ_c1": 0.5}

# Full 2 x 2 x 2 factorial in (kernel, alpha, nu0^2).  Bias / RMSE / uniform
# error / ||K_hat - K||_op are computed in every cell; the object-level
# bootstrap -- which costs ~4x a point estimate -- is run in two cells only
# (``boot``), and the estimated-noise arm, which needs a second measurement
# replicate and a second pass over Pi_B, in one (``est_arm``).
CONFIGS = [
    dict(cid=0, kernel="ou", alpha=0.0, nu2=0.25, est_arm=True, boot=True),  # primary
    dict(cid=1, kernel="ou", alpha=0.3, nu2=0.25, est_arm=False, boot=False),
    dict(cid=2, kernel="matern32", alpha=0.0, nu2=0.25, est_arm=False, boot=False),
    dict(cid=3, kernel="matern32", alpha=0.3, nu2=0.25, est_arm=False, boot=False),
    dict(cid=4, kernel="ou", alpha=0.0, nu2=0.0, est_arm=False, boot=True),
    dict(cid=5, kernel="ou", alpha=0.3, nu2=0.0, est_arm=False, boot=False),
    dict(cid=6, kernel="matern32", alpha=0.0, nu2=0.0, est_arm=False, boot=False),
    dict(cid=7, kernel="matern32", alpha=0.3, nu2=0.0, est_arm=False, boot=False),
]
if os.environ.get("S3_CONFIGS"):
    _keep = {int(x) for x in os.environ["S3_CONFIGS"].split(",")}
    CONFIGS = [c for c in CONFIGS if c["cid"] in _keep]
PRIMARY_CID = CONFIGS[0]["cid"]
_BY_CID = {c["cid"]: c for c in CONFIGS}


def config_name(cfg: dict) -> str:
    return f"{cfg['kernel']}_alpha{cfg['alpha']:g}_nu{cfg['nu2']:g}"


# --------------------------------------------------------------------------
# Per-configuration cached state
# --------------------------------------------------------------------------
_STATE: dict[int, dict] = {}


def state(cid: int) -> dict:
    if cid in _STATE:
        return _STATE[cid]
    cfg = _BY_CID[cid]
    grid = uniform_grid(HORIZON, P_GRID)
    K = trait_state_correlation(grid, cfg["alpha"], make_kernel(cfg["kernel"], tau=TAU))
    cand = [Action(time=float(t), width=0.0, n_segments=1, noise=PROTO_NOISE, cost=1.0)
            for t in bin_midpoints(HORIZON, N_CANDIDATES)]
    A_full = np.stack([action_vector(a, grid) for a in cand])
    subsets = [np.asarray(s) for s in itertools.combinations(range(N_CANDIDATES), BUDGET)]
    ref_pos = next(i for i, s in enumerate(subsets) if tuple(s) == REF_SUBSET)
    iu = np.triu_indices(P_GRID)
    om = grid.weights
    tri_w = om[iu[0]] * om[iu[1]] * np.where(iu[0] == iu[1], 1.0, 2.0)
    st = dict(cfg=cfg, grid=grid, K=K, omega=grid.weights, cand=cand,
              A_full=A_full, subsets=subsets, ref_pos=ref_pos, iu=iu, tri_w=tri_w,
              ref_actions=[cand[i] for i in REF_SUBSET],
              kappa=float(np.linalg.norm(K, 2)),
              r_eff_true=float(effective_rank(K)),
              # Theorem 14's concentration display is driven by the covariance of
              # the calibration OBSERVATIONS, K + R_0, not by K alone
              kappa_obs=float(np.linalg.norm(K + cfg["nu2"] * np.eye(P_GRID), 2)),
              r_eff_obs=float(effective_rank(K + cfg["nu2"] * np.eye(P_GRID))))
    st["I_true"] = family_ceilings(K, st)
    st["I_true_ref"] = {k: float(v[ref_pos]) for k, v in st["I_true"].items()}
    st["V_true"] = {k: float(st["omega"] @ lab.C(K) @ st["omega"])
                    for k, lab in LABELS.items()}
    _STATE[cid] = st
    return st


def family_ceilings(K: np.ndarray, st: dict) -> dict[str, np.ndarray]:
    """``I_g(S; K)`` for every ``S`` in ``Pi_B`` and every label.

    Algebraically identical to ``estimate_ceiling_family`` but three cheap
    rewrites make the 495 x 4 sweep affordable inside 200 x 6 replications:

    1. the 12 x p product ``A_full K`` and the explained covariance ``Q_S`` are
       shared by the four labels instead of being recomputed 4 x 495 times;
    2. ``sum_{jk} w_j w_k C_g(Q_jk)`` is evaluated on the upper triangle only,
       with off-diagonal weights doubled (``C_g`` is applied to a symmetric
       matrix), halving the number of ``C_g`` evaluations;
    3. protocols are processed in blocks so that each ``C_g`` call sees a large
       array (the threshold label's spline and the Hermite Horner recursion are
       both dominated by per-call overhead on 8 k-element inputs).

    Verified against ``estimate_protocol_ceiling`` in :func:`self_check`.
    """
    iu, wq = st["iu"], st["tri_w"]
    G = st["A_full"] @ K                      # 12 x p
    H = st["A_full"] @ G.T                    # 12 x 12
    eyeB = np.eye(BUDGET)
    subsets = st["subsets"]
    n = len(subsets)
    out = {name: np.empty(n) for name in LABELS}
    buf = np.empty((BLOCK, wq.size))
    for start in range(0, n, BLOCK):
        blk = subsets[start:start + BLOCK]
        for b, idx in enumerate(blk):
            Gs = G[idx]
            M = H[np.ix_(idx, idx)] + PROTO_NOISE * eyeB
            M = 0.5 * (M + M.T) + 1e-10 * eyeB
            buf[b] = (Gs.T @ np.linalg.solve(M, Gs))[iu]
        view = buf[:len(blk)]
        for name, lab in LABELS.items():
            out[name][start:start + len(blk)] = lab.C(view) @ wq
    Ktri = K[iu]
    for name, lab in LABELS.items():
        V = float(lab.C(Ktri) @ wq)
        out[name] = out[name] / V if V > 0 else out[name] * 0.0
    return out


# --------------------------------------------------------------------------
# One replication
# --------------------------------------------------------------------------
def one_rep(task: tuple[int, int, int]) -> list[dict]:
    cid, m, rep = task
    st = state(cid)
    cfg = st["cfg"]
    nu2 = cfg["nu2"]
    K_true, grid, omega = st["K"], st["grid"], st["omega"]
    rng = np.random.default_rng([SEED, cid, m, rep])

    Z = sample_paths(K_true, m, rng)
    if nu2 > 0.0:
        sd = np.sqrt(nu2)
        W = Z + sd * rng.standard_normal(Z.shape)
        arms = [("known", float(nu2))]
        if cfg["est_arm"]:
            W2 = Z + sd * rng.standard_normal(Z.shape)
            arms.append(("estimated", None))
    else:
        W = Z
        W2 = None
        arms = [("known", None)]

    rows: list[dict] = []
    for arm, nv in arms:
        if arm == "estimated":
            nu_hat = estimate_noise_from_replicates(np.stack([W, W2], axis=2))
            noise_arg = nu_hat
            nu_hat_mean = float(np.mean(nu_hat))
        else:
            noise_arg = nv
            nu_hat_mean = float(nu2)

        fit = fit_covariance(W, noise_var=noise_arg)
        K_hat = fit.K
        k_err = float(np.linalg.norm(K_hat - K_true, 2))
        r_eff = float(fit.effective_rank)
        I_hat = family_ceilings(K_hat, st)

        # bootstrap only for the "known"-noise arm (object-level, reused
        # covariance replicates shared across the four labels)
        if arm == "known" and cfg["boot"]:
            boot_rng = np.random.default_rng([SEED, 7919, cid, m, rep])
            Ks = bootstrap_covariances(W, n_bootstrap=N_BOOT, noise_var=noise_arg,
                                       rng=boot_rng)
        else:
            Ks = None

        for name, lab in LABELS.items():
            it = st["I_true"][name]
            ih = I_hat[name]
            unif = float(np.max(np.abs(ih - it)))
            ref_err = float(ih[st["ref_pos"]] - st["I_true_ref"][name])
            bnd = uniform_error_bound(lab, K_true, omega, budget=BUDGET,
                                      min_noise=PROTO_NOISE, k_error=k_err)
            if Ks is not None:
                br = bootstrap_protocol_ceiling(lab, W, grid, st["ref_actions"],
                                                n_bootstrap=N_BOOT, level=LEVEL,
                                                noise_var=noise_arg, covariances=Ks)
                lo, hi = br.lower, br.upper
                covered = int(lo <= st["I_true_ref"][name] <= hi)
                width = float(hi - lo)
            else:
                lo = hi = float("nan")
                covered = -1
                width = float("nan")
            rows.append(dict(
                config=config_name(cfg), cid=cid, arm=arm, m=m, rep=rep,
                label=name, I_true_ref=st["I_true_ref"][name],
                I_hat_ref=float(ih[st["ref_pos"]]), ref_err=ref_err,
                uniform_err=unif, k_err=k_err, r_eff=r_eff,
                nu2_used=nu_hat_mean, ci_lo=lo, ci_hi=hi, ci_width=width,
                covered=covered, bound=float(bnd["bound"]),
                bound_ratio=float(bnd["bound"]) / max(unif, 1e-300),
                bound_lt_one=int(bnd["bound"] < 1.0),
                small_pert_ok=int(BUDGET * k_err <= 0.5 * PROTO_NOISE),
            ))
    return rows


# --------------------------------------------------------------------------
# Diagnostics / self-check
# --------------------------------------------------------------------------
def self_check() -> dict:
    st = state(PRIMARY_CID)
    fam = family_ceilings(st["K"], st)
    errs = []
    for pos in (0, 17, st["ref_pos"], len(st["subsets"]) - 1):
        acts = [st["cand"][i] for i in st["subsets"][pos]]
        for name, lab in LABELS.items():
            ref = estimate_protocol_ceiling(lab, st["K"], st["grid"], acts)
            errs.append(abs(ref - fam[name][pos]))
    err = float(max(errs))
    print(f"  [check] shared-Q family evaluation vs estimate_protocol_ceiling: "
          f"max |diff| = {err:.2e}")
    assert err < 1e-11, "fast family evaluation disagrees with the package API"
    return {"family_vs_api_max_abs_diff": err}


# --------------------------------------------------------------------------
# Slope fitting
# --------------------------------------------------------------------------
def ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """OLS of ``y`` on ``x`` -> (slope, standard error, R^2, intercept)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    xb, yb = x.mean(), y.mean()
    sxx = float(((x - xb) ** 2).sum())
    slope = float(((x - xb) * (y - yb)).sum() / sxx)
    intercept = float(yb - slope * xb)
    res = y - (intercept + slope * x)
    dof = max(n - 2, 1)
    s2 = float((res ** 2).sum() / dof)
    se = float(np.sqrt(s2 / sxx))
    sst = float(((y - yb) ** 2).sum())
    r2 = float(1.0 - (res ** 2).sum() / sst) if sst > 0 else float("nan")
    return slope, se, r2, intercept


def fit_rate(reps: list[dict], key: str) -> dict:
    """Replication-level and mean-level log-log rate fits in ``m``."""
    m = np.array([r["m"] for r in reps], float)
    v = np.array([r[key] for r in reps], float)
    ok = v > 0
    s_rep, se_rep, r2_rep, b_rep = ols_slope(np.log(m[ok]), np.log(v[ok]))
    ms = np.array(sorted(set(m.tolist())))
    means = np.array([v[(m == mm) & ok].mean() for mm in ms])
    s_mean, se_mean, r2_mean, _ = ols_slope(np.log(ms), np.log(means))
    return {"slope": s_rep, "se": se_rep, "r2": r2_rep, "n": int(ok.sum()),
            "intercept": b_rep,
            "slope_of_mean": s_mean, "se_of_mean": se_mean, "r2_of_mean": r2_mean}


def fit_beta(reps: list[dict]) -> dict:
    """Direct estimate of the Holder exponent in the uniform bound.

    `thm:uniform-error` asserts ``sup_S |I_hat - I| <= C ||K_hat - K||_op^beta``.
    Regressing ``log(uniform error)`` on ``log ||K_hat - K||_op`` across all
    ``m`` and replications estimates the exponent that is actually attained,
    without going through the (separately estimated) rate of ``||K_hat-K||_op``.
    """
    e = np.array([r["uniform_err"] for r in reps], float)
    k = np.array([r["k_err"] for r in reps], float)
    ok = (e > 0) & (k > 0)
    s, se, r2, b = ols_slope(np.log(k[ok]), np.log(e[ok]))
    return {"beta_hat": s, "se": se, "r2": r2, "intercept": b, "n": int(ok.sum())}


def bound_threshold(label, K: np.ndarray, omega: np.ndarray, kappa: float) -> dict:
    """Covariance error at which ``uniform_error_bound`` first drops below 1.

    ``uniform_error_bound`` returns ``(L_g (L_Q e)^beta + L_g e^beta)/(V - L_g e^beta)``
    with ``e = ||K_hat - K||_op``.  Setting that equal to one gives the closed
    form ``e* = [V / (L_g (L_Q^beta + 2))]^{1/beta}``: for ``e > e*`` the
    certificate is weaker than the trivial bound ``|I_hat - I| <= 1`` and
    therefore vacuous.
    """
    b = uniform_error_bound(label, K, omega, budget=BUDGET,
                            min_noise=PROTO_NOISE, kappa=kappa, k_error=1.0)
    L_Q, L_g, beta, V = b["L_Q"], b["L_g"], b["beta"], b["label_variance"]
    e_star = (V / (L_g * (L_Q ** beta + 2.0))) ** (1.0 / beta)
    return {"L_Q": float(L_Q), "L_g": float(L_g), "beta": float(beta),
            "label_variance": float(V), "k_error_for_unit_bound": float(e_star)}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    t_start = time.perf_counter()
    print(f"[{NAME}] p={P_GRID}, T={HORIZON}, |Pi_B|={495}, "
          f"m={M_GRID}, reps={N_REP}, bootstrap={N_BOOT}, seed={SEED}")
    check = self_check()

    tasks = [(c["cid"], m, r) for c in CONFIGS for m in M_GRID for r in range(N_REP)]
    print(f"  {len(tasks)} replications over {len(CONFIGS)} configurations")

    import multiprocessing as mp
    n_proc = int(os.environ.get("S3_NPROC", os.cpu_count() or 4))
    reps: list[dict] = []
    with Timer("simulate"):
        with mp.get_context("fork").Pool(n_proc) as pool:
            done = 0
            for out in pool.imap_unordered(one_rep, tasks, chunksize=2):
                reps.extend(out)
                done += 1
                if done % 1000 == 0:
                    el = time.perf_counter() - t_start
                    print(f"    {done}/{len(tasks)} reps  ({el:.0f}s elapsed, "
                          f"ETA {el * (len(tasks) / done - 1):.0f}s)")

    # ---------------- aggregate ----------------
    groups: dict[tuple, list[dict]] = {}
    for r in reps:
        groups.setdefault((r["config"], r["arm"], r["label"], r["m"]), []).append(r)
    summary: list[dict] = []
    index: dict[tuple, dict] = {}
    for cfg in CONFIGS:
        cname = config_name(cfg)
        st = state(cfg["cid"])
        for arm in ("known", "estimated"):
            for label in LABEL_ORDER:
                for m in M_GRID:
                    sub = groups.get((cname, arm, label, m))
                    if not sub:
                        continue
                    err = np.array([r["ref_err"] for r in sub])
                    unif = np.array([r["uniform_err"] for r in sub])
                    kerr = np.array([r["k_err"] for r in sub])
                    ratio = np.array([r["bound_ratio"] for r in sub])
                    cov = np.array([r["covered"] for r in sub])
                    row = dict(
                        config=cname, kernel=cfg["kernel"], alpha=cfg["alpha"],
                        nu2=cfg["nu2"], arm=arm, label=label, beta=BETA[label], m=m,
                        n_rep=len(sub), I_true_ref=st["I_true_ref"][label],
                        bias=float(err.mean()),
                        bias_se=float(err.std(ddof=1) / np.sqrt(len(sub))),
                        rmse=float(np.sqrt((err ** 2).mean())),
                        sd=float(err.std(ddof=1)),
                        coverage=float(cov[cov >= 0].mean()) if (cov >= 0).any() else float("nan"),
                        ci_width=(float(np.mean([r["ci_width"] for r in sub]))
                                  if (cov >= 0).any() else float("nan")),
                        uniform_err_mean=float(unif.mean()),
                        # dispersion OF THE UNIFORM ERROR: the `sd` column above is
                        # the sd of the reference-protocol signed error, a different
                        # statistic, and must not be used as a band for this curve
                        uniform_err_sd=float(unif.std(ddof=1)) if len(unif) > 1 else 0.0,
                        uniform_err_median=float(np.median(unif)),
                        uniform_err_q90=float(np.quantile(unif, 0.9)),
                        k_err_mean=float(kerr.mean()),
                        r_eff_mean=float(np.mean([r["r_eff"] for r in sub])),
                        r_eff_true=st["r_eff_true"], kappa=st["kappa"],
                        bound_mean=float(np.mean([r["bound"] for r in sub])),
                        bound_ratio_median=float(np.median(ratio)),
                        bound_ratio_mean=float(ratio.mean()),
                        bound_valid_frac=float((ratio > 1.0).mean()),
                        bound_nonvacuous_frac=float(
                            np.mean([r["bound_lt_one"] for r in sub])),
                        small_pert_frac=float(np.mean([r["small_pert_ok"] for r in sub])),
                    )
                    summary.append(row)
                    index[(cname, arm, label, m)] = row

    save_csv(summary, NAME)
    save_csv(reps, NAME + "_reps")

    # ---------------- rates ----------------
    rates: dict[str, dict] = {}
    for cfg in CONFIGS:
        cname = config_name(cfg)
        for arm in ("known", "estimated"):
            for label in LABEL_ORDER:
                sub = [r for m in M_GRID
                       for r in groups.get((cname, arm, label, m), [])]
                if not sub:
                    continue
                key = f"{cname}|{arm}|{label}"
                st_c = state(cfg["cid"])
                thr = bound_threshold(LABELS[label], st_c["K"], st_c["omega"],
                                      st_c["kappa"])
                k_fit = fit_rate(sub, "k_err")
                # m at which ||K_hat - K||_op would reach e*, from log k = a + b log m
                m_req = float(np.exp((np.log(thr["k_error_for_unit_bound"])
                                      - k_fit["intercept"]) / k_fit["slope"]))
                rates[key] = {
                    "uniform": fit_rate(sub, "uniform_err"),
                    "k_err": k_fit,
                    "beta": fit_beta(sub),
                    "predicted_slope": -BETA[label] / 2.0,
                    "predicted_beta": BETA[label],
                    "bound_constants": thr,
                    "m_for_nonvacuous_bound": m_req,
                }
                # RMSE rate (reference protocol) from the per-m aggregates
                ms = np.array(M_GRID, float)
                rm = np.array([index[(cname, arm, label, int(mm))]["rmse"] for mm in ms])
                s, se, r2, _ = ols_slope(np.log(ms), np.log(rm))
                rates[key]["rmse"] = {"slope": s, "se": se, "r2": r2}

    # ---------------- figure ----------------
    make_figure(index, rates)

    # ---------------- headline ----------------
    pname = config_name(_BY_CID[PRIMARY_CID])
    st = state(PRIMARY_CID)
    headline = {
        "primary_config": pname,
        "grid_p": P_GRID, "horizon": HORIZON, "tau": TAU,
        "n_protocols_in_family": len(st["subsets"]),
        "budget": BUDGET, "protocol_noise": PROTO_NOISE,
        "reference_protocol_times": [float(st["cand"][i].time) for i in REF_SUBSET],
        "m_grid": list(M_GRID), "n_replications": N_REP, "n_bootstrap": N_BOOT,
        "seed": SEED,
        "kappa_primary": st["kappa"], "r_eff_true_primary": st["r_eff_true"],
        "kappa_obs_primary": st["kappa_obs"], "r_eff_obs_primary": st["r_eff_obs"],
    }
    for label in LABEL_ORDER:
        r = rates[f"{pname}|known|{label}"]
        headline[f"slope_uniform_{label}"] = r["uniform"]["slope"]
        headline[f"slope_uniform_se_{label}"] = r["uniform"]["se"]
        headline[f"slope_uniform_mean_{label}"] = r["uniform"]["slope_of_mean"]
        headline[f"slope_uniform_mean_se_{label}"] = r["uniform"]["se_of_mean"]
        headline[f"predicted_slope_{label}"] = -BETA[label] / 2.0
        # Tail slope over the largest three calibration sizes.  The full-range
        # fit is pre-asymptotic; the fixed-model delta-method rate is an
        # asymptotic statement, so what matters is whether the slope steepens
        # towards -1/2 as m grows.
        _ms = np.array(M_GRID, float)
        _ys = np.array([index[(pname, "known", label, int(mm))]["uniform_err_mean"]
                        for mm in _ms])
        _ok = np.isfinite(_ys) & (_ys > 0)
        if _ok.sum() >= 3:
            headline[f"tail_slope_{label}"] = float(
                np.polyfit(np.log(_ms[_ok][-3:]), np.log(_ys[_ok][-3:]), 1)[0])
        headline[f"slope_rmse_{label}"] = r["rmse"]["slope"]
        headline[f"slope_rmse_se_{label}"] = r["rmse"]["se"]
        headline[f"slope_k_err_{label}"] = r["k_err"]["slope"]
        headline[f"beta_hat_{label}"] = r["beta"]["beta_hat"]
        headline[f"beta_hat_se_{label}"] = r["beta"]["se"]
        headline[f"predicted_beta_{label}"] = BETA[label]
        headline[f"L_Q_{label}"] = r["bound_constants"]["L_Q"]
        headline[f"L_g_{label}"] = r["bound_constants"]["L_g"]
        headline[f"k_error_for_unit_bound_{label}"] = \
            r["bound_constants"]["k_error_for_unit_bound"]
        headline[f"m_for_nonvacuous_bound_{label}"] = r["m_for_nonvacuous_bound"]
        for m in (25, 1000):
            row = index[(pname, "known", label, m)]
            headline[f"bias_m{m}_{label}"] = row["bias"]
            headline[f"rmse_m{m}_{label}"] = row["rmse"]
            headline[f"coverage_m{m}_{label}"] = row["coverage"]
            headline[f"uniform_err_m{m}_{label}"] = row["uniform_err_mean"]
            headline[f"bound_ratio_median_m{m}_{label}"] = row["bound_ratio_median"]
            headline[f"bound_valid_frac_m{m}_{label}"] = row["bound_valid_frac"]
    for m in (25, 1000):
        row = index[(pname, "known", "mean", m)]
        headline[f"k_err_m{m}"] = row["k_err_mean"]
        headline[f"r_eff_hat_m{m}"] = row["r_eff_mean"]
    headline["k_err_slope_primary"] = rates[f"{pname}|known|mean"]["k_err"]["slope"]
    headline["coverage_min_primary"] = float(min(
        index[(pname, "known", lb, m)]["coverage"]
        for lb in LABEL_ORDER for m in M_GRID))
    headline["coverage_at_m1000_min"] = float(min(
        index[(pname, "known", lb, 1000)]["coverage"] for lb in LABEL_ORDER))
    headline["bound_valid_frac_overall"] = float(np.mean(
        [r["bound_valid_frac"] for r in summary]))
    headline["bound_ratio_median_overall"] = float(np.median(
        [r["bound_ratio_median"] for r in summary]))
    headline["bound_nonvacuous_frac_overall"] = float(np.mean(
        [r["bound_nonvacuous_frac"] for r in summary]))
    headline["small_pert_frac_overall"] = float(np.mean(
        [r["small_pert_frac"] for r in summary]))
    # slope stability across the 2 x 2 x 2 (kernel, alpha, nu0^2) grid
    for label in LABEL_ORDER:
        sl = [rates[f"{config_name(c)}|known|{label}"]["uniform"]["slope"]
              for c in CONFIGS]
        headline[f"slope_uniform_range_{label}"] = [float(min(sl)), float(max(sl))]
        bh = [rates[f"{config_name(c)}|known|{label}"]["beta"]["beta_hat"]
              for c in CONFIGS]
        headline[f"beta_hat_range_{label}"] = [float(min(bh)), float(max(bh))]
    # noise-arm comparison
    for label in LABEL_ORDER:
        for m in (25, 1000):
            k = (pname, "estimated", label, m)
            if k in index:
                headline[f"est_noise_rmse_m{m}_{label}"] = index[k]["rmse"]
                headline[f"est_noise_uniform_m{m}_{label}"] = index[k]["uniform_err_mean"]

    runtime = time.perf_counter() - t_start
    save_json({"headline": headline, "rates": rates, "self_check": check,
               "config_grid": [config_name(c) for c in CONFIGS],
               "labels": {k: {"beta": BETA[k], "modulus_L": float(v.modulus()[0]),
                              "variance": float(v.variance())}
                          for k, v in LABELS.items()},
               "runtime_seconds": runtime,
               "environment": environment_record()}, NAME)
    print(f"[{NAME}] total {runtime:.1f}s")


LABEL_SHORT = {"mean": "mean", "sigmoid": "sigmoid",
               "occ_c0": r"occ. $c{=}0$", "occ_c1": r"occ. $c{=}1$"}


def make_figure(index: dict, rates: dict) -> None:
    plt = setup_matplotlib()
    pname = config_name(_BY_CID[PRIMARY_CID])
    ms = np.array(M_GRID, float)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    def get(lb, field):
        return np.array([index[(pname, "known", lb, int(m))][field] for m in ms])

    def panel_tag(ax, tag):
        ax.text(-0.19, 1.03, tag, transform=ax.transAxes, fontweight="bold",
                va="bottom", ha="left")

    # (a) bias of the reference-protocol ceiling
    ax_a.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    for lb in LABEL_ORDER:
        ax_a.errorbar(ms, get(lb, "bias"), yerr=1.96 * get(lb, "bias_se"),
                      capsize=2, elinewidth=0.7, label=LABEL_PRETTY[lb],
                      **LABEL_STYLE[lb])
    ax_a.set_xscale("log")
    ax_a.set_xlabel(r"calibration sample size $m$")
    ax_a.set_ylabel(r"bias of $\hat I_g(S_{\mathrm{ref}})$")
    ax_a.legend(loc="upper right", handlelength=2.2)
    panel_tag(ax_a, "(a)")

    # (b) RMSE, log-log, with fitted slopes
    for lb in LABEL_ORDER:
        s = rates[f"{pname}|known|{lb}"]["rmse"]
        ax_b.plot(ms, get(lb, "rmse"),
                  label=f"{LABEL_SHORT[lb]}: ${s['slope']:.2f}$", **LABEL_STYLE[lb])
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"calibration sample size $m$")
    ax_b.set_ylabel(r"RMSE of $\hat I_g(S_{\mathrm{ref}})$")
    ax_b.legend(loc="upper right", title="fitted slope", title_fontsize=7,
                handlelength=2.2)
    panel_tag(ax_b, "(b)")

    # (c) bootstrap coverage against the nominal level
    mc = 1.96 * np.sqrt(LEVEL * (1 - LEVEL) / N_REP)
    ax_c.axhspan(LEVEL - mc, LEVEL + mc, color="0.88", zorder=0, lw=0)
    ax_c.axhline(LEVEL, color="k", lw=0.9, ls="--", zorder=1,
                 label=f"nominal {LEVEL:.0%} " + r"($\pm$ MC error)")
    for lb in LABEL_ORDER:
        ax_c.plot(ms, get(lb, "coverage"), label=LABEL_SHORT[lb], **LABEL_STYLE[lb])
    ax_c.set_xscale("log")
    ax_c.set_xlabel(r"calibration sample size $m$")
    ax_c.set_ylabel("95% bootstrap CI coverage")
    ax_c.set_ylim(0.80, 1.0)
    ax_c.legend(loc="lower right", ncol=2, columnspacing=1.0, handlelength=2.2)
    panel_tag(ax_c, "(c)")

    # (d) uniform error over Pi_B, with the theoretical -beta/2 reference slopes
    for lb in LABEL_ORDER:
        s = rates[f"{pname}|known|{lb}"]["uniform"]
        ax_d.plot(ms, get(lb, "uniform_err_mean"),
                  label=f"{LABEL_SHORT[lb]}: ${s['slope']:.2f}$", **LABEL_STYLE[lb])
    y0 = get("mean", "uniform_err_mean")[0]
    ax_d.plot(ms, y0 * (ms / ms[0]) ** (-0.5), color="0.35", lw=0.9,
              label=r"smooth: $m^{-1/2}$")
    y1 = get("occ_c1", "uniform_err_mean")[0]
    ax_d.plot(ms, y1 * (ms / ms[0]) ** (-0.25), color="0.35", lw=0.9, ls=":",
              label=r"threshold: $m^{-1/4}$")
    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlabel(r"calibration sample size $m$")
    ax_d.set_ylabel(r"$\sup_{S\in\Pi_B}|\hat I_g(S)-I_g(S)|$")
    ax_d.legend(loc="lower left", title="fitted slope / theory",
                title_fontsize=7, handlelength=2.2, frameon=True,
                framealpha=0.9, edgecolor="none")
    panel_tag(ax_d, "(d)")

    fig.tight_layout(pad=0.5, w_pad=1.6, h_pad=1.4)
    save_figure(fig, "fig_ceiling_estimation")
    plt.close(fig)


def figure_only() -> None:
    """Rebuild the figure from the saved CSV/JSON without re-simulating."""
    import csv as _csv
    import json as _json
    index = {}
    for r in _csv.DictReader((RESULTS / f"{NAME}.csv").open()):
        row = {k: (v if k in ("config", "arm", "label", "kernel") else
                   (float(v) if v not in ("", "nan") else float("nan")))
               for k, v in r.items()}
        index[(r["config"], r["arm"], r["label"], int(row["m"]))] = row
    rates = _json.loads((RESULTS / f"{NAME}.json").read_text())["rates"]
    make_figure(index, rates)


if __name__ == "__main__":
    if os.environ.get("S3_FIGURE_ONLY"):
        figure_only()
    else:
        main()
