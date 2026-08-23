"""S4 -- protocol-selection regret.

The question of this experiment is *not* how well a ceiling is estimated but how
much is lost by **acting** on an estimated ceiling.  The experimenter knows the
truth ``K``; the selector does not, and must pick a protocol
``S_hat in Pi_B`` from ``m`` dense calibration trajectories alone.  The loss is
the *selection regret*

    regret(S_hat) = I_g(S*; K) - I_g(S_hat; K),   S* = argmax_{S in Pi_B} I_g(S; K),

and `cor:regret` says it is controlled by the uniform plug-in error

    eps_m = sup_{S in Pi_B} |I_hat_g(S) - I_g(S)|

through ``regret <= 2 eps_m`` for an *exact* maximiser of the estimated
objective, and through ``I(S_hat) >= eta I(S*) - (1 + eta) eps_m`` for an
``eta``-approximate one.

Both arms are run:

* **eta = 1 arm.**  The candidate set is deliberately small enough
  (``|V| = 14``, ``B = 4``, ``C(14, 4) = 1001`` protocols) that
  ``argmax_{S in Pi_B} I_hat`` is found by *exhaustive* search, so the eta = 1
  theory applies exactly and the guarantee can be checked replication by
  replication with no algorithmic slack.
* **eta < 1 arm.**  The proposed selector -- greedy with 1-swap local search --
  is run on the same ``K_hat``; its realised objective ratio
  ``eta_hat = I_hat(S_greedy) / max_S I_hat(S)`` is *measured*, not assumed, and
  the ``eta``-approximate bound is checked with that measured ``eta_hat``.

A third, **robust** arm replaces the plug-in objective by a bootstrap lower
confidence bound (``select_protocol_robust``, quantile 0.1).  The claim under
test is that the LCB objective buys something at small ``m``; the comparison is
paired (same calibration sample, same greedy algorithm, only the objective
differs) so that any difference is attributable to the objective alone.

Setup (shared with S3): trait-state OU and two-scale OU correlations on
``T = 20`` with ``p = 128`` grid points; labels are the mean label and the
occupation labels ``1{Z > c}`` at ``c = 0`` and ``c = 1``.

Outputs
-------
results/s4_selection_regret.csv          one row per (kernel, label, m, rep)
results/s4_selection_regret_robust.csv   one row per paired robust comparison
results/s4_selection_regret.json         headline numbers
figures/fig_regret.pdf/.png              two-panel figure
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import (PALETTE, SEED, Timer, environment_record,
                                save_csv, save_figure, save_json,
                                setup_matplotlib)
from protocol_ceiling import (Action, MeanLabel, ThresholdLabel, bin_midpoints,
                              bootstrap_covariances, fit_covariance,
                              make_kernel, sample_paths, select_protocol_greedy,
                              select_protocol_robust, trait_state_correlation,
                              uniform_grid)
from protocol_ceiling.covariance import action_vector
from protocol_ceiling.risk import explained_covariance

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
HORIZON = 20.0
P_GRID = 128
BUDGET = 4                      # |S| = B actions, unit cost each
ACTION_NOISE = 0.5              # raw per-segment measurement noise nu^2

M_LIST = (25, 50, 100, 250, 500, 1000)
N_REPS = 200                    # replications per (kernel, m) cell
CHUNK = 25                      # replications per parallel task

ROBUST_M = (25, 50)             # small-m cells where the robust arm is run
N_REPS_ROBUST = 100             # paired replications for the robust arm
N_BOOT = 20                     # bootstrap covariance replicates
ROBUST_QUANTILE = 0.1

TOL = 1e-12                     # slack allowed when checking the guarantees

KERNEL_SPECS = {
    # trait-state OU: a stable between-object component plus a fast state
    "trait_ou": dict(alpha=0.2, kernel="ou", kwargs=dict(tau=2.0)),
    # two-scale OU: no trait, but a fast and a slow state channel
    "two_scale_ou": dict(alpha=0.0, kernel="two_scale_ou",
                         kwargs=dict(tau_fast=0.5, tau_slow=6.0, w_fast=0.5)),
}

LABEL_SPECS = ("mean", "occ_c0", "occ_c1")


def make_labels() -> dict:
    # NOTE: ThresholdLabel inherits ``name`` as its FIRST dataclass field, so the
    # threshold must always be passed by keyword (``ThresholdLabel(c=1.0)``).
    return {
        "mean": MeanLabel(),
        "occ_c0": ThresholdLabel(c=0.0),
        "occ_c1": ThresholdLabel(c=1.0),
    }


def make_grid():
    return uniform_grid(HORIZON, P_GRID)


def make_candidates(grid) -> list[Action]:
    """A 14-action candidate set with heterogeneous temporal support.

    Ten point occasions at the *bin midpoints* of the horizon (the correct
    dispersed placement), two window averages of width 3, and two point
    occasions with ``M = 3`` repeated segments -- the last pair buys a factor
    three in effective noise on a support already covered, so the selector has
    to trade new temporal support against sharper existing support.  All costs
    are one, so the budget ``B = 4`` is exactly "four actions".
    """
    acts = [Action(time=float(t), width=0.0, n_segments=1, noise=ACTION_NOISE,
                   cost=1.0, tag=f"pt{i}")
            for i, t in enumerate(bin_midpoints(HORIZON, 10))]
    acts += [Action(time=float(t), width=3.0, n_segments=1, noise=ACTION_NOISE,
                    cost=1.0, tag=f"win{i}")
             for i, t in enumerate((5.0, 15.0))]
    acts += [Action(time=float(t), width=0.0, n_segments=3, noise=ACTION_NOISE,
                    cost=1.0, tag=f"seg{i}")
             for i, t in enumerate((2.0, 18.0))]
    return acts


def make_truth(name: str, grid):
    spec = KERNEL_SPECS[name]
    rho = make_kernel(spec["kernel"], **spec["kwargs"])
    return trait_state_correlation(grid, spec["alpha"], rho)


# --------------------------------------------------------------------------
# Exhaustive sweep over Pi_B
# --------------------------------------------------------------------------
class Sweeper:
    """Evaluates ``I_g(S; K)`` for *every* ``S in Pi_B`` and every label."""

    def __init__(self, grid, candidates: list[Action], labels: dict, budget: int):
        self.grid = grid
        self.labels = labels
        self.candidates = candidates
        self.vectors = np.stack([action_vector(a, grid) for a in candidates])
        self.noises = np.array([a.effective_noise for a in candidates])
        self.subsets = [list(c) for c in
                        itertools.combinations(range(len(candidates)), budget)]
        self.index = {tuple(s): i for i, s in enumerate(self.subsets)}
        self.tag_index = {a.tag: i for i, a in enumerate(candidates)}
        self.omega = grid.weights

    def sweep(self, K: np.ndarray) -> dict[str, np.ndarray]:
        w = self.omega
        V = {k: float(w @ lab.C(K) @ w) for k, lab in self.labels.items()}
        out = {k: np.empty(len(self.subsets)) for k in self.labels}
        for n, idx in enumerate(self.subsets):
            Q = explained_covariance(K, self.vectors[idx], np.diag(self.noises[idx]))
            for k, lab in self.labels.items():
                out[k][n] = float(w @ lab.C(Q) @ w) / V[k] if V[k] > 0 else 0.0
        return out

    def subset_position(self, actions) -> int:
        """Position of a selected protocol in the enumerated family ``Pi_B``."""
        idx = tuple(sorted(self.tag_index[a.tag] for a in actions))
        return self.index[idx]


# --------------------------------------------------------------------------
# Worker-process globals
# --------------------------------------------------------------------------
CTX: dict = {}


def _init_worker() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    grid = make_grid()
    labels = make_labels()
    cands = make_candidates(grid)
    sw = Sweeper(grid, cands, labels, BUDGET)
    truth = {}
    for kname in KERNEL_SPECS:
        K = make_truth(kname, grid)
        truth[kname] = {"K": K, "I": sw.sweep(K)}
    CTX.update(grid=grid, labels=labels, candidates=cands, sweeper=sw, truth=truth)


def _rng(kernel_idx: int, m_idx: int, rep: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(entropy=SEED, spawn_key=(kernel_idx, m_idx, rep)))


def _run_cell(task) -> tuple[list[dict], list[dict]]:
    """Run replications ``[r0, r1)`` of one ``(kernel, m)`` cell."""
    kernel_idx, kname, m_idx, m, r0, r1 = task
    if not CTX:
        _init_worker()
    grid, labels = CTX["grid"], CTX["labels"]
    cands, sw = CTX["candidates"], CTX["sweeper"]
    K_true = CTX["truth"][kname]["K"]
    I_true = CTX["truth"][kname]["I"]

    main_rows: list[dict] = []
    robust_rows: list[dict] = []

    for rep in range(r0, r1):
        rng = _rng(kernel_idx, m_idx, rep)
        Z = sample_paths(K_true, m, rng)                 # dense calibration
        K_hat = fit_covariance(Z).K
        I_hat = sw.sweep(K_hat)
        k_err = float(np.linalg.norm(K_hat - K_true, 2))

        need_robust = (m in ROBUST_M) and (rep < N_REPS_ROBUST)
        K_boot = (bootstrap_covariances(Z, N_BOOT, rng=rng) if need_robust else None)

        for lname, lab in labels.items():
            It, Ih = I_true[lname], I_hat[lname]
            eps = float(np.max(np.abs(Ih - It)))
            j_star = int(np.argmax(It))
            I_star = float(It[j_star])

            # -- eta = 1 arm: exhaustive maximisation of the plug-in objective --
            j_hat = int(np.argmax(Ih))
            regret_exh = I_star - float(It[j_hat])
            ok_exh = bool(regret_exh <= 2.0 * eps + TOL)

            # -- eta < 1 arm: proposed greedy + 1-swap local search ------------
            res = select_protocol_greedy(lab, K_hat, grid, cands,
                                         budget=float(BUDGET), cost_aware=True,
                                         local_search=True)
            j_gr = sw.subset_position(res.actions)
            eta_hat = float(Ih[j_gr] / Ih[j_hat]) if Ih[j_hat] > 0 else 1.0
            regret_gr = I_star - float(It[j_gr])
            lhs = float(It[j_gr])
            rhs = eta_hat * I_star - (1.0 + eta_hat) * eps
            ok_eta = bool(lhs >= rhs - TOL)

            main_rows.append(dict(
                kernel=kname, label=lname, m=m, rep=rep,
                k_error_op=k_err, eps_m=eps, I_star=I_star,
                I_true_at_Shat=float(It[j_hat]), regret_exhaustive=regret_exh,
                bound_2eps=2.0 * eps, slack_exhaustive=2.0 * eps - regret_exh,
                bound_holds_exhaustive=int(ok_exh),
                eta_hat=eta_hat, n_greedy_actions=len(res.actions),
                greedy_is_optimal=int(j_gr == j_hat),
                I_true_at_Sgreedy=lhs, regret_greedy=regret_gr,
                eta_bound_rhs=rhs, slack_eta=lhs - rhs,
                bound_holds_eta=int(ok_eta),
                Ihat_at_Shat=float(Ih[j_hat]), Ihat_at_Sgreedy=float(Ih[j_gr]),
                ceiling_from_designresult=float(res.ceiling),
            ))

            if need_robust:
                rob = select_protocol_robust(lab, K_boot, grid, cands,
                                             budget=float(BUDGET),
                                             quantile=ROBUST_QUANTILE,
                                             cost_aware=True)
                plug = select_protocol_greedy(lab, K_hat, grid, cands,
                                              budget=float(BUDGET),
                                              cost_aware=True, local_search=False)
                j_rob = sw.subset_position(rob.actions)
                j_plug = sw.subset_position(plug.actions)
                r_rob = I_star - float(It[j_rob])
                r_plug = I_star - float(It[j_plug])
                robust_rows.append(dict(
                    kernel=kname, label=lname, m=m, rep=rep,
                    eps_m=eps, I_star=I_star,
                    regret_robust=r_rob, regret_plugin_greedy=r_plug,
                    regret_plugin_exhaustive=regret_exh,
                    diff_robust_minus_plugin=r_rob - r_plug,
                    robust_wins=int(r_rob < r_plug - 1e-15),
                    tie=int(abs(r_rob - r_plug) <= 1e-15),
                    same_protocol=int(j_rob == j_plug),
                ))

    return main_rows, robust_rows


# --------------------------------------------------------------------------
# Aggregation helpers
# --------------------------------------------------------------------------
def _cell_stats(rows: list[dict], keys: tuple, value: str) -> dict:
    buckets: dict = {}
    for r in rows:
        buckets.setdefault(tuple(r[k] for k in keys), []).append(r[value])
    return {k: np.asarray(v, dtype=float) for k, v in buckets.items()}


def _slope_loglog(m: np.ndarray, y: np.ndarray) -> float:
    ok = (y > 0) & (m > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(m[ok]), np.log(y[ok]), 1)[0])


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def make_figure(main_rows: list[dict], robust_rows: list[dict]) -> None:
    plt = setup_matplotlib()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ms = np.array(M_LIST, dtype=float)

    label_colour = {"mean": PALETTE[0], "occ_c0": PALETTE[1], "occ_c1": PALETTE[2]}
    label_text = {"mean": r"mean", "occ_c0": r"occupation $c=0$",
                  "occ_c1": r"occupation $c=1$"}
    kernel_style = {"trait_ou": dict(ls="-", marker="o"),
                    "two_scale_ou": dict(ls="--", marker="s")}
    kernel_text = {"trait_ou": "trait-state OU", "two_scale_ou": "two-scale OU"}

    # ---- panel (a): mean regret vs m, log-log, with the 2 eps_m envelope ----
    reg = _cell_stats(main_rows, ("kernel", "label", "m"), "regret_exhaustive")
    env = _cell_stats(main_rows, ("kernel", "label", "m"), "bound_2eps")

    env_lo, env_hi = [], []
    for m in M_LIST:
        vals = [env[(kn, ln, m)].mean() for kn in KERNEL_SPECS for ln in LABEL_SPECS]
        env_lo.append(min(vals))
        env_hi.append(max(vals))
    ax_a.fill_between(ms, env_lo, env_hi, color="0.75", alpha=0.55, lw=0,
                      label=r"$2\varepsilon_m$ envelope (theory)")

    for kn in KERNEL_SPECS:
        for ln in LABEL_SPECS:
            y = np.array([reg[(kn, ln, m)].mean() for m in M_LIST])
            y = np.where(y > 0, y, np.nan)
            ax_a.plot(ms, y, color=label_colour[ln], **kernel_style[kn], ms=3.2,
                      lw=1.2, alpha=0.95)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel(r"calibration sample size $m$")
    ax_a.set_ylabel(r"mean selection regret $I(S^\star)-I(\hat S)$")
    ax_a.set_xticks(list(M_LIST))
    ax_a.set_xticklabels([str(m) for m in M_LIST])
    ax_a.minorticks_off()
    # Extra head-room at the bottom so the legend never sits on the curves.
    y_lo = min(v for v in
               (reg[(kn, ln, m)].mean() for kn in KERNEL_SPECS
                for ln in LABEL_SPECS for m in M_LIST) if v > 0)
    ax_a.set_ylim(y_lo / 9.0, 1.6 * max(env_hi))

    handles = [plt.Line2D([], [], color=label_colour[ln], ls="-", marker="o",
                          ms=3.2, lw=1.2, label=label_text[ln]) for ln in LABEL_SPECS]
    handles += [plt.Line2D([], [], color="0.35", ms=3.2, lw=1.2,
                           label=kernel_text[kn], **kernel_style[kn])
                for kn in KERNEL_SPECS]
    handles += [plt.Rectangle((0, 0), 1, 1, color="0.75", alpha=0.55,
                              label=r"$2\varepsilon_m$ (theory)")]
    ax_a.legend(handles=handles, loc="lower left", ncol=2, handlelength=1.7,
                columnspacing=1.0, labelspacing=0.35)
    ax_a.text(0.97, 0.95, "(a)", transform=ax_a.transAxes, ha="right", va="top")

    # ---- panel (b): paired plug-in vs robust regret at small m -------------
    rob = _cell_stats(robust_rows, ("kernel", "label", "m"), "regret_robust")
    plg = _cell_stats(robust_rows, ("kernel", "label", "m"), "regret_plugin_greedy")
    marker_m = {ROBUST_M[0]: "o", ROBUST_M[1]: "^"}
    xs, ys = [], []
    for kn in KERNEL_SPECS:
        for ln in LABEL_SPECS:
            for m in ROBUST_M:
                x = plg[(kn, ln, m)]
                y = rob[(kn, ln, m)]
                xs.append(x.mean())
                ys.append(y.mean())
                ax_b.errorbar(x.mean(), y.mean(),
                              xerr=x.std(ddof=1) / np.sqrt(x.size),
                              yerr=y.std(ddof=1) / np.sqrt(y.size),
                              color=label_colour[ln], marker=marker_m[m],
                              ms=4.5, lw=0.9, mfc=("none" if kn == "two_scale_ou"
                                                   else label_colour[ln]),
                              mew=1.0, capsize=1.5, ls="none")
    lim = [0.0, 1.08 * max(max(xs), max(ys))]
    ax_b.plot(lim, lim, color="0.35", lw=0.9, ls=":")
    ax_b.set_xlim(lim)
    ax_b.set_ylim(lim)
    ax_b.set_xlabel(r"plug-in greedy regret (cell mean)")
    ax_b.set_ylabel(r"robust LCB greedy regret (cell mean)")
    hb = [plt.Line2D([], [], color=label_colour[ln], marker="o", ls="none",
                     ms=4.5, label=label_text[ln]) for ln in LABEL_SPECS]
    hb += [plt.Line2D([], [], color="0.35", marker=marker_m[m], ls="none", ms=4.5,
                      label=rf"$m={m}$") for m in ROBUST_M]
    hb += [plt.Line2D([], [], color="0.35", marker="o", ls="none", ms=4.5,
                      mfc="none", label="two-scale OU (open)")]
    hb += [plt.Line2D([], [], color="0.35", lw=0.9, ls=":", label="equal regret")]
    ax_b.legend(handles=hb, loc="upper left", handlelength=1.5)
    ax_b.text(0.97, 0.05, "(b)", transform=ax_b.transAxes, ha="right", va="bottom")

    fig.tight_layout()
    save_figure(fig, "fig_regret")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def _replot_from_csv() -> None:
    """Regenerate figures/fig_regret from the CSVs already on disk."""
    import csv
    from experiments.common import RESULTS

    def load(name: str) -> list[dict]:
        with (RESULTS / f"{name}.csv").open() as fh:
            rows = []
            for r in csv.DictReader(fh):
                rows.append({k: (v if k in ("kernel", "label") else float(v))
                             for k, v in r.items()})
            for r in rows:
                r["m"] = int(r["m"])
            return rows

    make_figure(load("s4_selection_regret"), load("s4_selection_regret_robust"))


def main() -> None:
    t_start = time.perf_counter()
    if "--replot" in sys.argv:
        _replot_from_csv()
        return
    quick = "--quick" in sys.argv
    n_reps = 6 if quick else N_REPS
    chunk = 3 if quick else CHUNK

    grid = make_grid()
    labels = make_labels()
    cands = make_candidates(grid)
    sw = Sweeper(grid, cands, labels, BUDGET)
    print(f"[S4] |V| = {len(cands)}, B = {BUDGET}, |Pi_B| = {len(sw.subsets)}, "
          f"p = {P_GRID}, T = {HORIZON}, reps = {n_reps}, seed = {SEED}")

    # Truth-side reference numbers (computed once, reported in the JSON).
    truth_ref = {}
    for kname in KERNEL_SPECS:
        K = make_truth(kname, grid)
        It = sw.sweep(K)
        truth_ref[kname] = {
            ln: {
                "I_star": float(It[ln].max()),
                "I_worst": float(It[ln].min()),
                "I_star_actions": [cands[i].tag
                                   for i in sw.subsets[int(np.argmax(It[ln]))]],
                "I_star_times": [cands[i].time
                                 for i in sw.subsets[int(np.argmax(It[ln]))]],
                "spread_over_Pi_B": float(It[ln].max() - It[ln].min()),
            } for ln in LABEL_SPECS
        }

    tasks = []
    for ki, kname in enumerate(KERNEL_SPECS):
        for mi, m in enumerate(M_LIST):
            for r0 in range(0, n_reps, chunk):
                tasks.append((ki, kname, mi, m, r0, min(r0 + chunk, n_reps)))

    main_rows: list[dict] = []
    robust_rows: list[dict] = []
    n_workers = min(16, max(1, (os.cpu_count() or 4) - 2))
    with Timer(f"S4 sweep ({len(tasks)} tasks, {n_workers} workers)"):
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_worker) as pool:
            for mr, rr in pool.map(_run_cell, tasks):
                main_rows.extend(mr)
                robust_rows.extend(rr)

    main_rows.sort(key=lambda r: (r["kernel"], r["label"], r["m"], r["rep"]))
    robust_rows.sort(key=lambda r: (r["kernel"], r["label"], r["m"], r["rep"]))
    save_csv(main_rows, "s4_selection_regret")
    save_csv(robust_rows, "s4_selection_regret_robust")

    # ---------------- aggregation ----------------
    viol_exh = sum(1 for r in main_rows if not r["bound_holds_exhaustive"])
    viol_eta = sum(1 for r in main_rows if not r["bound_holds_eta"])
    n_short = sum(1 for r in main_rows if r["n_greedy_actions"] != BUDGET)
    max_recon_err = max(abs(r["Ihat_at_Sgreedy"] - r["ceiling_from_designresult"])
                        for r in main_rows)

    curves: dict = {}
    for kname in KERNEL_SPECS:
        for ln in LABEL_SPECS:
            reg_e, reg_g, eps_c, eta_c, opt_c, slack_c = [], [], [], [], [], []
            for m in M_LIST:
                sub = [r for r in main_rows
                       if r["kernel"] == kname and r["label"] == ln and r["m"] == m]
                reg_e.append(float(np.mean([r["regret_exhaustive"] for r in sub])))
                reg_g.append(float(np.mean([r["regret_greedy"] for r in sub])))
                eps_c.append(float(np.mean([r["eps_m"] for r in sub])))
                eta_c.append(float(np.mean([r["eta_hat"] for r in sub])))
                opt_c.append(float(np.mean([r["greedy_is_optimal"] for r in sub])))
                slack_c.append(float(np.mean([r["slack_exhaustive"] for r in sub])))
            curves[f"{kname}|{ln}"] = {
                "m": list(M_LIST),
                "mean_regret_exhaustive": reg_e,
                "mean_regret_greedy": reg_g,
                "mean_eps_m": eps_c,
                "mean_eta_hat": eta_c,
                "frac_greedy_hits_plugin_optimum": opt_c,
                "mean_slack_2eps_minus_regret": slack_c,
                "loglog_slope_regret": _slope_loglog(np.array(M_LIST, float),
                                                     np.array(reg_e)),
                "loglog_slope_eps": _slope_loglog(np.array(M_LIST, float),
                                                  np.array(eps_c)),
            }

    # robust vs plug-in, paired
    robust_cells = {}
    wins = ties = losses = 0
    for kname in KERNEL_SPECS:
        for ln in LABEL_SPECS:
            for m in ROBUST_M:
                sub = [r for r in robust_rows if r["kernel"] == kname
                       and r["label"] == ln and r["m"] == m]
                d = np.array([r["diff_robust_minus_plugin"] for r in sub])
                rr = np.array([r["regret_robust"] for r in sub])
                pp = np.array([r["regret_plugin_greedy"] for r in sub])
                ee = np.array([r["regret_plugin_exhaustive"] for r in sub])
                n_win = int(np.sum(d < -1e-15))
                n_tie = int(np.sum(np.abs(d) <= 1e-15))
                n_loss = int(np.sum(d > 1e-15))
                wins += n_win
                ties += n_tie
                losses += n_loss
                se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else float("nan")
                robust_cells[f"{kname}|{ln}|m={m}"] = {
                    "n": int(d.size),
                    "mean_regret_robust": float(rr.mean()),
                    "mean_regret_plugin_greedy": float(pp.mean()),
                    "mean_regret_plugin_exhaustive": float(ee.mean()),
                    "mean_paired_difference": float(d.mean()),
                    "se_paired_difference": se,
                    "t_stat": float(d.mean() / se) if se and se > 0 else float("nan"),
                    "robust_wins": n_win, "ties": n_tie, "robust_losses": n_loss,
                    "frac_same_protocol": float(np.mean(
                        [r["same_protocol"] for r in sub])),
                }
    cells_robust_better = sum(1 for v in robust_cells.values()
                              if v["mean_paired_difference"] < 0)

    headline = {
        "n_replications": n_reps,
        "n_replications_robust": min(N_REPS_ROBUST, n_reps),
        "seed": SEED,
        "n_candidates": len(cands),
        "budget_B": BUDGET,
        "n_protocols_Pi_B": len(sw.subsets),
        "grid_p": P_GRID, "horizon_T": HORIZON,
        "action_noise": ACTION_NOISE,
        "n_bootstrap": N_BOOT, "robust_quantile": ROBUST_QUANTILE,
        # --- the guarantee checks ---
        "violations_regret_le_2eps": viol_exh,
        "violations_eta_bound": viol_eta,
        "n_checks": len(main_rows),
        "min_slack_2eps_minus_regret": float(min(r["slack_exhaustive"]
                                                 for r in main_rows)),
        "min_slack_eta_bound": float(min(r["slack_eta"] for r in main_rows)),
        "median_slack_ratio_regret_over_2eps": float(np.median(
            [r["regret_exhaustive"] / r["bound_2eps"] for r in main_rows
             if r["bound_2eps"] > 0])),
        "max_ratio_regret_over_2eps": float(max(
            r["regret_exhaustive"] / r["bound_2eps"] for r in main_rows
            if r["bound_2eps"] > 0)),
        # --- greedy arm ---
        "mean_eta_hat_overall": float(np.mean([r["eta_hat"] for r in main_rows])),
        "min_eta_hat_overall": float(min(r["eta_hat"] for r in main_rows)),
        "frac_greedy_hits_plugin_optimum": float(np.mean(
            [r["greedy_is_optimal"] for r in main_rows])),
        "n_greedy_short_of_budget": n_short,
        "max_ceiling_reconstruction_error": float(max_recon_err),
        # --- regret levels ---
        "mean_regret_at_m25": float(np.mean([r["regret_exhaustive"] for r in main_rows
                                             if r["m"] == 25])),
        "mean_regret_at_m1000": float(np.mean([r["regret_exhaustive"] for r in main_rows
                                               if r["m"] == 1000])),
        "mean_eps_at_m25": float(np.mean([r["eps_m"] for r in main_rows if r["m"] == 25])),
        "mean_eps_at_m1000": float(np.mean([r["eps_m"] for r in main_rows
                                            if r["m"] == 1000])),
        "frac_zero_regret_at_m1000": float(np.mean(
            [r["regret_exhaustive"] <= 1e-15 for r in main_rows if r["m"] == 1000])),
        "frac_zero_regret_at_m25": float(np.mean(
            [r["regret_exhaustive"] <= 1e-15 for r in main_rows if r["m"] == 25])),
        "mean_loglog_slope_regret": float(np.nanmean(
            [c["loglog_slope_regret"] for c in curves.values()])),
        "mean_loglog_slope_eps": float(np.nanmean(
            [c["loglog_slope_eps"] for c in curves.values()])),
        # --- robust arm ---
        "robust_cells_with_lower_mean_regret": cells_robust_better,
        "robust_n_cells": len(robust_cells),
        "robust_wins": wins, "robust_ties": ties, "robust_losses": losses,
        "robust_mean_paired_difference": float(np.mean(
            [r["diff_robust_minus_plugin"] for r in robust_rows])),
        "robust_overall_mean_regret": float(np.mean(
            [r["regret_robust"] for r in robust_rows])),
        "plugin_greedy_overall_mean_regret": float(np.mean(
            [r["regret_plugin_greedy"] for r in robust_rows])),
        "plugin_exhaustive_overall_mean_regret": float(np.mean(
            [r["regret_plugin_exhaustive"] for r in robust_rows])),
    }
    payload = {
        "headline": headline,
        "truth": truth_ref,
        "curves": curves,
        "robust_cells": robust_cells,
        "config": {
            "kernels": {k: {kk: (vv if not callable(vv) else str(vv))
                            for kk, vv in v.items()} for k, v in KERNEL_SPECS.items()},
            "labels": list(LABEL_SPECS),
            "m_list": list(M_LIST),
            "robust_m": list(ROBUST_M),
            "candidate_tags": [a.tag for a in cands],
            "candidate_times": [a.time for a in cands],
            "candidate_widths": [a.width for a in cands],
            "candidate_segments": [a.n_segments for a in cands],
            "calibration": "noiseless dense trajectories, K_hat = fit_covariance(Z).K",
        },
        "environment": environment_record(),
    }

    make_figure(main_rows, robust_rows)
    payload["headline"]["runtime_seconds"] = float(time.perf_counter() - t_start)
    save_json(payload, "s4_selection_regret")

    print("\n--- guarantee checks ---")
    print(f"  regret <= 2 eps_m       : {len(main_rows) - viol_exh}/{len(main_rows)}"
          f" ({viol_exh} violations)")
    print(f"  eta-approximate bound   : {len(main_rows) - viol_eta}/{len(main_rows)}"
          f" ({viol_eta} violations)")
    print(f"  min slack (2eps-regret) : {headline['min_slack_2eps_minus_regret']:.3e}")
    print(f"  median regret/(2 eps_m) : {headline['median_slack_ratio_regret_over_2eps']:.4f}")
    print(f"  mean eta_hat            : {headline['mean_eta_hat_overall']:.6f}"
          f"  (min {headline['min_eta_hat_overall']:.6f})")
    print("\n--- robust vs plug-in (paired, small m) ---")
    print(f"  cells where robust has lower mean regret: "
          f"{cells_robust_better}/{len(robust_cells)}")
    print(f"  per-replication wins/ties/losses: {wins}/{ties}/{losses}")
    print(f"  mean paired difference (robust - plug-in): "
          f"{headline['robust_mean_paired_difference']:+.5f}")
    print(f"\n[S4] total runtime {payload['headline']['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
