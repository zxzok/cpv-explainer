"""Experiment S5 -- label-aware observation design versus label-agnostic baselines.

Every method below produces a set of observation actions ``S`` and is then scored
by *the same* exact objective

    I_g(S) = F_g(S; K) / V_g(K),    F_g(S; K) = sum_jk w_j w_k C_g(Q_S(K)_jk),

so the comparison is a comparison of *placements*, never of surrogate criteria.
Relative efficiency is ``I_g(S_method) / I_g(S*)`` with ``S*`` the exhaustive
budget-feasible optimum, which is computed exactly on every instance here (the
candidate sets are deliberately small enough for that; ``|V|`` and the number of
enumerated subsets are recorded in the outputs).

Four instance families isolate three distinct ways of breaking the symmetry that
makes label-awareness pointless:

  (i)   stationary kernel (OU and Matern-3/2), uniform label weights.
        *Negative control.*  The problem is (up to boundary effects) translation
        symmetric, so every method lands on essentially the same placement and
        label-awareness buys nothing.
  (ii)  non-stationary kernel whose local correlation time increases across the
        horizon (``nonstationary_correlation``, copied from the test suite).
        The kernel breaks the symmetry.
  (iii) heterogeneous candidate actions: three window widths with cost
        proportional to width, and per-action measurement noise increasing over
        the horizon.  The *action set* breaks the symmetry.
  (iv)  recency-weighted label weights (``recency_weight``).  The *label* breaks
        the symmetry.

Outputs
-------
``results/s5_design.csv``              one row per (family, label, method)
``results/s5_design.json``             headline numbers + submodularity report
``figures/fig_design.pdf/.png``        (a) selected times on family (ii) over the
                                       local correlation time, (b) relative
                                       efficiency bars
``figures/fig_design_crossmatrix.pdf/.png``  design x label cross-efficiency

Run with::

    .venv/bin/python experiments/synthetic/s5_design.py
"""

from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import (METHOD_STYLE, PALETTE, SEED, Timer,
                                environment_record, save_csv, save_figure,
                                save_json, setup_matplotlib)
from protocol_ceiling import (Action, MeanLabel, ThresholdLabel, TimeGrid,
                              TwoSidedLabel, bin_midpoints, design_imse,
                              design_kernel_quadrature,
                              design_mutual_information, design_same_time,
                              design_uniform, find_submodularity_violation,
                              label_variance, make_kernel, project_psd,
                              recency_weight, select_protocol_exhaustive,
                              select_protocol_greedy, sigmoid_label,
                              submodularity_ratio_certificate, to_correlation,
                              trait_state_correlation, uniform_grid)
from protocol_ceiling.design import nonlinear_ratio_lower_bound, objective

Array = np.ndarray

# --------------------------------------------------------------------------
# Fixed experiment constants (all replication counts are stated here)
# --------------------------------------------------------------------------
HORIZON = 10.0
P_GRID = 128                # latent grid points
N_RANDOM_RESTARTS = 20      # "random" baseline = best of 20 restarts
N_VIOLATION_TRIALS = 600    # random search for a submodularity violation
N_RATIO_SUBSETS = 32        # extension sets per greedy prefix in the certificate
SIGMOID_KMAX = 60           # truncated Hermite spectrum (see the assertion below)

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
LABELS: list[tuple[str, str, object]] = [
    ("mean", r"mean", MeanLabel()),
    ("occupation_c0", r"occupation $c\!=\!0$", ThresholdLabel(c=0.0)),
    ("occupation_c1.5", r"occupation $c\!=\!1.5$", ThresholdLabel(c=1.5)),
    ("two_sided_c1.2", r"two-sided $c\!=\!1.2$", TwoSidedLabel(c=1.2)),
    ("sigmoid", r"sigmoid $s\!=\!2$", sigmoid_label(2.0, 0.0, kmax=SIGMOID_KMAX)),
]
LABEL_KEYS = [k for k, _, _ in LABELS]
LABEL_TEX = {k: t for k, t, _ in LABELS}
LABEL_OBJ = {k: g for k, _, g in LABELS}
LABEL_SHORT = {"mean": "mean", "occupation_c0": r"occ. $0$",
               "occupation_c1.5": r"occ. $1.5$", "two_sided_c1.2": r"2-sided $1.2$",
               "sigmoid": "sigmoid"}

METHODS = ["same_time", "uniform", "dispersed", "random", "mutual_information",
           "imse", "linear_target", "kernel_quadrature", "label_aware_greedy", "label_aware",
           "exhaustive"]
BAR_METHODS = ["same_time", "uniform", "random", "mutual_information", "imse",
               "linear_target", "kernel_quadrature", "label_aware_greedy", "label_aware"]

STYLE = dict(METHOD_STYLE)
STYLE["dispersed"] = dict(color=PALETTE[6], marker="d", ls=":",
                          label="Dispersed (bin midpoints)")


# --------------------------------------------------------------------------
# Non-stationary kernel (copied verbatim from tests/test_identifiability_design.py)
# --------------------------------------------------------------------------
def local_correlation_time(grid: TimeGrid, tau_lo: float, tau_hi: float) -> Array:
    t = grid.times
    return tau_lo + (tau_hi - tau_lo) * (t / t.max())


def nonstationary_correlation(grid: TimeGrid, tau_lo: float = 0.25,
                              tau_hi: float = 2.75) -> Array:
    """Local correlation time increasing across the horizon.

    A stationary kernel with uniform label weights makes the design problem
    translation-symmetric, so *every* label lands on the same symmetric
    placement and the label-dependence of the optimum is invisible.  Letting the
    local correlation time vary breaks that symmetry and exposes the mechanism:
    the mean label sees only tau_1, whereas order-k Hermite components see
    tau_k = int rho^k, which concentrates ever more sharply on the slowly
    decorrelating part of the horizon.
    """
    t = grid.times
    tau_t = tau_lo + (tau_hi - tau_lo) * (t / t.max())
    lag = np.abs(t[:, None] - t[None, :])
    K = np.exp(-lag / np.sqrt(np.outer(tau_t, tau_t)))
    np.fill_diagonal(K, 1.0)
    return to_correlation(project_psd(K, 1e-9))


# --------------------------------------------------------------------------
# Instance families
# --------------------------------------------------------------------------
@dataclass
class Family:
    key: str
    title: str
    grid: TimeGrid
    K: Array
    candidates: list[Action]
    budget: float
    n_select: int          # cardinality handed to the cardinality-based baselines
    max_size: int          # largest cardinality affordable under the budget
    homogeneous: bool      # all candidate costs equal to 1
    noise_fn: Callable[[float], float]
    tau_fn: Callable[[Array], Array] | None = None
    notes: str = ""
    n_subsets_enumerated: int = 0
    n_subsets_feasible: int = 0


def _point_candidates(times: Array, noise: float) -> list[Action]:
    return [Action(time=float(t), width=0.0, n_segments=1, noise=float(noise),
                   cost=1.0, tag=f"t={t:.3f}") for t in times]


def build_families() -> list[Family]:
    fams: list[Family] = []
    grid = uniform_grid(HORIZON, P_GRID)
    cand_times = np.linspace(0.5, 9.5, 12)

    # ---- (i) stationary, uniform weights ------------------------------
    for kern, tex in (("ou", "OU"), ("matern32", "Matern-3/2")):
        noise = 0.4
        fams.append(Family(
            key=f"i_stationary_{kern}",
            title=f"(i) stationary {tex}, uniform weights",
            grid=grid,
            K=trait_state_correlation(grid, 0.0, make_kernel(kern, tau=1.0)),
            candidates=_point_candidates(cand_times, noise),
            budget=4.0, n_select=4, max_size=4, homogeneous=True,
            noise_fn=lambda t, nz=noise: nz,
            notes=f"kernel={kern}(tau=1), T/tau=10, nu^2={noise}, point actions",
        ))

    # ---- (ii) non-stationary correlation time -------------------------
    tau_lo, tau_hi, noise = 0.5, 4.0, 0.15
    fams.append(Family(
        key="ii_nonstationary",
        title="(ii) non-stationary correlation time",
        grid=grid,
        K=nonstationary_correlation(grid, tau_lo, tau_hi),
        candidates=_point_candidates(cand_times, noise),
        budget=4.0, n_select=4, max_size=4, homogeneous=True,
        noise_fn=lambda t, nz=noise: nz,
        tau_fn=lambda t, a=tau_lo, b=tau_hi: a + (b - a) * (t / t.max()),
        notes=f"tau(t) linear from {tau_lo} to {tau_hi}, nu^2={noise}, point actions",
    ))

    # ---- (iii) heterogeneous actions ----------------------------------
    fams.append(_family_heterogeneous(grid))

    # ---- (iv) recency-weighted label ----------------------------------
    half_life, noise = 6.0, 0.4
    grid_r = uniform_grid(HORIZON, P_GRID, weight_fn=recency_weight(half_life))
    fams.append(Family(
        key="iv_recency",
        title="(iv) recency-weighted label",
        grid=grid_r,
        K=trait_state_correlation(grid_r, 0.0, make_kernel("ou", tau=1.0)),
        candidates=_point_candidates(cand_times, noise),
        budget=4.0, n_select=4, max_size=4, homogeneous=True,
        noise_fn=lambda t, nz=noise: nz,
        notes=f"OU(tau=1), omega_j prop. exp(-(T-t_j)/{half_life}), nu^2={noise}",
    ))
    return fams


# Configuration of family (iii).  Chosen (before any method was run) so that the
# three window widths are genuinely competitive: with a short correlation time
# and a large measurement noise a wide window is a *cheap* way of buying a
# low-noise view of a temporal average, which is exactly what a mean-type label
# wants and exactly what a strongly non-linear label does not.
HET_CONFIG = dict(tau=0.7, widths=(0.0, 1.5, 3.5), cost_per_width=0.15,
                  noise_lo=1.0, noise_hi=3.0, n_times=6, budget=4.0, max_size=4)


def _family_heterogeneous(grid: TimeGrid) -> Family:
    cfg = HET_CONFIG
    K = trait_state_correlation(grid, 0.0, make_kernel("ou", tau=cfg["tau"]))
    times = np.linspace(0.5, 9.5, cfg["n_times"])

    def noise_fn(t: float) -> float:
        return cfg["noise_lo"] + (cfg["noise_hi"] - cfg["noise_lo"]) * (t / HORIZON)

    cands: list[Action] = []
    for t in times:
        for w in cfg["widths"]:
            cands.append(Action(time=float(t), width=float(w), n_segments=1,
                                noise=float(noise_fn(float(t))),
                                cost=float(1.0 + cfg["cost_per_width"] * w),
                                tag=f"t={t:.2f},w={w:g}"))
    return Family(
        key="iii_heterogeneous",
        title="(iii) heterogeneous widths, costs and noise",
        grid=grid, K=K, candidates=cands,
        budget=cfg["budget"], n_select=4, max_size=cfg["max_size"],
        homogeneous=False, noise_fn=noise_fn,
        notes=(f"OU(tau={cfg['tau']}), widths {cfg['widths']}, "
               f"c_a = 1 + {cfg['cost_per_width']}*w_a, "
               f"nu_a^2 linear from {cfg['noise_lo']} to {cfg['noise_hi']}, B={cfg['budget']}"),
    )


# --------------------------------------------------------------------------
# Evaluation helpers -- every method is scored by the same exact objective
# --------------------------------------------------------------------------
def score(label, fam: Family, acts: Sequence[Action]) -> tuple[float, float, float]:
    F = objective(label, fam.K, fam.grid, list(acts)) if len(acts) else 0.0
    V = label_variance(label, fam.K, fam.grid.weights)
    return F, (F / V if V > 0 else 0.0), float(sum(a.cost for a in acts))


def truncate_to_budget(acts: Sequence[Action], budget: float) -> list[Action]:
    """Longest *prefix* of a greedy ordering that fits the budget."""
    out: list[Action] = []
    c = 0.0
    for a in acts:
        if c + a.cost > budget + 1e-12:
            break
        out.append(a)
        c += a.cost
    return out


def build_same_time(fam: Family, n: int) -> list[Action]:
    t = 0.5 * (fam.grid.times[0] + fam.grid.times[-1])
    return [Action(time=float(t), width=0.0, n_segments=int(n),
                   noise=float(fam.noise_fn(float(t))), cost=float(n),
                   tag="same-time")]


def build_uniform(fam: Family, n: int) -> list[Action]:
    ts = np.linspace(fam.grid.times[0], fam.grid.times[-1], n)
    return [Action(time=float(t), width=0.0, n_segments=1,
                   noise=float(fam.noise_fn(float(t))), cost=1.0, tag="uniform")
            for t in ts]


def build_dispersed(fam: Family, n: int) -> list[Action]:
    return [Action(time=float(t), width=0.0, n_segments=1,
                   noise=float(fam.noise_fn(float(t))), cost=1.0, tag="dispersed")
            for t in bin_midpoints(fam.grid.horizon, n)]


def random_designs(fam: Family, rng: np.random.Generator, n_restarts: int) -> list[list[Action]]:
    """``n_restarts`` random budget-feasible protocols, drawn once per family."""
    designs = []
    for _ in range(n_restarts):
        order = rng.permutation(len(fam.candidates))
        acts: list[Action] = []
        c = 0.0
        for i in order:
            a = fam.candidates[int(i)]
            if c + a.cost <= fam.budget + 1e-12:
                acts.append(a)
                c += a.cost
            if len(acts) >= fam.max_size:
                break
        designs.append(acts)
    return designs


def exhaustive_optimum(fam: Family, label) -> tuple[list[Action], int, int, float]:
    """Exact optimum; returns (actions, n_enumerated, n_evaluated, runtime)."""
    t0 = time.perf_counter()
    if fam.homogeneous:
        res = select_protocol_exhaustive(label, fam.K, fam.grid, fam.candidates,
                                         n_select=fam.n_select)
        return (list(res.actions), res.n_evaluations, res.n_evaluations,
                time.perf_counter() - t0)
    best_F, best = -np.inf, []
    n_enum = n_eval = 0
    for k in range(1, fam.max_size + 1):
        for combo in itertools.combinations(range(len(fam.candidates)), k):
            n_enum += 1
            acts = [fam.candidates[i] for i in combo]
            if sum(a.cost for a in acts) > fam.budget + 1e-12:
                continue
            F = objective(label, fam.K, fam.grid, acts)
            n_eval += 1
            if F > best_F:
                best_F, best = F, acts
    return best, n_enum, n_eval, time.perf_counter() - t0


# --------------------------------------------------------------------------
# Submodularity diagnostics
# --------------------------------------------------------------------------
def minimal_violation(label, fam: Family) -> dict | None:
    """Exhaustive scan of the *smallest possible* violating triples.

    ``S = {}``, ``T = {b}``: a pair ``(a, b)`` with
    ``Delta(a | {}) < Delta(a | {b})`` is the minimal certificate that ``F_g``
    is not submodular, and it can be found by ``O(|V|^2)`` exact evaluations.
    """
    cands = fam.candidates
    n = len(cands)
    F1 = np.array([objective(label, fam.K, fam.grid, [cands[i]]) for i in range(n)])
    Fpair = {}
    for i, j in itertools.combinations(range(n), 2):
        Fpair[(i, j)] = objective(label, fam.K, fam.grid, [cands[i], cands[j]])
    best = None
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            fij = Fpair[(i, j)] if i < j else Fpair[(j, i)]
            dS, dT = float(F1[i]), float(fij - F1[j])
            if dT > dS + 1e-14:
                rel = (dT - dS) / max(dS, 1e-14)
                if best is None or rel > best["relative_violation"]:
                    best = {"a_time": cands[i].time, "a_width": cands[i].width,
                            "b_time": cands[j].time, "b_width": cands[j].width,
                            "size_S": 0, "size_T": 1,
                            "gain_S": dS, "gain_T": dT,
                            "relative_violation": float(rel)}
    return best


def is_one_swap_local_optimum(fam: Family, label, acts: Sequence[Action]) -> bool:
    """Independent check that a design cannot be improved by any single swap.

    Used to attribute the (small) shortfall of the local search on some
    instances to the 1-swap *neighbourhood* rather than to a premature stop.
    """
    acts = list(acts)
    base = objective(label, fam.K, fam.grid, acts)
    for i in range(len(acts)):
        for cand in fam.candidates:
            if any(cand is c for c in acts):
                continue
            trial = acts[:i] + [cand] + acts[i + 1:]
            if sum(a.cost for a in trial) > fam.budget + 1e-12:
                continue
            if objective(label, fam.K, fam.grid, trial) > base + 1e-14:
                return False
    return True


def submodularity_report(fam: Family, rng: np.random.Generator) -> list[dict]:
    rows = []
    for key, _, label in LABELS:
        minimal = minimal_violation(label, fam)
        worst = find_submodularity_violation(label, fam.K, fam.grid, fam.candidates,
                                             rng, n_trials=N_VIOLATION_TRIALS)
        greedy = select_protocol_greedy(label, fam.K, fam.grid, fam.candidates,
                                        budget=fam.budget,
                                        cost_aware=not fam.homogeneous)
        prefixes = [greedy.actions[:k] for k in range(len(greedy.actions))]
        cert = submodularity_ratio_certificate(label, fam.K, fam.grid, fam.candidates,
                                               prefixes, rng=rng,
                                               n_subsets=N_RATIO_SUBSETS)
        row = {
            "family": fam.key, "label": key,
            "minimal_violation_found": minimal is not None,
            "minimal_size_S": None if minimal is None else minimal["size_S"],
            "minimal_size_T": None if minimal is None else minimal["size_T"],
            "minimal_gain_S": None if minimal is None else minimal["gain_S"],
            "minimal_gain_T": None if minimal is None else minimal["gain_T"],
            "minimal_a_time": None if minimal is None else minimal["a_time"],
            "minimal_a_width": None if minimal is None else minimal["a_width"],
            "minimal_b_time": None if minimal is None else minimal["b_time"],
            "minimal_b_width": None if minimal is None else minimal["b_width"],
            "minimal_relative_violation": None if minimal is None else minimal["relative_violation"],
            "random_search_action_time": None if worst is None else worst["action"].time,
            "random_search_action_width": None if worst is None else worst["action"].width,
            "random_search_size_S": None if worst is None else len(worst["S"]),
            "random_search_size_T": None if worst is None else len(worst["T"]),
            "random_search_gain_S": None if worst is None else worst["gain_S"],
            "random_search_gain_T": None if worst is None else worst["gain_T"],
            "random_search_relative_violation": None if worst is None else worst["relative_violation"],
            "gamma": cert["gamma"],
            "gamma_median": cert.get("gamma_median", float("nan")),
            "gamma_n_samples": cert["n_samples"],
            "greedy_factor_1_minus_exp_minus_gamma": cert["greedy_factor"],
            "analytic_transfer_bound": float(
                nonlinear_ratio_lower_bound(label, cert["gamma"], rmax=0.99)),
        }
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Main sweep
# --------------------------------------------------------------------------
def run_family(fam: Family, rng: np.random.Generator) -> tuple[list[dict], dict]:
    """Return (rows, per-family bookkeeping)."""
    n = fam.n_select

    # -- label-agnostic selections are computed once and reused ----------
    t0 = time.perf_counter()
    mi = design_mutual_information(LABEL_OBJ["mean"], fam.K, fam.grid,
                                   fam.candidates, n)
    mi_acts = truncate_to_budget(list(mi.actions), fam.budget)
    mi_time, mi_eval = time.perf_counter() - t0, mi.n_evaluations

    t0 = time.perf_counter()
    im = design_imse(LABEL_OBJ["mean"], fam.K, fam.grid, fam.candidates, n)
    im_acts = truncate_to_budget(list(im.actions), fam.budget)
    im_time, im_eval = time.perf_counter() - t0, im.n_evaluations

    t0 = time.perf_counter()
    kq = design_kernel_quadrature(LABEL_OBJ["mean"], fam.K, fam.grid,
                                  fam.candidates, n)
    kq_acts = truncate_to_budget(list(kq.actions), fam.budget)
    kq_time, kq_eval = time.perf_counter() - t0, kq.n_evaluations

    # Noise-aware linear-target reference. It is selected once with C(r)=r and
    # the actual action noises, then scored under every nonlinear target.
    t0 = time.perf_counter()
    lin = select_protocol_greedy(
        LABEL_OBJ["mean"], fam.K, fam.grid, fam.candidates,
        budget=fam.budget, cost_aware=not fam.homogeneous, local_search=True)
    lin_acts = list(lin.actions)
    lin_time, lin_eval = time.perf_counter() - t0, lin.n_evaluations

    fixed = {
        "same_time": (build_same_time(fam, n), 0.0, 1),
        "uniform": (build_uniform(fam, n), 0.0, 1),
        "dispersed": (build_dispersed(fam, n), 0.0, 1),
        "mutual_information": (mi_acts, mi_time, mi_eval),
        "imse": (im_acts, im_time, im_eval),
        "linear_target": (lin_acts, lin_time, lin_eval),
        "kernel_quadrature": (kq_acts, kq_time, kq_eval),
    }

    # cross-check the hand-built baselines against the library versions
    if fam.homogeneous:
        tmpl = fam.candidates[0]
        lib_u = design_uniform(LABEL_OBJ["mean"], fam.K, fam.grid, n, tmpl).objective
        own_u = objective(LABEL_OBJ["mean"], fam.K, fam.grid, fixed["uniform"][0])
        lib_s = design_same_time(LABEL_OBJ["mean"], fam.K, fam.grid, n, tmpl).objective
        own_s = objective(LABEL_OBJ["mean"], fam.K, fam.grid, fixed["same_time"][0])
        assert abs(lib_u - own_u) < 1e-12, (lib_u, own_u)
        assert abs(lib_s - own_s) < 1e-12, (lib_s, own_s)

    rand_pool = random_designs(fam, rng, N_RANDOM_RESTARTS)

    rows: list[dict] = []
    optima: dict[str, list[Action]] = {}
    local_opt: dict[str, bool] = {}
    for key, _, label in LABELS:
        designs: dict[str, tuple[list[Action], float, int]] = dict(fixed)

        t0 = time.perf_counter()
        best_acts, best_F = rand_pool[0], -np.inf
        for acts in rand_pool:
            F = objective(label, fam.K, fam.grid, acts)
            if F > best_F:
                best_F, best_acts = F, acts
        designs["random"] = (best_acts, time.perf_counter() - t0, N_RANDOM_RESTARTS)

        gr = select_protocol_greedy(label, fam.K, fam.grid, fam.candidates,
                                    budget=fam.budget,
                                    cost_aware=not fam.homogeneous,
                                    local_search=False)
        designs["label_aware_greedy"] = (list(gr.actions), gr.runtime, gr.n_evaluations)

        la = select_protocol_greedy(label, fam.K, fam.grid, fam.candidates,
                                    budget=fam.budget,
                                    cost_aware=not fam.homogeneous,
                                    local_search=True)
        designs["label_aware"] = (list(la.actions), la.runtime, la.n_evaluations)
        local_opt[key] = is_one_swap_local_optimum(fam, label, la.actions)

        ex_acts, n_enum, n_eval, ex_time = exhaustive_optimum(fam, label)
        designs["exhaustive"] = (ex_acts, ex_time, n_eval)
        fam.n_subsets_enumerated = n_enum
        fam.n_subsets_feasible = n_eval
        optima[key] = ex_acts

        F_star, I_star, _ = score(label, fam, ex_acts)
        for meth in METHODS:
            acts, rt, nev = designs[meth]
            F, I, cost = score(label, fam, acts)
            rows.append({
                "family": fam.key,
                "family_title": fam.title,
                "label": key,
                "method": meth,
                "objective_F": F,
                "ceiling_I": I,
                "relative_efficiency": (I / I_star) if I_star > 0 else float("nan"),
                "cost": cost,
                "n_actions": len(acts),
                "times": ";".join(f"{a.time:.3f}" for a in sorted(acts, key=lambda x: x.time)),
                "widths": ";".join(f"{a.width:g}" for a in sorted(acts, key=lambda x: x.time)),
                "segments": ";".join(str(a.n_segments) for a in sorted(acts, key=lambda x: x.time)),
                "runtime_s": float(rt),
                "n_evaluations": int(nev),
                "n_candidates": len(fam.candidates),
                "budget": fam.budget,
            })

    # -- design x label cross-efficiency --------------------------------
    cross = np.zeros((len(LABELS), len(LABELS)))
    for i, kd in enumerate(LABEL_KEYS):
        for j, kt in enumerate(LABEL_KEYS):
            lab = LABEL_OBJ[kt]
            F_d = objective(lab, fam.K, fam.grid, optima[kd])
            F_o = objective(lab, fam.K, fam.grid, optima[kt])
            cross[i, j] = F_d / F_o if F_o > 0 else float("nan")

    info = {
        "title": fam.title,
        "notes": fam.notes,
        "n_candidates": len(fam.candidates),
        "budget": fam.budget,
        "n_select": fam.n_select,
        "homogeneous_costs": fam.homogeneous,
        "n_subsets_enumerated": fam.n_subsets_enumerated,
        "n_subsets_evaluated": fam.n_subsets_feasible,
        "optimal_times": {k: [round(a.time, 4) for a in sorted(v, key=lambda x: x.time)]
                          for k, v in optima.items()},
        "optimal_widths": {k: [a.width for a in sorted(v, key=lambda x: x.time)]
                           for k, v in optima.items()},
        "n_distinct_optima": len({tuple(sorted((round(a.time, 6), a.width, a.n_segments)
                                               for a in v)) for v in optima.values()}),
        "label_aware_is_1swap_local_optimum": local_opt,
        "cross_efficiency": cross.tolist(),
        "cross_efficiency_min": float(np.nanmin(cross)),
    }
    return rows, info


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def figure_design(plt, fams: dict[str, Family], rows: list[dict],
                  infos: dict[str, dict]) -> None:
    fam = fams["ii_nonstationary"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2),
                             gridspec_kw={"width_ratios": [1.0, 1.45]})

    # -- (a) selected times over the local correlation time -------------
    ax = axes[0]
    axb = ax.twinx()
    t = fam.grid.times
    tau_t = fam.tau_fn(t)
    axb.fill_between(t, 0.0, tau_t, color="0.88", lw=0, zorder=0)
    axb.plot(t, tau_t, color="0.55", lw=1.0, zorder=1)
    axb.set_ylim(0.0, float(tau_t.max()) * 2.3)
    axb.set_ylabel(r"local correlation time $\tau(t)$", fontsize=8)
    axb.grid(False)
    ax.set_zorder(axb.get_zorder() + 1)
    ax.patch.set_visible(False)

    opt = infos["ii_nonstationary"]["optimal_times"]
    ncand = [a.time for a in fam.candidates]
    for i, key in enumerate(LABEL_KEYS):
        y = len(LABEL_KEYS) - i
        ax.plot([0.0, HORIZON], [y, y], color="0.8", lw=0.6, zorder=1)
        ax.plot(ncand, [y] * len(ncand), "|", color="0.8", ms=4, zorder=2)
        ax.plot(opt[key], [y] * len(opt[key]), "o", color=PALETTE[i], ms=5.5,
                mec="white", mew=0.5, zorder=4)
    ax.set_yticks(range(1, len(LABEL_KEYS) + 1))
    ax.set_yticklabels([LABEL_TEX[k] for k in LABEL_KEYS][::-1], fontsize=7.5)
    ax.set_ylim(0.4, len(LABEL_KEYS) + 0.6)
    ax.set_xlim(0.0, HORIZON)
    ax.set_xlabel(r"time $t$")
    ax.grid(False)
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, va="top", fontsize=9)

    # -- (b) relative efficiency bars -----------------------------------
    ax = axes[1]
    sub = {(r["label"], r["method"]): r["relative_efficiency"] for r in rows
           if r["family"] == "ii_nonstationary"}
    nb = len(BAR_METHODS)
    width = 0.84 / nb
    x = np.arange(len(LABEL_KEYS))
    for m, meth in enumerate(BAR_METHODS):
        vals = [sub[(k, meth)] for k in LABEL_KEYS]
        st = STYLE[meth]
        hatched = meth == "label_aware_greedy"
        ax.bar(x + (m - (nb - 1) / 2) * width, vals, width * 0.92,
               color=st["color"], label=st["label"],
               edgecolor="white" if hatched else "none",
               linewidth=0.0, hatch="///" if hatched else None)
    ax.axhline(1.0, color="k", lw=0.8, ls="-")
    lo = min(min(sub[(k, m)] for k in LABEL_KEYS) for m in BAR_METHODS)
    ax.set_ylim(max(0.0, lo - 0.06), 1.035)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_SHORT[k] for k in LABEL_KEYS], fontsize=7.5)
    ax.set_xlabel("label $g$")
    ax.set_ylabel(r"relative efficiency $I_g(S)/I_g(S^\ast)$")
    ax.text(0.01, 0.97, "(b)", transform=ax.transAxes, va="top", fontsize=9)

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))
    fig.legend(handles, labels, ncol=4, fontsize=7.0, loc="upper center",
               bbox_to_anchor=(0.5, 0.035), handlelength=1.3,
               columnspacing=1.1, handletextpad=0.45)
    save_figure(fig, "fig_design")
    plt.close(fig)


def figure_crossmatrix(plt, infos: dict[str, dict], family_keys: Sequence[str]) -> None:
    fig, axes = plt.subplots(1, len(family_keys), figsize=(7.0, 3.3))
    axes = np.atleast_1d(axes)
    for k, (fkey, ax) in enumerate(zip(family_keys, axes)):
        C = np.asarray(infos[fkey]["cross_efficiency"])
        lo = float(np.nanmin(C))
        im = ax.imshow(C, cmap="viridis", vmin=lo, vmax=1.0)
        ax.set_xticks(range(len(LABEL_KEYS)))
        ax.set_yticks(range(len(LABEL_KEYS)))
        ax.set_xticklabels([LABEL_TEX[j] for j in LABEL_KEYS], rotation=40,
                           ha="right", fontsize=6.5)
        ax.set_yticklabels([LABEL_TEX[j] for j in LABEL_KEYS] if k == 0 else [],
                           fontsize=6.5)
        ax.set_xlabel("true label $g$", fontsize=8)
        if k == 0:
            ax.set_ylabel("design optimised for $g'$", fontsize=8)
        ax.set_title(infos[fkey]["title"], fontsize=7.5)
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                ax.text(j, i, f"{C[i, j]:.3f}", ha="center", va="center",
                        fontsize=5.8,
                        color="white" if C[i, j] < 0.5 * (1.0 + lo) else "black")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"$I_g(S^\ast_{g'})/I_g(S^\ast_g)$", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    save_figure(fig, "fig_design_crossmatrix")
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> None:
    t_start = time.perf_counter()
    plt = setup_matplotlib()
    rng = np.random.default_rng(SEED)

    # the truncated sigmoid spectrum must be numerically indistinguishable
    full = sigmoid_label(2.0, 0.0, kmax=200)
    trunc = LABEL_OBJ["sigmoid"]
    r = np.linspace(-0.999, 0.999, 401)
    sig_err = float(np.max(np.abs(full.C(r) - trunc.C(r))))
    assert sig_err < 1e-9, sig_err

    fams = build_families()
    fam_by_key = {f.key: f for f in fams}
    rows: list[dict] = []
    infos: dict[str, dict] = {}
    for fam in fams:
        with Timer(f"S5 {fam.key}"):
            r_, info = run_family(fam, rng)
            rows.extend(r_)
            infos[fam.key] = info
            print(f"  |V| = {info['n_candidates']}, budget = {info['budget']}, "
                  f"subsets enumerated = {info['n_subsets_enumerated']}, "
                  f"evaluated = {info['n_subsets_evaluated']}, "
                  f"distinct optima = {info['n_distinct_optima']}/{len(LABELS)}")
            for k in LABEL_KEYS:
                print(f"    S*_{k:<16s} t = {info['optimal_times'][k]}"
                      + (f"  w = {info['optimal_widths'][k]}" if not fam.homogeneous else ""))

    # -- submodularity on families (ii) and (iii) -----------------------
    sub_rows: list[dict] = []
    for key in ("ii_nonstationary", "iii_heterogeneous"):
        with Timer(f"S5 submodularity {key}"):
            sub_rows.extend(submodularity_report(fam_by_key[key], rng))

    # -- headline numbers -----------------------------------------------
    def eff(family: str, label: str, method: str) -> float:
        for r_ in rows:
            if r_["family"] == family and r_["label"] == label and r_["method"] == method:
                return float(r_["relative_efficiency"])
        raise KeyError((family, label, method))

    references = ["mutual_information", "imse", "linear_target", "kernel_quadrature"]
    headline: dict = {}
    for fkey in infos:
        worst = min(eff(fkey, l, m) for l in LABEL_KEYS for m in references)
        mean_eff = float(np.mean([eff(fkey, l, m) for l in LABEL_KEYS for m in references]))
        headline[f"{fkey}_label_agnostic_worst_rel_eff"] = worst
        headline[f"{fkey}_label_agnostic_mean_rel_eff"] = mean_eff
        headline[f"{fkey}_label_aware_min_rel_eff"] = min(
            eff(fkey, l, "label_aware") for l in LABEL_KEYS)
        headline[f"{fkey}_label_aware_greedy_min_rel_eff"] = min(
            eff(fkey, l, "label_aware_greedy") for l in LABEL_KEYS)
        headline[f"{fkey}_n_distinct_optima"] = infos[fkey]["n_distinct_optima"]
        headline[f"{fkey}_cross_efficiency_min"] = infos[fkey]["cross_efficiency_min"]
        headline[f"{fkey}_n_subsets_enumerated"] = infos[fkey]["n_subsets_enumerated"]

    ev_la = {f: float(np.mean([r_["n_evaluations"] for r_ in rows
                               if r_["family"] == f and r_["method"] == "label_aware"]))
             for f in infos}
    ev_ex = {f: float(np.mean([r_["n_evaluations"] for r_ in rows
                               if r_["family"] == f and r_["method"] == "exhaustive"]))
             for f in infos}
    headline["label_aware_evaluations_mean"] = ev_la
    headline["exhaustive_evaluations_mean"] = ev_ex
    headline["evaluation_fraction"] = {f: ev_la[f] / ev_ex[f] for f in infos}
    headline["label_aware_attains_optimum_everywhere"] = bool(
        all(eff(f, l, "label_aware") > 1.0 - 1e-9 for f in infos for l in LABEL_KEYS))
    headline["label_aware_greedy_attains_optimum_everywhere"] = bool(
        all(eff(f, l, "label_aware_greedy") > 1.0 - 1e-9 for f in infos for l in LABEL_KEYS))

    pairs = [(f, l) for f in infos for l in LABEL_KEYS]
    exact_la = [p for p in pairs if eff(*p, "label_aware") > 1.0 - 1e-9]
    exact_gr = [p for p in pairs if eff(*p, "label_aware_greedy") > 1.0 - 1e-9]
    headline["n_instances"] = len(pairs)
    headline["label_aware_exact_count"] = len(exact_la)
    headline["label_aware_greedy_exact_count"] = len(exact_gr)
    headline["label_aware_min_rel_eff_overall"] = min(eff(*p, "label_aware") for p in pairs)
    headline["label_aware_greedy_min_rel_eff_overall"] = min(
        eff(*p, "label_aware_greedy") for p in pairs)
    headline["label_aware_shortfalls"] = [
        {"family": f, "label": l, "relative_efficiency": eff(f, l, "label_aware"),
         "is_1swap_local_optimum": infos[f]["label_aware_is_1swap_local_optimum"][l]}
        for f, l in pairs if eff(f, l, "label_aware") <= 1.0 - 1e-9]
    headline["all_label_aware_outputs_are_1swap_local_optima"] = bool(
        all(infos[f]["label_aware_is_1swap_local_optimum"][l] for f, l in pairs))
    headline["swap_search_improvement_max"] = max(
        eff(*p, "label_aware") - eff(*p, "label_aware_greedy") for p in pairs)

    # the label-dependence claim, spelled out
    headline["family_ii_optimal_times"] = infos["ii_nonstationary"]["optimal_times"]
    headline["family_iii_optimal_times"] = infos["iii_heterogeneous"]["optimal_times"]
    headline["family_iii_optimal_widths"] = infos["iii_heterogeneous"]["optimal_widths"]
    headline["families_with_mean_occ_twosided_all_distinct"] = [
        f for f in infos
        if len({tuple(infos[f]["optimal_times"][l]) + tuple(infos[f]["optimal_widths"][l])
                for l in ("mean", "occupation_c0", "occupation_c1.5", "two_sided_c1.2")}) == 4]

    # uniform (endpoint linspace) versus dispersed (bin midpoints)
    headline["uniform_mean_rel_eff"] = float(np.mean([eff(*p, "uniform") for p in pairs]))
    headline["dispersed_mean_rel_eff"] = float(np.mean([eff(*p, "dispersed") for p in pairs]))
    headline["same_time_mean_rel_eff"] = float(np.mean([eff(*p, "same_time") for p in pairs]))
    headline["unspent_budget_label_agnostic"] = {
        m: sorted({round(float(r_["cost"]), 3) for r_ in rows
                   if r_["family"] == "iii_heterogeneous" and r_["method"] == m})
        for m in references}

    # Directly comparable summaries: every method is aggregated over the same
    # five kernel--action configurations x five targets (the stationary family
    # contributes the OU and Matern-3/2 configurations), never over a
    # method-target cross-product of a different size.
    summary_methods = ["mutual_information", "imse", "linear_target",
                       "kernel_quadrature", "label_aware_greedy", "label_aware"]
    method_summary = []
    for method in summary_methods:
        vals = np.array([float(r_["relative_efficiency"]) for r_ in rows
                         if r_["method"] == method])
        method_summary.append({
            "method": method, "n_instances": int(vals.size),
            "minimum": float(vals.min()), "mean": float(vals.mean()),
            "median": float(np.median(vals)),
        })
    headline["method_summary"] = method_summary

    payload = {
        "seed": SEED,
        "sigmoid_kmax": SIGMOID_KMAX,
        "sigmoid_truncation_error": sig_err,
        "n_random_restarts": N_RANDOM_RESTARTS,
        "n_violation_trials": N_VIOLATION_TRIALS,
        "n_ratio_subsets": N_RATIO_SUBSETS,
        "grid": {"horizon": HORIZON, "p": P_GRID},
        "labels": LABEL_KEYS,
        "methods": METHODS,
        "families": infos,
        "submodularity": sub_rows,
        "headline": headline,
        "environment": environment_record(),
        "runtime_seconds": time.perf_counter() - t_start,
    }

    save_csv(rows, "s5_design")
    save_csv(method_summary, "s5_design_method_summary")
    save_csv(sub_rows, "s5_design_submodularity")
    save_json(payload, "s5_design")

    figure_design(plt, fam_by_key, rows, infos)
    figure_crossmatrix(plt, infos, ["ii_nonstationary", "iii_heterogeneous"])

    print(f"\n[S5] total runtime {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
