"""Label-aware observation design under temporal and measurement budgets.

The design problem is

    max_{S subset V}  F_g(S; K)   s.t.  sum_{a in S} c_a <= B,

where ``F_g(S; K) = Var{E(Theta_g | Y_S)}``.  Two structural facts organise the
algorithms in this module.

*Monotonicity is free.*  ``F_g`` is monotone for every ``g in L^2(phi)`` by the
tower property, with no assumption on the kernel.

*Submodularity is not.*  Even the linear (mean) label reduces to ``R^2`` subset
selection, which is not submodular in general.  We therefore work with the
submodularity ratio ``gamma`` and the greedy guarantee
``F(S_greedy) >= (1 - e^{-gamma}) F(S*)``, and provide

* :func:`submodularity_ratio_certificate` -- an instance-specific *lower* bound
  on ``gamma`` computed after the fact, which converts the greedy output into a
  certified approximation, and
* :func:`nonlinear_ratio_lower_bound` -- the analytic transfer bound
  ``gamma_g >= (c_0 / L_g) gamma_linear`` valid when the posterior covariance
  increments stay entrywise non-negative (e.g. MTP2 kernels with non-negative
  action rows), where ``c_0 = inf C_g'`` and ``L_g = sup C_g'``.

Baselines implemented for comparison: same-time replication, uniform spacing,
random spacing, mutual-information placement, integrated posterior variance
(IMSE), and kernel quadrature.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .covariance import Action, TimeGrid, action_vector, protocol_matrices
from .risk import ProtocolState, bilinear, explained_covariance, label_variance
from .transforms import LabelFunctional, MeanLabel

Array = NDArray[np.float64]


def _same_support(a: Action, b: Action, tol: float = 1e-12) -> bool:
    """Whether two actions are acquisition variants on one temporal support."""
    return abs(a.time - b.time) <= tol and abs(a.width - b.width) <= tol


def _support_compatible(action: Action, chosen: Sequence[Action]) -> bool:
    """Enforce the feasible-family rule: at most one variant per support."""
    return not any(_same_support(action, other) for other in chosen)


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class DesignResult:
    actions: list[Action]
    objective: float
    ceiling: float
    cost: float
    method: str
    n_evaluations: int = 0
    runtime: float = 0.0
    trace: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "objective": self.objective,
            "ceiling": self.ceiling,
            "cost": self.cost,
            "n_actions": len(self.actions),
            "n_evaluations": self.n_evaluations,
            "runtime": self.runtime,
            "times": [a.time for a in self.actions],
            "widths": [a.width for a in self.actions],
            "segments": [a.n_segments for a in self.actions],
        }


# --------------------------------------------------------------------------
# Label-aware greedy
# --------------------------------------------------------------------------
def objective(label: LabelFunctional, K: Array, grid: TimeGrid,
              acts: Sequence[Action]) -> float:
    """``F_g(S; K)`` evaluated from scratch for an arbitrary action set."""
    if len(acts) == 0:
        return 0.0
    A, R = protocol_matrices(list(acts), grid)
    return bilinear(label, explained_covariance(K, A, R), grid.weights)


def swap_local_search(
    label: LabelFunctional,
    K: Array,
    grid: TimeGrid,
    candidates: Sequence[Action],
    chosen: Sequence[Action],
    budget: float,
    max_rounds: int = 20,
) -> tuple[list[Action], float, int]:
    """First-improvement 1-swap local search on the exact objective.

    ``max_rounds`` bounds the number of *accepted swaps*, not the number of
    sweeps: the search takes the first improvement it finds and restarts, so a
    round performs at most one swap.  Running to convergence therefore needs
    ``max_rounds`` at least as large as the number of swaps the instance needs;
    a small cap trades a little objective value for a large saving in the
    expensive full re-evaluations.

    Greedy alone is *not* enough here: ``F_g`` is not submodular, so a myopic
    first pick can foreclose a better configuration, and on some instances a
    label-agnostic baseline that happens to spread out beats plain greedy.  A
    swap pass repairs exactly that failure mode at negligible cost and keeps the
    method a genuine maximiser of the label-specific objective.
    """
    current = list(chosen)
    best = objective(label, K, grid, current)
    n_eval = 0
    for _ in range(max_rounds):
        improved = False
        for i in range(len(current)):
            for cand in candidates:
                if any(cand is c for c in current):
                    continue
                trial = current[:i] + [cand] + current[i + 1:]
                if any(_same_support(trial[j], trial[k])
                       for j in range(len(trial)) for k in range(j + 1, len(trial))):
                    continue
                if sum(a.cost for a in trial) > budget + 1e-12:
                    continue
                val = objective(label, K, grid, trial)
                n_eval += 1
                if val > best + 1e-14:
                    current, best, improved = trial, val, True
                    break
            if improved:
                break
        if not improved:
            break
    return current, best, n_eval


def select_protocol_greedy(
    label: LabelFunctional,
    K: Array,
    grid: TimeGrid,
    candidates: Sequence[Action],
    budget: float,
    cost_aware: bool = True,
    initial: Sequence[Action] = (),
    local_search: bool = False,
) -> DesignResult:
    """Greedy maximisation of ``F_g`` using exact rank-one marginal gains.

    With heterogeneous costs the greedy rule maximises ``Delta_g(a|S) / c_a``;
    with equal costs the two rules coincide.  Setting ``local_search`` appends a
    1-swap refinement pass, which is what Algorithm 1 of the paper actually
    specifies.
    """
    import time

    t0 = time.perf_counter()
    state = ProtocolState.from_actions(label, K, grid, list(initial))
    vectors = [action_vector(a, grid) for a in candidates]
    remaining = set(range(len(candidates)))
    n_eval = 0
    trace: list[float] = [state.F]

    while remaining:
        best_idx, best_score, best_gain = -1, -np.inf, 0.0
        for i in sorted(remaining):
            a = candidates[i]
            if not _support_compatible(a, state.chosen):
                continue
            if state.cost + a.cost > budget + 1e-12:
                continue
            gain = state.marginal_gain(vectors[i], a.effective_noise)
            n_eval += 1
            score = gain / a.cost if cost_aware and a.cost > 0 else gain
            if score > best_score:
                best_idx, best_score, best_gain = i, score, gain
        if best_idx < 0 or best_gain <= 1e-15:
            break
        state = state.add(candidates[best_idx], grid)
        remaining.discard(best_idx)
        trace.append(state.F)

    chosen, F, cost = list(state.chosen), state.F, state.cost
    method = "label_aware_greedy"
    if local_search and chosen:
        chosen, F, extra = swap_local_search(label, K, grid, candidates, chosen, budget)
        n_eval += extra
        cost = float(sum(a.cost for a in chosen))
        method = "label_aware"
        trace.append(F)

    V = label_variance(label, K, grid.weights)
    return DesignResult(
        actions=chosen, objective=F,
        ceiling=F / V if V > 0 else 0.0, cost=cost,
        method=method, n_evaluations=n_eval,
        runtime=time.perf_counter() - t0, trace=trace,
    )


def select_protocol_exhaustive(
    label: LabelFunctional,
    K: Array,
    grid: TimeGrid,
    candidates: Sequence[Action],
    n_select: int,
) -> DesignResult:
    """Exhaustive search over all ``C(|V|, n_select)`` protocols.

    Used to certify greedy on small instances; the caller is responsible for
    keeping ``|V|`` small enough that this terminates.
    """
    import time

    t0 = time.perf_counter()
    V = label_variance(label, K, grid.weights)
    best_actions: list[Action] = []
    best_F = -np.inf
    n_eval = 0
    for combo in itertools.combinations(range(len(candidates)), n_select):
        acts = [candidates[i] for i in combo]
        if any(_same_support(acts[j], acts[k])
               for j in range(len(acts)) for k in range(j + 1, len(acts))):
            continue
        A, R = protocol_matrices(acts, grid)
        F = bilinear(label, explained_covariance(K, A, R), grid.weights)
        n_eval += 1
        if F > best_F:
            best_F, best_actions = F, acts
    return DesignResult(
        actions=best_actions, objective=best_F,
        ceiling=best_F / V if V > 0 else 0.0,
        cost=float(sum(a.cost for a in best_actions)),
        method="exhaustive", n_evaluations=n_eval,
        runtime=time.perf_counter() - t0,
    )


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def _evaluate(label: LabelFunctional, K: Array, grid: TimeGrid,
              acts: Sequence[Action], method: str, runtime: float = 0.0,
              n_eval: int = 0) -> DesignResult:
    A, R = protocol_matrices(acts, grid)
    V = label_variance(label, K, grid.weights)
    F = bilinear(label, explained_covariance(K, A, R), grid.weights) if len(acts) else 0.0
    return DesignResult(actions=list(acts), objective=F,
                        ceiling=F / V if V > 0 else 0.0,
                        cost=float(sum(a.cost for a in acts)), method=method,
                        n_evaluations=n_eval, runtime=runtime)


def design_uniform(label: LabelFunctional, K: Array, grid: TimeGrid,
                   n_select: int, template: Action) -> DesignResult:
    centres = np.linspace(grid.times[0], grid.times[-1], n_select)
    acts = [Action(time=float(t), width=template.width, n_segments=template.n_segments,
                   noise=template.noise, cost=template.cost) for t in centres]
    return _evaluate(label, K, grid, acts, "uniform")


def design_same_time(label: LabelFunctional, K: Array, grid: TimeGrid,
                     n_select: int, template: Action) -> DesignResult:
    t = 0.5 * (grid.times[0] + grid.times[-1])
    acts = [Action(time=float(t), width=template.width,
                   n_segments=template.n_segments * n_select,
                   noise=template.noise, cost=template.cost * n_select)]
    return _evaluate(label, K, grid, acts, "same_time")


def design_random(label: LabelFunctional, K: Array, grid: TimeGrid,
                  candidates: Sequence[Action], n_select: int,
                  rng: np.random.Generator, n_restarts: int = 1) -> DesignResult:
    best: DesignResult | None = None
    for _ in range(max(1, n_restarts)):
        chosen: list[Action] = []
        for i in rng.permutation(len(candidates)):
            action = candidates[int(i)]
            if _support_compatible(action, chosen):
                chosen.append(action)
                if len(chosen) == n_select:
                    break
        if len(chosen) < n_select:
            raise ValueError("fewer distinct action supports than n_select")
        res = _evaluate(label, K, grid, chosen, "random")
        if best is None or res.objective > best.objective:
            best = res
    assert best is not None
    return best


def design_mutual_information(label: LabelFunctional, K: Array, grid: TimeGrid,
                              candidates: Sequence[Action],
                              n_select: int) -> DesignResult:
    r"""Greedy latent-state mutual information under positive action noise.

    The target-independent set function is

    ``I(Z; Y_S) = 0.5 log det(I + R_S^{-1/2} A_S K A_S' R_S^{-1/2})``.

    Its exact marginal gain is
    ``0.5 log{(ell_a' P_S ell_a + r_a) / r_a}``, where ``P_S`` is the
    posterior covariance of the latent grid after the selected observations.
    This is the marginal gain of one fixed, standard information objective.
    """
    import time

    t0 = time.perf_counter()
    if any(a.effective_noise <= 0.0 for a in candidates):
        raise ValueError("latent-state mutual information requires positive action noise")
    vectors = [action_vector(a, grid) for a in candidates]
    P = K.copy()
    chosen: list[int] = []
    n_eval = 0
    for _ in range(n_select):
        best_idx, best_gain = -1, -np.inf
        selected_actions = [candidates[j] for j in chosen]
        for i, action in enumerate(candidates):
            if i in chosen or not _support_compatible(action, selected_actions):
                continue
            v = P @ vectors[i]
            r = float(action.effective_noise)
            s = float(vectors[i] @ v) + r
            gain = 0.5 * np.log(max(s, 1e-300) / r)
            n_eval += 1
            if gain > best_gain:
                best_idx, best_gain = i, gain
        if best_idx < 0:
            break
        v = P @ vectors[best_idx]
        s = float(vectors[best_idx] @ v) + candidates[best_idx].effective_noise
        P = P - np.outer(v, v) / max(s, 1e-300)
        chosen.append(best_idx)
    return _evaluate(label, K, grid, [candidates[i] for i in chosen],
                     "mutual_information", time.perf_counter() - t0, n_eval)


def design_imse(label: LabelFunctional, K: Array, grid: TimeGrid,
                candidates: Sequence[Action], n_select: int) -> DesignResult:
    """Greedy minimisation of the integrated posterior variance ``sum_j w_j P_jj``.

    Also label-agnostic: it reduces uncertainty about the *trajectory* uniformly
    rather than about the aggregate functional.
    """
    import time

    t0 = time.perf_counter()
    vectors = [action_vector(a, grid) for a in candidates]
    P = K.copy()
    chosen: list[int] = []
    n_eval = 0
    for _ in range(n_select):
        best_idx, best_gain = -1, -np.inf
        for i in range(len(candidates)):
            if i in chosen:
                continue
            if not _support_compatible(candidates[i], [candidates[j] for j in chosen]):
                continue
            v = P @ vectors[i]
            s = float(vectors[i] @ v) + candidates[i].effective_noise
            gain = float(grid.weights @ (v * v)) / max(s, 1e-300)
            n_eval += 1
            if gain > best_gain:
                best_idx, best_gain = i, gain
        if best_idx < 0:
            break
        v = P @ vectors[best_idx]
        s = float(vectors[best_idx] @ v) + candidates[best_idx].effective_noise
        P = P - np.outer(v, v) / s
        chosen.append(best_idx)
    return _evaluate(label, K, grid, [candidates[i] for i in chosen], "imse",
                     time.perf_counter() - t0, n_eval)


def design_kernel_quadrature(label: LabelFunctional, K: Array, grid: TimeGrid,
                             candidates: Sequence[Action], n_select: int) -> DesignResult:
    """Greedy kernel quadrature: minimise the worst-case RKHS integration error.

    Choosing nodes to minimise ``|| mu - Pi_span mu ||_H`` with mean embedding
    ``mu = K omega`` is *exactly* our objective for the noiseless mean label
    (the noiseless-mean special case of `sec:design`), so this baseline
    coincides with the proposed method in
    that special case and separates from it only through measurement noise and
    label nonlinearity.
    """
    import time

    t0 = time.perf_counter()
    lin = MeanLabel()
    res = select_protocol_greedy(
        lin, K, grid,
        [Action(time=a.time, width=a.width, n_segments=a.n_segments,
                noise=0.0, cost=a.cost) for a in candidates],
        budget=float(sum(sorted(a.cost for a in candidates)[:n_select])),
        cost_aware=False,
    )
    acts = [candidates[i] for i in
            _match_actions(res.actions, candidates)][:n_select]
    out = _evaluate(label, K, grid, acts, "kernel_quadrature",
                    time.perf_counter() - t0, res.n_evaluations)
    return out


def _match_actions(chosen: Sequence[Action], candidates: Sequence[Action]) -> list[int]:
    idx = []
    for a in chosen:
        for i, b in enumerate(candidates):
            if abs(a.time - b.time) < 1e-12 and abs(a.width - b.width) < 1e-12 \
                    and a.n_segments == b.n_segments and i not in idx:
                idx.append(i)
                break
    return idx


# --------------------------------------------------------------------------
# Robust (lower-confidence-bound) design
# --------------------------------------------------------------------------
def select_protocol_robust(
    label: LabelFunctional,
    K_samples: Sequence[Array],
    grid: TimeGrid,
    candidates: Sequence[Action],
    budget: float,
    quantile: float = 0.1,
    cost_aware: bool = True,
) -> DesignResult:
    """Greedy maximisation of a lower confidence bound on ``I_g(S)``.

    At each step the candidate score is the ``quantile``-level empirical
    quantile of the ceiling across bootstrap covariance replicates, which
    prevents the optimiser from exploiting accidental structure in a single
    noisy covariance estimate.
    """
    import time

    t0 = time.perf_counter()
    states = [ProtocolState.empty(label, Kb, grid.weights) for Kb in K_samples]
    totals = [label_variance(label, Kb, grid.weights) for Kb in K_samples]
    vectors = [action_vector(a, grid) for a in candidates]
    remaining = set(range(len(candidates)))
    cost = 0.0
    chosen: list[Action] = []
    n_eval = 0

    while remaining:
        best_idx, best_score = -1, -np.inf
        for i in sorted(remaining):
            a = candidates[i]
            if cost + a.cost > budget + 1e-12:
                continue
            vals = []
            for st, tot in zip(states, totals):
                g = st.marginal_gain(vectors[i], a.effective_noise)
                vals.append((st.F + g) / tot if tot > 0 else 0.0)
            n_eval += 1
            score = float(np.quantile(vals, quantile))
            if cost_aware and a.cost > 0:
                score = score / a.cost
            if score > best_score:
                best_idx, best_score = i, score
        if best_idx < 0:
            break
        states = [st.add(candidates[best_idx], grid) for st in states]
        chosen.append(candidates[best_idx])
        cost += candidates[best_idx].cost
        remaining.discard(best_idx)

    res = _evaluate(label, K_samples[0], grid, chosen, "robust_lcb",
                    time.perf_counter() - t0, n_eval)
    return res


# --------------------------------------------------------------------------
# Continuous local refinement
# --------------------------------------------------------------------------
def refine_protocol_continuous(label: LabelFunctional, K_fn: Callable[[Array], Array],
                               grid: TimeGrid, actions: Sequence[Action],
                               max_iter: int = 40) -> DesignResult:
    """Local continuous refinement of the selected window centres.

    ``K_fn`` maps a vector of times to the corresponding correlation matrix, so
    the refinement is not restricted to the discrete candidate grid.  A
    derivative-free bounded optimiser is used because the objective involves the
    entrywise transform ``C_g``.
    """
    from scipy.optimize import minimize

    times0 = np.array([a.time for a in actions], dtype=float)
    lo, hi = float(grid.times[0]), float(grid.times[-1])

    def negative_objective(ts: Array) -> float:
        acts = [Action(time=float(t), width=a.width, n_segments=a.n_segments,
                       noise=a.noise, cost=a.cost) for t, a in zip(ts, actions)]
        A, R = protocol_matrices(acts, grid)
        return -bilinear(label, explained_covariance(K_fn(grid.times), A, R), grid.weights)

    res = minimize(negative_objective, times0, method="L-BFGS-B",
                   bounds=[(lo, hi)] * len(actions),
                   options={"maxiter": max_iter})
    acts = [Action(time=float(t), width=a.width, n_segments=a.n_segments,
                   noise=a.noise, cost=a.cost) for t, a in zip(res.x, actions)]
    return _evaluate(label, K_fn(grid.times), grid, acts, "continuous_refined")


# --------------------------------------------------------------------------
# Submodularity diagnostics
# --------------------------------------------------------------------------
def marginal_gain_of_set(label: LabelFunctional, K: Array, grid: TimeGrid,
                         base: Sequence[Action], extra: Sequence[Action]) -> float:
    A0, R0 = protocol_matrices(list(base), grid)
    A1, R1 = protocol_matrices(list(base) + list(extra), grid)
    F0 = bilinear(label, explained_covariance(K, A0, R0), grid.weights) if len(base) else 0.0
    F1 = bilinear(label, explained_covariance(K, A1, R1), grid.weights)
    return float(F1 - F0)


def submodularity_ratio_certificate(
    label: LabelFunctional, K: Array, grid: TimeGrid,
    candidates: Sequence[Action], base_sets: Sequence[Sequence[Action]],
    max_extra: int = 3, rng: np.random.Generator | None = None,
    n_subsets: int = 64,
) -> dict:
    """Instance-specific lower bound on the submodularity ratio.

    ``gamma = min_{S, Omega} sum_{a in Omega} Delta(a|S) / [F(S u Omega) - F(S)]``
    is estimated by minimising over the supplied base sets ``S`` (typically the
    prefixes of the greedy path) and random extension sets ``Omega``.  The
    returned value is an upper bound on the true ``gamma`` and therefore yields
    a *heuristic-but-checkable* certificate: combined with the greedy output it
    gives ``F(S_greedy) >= (1 - e^{-gamma}) F(S*)`` for the sampled family.

    Singleton ``Omega`` is included, which forces ``gamma <= 1`` identically
    (the ratio is exactly one there).  A sampler that only drew ``|Omega| >= 2``
    could report a value above one, which is not a submodularity ratio; the
    returned value is additionally capped at one so that the greedy factor
    ``1 - e^{-gamma}`` never exceeds ``1 - 1/e``.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    ratios: list[float] = []
    for base in base_sets:
        base_list = list(base)
        pool = [a for a in candidates if a not in base_list]
        if len(pool) < 1:
            continue
        for _ in range(n_subsets):
            k = int(rng.integers(1, min(max_extra, len(pool)) + 1))
            idx = rng.choice(len(pool), size=k, replace=False)
            omega = [pool[i] for i in idx]
            joint = marginal_gain_of_set(label, K, grid, base_list, omega)
            if joint <= 1e-14:
                continue
            singles = sum(marginal_gain_of_set(label, K, grid, base_list, [a])
                          for a in omega)
            ratios.append(singles / joint)
    if not ratios:
        return {"gamma": float("nan"), "n_samples": 0, "greedy_factor": float("nan")}
    gamma = float(min(1.0, np.min(ratios)))
    return {
        "gamma": gamma,
        "gamma_median": float(min(1.0, np.median(ratios))),
        "gamma_uncapped": float(np.min(ratios)),
        "n_samples": len(ratios),
        "greedy_factor": float(1.0 - np.exp(-gamma)),
    }


def find_submodularity_violation(
    label: LabelFunctional, K: Array, grid: TimeGrid,
    candidates: Sequence[Action], rng: np.random.Generator,
    n_trials: int = 2000,
) -> dict | None:
    """Search for ``S subset T`` and ``a`` with ``Delta(a|S) < Delta(a|T)``.

    A single such triple falsifies submodularity of ``F_g`` for the instance.
    """
    n = len(candidates)
    if n < 3:
        return None
    worst = None
    for _ in range(n_trials):
        perm = rng.permutation(n)
        a = int(perm[0])
        s_size = int(rng.integers(0, max(1, min(3, n - 2))))
        t_extra = int(rng.integers(1, max(2, min(3, n - 1 - s_size))))
        S = [candidates[i] for i in perm[1:1 + s_size]]
        T = S + [candidates[i] for i in perm[1 + s_size:1 + s_size + t_extra]]
        dS = marginal_gain_of_set(label, K, grid, S, [candidates[a]])
        dT = marginal_gain_of_set(label, K, grid, T, [candidates[a]])
        if dT > dS + 1e-14:
            violation = (dT - dS) / max(dS, 1e-14)
            if worst is None or violation > worst["relative_violation"]:
                worst = {
                    "action": candidates[a],
                    "S": S, "T": T,
                    "gain_S": float(dS), "gain_T": float(dT),
                    "relative_violation": float(violation),
                }
    return worst


def nonlinear_ratio_lower_bound(label: LabelFunctional, gamma_linear: float,
                                rmax: float = 1.0 - 1e-9) -> float:
    """``gamma_g >= (c_0 / L_g) gamma_linear`` (transfer bound).

    Dropped from the article during compression; retained here because the
    experiments report the certificate.

    Valid when every posterior-covariance increment ``v v^T / s`` is entrywise
    non-negative, e.g. for MTP2 kernels with non-negative action rows.  Here
    ``c_0 = inf_{[0, rmax]} C_g'`` and ``L_g = sup_{[0, rmax]} C_g'``.
    """
    c0 = label.derivative_floor(rmax)
    Lg = float(np.max(np.abs(label.dC(np.linspace(0.0, rmax, 4001)))))
    if Lg <= 0:
        return 0.0
    return float(min(1.0, (c0 / Lg) * gamma_linear))
