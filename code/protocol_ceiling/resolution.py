"""Resolution-adaptive protocol selection.

The design experiments of this package produce a result that looks like a
failure and is in fact a prediction: with a small calibration sample, choosing
the *coarse geometry* of a protocol (contiguous versus dispersed) is reliable,
while choosing *exact placements* is not.  This module makes that a theorem
rather than an anecdote.

Order the feasible protocols into nested classes

    Pi^(1) subset Pi^(2) subset ... subset Pi^(L),

from coarse to fine -- for example: (1) contiguous versus dispersed, (2) the
number of windows and the broad phase they occupy, (3) coarse temporal bins,
(4) exact grid locations.  A larger class has smaller approximation error but a
larger uniform estimation error ``eps_l``.  Selecting

    (l_hat, S_hat) = argmax_{l, S in Pi^(l)} { I_hat(S) - eps_l }

balances the two, and on the event that every ``eps_l`` is a valid uniform
bound,

    I(S*) - I(S_hat) <= min_l [ I(S*) - max_{S in Pi^(l)} I(S) + 2 eps_l ],

which is `cor:regret` in the paper.  The first term is the price of a class too
coarse to contain a good protocol; the second is the price of a class too rich
to be estimated.  Neither is avoidable, and the selector pays only the better of
the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .covariance import Action, TimeGrid, bin_midpoints
from .estimation import estimate_protocol_ceiling
from .transforms import LabelFunctional

Array = NDArray[np.float64]


# --------------------------------------------------------------------------
# Nested protocol classes
# --------------------------------------------------------------------------
@dataclass
class ProtocolClass:
    """One level of the nested hierarchy."""

    name: str
    level: int
    protocols: list[list[Action]]

    def __len__(self) -> int:
        return len(self.protocols)


def nested_classes(grid: TimeGrid, budget: int, noise: float,
                   width: float = 0.0, n_fine: int = 24,
                   n_coarse_bins: int = 4) -> list[ProtocolClass]:
    """Build a coarse-to-fine hierarchy of protocols at a fixed budget.

    Level 1  geometry only: one contiguous block against uniform dispersion.
    Level 2  phase: the dispersed pattern restricted to an early, middle, late
             or full-horizon phase.
    Level 3  coarse bins: one window per chosen coarse bin.  This is a genuine
             refinement of level 2 only when ``n_coarse_bins > budget``; with
             ``n_coarse_bins == budget`` there is a single admissible choice and
             it coincides with the full-horizon dispersed protocol, so levels 2
             and 3 collapse onto each other.  The caller is responsible for
             choosing more bins than windows.
    Level 4  exact placements: any ``budget``-subset of a fine candidate grid.

    Each level is the union of the previous one with its own protocols, with
    exact duplicates removed: the dispersed protocol of level 1 reappears as the
    full-horizon phase of level 2 and (when the bin grid matches) as a coarse-bin
    choice, and counting it twice would inflate the class size and make one
    level look like a refinement of another when it is the same set.
    """
    import itertools

    T = grid.horizon

    def acts(times) -> list[Action]:
        return [Action(time=float(t), width=width, noise=noise, cost=1.0)
                for t in times]

    def _key(protocol: list[Action]) -> tuple:
        return tuple(sorted((round(a.time, 12), a.width, a.noise, a.cost)
                            for a in protocol))

    def dedup(protocols: list[list[Action]]) -> list[list[Action]]:
        seen: set[tuple] = set()
        out: list[list[Action]] = []
        for pr in protocols:
            k = _key(pr)
            if k not in seen:
                seen.add(k)
                out.append(pr)
        return out

    # ---- level 1: contiguous vs dispersed -------------------------------
    dt = T / max(n_fine, 1)
    contiguous = acts([0.5 * T + (i - (budget - 1) / 2) * dt for i in range(budget)])
    dispersed = acts(bin_midpoints(T, budget))
    lvl1 = ProtocolClass("geometry", 1, dedup([contiguous, dispersed]))

    # ---- level 2: phase-restricted dispersion ---------------------------
    phases = {"early": (0.0, 0.4), "middle": (0.3, 0.7),
              "late": (0.6, 1.0), "full": (0.0, 1.0)}
    lvl2 = ProtocolClass("phase", 2, dedup(lvl1.protocols + [
        acts(lo * T + (hi - lo) * T * (np.arange(budget) + 0.5) / budget)
        for lo, hi in phases.values()]))

    # ---- level 3: one window per chosen coarse bin ----------------------
    edges = np.linspace(0.0, T, n_coarse_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    lvl3_protocols: list[list[Action]] = []
    if budget <= n_coarse_bins:
        for combo in itertools.combinations(range(n_coarse_bins), budget):
            lvl3_protocols.append(acts([centres[i] for i in combo]))
    else:  # more windows than bins: spread them evenly inside chosen bins
        reps = int(np.ceil(budget / n_coarse_bins))
        for combo in itertools.combinations(range(n_coarse_bins),
                                            min(n_coarse_bins, budget)):
            times = []
            for i in combo:
                lo, hi = edges[i], edges[i + 1]
                times += list(lo + (hi - lo) * (np.arange(reps) + 0.5) / reps)
            lvl3_protocols.append(acts(times[:budget]))
    lvl3 = ProtocolClass("coarse bins", 3, dedup(lvl2.protocols + lvl3_protocols))

    # ---- level 4: exact placements on a fine grid -----------------------
    fine = bin_midpoints(T, n_fine)
    lvl4_protocols = [acts([fine[i] for i in combo])
                      for combo in itertools.combinations(range(n_fine), budget)]
    lvl4 = ProtocolClass("exact placement", 4, dedup(lvl3.protocols + lvl4_protocols))

    return [lvl1, lvl2, lvl3, lvl4]


# --------------------------------------------------------------------------
# The selector
# --------------------------------------------------------------------------
@dataclass
class ResolutionSelection:
    level: int
    level_name: str
    actions: list[Action]
    penalised_score: float
    estimated_value: float
    true_value: float
    regret: float
    eps: dict[int, float] = field(default_factory=dict)
    per_level: dict[int, dict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "selected_level": self.level, "selected_level_name": self.level_name,
            "penalised_score": self.penalised_score,
            "estimated_value": self.estimated_value,
            "true_value": self.true_value, "regret": self.regret,
            "eps": {str(k): v for k, v in self.eps.items()},
            "times": [a.time for a in self.actions],
        }


def uniform_error(label: LabelFunctional, K_hat: Array, K_true: Array,
                  grid: TimeGrid, protocols: Sequence[Sequence[Action]]) -> float:
    """``sup_{S in Pi} |I_hat(S) - I(S)|`` -- the oracle version, for analysis."""
    return float(max(
        abs(estimate_protocol_ceiling(label, K_hat, grid, S)
            - estimate_protocol_ceiling(label, K_true, grid, S))
        for S in protocols))


def bootstrap_uniform_error(label: LabelFunctional, K_samples: Sequence[Array],
                            K_hat: Array, grid: TimeGrid,
                            protocols: Sequence[Sequence[Action]],
                            quantile: float = 0.9) -> float:
    """Practical ``eps`` from bootstrap covariance replicates.

    For each replicate we take the sup over the class of the deviation from the
    point estimate, then read off an upper quantile.  This is the quantity a
    practitioner can actually compute; the oracle version above is used only to
    check the theorem.
    """
    base = np.array([estimate_protocol_ceiling(label, K_hat, grid, S)
                     for S in protocols])
    sups = []
    for Kb in K_samples:
        vals = np.array([estimate_protocol_ceiling(label, Kb, grid, S)
                         for S in protocols])
        sups.append(float(np.max(np.abs(vals - base))))
    return float(np.quantile(sups, quantile))


def resolution_adaptive_select(
    label: LabelFunctional,
    K_hat: Array,
    grid: TimeGrid,
    classes: Sequence[ProtocolClass],
    eps: dict[int, float],
    K_true: Array | None = None,
) -> ResolutionSelection:
    """``argmax_{l, S in Pi^(l)} { I_hat(S) - eps_l }``.

    Returns the selection together with the true value and regret when the truth
    is supplied (for analysis); in deployment ``K_true`` is unavailable and only
    the penalised score and estimate are meaningful.
    """
    best = None
    per_level: dict[int, dict] = {}
    for cls in classes:
        vals = np.array([estimate_protocol_ceiling(label, K_hat, grid, S)
                         for S in cls.protocols])
        j = int(np.argmax(vals))
        score = float(vals[j] - eps[cls.level])
        entry = {"n_protocols": len(cls), "best_estimated": float(vals[j]),
                 "penalised": score, "eps": eps[cls.level]}
        if K_true is not None:
            true_vals = np.array([estimate_protocol_ceiling(label, K_true, grid, S)
                                  for S in cls.protocols])
            entry["best_true_in_class"] = float(true_vals.max())
            entry["true_value_of_estimated_argmax"] = float(true_vals[j])
        per_level[cls.level] = entry
        if best is None or score > best[0]:
            best = (score, cls, j, vals[j])

    assert best is not None
    score, cls, j, est = best
    actions = list(cls.protocols[j])
    true_value, regret = float("nan"), float("nan")
    if K_true is not None:
        true_value = estimate_protocol_ceiling(label, K_true, grid, actions)
        finest = classes[-1]
        best_overall = max(estimate_protocol_ceiling(label, K_true, grid, S)
                           for S in finest.protocols)
        regret = float(best_overall - true_value)
    return ResolutionSelection(
        level=cls.level, level_name=cls.name, actions=actions,
        penalised_score=score, estimated_value=float(est),
        true_value=true_value, regret=regret,
        eps=dict(eps), per_level=per_level,
    )


def theorem_bound(classes: Sequence[ProtocolClass], eps: dict[int, float],
                  label: LabelFunctional, K_true: Array, grid: TimeGrid) -> dict:
    """Right-hand side of `cor:regret`, level by level.

    ``min_l [ I(S*) - max_{S in Pi^(l)} I(S) + 2 eps_l ]``.
    """
    finest = classes[-1]
    best_overall = max(estimate_protocol_ceiling(label, K_true, grid, S)
                       for S in finest.protocols)
    per_level = {}
    for cls in classes:
        best_in_class = max(estimate_protocol_ceiling(label, K_true, grid, S)
                            for S in cls.protocols)
        per_level[cls.level] = {
            "approximation_error": float(best_overall - best_in_class),
            "estimation_term": float(2.0 * eps[cls.level]),
            "bound": float(best_overall - best_in_class + 2.0 * eps[cls.level]),
        }
    return {"best_overall": float(best_overall), "per_level": per_level,
            "bound": float(min(v["bound"] for v in per_level.values()))}
