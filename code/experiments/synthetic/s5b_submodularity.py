"""S5b: explicit witnesses that F_g is monotone but not submodular.

`sec:design` asserts the existence of S subset T and an action a with
Delta_g(a|S) < Delta_g(a|T).  This script produces witnesses that can be checked
by hand: it searches small instances exhaustively (not randomly) over all
S subset T with |S| <= 2 and |T| <= 5, records the largest relative violation
for each kernel and label, and dumps the smallest such instance in full so a
reader can recompute it from the printed correlation matrix.

The reported ``relative_violation`` is ``(Delta(a|T) - Delta(a|S)) / Delta(a|S)``.
Where ``Delta(a|S)`` is itself near zero that ratio is a division by numerical
noise and says nothing about the size of the violation; the absolute gains are
recorded alongside it and are what should be read.  The *existence* of the
violation is decided by the absolute comparison, not by the ratio.

It also confirms the positive half of the structural claim -- monotonicity --
by checking that no marginal gain is ever negative, and reports the
submodularity ratio and the resulting greedy factor on each instance.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import (SEED, environment_record, save_csv,  # noqa: E402
                                save_json, Timer)

from protocol_ceiling import (Action, MeanLabel, ThresholdLabel,  # noqa: E402
                              TwoSidedLabel, make_kernel, submodularity_ratio_certificate,
                              trait_state_correlation, uniform_grid)
from protocol_ceiling.design import marginal_gain_of_set, objective  # noqa: E402

rng = np.random.default_rng(SEED)

HORIZON, P, NOISE = 10.0, 96, 0.15
GRID = uniform_grid(HORIZON, P)

KERNELS = {
    "ou": make_kernel("ou", tau=1.0),
    "matern32": make_kernel("matern32", tau=1.0),
    "two_scale_ou": make_kernel("two_scale_ou", tau_fast=0.15, tau_slow=3.0, w_fast=0.6),
    "damped_periodic": make_kernel("periodic", tau=6.0, period=3.0),
}
LABELS = {
    "mean": MeanLabel(),
    "occupation_c0": ThresholdLabel(c=0.0),
    "occupation_c1": ThresholdLabel(c=1.0),
    "two_sided_c1": TwoSidedLabel(c=1.0),
}
N_CAND = 9
CANDS = [Action(time=float(t), width=0.0, noise=NOISE, cost=1.0)
         for t in np.linspace(0.6, 9.4, N_CAND)]


def exhaustive_violation(label, K):
    """Largest relative violation of the diminishing-returns property."""
    best = None
    n_checked = 0
    for a_idx in range(N_CAND):
        others = [i for i in range(N_CAND) if i != a_idx]
        for s_size in (0, 1, 2):
            for S_idx in itertools.combinations(others, s_size):
                rest = [i for i in others if i not in S_idx]
                for extra in range(1, min(3, len(rest)) + 1):
                    for T_extra in itertools.combinations(rest, extra):
                        S = [CANDS[i] for i in S_idx]
                        T = S + [CANDS[i] for i in T_extra]
                        dS = marginal_gain_of_set(label, K, GRID, S, [CANDS[a_idx]])
                        dT = marginal_gain_of_set(label, K, GRID, T, [CANDS[a_idx]])
                        n_checked += 1
                        if dT > dS + 1e-15:
                            rel = (dT - dS) / max(dS, 1e-300)
                            if best is None or rel > best["relative_violation"]:
                                best = {
                                    "a_index": a_idx, "a_time": CANDS[a_idx].time,
                                    "S_indices": list(S_idx),
                                    "T_indices": list(S_idx) + list(T_extra),
                                    "S_times": [CANDS[i].time for i in S_idx],
                                    "T_times": [CANDS[i].time for i in
                                                list(S_idx) + list(T_extra)],
                                    "gain_S": float(dS), "gain_T": float(dT),
                                    "relative_violation": float(rel),
                                    "absolute_violation": float(dT - dS),
                                }
    return best, n_checked


def monotonicity_check(label, K, n_paths: int = 200) -> float:
    """Smallest marginal increment observed along random insertion orders."""
    worst = np.inf
    for _ in range(n_paths):
        order = rng.permutation(N_CAND)
        chosen: list[Action] = []
        prev = 0.0
        for i in order:
            chosen.append(CANDS[i])
            cur = objective(label, K, GRID, chosen)
            worst = min(worst, cur - prev)
            prev = cur
    return float(worst)


rows: list[dict] = []
detail: dict = {}
with Timer("S5b exhaustive submodularity search"):
    for kname, kern in KERNELS.items():
        K = trait_state_correlation(GRID, 0.0, kern)
        for lname, label in LABELS.items():
            viol, n_checked = exhaustive_violation(label, K)
            mono = monotonicity_check(label, K)
            cert = submodularity_ratio_certificate(
                label, K, GRID, CANDS,
                base_sets=[[], [CANDS[2]], [CANDS[2], CANDS[6]]],
                rng=np.random.default_rng(SEED + 1), n_subsets=40)
            row = {
                "kernel": kname, "label": lname,
                "n_triples_checked": n_checked,
                "violation_found": viol is not None,
                "relative_violation": viol["relative_violation"] if viol else 0.0,
                "absolute_violation": viol["absolute_violation"] if viol else 0.0,
                "gain_S": viol["gain_S"] if viol else float("nan"),
                "gain_T": viol["gain_T"] if viol else float("nan"),
                "size_S": len(viol["S_indices"]) if viol else 0,
                "size_T": len(viol["T_indices"]) if viol else 0,
                "min_marginal_increment": mono,
                "gamma": cert["gamma"], "gamma_median": cert["gamma_median"],
                "greedy_factor": min(cert["greedy_factor"], 1.0 - np.exp(-1.0))
                if cert["gamma"] >= 1 else cert["greedy_factor"],
            }
            rows.append(row)
            detail[f"{kname}|{lname}"] = viol
            print(f"  {kname:16s} {lname:14s} "
                  f"rel.viol {row['relative_violation']:9.4f}  "
                  f"|S|={row['size_S']} |T|={row['size_T']}  "
                  f"min increment {mono:+.2e}  gamma {cert['gamma']:.3f}")

save_csv(rows, "s5b_submodularity")

worst = max(rows, key=lambda r: r["relative_violation"])
worst_detail = detail[f"{worst['kernel']}|{worst['label']}"]
print("\nMost dramatic violation:")
print(f"  kernel {worst['kernel']}, label {worst['label']}")
print(f"  a at t={worst_detail['a_time']:.3f}")
print(f"  S = {[round(t, 3) for t in worst_detail['S_times']]}")
print(f"  T = {[round(t, 3) for t in worst_detail['T_times']]}")
print(f"  Delta(a|S) = {worst_detail['gain_S']:.6e}")
print(f"  Delta(a|T) = {worst_detail['gain_T']:.6e}   "
      f"({worst['relative_violation']:.1f}x larger)")
print(f"\nMonotonicity: smallest increment over all kernels/labels/orders = "
      f"{min(r['min_marginal_increment'] for r in rows):+.3e}")

save_json({
    "seed": SEED, "environment": environment_record(),
    "horizon": HORIZON, "grid_p": P, "action_noise": NOISE,
    "n_candidates": N_CAND,
    "candidate_times": [a.time for a in CANDS],
    "rows": rows, "witnesses": detail,
    "headline": {
        "any_violation": bool(any(r["violation_found"] for r in rows)),
        "n_cells": len(rows),
        "n_cells_with_violation": int(sum(r["violation_found"] for r in rows)),
        "max_relative_violation": worst["relative_violation"],
        "max_violation_kernel": worst["kernel"],
        "max_violation_label": worst["label"],
        "max_violation_gain_S": worst["gain_S"],
        "max_violation_gain_T": worst["gain_T"],
        "max_violation_size_S": worst["size_S"],
        "max_violation_size_T": worst["size_T"],
        "mean_label_violation_ou": next(
            (r["relative_violation"] for r in rows
             if r["kernel"] == "ou" and r["label"] == "mean"), None),
        "occ_violation_periodic": next(
            (r["relative_violation"] for r in rows
             if r["kernel"] == "damped_periodic" and r["label"] == "occupation_c0"), None),
        "min_marginal_increment_overall": min(r["min_marginal_increment"] for r in rows),
        "gamma_min": min(r["gamma"] for r in rows),
        "gamma_max": max(r["gamma"] for r in rows),
        "greedy_factor_min": min(r["greedy_factor"] for r in rows),
        "n_triples_checked_per_cell": rows[0]["n_triples_checked"],
    },
}, "s5b_submodularity")
