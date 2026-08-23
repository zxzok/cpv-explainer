"""S8: calibration size determines the resolution at which a protocol can be designed.

The real-data analyses of this paper find that dispersing observations is a
robust gain while optimising exact placements is not.  That is not an anomaly of
those cohorts: it is what `cor:regret` predicts.  Ordering protocols into nested
classes from coarse geometry to exact placement, a larger class has smaller
approximation error and larger uniform estimation error, and the selector that
maximises ``I_hat(S) - eps_l`` pays only the better of the two.

This script measures the whole trade-off as a function of the calibration sample
size ``m``:

* which resolution level the adaptive selector picks;
* the realised regret of the adaptive selector against always committing to a
  fixed level;
* whether the realised regret ever exceeds the `cor:regret` bound.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import (PALETTE, SEED, environment_record, save_csv,  # noqa: E402
                                save_figure, save_json, setup_matplotlib, Timer)

from protocol_ceiling import (MeanLabel, ThresholdLabel, fit_covariance,  # noqa: E402
                              make_kernel, project_psd, sample_paths,
                              to_correlation, uniform_grid)
from protocol_ceiling.estimation import estimate_protocol_ceiling  # noqa: E402
from protocol_ceiling.resolution import (bootstrap_uniform_error,  # noqa: E402
                                         nested_classes,
                                         resolution_adaptive_select,
                                         theorem_bound, uniform_error)

plt = setup_matplotlib()
rng = np.random.default_rng(SEED)

T, P, BUDGET, NOISE = 20.0, 64, 4, 0.4
M_GRID = (25, 50, 100, 250, 500, 1000)
N_REP = 30
N_BOOT = 24
LABELS = {"mean": MeanLabel(), "occupation": ThresholdLabel(c=0.0)}
GRID = uniform_grid(T, P)


def nonstationary_K(grid, tau_lo=0.3, tau_hi=3.0, alpha=0.15):
    """Horizon-varying local correlation time, so exact placement really matters."""
    t = grid.times
    tau_t = tau_lo + (tau_hi - tau_lo) * (t / t.max())
    lag = np.abs(t[:, None] - t[None, :])
    K = np.exp(-lag / np.sqrt(np.outer(tau_t, tau_t)))
    np.fill_diagonal(K, 1.0)
    K = alpha + (1.0 - alpha) * K
    np.fill_diagonal(K, 1.0)
    return to_correlation(project_psd(K, 1e-9))


K_TRUE = nonstationary_K(GRID)
# More coarse bins than windows, or level 3 has exactly one admissible choice
# and collapses onto the full-horizon dispersed protocol of level 2 --- the
# ladder then has three distinct rungs while the figure draws four.
CLASSES = nested_classes(GRID, budget=BUDGET, noise=NOISE, n_fine=12,
                         n_coarse_bins=8)
LEVEL_NAMES = {c.level: c.name for c in CLASSES}
print("nested classes: " + ", ".join(f"L{c.level} {c.name} ({len(c)})" for c in CLASSES))

rows: list[dict] = []
with Timer("S8 resolution-adaptive selection"):
    for lab_name, label in LABELS.items():
        finest = CLASSES[-1]
        best_overall = max(estimate_protocol_ceiling(label, K_TRUE, GRID, S)
                           for S in finest.protocols)
        for m in M_GRID:
            for rep in range(N_REP):
                W = sample_paths(K_TRUE, m, rng)
                K_hat = fit_covariance(W).K
                boots = [fit_covariance(W[rng.integers(0, m, size=m)]).K
                         for _ in range(N_BOOT)]

                eps_boot, eps_true = {}, {}
                for cls in CLASSES:
                    eps_boot[cls.level] = bootstrap_uniform_error(
                        label, boots, K_hat, GRID, cls.protocols, quantile=0.9)
                    eps_true[cls.level] = uniform_error(
                        label, K_hat, K_TRUE, GRID, cls.protocols)

                # Two arms.  The ORACLE arm feeds the selector the same eps that
                # the bound uses, which is the only setting in which `cor:regret`
                # formally applies; it is the honest test of the theorem.  The
                # BOOTSTRAP arm is what a practitioner can actually run, and its
                # guarantee holds only on the event that every bootstrap eps is a
                # valid uniform bound, which we record.
                sel_oracle = resolution_adaptive_select(label, K_hat, GRID, CLASSES,
                                                        eps_true, K_true=K_TRUE)
                bnd_oracle = theorem_bound(CLASSES, eps_true, label, K_TRUE, GRID)
                sel = resolution_adaptive_select(label, K_hat, GRID, CLASSES,
                                                 eps_boot, K_true=K_TRUE)
                bnd = theorem_bound(CLASSES, eps_boot, label, K_TRUE, GRID)
                eps_valid = all(eps_boot[k] >= eps_true[k] for k in eps_true)

                # regret of committing to each single level
                fixed = {}
                for cls in CLASSES:
                    vals = np.array([estimate_protocol_ceiling(label, K_hat, GRID, S)
                                     for S in cls.protocols])
                    pick = cls.protocols[int(np.argmax(vals))]
                    fixed[cls.level] = float(
                        best_overall - estimate_protocol_ceiling(label, K_TRUE,
                                                                 GRID, pick))

                rows.append({
                    "label": lab_name, "m": m, "rep": rep,
                    "selected_level": sel.level,
                    "adaptive_regret": sel.regret,
                    "theorem_bound": bnd["bound"],
                    "bound_satisfied": bool(sel.regret <= bnd["bound"] + 1e-12),
                    "selected_level_oracle": sel_oracle.level,
                    "adaptive_regret_oracle": sel_oracle.regret,
                    "theorem_bound_oracle": bnd_oracle["bound"],
                    "bound_satisfied_oracle":
                        bool(sel_oracle.regret <= bnd_oracle["bound"] + 1e-12),
                    "eps_boot_valid": bool(eps_valid),
                    **{f"eps_boot_L{k}": v for k, v in eps_boot.items()},
                    **{f"eps_true_L{k}": v for k, v in eps_true.items()},
                    **{f"regret_fixed_L{k}": v for k, v in fixed.items()},
                })

save_csv(rows, "s8_resolution")

# ------------------------------------------------------------------ analysis --
import collections  # noqa: E402

summary: dict = {}
for lab_name in LABELS:
    sub = [r for r in rows if r["label"] == lab_name]
    per_m = {}
    for m in M_GRID:
        cell = [r for r in sub if r["m"] == m]
        counts = collections.Counter(r["selected_level"] for r in cell)
        per_m[m] = {
            "modal_level": int(counts.most_common(1)[0][0]),
            "level_distribution": {str(k): v / len(cell) for k, v in sorted(counts.items())},
            "mean_selected_level": float(np.mean([r["selected_level"] for r in cell])),
            "mean_selected_level_oracle":
                float(np.mean([r["selected_level_oracle"] for r in cell])),
            "adaptive_regret": float(np.mean([r["adaptive_regret"] for r in cell])),
            "adaptive_regret_oracle":
                float(np.mean([r["adaptive_regret_oracle"] for r in cell])),
            "eps_boot_valid_frac": float(np.mean([r["eps_boot_valid"] for r in cell])),
            **{f"regret_fixed_L{k}": float(np.mean([r[f"regret_fixed_L{k}"] for r in cell]))
               for k in LEVEL_NAMES},
            **{f"eps_true_L{k}": float(np.mean([r[f"eps_true_L{k}"] for r in cell]))
               for k in LEVEL_NAMES},
        }
    summary[lab_name] = per_m
    print(f"\n{lab_name}:")
    print(f"  {'m':>5} {'level':>6} " + " ".join(f"{'reg L'+str(k):>9}" for k in LEVEL_NAMES)
          + f" {'adaptive':>9}")
    for m in M_GRID:
        d = per_m[m]
        print(f"  {m:>5} {d['mean_selected_level']:>6.2f} "
              + " ".join(f"{d[f'regret_fixed_L{k}']:>9.4f}" for k in LEVEL_NAMES)
              + f" {d['adaptive_regret']:>9.4f}")

violations = sum(1 for r in rows if not r["bound_satisfied"])
violations_oracle = sum(1 for r in rows if not r["bound_satisfied_oracle"])
eps_valid_frac = float(np.mean([r["eps_boot_valid"] for r in rows]))
print(f"\ncor:regret, oracle-eps arm (the setting the corollary covers): "
      f"{violations_oracle} violations in {len(rows)} selections")
print(f"Bootstrap-eps arm: {violations} violations; the bootstrap eps was a valid "
      f"uniform bound at every level in {eps_valid_frac:.1%} of replications")
best_fixed = {}
for lab_name in LABELS:
    for m in M_GRID:
        d = summary[lab_name][m]
        bf = min(d[f"regret_fixed_L{k}"] for k in LEVEL_NAMES)
        best_fixed[(lab_name, m)] = bf
adv = [best_fixed[(l, m)] - summary[l][m]["adaptive_regret"]
       for l in LABELS for m in M_GRID]
print(f"Adaptive minus best fixed level: mean {np.mean(adv):+.4f}, "
      f"wins in {sum(a > 0 for a in adv)}/{len(adv)} cells")

# --------------------------------------------------------------------- figure --
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)

ax = axes[0]
for i, lab_name in enumerate(LABELS):
    ax.plot(M_GRID, [summary[lab_name][m]["mean_selected_level"] for m in M_GRID],
            "o-", color=PALETTE[i], label=lab_name)
ax.set_xscale("log")
ax.set_yticks(sorted(LEVEL_NAMES))
ax.set_yticklabels([f"{k}: {LEVEL_NAMES[k]}" for k in sorted(LEVEL_NAMES)], fontsize=6.2)
ax.set_xlabel("calibration objects $m$")
ax.set_ylabel("selected resolution")
ax.set_title("(a) reliably selectable resolution", fontsize=8)
ax.legend(fontsize=6.5)

ax = axes[1]
lab_name = "occupation"
for k in sorted(LEVEL_NAMES):
    ax.plot(M_GRID, [summary[lab_name][m][f"regret_fixed_L{k}"] for m in M_GRID],
            "o--", ms=3, color=PALETTE[k - 1], lw=1.0,
            label=f"fixed L{k}")
ax.plot(M_GRID, [summary[lab_name][m]["adaptive_regret"] for m in M_GRID],
        "s-", color="k", lw=1.5, label="resolution-adaptive")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("calibration objects $m$")
ax.set_ylabel("true selection regret")
ax.set_title("(b) occupation label", fontsize=8)
ax.legend(fontsize=6.0, ncol=2)

ax = axes[2]
for k in sorted(LEVEL_NAMES):
    ax.plot(M_GRID, [summary[lab_name][m][f"eps_true_L{k}"] for m in M_GRID],
            "o-", ms=3, color=PALETTE[k - 1], label=f"$\\varepsilon_{k}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("calibration objects $m$")
ax.set_ylabel(r"uniform error $\varepsilon_\ell$")
ax.set_title("(c) estimation cost of resolution", fontsize=8)
ax.legend(fontsize=6.2, ncol=2)

# NOT "fig_resolution": that name belongs to experiments/make_fig_resolution.py,
# whose output feeds panels (c) and (d) of Figure 2 of the manuscript.  This
# three-panel diagnostic includes the data-driven level selector, which the
# article deliberately does not present, so it must not share the file name.
save_figure(fig, "fig_resolution_selector")

head = {
    "n_replications": N_REP, "n_bootstrap": N_BOOT, "budget": BUDGET,
    "grid_p": P, "horizon": T, "noise": NOISE, "seed": SEED,
    "class_sizes": {str(c.level): len(c) for c in CLASSES},
    "class_names": {str(k): v for k, v in LEVEL_NAMES.items()},
    "bound_violations": violations,
    "bound_violations_oracle": violations_oracle,
    "eps_boot_valid_frac": eps_valid_frac,
    "adaptive_minus_best_fixed_mean": float(np.mean(adv)),
    "adaptive_beats_best_fixed_cells": int(sum(a > 0 for a in adv)),
    "n_cells": int(len(adv)),
    "n_selections": len(rows),
    "selected_level_at_m25": summary["occupation"][25]["mean_selected_level"],
    "selected_level_at_m1000": summary["occupation"][1000]["mean_selected_level"],
    "adaptive_regret_m25": summary["occupation"][25]["adaptive_regret"],
    "adaptive_regret_m1000": summary["occupation"][1000]["adaptive_regret"],
    "regret_fixed_finest_m25": summary["occupation"][25]["regret_fixed_L4"],
    "regret_fixed_coarsest_m25": summary["occupation"][25]["regret_fixed_L1"],
    "regret_fixed_finest_m1000": summary["occupation"][1000]["regret_fixed_L4"],
    "regret_fixed_coarsest_m1000": summary["occupation"][1000]["regret_fixed_L1"],
    "eps_L1_m25": summary["occupation"][25]["eps_true_L1"],
    "eps_L4_m25": summary["occupation"][25]["eps_true_L4"],
    "eps_L1_m1000": summary["occupation"][1000]["eps_true_L1"],
    "eps_L4_m1000": summary["occupation"][1000]["eps_true_L4"],
}
save_json({"headline": head, "summary": summary, "environment": environment_record()},
          "s8_resolution")
