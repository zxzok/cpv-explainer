"""Framework schematic. NOT included in the article: retained as a regression run.

Panel (a) shows a real simulated trait-state trajectory, the occupation-time
label it induces, and the two equal-budget protocols under comparison.
Panel (b) shows the exact decomposition of Section 2.4 computed from the
package rather than drawn as a cartoon: the protocol gap is an exact Bayes-risk
difference, and the remaining gap is measured by actually training ridge
predictors at two training-set sizes.  The headline contrast is between a low
ceiling with a small model gap and a high ceiling with a larger one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import (PALETTE, SEED, save_json, save_figure,  # noqa: E402
                                setup_matplotlib)

from protocol_ceiling import (ThresholdLabel, dispersed_protocol,  # noqa: E402
                              evaluate_protocol, make_kernel, r2_score,
                              same_time_protocol, trait_state_correlation,
                              uniform_grid)
from protocol_ceiling.covariance import protocol_matrices  # noqa: E402
from protocol_ceiling.diagnostics import ridge_fit_predict  # noqa: E402

plt = setup_matplotlib()
rng = np.random.default_rng(SEED)

T, P, ALPHA, TAU, NOISE, N_SEG = 20.0, 256, 0.30, 1.0, 0.5, 8
grid = uniform_grid(T, P)
label = ThresholdLabel(c=0.0)
K = trait_state_correlation(grid, ALPHA, make_kernel("ou", tau=TAU))
L = np.linalg.cholesky(K + 1e-10 * np.eye(P))

protocols = {
    "same-time": same_time_protocol(grid, N_SEG, noise=NOISE),
    "dispersed": dispersed_protocol(grid, N_SEG, noise=NOISE),
}
ceilings = {k: evaluate_protocol(label, K, grid, v).ceiling
            for k, v in protocols.items()}


def simulate(n: int, actions):
    Z = rng.standard_normal((n, P)) @ L.T
    theta = label.apply(Z) @ grid.weights
    A, R = protocol_matrices(actions, grid)
    Y = Z @ A.T + rng.standard_normal((n, A.shape[0])) @ np.sqrt(R)
    return Y, theta


achieved: dict[tuple[str, int], float] = {}
for name, acts in protocols.items():
    Yte, tte = simulate(20000, acts)
    for n in (100, 5000):
        Ytr, ttr = simulate(n, acts)
        achieved[(name, n)] = r2_score(tte, ridge_fit_predict(Ytr, ttr, Yte, rng=rng))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75),
                         gridspec_kw={"width_ratios": [1.45, 1.0]})

# ---------------------------------------------------------------- panel (a) --
ax = axes[0]
z = rng.standard_normal(P) @ L.T
ax.plot(grid.times, z, color="0.4", lw=0.9, zorder=2)
ax.fill_between(grid.times, 0, z, where=(z > 0), color=PALETTE[0], alpha=0.20,
                interpolate=True, zorder=1)
ax.axhline(0.0, color="0.2", lw=0.8, ls="--", zorder=3)

t_disp = np.array([a.time for a in protocols["dispersed"]])
z_disp = np.array([z[int(np.argmin(np.abs(grid.times - t)))] for t in t_disp])
ax.plot(t_disp, z_disp, "o", ms=4.5, color=PALETTE[2], zorder=5,
        label=r"dispersed: $D{=}8,\,M{=}1$")
t_same = protocols["same-time"][0].time
z_same = z[int(np.argmin(np.abs(grid.times - t_same)))]
ax.plot([t_same] * 8, z_same + np.linspace(-0.30, 0.30, 8), "_", ms=7,
        color=PALETTE[1], mew=1.4, zorder=5, label=r"same-time: $D{=}1,\,M{=}8$")

occ = float((z > 0).mean())
ax.set_xlabel("time $t$")
ax.set_ylabel("latent state $Z(t)$")
ax.set_xlim(0, T)
ax.set_ylim(z.min() - 0.9, z.max() + 1.25)
ax.legend(loc="upper left", ncol=2, columnspacing=1.0, handletextpad=0.4)
ax.text(0.985, 0.05,
        rf"$\Theta=\int_0^T\!\mathbf{{1}}\{{Z(t)>c\}}\,\mathrm{{d}}t/T={occ:.2f}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8)

# ---------------------------------------------------------------- panel (b) --
ax = axes[1]
bars = [
    ("same-time\n$n=5000$", "same-time", 5000),
    ("dispersed\n$n=100$", "dispersed", 100),
    ("dispersed\n$n=5000$", "dispersed", 5000),
]
for x, (tick, proto, n) in enumerate(bars):
    r2 = achieved[(proto, n)]
    ceil = ceilings[proto]
    parts = [(r2, PALETTE[2], "achieved $R^2$"),
             (ceil - r2, PALETTE[3], "approximation $+$\nestimation gap"),
             (1.0 - ceil, PALETTE[1], "protocol gap")]
    bottom = 0.0
    for h, col, lab in parts:
        ax.bar(x, h, bottom=bottom, color=col, width=0.62,
               edgecolor="white", lw=0.7, label=lab if x == 0 else None)
        bottom += h
    ax.plot([x - 0.31, x + 0.31], [ceil, ceil], color="k", lw=1.1, zorder=6)

ax.set_xticks(range(len(bars)))
ax.set_xticklabels([b[0] for b in bars], fontsize=7.5)
ax.set_ylabel(r"share of $\mathrm{Var}(\Theta)$")
ax.set_ylim(0, 1)
ax.set_xlim(-0.6, len(bars) - 0.4)
from matplotlib.lines import Line2D
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([0], [0], color="k", lw=1.1))
labels.append(r"exact ceiling $\mathcal{I}(S)$")
ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)

save_figure(fig, "fig_framework")

summary = {
    "alpha": ALPHA, "tau": TAU, "T": T, "p": P, "noise": NOISE,
    "n_segments": N_SEG, "label": "occupation c=0", "seed": SEED,
    "ceilings": ceilings,
    "achieved_r2": {f"{k[0]}_n{k[1]}": v for k, v in achieved.items()},
    "utilisation": {f"{k[0]}_n{k[1]}": v / ceilings[k[0]] for k, v in achieved.items()},
}
save_json(summary, "fig1_framework")
for k, v in summary["ceilings"].items():
    print(f"  ceiling[{k}] = {v:.4f}")
for k, v in summary["achieved_r2"].items():
    print(f"  ridge R2[{k}] = {v:.4f}   utilisation = {summary['utilisation'][k]:.3f}")
