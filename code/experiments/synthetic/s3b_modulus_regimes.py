"""S3b: is the Holder-1/2 exponent sharp, and when does it bind?

Experiment S3 measures an empirical uniform-error slope of about -0.41 for
*every* label it runs -- mean and sigmoid included, not only the threshold ones
-- so that slope is a property of the finite-m estimation pipeline and not of
the Holder exponent.  Two separate questions follow, and this script answers
both: is the Holder-1/2 exponent sharp at all (part A), and does it bind on the
correlation shapes a time grid produces (part B)?

On correlations bounded away from +-1, C_g is
Lipschitz with the finite constant

    L_g(r_max) = exp{-c^2/(1+r_max)} / {2 pi sqrt(1 - r_max^2)},

which diverges only as r_max -> 1.  So beta = 1/2 is a statement *uniform over
all correlation matrices*, and it binds only when near-unit correlations carry
non-negligible weight in the aggregate.

Three parts settle both questions.

A (sharpness).  On the equicorrelated boundary family
K(delta) = (1-delta) J + delta I, every off-diagonal entry sits at distance
delta from the boundary.  Here the bound must be attained.

B (temporal covariances).  On banded/AR-type correlations -- the shape that
actual time grids produce -- only O(p) of the p^2 entries are near the
boundary, so they carry O(1/p) of the weight and the aggregate behaves
Lipschitz.

C (pipeline).  The full calibration -> ceiling estimation slope, sweeping the
grid resolution that controls r_max in practice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import (PALETTE, SEED, environment_record, save_csv,  # noqa: E402
                                save_figure, save_json, setup_matplotlib, Timer)

from protocol_ceiling import (Action, MeanLabel, ThresholdLabel,  # noqa: E402
                              fit_covariance, make_kernel, sample_paths,
                              sigmoid_label, trait_state_correlation,
                              uniform_grid)
from protocol_ceiling.estimation import estimate_protocol_ceiling  # noqa: E402
from protocol_ceiling.risk import label_variance  # noqa: E402

plt = setup_matplotlib()
rng = np.random.default_rng(SEED)

LABELS = {
    "mean": MeanLabel(),
    "sigmoid": sigmoid_label(slope=2.0, c=0.0),
    "occ_c0": ThresholdLabel(c=0.0),
    "occ_c1": ThresholdLabel(c=1.0),
}
PRETTY = {"mean": "mean", "sigmoid": "sigmoid", "occ_c0": r"occupation $c=0$",
          "occ_c1": r"occupation $c=1$"}


def loglog_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0)
    lx, ly = np.log(x[ok]), np.log(y[ok])
    b, a = np.polyfit(lx, ly, 1)
    resid = ly - (a + b * lx)
    se = float(np.sqrt(np.sum(resid**2) / max(lx.size - 2, 1)
                       / max(np.sum((lx - lx.mean()) ** 2), 1e-30)))
    return float(b), se


# ---------------------------------------------------------------------------
# A. sharpness on the equicorrelated boundary family
# ---------------------------------------------------------------------------
def part_a(p: int = 40, n_points: int = 15) -> list[dict]:
    omega = np.full(p, 1.0 / p)

    def K_of(delta: float) -> np.ndarray:
        K = np.full((p, p), 1.0 - delta)
        np.fill_diagonal(K, 1.0)
        return K

    rows = []
    for delta in np.logspace(-1, -8, n_points):
        K1, K2 = K_of(delta), K_of(delta / 2.0)
        e = float(np.linalg.norm(K2 - K1, 2))
        rows.append({"delta": float(delta), "e_norm": e,
                     **{f"dV_{k}": abs(label_variance(l, K2, omega)
                                       - label_variance(l, K1, omega))
                        for k, l in LABELS.items()}})
    return rows


# ---------------------------------------------------------------------------
# B. banded (temporal-like) correlations
# ---------------------------------------------------------------------------
def part_b(p: int = 40, n_sizes: int = 12, n_draws: int = 30) -> list[dict]:
    omega = np.full(p, 1.0 / p)
    lags = np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
    rows = []
    for r_max in (0.30, 0.90, 0.99, 0.999, 0.9999):
        K = r_max ** lags
        np.fill_diagonal(K, 1.0)
        base = {k: label_variance(l, K, omega) for k, l in LABELS.items()}
        for s in np.logspace(-6, -1.2, n_sizes):
            dV = {k: [] for k in LABELS}
            enorm = []
            for _ in range(n_draws):
                G = rng.standard_normal((p, p))
                E = 0.5 * (G + G.T)
                np.fill_diagonal(E, 0.0)
                E = s * E / np.linalg.norm(E, 2)
                Kp = np.clip(K + E, -1.0, 1.0)
                np.fill_diagonal(Kp, 1.0)
                enorm.append(float(np.linalg.norm(Kp - K, 2)))
                for k, l in LABELS.items():
                    dV[k].append(abs(label_variance(l, Kp, omega) - base[k]))
            rows.append({"r_max": r_max, "scale": float(s),
                         "e_norm": float(np.mean(enorm)),
                         **{f"dV_{k}": float(np.mean(v)) for k, v in dV.items()}})
    return rows


# ---------------------------------------------------------------------------
# C. full estimation pipeline across grid resolutions
# ---------------------------------------------------------------------------
def part_c(horizon: float = 20.0, tau: float = 1.0, noise: float = 0.5,
           m_grid=(50, 100, 200, 400, 800, 1600), n_rep: int = 12) -> list[dict]:
    kern = make_kernel("ou", tau=tau)
    protocols = [[Action(time=float(t), noise=noise) for t in ts]
                 for ts in ([2.5, 7.5, 12.5, 17.5], [5.0, 10.0, 15.0],
                            [1.0, 6.0, 11.0, 16.0], [10.0], [3.0, 17.0])]
    rows = []
    for p in (32, 64, 128, 256):
        grid = uniform_grid(horizon, p)
        K = trait_state_correlation(grid, 0.0, kern)
        r_max = float(np.max(np.abs(K - np.diag(np.diag(K)))))
        truth = {k: np.array([estimate_protocol_ceiling(l, K, grid, S)
                              for S in protocols]) for k, l in LABELS.items()}
        for m in m_grid:
            unif = {k: [] for k in LABELS}
            kerr = []
            for _ in range(n_rep):
                Khat = fit_covariance(sample_paths(K, m, rng)).K
                kerr.append(float(np.linalg.norm(Khat - K, 2)))
                for k, l in LABELS.items():
                    est = np.array([estimate_protocol_ceiling(l, Khat, grid, S)
                                    for S in protocols])
                    unif[k].append(float(np.max(np.abs(est - truth[k]))))
            rows.append({"p": p, "r_max": r_max, "m": m,
                         "k_err": float(np.mean(kerr)),
                         **{f"unif_{k}": float(np.mean(v)) for k, v in unif.items()}})
    return rows


with Timer("A: sharpness on the boundary family"):
    rows_a = part_a()
with Timer("B: banded temporal-like correlations"):
    rows_b = part_b()
with Timer("C: estimation pipeline vs grid resolution"):
    rows_c = part_c()

save_csv(rows_a, "s3b_sharpness_boundary")
save_csv(rows_b, "s3b_banded")
save_csv(rows_c, "s3b_pipeline_resolution")

exp_a = {k: loglog_slope([r["e_norm"] for r in rows_a],
                         [r[f"dV_{k}"] for r in rows_a]) for k in LABELS}
exp_b: dict[str, dict[float, tuple[float, float]]] = {k: {} for k in LABELS}
for r_max in sorted({r["r_max"] for r in rows_b}):
    sub = [r for r in rows_b if r["r_max"] == r_max]
    for k in LABELS:
        exp_b[k][r_max] = loglog_slope([r["e_norm"] for r in sub],
                                       [r[f"dV_{k}"] for r in sub])
slope_c: dict[str, dict[int, tuple[float, float]]] = {k: {} for k in LABELS}
kerr_c: dict[int, tuple[float, float]] = {}
for p in sorted({r["p"] for r in rows_c}):
    sub = [r for r in rows_c if r["p"] == p]
    kerr_c[p] = loglog_slope([r["m"] for r in sub], [r["k_err"] for r in sub])
    for k in LABELS:
        slope_c[k][p] = loglog_slope([r["m"] for r in sub],
                                     [r[f"unif_{k}"] for r in sub])

print("\nA. equicorrelated boundary family: exponent of |dV_g| in ||E||_op")
for k in LABELS:
    print(f"   {k:>10}: {exp_a[k][0]:.4f}  (se {exp_a[k][1]:.4f})")
print("   -> predicted 1.0 for smooth labels, 0.5 for threshold labels")

print("\nB. banded correlations: exponent of |dV_g| in ||E||_op")
print(f"   {'r_max':>8}" + "".join(f"{k:>12}" for k in LABELS))
for r_max in sorted(exp_b["mean"]):
    print(f"   {r_max:>8.4f}" + "".join(f"{exp_b[k][r_max][0]:>12.3f}" for k in LABELS))

print("\nC. pipeline: log-log slope of the uniform ceiling error in m")
print(f"   {'p':>5} {'r_max':>8} {'||K-K||':>9}" + "".join(f"{k:>11}" for k in LABELS))
for p in sorted(slope_c["mean"]):
    r_max = [r["r_max"] for r in rows_c if r["p"] == p][0]
    print(f"   {p:>5} {r_max:>8.4f} {kerr_c[p][0]:>9.3f}"
          + "".join(f"{slope_c[k][p][0]:>11.3f}" for k in LABELS))

# ------------------------------------------------------------------- figure --
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)

ax = axes[0]
for i, k in enumerate(LABELS):
    ax.loglog([r["e_norm"] for r in rows_a], [r[f"dV_{k}"] for r in rows_a],
              "o-", ms=3, color=PALETTE[i],
              label=f"{PRETTY[k]}: $\\beta={exp_a[k][0]:.2f}$")
ax.set_xlabel(r"$\|E\|_{\mathrm{op}}$")
ax.set_ylabel(r"$|V_g(K+E)-V_g(K)|$")
ax.set_title("(a) boundary family: bound is sharp", fontsize=8)
ax.legend(loc="upper left", fontsize=5.8, handlelength=1.4,
              labelspacing=0.25, borderpad=0.25)

ax = axes[1]
rmaxes = sorted(exp_b["mean"])
for i, k in enumerate(LABELS):
    ax.plot(range(len(rmaxes)), [exp_b[k][r][0] for r in rmaxes], "o-",
            color=PALETTE[i], label=PRETTY[k])
ax.axhline(1.0, color="0.35", ls="--", lw=0.9)
ax.axhline(0.5, color="0.35", ls=":", lw=0.9)
ax.set_xticks(range(len(rmaxes)))
ax.set_xticklabels([f"{r:g}" for r in rmaxes], fontsize=6.5)
ax.set_xlabel(r"$r_{\max}$ of a banded correlation")
ax.set_ylabel(r"measured exponent $\beta$")
ax.set_ylim(0.35, 1.25)
ax.set_title("(b) temporal-like: bound is slack", fontsize=8)
ax.legend(loc="lower left", fontsize=6.2, ncol=2)

ax = axes[2]
ps = sorted(slope_c["mean"])
rm = [[r["r_max"] for r in rows_c if r["p"] == p][0] for p in ps]
for i, k in enumerate(LABELS):
    ax.plot(rm, [slope_c[k][p][0] for p in ps], "o-", color=PALETTE[i],
            label=PRETTY[k])
ax.plot(rm, [kerr_c[p][0] for p in ps], "s--", color="0.45",
        label=r"$\|\widehat K-K\|_{\mathrm{op}}$")
ax.axhline(-0.5, color="0.35", ls="--", lw=0.9)
ax.axhline(-0.25, color="0.35", ls=":", lw=0.9)
ax.set_xlabel(r"$r_{\max}$ (grid $p=32,64,128,256$)")
ax.set_ylabel(r"log--log slope in $m$")
ax.set_ylim(-0.62, -0.18)
ax.set_title("(c) pipeline rate", fontsize=8)
ax.legend(loc="upper left", fontsize=6.0, ncol=2)

save_figure(fig, "fig_modulus_regimes")

save_json({
    "seed": SEED, "environment": environment_record(),
    "boundary_family_exponents": {k: {"slope": v[0], "se": v[1]}
                                  for k, v in exp_a.items()},
    "banded_exponents": {k: {str(r): {"slope": v[0], "se": v[1]}
                             for r, v in d.items()} for k, d in exp_b.items()},
    "pipeline_slopes": {k: {str(p): {"slope": v[0], "se": v[1]}
                            for p, v in d.items()} for k, d in slope_c.items()},
    "pipeline_kerr_slopes": {str(p): {"slope": v[0], "se": v[1]}
                             for p, v in kerr_c.items()},
    "headline": {
        "boundary_exponent_occ_c0": exp_a["occ_c0"][0],
        "boundary_exponent_occ_c1": exp_a["occ_c1"][0],
        "boundary_exponent_mean": exp_a["mean"][0],
        "boundary_exponent_sigmoid": exp_a["sigmoid"][0],
        "banded_exponent_occ_c0_at_0.9999": exp_b["occ_c0"][0.9999][0],
        "pipeline_slope_occ_c0_p256": slope_c["occ_c0"][256][0],
        "pipeline_slope_mean_p256": slope_c["mean"][256][0],
        "pipeline_kerr_slope_p256": kerr_c[256][0],
    },
}, "s3b_modulus_regimes")
