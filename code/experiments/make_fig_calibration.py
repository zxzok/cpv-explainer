"""Four-panel calibration figure: value error, regret, and protocol classes.

The four panels prove one thing, so they belong in one figure. (a) the uniform
protocol-value error against calibration size, for a smooth and a threshold
target, on log-log axes with fitted slopes; (b) protocol-selection regret
against calibration size with the 2*eps_m envelope of the regret theorem
overlaid; (c) the true selection regret from committing to each fixed
protocol class; (d) the uniform error eps_l within each class, which is the
estimation term of the nested-class bound.

Reads results/s3_ceiling_estimation.csv, results/s4_selection_regret.csv and
results/s8_resolution.csv; writes figures/fig_calibration.pdf/.png.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import PALETTE, save_figure, setup_matplotlib  # noqa: E402

setup_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402

LEVELS = (1, 2, 3, 4)
LEVEL_NAMES = {1: "layouts", 2: "phase", 3: "coarse bins", 4: "fine supports"}
RES_LABEL = "occupation"        # s8 runs two targets; averaging them would mix curves


def _rows(name):
    with (ROOT / "results" / f"{name}.csv").open() as fh:
        return list(csv.DictReader(fh))


def curve(rows, key, value, group="m"):
    """Mean of `value` by `group`; raises if nothing matches.

    A silently empty panel is worse than a crash: the figure still compiles and
    the caption still describes curves that are not there.
    """
    if rows and value not in rows[0]:
        raise SystemExit(f"column {value!r} not in the CSV; have {sorted(rows[0])[:12]}")
    acc = defaultdict(list)
    for r in rows:
        if key and r.get(key[0]) != key[1]:
            continue
        v = r.get(value)
        if v in (None, "", "nan"):
            continue
        acc[float(r[group])].append(float(v))
    if not acc:
        raise SystemExit(f"no rows for {value!r} (key={key!r}) -- panel would be blank")
    xs = np.array(sorted(acc))
    ys = np.array([np.mean(acc[x]) for x in xs])
    # standard error of the plotted mean; for s3 each row is already a per-cell
    # mean over n_rep replications, so its own sd column carries the dispersion
    ses = []
    for x in xs:
        v = np.asarray(acc[x], dtype=float)
        ses.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)
    return xs, ys, np.array(ses)


def _s4_clustered_curve(rows, value, group="m"):
    """Pooled S4 mean and SE, clustering targets within each calibration draw.

    Each ``(kernel, m, rep)`` cell uses one dense calibration sample and the
    resulting covariance estimate is scored for three targets.  The targets
    therefore share calibration error.  We average them within replication,
    estimate replication variance separately for each kernel, and then combine
    the two equally weighted kernel strata.
    """
    acc = defaultdict(list)
    for r in rows:
        v = r.get(value)
        if v in (None, "", "nan"):
            continue
        acc[(float(r[group]), r["kernel"], r["rep"])].append(float(v))
    if not acc:
        raise SystemExit(f"no rows for {value!r} -- panel would be blank")

    xs = np.array(sorted({key[0] for key in acc}))
    ys, ses = [], []
    for x in xs:
        by_kernel = defaultdict(list)
        for (m, kernel, _rep), vals in acc.items():
            if m == x:
                by_kernel[kernel].append(float(np.mean(vals)))
        kernel_means = []
        kernel_variances = []
        for kernel in sorted(by_kernel):
            vals = np.asarray(by_kernel[kernel], dtype=float)
            kernel_means.append(float(vals.mean()))
            kernel_variances.append(float(vals.var(ddof=1) / vals.size)
                                    if vals.size > 1 else 0.0)
        n_kernel = len(kernel_means)
        ys.append(float(np.mean(kernel_means)))
        ses.append(float(np.sqrt(sum(kernel_variances)) / n_kernel))
    return xs, np.asarray(ys), np.asarray(ses)


def _se_from_sd(rows, key, group="m", sd="uniform_err_sd", n="n_rep"):
    """Standard error of a per-cell mean, from the stored sd and replication count.

    ``sd`` must name the dispersion OF THE PLOTTED STATISTIC.  The CSV also has a
    bare ``sd`` column, which is the sd of the reference-protocol signed error --
    a different quantity that must not be used as a band for the uniform error.
    """
    acc = defaultdict(list)
    for r in rows:
        if key and r.get(key[0]) != key[1]:
            continue
        try:
            acc[float(r[group])].append((float(r[sd]), float(r[n])))
        except (TypeError, ValueError):
            continue
    xs = np.array(sorted(acc))
    out = []
    for x in xs:
        cells = acc[x]                      # one entry per pooled cell
        var = sum((sd_ ** 2) / max(n_, 1) for sd_, n_ in cells) / max(len(cells), 1) ** 2
        out.append(float(np.sqrt(var)))
    return xs, np.array(out)


PRIMARY_CONFIG = "ou_alpha0_nu0.25"   # CONFIGS[0] of s3_ceiling_estimation.py


def main() -> None:
    s3, s4, s8 = _rows("s3_ceiling_estimation"), _rows("s4_selection_regret"), _rows("s8_resolution")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=True)

    # (a) uniform value error, with the log-log fit the text quotes drawn on top
    # (the caption promises fitted slopes; a caption that names a line the panel
    # does not draw is the same defect as a blank panel).
    ax = axes[0, 0]
    slopes = json.loads((ROOT / "results" / "s3_ceiling_estimation.json").read_text())["headline"]
    for i, (lab, name, skey) in enumerate((("mean", "temporal mean", "slope_uniform_mean"),
                                           ("occ_c0", "occupation above zero",
                                            "slope_uniform_occ_c0"))):
        # the quoted slope is fitted on the primary cell alone (make_numbers reads
        # headline["slope_uniform_*"], which s3 computes for PRIMARY_CID/known);
        # pooling the whole 2x2x2 factorial here would draw a line that does not
        # fit its own markers.
        sel = [r for r in s3 if r.get("label") == lab
               and r.get("config") == PRIMARY_CONFIG and r.get("arm") == "known"]
        x, y, _ = curve(sel, None, "uniform_err_mean")
        if not x.size:
            continue
        _, se = _se_from_sd(sel, None)
        ax.fill_between(x, y - se, y + se, color=PALETTE[i], alpha=0.18, lw=0)
        ax.plot(x, y, "o-", color=PALETTE[i], label=name)
        b = slopes[skey]                       # the same number make_numbers emits
        a = np.mean(np.log(y) - b * np.log(x))
        ax.plot(x, np.exp(a) * x ** b, "--", color=PALETTE[i], lw=0.9, alpha=0.85,
                label=f"fit, slope {b:.4f}")   # 4 dp, as the text quotes it
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("calibration trajectories $m$"); ax.set_ylabel(r"uniform error $\varepsilon_m$")
    ax.set_title("(a) value estimation", fontsize=8); ax.legend(fontsize=6.5)

    # (b) selection regret against the 2 eps_m envelope
    ax = axes[0, 1]
    x, y, se = _s4_clustered_curve(s4, "regret_exhaustive")
    xe, ye, _ = _s4_clustered_curve(s4, "bound_2eps")
    if x.size:
        ax.fill_between(x, np.maximum(y - se, 1e-6), y + se,
                        color=PALETTE[0], alpha=0.18, lw=0)
        ax.plot(x, np.maximum(y, 1e-6), "o-", color=PALETTE[0], label="realised regret")
    if xe.size:
        ax.plot(xe, ye, "k--", lw=1.0, label=r"$2\varepsilon_m$ envelope")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("calibration trajectories $m$"); ax.set_ylabel("selection regret")
    ax.set_title("(b) selection regret and error envelope", fontsize=8); ax.legend(fontsize=6.5)

    res = [r for r in s8 if r.get("label") == RES_LABEL]
    # (c) regret of committing to each fixed protocol class
    ax = axes[1, 0]
    for lv in LEVELS:
        x, y, se = curve(res, None, f"regret_fixed_L{lv}")
        if x.size:
            ax.fill_between(x, np.maximum(y - se, 1e-9), y + se,
                            color=PALETTE[lv - 1], alpha=0.15, lw=0)
            ax.plot(x, np.maximum(y, 1e-6), "o-", ms=3, color=PALETTE[lv - 1],
                    label=f"L{lv} {LEVEL_NAMES[lv]}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("calibration trajectories $m$"); ax.set_ylabel("true selection regret")
    ax.set_title("(c) selection regret by protocol class", fontsize=8); ax.legend(fontsize=6, ncol=2)

    # (d) uniform estimation error within each class
    ax = axes[1, 1]
    for lv in LEVELS:
        x, y, se = curve(res, None, f"eps_true_L{lv}")
        if x.size:
            ax.fill_between(x, np.maximum(y - se, 1e-9), y + se,
                            color=PALETTE[lv - 1], alpha=0.15, lw=0)
            ax.plot(x, y, "o-", ms=3, color=PALETTE[lv - 1], label=rf"$\varepsilon_{lv}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("calibration trajectories $m$"); ax.set_ylabel(r"uniform error $\varepsilon_\ell$")
    ax.set_title("(d) uniform error by protocol class", fontsize=8); ax.legend(fontsize=6, ncol=2)

    save_figure(fig, "fig_calibration")
    plt.close(fig)


main()
