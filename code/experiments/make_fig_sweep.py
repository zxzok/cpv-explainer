"""Selection-sample size against fine-placement resolution (Figure 4).

This figure asks how support-selection sample size affects fine-placement
resolution within the Sleep outer-training folds. It reports the cross-fitted
advantage of target-aware placement over each comparison protocol as a function
of how many training subjects were used to select the support; these subsets are
not separate external calibration databases.

Panel (a) is the advantage against the strongest learned baseline (kernel
quadrature, which also consumes the estimated covariance) and against the fixed
uniformly dispersed schedule. Panel (b) is the study-stratified selection-aware
repeated-subsample distribution and marks the original-sample statistic
separately, so a shifted resampling centre cannot be read as the effect size.

Reads results/calibration_sweep.json; writes figures/fig_sweep.pdf/.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import PALETTE, save_figure, setup_matplotlib  # noqa: E402

setup_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    d = json.load(open(ROOT / "results" / "calibration_sweep.json"))
    sweep = d["sweep"]
    subsample = d["repeated_subsampling"]
    original = d["original_sample"]
    keys = sorted(sweep, key=lambda k: sweep[k]["m_train_subjects"])
    m = np.array([sweep[k]["m_train_subjects"] for k in keys], float)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))

    # ---- (a) advantage against each baseline, versus selection size --------
    ax = axes[0]
    ax.axhline(0.0, color="0.4", lw=0.9, ls="-", zorder=1)
    for field, lab, col, mk in (
            ("delta_vs_kq", "vs. kernel quadrature (learned)", PALETTE[0], "o"),
            ("delta_vs_uniform", "vs. uniformly dispersed (fixed)", PALETTE[1], "s")):
        y = np.array([sweep[k][field] for k in keys], float)
        se = np.array([sweep[k].get(f"{field}_se", 0.0) for k in keys], float)
        ax.fill_between(m, y - se, y + se, color=col, alpha=0.18, lw=0)
        ax.plot(m, y, color=col, marker=mk, lw=1.3, ms=3.8, label=lab)
    ax.set_xlabel("training subjects used to select the support")
    ax.set_ylabel(r"cross-fitted $R^2$ advantage")
    ax.legend(loc="best", fontsize=6.2)
    ax.text(-0.18, 1.03, "(a)", transform=ax.transAxes, fontweight="bold")
    ax.set_title("Target-aware advantage", fontsize=8.5)

    # ---- (b) selection-aware repeated subsampling -------------------------
    ax = axes[1]
    ax.axvline(0.0, color="0.4", lw=0.9, zorder=1)
    ypos, labels = [], []
    for i, (field, lab, col) in enumerate((
            ("delta_vs_kq", "vs. kernel\nquadrature", PALETTE[0]),
            ("delta_vs_uniform", "vs. uniformly\ndispersed", PALETTE[1]))):
        b = subsample[field]
        ax.plot([b["p025"], b["p975"]], [i, i], color=col, lw=2.6,
                solid_capstyle="butt")
        ax.plot([b["median"]], [i], color=col, marker="D", ms=5.0,
                label="subsample median" if i == 0 else None)
        ax.plot([original[field]], [i], color="black", marker="|", ms=11,
                mew=1.5, label="original sample" if i == 0 else None)
        ypos.append(i); labels.append(lab)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_ylim(-0.6, len(ypos) - 0.4)
    ax.set_xlabel(r"cross-fitted $R^2$ advantage")
    ax.text(-0.28, 1.03, "(b)", transform=ax.transAxes, fontweight="bold")
    ax.set_title(
        f"Study-stratified 80% subsamples ({subsample['delta_vs_kq']['n']})",
        fontsize=8.5)
    ax.legend(loc="lower right", fontsize=5.9, frameon=False)

    fig.tight_layout()
    save_figure(fig, "fig_sweep")
    print("  [fig] figures/fig_sweep.pdf")
    for k in keys:
        print(f"    m={sweep[k]['m_train_subjects']:3d}  "
              f"delta_kq {sweep[k]['delta_vs_kq']:+.3f}  "
              f"delta_uniform {sweep[k]['delta_vs_uniform']:+.3f}")


if __name__ == "__main__":
    main()
