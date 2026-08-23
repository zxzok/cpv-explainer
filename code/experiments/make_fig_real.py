"""Combined figure for the two full-trajectory physiological tasks.

Panel (a) is the pooled baseline-adjusted Sleep-EDF budget curve, panel
(b) the atrial-fibrillation observed-fraction curve. Conditional paired
percentile ranges for fixed templates are computed in ``crossfit_real.py``.

Reads results/crossfit_real.json; writes figures/fig_real.pdf/.png.
"""

from __future__ import annotations

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

SLEEP_LABEL = "REM"
SLEEP_METHODS = [
    ("label_aware", "Target-aware", PALETTE[0], "o", "-"),
    ("kernel_quadrature", "Kernel quadrature (learned)", PALETTE[2], "^", "-."),
    ("uniform", "Uniformly dispersed", PALETTE[1], "s", "--"),
    ("consecutive", "Contiguous block", PALETTE[3], "v", ":"),
]
AF_METHODS = [("dispersed", "Dispersed windows", PALETTE[1], "s", "--"),
              ("contiguous", "Contiguous windows", PALETTE[3], "v", ":")]


def sleep_curves(d: dict) -> dict[str, tuple[list[float], list[float]]]:
    rows = d["sleep"][SLEEP_LABEL]
    out: dict[str, tuple[list, list]] = {}
    for key, *_ in SLEEP_METHODS:
        pts = sorted((int(k.split("N=")[1]), v["cross_fitted_r2"])
                     for k, v in rows.items() if k.startswith(key + "|"))
        if pts:
            out[key] = ([float(b) for b, _ in pts], [v for _, v in pts])
    return out


def af_curves(d: dict) -> dict[str, tuple[list[float], list[float]]]:
    out: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for key, v in d["af"].items():
        meth = key.split("|")[0]
        out[meth][0].append(100.0 * v["observed_fraction"])
        out[meth][1].append(v["cross_fitted_r2"])
    for k in out:
        order = np.argsort(out[k][0])
        out[k] = ([out[k][0][i] for i in order], [out[k][1][i] for i in order])
    return dict(out)


def main() -> None:
    d = json.load(open(ROOT / "results" / "crossfit_real.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))

    # ---- (a) Sleep-EDF: value against scored-epoch budget -----------------
    ax = axes[0]
    ax.axhline(0.0, color="0.4", lw=0.8, zorder=0)
    sc = sleep_curves(d)
    for key, lab, col, mk, ls in SLEEP_METHODS:
        # The caption names every series; dropping one silently is how a panel
        # ends up describing curves it does not draw.
        if key not in sc:
            raise SystemExit(f"no Sleep-EDF rows for {key!r} -- the caption names it")
        x, y = sc[key]
        ax.plot(x, y, color=col, marker=mk, ls=ls, label=lab, lw=1.3, ms=3.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"budget $N$ (scored 30 s epochs)")
    ax.set_ylabel(r"cross-fitted $R^2$")
    ax.set_ylim(-0.08, 1.02)
    ax.legend(loc="lower right", fontsize=6.0)
    ax.text(-0.16, 1.03, "(a)", transform=ax.transAxes, fontweight="bold")
    ax.set_title("Pooled baseline-adjusted analysis", fontsize=8.2)

    # ---- (b) LTAF: value against observed record fraction ----------------
    ax = axes[1]
    ax.axhline(0.0, color="0.4", lw=0.8, zorder=0)
    af = af_curves(d)
    for key, lab, col, mk, ls in AF_METHODS:
        if key not in af:
            raise SystemExit(f"no LTAF rows for {key!r} -- the caption names it")
        x, y = af[key]
        ax.plot(x, y, color=col, marker=mk, ls=ls, label=lab, lw=1.3, ms=3.6)
    ax.set_xscale("log", base=2)
    ticks = sorted({v for x, _ in af.values() for v in x})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:.2f}".rstrip("0").rstrip(".") for v in ticks],
                       fontsize=6.2)
    ax.set_xlabel("observed fraction of analysable record (%)")
    ax.set_ylabel(r"cross-fitted $R^2$")
    ax.set_ylim(-0.08, 1.02)                       # same range as panel (a)
    ax.legend(loc="lower right", fontsize=6.0)
    ax.text(-0.16, 1.03, "(b)", transform=ax.transAxes, fontweight="bold")
    ax.set_title("Atrial-fibrillation burden", fontsize=8.5)

    fig.tight_layout()
    save_figure(fig, "fig_real")
    print(f"  sleep methods: {sorted(sc)}")
    print(f"  af methods:    {sorted(af)}")


if __name__ == "__main__":
    main()
