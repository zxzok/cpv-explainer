"""Shared configuration, plotting style, and IO for all experiments.

Every experiment script writes
  * a CSV of raw per-replication numbers into ``results/``,
  * a JSON of headline numbers into ``results/`` (consumed by the manuscript),
  * one or more PDF/PNG figures into ``figures/``.

The manuscript never hard-codes a number: ``paper/numbers.tex`` is generated
from the JSON files by ``experiments/make_numbers.py``.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATA = ROOT / "data"
for _d in (RESULTS, FIGURES, DATA):
    _d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

SEED = 20260802


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "pdf.fonttype": 42,   # TrueType, not Type 3
        "ps.fonttype": 42,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": False,
    })
    return plt


# Colour-blind-safe qualitative palette (Okabe--Ito).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000"]

METHOD_STYLE = {
    "label_aware": dict(color=PALETTE[0], marker="o", ls="-", label="Target-aware (ours)"),
    "label_aware_greedy": dict(color=PALETTE[0], marker="o", ls="--",
                               label="Target-aware, greedy only"),
    "mutual_information": dict(color=PALETTE[1], marker="s", ls="-",
                               label="Latent-state mutual information"),
    "imse": dict(color=PALETTE[2], marker="^", ls="-", label="Integrated posterior var."),
    "linear_target": dict(color=PALETTE[6], marker="P", ls="-", label="Linear-target design"),
    "kernel_quadrature": dict(color=PALETTE[3], marker="v", ls="-", label="Noiseless kernel quadrature"),
    "uniform": dict(color=PALETTE[4], marker="D", ls=":", label="Uniform spacing"),
    "random": dict(color=PALETTE[5], marker="x", ls=":", label="Random spacing"),
    "same_time": dict(color="0.45", marker="+", ls=":", label="Same-time replication"),
    "exhaustive": dict(color="k", marker="*", ls="-", label="Exhaustive optimum"),
}


def save_figure(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    print(f"  [fig] figures/{name}.pdf")


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def _default(o):
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    if isinstance(o, np.bool_):
        return bool(o)          # before np.integer: np.bool_ is not an np.integer,
                                # but without this it falls through to str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _finite(o):
    """Map NaN/Inf to ``null``.

    ``json.dumps`` writes bare ``NaN`` and ``Infinity``, which Python reads back
    but which are not valid JSON: the results files ship with the paper and have
    to parse in any language.  A missing number is ``null``.
    """
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, np.floating):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def save_json(obj: dict, name: str) -> None:
    path = RESULTS / f"{name}.json"
    # round-trip through _default first so dataclasses/arrays become plain types,
    # then strip non-finite floats before the final, strictly valid dump
    plain = json.loads(json.dumps(obj, default=_default))
    path.write_text(json.dumps(_finite(plain), indent=2, allow_nan=False))
    print(f"  [json] results/{name}.json")


def load_json(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def save_csv(rows: list[dict], name: str) -> None:
    import csv
    if not rows:
        return
    path = RESULTS / f"{name}.csv"
    keys = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  [csv]  results/{name}.csv ({len(rows)} rows)")


def environment_record() -> dict:
    import scipy
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        print(f"[{self.label}] start")
        return self

    def __exit__(self, *exc):
        print(f"[{self.label}] done in {time.perf_counter() - self.t0:.1f}s")
