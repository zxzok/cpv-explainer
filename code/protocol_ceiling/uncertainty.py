"""Subject-level bootstrap uncertainty for protocol ceilings.

The theory gives high-probability uniform error bounds; in practice the useful
object is an interval.  Because objects (subjects, recordings, patients) are the
independent replication unit, every resample here is taken at the *object*
level: resampling time points inside an object would destroy exactly the
temporal dependence the ceiling depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .covariance import Action, TimeGrid
from .estimation import estimate_protocol_ceiling, fit_covariance
from .transforms import LabelFunctional

Array = NDArray[np.float64]


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lower: float
    upper: float
    replicates: Array
    level: float

    def as_dict(self) -> dict:
        return {
            "ceiling": self.point,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "level": self.level,
            "n_bootstrap": int(self.replicates.size),
            "bootstrap_sd": float(np.std(self.replicates, ddof=1))
            if self.replicates.size > 1 else float("nan"),
        }


def bootstrap_covariances(
    W: Array,
    n_bootstrap: int = 200,
    noise_var: Array | float | None = None,
    shrinkage: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[Array]:
    """Object-level bootstrap replicates of the standardised covariance."""
    rng = np.random.default_rng(0) if rng is None else rng
    W = np.asarray(W, dtype=float)
    m = W.shape[0]
    out = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, m, size=m)
        out.append(fit_covariance(W[idx], noise_var=noise_var,
                                  shrinkage=shrinkage).K)
    return out


def bootstrap_protocol_ceiling(
    label: LabelFunctional,
    W: Array,
    grid: TimeGrid,
    actions: Sequence[Action],
    n_bootstrap: int = 200,
    level: float = 0.95,
    noise_var: Array | float | None = None,
    shrinkage: float = 0.0,
    rng: np.random.Generator | None = None,
    covariances: Sequence[Array] | None = None,
) -> BootstrapResult:
    """Percentile bootstrap interval for ``I_g(S)``."""
    rng = np.random.default_rng(0) if rng is None else rng
    K_hat = fit_covariance(W, noise_var=noise_var, shrinkage=shrinkage).K
    point = estimate_protocol_ceiling(label, K_hat, grid, actions)
    Ks = (list(covariances) if covariances is not None
          else bootstrap_covariances(W, n_bootstrap, noise_var, shrinkage, rng))
    reps = np.array([estimate_protocol_ceiling(label, Kb, grid, actions) for Kb in Ks])
    a = (1.0 - level) / 2.0
    return BootstrapResult(point=float(point),
                           lower=float(np.quantile(reps, a)),
                           upper=float(np.quantile(reps, 1.0 - a)),
                           replicates=reps, level=level)


def lower_confidence_bound(reps: Array, quantile: float = 0.1) -> float:
    """Scalar LCB used as the objective of the robust design algorithm."""
    return float(np.quantile(np.asarray(reps, dtype=float), quantile))


def coverage(intervals: Sequence[tuple[float, float]], truth: float) -> float:
    """Empirical coverage of a family of intervals for a fixed truth."""
    hits = sum(1 for lo, hi in intervals if lo <= truth <= hi)
    return float(hits / max(len(intervals), 1))
