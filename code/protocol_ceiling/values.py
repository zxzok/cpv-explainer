"""Protocol values: Bayes, best-linear, and achieved.

Three different numbers are attached to an acquisition protocol, and conflating
them is the easiest way to overstate a real-data result.

``I_B(S) = Var{E(Theta | Y_S)} / Var(Theta)``
    The *Bayes* protocol value: the population ``R^2`` of the best measurable
    predictor.  Available in closed form only under a distributional model; in
    this package that means the Gaussian temporal-aggregate model of
    :mod:`protocol_ceiling.risk`.

``I_L(S) = c_S^T Sigma_S^+ c_S / Var(Theta)``
    The *best-linear* protocol value, with ``Sigma_S = Var(Y_S)`` and
    ``c_S = Cov(Y_S, Theta)``.  Estimable from second moments alone, for an
    arbitrary square-integrable target, with no distributional assumption.

``R^2_model(S)``
    The *achieved* out-of-sample value of a fitted predictor: an empirical lower
    bound on ``I_B(S)``.

They are ordered ``I_L(S) <= I_B(S) <= 1`` for every square-integrable target,
with equality in the first inequality exactly when ``E(Theta | Y_S)`` lies in
the closed affine span of ``Y_S`` -- in particular under joint Gaussianity.
Reporting an empirical second-moment quantity as a "Bayes ceiling" is therefore
correct only in the Gaussian case, and this module exists to keep the three
apart by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ProtocolValues:
    """The three quantities, reported together."""

    linear: float
    bayes: float | None = None
    achieved: float | None = None
    n_objects: int = 0

    def as_dict(self) -> dict:
        return {
            "best_linear_value": self.linear,
            "model_based_bayes_value": self.bayes,
            "achieved_r2": self.achieved,
            "n_objects": self.n_objects,
        }

    def check_ordering(self, tol: float = 1e-9) -> bool:
        """``I_L <= I_B <= 1`` and ``R^2_model <= I_B`` in population."""
        ok = 0.0 - tol <= self.linear <= 1.0 + tol
        if self.bayes is not None:
            ok = ok and (self.linear <= self.bayes + tol) and (self.bayes <= 1.0 + tol)
        return bool(ok)


def best_linear_value_from_moments(Sigma_S: Array, c_S: Array, var_theta: float,
                                   rcond: float = 1e-10) -> float:
    """``c_S^T Sigma_S^+ c_S / Var(Theta)`` using the Moore--Penrose inverse.

    The pseudo-inverse rather than an inverse: an observation set can be
    rank-deficient (repeated windows, or windows that coincide after gridding),
    and the best-linear predictor is still well defined on the column space.
    """
    Sigma_S = np.atleast_2d(np.asarray(Sigma_S, dtype=float))
    c_S = np.asarray(c_S, dtype=float).ravel()
    if var_theta <= 0 or c_S.size == 0:
        return 0.0
    explained = float(c_S @ np.linalg.pinv(Sigma_S, rcond=rcond) @ c_S)
    return float(np.clip(explained / var_theta, 0.0, 1.0))


def best_linear_value(Y: Array, theta: Array, center: bool = True) -> float:
    """Empirical best-linear protocol value from paired observations.

    ``Y`` has shape ``(n, d)`` -- one row per object, one column per action --
    and ``theta`` has length ``n``.  This is a purely second-moment quantity: it
    requires no model for the latent process and no Gaussian assumption, and it
    is what a real-data analysis can honestly report when the conditional mean
    is not known to be affine.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    theta = np.asarray(theta, dtype=float).ravel()
    if Y.shape[0] != theta.size:
        raise ValueError("Y and theta must have the same number of objects")
    if center:
        Y = Y - Y.mean(axis=0, keepdims=True)
        theta = theta - theta.mean()
    n = max(Y.shape[0] - 1, 1)
    return best_linear_value_from_moments(Y.T @ Y / n, Y.T @ theta / n,
                                          float(theta @ theta / n))


def gaussian_bayes_value(label, K: Array, grid, actions) -> float:
    """Model-based Gaussian Bayes value; a thin alias for the closed form.

    Kept here so that a caller choosing between the two quantities has to name
    which one it wants.
    """
    from .risk import evaluate_protocol

    return float(evaluate_protocol(label, K, grid, actions).ceiling)


def linear_value_of_protocol(K: Array, A: Array, R: Array, h: Array) -> float:
    """Best-linear value of a linear target ``Theta = h^T Z`` under ``(A, R)``.

    For a linear target and linear observations the best-linear and Bayes values
    coincide when ``(Z, Y_S)`` is jointly Gaussian; outside that case this is
    still exactly the best-linear value, because both are determined by second
    moments.
    """
    A = np.atleast_2d(A)
    KA = K @ A.T
    Sigma = A @ KA + R
    return best_linear_value_from_moments(Sigma, KA.T @ h, float(h @ K @ h))


# --------------------------------------------------------------------------
# Empirical best-linear design directly on the acquired variables
# --------------------------------------------------------------------------
@dataclass
class MomentState:
    """Sequential-regression state for best-linear selection on ``(Sigma, c, v)``.

    ``Sigma`` is the covariance of the *acquired* variables, ``c`` their
    covariance with the target and ``v`` the target variance.  No latent
    process, no measurement-error model and no transform are involved: the
    selection criterion is the population ``R^2`` of the best linear predictor
    of the target from the selected coordinates, which is what
    :func:`best_linear_value_from_moments` evaluates in closed form.

    The state carries the residual covariance ``R`` of every coordinate given
    the chosen ones and the residual covariance ``d`` with the target, so the
    Schur-complement increment of a candidate is ``d_a^2 / (v R_aa)`` and the
    update after choosing ``a`` is rank one in both.
    """

    R: Array
    d: Array
    v: float
    value: float
    chosen: tuple[int, ...]

    @classmethod
    def empty(cls, Sigma: Array, c: Array, v: float) -> "MomentState":
        return cls(R=np.array(Sigma, dtype=float, copy=True),
                   d=np.asarray(c, dtype=float).ravel().copy(),
                   v=float(v), value=0.0, chosen=())

    def gain(self, a: int, ridge: float = 1e-12) -> float:
        s = float(self.R[a, a])
        if s <= ridge or self.v <= 0:
            return 0.0
        return float(self.d[a] ** 2 / (self.v * s))

    def add(self, a: int, ridge: float = 1e-12) -> "MomentState":
        s = float(self.R[a, a])
        if s <= ridge:
            return MomentState(self.R, self.d, self.v, self.value, self.chosen + (a,))
        col = self.R[:, a].copy()
        g = float(self.d[a] ** 2 / (self.v * s)) if self.v > 0 else 0.0
        return MomentState(R=self.R - np.outer(col, col) / s,
                           d=self.d - col * (self.d[a] / s),
                           v=self.v, value=self.value + g,
                           chosen=self.chosen + (a,))


def best_linear_greedy(Sigma: Array, c: Array, v: float, budget: int,
                       forbidden: Sequence[int] = (),
                       max_swap_rounds: int = 0) -> list[int]:
    """Forward selection on ``I_L``, optionally refined by one-swap.

    Returns the selected column indices.  ``max_swap_rounds`` bounds the number
    of *accepted* swaps, matching :func:`protocol_ceiling.design.swap_local_search`.
    """
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    p = Sigma.shape[0]
    banned = set(int(i) for i in forbidden)
    state = MomentState.empty(Sigma, c, v)
    chosen: list[int] = []
    for _ in range(int(budget)):
        best, best_gain = -1, 0.0
        for a in range(p):
            if a in banned or a in chosen:
                continue
            g = state.gain(a)
            if g > best_gain:
                best, best_gain = a, g
        if best < 0:
            break
        state = state.add(best)
        chosen.append(best)
    if not max_swap_rounds or not chosen:
        return sorted(chosen)

    def value_of(idx: Sequence[int]) -> float:
        idx = list(idx)
        if not idx:
            return 0.0
        return best_linear_value_from_moments(
            Sigma[np.ix_(idx, idx)], np.asarray(c, dtype=float).ravel()[idx], v)

    cur, best_val = list(chosen), value_of(chosen)
    pool = [a for a in range(p) if a not in banned]
    for _ in range(int(max_swap_rounds)):
        improved = False
        for i in range(len(cur)):
            for a in pool:
                if a in cur:
                    continue
                trial = cur[:i] + [a] + cur[i + 1:]
                val = value_of(trial)
                if val > best_val + 1e-14:
                    cur, best_val, improved = trial, val, True
                    break
            if improved:
                break
        if not improved:
            break
    return sorted(cur)
