"""Calibration-based estimation of protocol ceilings and its error analysis.

A benchmark collected under one snapshot protocol identifies the joint law of
``(Y_S, Theta)`` and hence *its own* Bayes risk, but generally not the ceiling
of a counterfactual protocol (see :mod:`protocol_ceiling.identifiability`).
What restores estimability is a small densely sampled *calibration* sample of
``m`` objects whose latent trajectories are observed on the full grid,

    W^(i) = Z^(i) + eta^(i),   eta^(i) ~ N(0, R_0),   i = 1..m.

The plug-in pipeline is

    Sigma_hat  = (1/m) sum_i W^(i) W^(i)T          (subject-level moments)
    K_tilde    = Pi_PSD(Sigma_hat - R_0_hat)       (deconvolve, repair)
    K_hat      = to_correlation(K_tilde)           (standardise)
    I_hat(S)   = F_g(S; K_hat) / V_g(K_hat)        (plug in, for every S)

and `thm:uniform-error` of the paper controls ``sup_{S in Pi_B} |I_hat(S) - I(S)|`` by
``C * ||K_hat - K||_op^beta`` with ``beta = 1`` for smooth labels and
``beta = 1/2`` for threshold labels.  The constants are *computed*, not merely
asserted: :func:`uniform_error_bound` returns the same expression that the
proof produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .covariance import (Action, TimeGrid, project_psd, protocol_matrices,
                         shrink, to_correlation)
from .risk import label_variance, protocol_ceiling
from .transforms import LabelFunctional

Array = NDArray[np.float64]


# --------------------------------------------------------------------------
# Covariance estimation from dense calibration trajectories
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CovarianceFit:
    K: Array                 # standardised (unit-diagonal) estimate
    raw: Array               # deconvolved, PSD-projected, unstandardised
    noise_var: Array         # estimated measurement-noise variances
    n_subjects: int
    effective_rank: float

    def as_dict(self) -> dict:
        return {
            "n_subjects": self.n_subjects,
            "effective_rank": self.effective_rank,
            "mean_noise_var": float(np.mean(self.noise_var)),
        }


def effective_rank(K: Array) -> float:
    """``r_eff(K) = tr(K) / ||K||_op``, the quantity in the concentration rate."""
    op = float(np.linalg.norm(K, 2))
    return float(np.trace(K) / op) if op > 0 else 0.0


def estimate_noise_from_replicates(replicates: Array) -> Array:
    """Per-time measurement-noise variance from within-support replicates.

    ``replicates`` has shape ``(m, p, n_rep)``; the within-support sample
    variance is an unbiased estimate of ``nu^2`` because repeated segments on a
    fixed temporal support share the same latent state.
    """
    if replicates.ndim != 3 or replicates.shape[2] < 2:
        raise ValueError("replicates must have shape (m, p, n_rep) with n_rep >= 2")
    return np.asarray(replicates.var(axis=2, ddof=1).mean(axis=0), dtype=float)


def fit_covariance(
    W: Array,
    noise_var: Array | float | None = None,
    shrinkage: float = 0.0,
    center: bool = True,
    eig_floor: float | None = None,
) -> CovarianceFit:
    """Estimate the standardised latent correlation ``K`` from calibration data.

    Parameters
    ----------
    W : ``(m, p)`` array of densely observed (possibly noisy) trajectories.
    noise_var : known or separately estimated measurement-noise variance,
        either a scalar or a length-``p`` vector.  ``None`` means noise-free.
    shrinkage : linear shrinkage intensity towards the identity, applied before
        the PSD projection; stabilises the small-``m`` regime.
    eig_floor : relative eigenvalue floor for the PSD projection, as a multiple
        of the mean diagonal.  ``None`` uses ``1/m``, which keeps the estimator
        defined for every sample while vanishing with the calibration size.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be a 2-d array of shape (m, p)")
    m, p = W.shape
    Wc = W - W.mean(axis=0, keepdims=True) if center else W
    denom = max(m - 1, 1) if center else m
    Sigma = (Wc.T @ Wc) / denom

    if noise_var is None:
        nu = np.zeros(p)
    elif np.isscalar(noise_var):
        nu = np.full(p, float(noise_var))
    else:
        nu = np.asarray(noise_var, dtype=float)
        if nu.shape != (p,):
            raise ValueError("noise_var must be scalar or of length p")

    raw = Sigma - np.diag(nu)
    raw = shrink(raw, target=np.eye(p) * max(np.mean(np.diag(raw)), 1e-12),
                 intensity=shrinkage)
    # Projecting onto the PSD cone with floor 0 can return a matrix with zero
    # diagonal entries, and `to_correlation` is then undefined.  Project onto
    # {M : M >= tau I} instead, so every diagonal entry is at least tau and the
    # standardisation is well defined on every sample.  The default
    # tau = 1/m vanishes with the calibration size, and the perturbation
    # analysis is stated on the event where the floor is inactive.
    tau = float(eig_floor) if eig_floor is not None else 1.0 / max(m, 1)
    scale = max(np.mean(np.diag(raw)), 1e-12)
    raw = project_psd(raw, floor=tau * scale)
    K = to_correlation(raw)
    return CovarianceFit(K=K, raw=raw, noise_var=nu, n_subjects=m,
                         effective_rank=effective_rank(K))


# --------------------------------------------------------------------------
# Plug-in ceiling estimation
# --------------------------------------------------------------------------
def estimate_protocol_ceiling(label: LabelFunctional, K_hat: Array, grid: TimeGrid,
                              actions: Sequence[Action]) -> float:
    """Plug-in estimate ``I_hat_g(S)`` for one candidate protocol."""
    A, R = protocol_matrices(actions, grid)
    return protocol_ceiling(label, K_hat, A, R, grid.weights)


def estimate_ceiling_family(label: LabelFunctional, K_hat: Array, grid: TimeGrid,
                            protocols: Sequence[Sequence[Action]]) -> Array:
    """Vector of plug-in ceilings over a family of candidate protocols."""
    return np.array([estimate_protocol_ceiling(label, K_hat, grid, S) for S in protocols])


# --------------------------------------------------------------------------
# The constants appearing in the finite-sample bound
# --------------------------------------------------------------------------
def q_perturbation_constant(kappa: float, a_norm_sq: float, lambda0: float) -> float:
    """``L_Q`` in ``||Q_S(K_hat) - Q_S(K)||_op <= L_Q ||K_hat - K||_op``.

    Derived in the paper by splitting

        Q_hat - Q = E A^T M_hat^{-1} A K_hat + K A^T M_hat^{-1} A E
                    + K A^T (M_hat^{-1} - M^{-1}) A K

    and using ``lambda_min(M_hat) >= lambda0 / 2`` in the small-perturbation
    regime ``||A E A^T|| <= lambda0 / 2``.  Here ``kappa`` bounds the operator
    norm of *both* ``K`` and ``K_hat`` (both are correlation matrices),
    ``a_norm_sq >= ||A_S||_op^2`` (bounded by the budget ``B`` because every
    action row is a weighted average with unit L1 norm, so ||A||_F^2 <= D) and
    ``lambda0 <= lambda_min(A_S K A_S^T + R_S)`` (bounded below by the smallest
    per-action measurement-noise variance, since A K A^T is PSD).
    """
    return float(4.0 * kappa * a_norm_sq / lambda0
                 + 2.0 * kappa**2 * a_norm_sq**2 / lambda0**2)


def uniform_error_bound(label: LabelFunctional, K: Array, omega: Array,
                        budget: int, min_noise: float, kappa: float | None = None,
                        k_error: float = 1.0) -> dict:
    """Computable version of the uniform bound of `thm:uniform-error`.

    Returns ``L_Q``, the label modulus ``(L_g, beta)``, and the resulting bound
    on ``sup_{S in Pi_B} |I_hat(S) - I(S)|`` as a function of the covariance
    error ``k_error = ||K_hat - K||_op``.
    """
    # kappa must bound BOTH ||K|| and ||K_hat||; the default bounds only ||K||,
    # which is correct exactly when the caller has already checked that the
    # estimate is no larger in operator norm.  Callers with K_hat in hand should
    # pass max(||K||, ||K_hat||).
    kappa = float(np.linalg.norm(K, 2)) if kappa is None else float(kappa)
    a_norm_sq = float(budget)          # ||A_S||_op^2 <= D <= B
    lambda0 = float(min_noise)         # lambda_min(A K A^T + R) >= min nu^2
    if not (lambda0 > 0):
        raise ValueError("min_noise must be positive: the bound divides by it")
    L_Q = q_perturbation_constant(kappa, a_norm_sq, lambda0)
    L_g, beta = label.modulus()
    V = label_variance(label, K, omega)
    # |F_hat - F| <= L_g * ||Q_hat - Q||^beta ; |V_hat - V| <= L_g * ||K_hat - K||^beta
    dF = L_g * (L_Q * k_error) ** beta
    dV = L_g * k_error**beta
    # I = F/V with F <= V, so |I_hat - I| <= (dF + dV) / (V - dV), and only when
    # dV < V.  Off that event the theorem asserts nothing, and a clamped
    # denominator would return a finite number that is not a bound -- so say so.
    if dV >= V:
        return {
            "L_Q": L_Q, "L_g": float(L_g), "beta": float(beta),
            "label_variance": float(V), "bound": float("inf"),
            "hypothesis_holds": False,
        }
    return {
        "L_Q": L_Q,
        "L_g": float(L_g),
        "beta": float(beta),
        "label_variance": float(V),
        "bound": float((dF + dV) / (V - dV)),
        "hypothesis_holds": True,
    }


def selection_regret_bound(eps: float, eta: float = 1.0,
                           i_star: float | None = None) -> float:
    """Protocol-selection regret guarantee of `cor:regret`.

    Exact maximisation (``eta = 1``) gives ``I(S*) - I(S_hat) <= 2 eps``.  An
    ``eta``-approximate maximiser of the *estimated* objective gives
    ``I(S_hat) >= eta I(S*) - (1 + eta) eps``, i.e. a regret of at most
    ``(1 - eta) I(S*) + (1 + eta) eps``.  The first term does not vanish with
    ``eps``, so it cannot be dropped: ``i_star`` must be supplied whenever
    ``eta < 1``, and the returned value is the full bound.
    """
    if not 0.0 < eta <= 1.0:
        raise ValueError("eta must lie in (0, 1]")
    if eta == 1.0:
        return float(2.0 * eps)
    if i_star is None:
        raise ValueError(
            "eta < 1 needs i_star: the regret bound carries a (1 - eta) I(S*) "
            "term that does not shrink with the calibration sample")
    return float((1.0 - eta) * i_star + (1.0 + eta) * eps)


# --------------------------------------------------------------------------
# Partial identification of the trait share
# --------------------------------------------------------------------------
def trait_share_interval(r_lag: float, rho_bound: float) -> tuple[float, float]:
    """Interval for ``alpha`` from a long-lag correlation ``r_L``.

    If ``r_L = alpha + (1 - alpha) rho(L)`` and only ``0 <= rho(L) <= delta`` is
    known, then ``alpha`` is decreasing in ``rho(L)`` and

        max{0, (r_L - delta) / (1 - delta)} <= alpha <= r_L.
    """
    if not 0.0 <= rho_bound < 1.0:
        raise ValueError("rho_bound must lie in [0, 1)")
    lo = max(0.0, (r_lag - rho_bound) / (1.0 - rho_bound))
    return float(lo), float(min(max(r_lag, lo), 1.0))


def trait_ceiling(alpha: float, D: int, M: int, sigma_eps_sq: float) -> float:
    """``I_trait(D, M) = alpha / {alpha + (1-alpha)/D + sigma^2/(D M)}``.

    Monotone increasing in ``alpha``, so a partial-identification interval for
    ``alpha`` maps to an interval for the trait ceiling.
    """
    denom = alpha + (1.0 - alpha) / D + sigma_eps_sq / (D * M)
    return float(alpha / denom) if denom > 0 else 0.0


def trait_ceiling_interval(alpha_interval: tuple[float, float], D: int, M: int,
                           sigma_eps_sq: float) -> tuple[float, float]:
    lo, hi = alpha_interval
    return trait_ceiling(lo, D, M, sigma_eps_sq), trait_ceiling(hi, D, M, sigma_eps_sq)
