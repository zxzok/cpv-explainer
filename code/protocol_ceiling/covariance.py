"""Latent-process covariances, observation actions, and protocol matrices.

The discrete computational model used throughout the package is

    Z_i = (Z_i(t_1), ..., Z_i(t_p))^T ~ N(0, K),   diag(K) = 1,
    Theta_{i,g} = sum_j omega_j g(Z_ij),           omega >= 0, sum omega = 1,
    Y_{i,S}     = A_S Z_i + eps_{i,S},             eps ~ N(0, R_S).

A candidate *action* ``a = (t_a, w_a, M_a, nu_a^2, c_a)`` contributes one row
``ell_a`` to ``A_S`` (a normalised window average over the grid) and one
diagonal entry ``nu_a^2 / M_a`` to ``R_S``: ``M_a`` repeated segments inside a
fixed temporal support are exactly equivalent to dividing the measurement-noise
variance, which is why the design problem can trade "new temporal support"
against "less noise on support already held".
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Kernel = Callable[[Array], Array]


# --------------------------------------------------------------------------
# Stationary correlation kernels
# --------------------------------------------------------------------------
def ou(tau: float = 1.0) -> Kernel:
    """Ornstein--Uhlenbeck / exponential kernel with integral time ``tau``."""

    def k(u: Array) -> Array:
        return np.exp(-np.abs(u) / tau)

    return k


def matern32(tau: float = 1.0) -> Kernel:
    """Matern-3/2 parameterised so that ``int_0^inf rho = tau``."""
    a = 2.0 / tau

    def k(u: Array) -> Array:
        z = a * np.abs(u)
        return (1.0 + z) * np.exp(-z)

    return k


def matern52(tau: float = 1.0) -> Kernel:
    """Matern-5/2 parameterised so that ``int_0^inf rho = tau``."""
    # int_0^inf (1 + z + z^2/3) e^{-z} dz = 1 + 1 + 2/3 = 8/3, in units of 1/a.
    a = (8.0 / 3.0) / tau

    def k(u: Array) -> Array:
        z = a * np.abs(u)
        return (1.0 + z + z * z / 3.0) * np.exp(-z)

    return k


def squared_exponential(tau: float = 1.0) -> Kernel:
    """Squared-exponential kernel with integral time ``tau``."""
    ell = tau * np.sqrt(2.0 / np.pi)

    def k(u: Array) -> Array:
        return np.exp(-0.5 * (np.asarray(u) / ell) ** 2)

    return k


def damped_periodic(tau: float = 1.0, period: float = 2.0) -> Kernel:
    """Oscillatory kernel ``e^{-|u|/tau} cos(2 pi u / period)``.

    Produces *negative* correlations, exercising the parts of the theory that
    do not assume ``rho >= 0``.
    """

    def k(u: Array) -> Array:
        u = np.asarray(u, dtype=float)
        return np.exp(-np.abs(u) / tau) * np.cos(2.0 * np.pi * u / period)

    return k


def cauchy(tau: float = 1.0, beta: float = 1.0) -> Kernel:
    """Long-memory Cauchy kernel ``(1 + (u/tau)^2)^{-beta/2}``.

    For ``beta <= 1`` the kernel is not integrable, so ``tau_1 = infinity`` and
    the ``O(T^{-1})`` state term of the short-memory theory fails; this family
    is used in the misspecification experiments.
    """

    def k(u: Array) -> Array:
        return (1.0 + (np.asarray(u, dtype=float) / tau) ** 2) ** (-beta / 2.0)

    return k


def mixture(kernels: Sequence[Kernel], weights: Sequence[float]) -> Kernel:
    """Convex mixture of kernels; models multi-timescale dynamics."""
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("mixture weights must be non-negative and sum to one")

    def k(u: Array) -> Array:
        return sum(wi * ki(u) for wi, ki in zip(w, kernels))

    return k


def two_scale_ou(tau_fast: float = 0.2, tau_slow: float = 3.0, w_fast: float = 0.5) -> Kernel:
    return mixture([ou(tau_fast), ou(tau_slow)], [w_fast, 1.0 - w_fast])


def spectral_mixture(taus: Sequence[float], periods: Sequence[float],
                     weights: Sequence[float]) -> Kernel:
    comps = [damped_periodic(t, p) for t, p in zip(taus, periods)]
    return mixture(comps, weights)


KERNELS: dict[str, Callable[..., Kernel]] = {
    "ou": ou,
    "matern32": matern32,
    "matern52": matern52,
    "se": squared_exponential,
    "periodic": damped_periodic,
    "cauchy": cauchy,
    "two_scale_ou": two_scale_ou,
}


def make_kernel(name: str, **kwargs) -> Kernel:
    if name not in KERNELS:
        raise ValueError(f"unknown kernel {name!r}; available: {sorted(KERNELS)}")
    return KERNELS[name](**kwargs)


# --------------------------------------------------------------------------
# Time grids and trait--state covariances
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TimeGrid:
    """Discretised horizon ``[0, T]`` with label weights ``omega``."""

    times: Array
    weights: Array
    horizon: float

    @property
    def p(self) -> int:
        return int(self.times.size)

    @property
    def dt(self) -> float:
        return float(self.times[1] - self.times[0]) if self.p > 1 else float(self.horizon)


def uniform_grid(horizon: float, p: int, weight_fn: Callable[[Array], Array] | None = None) -> TimeGrid:
    """Midpoint grid on ``[0, T]`` with uniform (or user-weighted) label weights.

    Midpoints rather than endpoints are used so that ``sum_j omega_j g(Z_j)``
    is the midpoint quadrature of ``T^{-1} int_0^T g{Z(t)} dt`` and inherits its
    ``O(p^{-2})`` accuracy.
    """
    if p < 2:
        raise ValueError("p must be at least 2")
    edges = np.linspace(0.0, horizon, p + 1)
    times = 0.5 * (edges[:-1] + edges[1:])
    if weight_fn is None:
        w = np.full(p, 1.0 / p)
    else:
        w = np.asarray(weight_fn(times), dtype=float)
        if np.any(w < 0):
            raise ValueError("label weights must be non-negative")
        w = w / w.sum()
    return TimeGrid(times=times, weights=w, horizon=float(horizon))


def recency_weight(half_life: float) -> Callable[[Array], Array]:
    """Retrospective-scale weighting: recent states count more."""

    def wf(t: Array) -> Array:
        t = np.asarray(t, dtype=float)
        return np.exp(-(t.max() - t) / half_life)

    return wf


def trait_state_correlation(grid: TimeGrid, alpha: float, rho: Kernel) -> Array:
    """Correlation matrix of ``Z(t) = sqrt(alpha) M + sqrt(1-alpha) X(t)``."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    lag = grid.times[:, None] - grid.times[None, :]
    K = alpha + (1.0 - alpha) * np.asarray(rho(lag), dtype=float)
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 1.0)
    return K


def kernel_matrix(grid: TimeGrid, rho: Kernel) -> Array:
    return trait_state_correlation(grid, 0.0, rho)


# --------------------------------------------------------------------------
# Observation actions and protocol matrices
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    """One candidate observation.

    Attributes
    ----------
    time : centre of the observation window.
    width : window length ``w_a``; ``0`` gives a point observation.
    n_segments : ``M_a`` repeated segments inside the same support.
    noise : raw per-segment measurement-noise variance ``nu_a^2``.
    cost : acquisition cost ``c_a``.
    """

    time: float
    width: float = 0.0
    n_segments: int = 1
    noise: float = 1.0
    cost: float = 1.0
    tag: str = ""

    @property
    def effective_noise(self) -> float:
        """``nu_a^2 / M_a``: repeated segments only sharpen a fixed support."""
        return float(self.noise) / float(self.n_segments)

    def with_segments(self, m: int) -> "Action":
        return replace(self, n_segments=int(m))


def action_vector(action: Action, grid: TimeGrid) -> Array:
    """Row ``ell_a`` of ``A_S``: the normalised window average on the grid."""
    if action.width <= 0.0:
        ell = np.zeros(grid.p)
        ell[int(np.argmin(np.abs(grid.times - action.time)))] = 1.0
        return ell
    lo = action.time - 0.5 * action.width
    hi = action.time + 0.5 * action.width
    mask = (grid.times >= lo) & (grid.times <= hi)
    if not mask.any():  # window narrower than the grid spacing
        mask[int(np.argmin(np.abs(grid.times - action.time)))] = True
    ell = mask.astype(float)
    return ell / ell.sum()


def protocol_matrices(actions: Sequence[Action], grid: TimeGrid) -> tuple[Array, Array]:
    """Return ``(A_S, R_S)`` for a protocol given as a sequence of actions."""
    if len(actions) == 0:
        return np.zeros((0, grid.p)), np.zeros((0, 0))
    A = np.stack([action_vector(a, grid) for a in actions])
    R = np.diag([a.effective_noise for a in actions])
    return A, R


def candidate_actions(
    grid: TimeGrid,
    n_times: int = 32,
    widths: Sequence[float] = (0.0,),
    segments: Sequence[int] = (1,),
    noise: float = 1.0,
    cost_fixed: float = 1.0,
    cost_per_time: float = 0.0,
    cost_per_segment: float = 0.0,
) -> list[Action]:
    """Build the candidate set ``V`` on which the design problem is solved."""
    centres = np.linspace(grid.times[0], grid.times[-1], n_times)
    out: list[Action] = []
    for t in centres:
        for w in widths:
            for m in segments:
                cost = cost_fixed + cost_per_time * w + cost_per_segment * (m - 1)
                out.append(
                    Action(time=float(t), width=float(w), n_segments=int(m),
                           noise=float(noise), cost=float(cost),
                           tag=f"t={t:.3f},w={w:g},M={m}")
                )
    return out


def same_time_protocol(grid: TimeGrid, n_total: int, noise: float = 1.0,
                       width: float = 0.0, time: float | None = None) -> list[Action]:
    """``D = 1, M = N``: the whole budget spent replicating one occasion."""
    t = 0.5 * (grid.times[0] + grid.times[-1]) if time is None else time
    return [Action(time=float(t), width=width, n_segments=int(n_total), noise=noise,
                   cost=float(n_total), tag="same-time")]


def bin_midpoints(horizon: float, n: int) -> Array:
    """Centres of ``n`` equal bins on ``[0, T]``: ``(j + 1/2) T / n``.

    Midpoint placement -- not ``linspace(0, T, n)`` -- is the correct
    "temporally dispersed" comparator: endpoint placement wastes half of the
    first and last window on the boundary, and at ``n = 1`` midpoints coincide
    with the single-occasion protocol so the equal-budget comparison starts from
    an identical baseline.
    """
    return (np.arange(int(n), dtype=float) + 0.5) * float(horizon) / float(n)


def dispersed_protocol(grid: TimeGrid, n_total: int, noise: float = 1.0,
                       width: float = 0.0) -> list[Action]:
    """``D = N, M = 1``: the whole budget spent on distinct occasions."""
    return [Action(time=float(t), width=width, n_segments=1, noise=noise,
                   cost=1.0, tag="dispersed")
            for t in bin_midpoints(grid.horizon, n_total)]


def allocation_protocol(grid: TimeGrid, n_times: int, n_segments: int,
                        noise: float = 1.0, width: float = 0.0) -> list[Action]:
    """A ``D x M`` allocation of a fixed raw-segment budget ``N = D M``."""
    if n_times == 1:
        return same_time_protocol(grid, n_segments, noise=noise, width=width)
    return [Action(time=float(t), width=width, n_segments=int(n_segments),
                   noise=noise, cost=float(n_segments), tag=f"D={n_times},M={n_segments}")
            for t in bin_midpoints(grid.horizon, n_times)]


# --------------------------------------------------------------------------
# Covariance repair and regularisation
# --------------------------------------------------------------------------
def project_psd(S: Array, floor: float = 0.0) -> Array:
    """Project a symmetric matrix onto the PSD cone in Frobenius norm."""
    S = 0.5 * (np.asarray(S, dtype=float) + np.asarray(S, dtype=float).T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, floor)
    return (vecs * vals) @ vecs.T


def to_correlation(S: Array, eps: float = 1e-10) -> Array:
    """Rescale a PSD matrix to unit diagonal (the standardised process)."""
    d = np.sqrt(np.maximum(np.diag(S), eps))
    C = S / np.outer(d, d)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)


def shrink(S: Array, target: Array | None = None, intensity: float = 0.0) -> Array:
    """Linear shrinkage towards ``target`` (default: the identity)."""
    if intensity <= 0.0:
        return S
    T = np.eye(S.shape[0]) if target is None else target
    return (1.0 - intensity) * S + intensity * T


def low_rank_approx(K: Array, rank: int) -> tuple[Array, Array]:
    """Return ``(U, K_r)`` with ``K_r = U U^T`` the best rank-``r`` PSD fit."""
    vals, vecs = np.linalg.eigh(K)
    idx = np.argsort(vals)[::-1][:rank]
    U = vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0.0))
    return U, U @ U.T


def sample_paths(K: Array, n: int, rng: np.random.Generator) -> Array:
    """Draw ``n`` latent trajectories ``Z ~ N(0, K)`` (rows are objects)."""
    L = np.linalg.cholesky(K + 1e-11 * np.eye(K.shape[0]))
    return rng.standard_normal((n, K.shape[0])) @ L.T


def stationary_toeplitz(rho_values: Array) -> Array:
    """Toeplitz correlation matrix from ``rho(0), rho(1), ..., rho(p-1)``."""
    from scipy.linalg import toeplitz

    K = toeplitz(np.asarray(rho_values, dtype=float))
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, rho_values[0])
    return K
