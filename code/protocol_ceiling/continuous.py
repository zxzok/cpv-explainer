"""Continuous-time trait--state theory. NOT reported in the current article;
retained for the companion preprint and the regression tests.

This module implements the general finite-horizon variance decomposition

    Var(Theta_{g,T}) = Var{mu_g(M)}
                       + (2 / T^2) E_M int_0^T (T - u) gamma_g(u | M) du,

its short-memory expansion ``V_trait + A_state / T + o(T^{-1})``, and the
Gaussian--Hermite special case that the conference version established.  The
Gaussian routines are kept bit-comparable with the released conference code so
that :mod:`tests.test_regression_conference` can verify that the journal
pipeline reproduces every previously published number.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.signal import fftconvolve

from .transforms import ThresholdLabel, normalised_hermite_values

Array = NDArray[np.float64]
Kernel = Callable[[Array], Array]


# --------------------------------------------------------------------------
# Generic (non-Gaussian) finite-time decomposition
# --------------------------------------------------------------------------
def finite_time_variance(
    gamma: Callable[[float], float],
    v_trait: float,
    T: float,
    limit: int = 400,
) -> float:
    """Exact ``Var(Theta_{g,T})`` from a conditional autocovariance ``gamma``.

    ``gamma(u)`` must already be averaged over the stable factor ``M``, i.e.
    ``gamma(u) = E_M Cov[g{Z(t)}, g{Z(t+u)} | M]``.
    """
    val = quad(lambda u: (T - u) * gamma(u), 0.0, T, limit=limit,
               epsabs=1e-11, epsrel=1e-10)[0]
    return float(v_trait + 2.0 * val / (T * T))


def state_coefficient_general(gamma: Callable[[float], float],
                              upper: float = 200.0, limit: int = 500) -> float:
    """``A_state = 2 int_0^infty gamma(u) du``."""
    return float(2.0 * quad(gamma, 0.0, upper, limit=limit,
                            epsabs=1e-11, epsrel=1e-10)[0])


def finite_time_error_bound(gamma_abs: Callable[[float], float], T: float,
                            upper: float = 200.0) -> float:
    """Explicit finite-horizon remainder bound,

        |Var - V_trait - A_state / T|
            <= (2 / T^2) int_0^T u |gamma(u)| du
               + (2 / T) int_T^infty |gamma(u)| du.
    """
    first = quad(lambda u: u * gamma_abs(u), 0.0, T, limit=400,
                 epsabs=1e-11, epsrel=1e-10)[0]
    tail = quad(gamma_abs, T, max(upper, 2.0 * T), limit=400,
                epsabs=1e-11, epsrel=1e-10)[0]
    return float(2.0 * first / (T * T) + 2.0 * tail / T)


# --------------------------------------------------------------------------
# Gaussian special case (conference results)
# --------------------------------------------------------------------------
def trait_state_correlation(u: Array | float, alpha: float, rho: Kernel) -> Array:
    return alpha + (1.0 - alpha) * np.asarray(rho(u), dtype=float)


def _cg(label: str, r: float, threshold: float) -> float:
    if label == "mean":
        return float(r)
    return float(ThresholdLabel(c=threshold).C(r))


def exact_label_variance(T: float, alpha: float, rho: Kernel,
                         label: str = "occupation", threshold: float = 0.0) -> float:
    """Exact ``Var[T^{-1} int_0^T g{Z(t)} dt]`` by one-dimensional quadrature."""
    if T <= 0:
        raise ValueError("T must be positive")

    def integrand(u: float) -> float:
        r = float(trait_state_correlation(u, alpha, rho))
        return (T - u) * _cg(label, r, threshold)

    val = quad(integrand, 0.0, T, epsabs=2e-11, epsrel=2e-10, limit=400)[0]
    return float(2.0 * val / (T * T))


def trait_floor(alpha: float, label: str = "occupation", threshold: float = 0.0) -> float:
    return _cg(label, alpha, threshold)


def state_coefficient(alpha: float, rho: Kernel, label: str = "occupation",
                      threshold: float = 0.0, upper: float = 80.0) -> float:
    floor = trait_floor(alpha, label, threshold)

    def integrand(u: float) -> float:
        r = float(trait_state_correlation(u, alpha, rho))
        return _cg(label, r, threshold) - floor

    return float(2.0 * quad(integrand, 0.0, upper, epsabs=2e-11,
                            epsrel=2e-10, limit=500)[0])


def occupation_state_coefficient_ou(a: float, tau: float = 1.0) -> float:
    """OU worked example ``A_a = 2 tau int_0^1 G_a(r) / r dr``.

    At ``a = 0`` this equals ``tau log(2) / 2`` in closed form.
    """
    lab = ThresholdLabel(c=a)

    def integrand(r: float) -> float:
        if r < 1e-12:
            return float(np.exp(-a * a) / (2.0 * np.pi))
        return float(lab.C(r) / r)

    return float(2.0 * tau * quad(integrand, 0.0, 1.0, epsabs=2e-12,
                                  epsrel=2e-11, limit=500, points=[0.999999])[0])


def eta_window(w: float, rho: Kernel) -> float:
    """Variance of a unit-process window average of length ``w``."""
    if w <= 1e-10:
        return 1.0
    val = quad(lambda u: (w - u) * float(rho(u)), 0.0, w,
               epsabs=1e-11, epsrel=1e-10, limit=300)[0]
    return float(2.0 * val / (w * w))


def mean_effective_span(w: float, rho: Kernel, tau1: float = 1.0,
                        noise_var: float = 0.0) -> float:
    return float(2.0 * tau1 / (eta_window(w, rho) + noise_var))


@lru_cache(maxsize=64)
def _indicator_weights(a: float, kmax: int) -> Array:
    """``b_k = phi(a)^2 H_{k-1}(a)^2 / k!`` for ``k = 1 .. kmax``."""
    h = normalised_hermite_values(np.array(float(a)), kmax - 1)  # h_0 .. h_{kmax-1}
    ks = np.arange(1, kmax + 1, dtype=float)
    phi2 = np.exp(-a * a) / (2.0 * np.pi)
    return phi2 * np.asarray(h).ravel() ** 2 / ks


def occupation_effective_span(a: float, w: float, rho: Kernel, tau1: float = 1.0,
                              noise_var: float = 0.0, kmax: int = 120,
                              domain: float = 30.0, dx: float = 0.005
                              ) -> tuple[float, float, float]:
    """Return ``(ell_a(w), A_a, B_a(w))`` from the Hermite expansion."""
    x = np.arange(-domain, domain + dx, dx, dtype=float)
    base = np.asarray(rho(x), dtype=float)
    if w <= dx:
        conv, eta = base.copy(), 1.0
    else:
        n = max(1, int(round(w / dx)))
        if n % 2 == 0:
            n += 1
        eff_w = n * dx
        conv = fftconvolve(base, np.ones(n) / eff_w, mode="same") * dx
        eta = eta_window(eff_w, rho)
    corr = np.clip(conv / np.sqrt(eta + noise_var), -1.0, 1.0)

    b = _indicator_weights(float(a), kmax)
    pos = np.arange(0.0, domain + dx, dx)
    rho_pos = np.asarray(rho(pos), dtype=float)
    ks = np.arange(1, kmax + 1, dtype=int)
    tau_k = np.array([np.trapezoid(rho_pos**k, pos) for k in ks])
    j_k = np.array([np.trapezoid(corr**k, x) for k in ks])

    A_a = float(2.0 * np.sum(b * tau_k))
    B_a = float(np.sum(b * j_k * j_k))
    return B_a / A_a, A_a, B_a


def point_protocol_explainability(T: float, times: Array, noise_vars: Array,
                                  rho: Kernel, alpha: float = 0.0,
                                  threshold: float = 0.0,
                                  grid_size: int = 801) -> float:
    """Exact ``I_pi`` for noisy point observations of an occupation label.

    This is the continuous-time routine used by the conference paper's
    equal-budget experiment; the journal pipeline reproduces its values through
    the discrete interface in :mod:`protocol_ceiling.risk`.
    """
    times = np.asarray(times, dtype=float).ravel()
    noise_vars = np.asarray(noise_vars, dtype=float).ravel()
    if times.size == 0 or times.size != noise_vars.size:
        raise ValueError("times and noise_vars must have equal non-zero length")

    grid = np.linspace(0.0, T, grid_size)
    dx = T / (grid_size - 1)
    weights = np.full(grid_size, dx)
    weights[[0, -1]] *= 0.5

    def corr(d: Array) -> Array:
        return alpha + (1.0 - alpha) * np.asarray(rho(d), dtype=float)

    k_oo = corr(times[:, None] - times[None, :]) + np.diag(noise_vars)
    k_go = corr(grid[:, None] - times[None, :])
    q = np.clip(k_go @ np.linalg.solve(k_oo, k_go.T), -1.0, 1.0)

    lab = ThresholdLabel(c=threshold)
    explained = float(weights @ lab.C(q) @ weights / (T * T))
    total = exact_label_variance(T, alpha, rho, "occupation", threshold)
    return float(explained / total)
