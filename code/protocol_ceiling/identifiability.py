"""Counterfactual protocol identifiability: constructions and certificates.

A benchmark observed under protocol ``A`` reveals the joint law of
``(Y_A, Theta)``.  In the Gaussian model ``Z ~ N(0, K)``, ``Theta = h^T Z``,
``Y_A = A Z + eps_A``, that law depends on ``K`` only through

    A K A^T,    A K h,    h^T K h.

Any symmetric perturbation ``Delta`` annihilating all three leaves the observed
benchmark *exactly* unchanged.  If some alternative protocol ``B`` has
``B Delta B^T != 0`` or ``B Delta h != 0``, then ``K + eps Delta`` and
``K - eps Delta`` are observationally equivalent under ``A`` yet assign
different Bayes ceilings to ``B``: the counterfactual ceiling is not identified.

This module supplies

* :func:`nonidentified_directions` -- a basis for the space of such ``Delta``
  (optionally constrained to preserve a unit diagonal, i.e. a standardised
  process, and/or to be stationary/Toeplitz);
* :func:`minimal_stationary_example` -- the sharp four-point stationary
  counterexample of `thm:minimal`, together with the ``p = 3`` local
  identifiability that makes ``p >= 4`` minimal;
* :func:`max_psd_step` and :func:`ceiling_gap` -- numerical certificates that
  both perturbed covariances are genuine correlation matrices and that the
  counterfactual ceilings really differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import toeplitz

Array = NDArray[np.float64]


# --------------------------------------------------------------------------
# Symmetric-matrix vectorisation helpers
# --------------------------------------------------------------------------
def _sym_basis(p: int, unit_diagonal: bool) -> list[Array]:
    """Basis of the symmetric matrices (with zero diagonal if requested)."""
    basis = []
    for i in range(p):
        for j in range(i, p):
            if unit_diagonal and i == j:
                continue
            E = np.zeros((p, p))
            E[i, j] = 1.0
            E[j, i] = 1.0
            basis.append(E)
    return basis


def _toeplitz_basis(p: int, unit_diagonal: bool) -> list[Array]:
    """Basis of symmetric Toeplitz matrices (stationary perturbations)."""
    basis = []
    start = 1 if unit_diagonal else 0
    for lag in range(start, p):
        row = np.zeros(p)
        row[lag] = 1.0
        basis.append(toeplitz(row))
    return basis


def nonidentified_directions(
    A: Array,
    h: Array,
    stationary: bool = False,
    unit_diagonal: bool = True,
    tol: float = 1e-10,
) -> Array:
    """Basis of perturbations invisible to protocol ``A`` and the label.

    Returns an array of shape ``(n_dir, p, p)``.  Every returned ``Delta``
    satisfies ``A Delta A^T = 0``, ``A Delta h = 0`` and ``h^T Delta h = 0``
    (and ``diag(Delta) = 0`` when ``unit_diagonal`` is set, so that ``K +- eps
    Delta`` remains a correlation matrix).
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    h = np.asarray(h, dtype=float).ravel()
    p = h.size
    basis = _toeplitz_basis(p, unit_diagonal) if stationary else _sym_basis(p, unit_diagonal)
    if not basis:
        return np.zeros((0, p, p))

    rows = []
    for E in basis:
        AEA = A @ E @ A.T
        iu = np.triu_indices(AEA.shape[0])
        rows.append(np.concatenate([AEA[iu], (A @ E @ h).ravel(), [float(h @ E @ h)]]))
    Mconstraints = np.stack(rows, axis=1)          # (n_constraints, n_basis)

    _, s, Vt = np.linalg.svd(Mconstraints, full_matrices=True)
    rank = int(np.sum(s > tol * max(1.0, s[0] if s.size else 1.0)))
    null = Vt[rank:]
    out = np.array([sum(c * E for c, E in zip(vec, basis)) for vec in null])
    if out.size == 0:
        return np.zeros((0, p, p))
    # Normalise each direction to unit spectral norm for interpretability.
    return np.array([D / max(np.linalg.norm(D, 2), 1e-300) for D in out])


def max_psd_step(K: Array, Delta: Array, safety: float = 0.98) -> float:
    """Largest ``eps`` with ``K +- eps Delta`` positive definite."""
    lo, hi = 0.0, 10.0
    def ok(e: float) -> bool:
        return (np.linalg.eigvalsh(K + e * Delta).min() > 0
                and np.linalg.eigvalsh(K - e * Delta).min() > 0)
    if not ok(1e-12):
        return 0.0
    while ok(hi) and hi < 1e6:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return float(safety * lo)


# --------------------------------------------------------------------------
# Observational-equivalence and ceiling-gap certificates
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class IdentifiabilityCertificate:
    K_plus: Array
    K_minus: Array
    eps: float
    observed_discrepancy: float
    ceiling_plus: float
    ceiling_minus: float

    @property
    def ceiling_gap(self) -> float:
        return float(abs(self.ceiling_plus - self.ceiling_minus))

    def as_dict(self) -> dict:
        return {
            "eps": self.eps,
            "observed_discrepancy": self.observed_discrepancy,
            "ceiling_plus": self.ceiling_plus,
            "ceiling_minus": self.ceiling_minus,
            "ceiling_gap": self.ceiling_gap,
        }


def observed_discrepancy(K1: Array, K2: Array, A: Array, h: Array,
                         R: Array | None = None) -> float:
    """Max absolute difference of every functional the benchmark identifies."""
    A = np.atleast_2d(A)
    R = np.zeros((A.shape[0], A.shape[0])) if R is None else R
    d1 = [A @ K1 @ A.T + R, (A @ K1 @ h).reshape(-1, 1), np.array([[h @ K1 @ h]])]
    d2 = [A @ K2 @ A.T + R, (A @ K2 @ h).reshape(-1, 1), np.array([[h @ K2 @ h]])]
    return float(max(np.max(np.abs(x - y)) for x, y in zip(d1, d2)))


def linear_ceiling(K: Array, B: Array, h: Array, R: Array | None = None) -> float:
    """Bayes ceiling ``I_B = h^T Q_B h / h^T K h`` for a linear label."""
    B = np.atleast_2d(B)
    R = np.zeros((B.shape[0], B.shape[0])) if R is None else R
    KB = K @ B.T
    M = B @ KB + R
    explained = float((h @ KB) @ np.linalg.solve(M, (KB.T @ h)))
    total = float(h @ K @ h)
    return explained / total if total > 0 else 0.0


def certify(K0: Array, Delta: Array, A: Array, B: Array, h: Array,
            R_A: Array | None = None, R_B: Array | None = None,
            eps: float | None = None) -> IdentifiabilityCertificate:
    """Build and verify an observational-equivalence pair around ``K0``."""
    e = max_psd_step(K0, Delta) if eps is None else float(eps)
    Kp, Km = K0 + e * Delta, K0 - e * Delta
    return IdentifiabilityCertificate(
        K_plus=Kp, K_minus=Km, eps=e,
        observed_discrepancy=observed_discrepancy(Kp, Km, A, h, R_A),
        ceiling_plus=linear_ceiling(Kp, B, h, R_B),
        ceiling_minus=linear_ceiling(Km, B, h, R_B),
    )


# --------------------------------------------------------------------------
# The sharp minimal stationary example
# --------------------------------------------------------------------------
def minimal_stationary_example(tau: float = 1.0, eps: float | None = None,
                               noise: float = 0.0) -> dict:
    """The four-point stationary counterexample of `thm:minimal`.

    Grid ``t = 0, 1, 2, 3``; stationary unit-variance Gaussian process; mean
    label ``Theta = (Z_0 + Z_1 + Z_2 + Z_3)/4``; observed protocol
    ``A = {Z_0}``; counterfactual protocol ``B = {Z_1, Z_2}``.

    The stationary perturbation ``delta = (0, 1, -2, 1)`` on lags satisfies the
    two linear constraints imposed by the observable functionals
    (``Cov(Y_A, Theta)`` and ``Var(Theta)``) while changing ``rho(1)``, hence
    the ceiling of ``B``.  Because a stationary unit-variance model on ``p``
    points has ``p - 1`` free correlations and a single-point protocol imposes
    exactly two constraints, ``p >= 4`` is the smallest dimension in which such
    a direction exists.
    """
    p = 4
    rho0 = np.exp(-np.arange(p) / tau)
    K0 = toeplitz(rho0)
    delta = np.array([0.0, 1.0, -2.0, 1.0])
    Delta = toeplitz(delta)

    h = np.full(p, 1.0 / p)
    A = np.zeros((1, p)); A[0, 0] = 1.0
    B = np.zeros((2, p)); B[0, 1] = 1.0; B[1, 2] = 1.0
    R_A = np.array([[noise]])
    R_B = np.eye(2) * noise

    cert = certify(K0, Delta / np.linalg.norm(Delta, 2), A, B, h, R_A, R_B, eps)
    return {
        "K0": K0, "Delta": Delta, "h": h, "A": A, "B": B,
        "certificate": cert,
        "delta_lags": delta,
    }


def stationary_identification_jacobian(p: int, obs_index: int = 0) -> Array:
    """Jacobian of the observed functionals w.r.t. ``(rho(1), ..., rho(p-1))``.

    For a single-point protocol the observed functionals are
    ``Cov(Y, Theta) = p^{-1} sum_j rho(|j - obs|)`` and
    ``Var(Theta) = p^{-2} sum_{j,k} rho(|j-k|)``.  The matrix returned has two
    rows; it has full row rank for ``p = 3`` (so the model is locally
    identified) and a non-trivial kernel for ``p >= 4``.
    """
    rows = np.zeros((2, p - 1))
    for lag in range(1, p):
        # d Cov(Y, Theta) / d rho(lag)
        count = sum(1 for j in range(p) if abs(j - obs_index) == lag)
        rows[0, lag - 1] = count / p
        # d Var(Theta) / d rho(lag)
        pair = 2 * (p - lag)
        rows[1, lag - 1] = pair / (p * p)
    return rows


def counting_bound(p: int, d_obs: int, stationary: bool) -> dict:
    """Dimension count behind the genericity statement of `prop:counting`.

    The protocol is taken to select ``d_obs`` coordinates, which is the case the
    experiments use.  Then the ``d_obs`` diagonal entries of ``A K A^T`` are
    identically one under the standardisation that already fixed ``diag(K)``, so
    they are not constraints: the non-trivial count is

        d(d-1)/2  (off-diagonal blocks)  +  d  (``A K h``)  +  1  (``h^T K h``).

    The published statement of `prop:counting` uses the looser ``d(d+1)/2+d+1``
    and asserts the invisible space has dimension *at least* the corresponding
    deficiency; that remains true, and ``constraints_published`` is returned so
    the two can be compared.  ``deficiency`` uses the tight count, which matches
    ``len(nonidentified_directions(...))`` exactly for coordinate protocols.

    For a stationary model the entries of ``A K A^T`` repeat, so no closed count
    is attempted beyond ``d_obs = 1`` (where two constraints act on
    ``rho(1), ..., rho(p-1)``); ``constraints`` is then ``None`` and
    ``non_identified`` is ``None`` rather than a value that may be wrong.
    """
    published = d_obs * (d_obs + 1) // 2 + d_obs + 1
    if stationary:
        free = p - 1                     # rho(1), ..., rho(p-1)
        # a single-point protocol contributes only 2 non-trivial constraints
        constraints = 2 if d_obs == 1 else None
    else:
        free = p * (p - 1) // 2          # unit diagonal fixed
        constraints = d_obs * (d_obs - 1) // 2 + d_obs + 1
    if constraints is None:
        return {"free_parameters": int(free), "constraints": None,
                "constraints_published": int(published),
                "non_identified": None, "deficiency": None}
    return {"free_parameters": int(free), "constraints": int(constraints),
            "constraints_published": int(published),
            "non_identified": bool(free > constraints),
            "deficiency": int(max(free - constraints, 0))}
