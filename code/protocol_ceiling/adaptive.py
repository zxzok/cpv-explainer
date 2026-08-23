"""Subject-adaptive acquisition, and the exact boundary of its usefulness.

A *static* protocol fixes the same observation set for every object.  An
*adaptive* policy chooses the next window after seeing the previous ones.  This
module implements adaptive acquisition and, more importantly, makes precise when
it can possibly help.

The dividing line is whether the residual risk depends on the *values* observed
or only on *which* actions were taken.

* For a linear label ``Theta = h^T Z`` under Gaussian linear observations, the
  posterior covariance ``P_S = K - Q_S(K)`` does not depend on the realised
  ``Y_S`` at all.  Hence ``Var(Theta | Y_S) = h^T P_S h`` is a deterministic
  function of the action set, every adaptive policy is a randomisation over
  static sets, and no adaptive policy beats the best static one
  (:func:`static_optimum` versus :func:`adaptive_risk`).  Adaptive design is
  outside the scope of the article, which treats static protocols only.

* For a nonlinear label the posterior variance *is* value-dependent.  For a
  threshold label ``g_c``, ``Var(Theta_g | Y)`` is largest when the posterior
  mean sits near the threshold -- the boundary-localisation phenomenon of the
  conference version -- so a policy that has seen where the object actually sits
  can spend its remaining budget where the residual uncertainty is, and the gain
  is strict.

The expected value of information of action ``a`` at posterior ``(m, P)`` is

    EVI(a) = Var(Theta | m, P) - E_y[ Var(Theta | m^+(y), P^+) ],

with the standard Gaussian update ``m^+(y) = m + P l (y - l^T m)/s`` and
``P^+ = P - P l l^T P / s``, ``s = l^T P l + nu^2``.  Because ``P^+`` is
value-free, the outer expectation is a one-dimensional Gaussian integral over
``y``, which we evaluate by Gauss--Hermite quadrature.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr, roots_hermitenorm, roots_legendre

from .covariance import Action, TimeGrid, action_vector, protocol_matrices
from .risk import explained_covariance
from .transforms import MeanLabel, LabelFunctional, ThresholdLabel

Array = NDArray[np.float64]

_GL_NODES, _GL_WEIGHTS = roots_legendre(64)


# --------------------------------------------------------------------------
# Bivariate normal upper-orthant probability with unequal thresholds
# --------------------------------------------------------------------------
def bvn_upper_tail(a: Array, b: Array, r: Array) -> Array:
    """``P(U > a, V > b)`` for standard bivariate normal with correlation ``r``.

    Uses the Plackett/Sheppard derivative identity in the arcsine variable,

        P(U>a, V>b) = Phibar(a) Phibar(b)
                      + int_0^{arcsin r} exp{-(a^2 - 2ab sin t + b^2)/(2 cos^2 t)}
                        / (2 pi) dt,

    which has a bounded, smooth integrand right up to ``|r| = 1`` -- the same
    change of variable that keeps :mod:`protocol_ceiling.transforms` accurate.
    All arguments broadcast, so a whole ``p x p`` array of pairs is done at once.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = np.clip(np.asarray(r, float), -1.0 + 1e-12, 1.0 - 1e-12)
    theta = np.arcsin(r)
    # map the fixed Gauss-Legendre rule on [-1, 1] onto [0, theta]
    t = 0.5 * theta[..., None] * (_GL_NODES + 1.0)
    w = 0.5 * theta[..., None] * _GL_WEIGHTS
    sin_t, cos2 = np.sin(t), np.cos(t) ** 2
    expo = -(a[..., None] ** 2 - 2.0 * sin_t * a[..., None] * b[..., None]
             + b[..., None] ** 2) / (2.0 * cos2)
    integral = np.sum(w * np.exp(expo) / (2.0 * np.pi), axis=-1)
    return (1.0 - ndtr(a)) * (1.0 - ndtr(b)) + integral


# --------------------------------------------------------------------------
# Posterior label variance
# --------------------------------------------------------------------------
def posterior_label_variance(label: LabelFunctional, m: Array, P: Array,
                             omega: Array, eps: float = 1e-12) -> float:
    """``Var(Theta_g | Y)`` given the posterior mean ``m`` and covariance ``P``.

    Exact for the mean label (where it is value-free) and for threshold labels
    (where it is not).  This asymmetry is the whole content of the adaptivity
    boundary.
    """
    if isinstance(label, MeanLabel):
        return float(omega @ P @ omega)
    if isinstance(label, ThresholdLabel):
        sd = np.sqrt(np.maximum(np.diag(P), eps))
        a = (label.c - m) / sd
        corr = P / np.outer(sd, sd)
        joint = bvn_upper_tail(a[:, None] * np.ones_like(corr),
                               a[None, :] * np.ones_like(corr), corr)
        tail = 1.0 - ndtr(a)
        return float(omega @ (joint - np.outer(tail, tail)) @ omega)
    raise NotImplementedError(
        "posterior_label_variance supports mean and threshold labels exactly; "
        "use posterior_label_variance_mc for other functionals")


def posterior_label_variance_mc(label: LabelFunctional, m: Array, P: Array,
                                omega: Array, rng: np.random.Generator,
                                n: int = 20000) -> float:
    """Monte Carlo fallback for arbitrary label functionals."""
    L = np.linalg.cholesky(P + 1e-10 * np.eye(P.shape[0]))
    Z = m + rng.standard_normal((n, m.size)) @ L.T
    return float(np.var(label.apply(Z) @ omega, ddof=1))


# --------------------------------------------------------------------------
# Expected value of information
# --------------------------------------------------------------------------
def gaussian_update(m: Array, P: Array, ell: Array, noise: float, y: float
                    ) -> tuple[Array, Array, float]:
    """Return ``(m_plus, P_plus, s)`` after observing ``Y = ell^T Z + eps = y``."""
    v = P @ ell
    s = float(ell @ v) + float(noise)
    s = max(s, 1e-300)
    m_plus = m + v * (y - float(ell @ m)) / s
    P_plus = P - np.outer(v, v) / s
    return m_plus, P_plus, s


def expected_value_of_information(label: LabelFunctional, m: Array, P: Array,
                                  omega: Array, ell: Array, noise: float,
                                  n_nodes: int = 24) -> float:
    """``EVI(a) = Var(Theta|now) - E_y Var(Theta|now, Y_a=y)``.

    For the mean label the inner variance is value-free and this reduces
    exactly to the static marginal gain ``(omega^T P ell)^2 / s`` -- which is
    the computational shadow of the no-gain-from-adaptivity fact above.
    """
    v = P @ ell
    s = float(ell @ v) + float(noise)
    s = max(s, 1e-300)
    if isinstance(label, MeanLabel):
        return float((omega @ v) ** 2 / s)

    P_plus = P - np.outer(v, v) / s
    current = posterior_label_variance(label, m, P, omega)
    nodes, weights = roots_hermitenorm(n_nodes)
    weights = weights / np.sqrt(2.0 * np.pi)
    mu_y = float(ell @ m)
    total = 0.0
    for z, w in zip(nodes, weights):
        y = mu_y + np.sqrt(s) * z
        m_plus = m + v * (y - mu_y) / s
        total += w * posterior_label_variance(label, m_plus, P_plus, omega)
    return float(current - total)


# --------------------------------------------------------------------------
# Static optimum and adaptive policy
# --------------------------------------------------------------------------
@dataclass
class AdaptiveResult:
    risk: float
    risk_se: float
    static_risk: float
    static_actions: list[Action]
    n_objects: int
    selection_counts: dict[tuple, int] = field(default_factory=dict)

    @property
    def gain(self) -> float:
        """Reduction in Bayes risk relative to the best static protocol."""
        return float(self.static_risk - self.risk)

    def as_dict(self) -> dict:
        return {
            "adaptive_risk": self.risk, "adaptive_risk_se": self.risk_se,
            "static_risk": self.static_risk, "adaptivity_gain": self.gain,
            "static_times": [a.time for a in self.static_actions],
            "n_objects": self.n_objects,
            "n_distinct_adaptive_sets": len(self.selection_counts),
        }


def static_optimum(label: LabelFunctional, K: Array, grid: TimeGrid,
                   candidates: Sequence[Action], budget: int
                   ) -> tuple[list[Action], float]:
    """Exhaustive best static protocol and its exact Bayes risk.

    The risk is the closed form ``E Var(Theta_g | Y_S) = V_g(K) - F_g(S; K)``,
    which is exact for every label by the law of total variance together with
    ``Var{E(Theta_g | Y_S)} = omega^T C_g(Q_S) omega``.  No Monte Carlo and no
    quadrature are involved.
    """
    from .risk import bilinear, label_variance

    V = label_variance(label, K, grid.weights)
    best, best_risk = [], np.inf
    for combo in itertools.combinations(range(len(candidates)), budget):
        acts = [candidates[i] for i in combo]
        A, R = protocol_matrices(acts, grid)
        F = bilinear(label, explained_covariance(K, A, R), grid.weights)
        risk = V - F
        if risk < best_risk:
            best, best_risk = acts, float(risk)
    return best, best_risk


def adaptive_risk(label: LabelFunctional, K: Array, grid: TimeGrid,
                  candidates: Sequence[Action], budget: int,
                  rng: np.random.Generator, n_objects: int = 4000,
                  n_evi_nodes: int = 24) -> AdaptiveResult:
    """Monte Carlo risk of the greedy expected-value-of-information policy.

    The policy is strictly sequential: at each step it may use only the windows
    it has already acquired, never any unobserved part of the trajectory.  This
    is enforced structurally -- the state passed to the policy is exactly
    ``(m, P)``, the posterior given the acquired observations.
    """
    p = K.shape[0]
    L = np.linalg.cholesky(K + 1e-10 * np.eye(p))
    vectors = [action_vector(a, grid) for a in candidates]
    omega = grid.weights

    static_actions, static_risk = static_optimum(label, K, grid, candidates, budget)

    per_object = np.empty(n_objects)
    counts: dict[tuple, int] = {}
    for i in range(n_objects):
        z = rng.standard_normal(p) @ L.T
        m, P = np.zeros(p), K.copy()
        chosen: list[int] = []
        for _ in range(budget):
            best_idx, best_evi = -1, -np.inf
            for j in range(len(candidates)):
                if j in chosen:
                    continue
                evi = expected_value_of_information(
                    label, m, P, omega, vectors[j],
                    candidates[j].effective_noise, n_evi_nodes)
                if evi > best_evi:
                    best_idx, best_evi = j, evi
            if best_idx < 0:
                break
            ell = vectors[best_idx]
            y = float(ell @ z) + np.sqrt(candidates[best_idx].effective_noise) * rng.standard_normal()
            m, P, _ = gaussian_update(m, P, ell, candidates[best_idx].effective_noise, y)
            chosen.append(best_idx)
        per_object[i] = posterior_label_variance(label, m, P, omega)
        key = tuple(sorted(chosen))
        counts[key] = counts.get(key, 0) + 1

    return AdaptiveResult(
        risk=float(per_object.mean()),
        risk_se=float(per_object.std(ddof=1) / np.sqrt(n_objects)),
        static_risk=static_risk, static_actions=static_actions,
        n_objects=n_objects, selection_counts=counts,
    )


# --------------------------------------------------------------------------
# The construction behind the strict adaptive gain
# --------------------------------------------------------------------------
def block_construction(n_blocks: int = 2, block_size: int = 8,
                       within: float = 0.995) -> Array:
    """Independent blocks, each nearly constant inside.

    With a threshold label, each block contributes an almost binary
    ``1{block value > c}``; one observation localises its own block and tells
    the policy whether that block is still ambiguous.  This is the minimal
    structure in which value-dependent residual risk -- hence a strict adaptive
    gain -- can appear.
    """
    p = n_blocks * block_size
    K = np.zeros((p, p))
    for b in range(n_blocks):
        s = slice(b * block_size, (b + 1) * block_size)
        K[s, s] = within
    np.fill_diagonal(K, 1.0)
    return K


def block_actions(n_blocks: int, block_size: int, noise: float,
                  grid: TimeGrid) -> list[Action]:
    """One window per block, covering exactly that block."""
    step = grid.horizon / n_blocks
    return [Action(time=float((b + 0.5) * step), width=float(step * 0.98),
                   noise=noise, cost=1.0, tag=f"block{b}")
            for b in range(n_blocks)]
