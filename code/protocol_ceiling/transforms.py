"""Label functionals and their Gaussian covariance transforms.

For a label functional ``g`` and standard bivariate normal ``(U, V_r)`` with
correlation ``r``, the central object of this package is

    C_g(r) = Cov{g(U), g(V_r)}.

Every protocol-ceiling quantity in :mod:`protocol_ceiling.risk` is a bilinear
form in ``C_g`` evaluated entrywise on a covariance matrix, so ``C_g`` must be

* accurate (used in regression tests against closed forms),
* vectorised (evaluated on ``p x p`` matrices inside greedy loops),
* differentiable-in-principle (Lipschitz/Holder constants enter the
  finite-sample error bounds of Section 5: Proposition 11 supplies the constants
  that Theorems 12 and 14 consume).

Mehler's formula gives the Hermite representation

    g(z) - E g(U) = sum_{k>=1} (a_k / k!) H_k(z),
    C_g(r)        = sum_{k>=1} (a_k^2 / k!) r^k,

which we store in the numerically stable normalised form
``atilde_k = a_k / sqrt(k!)`` so that ``C_g(r) = sum_k atilde_k^2 r^k``.

Two analytic facts used by the theory are implemented and unit-tested here.

1. For ``g in W^{1,2}(phi)`` (i.e. ``g'`` square integrable under the standard
   normal law), ``C_g`` is Lipschitz on ``[-1, 1]`` with the sharp constant

       sup_{|r|<=1} |C_g'(r)| = E[g'(U)^2] = sum_k k a_k^2 / k!.

2. For a threshold label ``g_c(z) = 1{z > c}``, Plackett's identity gives

       C_g(r) = G_c(r) = int_0^r exp{-c^2/(1+s)} / (2 pi sqrt(1-s^2)) ds,

   which under ``s = sin(theta)`` becomes the *smooth, bounded* integral

       G_c(r) = int_0^{arcsin r} exp{-c^2/(1+sin u)} / (2 pi) du.

   Consequently ``|G_c(r_2) - G_c(r_1)| <= |r_2 - r_1|^{1/2} / (2 sqrt 2)``:
   threshold labels are Holder-1/2 but not Lipschitz at ``|r| = 1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.special import roots_hermitenorm

Array = NDArray[np.float64]

# Correlations are clipped into [-CLIP, CLIP] before any transform is applied.
# The clip is far tighter than machine precision but keeps arcsin derivatives
# and Hermite tails finite; every estimator in the package is stated for
# covariances whose entries stay strictly inside the open interval (-1, 1).
CLIP = 1.0 - 1e-12

# Sharp constant in |arcsin(b) - arcsin(a)| <= ARCSIN_HOLDER * |b - a|^{1/2},
# attained at (a, b) = (-1, 1).  See `prop:regularity`(ii) in the paper.
ARCSIN_HOLDER = np.pi / np.sqrt(2.0)


def _clip(r: Array | float) -> Array:
    return np.clip(np.asarray(r, dtype=float), -CLIP, CLIP)


# --------------------------------------------------------------------------
# Normalised Hermite machinery
# --------------------------------------------------------------------------
def normalised_hermite_values(z: Array, kmax: int, scale: Array | float = 1.0) -> Array:
    """Return ``scale * h_k(z)`` with ``h_k = H_k / sqrt(k!)``, ``k = 0 .. kmax``.

    Uses the stable three-term recurrence
    ``h_{k+1} = (z h_k - sqrt(k) h_{k-1}) / sqrt(k+1)``, which keeps every
    entry ``O(1)`` for ``|z| < 2 sqrt(k)`` instead of overflowing like the raw
    probabilists' polynomials.

    The optional ``scale`` is folded into the *initial* values rather than
    applied afterwards.  Because the recurrence is linear in ``h``, this is
    algebraically identical but numerically essential: with ``scale`` equal to a
    Gaussian quadrature weight, the far tail nodes (where ``h_k`` itself
    overflows but ``h_k * weight`` decays) stay finite.
    """
    z = np.asarray(z, dtype=float)
    s = np.broadcast_to(np.asarray(scale, dtype=float), z.shape)
    out = np.empty((kmax + 1,) + z.shape, dtype=float)
    out[0] = s
    if kmax == 0:
        return out
    out[1] = z * s
    for k in range(1, kmax):
        out[k + 1] = (z * out[k] - np.sqrt(k) * out[k - 1]) / np.sqrt(k + 1.0)
    return out


def hermite_coefficients(
    g, kmax: int = 200, n_nodes: int | None = None
) -> Array:
    """Normalised Hermite coefficients ``atilde_k``, ``k = 1 .. kmax``.

    ``atilde_k = E[g(U) h_k(U)]`` under ``U ~ N(0, 1)``, computed by
    Gauss--Hermite quadrature against the probabilists' weight.  The ``k = 0``
    term (the mean of ``g``) is discarded because ``C_g`` is a covariance.
    """
    if n_nodes is None:
        n_nodes = max(400, 4 * kmax)
    nodes, weights = roots_hermitenorm(n_nodes)
    weights = weights / np.sqrt(2.0 * np.pi)  # probabilists' normalisation
    gv = np.asarray(g(nodes), dtype=float)
    # Fold the weight into the recurrence so that extreme nodes cannot overflow.
    hw = normalised_hermite_values(nodes, kmax, scale=weights)
    return (hw[1:] * gv).sum(axis=1)


def indicator_hermite_coefficients(c: float, kmax: int = 200,
                                   two_sided: bool = False) -> Array:
    """Exact normalised Hermite spectrum of an indicator label.

    Because ``H_k(z) phi(z) = -{H_{k-1}(z) phi(z)}'``, integration by parts gives
    the closed forms

        g(z) = 1{z > c}   :  a_k = phi(c) H_{k-1}(c),
        g(z) = 1{|z| > c} :  a_k = phi(c) H_{k-1}(c) {1 - (-1)^{k-1}},

    so the two-sided label carries only *even* orders.  These are returned in
    the normalised scaling ``atilde_k = a_k / sqrt(k!) = phi(c) h_{k-1}(c)/sqrt(k)``.

    Gauss--Hermite quadrature must not be used here: the integrand is
    discontinuous and the quadrature error swamps the coefficients.
    """
    h = normalised_hermite_values(np.array(float(c)), kmax - 1).ravel()  # h_0..h_{kmax-1}
    ks = np.arange(1, kmax + 1, dtype=float)
    phi = np.exp(-0.5 * c * c) / np.sqrt(2.0 * np.pi)
    coeffs = phi * h / np.sqrt(ks)
    if two_sided:
        parity = 1.0 - (-1.0) ** (ks - 1.0)   # 0 for odd k, 2 for even k
        coeffs = coeffs * parity
    return coeffs


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LabelFunctional:
    """Abstract base: a label functional ``g`` with its transform ``C_g``."""

    name: str = "label"

    # -- interface -------------------------------------------------------
    def apply(self, z: Array) -> Array:  # pragma: no cover - overridden
        raise NotImplementedError

    def C(self, r: Array | float) -> Array:  # pragma: no cover - overridden
        raise NotImplementedError

    def dC(self, r: Array | float) -> Array:  # pragma: no cover - overridden
        raise NotImplementedError

    # -- constants entering the finite-sample bounds ---------------------
    def modulus(self, rmax: float = CLIP) -> tuple[float, float]:
        """Return ``(L, beta)`` with ``|C_g(r2)-C_g(r1)| <= L |r2-r1|^beta``
        for all ``r1, r2`` in ``[-rmax, rmax]``."""
        return float(np.max(np.abs(self.dC(np.linspace(-rmax, rmax, 4001))))), 1.0

    def derivative_floor(self, rmax: float = CLIP) -> float:
        """``inf_{0<=r<=rmax} C_g'(r)``; enters the submodularity-ratio bound."""
        grid = np.linspace(0.0, rmax, 4001)
        return float(np.min(self.dC(grid)))

    def variance(self) -> float:
        """``Var g(U) = C_g(1)``."""
        return float(self.C(CLIP))


# --------------------------------------------------------------------------
# Mean label
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MeanLabel(LabelFunctional):
    name: str = "mean"

    def apply(self, z: Array) -> Array:
        return np.asarray(z, dtype=float)

    def C(self, r: Array | float) -> Array:
        return _clip(r)

    def dC(self, r: Array | float) -> Array:
        return np.ones_like(_clip(r))

    def modulus(self, rmax: float = CLIP) -> tuple[float, float]:
        return 1.0, 1.0

    def derivative_floor(self, rmax: float = CLIP) -> float:
        return 1.0

    def variance(self) -> float:
        return 1.0


# --------------------------------------------------------------------------
# Threshold (occupation-time) label
# --------------------------------------------------------------------------
class _PlackettTable:
    """Cubic-spline table of ``theta -> int_0^theta e^{-c^2/(1+sin u)} du``."""

    @staticmethod
    def _integrand(theta: Array, c: float) -> Array:
        """``exp{-c^2 / (1 + sin theta)} / (2 pi)``, safe at ``theta = -pi/2``.

        The denominator vanishes at the left endpoint, where the integrand tends
        to ``0`` for ``c != 0`` and to ``1/(2 pi)`` for ``c = 0``; both limits are
        supplied explicitly so that no ``0/0`` or ``x/0`` ever forms.
        """
        s = 1.0 + np.sin(theta)
        if abs(c) < 1e-14:
            return np.full_like(s, 1.0 / (2.0 * np.pi))
        out = np.zeros_like(s)
        ok = s > 1e-300
        out[ok] = np.exp(-c**2 / s[ok]) / (2.0 * np.pi)
        return out

    def __init__(self, c: float, n: int = 6001):
        self.c = float(c)
        theta = np.linspace(-np.pi / 2.0, np.pi / 2.0, n)
        f = self._integrand(theta, self.c)
        # Composite Simpson on each consecutive pair of panels, refined by a
        # midpoint evaluation, gives O(h^4) cumulative accuracy.
        mid = 0.5 * (theta[:-1] + theta[1:])
        fmid = self._integrand(mid, self.c)
        h = np.diff(theta)
        panel = h * (f[:-1] + 4.0 * fmid + f[1:]) / 6.0
        cum = np.concatenate([[0.0], np.cumsum(panel)])
        # Re-centre so that the table is exactly zero at theta = 0.
        zero = np.interp(0.0, theta, cum)
        self.spline = CubicSpline(theta, cum - zero, extrapolate=True)

    def __call__(self, r: Array) -> Array:
        return self.spline(np.arcsin(r))


@dataclass(frozen=True)
class ThresholdLabel(LabelFunctional):
    """``g_c(z) = 1{z > c}``: the occupation-time label."""

    c: float = 0.0
    name: str = "occupation"

    @cached_property
    def _table(self) -> _PlackettTable:
        return _PlackettTable(self.c)

    def apply(self, z: Array) -> Array:
        return (np.asarray(z, dtype=float) > self.c).astype(float)

    def C(self, r: Array | float) -> Array:
        rr = _clip(r)
        if abs(self.c) < 1e-14:
            return np.arcsin(rr) / (2.0 * np.pi)
        return self._table(rr)

    def dC(self, r: Array | float) -> Array:
        rr = _clip(r)
        return np.exp(-self.c**2 / (1.0 + rr)) / (2.0 * np.pi * np.sqrt(1.0 - rr**2))

    def modulus(self, rmax: float = CLIP) -> tuple[float, float]:
        """Threshold labels are Holder-1/2 with the explicit constant
        ``ARCSIN_HOLDER / (2 pi)``, uniformly in the threshold ``c``.

        The bound follows from ``|C_g'(r)| <= exp(-c^2/(1+r))/(2 pi (1-r^2)^{1/2})``
        and the sharp arcsine Holder inequality; it is uniform over ``[-1, 1]``
        and therefore does not degrade as ``rmax -> 1``.  The ``exp(-c^2/(1+r))``
        factor is bounded by one and is *not* carried into the constant: keeping
        it would give a smaller constant only at ``r`` bounded away from ``-1``,
        which is exactly where the uniform statement must not assume anything.
        This is the constant the manuscript quotes.
        """
        return float(ARCSIN_HOLDER / (2.0 * np.pi)), 0.5

    def derivative_floor(self, rmax: float = CLIP) -> float:
        # C_g' is increasing on [0, 1) for every c, so the floor is at r = 0.
        return float(np.exp(-self.c**2) / (2.0 * np.pi))

    def variance(self) -> float:
        from scipy.special import ndtr

        p = float(1.0 - ndtr(self.c))
        return p * (1.0 - p)


@dataclass(frozen=True)
class TwoSidedLabel(LabelFunctional):
    """``g_c(z) = 1{|z| > c}``: a two-sided excursion label.

    ``C_g(r) = 2 [G_c(r) + G_c(-r)]``, which vanishes identically at ``c = 0``
    and, unlike the one-sided label, has only *even* Hermite orders.
    """

    c: float = 1.0
    name: str = "two_sided"

    @cached_property
    def _one_sided(self) -> ThresholdLabel:
        return ThresholdLabel(c=self.c)

    def apply(self, z: Array) -> Array:
        return (np.abs(np.asarray(z, dtype=float)) > self.c).astype(float)

    def C(self, r: Array | float) -> Array:
        rr = _clip(r)
        return 2.0 * (self._one_sided.C(rr) + self._one_sided.C(-rr))

    def dC(self, r: Array | float) -> Array:
        rr = _clip(r)
        return 2.0 * (self._one_sided.dC(rr) - self._one_sided.dC(-rr))

    def modulus(self, rmax: float = CLIP) -> tuple[float, float]:
        return float(2.0 * ARCSIN_HOLDER / np.pi), 0.5

    def derivative_floor(self, rmax: float = CLIP) -> float:
        grid = np.linspace(0.0, min(rmax, 1.0 - 1e-9), 4001)
        return float(np.min(self.dC(grid)))

    def variance(self) -> float:
        from scipy.special import ndtr

        p = float(2.0 * (1.0 - ndtr(self.c)))
        return p * (1.0 - p)


# --------------------------------------------------------------------------
# Smooth labels via the Hermite series
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class HermiteLabel(LabelFunctional):
    """A generic label represented by its normalised Hermite spectrum.

    ``C_g(r) = sum_{k>=1} atilde_k^2 r^k`` and
    ``C_g'(r) = sum_{k>=1} k atilde_k^2 r^{k-1}``, so
    ``sup_{|r|<=1}|C_g'| = sum_k k atilde_k^2 = E[g'(U)^2]``.
    """

    g: object = None
    kmax: int = 200
    name: str = "hermite"

    @cached_property
    def coeffs(self) -> Array:
        return hermite_coefficients(self.g, self.kmax)

    @cached_property
    def weights(self) -> Array:
        return self.coeffs**2

    def apply(self, z: Array) -> Array:
        return np.asarray(self.g(np.asarray(z, dtype=float)), dtype=float)

    def C(self, r: Array | float) -> Array:
        rr = _clip(r)
        shape = rr.shape
        flat = rr.ravel()
        # Horner evaluation of sum_k w_k r^k = r * (w_1 + r*(w_2 + ...)).
        acc = np.zeros_like(flat)
        for w in self.weights[::-1]:
            acc = flat * (acc + w)
        return acc.reshape(shape)

    def dC(self, r: Array | float) -> Array:
        rr = _clip(r)
        shape = rr.shape
        flat = rr.ravel()
        ks = np.arange(1, len(self.weights) + 1, dtype=float)
        dw = ks * self.weights
        acc = np.zeros_like(flat)
        for w in dw[:0:-1]:
            acc = flat * (acc + w)
        return (acc + dw[0]).reshape(shape)

    def modulus(self, rmax: float = CLIP) -> tuple[float, float]:
        return float(np.sum(np.arange(1, len(self.weights) + 1) * self.weights)), 1.0

    def derivative_floor(self, rmax: float = CLIP) -> float:
        return float(np.min(self.dC(np.linspace(0.0, rmax, 2001))))

    def variance(self) -> float:
        return float(np.sum(self.weights))


@dataclass(frozen=True)
class SquareLabel(LabelFunctional):
    """``g(z) = z^2``; exactly ``C_g(r) = 2 r^2``."""

    name: str = "square"

    def apply(self, z: Array) -> Array:
        return np.asarray(z, dtype=float) ** 2

    def C(self, r: Array | float) -> Array:
        return 2.0 * _clip(r) ** 2

    def dC(self, r: Array | float) -> Array:
        return 4.0 * _clip(r)

    def modulus(self, rmax: float = CLIP) -> tuple[float, float]:
        return 4.0, 1.0

    def derivative_floor(self, rmax: float = CLIP) -> float:
        return 0.0

    def variance(self) -> float:
        return 2.0


def sigmoid_label(slope: float = 2.0, c: float = 0.0, kmax: int = 200) -> HermiteLabel:
    """Smooth surrogate ``g(z) = sigmoid{slope (z - c)}`` of a threshold label."""

    def g(z: Array) -> Array:
        return 1.0 / (1.0 + np.exp(-slope * (z - c)))

    return HermiteLabel(g=g, kmax=kmax, name=f"sigmoid(s={slope},c={c})")


LABELS: dict[str, LabelFunctional] = {
    "mean": MeanLabel(),
    "occupation": ThresholdLabel(c=0.0),
    "square": SquareLabel(),
}


def make_label(spec: str) -> LabelFunctional:
    """Parse a short textual label specification used by the experiment configs.

    Examples: ``"mean"``, ``"occ@0"``, ``"occ@0.5"``, ``"two_sided@1"``,
    ``"square"``, ``"sigmoid@2,0"``.
    """
    spec = spec.strip()
    if spec in LABELS:
        return LABELS[spec]
    head, _, tail = spec.partition("@")
    head = head.strip()
    if head in ("occ", "occupation", "threshold"):
        return ThresholdLabel(c=float(tail))
    if head in ("two_sided", "twosided", "abs"):
        return TwoSidedLabel(c=float(tail))
    if head == "sigmoid":
        parts = [float(x) for x in tail.split(",")]
        return sigmoid_label(*parts)
    raise ValueError(f"unknown label specification: {spec!r}")
