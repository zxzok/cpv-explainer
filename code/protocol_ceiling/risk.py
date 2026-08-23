"""Protocol-conditioned Bayes risk, ceilings, and rank-one design updates.

Everything here is a consequence of one identity.  Let ``Z ~ N(0, K)`` with
``diag(K) = 1`` and let ``Y_S = A_S Z + eps``, ``eps ~ N(0, R_S)``.  Draw two
posterior replicas ``Z^(1), Z^(2)`` conditionally independent given ``Y_S``.
Each has the *unconditional* marginal law ``N(0, K)``, while

    Cov{Z^(1)_j, Z^(2)_k} = Q_S(K)_{jk},
    Q_S(K) = K A_S^T (A_S K A_S^T + R_S)^{-1} A_S K.

Hence ``(Z^(1)_j, Z^(2)_k)`` is standard bivariate normal with correlation
``Q_{S,jk}``, and for any ``g in L^2(phi)``

    Var{E(Theta_g | Y_S)} = sum_{j,k} omega_j omega_k C_g(Q_{S,jk}) =: F_g(S; K),
    Var(Theta_g)          = sum_{j,k} omega_j omega_k C_g(K_{jk})   =: V_g(K),
    R_S^*                 = V_g(K) - F_g(S; K),
    I_g(S; K)             = F_g(S; K) / V_g(K).

The residual (posterior) covariance is ``P_S = K - Q_S``.  Adding one action
with measurement row ``ell_a`` and noise ``nu_a^2`` gives the exact rank-one
update

    v = P_S ell_a,   s = ell_a^T P_S ell_a + nu_a^2,
    Q_{S+a} = Q_S + v v^T / s,   P_{S+a} = P_S - v v^T / s,

which is what makes label-aware greedy design cheap: no matrix inverse is
recomputed along the greedy path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .covariance import Action, TimeGrid, protocol_matrices
from .transforms import LabelFunctional

Array = NDArray[np.float64]


# --------------------------------------------------------------------------
# Core quantities
# --------------------------------------------------------------------------
def explained_covariance(K: Array, A: Array, R: Array, jitter: float = 1e-10) -> Array:
    """``Q_S(K) = K A^T (A K A^T + R)^{-1} A K``."""
    if A.shape[0] == 0:
        return np.zeros_like(K)
    KA = K @ A.T                       # p x d
    M = A @ KA + R                     # d x d
    M = 0.5 * (M + M.T) + jitter * np.eye(M.shape[0])
    return KA @ np.linalg.solve(M, KA.T)


def residual_covariance(K: Array, A: Array, R: Array, jitter: float = 1e-10) -> Array:
    """``P_S = K - Q_S``: the posterior covariance of ``Z`` given ``Y_S``."""
    return K - explained_covariance(K, A, R, jitter)


def bilinear(label: LabelFunctional, M: Array, omega: Array) -> float:
    """``sum_{j,k} omega_j omega_k C_g(M_{jk})``."""
    return float(omega @ label.C(M) @ omega)


def label_variance(label: LabelFunctional, K: Array, omega: Array) -> float:
    """``V_g(K) = Var(Theta_g)``."""
    return bilinear(label, K, omega)


def explained_variance(label: LabelFunctional, K: Array, A: Array, R: Array,
                       omega: Array) -> float:
    """``F_g(S; K) = Var{E(Theta_g | Y_S)}``."""
    return bilinear(label, explained_covariance(K, A, R), omega)


def protocol_ceiling(label: LabelFunctional, K: Array, A: Array, R: Array,
                     omega: Array) -> float:
    """``I_g(S; K) = F_g(S; K) / V_g(K)``, the protocol-explained fraction."""
    V = label_variance(label, K, omega)
    if V <= 0.0:
        return 0.0
    return explained_variance(label, K, A, R, omega) / V


def bayes_risk(label: LabelFunctional, K: Array, A: Array, R: Array,
               omega: Array) -> float:
    """``R_S^* = E Var(Theta_g | Y_S)``."""
    return label_variance(label, K, omega) - explained_variance(label, K, A, R, omega)


# --------------------------------------------------------------------------
# Protocol-level convenience wrappers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CeilingReport:
    ceiling: float
    explained: float
    total: float
    risk: float
    n_actions: int
    cost: float

    def as_dict(self) -> dict:
        return {
            "ceiling": self.ceiling,
            "explained_variance": self.explained,
            "label_variance": self.total,
            "bayes_risk": self.risk,
            "n_actions": self.n_actions,
            "cost": self.cost,
        }


def evaluate_protocol(label: LabelFunctional, K: Array, grid: TimeGrid,
                      actions: Sequence[Action]) -> CeilingReport:
    """Full ceiling report for a protocol expressed as a list of actions."""
    A, R = protocol_matrices(actions, grid)
    V = label_variance(label, K, grid.weights)
    F = explained_variance(label, K, A, R, grid.weights) if len(actions) else 0.0
    return CeilingReport(
        ceiling=F / V if V > 0 else 0.0,
        explained=F,
        total=V,
        risk=V - F,
        n_actions=len(actions),
        cost=float(sum(a.cost for a in actions)),
    )


# --------------------------------------------------------------------------
# Incremental (rank-one) machinery used by the design algorithms
# --------------------------------------------------------------------------
@dataclass
class ProtocolState:
    """Mutable state carried along a greedy design path.

    Holds ``Q`` (explained covariance), ``P = K - Q`` (residual covariance),
    the cached transform ``C_g(Q)`` and the current objective ``F_g``.
    """

    K: Array
    omega: Array
    label: LabelFunctional
    Q: Array
    P: Array
    F: float
    chosen: list[Action]
    cost: float

    @classmethod
    def empty(cls, label: LabelFunctional, K: Array, omega: Array) -> "ProtocolState":
        Q = np.zeros_like(K)
        return cls(K=K, omega=omega, label=label, Q=Q, P=K.copy(),
                   F=bilinear(label, Q, omega), chosen=[], cost=0.0)

    @classmethod
    def from_actions(cls, label: LabelFunctional, K: Array, grid: TimeGrid,
                     actions: Sequence[Action]) -> "ProtocolState":
        st = cls.empty(label, K, grid.weights)
        for a in actions:
            st = st.add(a, grid)
        return st

    # -- rank-one update ------------------------------------------------
    def gain_vector(self, ell: Array, noise: float) -> tuple[Array, float]:
        """Return ``(v, s)`` with ``v = P ell`` and ``s = ell^T P ell + nu^2``."""
        v = self.P @ ell
        s = float(ell @ v) + float(noise)
        return v, max(s, 1e-300)

    def marginal_gain(self, ell: Array, noise: float) -> float:
        """Exact ``Delta_g(a | S) = F_g(S + a) - F_g(S)``.

        For a linear (mean) label this collapses to the closed form
        ``(h^T P ell)^2 / (ell^T P ell + nu^2)``; for a nonlinear label the
        rank-one perturbation is pushed through ``C_g`` entrywise, which is the
        exact marginal gain rather than a surrogate.
        """
        v, s = self.gain_vector(ell, noise)
        if self.label.name == "mean":
            return float((self.omega @ v) ** 2 / s)
        Qnew = self.Q + np.outer(v, v) / s
        return bilinear(self.label, Qnew, self.omega) - self.F

    def add(self, action: Action, grid: TimeGrid) -> "ProtocolState":
        from .covariance import action_vector

        ell = action_vector(action, grid)
        v, s = self.gain_vector(ell, action.effective_noise)
        upd = np.outer(v, v) / s
        Q = self.Q + upd
        P = self.P - upd
        return ProtocolState(
            K=self.K, omega=self.omega, label=self.label, Q=Q, P=P,
            F=bilinear(self.label, Q, self.omega),
            chosen=self.chosen + [action], cost=self.cost + action.cost,
        )

    # -- reporting -------------------------------------------------------
    @property
    def ceiling(self) -> float:
        V = bilinear(self.label, self.K, self.omega)
        return self.F / V if V > 0 else 0.0


def marginal_gain_mean(K_residual: Array, h: Array, ell: Array, noise: float) -> float:
    """Closed-form mean-label marginal gain (`prop:marginal` in the paper)."""
    v = K_residual @ ell
    s = float(ell @ v) + float(noise)
    return float((h @ v) ** 2 / max(s, 1e-300))


# --------------------------------------------------------------------------
# Trait / state decomposition of the ceiling
# --------------------------------------------------------------------------
def trait_state_split(label: LabelFunctional, K: Array, A: Array, R: Array,
                      omega: Array, alpha: float) -> dict:
    """Split total explainability into trait and state channels.

    The trait channel conditions on nothing; the state channel conditions on
    the stable trait ``M``, i.e. it replaces ``K`` by the *state-only*
    correlation ``(K - alpha 1 1^T) / (1 - alpha)`` and the label by its
    trait-conditioned version.  Reporting only the total confuses ranking
    individuals with tracking them.
    """
    p = K.shape[0]
    ones = np.ones((p, p))
    total = protocol_ceiling(label, K, A, R, omega)
    if alpha >= 1.0:
        return {"total": total, "state": 0.0, "trait_share": 1.0}
    K_state = (K - alpha * ones) / (1.0 - alpha)
    K_state = 0.5 * (K_state + K_state.T)
    np.fill_diagonal(K_state, 1.0)
    state = protocol_ceiling(label, K_state, A, R, omega)
    return {"total": float(total), "state": float(state),
            "trait_share": float(alpha)}


def ceiling_utilization(model_r2: float, ceiling: float) -> dict:
    """Ceiling-utilisation ratio *and* absolute gap.

    The ratio is unstable when the ceiling is small, so both are always
    reported together.  Ceiling utilisation is not reported in the article.
    """
    gap = float(ceiling - model_r2)
    ratio = float(model_r2 / ceiling) if ceiling > 1e-12 else float("nan")
    return {"absolute_gap": gap, "utilization": ratio}
