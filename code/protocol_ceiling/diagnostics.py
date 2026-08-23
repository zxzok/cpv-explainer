"""Risk decomposition, ceiling utilisation, and simple learners.

The companion preprint splits the excess risk of any *learned* predictor into
three additive pieces,

    R(f_hat_S) - R_full* = [R_S* - R_full*]              protocol gap
                           + [inf_{f in F} R(f) - R_S*]  approximation gap
                           + [R(f_hat_S) - inf_F R(f)]   estimation gap,

which respond to three different interventions: change the acquisition, change
the model class, change the training-set size.  The routines here compute the
empirical counterparts, together with the learners used in the
protocol-versus-architecture experiment.  The learners are deliberately
dependency-free (ridge, kernel ridge / GP regression, and a small numpy MLP) so
that the whole pipeline runs with numpy and scipy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


# --------------------------------------------------------------------------
# Risk decomposition
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskDecomposition:
    full_risk: float
    protocol_risk: float
    class_risk: float
    learned_risk: float

    @property
    def protocol_gap(self) -> float:
        return float(self.protocol_risk - self.full_risk)

    @property
    def approximation_gap(self) -> float:
        return float(self.class_risk - self.protocol_risk)

    @property
    def estimation_gap(self) -> float:
        return float(self.learned_risk - self.class_risk)

    def as_dict(self) -> dict:
        return {
            "full_risk": self.full_risk,
            "protocol_risk": self.protocol_risk,
            "class_risk": self.class_risk,
            "learned_risk": self.learned_risk,
            "protocol_gap": self.protocol_gap,
            "approximation_gap": self.approximation_gap,
            "estimation_gap": self.estimation_gap,
        }


def r2_score(y: Array, pred: Array, sample_weight: Array | None = None) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if sample_weight is None:
        w = np.ones(y.size, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float)
        if w.shape != y.shape or np.any(w < 0) or not np.any(w > 0):
            raise ValueError("sample_weight must be nonnegative and match y")
    mean_y = float(np.sum(w * y) / np.sum(w))
    ss_res = float(np.sum(w * (y - pred) ** 2))
    ss_tot = float(np.sum(w * (y - mean_y) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def within_between_r2(y: Array, pred: Array, subject: Array) -> dict:
    """Split predictive accuracy into between- and within-subject components.

    A model can rank subjects well (high between-subject ``R^2``) while
    tracking none of them (near-zero within-subject ``R^2``); reporting only the
    total conflates the two.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    subject = np.asarray(subject)
    uniq = np.unique(subject)
    ybar = np.array([y[subject == s].mean() for s in uniq])
    pbar = np.array([pred[subject == s].mean() for s in uniq])
    y_between = np.zeros_like(y)
    p_between = np.zeros_like(pred)
    for s, yb, pb in zip(uniq, ybar, pbar):
        y_between[subject == s] = yb
        p_between[subject == s] = pb
    return {
        "total": r2_score(y, pred),
        "between": r2_score(ybar, pbar),
        "within": r2_score(y - y_between, pred - p_between),
    }


# --------------------------------------------------------------------------
# Learners (numpy only)
# --------------------------------------------------------------------------
def ridge_fit_predict(Xtr: Array, ytr: Array, Xte: Array,
                      alphas: tuple[float, ...] = tuple(np.logspace(-6.0, 2.0, 25)),
                      n_folds: int = 3, rng: np.random.Generator | None = None,
                      groups: Array | None = None,
                      sample_weight: Array | None = None,
                      n_unpenalized: int = 0) -> Array:
    """Ridge with an intercept and leakage-free inner-CV standardisation.

    Features are centred and scaled using each inner-training split, squared
    error selects the penalty, the smallest penalty wins an exact tie, and the
    final fit uses the complete outer-training set. Predictions are not clipped.
    When ``groups`` is supplied, all rows of a group remain in one inner fold.
    ``sample_weight`` is used consistently for standardisation, ridge fitting
    and inner-validation loss. The first ``n_unpenalized`` columns are baseline
    covariates: they are centred and scaled but receive penalty factor zero;
    the ridge penalty is applied only to the remaining columns. A feature with weighted variance at most
    ``1e-12`` is retained with scale one, so constant columns cannot divide by
    zero and receive a zero fitted coefficient after centring.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    Xtr = np.asarray(Xtr, dtype=float)
    ytr = np.asarray(ytr, dtype=float)
    n = Xtr.shape[0]
    if not 0 <= n_unpenalized <= Xtr.shape[1]:
        raise ValueError("n_unpenalized must lie between zero and n_features")
    if sample_weight is None:
        sample_weight = np.ones(n, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if (sample_weight.shape != (n,) or np.any(sample_weight < 0)
                or not np.any(sample_weight > 0)):
            raise ValueError("sample_weight must be nonnegative with one entry per row")
    if groups is None:
        group_values = np.arange(n)
    else:
        group_values = np.asarray(groups)
        if group_values.shape != (n,):
            raise ValueError("groups must have one entry per training row")
    unique_groups = np.unique(group_values)
    n_inner = max(2, min(n_folds, unique_groups.size))
    group_order = rng.permutation(unique_groups)
    group_fold = {g: i % n_inner for i, g in enumerate(group_order)}
    folds = np.array([group_fold[g] for g in group_values], dtype=int)
    best_a, best_err = alphas[0], np.inf
    penalty = np.ones(Xtr.shape[1], dtype=float)
    penalty[:n_unpenalized] = 0.0

    def solve(G: Array, rhs: Array) -> Array:
        """Solve the penalised equations, tolerating redundant baselines."""
        try:
            return np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(G, rcond=1e-12) @ rhs

    def weighted_location_scale(X: Array, y: Array, weight: Array
                                ) -> tuple[Array, Array, float, Array]:
        w = weight / np.sum(weight)
        mu_x = w @ X
        var_x = w @ ((X - mu_x) ** 2)
        sd_x = np.sqrt(np.maximum(var_x, 0.0))
        sd_x = np.where(sd_x > 1e-12, sd_x, 1.0)
        mu_y = float(w @ y)
        # Mean-one weights keep the ridge-penalty scale identical to the
        # unweighted implementation when every row has equal weight.
        fit_w = weight * (weight.size / np.sum(weight))
        return mu_x, sd_x, mu_y, fit_w

    for a in alphas:
        err = 0.0
        for f in np.unique(folds):
            tr, va = folds != f, folds == f
            mu_x, sd_x, mu_y, fit_w = weighted_location_scale(
                Xtr[tr], ytr[tr], sample_weight[tr])
            Ztr = (Xtr[tr] - mu_x) / sd_x
            Zva = (Xtr[va] - mu_x) / sd_x
            yc = ytr[tr] - mu_y
            G = Ztr.T @ (fit_w[:, None] * Ztr) + a * np.diag(penalty)
            coef = solve(G, Ztr.T @ (fit_w * yc))
            err += float(np.sum(sample_weight[va]
                                * (ytr[va] - (mu_y + Zva @ coef)) ** 2))
        if err < best_err:
            best_a, best_err = a, err
    mu_x, sd_x, mu_y, fit_w = weighted_location_scale(Xtr, ytr, sample_weight)
    Z = (Xtr - mu_x) / sd_x
    yc = ytr - mu_y
    G = Z.T @ (fit_w[:, None] * Z) + best_a * np.diag(penalty)
    coef = solve(G, Z.T @ (fit_w * yc))
    return ((np.asarray(Xte, dtype=float) - mu_x) / sd_x) @ coef + mu_y


def kernel_ridge_fit_predict(Xtr: Array, ytr: Array, Xte: Array,
                             length_scale: float | None = None,
                             reg: float = 1e-3) -> Array:
    """Gaussian-process regression (kernel ridge with an RBF kernel)."""
    Xtr = np.asarray(Xtr, dtype=float)
    Xte = np.asarray(Xte, dtype=float)
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-12
    A = (Xtr - mu_x) / sd_x
    B = (Xte - mu_x) / sd_x
    if length_scale is None:
        d2 = np.sum((A[:200, None, :] - A[None, :200, :]) ** 2, axis=-1)
        length_scale = float(np.sqrt(np.median(d2[d2 > 0]))) if np.any(d2 > 0) else 1.0
    def rbf(P: Array, Q: Array) -> Array:
        d2 = (np.sum(P**2, 1)[:, None] + np.sum(Q**2, 1)[None, :] - 2.0 * P @ Q.T)
        return np.exp(-np.maximum(d2, 0.0) / (2.0 * length_scale**2))
    mu_y = ytr.mean()
    Ktr = rbf(A, A) + reg * np.eye(A.shape[0])
    alpha = np.linalg.solve(Ktr, ytr - mu_y)
    return rbf(B, A) @ alpha + mu_y


def mlp_fit_predict(Xtr: Array, ytr: Array, Xte: Array, hidden: int = 64,
                    epochs: int = 400, lr: float = 3e-3, weight_decay: float = 1e-4,
                    rng: np.random.Generator | None = None) -> Array:
    """Two-layer tanh MLP trained by full-batch Adam (numpy implementation)."""
    rng = np.random.default_rng(0) if rng is None else rng
    Xtr = np.asarray(Xtr, dtype=float)
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-12
    A = (Xtr - mu_x) / sd_x
    B = (np.asarray(Xte, dtype=float) - mu_x) / sd_x
    mu_y, sd_y = ytr.mean(), ytr.std() + 1e-12
    y = (ytr - mu_y) / sd_y

    d = A.shape[1]
    W1 = rng.standard_normal((d, hidden)) / np.sqrt(d)
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, 1)) / np.sqrt(hidden)
    b2 = np.zeros(1)
    params = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1_, b2_, eps = 0.9, 0.999, 1e-8

    for step in range(1, epochs + 1):
        H = np.tanh(A @ W1 + b1)
        pred = (H @ W2 + b2).ravel()
        resid = pred - y
        gpred = (2.0 / len(y)) * resid[:, None]
        gW2 = H.T @ gpred + weight_decay * W2
        gb2 = gpred.sum(0)
        gH = gpred @ W2.T * (1.0 - H**2)
        gW1 = A.T @ gH + weight_decay * W1
        gb1 = gH.sum(0)
        grads = [gW1, gb1, gW2, gb2]
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = b1_ * m[i] + (1 - b1_) * g
            v[i] = b2_ * v[i] + (1 - b2_) * g * g
            mhat = m[i] / (1 - b1_**step)
            vhat = v[i] / (1 - b2_**step)
            p -= lr * mhat / (np.sqrt(vhat) + eps)
    H = np.tanh(B @ W1 + b1)
    return ((H @ W2 + b2).ravel()) * sd_y + mu_y


LEARNERS = {
    "ridge": ridge_fit_predict,
    "gp": kernel_ridge_fit_predict,
    "mlp": mlp_fit_predict,
}
