"""S9: numerical checks of two proof hypotheses quoted in the manuscript.

(a) The calibration chain of Claim 1 of the uniform-error theorem claims
    ||K_hat - K|| <= (6||K|| + 4) e0 on the event e0 <= min{1/2, lam_min(K)/2}.
    We report the worst realised ratio to that bound; the manuscript quotes it
    to say the bound is conservative rather than tight.

(b) the submodularity-ratio transfer argument needs Q_S entrywise non-negative.  We verify it
    holds on MTP2 configurations and fails on a non-MTP2 control kernel, so the
    structural hypothesis is doing real work rather than being vacuous.

Writes results/s9_proof_checks.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from protocol_ceiling import (explained_covariance, make_kernel,
                              to_correlation, trait_state_correlation,
                              uniform_grid)
from protocol_ceiling.covariance import Action, protocol_matrices

N_CHAIN = 4000
N_PROTOCOLS = 400
HORIZON, P_GRID = 10.0, 40


def calibration_chain(n: int = N_CHAIN) -> dict:
    """Worst realised ratio of ||K_hat - K|| to the the covariance-repair claim bound."""
    rng = np.random.default_rng(20260803)
    worst_bound, worst_naive, kept = -np.inf, -np.inf, 0
    for _ in range(n):
        p = int(rng.integers(3, 9))
        A = rng.standard_normal((p, p))
        K = to_correlation(A @ A.T / p + 0.6 * np.eye(p))
        lam = float(np.linalg.eigvalsh(K).min())
        e0 = float(rng.uniform(1e-4, min(0.5, lam / 2)))
        E = rng.standard_normal((p, p))
        E = 0.5 * (E + E.T)
        E = e0 * E / np.linalg.norm(E, 2)
        Ktilde = K + E
        if np.linalg.eigvalsh(Ktilde).min() <= 0:
            continue          # off the event; the lemma says nothing there
        kept += 1
        d = np.sqrt(np.diag(Ktilde))
        Khat = Ktilde / np.outer(d, d)
        err = float(np.linalg.norm(Khat - K, 2))
        nrm = float(np.linalg.norm(K, 2))
        worst_bound = max(worst_bound, err / ((6.0 * nrm + 4.0) * e0))
        worst_naive = max(worst_naive, err / ((1.0 + nrm) * e0))
    return {"n_draws": n, "n_on_event": kept,
            "worst_ratio_to_bound": worst_bound,
            "worst_ratio_to_one_plus_norm": worst_naive}


def q_entrywise(kernel, alpha: float, n: int = N_PROTOCOLS,
                seed: int = 7) -> float:
    """Smallest entry of Q_S over n random protocols."""
    rng = np.random.default_rng(seed)
    grid = uniform_grid(HORIZON, P_GRID)
    K = trait_state_correlation(grid, alpha, kernel)
    worst = np.inf
    for _ in range(n):
        k = int(rng.integers(1, 5))
        acts = [Action(time=float(t), noise=0.3)
                for t in rng.uniform(0.3, HORIZON - 0.3, k)]
        A, R = protocol_matrices(acts, grid)
        worst = min(worst, float(explained_covariance(K, A, R).min()))
    return worst


def perturbation_bounds(n: int = 300, budget: int = 3, p_grid: int = 12,
                        noise: float = 0.5) -> dict:
    """Stress the two explicit-constant bounds of the resolvent claim and the uniform-error theorem.

    For random (K, K_hat) pairs and every protocol of size <= budget we compare
    the realised quantities against the constants the paper prints, and record
    the worst realised ratio.  A ratio above 1 falsifies the theorem.
    """
    from itertools import combinations

    from protocol_ceiling import (ThresholdLabel, evaluate_protocol,
                                  label_variance)
    from protocol_ceiling.estimation import (q_perturbation_constant,
                                             uniform_error_bound)

    rng = np.random.default_rng(11)
    grid = uniform_grid(HORIZON, p_grid)
    omega = grid.weights
    label = ThresholdLabel(c=0.0)
    L_g, beta = label.modulus()
    worst_q, worst_i, kept = -np.inf, -np.inf, 0

    for _ in range(n):
        K = trait_state_correlation(grid, float(rng.uniform(0.0, 0.4)),
                                    make_kernel("ou", tau=float(rng.uniform(0.5, 3.0))))
        E = rng.standard_normal((p_grid, p_grid))
        E = 0.5 * (E + E.T)
        np.fill_diagonal(E, 0.0)                  # keep the unit diagonal
        e = float(rng.uniform(1e-4, 2e-2))
        E *= e / np.linalg.norm(E, 2)
        Khat = K + E
        if np.linalg.eigvalsh(Khat).min() <= 1e-9:
            continue
        e = float(np.linalg.norm(Khat - K, 2))

        times = rng.uniform(0.3, HORIZON - 0.3, 6)
        kappa = max(float(np.linalg.norm(K, 2)), float(np.linalg.norm(Khat, 2)))
        a_B, lam_B = float(budget), noise
        if a_B * e > lam_B / 2:
            continue                              # off the event of the resolvent claim
        V = label_variance(label, K, omega)
        if L_g * e**beta >= V:
            continue                              # off the event of the uniform-error theorem
        kept += 1

        L_Q = q_perturbation_constant(kappa, a_B, lam_B)
        bnd = uniform_error_bound(label, K, omega, budget=budget,
                                  min_noise=lam_B, kappa=kappa, k_error=e)["bound"]
        for k in range(1, budget + 1):
            for combo in combinations(times, k):
                acts = [Action(time=float(t), noise=noise) for t in combo]
                A, R = protocol_matrices(acts, grid)
                dQ = np.linalg.norm(explained_covariance(Khat, A, R)
                                    - explained_covariance(K, A, R), 2)
                worst_q = max(worst_q, dQ / (L_Q * e))
                gap = abs(evaluate_protocol(label, Khat, grid, acts).ceiling
                          - evaluate_protocol(label, K, grid, acts).ceiling)
                worst_i = max(worst_i, gap / bnd)
    return {"n_draws": n, "n_on_event": kept,
            "worst_ratio_LQ": worst_q, "worst_ratio_uniform": worst_i}


def invisible_dimension(rows, p: int = 4, stationary: bool = True) -> int:
    """Dimension of the directions invisible to the stacked protocol.

    A direction is invisible when the augmented benchmark cannot see it:
    ``A+ D A+' = 0``, ``A+ D h = 0`` and ``h' D h = 0``, with ``D`` symmetric
    and zero-diagonal.  Both the constraint map and the value derivative are
    linear in ``D``, so the augmentation test of the paper is a rank
    computation.  ``stationary=True`` restricts ``D`` to the Toeplitz
    directions of the stationary model.
    """
    import itertools
    Ap = np.atleast_2d(np.asarray(rows, float))
    h = np.full(p, 1.0 / p)
    if stationary:
        basis = []
        for lag in range(1, p):
            D = np.zeros((p, p))
            for i in range(p):
                for j in range(p):
                    if abs(i - j) == lag:
                        D[i, j] = 1.0
            basis.append(D)
    else:
        basis = []
        for i, j in itertools.combinations(range(p), 2):
            D = np.zeros((p, p))
            D[i, j] = D[j, i] = 1.0
            basis.append(D)
    C = np.array([list((Ap @ D @ Ap.T).ravel())
                  + list((Ap @ D @ h).ravel()) + [h @ D @ h] for D in basis]).T
    s = np.linalg.svd(C, compute_uv=False)
    tol = max(C.shape) * np.finfo(float).eps * (s[0] if s.size else 1.0)
    return len(basis) - int((s > tol).sum())


def augmentation_dimensions() -> dict:
    """How fast one extra point observation collapses the invisible space."""
    e = np.eye(4)
    return {
        "stationary_A_only": invisible_dimension([e[0]], stationary=True),
        "stationary_plus_z1": invisible_dimension([e[0], e[1]], stationary=True),
        "free_A_only": invisible_dimension([e[0]], stationary=False),
        "free_plus_z1": invisible_dimension([e[0], e[1]], stationary=False),
        "free_plus_z1_z2": invisible_dimension([e[0], e[1], e[2]], stationary=False),
    }


def permutation_witness(p: int = 6, c: float = 0.4, noise: float = 0.05,
                        seed: int = 11) -> dict:
    """Numerical witness for the permutation form of non-identification.

    With ``A P = A`` and ``P' omega = omega`` the joint law of ``(Y_A, Theta_g)``
    is identical under ``K`` and ``K' = P K P'`` for *every* square-integrable
    ``g`` -- exactly, not to second order.  A non-``P``-invariant row space is
    *necessary* for the two values to differ but not sufficient: on a
    ``P``-invariant ``K`` they coincide however ``B`` moves.  What the witness
    below exhibits is a concrete ``(K, B)`` on which they do differ.  The
    joint-law check is done on the exact second and third moments of the pair
    rather than by simulation, so the residual below is machine precision and
    not Monte Carlo noise.
    """
    from protocol_ceiling import ThresholdLabel, to_correlation
    from protocol_ceiling.risk import bilinear, explained_covariance, label_variance

    rng = np.random.default_rng(seed)
    W = rng.standard_normal((p, p + 4))
    K = to_correlation(W @ W.T / (p + 4))
    omega = np.full(p, 1.0 / p)
    A = np.eye(p)[[0]]
    perm = np.array([0, 3, 1, 5, 2, 4])          # fixes the observed coordinate
    P = np.eye(p)[perm]
    Kp = P @ K @ P.T
    lab = ThresholdLabel(c=c)

    # every functional the benchmark identifies, for the threshold target
    def blocks(M):
        return [float((A @ M @ A.T).ravel()[0]), float((A @ M @ omega).ravel()[0]),
                float(omega @ M @ omega),
                float(label_variance(lab, M, omega)),
                float(bilinear(lab, explained_covariance(M, A, np.array([[noise]])),
                               omega))]
    d_joint = max(abs(a - b) for a, b in zip(blocks(K), blocks(Kp)))

    gaps = {}
    for idx in ([1, 2], [2, 4], [1, 5]):
        B = np.eye(p)[idx]
        R = noise * np.eye(len(idx))
        v = bilinear(lab, explained_covariance(K, B, R), omega) / label_variance(lab, K, omega)
        vp = bilinear(lab, explained_covariance(Kp, B, R), omega) / label_variance(lab, Kp, omega)
        gaps["+".join(f"Z{i}" for i in idx)] = {"I_K": v, "I_Kprime": vp,
                                                "gap": abs(v - vp)}
    return {"p": p, "threshold_c": c, "noise": noise, "permutation": perm.tolist(),
            "AP_equals_A": bool(np.allclose(A @ P, A)),
            "P_preserves_weights": bool(np.allclose(P.T @ omega, omega)),
            "K_differs": float(np.abs(Kp - K).max()),
            "lambda_min_Kprime": float(np.linalg.eigvalsh(Kp).min()),
            "max_abs_discrepancy_identified_functionals": d_joint,
            "value_gaps": gaps,
            "max_value_gap": max(v["gap"] for v in gaps.values())}


def main() -> None:
    chain = calibration_chain()
    mtp2 = {}
    for name, kw in (("ou", dict(tau=1.0)),
                     ("two_scale_ou",
                      dict(tau_fast=0.15, tau_slow=3.0, w_fast=0.6))):
        for alpha in (0.0, 0.3):
            mtp2[f"{name}_alpha{alpha:g}"] = q_entrywise(
                make_kernel(name, **kw), alpha)
    control = q_entrywise(make_kernel("matern32", tau=1.0), 0.0)
    pert = perturbation_bounds()
    aug = augmentation_dimensions()
    perm = permutation_witness()

    out = {
        "headline": {
            "chain_n_draws": chain["n_draws"],
            "chain_worst_ratio": chain["worst_ratio_to_bound"],
            "n_protocols_per_config": N_PROTOCOLS,
            "n_mtp2_configs": len(mtp2),
            "mtp2_min_q_entry": min(mtp2.values()),
            "control_min_q_entry": control,
            "worst_ratio_LQ": pert["worst_ratio_LQ"],
            "worst_ratio_uniform": pert["worst_ratio_uniform"],
            **{f"aug_{k}": v for k, v in aug.items()},
            "perm_joint_discrepancy":
                perm["max_abs_discrepancy_identified_functionals"],
            "perm_max_value_gap": perm["max_value_gap"],
        },
        "calibration_chain": chain,
        "mtp2_min_q_entry_by_config": mtp2,
        "control_matern32_min_q_entry": control,
        "perturbation_bounds": pert,
        "augmentation_dimensions": aug,
        "permutation_witness": perm,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "s9_proof_checks.json").write_text(
        json.dumps(out, indent=2))

    print(f"calibration chain: worst ratio to bound "
          f"{chain['worst_ratio_to_bound']:.4f} over {chain['n_on_event']} "
          f"on-event draws (of {chain['n_draws']})")
    for k, v in mtp2.items():
        print(f"  MTP2 {k:26s} min Q_S entry {v:+.3e}")
    print(f"  control matern32           min Q_S entry {control:+.3e}")
    print(f"L_Q bound: worst realised ratio {pert['worst_ratio_LQ']:.4f} "
          f"over {pert['n_on_event']} on-event draws")
    print(f"uniform-error bound: worst realised ratio "
          f"{pert['worst_ratio_uniform']:.4e}")
    assert chain["worst_ratio_to_bound"] <= 1.0, "the covariance-repair claim bound violated"
    assert min(mtp2.values()) > -1e-8, "MTP2 hypothesis violated"
    assert control < 0.0, "control kernel failed to violate the hypothesis"
    assert pert["worst_ratio_LQ"] <= 1.0, "the resolvent claim constant violated"
    assert pert["worst_ratio_uniform"] <= 1.0, "the uniform-error theorem bound violated"
    assert perm["max_abs_discrepancy_identified_functionals"] < 1e-12, \
        "the permutation pair is not observationally equivalent"
    assert perm["max_value_gap"] > 1e-3, \
        "the permutation pair does not separate any alternative protocol"
    print("wrote results/s9_proof_checks.json")


if __name__ == "__main__":
    main()
