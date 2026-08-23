"""Tests for the identifiability constructions and the design algorithms."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol_ceiling.risk import ProtocolState
from protocol_ceiling import (Action, MeanLabel, ThresholdLabel, TwoSidedLabel,
                              candidate_actions, certify, counting_bound,
                              explained_covariance, to_correlation,
                              design_imse, design_kernel_quadrature,
                              design_mutual_information, design_uniform,
                              evaluate_protocol, find_submodularity_violation,
                              linear_ceiling, make_kernel, max_psd_step,
                              minimal_stationary_example,
                              nonidentified_directions, nonlinear_ratio_lower_bound,
                              observed_discrepancy, select_protocol_exhaustive,
                              select_protocol_greedy,
                              stationary_identification_jacobian,
                              submodularity_ratio_certificate,
                              trait_ceiling_interval, trait_share_interval,
                              trait_state_correlation, uniform_grid)
from protocol_ceiling.covariance import action_vector, protocol_matrices

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# --------------------------------------------------------------------------
def test_minimal_stationary_counterexample() -> None:
    ex = minimal_stationary_example(tau=1.0)
    cert = ex["certificate"]
    Kp, Km = cert.K_plus, cert.K_minus
    print(f"       eps = {cert.eps:.6f}")
    print(f"       rho_+ = {np.round(Kp[0], 6)}")
    print(f"       rho_- = {np.round(Km[0], 6)}")
    print(f"       I_B(K_+) = {cert.ceiling_plus:.6f}   I_B(K_-) = {cert.ceiling_minus:.6f}")
    check("both perturbed matrices are correlation matrices",
          abs(np.diag(Kp) - 1).max() < 1e-12 and abs(np.diag(Km) - 1).max() < 1e-12)
    check("both perturbed matrices are positive definite",
          np.linalg.eigvalsh(Kp).min() > 0 and np.linalg.eigvalsh(Km).min() > 0,
          f"lambda_min {min(np.linalg.eigvalsh(Kp).min(), np.linalg.eigvalsh(Km).min()):.4f}")
    check("observed benchmark is exactly unchanged",
          cert.observed_discrepancy < 1e-14, f"discrepancy {cert.observed_discrepancy:.2e}")
    check("counterfactual ceilings differ", cert.ceiling_gap > 1e-3,
          f"gap {cert.ceiling_gap:.6f}")


def test_full_joint_law_is_identical() -> None:
    """Beyond the three functionals: the whole Gaussian law of (Y_A, Theta) matches."""
    ex = minimal_stationary_example(tau=1.0, noise=0.25)
    cert = ex["certificate"]
    A, h = ex["A"], ex["h"]
    for K, tag in ((cert.K_plus, "+"), (cert.K_minus, "-")):
        G = np.vstack([A, h[None, :]])
        cov = G @ K @ G.T
        cov[0, 0] += 0.25
        if tag == "+":
            ref = cov
        else:
            err = float(np.max(np.abs(cov - ref)))
            check("joint covariance of (Y_A, Theta) is identical", err < 1e-14,
                  f"max err {err:.2e}")
            print(f"       Cov(Y_A,Theta) = {ref[0,1]:.10f}, Var(Theta) = {ref[1,1]:.10f}")




def test_closed_form_gap_matches_theorem() -> None:
    """Verify Eq. (3.9)-(3.10) of the paper against the numerical construction.

        I(B; K_pm) = 2 b^2 / [{1 + nu_B^2 + rho(1) +- eps} Var(Theta)],
        b = {1 + 2 rho(1) + rho(2)} / 4,

    and the resulting gap
        I(B; K_-) - I(B; K_+)
            = 4 b^2 eps / [Var(Theta) ({1+nu_B^2+rho(1)}^2 - eps^2)].
    """
    for nu_b in (0.0, 0.25):
        ex = minimal_stationary_example(tau=1.0, noise=nu_b)
        cert = ex["certificate"]
        rho0 = ex["K0"][0]
        # eps on the raw lag scale: the package normalises Delta by its spectral norm
        eps = float(cert.K_plus[0, 1] - rho0[1])
        b = (1.0 + 2.0 * rho0[1] + rho0[2]) / 4.0
        var_theta = float(ex["h"] @ ex["K0"] @ ex["h"])
        pred_plus = 2 * b**2 / ((1 + nu_b + rho0[1] + eps) * var_theta)
        pred_minus = 2 * b**2 / ((1 + nu_b + rho0[1] - eps) * var_theta)
        pred_gap = (4 * b**2 * eps
                    / (var_theta * ((1 + nu_b + rho0[1]) ** 2 - eps**2)))
        print(f"       nu_B^2={nu_b}: eps={eps:.6f}  b={b:.6f}  Var(Theta)={var_theta:.6f}")
        print(f"       closed form  I+={pred_plus:.9f}  I-={pred_minus:.9f}  gap={pred_gap:.9f}")
        print(f"       numerical    I+={cert.ceiling_plus:.9f}  I-={cert.ceiling_minus:.9f}"
              f"  gap={cert.ceiling_gap:.9f}")
        check(f"closed-form I(B;K_+) at nu_B^2={nu_b}",
              abs(pred_plus - cert.ceiling_plus) < 1e-10,
              f"|diff| {abs(pred_plus - cert.ceiling_plus):.2e}")
        check(f"closed-form I(B;K_-) at nu_B^2={nu_b}",
              abs(pred_minus - cert.ceiling_minus) < 1e-10,
              f"|diff| {abs(pred_minus - cert.ceiling_minus):.2e}")
        check(f"closed-form gap at nu_B^2={nu_b}",
              abs(pred_gap - cert.ceiling_gap) < 1e-10,
              f"|diff| {abs(pred_gap - cert.ceiling_gap):.2e}")
        # b is invariant under the perturbation: 2(+eps) + (-2 eps) = 0
        b_plus = (1.0 + 2.0 * cert.K_plus[0, 1] + cert.K_plus[0, 2]) / 4.0
        b_minus = (1.0 + 2.0 * cert.K_minus[0, 1] + cert.K_minus[0, 2]) / 4.0
        check(f"Cov(Z_1,Theta) invariant at nu_B^2={nu_b}",
              abs(b_plus - b) < 1e-12 and abs(b_minus - b) < 1e-12,
              f"b={b:.9f}, b+={b_plus:.9f}, b-={b_minus:.9f}")


def test_sharpness_p_equals_three() -> None:
    """p = 3 is locally identified; p >= 4 is not.  This makes p = 4 minimal."""
    for p in (3, 4, 5, 6):
        J = stationary_identification_jacobian(p, obs_index=0)
        rank = int(np.linalg.matrix_rank(J, tol=1e-10))
        dirs = nonidentified_directions(
            np.eye(4)[:1] if p == 4 else np.eye(p)[:1],
            np.full(p, 1.0 / p), stationary=True, unit_diagonal=True)
        cb = counting_bound(p, 1, stationary=True)
        print(f"       p={p}: free={cb['free_parameters']} constraints={cb['constraints']}"
              f" rank(J)={rank} n_directions={len(dirs)}")
        if p == 3:
            check("p=3 is locally identified (full-rank Jacobian, no direction)",
                  rank == 2 and len(dirs) == 0)
        else:
            check(f"p={p} admits a non-identified direction", len(dirs) == p - 1 - 2)


def test_general_direction_search() -> None:
    """Random higher-dimensional instances also admit invisible perturbations."""
    rng = np.random.default_rng(11)
    p, d = 12, 3
    hits = 0
    for _ in range(20):
        idx = rng.choice(p, size=d, replace=False)
        A = np.eye(p)[idx]
        h = rng.uniform(0.5, 1.5, p)
        h = h / h.sum()
        dirs = nonidentified_directions(A, h, stationary=False, unit_diagonal=True)
        if len(dirs) > 0:
            hits += 1
    check("non-identified directions exist generically in higher dimension",
          hits == 20, f"{hits}/20 instances")
    cb = counting_bound(12, 3, stationary=False)
    print(f"       p=12,d=3: free={cb['free_parameters']} constraints={cb['constraints']}"
          f" deficiency={cb['deficiency']}")


def test_partial_identification_interval() -> None:
    alpha_true, rho_L = 0.30, 0.05
    r_L = alpha_true + (1 - alpha_true) * rho_L
    lo, hi = trait_share_interval(r_L, rho_bound=0.10)
    print(f"       r_L={r_L:.4f} -> alpha in [{lo:.4f}, {hi:.4f}] (truth {alpha_true})")
    check("trait-share interval covers the truth", lo <= alpha_true <= hi)
    clo, chi = trait_ceiling_interval((lo, hi), D=4, M=4, sigma_eps_sq=0.5)
    print(f"       I_trait(4,4) in [{clo:.4f}, {chi:.4f}]")
    check("ceiling interval is ordered and in [0,1]", 0 <= clo <= chi <= 1)
    check("interval degenerates when rho(L) is known to vanish",
          abs(trait_share_interval(r_L, 0.0)[0] - trait_share_interval(r_L, 0.0)[1]) < 1e-12)


# --------------------------------------------------------------------------


def test_mtp2_classification() -> None:
    """Which kernels make the posterior-covariance increments entrywise non-negative.

    ``Z`` is MTP2 iff ``K^{-1}`` is an M-matrix, i.e. has no positive
    off-diagonal entry.  The paper claims OU, its trait-state extension and OU
    mixtures qualify while Matern / squared-exponential / periodic / Cauchy do
    not; this pins that claim down.
    """
    grid = uniform_grid(10.0, 40)

    def max_offdiag_precision(K):
        P = np.linalg.inv(K)
        return float((P - np.diag(np.diag(P))).max())

    expected = {"ou": True, "two_scale_ou": True, "matern32": False,
                "matern52": False, "se": False, "periodic": False,
                "cauchy": False}
    kwargs = {"ou": dict(tau=1.0), "two_scale_ou": dict(tau_fast=0.15, tau_slow=3.0,
                                                        w_fast=0.6),
              "matern32": dict(tau=1.0), "matern52": dict(tau=1.0),
              "se": dict(tau=1.0), "periodic": dict(tau=6.0, period=3.0),
              "cauchy": dict(tau=1.0, beta=0.7)}
    for name, is_mtp2 in expected.items():
        for alpha in (0.0, 0.3):
            K = trait_state_correlation(grid, alpha, make_kernel(name, **kwargs[name]))
            m = max_offdiag_precision(K)
            got = m <= 1e-9
            print(f"       {name:14s} alpha={alpha}: max off-diagonal of K^-1 = {m:+.3e}"
                  f"  -> {'MTP2' if got else 'not MTP2'}")
            check(f"{name} (alpha={alpha}) MTP2 classification", got == is_mtp2)

    # The property the theorem actually uses: non-negative posterior covariance.
    rng = np.random.default_rng(0)
    for name, is_mtp2 in (("ou", True), ("matern32", False)):
        K = trait_state_correlation(grid, 0.0, make_kernel(name, **kwargs[name]))
        worst = np.inf
        for _ in range(200):
            acts = [Action(time=float(t), noise=0.3)
                    for t in rng.uniform(0.3, 9.7, int(rng.integers(0, 4)))]
            st = ProtocolState.from_actions(MeanLabel(), K, grid, acts)
            worst = min(worst, float(st.P.min()))
        print(f"       {name:14s}: min entry of P_S over 200 protocols = {worst:+.3e}")
        check(f"{name}: posterior covariance sign matches MTP2 status",
              (worst > -1e-9) == is_mtp2, f"min {worst:+.3e}")




def test_paper_nonsubmodularity_witness() -> None:
    """The explicit 4x4 non-submodularity witness must reproduce exactly.

    A regression target retained from v3.0.  The matrix and its two marginal
    gains are no longer printed in the article, which reports the aggregate
    exhaustive search of sec:synthetic (7.2) instead.
    """
    K = np.array([
        [1.0, 0.699995, -0.297399, -0.614794],
        [0.699995, 1.0, -0.311964, -0.241546],
        [-0.297399, -0.311964, 1.0, 0.614067],
        [-0.614794, -0.241546, 0.614067, 1.0]])
    h = np.full(4, 0.25)
    nu = 0.1
    minors = [float(np.linalg.det(K[:k, :k])) for k in (1, 2, 3, 4)]
    print(f"       leading minors: {[round(m, 6) for m in minors]}")
    check("witness matrix is positive definite",
          all(m > 0 for m in minors) and float(np.linalg.eigvalsh(K).min()) > 0,
          f"lambda_min {float(np.linalg.eigvalsh(K).min()):.6f}")
    for got, want in zip(minors, (1.0, 0.510007, 0.454127, 0.138245)):
        check(f"minor {want}", abs(got - want) < 5e-7, f"{got:.6f}")

    from protocol_ceiling.risk import bilinear, explained_covariance

    def F(label, idx):
        if not idx:
            return 0.0
        A = np.eye(4)[list(idx)]
        return bilinear(label, explained_covariance(K, A, nu * np.eye(len(idx))), h)

    def delta(label, a, S):
        return F(label, tuple(sorted(set(S) | {a}))) - F(label, tuple(sorted(S)))

    # 1-based indexing of the recorded witness: S = {}, T = {1}, a = 3
    for name, lab, want_S, want_T in (
            ("mean", MeanLabel(), 0.057354, 0.090893),
            ("occupation c=0", ThresholdLabel(0.0), 0.012008, 0.017524)):
        dS, dT = delta(lab, 2, ()), delta(lab, 2, (0,))
        print(f"       {name:16s} Delta(a|S)={dS:.6f}  Delta(a|T)={dT:.6f}")
        check(f"{name}: Delta(a|S) matches the recorded witness", abs(dS - want_S) < 1e-6,
              f"{dS:.6f} vs {want_S}")
        check(f"{name}: Delta(a|T) matches the recorded witness", abs(dT - want_T) < 1e-6,
              f"{dT:.6f} vs {want_T}")
        check(f"{name}: diminishing returns fails", dT > dS)


def test_submodularity_ratio_is_capped() -> None:
    """Singleton Omega is admissible, so the certified ratio cannot exceed one."""
    grid = uniform_grid(10.0, 64)
    K = trait_state_correlation(grid, 0.15, make_kernel("ou", tau=1.0))
    cands = candidate_actions(grid, n_times=8, noise=0.5)
    for lab in (MeanLabel(), ThresholdLabel(c=0.5)):
        cert = submodularity_ratio_certificate(
            lab, K, grid, cands, base_sets=[[], [cands[2]]],
            rng=np.random.default_rng(3), n_subsets=30)
        print(f"       {lab.name:12s} gamma={cert['gamma']:.4f}  "
              f"greedy factor={cert['greedy_factor']:.4f}")
        check(f"{lab.name}: gamma <= 1", cert["gamma"] <= 1.0 + 1e-12)
        check(f"{lab.name}: greedy factor <= 1 - 1/e",
              cert["greedy_factor"] <= 1.0 - np.exp(-1.0) + 1e-12)




def test_no_adaptivity_gain_linear() -> None:
    """Theorem: for a linear target the EVI is free of the observed values."""
    from protocol_ceiling.adaptive import expected_value_of_information
    rng = np.random.default_rng(0)
    grid = uniform_grid(10.0, 12)
    A = rng.standard_normal((12, 12))
    P = A @ A.T / 12 + 0.3 * np.eye(12)
    om = grid.weights
    ell = np.zeros(12); ell[4] = 1.0
    nu = 0.4
    base = expected_value_of_information(MeanLabel(), np.zeros(12), P, om, ell, nu)
    closed = float((om @ (P @ ell)) ** 2 / (float(ell @ P @ ell) + nu))
    check("mean-label EVI equals the closed-form static gain",
          abs(base - closed) < 1e-12, f"|diff| {abs(base - closed):.2e}")
    shifts = [expected_value_of_information(MeanLabel(), np.full(12, d), P, om, ell, nu)
              for d in (-3.0, -1.0, 0.0, 1.0, 3.0)]
    spread = float(max(shifts) - min(shifts))
    print(f"       EVI across posterior-mean shifts: spread {spread:.2e}")
    check("mean-label EVI is value-free", spread < 1e-12)

    lab = ThresholdLabel(c=0.0)
    nz = [expected_value_of_information(lab, np.full(12, d), P, om, ell, nu, n_nodes=24)
          for d in (-2.0, -0.5, 0.0, 0.5, 2.0)]
    ratio = max(nz) / max(min(nz), 1e-300)
    print(f"       threshold-label EVI varies by a factor of {ratio:.1f}")
    check("threshold-label EVI is value-dependent", ratio > 2.0)




def test_calibration_chain_constant() -> None:
    """Claim 1 of sec:estimation (covariance repair): ||K_hat-K|| <= (6||K||+4) e0."""
    rng = np.random.default_rng(0)
    worst = -np.inf
    for _ in range(2000):
        p = int(rng.integers(3, 9))
        A = rng.standard_normal((p, p))
        K = to_correlation(A @ A.T / p + 0.6 * np.eye(p))
        lam = float(np.linalg.eigvalsh(K).min())
        e0 = float(rng.uniform(1e-4, min(0.5, lam / 2)))
        E = rng.standard_normal((p, p)); E = 0.5 * (E + E.T)
        E = e0 * E / np.linalg.norm(E, 2)
        Kt = K + E
        if np.linalg.eigvalsh(Kt).min() <= 0:
            continue
        D = np.sqrt(np.diag(Kt))
        Khat = Kt / np.outer(D, D)
        bound = (6.0 * float(np.linalg.norm(K, 2)) + 4.0) * e0
        worst = max(worst, float(np.linalg.norm(Khat - K, 2)) / bound)
    print(f"       worst realised ratio to the bound: {worst:.4f}")
    check("calibration-chain bound holds on 2000 instances", worst <= 1.0,
          f"max ratio {worst:.4f}")
    check("the bound is conservative but not absurd", 0.001 < worst <= 1.0)


def test_mtp2_hypothesis_of_transfer_theorem() -> None:
    """The transfer bound needs Q_S >= 0 entrywise; MTP2 delivers it, Matern does not.

    The bound itself was dropped from the article during compression; the
    hypothesis is kept under test because the design experiments rely on it.
    """
    from protocol_ceiling.covariance import protocol_matrices
    rng = np.random.default_rng(1)
    grid = uniform_grid(10.0, 40)

    def min_entry(kernel, alpha, n=200):
        K = trait_state_correlation(grid, alpha, kernel)
        worst = np.inf
        for _ in range(n):
            acts = [Action(time=float(t), noise=0.3)
                    for t in rng.uniform(0.3, 9.7, int(rng.integers(1, 5)))]
            A, R = protocol_matrices(acts, grid)
            worst = min(worst, float(explained_covariance(K, A, R).min()))
        return worst

    for name, kw in (("ou", dict(tau=1.0)),
                     ("two_scale_ou", dict(tau_fast=0.15, tau_slow=3.0, w_fast=0.6))):
        for alpha in (0.0, 0.3):
            v = min_entry(make_kernel(name, **kw), alpha)
            print(f"       MTP2 {name:14s} alpha={alpha}: min Q_S entry {v:+.2e}")
            check(f"{name} (alpha={alpha}): Q_S entrywise non-negative", v > -1e-8)
    v = min_entry(make_kernel("matern32", tau=1.0), 0.0)
    print(f"       non-MTP2 matern32       alpha=0.0: min Q_S entry {v:+.2e}")
    check("matern32: Q_S acquires negative entries (hypothesis fails)", v < 0)


def test_explicit_constant_bounds_hold() -> None:
    """Claim 2 (L_Q) and `thm:uniform-error` must not be violated on their events."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from synthetic.s9_proof_checks import perturbation_bounds
    r = perturbation_bounds(n=120)
    print(f"       worst L_Q ratio {r['worst_ratio_LQ']:.4f}, "
          f"worst uniform-error ratio {r['worst_ratio_uniform']:.3e}, "
          f"{r['n_on_event']} on-event draws")
    check("some draws land on the event (the test is not vacuous)",
          r["n_on_event"] > 50, f"n={r['n_on_event']}")
    check("Claim 2 constant L_Q is not violated",
          r["worst_ratio_LQ"] <= 1.0, f"ratio {r['worst_ratio_LQ']:.4f}")
    check("thm:uniform-error bound is not violated",
          r["worst_ratio_uniform"] <= 1.0, f"ratio {r['worst_ratio_uniform']:.3e}")


def test_frechet_derivative_of_Q() -> None:
    """eq:dQ quotes dQ_S[E]; check it against central differences."""
    from protocol_ceiling.covariance import protocol_matrices
    rng = np.random.default_rng(3)
    grid = uniform_grid(10.0, 14)
    K = trait_state_correlation(grid, 0.25, make_kernel("ou", tau=1.3))
    acts = [Action(time=t, noise=0.4) for t in (1.5, 4.0, 8.5)]
    A, R = protocol_matrices(acts, grid)
    Minv = np.linalg.inv(A @ K @ A.T + R)
    worst, h = 0.0, 1e-6
    for _ in range(50):
        E = rng.standard_normal(K.shape)
        E = 0.5 * (E + E.T)
        E /= np.linalg.norm(E, 2)
        dQ = (E @ A.T @ Minv @ A @ K + K @ A.T @ Minv @ A @ E
              - K @ A.T @ Minv @ (A @ E @ A.T) @ Minv @ A @ K)
        fd = (explained_covariance(K + h * E, A, R)
              - explained_covariance(K - h * E, A, R)) / (2 * h)
        worst = max(worst, float(np.abs(dQ - fd).max()))
    check("eq:dQ matches central finite differences", worst < 1e-6,
          f"max discrepancy {worst:.2e}")


def test_transfer_inequality_holds() -> None:
    """Transfer inequality gamma_g >= (c_0/L_g) gamma_lin on MTP2 configurations.

    Dropped from the article during compression; retained as a checked fact.
    """
    import itertools

    from protocol_ceiling.covariance import protocol_matrices
    grid = uniform_grid(10.0, 24)
    cands = [Action(time=float(t), noise=0.4) for t in np.linspace(0.5, 9.5, 7)]

    def gamma(label, K):
        worst = np.inf
        for ns in (0, 1, 2):
            for S in itertools.combinations(cands, ns):
                pool = [a for a in cands if a not in S]
                base = evaluate_protocol(label, K, grid, list(S)).explained
                # the recorded submodularity-ratio definition includes singleton
                # Omega (it is no longer stated in the article); omitting it is exactly
                # the sampler bug that produced the impossible gamma > 1 in v1.0.
                for no in (1, 2, 3):
                    for Om in itertools.combinations(pool, no):
                        j = evaluate_protocol(label, K, grid,
                                              list(S) + list(Om)).explained - base
                        if j <= 1e-13:
                            continue
                        sgl = sum(evaluate_protocol(label, K, grid,
                                                    list(S) + [a]).explained - base
                                  for a in Om)
                        worst = min(worst, sgl / j)
        return float(worst)

    def r_max(K):
        m = 0.0
        for ns in range(1, 4):
            for S in itertools.combinations(cands, ns):
                A, R = protocol_matrices(list(S), grid)
                m = max(m, float(explained_covariance(K, A, R).max()))
        return m

    for tau in (1.0, 2.5):
        K = trait_state_correlation(grid, 0.2, make_kernel("ou", tau=tau))
        rm = r_max(K)
        ratio = float(np.sqrt(1.0 - rm**2))     # c_0/L_g for the c=0 threshold label
        g_lin, g_thr = gamma(MeanLabel(), K), gamma(ThresholdLabel(c=0.0), K)
        print(f"       tau={tau}: r_max={rm:.4f} c0/Lg={ratio:.4f} "
              f"gamma_lin={g_lin:.4f} gamma_g={g_thr:.4f}")
        check(f"transfer inequality holds (tau={tau})",
              g_thr >= ratio * g_lin - 1e-12,
              f"{g_thr:.4f} >= {ratio * g_lin:.4f}")
        check(f"gamma <= 1 as the recorded definition forces (tau={tau})",
              g_lin <= 1.0 + 1e-12 and g_thr <= 1.0 + 1e-12,
              f"lin={g_lin:.4f}, g={g_thr:.4f}")

    # Controls: the scan really can detect gamma < 1, so the checks above are
    # not passing merely because the search is blind.
    K_non = trait_state_correlation(grid, 0.0, make_kernel("matern32", tau=1.0))
    g_non = gamma(ThresholdLabel(c=0.0), K_non)
    K_ou = trait_state_correlation(grid, 0.0, make_kernel("ou", tau=1.0))
    g_two = gamma(TwoSidedLabel(c=1.0), K_ou)
    print(f"       controls: matern32 gamma={g_non:.4f}, "
          f"ou+two-sided gamma={g_two:.4f}")
    check("non-MTP2 kernel drops below gamma=1 (scan is not blind)", g_non < 1.0,
          f"gamma={g_non:.4f}")
    check("two-sided under MTP2 drops below 1, as c_0=0 predicts", g_two < 1.0,
          f"gamma={g_two:.4f}")


def test_theorem_minimal_b_zero_proviso() -> None:
    """The strict gap of eq:gap-value needs b != 0; this witness is the b = 0 slice."""
    from fractions import Fraction as Fr
    r = [Fr(1), Fr(-1, 2), Fr(0), Fr(1, 4)]
    K = np.array([[float(r[abs(i - j)]) for j in range(4)] for i in range(4)])
    ev = np.sort(np.linalg.eigvalsh(K))
    b = float((1 + 2 * r[1] + r[2]) / 4)
    print(f"       rho=(-1/2, 0, 1/4): spectrum {np.round(ev, 12)}, b={b:g}")
    check("the witness is a genuine correlation matrix", ev[0] > 0,
          f"lambda_min={ev[0]:.6f}")
    check("its spectrum is exactly {1/4, 1/2, 3/2, 7/4}",
          np.allclose(ev, [0.25, 0.5, 1.5, 1.75], atol=1e-12))
    check("b vanishes exactly there, so the gap of eq:gap-value is zero", b == 0.0)


def test_ceiling_derivative_formula() -> None:
    """eq:ceiling-derivative of `thm:impossibility`, against central differences."""
    rng = np.random.default_rng(0)

    def ceiling(K, B, RB, h):
        W = np.linalg.inv(B @ K @ B.T + RB)
        return float(h @ K @ B.T @ W @ B @ K @ h / (h @ K @ h))

    p, worst, n_ok = 5, 0.0, 0
    iu = np.triu_indices(p, 1)
    for _ in range(120):
        M = rng.standard_normal((p, p))
        K0 = M @ M.T / p + 0.8 * np.eye(p)
        d = np.sqrt(np.diag(K0))
        K0 = K0 / np.outer(d, d)
        h = rng.standard_normal(p)
        A = np.zeros((1, p))
        A[0, 0] = 1.0
        # project a random zero-diagonal symmetric matrix onto {ADh=0, hDh=0}
        rows = []
        for G in (np.outer(A[0], h), np.outer(h, h)):
            G = 0.5 * (G + G.T)
            np.fill_diagonal(G, 0.0)
            rows.append(2.0 * G[iu])
        C = np.array(rows)
        D0 = rng.standard_normal((p, p))
        D0 = 0.5 * (D0 + D0.T)
        np.fill_diagonal(D0, 0.0)
        v = D0[iu]
        v = v - C.T @ np.linalg.solve(C @ C.T, C @ v)
        if np.linalg.norm(v) < 1e-9:
            continue
        D = np.zeros((p, p))
        D[iu] = v
        D = D + D.T
        D /= np.linalg.norm(D, 2)
        a = A[0]
        if max(abs(float(a @ D @ h)), abs(float(h @ D @ h)),
               abs(float(a @ D @ a))) > 1e-9:
            continue
        n_ok += 1
        B = rng.standard_normal((2, p))
        RB = np.diag(rng.uniform(0.2, 0.6, 2))
        W = np.linalg.inv(B @ K0 @ B.T + RB)
        paper = (2 * h @ D @ B.T @ W @ B @ K0 @ h
                 - h @ K0 @ B.T @ W @ (B @ D @ B.T) @ W @ B @ K0 @ h) / (h @ K0 @ h)
        e = 1e-6
        fd = (ceiling(K0 + e * D, B, RB, h) - ceiling(K0 - e * D, B, RB, h)) / (2 * e)
        worst = max(worst, abs(float(paper) - fd))
    check("invisible directions were actually constructed", n_ok > 50,
          f"n={n_ok}")
    check("eq:ceiling-derivative matches central differences", worst < 1e-6,
          f"max discrepancy {worst:.2e}")


def test_greedy_versus_exhaustive() -> None:
    grid = uniform_grid(10.0, 128)
    K = trait_state_correlation(grid, 0.1, make_kernel("ou", tau=1.0))
    cands = candidate_actions(grid, n_times=10, noise=0.5)
    for lab in (MeanLabel(), ThresholdLabel(c=0.0), ThresholdLabel(c=1.0),
                TwoSidedLabel(c=1.0)):
        ex = select_protocol_exhaustive(lab, K, grid, cands, n_select=3)
        gr = select_protocol_greedy(lab, K, grid, cands, budget=3.0, cost_aware=False)
        ratio = gr.objective / ex.objective if ex.objective > 0 else 1.0
        print(f"       {lab.name:12s} greedy/optimal = {ratio:.6f}"
              f"  (greedy {gr.ceiling:.4f} vs optimal {ex.ceiling:.4f})")
        check(f"greedy is near-optimal for {lab.name}", ratio > 0.985,
              f"ratio {ratio:.6f}")
        check(f"greedy never beats exhaustive for {lab.name}", ratio <= 1.0 + 1e-9)


def nonstationary_correlation(grid, tau_lo: float = 0.25, tau_hi: float = 2.75):
    """Local correlation time increasing across the horizon.

    A stationary kernel with uniform label weights makes the design problem
    translation-symmetric, so *every* label lands on the same symmetric
    placement and the label-dependence of the optimum is invisible.  Letting the
    local correlation time vary breaks that symmetry and exposes the mechanism:
    the mean label sees only tau_1, whereas order-k Hermite components see
    tau_k = int rho^k, which concentrates ever more sharply on the slowly
    decorrelating part of the horizon.
    """
    from protocol_ceiling import project_psd, to_correlation
    t = grid.times
    tau_t = tau_lo + (tau_hi - tau_lo) * (t / t.max())
    lag = np.abs(t[:, None] - t[None, :])
    K = np.exp(-lag / np.sqrt(np.outer(tau_t, tau_t)))
    np.fill_diagonal(K, 1.0)
    return to_correlation(project_psd(K, 1e-9))


def test_label_dependent_optima() -> None:
    """Different labels must select different optimal placements."""
    grid = uniform_grid(10.0, 128)
    K = nonstationary_correlation(grid)
    cands = [Action(time=float(t), width=0.0, noise=0.6, cost=1.0)
             for t in np.linspace(0.7, 9.3, 9)]
    picks = {}
    for lab in (MeanLabel(), ThresholdLabel(c=0.0), ThresholdLabel(c=1.5),
                TwoSidedLabel(c=1.2)):
        res = select_protocol_exhaustive(lab, K, grid, cands, n_select=3)
        key = f"{lab.name}(c={getattr(lab, 'c', '-')})"
        picks[key] = tuple(round(a.time, 2) for a in res.actions)
        print(f"       {key:22s} -> {picks[key]}   I = {res.ceiling:.4f}")
    distinct = len(set(picks.values()))
    check("labels induce different optimal protocols", distinct >= 3,
          f"{distinct} distinct designs among {len(picks)} labels")
    check("mean and two-sided optima differ",
          picks["mean(c=-)"] != picks["two_sided(c=1.2)"])


def test_greedy_needs_local_search() -> None:
    """Plain greedy can be beaten; the swap pass must repair it."""
    grid = uniform_grid(10.0, 128)
    K = trait_state_correlation(grid, 0.2, make_kernel("ou", tau=0.8))
    cands = candidate_actions(grid, n_times=16, noise=0.7)
    lab = ThresholdLabel(c=1.0)
    plain = select_protocol_greedy(lab, K, grid, cands, budget=4.0,
                                   cost_aware=False, local_search=False)
    refined = select_protocol_greedy(lab, K, grid, cands, budget=4.0,
                                     cost_aware=False, local_search=True)
    best = select_protocol_exhaustive(lab, K, grid, cands, n_select=4)
    print(f"       plain greedy   I = {plain.ceiling:.6f}")
    print(f"       + swap search  I = {refined.ceiling:.6f}  ({refined.n_evaluations} evals)")
    print(f"       exhaustive     I = {best.ceiling:.6f}  ({best.n_evaluations} evals)")
    check("swap search does not hurt", refined.objective >= plain.objective - 1e-14)
    check("swap search reaches (near) the exhaustive optimum",
          refined.objective >= 0.9999 * best.objective,
          f"ratio {refined.objective / best.objective:.6f}")
    check("swap search is far cheaper than exhaustive search",
          refined.n_evaluations < best.n_evaluations)


def test_kernel_quadrature_equivalence() -> None:
    """Noiseless mean label: our objective must coincide with kernel quadrature."""
    grid = uniform_grid(10.0, 128)
    K = trait_state_correlation(grid, 0.2, make_kernel("matern32", tau=1.0))
    cands = candidate_actions(grid, n_times=16, noise=0.0)
    lab = MeanLabel()
    ours = select_protocol_greedy(lab, K, grid, cands, budget=4.0, cost_aware=False)
    kq = design_kernel_quadrature(lab, K, grid, cands, n_select=4)
    print(f"       ours {ours.objective:.10f}   kernel-quadrature {kq.objective:.10f}")
    check("noiseless mean label == kernel quadrature",
          abs(ours.objective - kq.objective) < 1e-10,
          f"|diff| {abs(ours.objective - kq.objective):.2e}")


def test_baselines_are_dominated() -> None:
    grid = uniform_grid(10.0, 128)
    K = trait_state_correlation(grid, 0.2, make_kernel("ou", tau=0.8))
    cands = candidate_actions(grid, n_times=16, noise=0.7)
    lab = ThresholdLabel(c=1.0)
    n = 4
    ours = select_protocol_greedy(lab, K, grid, cands, budget=float(n),
                                  cost_aware=False, local_search=True)
    tmpl = cands[0]
    rows = {
        "label-aware": ours.ceiling,
        "uniform": design_uniform(lab, K, grid, n, tmpl).ceiling,
        "mutual-information": design_mutual_information(lab, K, grid, cands, n).ceiling,
        "imse": design_imse(lab, K, grid, cands, n).ceiling,
        "kernel-quadrature": design_kernel_quadrature(lab, K, grid, cands, n).ceiling,
    }
    for k, v in rows.items():
        print(f"       {k:20s} I = {v:.6f}")
    check("label-aware design is best among the baselines",
          all(ours.ceiling >= v - 1e-12 for v in rows.values()))


def test_mutually_exclusive_action_variants() -> None:
    """Every selector uses at most one noise/segment variant per support."""
    grid = uniform_grid(6.0, 64)
    K = trait_state_correlation(grid, 0.1, make_kernel("ou", tau=1.0))
    lab = MeanLabel()
    cands = []
    for t in (1.0, 3.0, 5.0):
        cands.extend([
            Action(time=t, width=0.0, n_segments=1, noise=1.0, cost=1.0,
                   tag=f"{t}:single"),
            Action(time=t, width=0.0, n_segments=4, noise=1.0, cost=1.0,
                   tag=f"{t}:replicated"),
        ])

    results = {
        "target-aware": select_protocol_greedy(
            lab, K, grid, cands, budget=3.0, cost_aware=True),
        "mutual-information": design_mutual_information(lab, K, grid, cands, 3),
        "integrated-posterior-variance": design_imse(lab, K, grid, cands, 3),
        "kernel-quadrature": design_kernel_quadrature(lab, K, grid, cands, 3),
        "exhaustive": select_protocol_exhaustive(lab, K, grid, cands, n_select=3),
    }
    for name, res in results.items():
        supports = [(round(a.time, 12), round(a.width, 12)) for a in res.actions]
        check(f"{name} respects mutually exclusive variants",
              len(supports) == len(set(supports)), detail=str(supports))


def test_submodularity_violation_exists() -> None:
    """Find an explicit S subset T with Delta(a|S) < Delta(a|T)."""
    rng = np.random.default_rng(4)
    grid = uniform_grid(10.0, 96)
    K = trait_state_correlation(grid, 0.0, make_kernel("periodic", tau=6.0, period=3.0))
    cands = candidate_actions(grid, n_times=10, noise=0.15)
    for lab in (MeanLabel(), ThresholdLabel(c=0.0)):
        v = find_submodularity_violation(lab, K, grid, cands, rng, n_trials=600)
        if v is None:
            print(f"       {lab.name}: no violation found")
            check(f"{lab.name}: submodularity violation found", False)
            continue
        print(f"       {lab.name}: Delta(a|S)={v['gain_S']:.6e} < Delta(a|T)={v['gain_T']:.6e}"
              f"  (|S|={len(v['S'])}, |T|={len(v['T'])}, rel {v['relative_violation']:.3f})")
        check(f"{lab.name}: F_g is provably not submodular", v["gain_T"] > v["gain_S"])


def test_submodularity_ratio_certificate() -> None:
    grid = uniform_grid(10.0, 96)
    K = trait_state_correlation(grid, 0.15, make_kernel("ou", tau=1.0))
    cands = candidate_actions(grid, n_times=10, noise=0.5)
    lab = ThresholdLabel(c=0.5)
    gr = select_protocol_greedy(lab, K, grid, cands, budget=4.0, cost_aware=False)
    prefixes = [gr.actions[:k] for k in range(len(gr.actions))]
    cert = submodularity_ratio_certificate(lab, K, grid, cands, prefixes,
                                           rng=np.random.default_rng(1), n_subsets=24)
    print(f"       gamma >= {cert['gamma']:.4f} (median {cert['gamma_median']:.4f},"
          f" {cert['n_samples']} samples) -> greedy factor {cert['greedy_factor']:.4f}")
    check("submodularity ratio certificate is positive", cert["gamma"] > 0)
    check("greedy factor is a valid approximation factor",
          0 < cert["greedy_factor"] <= 1.0)
    lin_gamma = cert["gamma"]
    tr = nonlinear_ratio_lower_bound(lab, lin_gamma, rmax=0.99)
    print(f"       analytic transfer bound gamma_g >= (c0/Lg) gamma_lin = {tr:.6f}")
    check("transfer bound is a valid probability", 0 <= tr <= 1)


def test_cost_aware_budget_respected() -> None:
    grid = uniform_grid(10.0, 96)
    K = trait_state_correlation(grid, 0.2, make_kernel("ou", tau=1.0))
    cands = candidate_actions(grid, n_times=12, widths=(0.0, 1.0, 2.0),
                              noise=0.5, cost_fixed=1.0, cost_per_time=0.5)
    lab = ThresholdLabel(c=0.0)
    res = select_protocol_greedy(lab, K, grid, cands, budget=5.0, cost_aware=True)
    print(f"       chose {len(res.actions)} actions at total cost {res.cost:.2f} <= 5.0"
          f"  (I = {res.ceiling:.4f})")
    check("cost-aware greedy respects the budget", res.cost <= 5.0 + 1e-9)


def test_mutual_information_matches_logdet() -> None:
    """The standard latent-state MI comparator must use its stated gains."""
    grid = uniform_grid(6.0, 48)
    K = trait_state_correlation(grid, 0.1, make_kernel("ou", tau=1.2))
    cands = [Action(time=float(t), noise=float(r), tag=f"a{j}")
             for j, (t, r) in enumerate(zip(np.linspace(0.2, 5.8, 8),
                                             [0.8, 0.3, 1.1, 0.5,
                                              0.2, 0.9, 0.4, 0.7]))]
    res = design_mutual_information(MeanLabel(), K, grid, cands, 4)

    # Reconstruct exact greedy MI gains independently.
    P = K.copy()
    available = list(range(len(cands)))
    expected = []
    for _ in range(4):
        gains = []
        for j in available:
            ell = action_vector(cands[j], grid)
            gains.append(0.5 * np.log((ell @ P @ ell + cands[j].effective_noise)
                                      / cands[j].effective_noise))
        chosen = available[int(np.argmax(gains))]
        expected.append(cands[chosen].tag)
        ell = action_vector(cands[chosen], grid)
        v = P @ ell
        P -= np.outer(v, v) / (ell @ v + cands[chosen].effective_noise)
        available.remove(chosen)
    check("latent-state MI uses the exact posterior-variance greedy gains",
          [a.tag for a in res.actions] == expected,
          detail=f"got={[a.tag for a in res.actions]}, expected={expected}")

    A, R = protocol_matrices(res.actions, grid)
    sign, logdet = np.linalg.slogdet(np.eye(len(res.actions))
                                    + np.linalg.solve(R, A @ K @ A.T))
    direct = 0.5 * float(logdet)
    check("selected latent-state MI log-determinant is finite and positive",
          sign > 0 and np.isfinite(direct) and direct > 0.0,
          detail=f"MI={direct:.12f}")



def test_augmentation_test_collapses_invisible_space():
    """Proposition: one extra point observation can restore identification.

    The paper claims that in the four-point stationary model a single added
    point observation of Z_1 takes the stationary invisible space from
    dimension 1 to 0, and that without stationarity the free dimensions run
    4 -> 2 -> 0.  These are the numbers printed as \numAug* macros, and they
    are what makes the augmentation test non-vacuous: the minimal sufficient
    augmentation is a second measurement, not a second trajectory.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                           / "experiments" / "synthetic"))
    from s9_proof_checks import invisible_dimension
    import numpy as _np
    e = _np.eye(4)
    check("stationary, A = {Z0}", invisible_dimension([e[0]], stationary=True) == 1)
    check("stationary, A + {Z1}", invisible_dimension([e[0], e[1]], stationary=True) == 0)
    check("free, A = {Z0}", invisible_dimension([e[0]], stationary=False) == 4)
    check("free, A + {Z1}", invisible_dimension([e[0], e[1]], stationary=False) == 2)
    check("free, A + {Z1,Z2}",
          invisible_dimension([e[0], e[1], e[2]], stationary=False) == 0)
    # monotone: adding rows can only shrink the invisible space
    dims = [invisible_dimension([e[0]], stationary=False),
            invisible_dimension([e[0], e[1]], stationary=False),
            invisible_dimension([e[0], e[1], e[2]], stationary=False)]
    check("monotone in the augmentation", all(a >= b for a, b in zip(dims, dims[1:])))



if __name__ == "__main__":
    for fn in [test_minimal_stationary_counterexample, test_full_joint_law_is_identical,
               test_closed_form_gap_matches_theorem,
               test_sharpness_p_equals_three, test_general_direction_search,
               test_partial_identification_interval, test_mtp2_classification,
               test_paper_nonsubmodularity_witness, test_submodularity_ratio_is_capped,
               test_augmentation_test_collapses_invisible_space,
               test_no_adaptivity_gain_linear,
               test_calibration_chain_constant,
               test_mtp2_hypothesis_of_transfer_theorem,
               test_explicit_constant_bounds_hold,
               test_frechet_derivative_of_Q,
               test_transfer_inequality_holds,
               test_theorem_minimal_b_zero_proviso,
               test_ceiling_derivative_formula,
               test_greedy_versus_exhaustive,
               test_greedy_needs_local_search,
               test_label_dependent_optima, test_kernel_quadrature_equivalence,
               test_baselines_are_dominated, test_mutually_exclusive_action_variants,
               test_submodularity_violation_exists,
               test_submodularity_ratio_certificate, test_cost_aware_budget_respected,
               test_mutual_information_matches_logdet]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n" + "=" * 68)
    if failures:
        print(f"{len(failures)} FAILURE(S): " + ", ".join(failures))
        raise SystemExit(1)
    print("all identifiability and design tests passed")
