"""Unit tests for the numerical core.

Every closed form used in the paper is checked here against an independent
computation, and the conference-version numbers are reproduced through the new
discrete interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol_ceiling import (Action, MeanLabel, SquareLabel, ThresholdLabel,
                              TwoSidedLabel, allocation_protocol, bin_midpoints,
                              dispersed_protocol, evaluate_protocol,
                              explained_covariance, label_variance,
                              make_kernel, project_psd, same_time_protocol,
                              sigmoid_label, trait_state_correlation,
                              uniform_grid)
from protocol_ceiling.covariance import protocol_matrices, action_vector
from protocol_ceiling.risk import ProtocolState, bilinear
from protocol_ceiling import continuous as cont
from protocol_ceiling.transforms import (ARCSIN_HOLDER, hermite_coefficients,
                                         indicator_hermite_coefficients,
                                         normalised_hermite_values)

TOL = 1e-8
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# --------------------------------------------------------------------------
def test_threshold_zero_arcsine() -> None:
    lab = ThresholdLabel(c=0.0)
    r = np.linspace(-0.999, 0.999, 401)
    err = float(np.max(np.abs(lab.C(r) - np.arcsin(r) / (2 * np.pi))))
    check("threshold c=0 equals arcsin(r)/(2 pi)", err < 1e-14, f"max err {err:.2e}")


def test_threshold_table_against_quadrature() -> None:
    from scipy.integrate import quad
    for c in (0.3, 0.8, 1.5, -1.0):
        lab = ThresholdLabel(c=c)
        for r in (0.1, 0.5, 0.9, 0.99, -0.7):
            ref = quad(lambda s: np.exp(-c**2 / (1 + s)) / (2 * np.pi * np.sqrt(1 - s**2)),
                       0.0, r, epsabs=1e-13, epsrel=1e-12, limit=400)[0]
            err = abs(float(lab.C(r)) - ref)
            check(f"Plackett table c={c} r={r}", err < 1e-10, f"err {err:.2e}")


def test_threshold_matches_bivariate_normal() -> None:
    """C_g(r) must equal Phi_2(c, c; r) - Phibar(c)^2 from an independent route."""
    from scipy.stats import multivariate_normal
    from scipy.special import ndtr
    for c in (0.0, 0.5, 1.2):
        lab = ThresholdLabel(c=c)
        for r in (0.2, 0.6, 0.85):
            cov = np.array([[1.0, r], [r, 1.0]])
            joint = float(multivariate_normal(mean=[0, 0], cov=cov).cdf([-c, -c]))
            ref = joint - float(1 - ndtr(c)) ** 2
            err = abs(float(lab.C(r)) - ref)
            check(f"threshold vs bivariate normal c={c} r={r}", err < 1e-8,
                  f"err {err:.2e}")


def test_square_and_hermite() -> None:
    sq = SquareLabel()
    r = np.linspace(-0.99, 0.99, 51)
    check("square label C(r) = 2 r^2", float(np.max(np.abs(sq.C(r) - 2 * r**2))) < 1e-14)
    coeffs = hermite_coefficients(lambda z: z**2, kmax=8)
    # a_2 = 2 so atilde_2 = 2/sqrt(2) = sqrt(2); all others vanish.
    ok = abs(coeffs[1] - np.sqrt(2.0)) < 1e-9 and np.max(np.abs(np.delete(coeffs, 1))) < 1e-9
    check("Hermite spectrum of z^2", ok, f"coeffs[:4]={np.round(coeffs[:4], 6)}")


def test_hermite_recurrence() -> None:
    from scipy.special import eval_hermitenorm
    from scipy.special import factorial
    z = np.linspace(-3, 3, 7)
    h = normalised_hermite_values(z, 10)
    ref = np.array([eval_hermitenorm(k, z) / np.sqrt(float(factorial(k)))
                    for k in range(11)])
    err = float(np.max(np.abs(h - ref)))
    check("normalised Hermite recurrence", err < 1e-9, f"max err {err:.2e}")


def test_lipschitz_constant_identity() -> None:
    """sup |C_g'| must equal E[g'(U)^2] for smooth labels."""
    slope = 1.7
    lab = sigmoid_label(slope=slope, c=0.3, kmax=300)
    L, beta = lab.modulus()
    nodes = np.linspace(-9, 9, 200001)
    w = np.exp(-nodes**2 / 2) / np.sqrt(2 * np.pi)
    s = 1.0 / (1.0 + np.exp(-slope * (nodes - 0.3)))
    dg = slope * s * (1 - s)
    ref = float(np.trapezoid(dg**2 * w, nodes))
    check("sup |C_g'| = E[g'(U)^2]", abs(L - ref) / ref < 2e-4,
          f"series {L:.8f} vs quadrature {ref:.8f}")
    check("smooth label is Lipschitz (beta=1)", beta == 1.0)


def test_arcsine_holder_constant() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.uniform(-1, 1, 200000), rng.uniform(-1, 1, 200000)
    lhs = np.abs(np.arcsin(b) - np.arcsin(a))
    rhs = ARCSIN_HOLDER * np.sqrt(np.abs(b - a))
    check("arcsin Holder-1/2 constant pi/sqrt(2)", bool(np.all(lhs <= rhs + 1e-12)),
          f"max ratio {float(np.max(lhs / np.maximum(rhs, 1e-12))):.6f}")
    # sharpness at the extreme pair
    ratio = abs(np.arcsin(1.0) - np.arcsin(-1.0)) / (ARCSIN_HOLDER * np.sqrt(2.0))
    check("arcsin constant is attained", abs(ratio - 1.0) < 1e-12, f"ratio {ratio:.12f}")


def test_two_sided_label() -> None:
    """Two-sided label: exact bivariate-normal check and even-order structure."""
    from scipy.stats import multivariate_normal
    from scipy.special import ndtr
    c = 1.0
    lab = TwoSidedLabel(c=c)
    p = float(2.0 * (1.0 - ndtr(c)))
    for r in (0.2, 0.6, 0.9, -0.5):
        cov = np.array([[1.0, r], [r, 1.0]])
        mvn = multivariate_normal(mean=[0, 0], cov=cov)
        # P(|U|>c, |V|>c) via inclusion-exclusion on the four tail quadrants.
        joint = (float(mvn.cdf([-c, -c]))                       # U<-c, V<-c
                 + float(mvn.cdf([-c, np.inf])) - float(mvn.cdf([-c, c]))   # U<-c, V>c
                 + float(mvn.cdf([np.inf, -c])) - float(mvn.cdf([c, -c]))   # U>c, V<-c
                 + 1.0 - float(mvn.cdf([c, np.inf])) - float(mvn.cdf([np.inf, c]))
                 + float(mvn.cdf([c, c])))                      # U>c, V>c
        ref = joint - p * p
        err = abs(float(lab.C(r)) - ref)
        check(f"two-sided vs bivariate normal r={r}", err < 1e-8, f"err {err:.2e}")

    coeffs = indicator_hermite_coefficients(c, kmax=600, two_sided=True)
    odd = float(np.max(np.abs(coeffs[0::2])))   # k = 1, 3, 5, ...
    check("two-sided label has no odd Hermite orders", odd < 1e-14, f"max odd {odd:.2e}")
    # The truncated Hermite series converges to C_g away from |r| = 1.
    r = np.linspace(-0.9, 0.9, 41)
    series = np.zeros_like(r)
    for k, a in enumerate(coeffs, start=1):
        series += a**2 * r**k
    err = float(np.max(np.abs(lab.C(r) - series)))
    check("two-sided label vs analytic Hermite series (|r|<=0.9)", err < 1e-9,
          f"max err {err:.2e}")
    # The spectrum sums to Var g(U) only in the limit, and the truncation tail
    # decays like K^{-1/2}: this *is* the square-root singularity of C_g at
    # r = 1 that forces the Holder-1/2 (rather than Lipschitz) rate for
    # threshold labels (the beta = 1/2 branch of `thm:uniform-error`).
    target = p * (1 - p)
    tails = []
    for K in (500, 2000, 8000, 32000):
        tot = float(np.sum(indicator_hermite_coefficients(c, K, True) ** 2))
        tails.append((K, target - tot))
    for K, t in tails:
        print(f"       K={K:6d}  truncation tail {t:.6e}  tail*sqrt(K) {t * np.sqrt(K):.4f}")
    check("indicator spectrum converges upward to Var g(U)",
          all(t > 0 for _, t in tails) and tails[-1][1] < 1e-3,
          f"tail at K=32000 is {tails[-1][1]:.2e}")
    scaled = [t * np.sqrt(K) for K, t in tails]
    check("truncation tail decays at the K^{-1/2} rate",
          max(scaled) / min(scaled) < 1.3,
          f"tail*sqrt(K) ranges over {min(scaled):.4f}-{max(scaled):.4f}")


def test_one_sided_indicator_spectrum() -> None:
    """One-sided threshold: analytic spectrum must reproduce the Plackett table."""
    for c in (0.0, 0.6, 1.4):
        lab = ThresholdLabel(c=c)
        coeffs = indicator_hermite_coefficients(c, kmax=800, two_sided=False)
        r = np.linspace(-0.9, 0.9, 41)
        series = np.zeros_like(r)
        for k, a in enumerate(coeffs, start=1):
            series += a**2 * r**k
        err = float(np.max(np.abs(lab.C(r) - series)))
        check(f"threshold c={c}: Plackett table vs Hermite series", err < 1e-9,
              f"max err {err:.2e}")


def test_rank_one_update_matches_direct_solve() -> None:
    rng = np.random.default_rng(3)
    grid = uniform_grid(10.0, 48)
    K = trait_state_correlation(grid, 0.2, make_kernel("ou", tau=1.3))
    lab = ThresholdLabel(c=0.4)
    acts = [Action(time=t, width=0.0, noise=0.35) for t in (1.0, 4.0, 7.5, 9.0)]

    state = ProtocolState.empty(lab, K, grid.weights)
    seq = []
    for a in acts:
        seq.append(state.marginal_gain(action_vector(a, grid), a.effective_noise))
        state = state.add(a, grid)

    A, R = protocol_matrices(acts, grid)
    F_direct = bilinear(lab, explained_covariance(K, A, R), grid.weights)
    check("rank-one greedy path equals direct solve",
          abs(state.F - F_direct) < 1e-11, f"|diff| {abs(state.F - F_direct):.2e}")
    check("marginal gains telescope to the objective",
          abs(sum(seq) - F_direct) < 1e-11, f"|diff| {abs(sum(seq) - F_direct):.2e}")


def test_mean_label_closed_form() -> None:
    rng = np.random.default_rng(5)
    grid = uniform_grid(8.0, 40)
    K = trait_state_correlation(grid, 0.15, make_kernel("matern32", tau=1.0))
    lab = MeanLabel()
    acts = [Action(time=t, width=0.5, noise=0.2) for t in (2.0, 5.5)]
    A, R = protocol_matrices(acts, grid)
    h = grid.weights
    Q = explained_covariance(K, A, R)
    check("mean label F = h^T Q h",
          abs(bilinear(lab, Q, h) - float(h @ Q @ h)) < 1e-13)
    # exact marginal gain formula
    st = ProtocolState.from_actions(lab, K, grid, acts[:1])
    ell = action_vector(acts[1], grid)
    v = st.P @ ell
    s = float(ell @ v) + acts[1].effective_noise
    check("mean marginal gain = (h^T P l)^2 / (l^T P l + nu^2)",
          abs(st.marginal_gain(ell, acts[1].effective_noise) - (h @ v) ** 2 / s) < 1e-13)


def test_monotonicity_of_objective() -> None:
    rng = np.random.default_rng(11)
    grid = uniform_grid(12.0, 40)
    K = trait_state_correlation(grid, 0.1, make_kernel("ou", tau=1.0))
    for lab in (MeanLabel(), ThresholdLabel(c=0.0), ThresholdLabel(c=0.7),
                TwoSidedLabel(c=0.8), SquareLabel()):
        st = ProtocolState.empty(lab, K, grid.weights)
        vals = [st.F]
        for t in rng.uniform(0, 12, 8):
            st = st.add(Action(time=float(t), noise=0.4), grid)
            vals.append(st.F)
        diffs = np.diff(vals)
        check(f"monotone F_g for {lab.name}", bool(np.all(diffs >= -1e-12)),
              f"min increment {float(diffs.min()):.2e}")


def test_psd_projection() -> None:
    rng = np.random.default_rng(7)
    S = rng.standard_normal((20, 20))
    S = 0.5 * (S + S.T)
    P = project_psd(S)
    vals = np.linalg.eigvalsh(P)
    check("PSD projection has no negative eigenvalues", float(vals.min()) > -1e-12,
          f"lambda_min {float(vals.min()):.2e}")


# --------------------------------------------------------------------------
# Conference-version regression
# --------------------------------------------------------------------------
def test_conference_ou_closed_form() -> None:
    tau = 1.0
    val = cont.occupation_state_coefficient_ou(0.0, tau)
    ref = tau * np.log(2.0) / 2.0
    check("OU A_0 = tau log 2 / 2", abs(val - ref) < 1e-9,
          f"{val:.12f} vs {ref:.12f}")


def test_conference_equal_budget_values() -> None:
    """Reproduce I_same = 0.0970 and I_dispersed = 0.8083 at N = 64."""
    T, tau, N = 20.0, 1.0, 64
    rho = make_kernel("ou", tau=tau)
    same = cont.point_protocol_explainability(T, np.array([T / 2]), np.array([1.0 / N]),
                                              rho, alpha=0.0, grid_size=4001)
    times = bin_midpoints(T, N)
    disp = cont.point_protocol_explainability(T, times, np.ones(N), rho, alpha=0.0,
                                              grid_size=4001)
    check("conference same-time value 0.0970", abs(same - 0.09701591800567386) < 1e-4,
          f"got {same:.6f}")
    check("conference dispersed value 0.8083", abs(disp - 0.8083429396705237) < 1e-4,
          f"got {disp:.6f}")
    print(f"       exact: same-time {same:.6f}, dispersed {disp:.6f}")


def test_discrete_matches_continuous() -> None:
    """The discrete interface must converge to the continuous-time routine."""
    T, tau, N = 20.0, 1.0, 16
    rho = make_kernel("ou", tau=tau)
    lab = ThresholdLabel(c=0.0)
    times = np.linspace(0.0, T, N)
    ref = cont.point_protocol_explainability(T, times, np.ones(N), rho, alpha=0.0,
                                             grid_size=6001)
    for p in (256, 512, 1024):
        grid = uniform_grid(T, p)
        K = trait_state_correlation(grid, 0.0, rho)
        acts = [Action(time=float(t), width=0.0, noise=1.0) for t in times]
        got = evaluate_protocol(lab, K, grid, acts).ceiling
        print(f"       p={p:5d}  discrete {got:.6f}  continuous {ref:.6f}"
              f"  diff {abs(got - ref):.2e}")
    check("discrete interface agrees with continuous quadrature",
          abs(got - ref) < 5e-3, f"|diff| {abs(got - ref):.2e}")


def test_segments_versus_time_discrete() -> None:
    """Same-time replication saturates; dispersed occasions keep improving."""
    T, p = 20.0, 512
    grid = uniform_grid(T, p)
    K = trait_state_correlation(grid, 0.0, make_kernel("ou", tau=1.0))
    lab = ThresholdLabel(c=0.0)
    rows = []
    for N in (1, 2, 4, 8, 16, 32, 64):
        s = evaluate_protocol(lab, K, grid, same_time_protocol(grid, N, noise=1.0)).ceiling
        d = evaluate_protocol(lab, K, grid, dispersed_protocol(grid, N, noise=1.0)).ceiling
        rows.append((N, s, d))
        print(f"       N={N:3d}  same-time {s:.4f}   dispersed {d:.4f}")
    same64, disp64 = rows[-1][1], rows[-1][2]
    check("discrete same-time saturates below 0.12", same64 < 0.12, f"{same64:.4f}")
    check("discrete dispersed exceeds 0.78", disp64 > 0.78, f"{disp64:.4f}")
    check("the two protocols coincide at N = 1", abs(rows[0][1] - rows[0][2]) < 1e-12)
    check("dispersed dominates same-time for every N >= 2",
          all(d > s for _, s, d in rows[1:]))


def test_segments_equal_noise_division() -> None:
    """M repeated segments must be exactly equivalent to nu^2 / M."""
    grid = uniform_grid(10.0, 64)
    K = trait_state_correlation(grid, 0.3, make_kernel("ou", tau=1.0))
    lab = ThresholdLabel(c=0.2)
    a1 = [Action(time=5.0, noise=1.0, n_segments=8)]
    a2 = [Action(time=5.0, noise=0.125, n_segments=1)]
    c1 = evaluate_protocol(lab, K, grid, a1).ceiling
    c2 = evaluate_protocol(lab, K, grid, a2).ceiling
    check("M segments == noise / M", abs(c1 - c2) < 1e-13, f"|diff| {abs(c1 - c2):.2e}")


def test_boundary_localisation() -> None:
    """Prop B.4: G_a is even in a and strictly decreasing in |a| for r > 0."""
    from scipy.integrate import quad
    G = lambda a, r: float(ThresholdLabel(c=a).C(r))
    for r in (0.2, 0.5, 0.9):
        even = max(abs(G(a, r) - G(-a, r)) for a in np.linspace(0.1, 3.0, 15))
        vals = np.array([G(a, r) for a in np.linspace(0.0, 3.0, 13)])
        check(f"G_a is even in a (r={r})", even < 1e-14, f"max diff {even:.1e}")
        check(f"G_a strictly decreasing in |a| (r={r})", bool(np.all(np.diff(vals) < 0)))
    A = [2 * quad(lambda u: G(a, np.exp(-u)), 0, 60, limit=400)[0]
         for a in (0.0, 0.5, 1.0, 1.5, 2.0)]
    check("A_a decreases in |a| (OU, tau=1)", bool(np.all(np.diff(A) < 0)),
          f"{A[0]:.4f} -> {A[-1]:.4f}")
    check("A_0 = tau*log(2)/2, independently of the closed-form test",
          abs(A[0] - np.log(2) / 2) < 1e-8, f"{A[0]:.8f}")


def test_monte_carlo_agreement() -> None:
    """Exact ceiling must match a Monte Carlo Bayes-risk simulation."""
    rng = np.random.default_rng(2024)
    T, p = 6.0, 64
    grid = uniform_grid(T, p)
    K = trait_state_correlation(grid, 0.25, make_kernel("ou", tau=1.0))
    lab = ThresholdLabel(c=0.0)
    acts = [Action(time=1.5, noise=0.3), Action(time=4.5, noise=0.3)]
    A, R = protocol_matrices(acts, grid)
    exact = evaluate_protocol(lab, K, grid, acts)

    n = 200_000
    L = np.linalg.cholesky(K + 1e-10 * np.eye(p))
    Z = rng.standard_normal((n, p)) @ L.T
    theta = lab.apply(Z) @ grid.weights
    Y = Z @ A.T + rng.standard_normal((n, A.shape[0])) @ np.sqrt(R)
    # Exact posterior mean of Theta given Y (Gaussian conditioning + Phibar).
    from scipy.special import ndtr
    KA = K @ A.T
    M = A @ KA + R
    Minv = np.linalg.inv(M)
    mpost = Y @ Minv @ KA.T
    qdiag = np.einsum("ij,jk,ik->i", KA, Minv, KA)
    sd = np.sqrt(np.maximum(1.0 - qdiag, 1e-12))
    pred = (1.0 - ndtr((lab.c - mpost) / sd)) @ grid.weights

    mc_var = float(np.var(theta, ddof=1))
    mc_risk = float(np.mean((theta - pred) ** 2))
    mc_ceiling = 1.0 - mc_risk / mc_var
    print(f"       exact  Var {exact.total:.6f}  risk {exact.risk:.6f}  I {exact.ceiling:.6f}")
    print(f"       MC     Var {mc_var:.6f}  risk {mc_risk:.6f}  I {mc_ceiling:.6f}")
    check("Monte Carlo label variance matches", abs(mc_var - exact.total) < 3e-4)
    check("Monte Carlo Bayes risk matches", abs(mc_risk - exact.risk) < 3e-4)
    check("Monte Carlo ceiling matches", abs(mc_ceiling - exact.ceiling) < 4e-3,
          f"|diff| {abs(mc_ceiling - exact.ceiling):.2e}")


if __name__ == "__main__":
    for fn in [test_threshold_zero_arcsine, test_threshold_table_against_quadrature,
               test_threshold_matches_bivariate_normal, test_square_and_hermite,
               test_hermite_recurrence, test_lipschitz_constant_identity,
               test_arcsine_holder_constant, test_two_sided_label,
               test_one_sided_indicator_spectrum,
               test_rank_one_update_matches_direct_solve, test_mean_label_closed_form,
               test_monotonicity_of_objective, test_psd_projection,
               test_conference_ou_closed_form, test_conference_equal_budget_values,
               test_discrete_matches_continuous, test_segments_versus_time_discrete,
               test_segments_equal_noise_division, test_boundary_localisation,
               test_monte_carlo_agreement]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n" + "=" * 68)
    if failures:
        print(f"{len(failures)} FAILURE(S): " + ", ".join(failures))
        raise SystemExit(1)
    print("all core tests passed")
