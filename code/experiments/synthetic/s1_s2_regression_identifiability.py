"""S1 (conference regression) and S2 (non-identifiability of counterfactual protocol values).

S1 -- every conference-version number is recomputed twice: once through
``protocol_ceiling.continuous`` (the one-dimensional quadrature route the
conference code used) and once through the *new* discrete interface
``evaluate_protocol`` / ``label_variance``.  Both routes are then checked against
a Monte Carlo simulation of the actual Bayes risk.

  (a) one-snapshot occupation-label explainability, ``T/tau = 14``, point-noise
      variance ``0.2``, ``alpha in {0, 0.20, 0.35}``;
  (b) the equal-budget ``D``-versus-``M`` curve, ``alpha = 0``, ``c = 0``,
      ``T/tau = 20``, unit raw-segment noise, ``N in {1,...,64}``;
  (c) the OU closed form ``A_0 = tau log(2) / 2``.

S2 -- the counterfactual protocol value is not identified from a single-protocol
benchmark.  The four-point stationary witness, the sharp ``p = 3`` / ``p >= 4``
threshold, a genericity sweep over random higher-dimensional instances, and a
demonstration that the gap is not an artefact of linear labels.

Run with::

    .venv/bin/python experiments/synthetic/s1_s2_regression_identifiability.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common import (FIGURES, PALETTE, RESULTS, SEED, Timer,
                                environment_record, save_csv, save_figure,
                                save_json, setup_matplotlib)
from protocol_ceiling import continuous as cont
from protocol_ceiling import (Action, MeanLabel, ThresholdLabel, TimeGrid,
                              bin_midpoints, counting_bound, dispersed_protocol,
                              evaluate_protocol, label_variance, linear_ceiling,
                              make_kernel, max_psd_step, minimal_stationary_example,
                              nonidentified_directions, observed_discrepancy,
                              same_time_protocol,
                              stationary_identification_jacobian,
                              to_correlation, trait_state_correlation,
                              uniform_grid)
from protocol_ceiling.covariance import protocol_matrices

# --------------------------------------------------------------------------
# Fixed experimental constants (every one of these is cited in the paper)
# --------------------------------------------------------------------------
TAU = 1.0
T_SNAPSHOT = 14.0            # T / tau = 14
SNAPSHOT_NOISE = 0.2         # point-observation noise variance
ALPHAS = (0.0, 0.20, 0.35)
CONF_CEILINGS = {0.0: 0.119, 0.20: 0.256, 0.35: 0.355}

T_BUDGET = 20.0              # T / tau = 20
BUDGETS = (1, 2, 4, 8, 16, 32, 64)
RAW_NOISE = 1.0              # unit raw-segment noise
CONF_SAME_64 = 0.09701591800567386
CONF_DISP_64 = 0.8083429396705237

P_GRID = 512                 # headline discretisation
P_LADDER = (128, 256, 512, 1024)
CONT_GRID = 8001             # continuous-route quadrature resolution

MC_N = 25_000                # objects per Monte Carlo repetition (>= 20000)
MC_REPS = 40                 # repetitions (>= 10); 1.0e6 simulated objects in total

S2_P_VALUES = (8, 12, 16)
S2_D_OBS = (1, 2, 3)
S2_INSTANCES = 200
S2_DIR_TRIALS = 12           # random directions probed inside each null space

rows: list[dict] = []        # long-format record for the summary CSV


def rec(block: str, scenario: str, quantity: str, route: str,
        index: float, value: float, note: str = "") -> None:
    rows.append({"block": block, "scenario": scenario, "quantity": quantity,
                 "route": route, "index": index, "value": value, "note": note})


# ==========================================================================
# Monte Carlo machinery (shared by every S1 scenario)
# ==========================================================================
def mc_ceiling_reps(K, grid, label, protocols, n, reps, rng):
    """Paired Monte Carlo estimate of ``I_g`` for several protocols at once.

    One latent sample ``Z ~ N(0, K)`` per repetition is reused by every
    protocol, so the curves share their Monte Carlo noise (the comparison
    between protocols is far more precise than the individual error bars).
    The posterior mean of ``Theta`` is available in closed form for a threshold
    label -- ``E(Theta | Y) = sum_j omega_j Phibar{(c - m_j) / s_j}`` -- so the
    simulation estimates the true Bayes risk, not a plug-in surrogate.
    """
    p = K.shape[0]
    L = np.linalg.cholesky(K + 1e-10 * np.eye(p))
    omega = grid.weights
    c = float(getattr(label, "c", 0.0))

    prepared = []
    for name, acts in protocols.items():
        A, R = protocol_matrices(acts, grid)
        KA = K @ A.T
        M = A @ KA + R
        Minv = np.linalg.inv(M)
        gain = Minv @ KA.T                                   # d x p
        qdiag = np.einsum("ij,jk,ik->i", KA, Minv, KA)
        sd = np.sqrt(np.maximum(1.0 - qdiag, 1e-12))
        Rchol = np.linalg.cholesky(R + 1e-14 * np.eye(R.shape[0]))
        prepared.append((name, A, Rchol, gain, sd))

    out = {name: {"var": [], "risk": [], "ceiling": []} for name in protocols}
    for _ in range(reps):
        Z = rng.standard_normal((n, p)) @ L.T
        theta = label.apply(Z) @ omega
        var = float(np.var(theta, ddof=1))
        for name, A, Rchol, gain, sd in prepared:
            Y = Z @ A.T + rng.standard_normal((n, A.shape[0])) @ Rchol.T
            mpost = Y @ gain
            pred = (1.0 - ndtr((c - mpost) / sd)) @ omega
            risk = float(np.mean((theta - pred) ** 2))
            out[name]["var"].append(var)
            out[name]["risk"].append(risk)
            out[name]["ceiling"].append(1.0 - risk / var)

    summary = {}
    for name, d in out.items():
        s = {}
        for key, vals in d.items():
            a = np.asarray(vals, dtype=float)
            m = float(a.mean())
            half = float(1.96 * a.std(ddof=1) / np.sqrt(a.size))
            s[key] = m
            s[key + "_ci95"] = half
        summary[name] = s
    return summary


# ==========================================================================
# S1 (a): one-snapshot explainability
# ==========================================================================
def s1_snapshot(rng) -> dict:
    rho = make_kernel("ou", tau=TAU)
    lab = ThresholdLabel(c=0.0)
    t_obs = 0.5 * T_SNAPSHOT
    out: dict = {"alphas": list(ALPHAS), "per_alpha": {}}
    snap_rows: list[dict] = []

    for alpha in ALPHAS:
        # --- continuous route (the conference code path) ------------------
        V_cont = cont.exact_label_variance(T_SNAPSHOT, alpha, rho, "occupation", 0.0)
        I_cont = cont.point_protocol_explainability(
            T_SNAPSHOT, np.array([t_obs]), np.array([SNAPSHOT_NOISE]), rho,
            alpha=alpha, threshold=0.0, grid_size=CONT_GRID)

        # --- new discrete route, over a discretisation ladder --------------
        ladder = {}
        for p in P_LADDER:
            grid = uniform_grid(T_SNAPSHOT, p)
            K = trait_state_correlation(grid, alpha, rho)
            act = [Action(time=t_obs, width=0.0, n_segments=1,
                          noise=SNAPSHOT_NOISE, cost=1.0)]
            report = evaluate_protocol(lab, K, grid, act)
            ladder[p] = {"label_variance": report.total, "ceiling": report.ceiling,
                         "bayes_risk": report.risk}
            snap_rows.append({"alpha": alpha, "p": p,
                              "V_discrete": report.total, "I_discrete": report.ceiling,
                              "V_continuous": V_cont, "I_continuous": I_cont,
                              "I_conference": CONF_CEILINGS[alpha]})
        V_disc = ladder[P_GRID]["label_variance"]
        I_disc = ladder[P_GRID]["ceiling"]

        # --- Monte Carlo at the headline discretisation --------------------
        grid = uniform_grid(T_SNAPSHOT, P_GRID)
        K = trait_state_correlation(grid, alpha, rho)
        acts = {"snapshot": [Action(time=t_obs, width=0.0, n_segments=1,
                                    noise=SNAPSHOT_NOISE, cost=1.0)]}
        mc = mc_ceiling_reps(K, grid, lab, acts, MC_N, MC_REPS, rng)["snapshot"]

        risk_disc = ladder[P_GRID]["bayes_risk"]
        out["per_alpha"][f"{alpha:g}"] = {
            "label_variance_continuous": V_cont,
            "label_variance_discrete_p512": V_disc,
            "label_variance_mc": mc["var"],
            "label_variance_mc_ci95": mc["var_ci95"],
            "bayes_risk_discrete_p512": risk_disc,
            "bayes_risk_mc": mc["risk"],
            "bayes_risk_mc_ci95": mc["risk_ci95"],
            "ceiling_continuous": I_cont,
            "ceiling_discrete_p512": I_disc,
            "ceiling_mc": mc["ceiling"],
            "ceiling_mc_ci95": mc["ceiling_ci95"],
            "ceiling_conference": CONF_CEILINGS[alpha],
            "abs_diff_continuous_vs_conference": abs(I_cont - CONF_CEILINGS[alpha]),
            "abs_diff_discrete_vs_continuous": abs(I_disc - I_cont),
            "abs_diff_mc_vs_discrete": abs(mc["ceiling"] - I_disc),
            "z_mc_vs_discrete": abs(mc["ceiling"] - I_disc) / max(mc["ceiling_ci95"] / 1.96, 1e-300),
            "mc_within_ci_of_discrete": bool(abs(mc["ceiling"] - I_disc) <= mc["ceiling_ci95"]),
            "discretisation_ladder": ladder,
        }
        for route, val in (("continuous", I_cont), ("discrete_p512", I_disc),
                           ("monte_carlo", mc["ceiling"]),
                           ("conference", CONF_CEILINGS[alpha])):
            rec("S1a", "one_snapshot", "ceiling", route, alpha, val)
        rec("S1a", "one_snapshot", "label_variance", "continuous", alpha, V_cont)
        rec("S1a", "one_snapshot", "label_variance", "discrete_p512", alpha, V_disc)
        rec("S1a", "one_snapshot", "label_variance", "monte_carlo", alpha, mc["var"])

        print(f"  alpha={alpha:4.2f}  V: cont {V_cont:.6f} / disc {V_disc:.6f}"
              f" / MC {mc['var']:.6f}+-{mc['var_ci95']:.6f}")
        print(f"             I: cont {I_cont:.6f} / disc {I_disc:.6f}"
              f" / MC {mc['ceiling']:.6f}+-{mc['ceiling_ci95']:.6f}"
              f" / conference {CONF_CEILINGS[alpha]:.3f}")

    save_csv(snap_rows, "s1_s2_snapshot")
    return out


# ==========================================================================
# S1 (b): equal-budget D-versus-M curve
# ==========================================================================
def s1_equal_budget(rng) -> dict:
    rho = make_kernel("ou", tau=TAU)
    lab = ThresholdLabel(c=0.0)
    grid = uniform_grid(T_BUDGET, P_GRID)
    K = trait_state_correlation(grid, 0.0, rho)
    t_mid = 0.5 * (grid.times[0] + grid.times[-1])

    protocols = {}
    for N in BUDGETS:
        protocols[f"same_time_{N}"] = same_time_protocol(grid, N, noise=RAW_NOISE)
        protocols[f"dispersed_{N}"] = dispersed_protocol(grid, N, noise=RAW_NOISE)

    with Timer("S1b Monte Carlo"):
        mc = mc_ceiling_reps(K, grid, lab, protocols, MC_N, MC_REPS, rng)

    curve_rows: list[dict] = []
    per_N: dict = {}
    for N in BUDGETS:
        same_disc = evaluate_protocol(lab, K, grid, protocols[f"same_time_{N}"]).ceiling
        disp_disc = evaluate_protocol(lab, K, grid, protocols[f"dispersed_{N}"]).ceiling
        same_cont = cont.point_protocol_explainability(
            T_BUDGET, np.array([t_mid]), np.array([RAW_NOISE / N]), rho,
            alpha=0.0, threshold=0.0, grid_size=CONT_GRID)
        disp_cont = cont.point_protocol_explainability(
            T_BUDGET, bin_midpoints(T_BUDGET, N), np.full(N, RAW_NOISE), rho,
            alpha=0.0, threshold=0.0, grid_size=CONT_GRID)
        ms, md = mc[f"same_time_{N}"], mc[f"dispersed_{N}"]
        curve_rows.append({
            "N": N,
            "same_continuous": same_cont, "same_discrete": same_disc,
            "same_mc": ms["ceiling"], "same_mc_ci95": ms["ceiling_ci95"],
            "dispersed_continuous": disp_cont, "dispersed_discrete": disp_disc,
            "dispersed_mc": md["ceiling"], "dispersed_mc_ci95": md["ceiling_ci95"],
        })
        per_N[str(N)] = curve_rows[-1]
        for route, val in (("continuous", same_cont), ("discrete_p512", same_disc),
                           ("monte_carlo", ms["ceiling"])):
            rec("S1b", "same_time", "ceiling", route, N, val)
        for route, val in (("continuous", disp_cont), ("discrete_p512", disp_disc),
                           ("monte_carlo", md["ceiling"])):
            rec("S1b", "dispersed", "ceiling", route, N, val)
        print(f"  N={N:3d}  same: cont {same_cont:.6f} disc {same_disc:.6f}"
              f" MC {ms['ceiling']:.4f}+-{ms['ceiling_ci95']:.4f} |"
              f"  disp: cont {disp_cont:.6f} disc {disp_disc:.6f}"
              f" MC {md['ceiling']:.4f}+-{md['ceiling_ci95']:.4f}")

    rec("S1b", "same_time", "ceiling", "conference", 64, CONF_SAME_64)
    rec("S1b", "dispersed", "ceiling", "conference", 64, CONF_DISP_64)
    save_csv(curve_rows, "s1_s2_equal_budget")

    last = per_N["64"]
    out = {
        "curve": per_N,
        "N64": {
            "same_time_continuous": last["same_continuous"],
            "same_time_discrete_p512": last["same_discrete"],
            "same_time_mc": last["same_mc"],
            "same_time_mc_ci95": last["same_mc_ci95"],
            "same_time_conference": CONF_SAME_64,
            "same_time_abs_diff_continuous_vs_conference":
                abs(last["same_continuous"] - CONF_SAME_64),
            "dispersed_continuous": last["dispersed_continuous"],
            "dispersed_discrete_p512": last["dispersed_discrete"],
            "dispersed_mc": last["dispersed_mc"],
            "dispersed_mc_ci95": last["dispersed_mc_ci95"],
            "dispersed_conference": CONF_DISP_64,
            "dispersed_abs_diff_continuous_vs_conference":
                abs(last["dispersed_continuous"] - CONF_DISP_64),
            "ratio_dispersed_over_same_continuous":
                last["dispersed_continuous"] / last["same_continuous"],
        },
        "coincide_at_N1": abs(per_N["1"]["same_discrete"] - per_N["1"]["dispersed_discrete"]),
        "curve_rows": curve_rows,
    }
    return out


# ==========================================================================
# S1 (c): the OU closed form
# ==========================================================================
def s1_ou_closed_form() -> dict:
    rho = make_kernel("ou", tau=TAU)
    numeric = cont.occupation_state_coefficient_ou(0.0, TAU)
    closed = TAU * np.log(2.0) / 2.0
    via_state = cont.state_coefficient(0.0, rho, "occupation", 0.0)
    rec("S1c", "ou_state_coefficient", "A_0", "hermite_quadrature", 0.0, numeric)
    rec("S1c", "ou_state_coefficient", "A_0", "closed_form", 0.0, closed)
    rec("S1c", "ou_state_coefficient", "A_0", "direct_quadrature", 0.0, via_state)
    print(f"  A_0 numeric {numeric:.12f}  closed form tau log2/2 {closed:.12f}"
          f"  |diff| {abs(numeric - closed):.2e}")
    return {"A_0_numeric": numeric, "A_0_closed_form": closed,
            "A_0_via_state_coefficient": via_state,
            "abs_diff": abs(numeric - closed),
            "abs_diff_alternative_route": abs(via_state - closed)}


# ==========================================================================
# S2 (a): the minimal stationary counterexample
# ==========================================================================
def s2_minimal_example() -> dict:
    ex = minimal_stationary_example(tau=TAU)
    cert = ex["certificate"]
    Kp, Km = cert.K_plus, cert.K_minus
    A, B, h = ex["A"], ex["B"], ex["h"]

    # The three functionals the benchmark identifies, computed by hand.
    var_YA_p, var_YA_m = (A @ Kp @ A.T).item(), (A @ Km @ A.T).item()
    cov_p, cov_m = (A @ Kp @ h).item(), (A @ Km @ h).item()
    var_th_p, var_th_m = float(h @ Kp @ h), float(h @ Km @ h)
    joint_gap = max(abs(var_YA_p - var_YA_m), abs(cov_p - cov_m),
                    abs(var_th_p - var_th_m))

    # ... and the whole joint Gaussian law of (Y_A, Theta) as a cross-check.
    G = np.vstack([A, h[None, :]])
    law_gap = float(np.max(np.abs(G @ Kp @ G.T - G @ Km @ G.T)))

    I_plus, I_minus = cert.ceiling_plus, cert.ceiling_minus

    # Occupation-label ceiling of B through the *discrete* interface: the
    # non-identifiability is a property of K, not of the linearity of the label.
    grid4 = TimeGrid(times=np.arange(4.0), weights=np.full(4, 0.25), horizon=3.0)
    acts_B = [Action(time=1.0, width=0.0, noise=0.0, cost=1.0),
              Action(time=2.0, width=0.0, noise=0.0, cost=1.0)]
    occ = ThresholdLabel(c=0.0)
    occ_p = evaluate_protocol(occ, Kp, grid4, acts_B).ceiling
    occ_m = evaluate_protocol(occ, Km, grid4, acts_B).ceiling
    mean_p = evaluate_protocol(MeanLabel(), Kp, grid4, acts_B).ceiling
    mean_m = evaluate_protocol(MeanLabel(), Km, grid4, acts_B).ceiling

    # The observed protocol A is *also* re-evaluated under the occupation label.
    # `thm:impossibility` is stated for a linear label, where observational equivalence
    # is exact; for a nonlinear label the benchmark sees higher-order Hermite
    # functionals of K as well, so equivalence is only approximate.  Quantifying
    # the residual movement of the benchmark is what makes the B-side gap
    # interpretable: it must be far larger than the A-side movement.
    acts_A = [Action(time=0.0, width=0.0, noise=0.0, cost=1.0)]
    occ_A_p = evaluate_protocol(occ, Kp, grid4, acts_A).ceiling
    occ_A_m = evaluate_protocol(occ, Km, grid4, acts_A).ceiling
    V_occ_p = label_variance(occ, Kp, grid4.weights)
    V_occ_m = label_variance(occ, Km, grid4.weights)
    # Cov(Y_A, Theta_occ) = phi(0) sum_j omega_j rho(|j|)  (Stein), which is an
    # exact multiple of the identified linear functional and therefore equal.
    phi0 = 1.0 / np.sqrt(2.0 * np.pi)
    cov_occ_p = phi0 * (A @ Kp @ h).item()
    cov_occ_m = phi0 * (A @ Km @ h).item()

    print(f"  eps = {cert.eps:.10f}")
    print(f"  rho_+ = {np.round(Kp[0], 8)}")
    print(f"  rho_- = {np.round(Km[0], 8)}")
    print(f"  Var(Y_A) {var_YA_p:.12f} / {var_YA_m:.12f}")
    print(f"  Cov(Y_A,Theta) {cov_p:.12f} / {cov_m:.12f}")
    print(f"  Var(Theta) {var_th_p:.12f} / {var_th_m:.12f}")
    print(f"  max |discrepancy| over the identified functionals = {joint_gap:.3e}")
    print(f"  I_B(K_+) = {I_plus:.6f}   I_B(K_-) = {I_minus:.6f}"
          f"   gap {abs(I_plus - I_minus):.6f}")
    print(f"  occupation label:  I_B(K_+) = {occ_p:.6f}  I_B(K_-) = {occ_m:.6f}"
          f"   gap {abs(occ_p - occ_m):.6f}")
    print(f"  occupation label:  benchmark A moves by {abs(occ_A_p - occ_A_m):.2e}"
          f"  (Var Theta_occ {V_occ_p:.8f} / {V_occ_m:.8f},"
          f" Cov(Y_A,Theta_occ) {cov_occ_p:.8f} / {cov_occ_m:.8f})")

    rec("S2a", "minimal_example", "ceiling_B_mean", "linear_ceiling", 1, I_plus)
    rec("S2a", "minimal_example", "ceiling_B_mean", "linear_ceiling", -1, I_minus)
    rec("S2a", "minimal_example", "ceiling_B_occupation", "discrete", 1, occ_p)
    rec("S2a", "minimal_example", "ceiling_B_occupation", "discrete", -1, occ_m)

    return {
        "eps": cert.eps,
        "delta_lags": ex["delta_lags"].tolist(),
        "rho_plus": Kp[0].tolist(),
        "rho_minus": Km[0].tolist(),
        "K0_rho": ex["K0"][0].tolist(),
        "var_Y_A_plus": var_YA_p, "var_Y_A_minus": var_YA_m,
        "cov_Y_A_theta_plus": cov_p, "cov_Y_A_theta_minus": cov_m,
        "var_theta_plus": var_th_p, "var_theta_minus": var_th_m,
        "max_abs_discrepancy_identified_functionals": joint_gap,
        "max_abs_discrepancy_joint_law": law_gap,
        "package_observed_discrepancy": cert.observed_discrepancy,
        "lambda_min_plus": float(np.linalg.eigvalsh(Kp).min()),
        "lambda_min_minus": float(np.linalg.eigvalsh(Km).min()),
        "ceiling_B_plus_mean": I_plus, "ceiling_B_minus_mean": I_minus,
        "ceiling_gap_mean": abs(I_plus - I_minus),
        "ceiling_B_plus_mean_discrete": mean_p,
        "ceiling_B_minus_mean_discrete": mean_m,
        "ceiling_B_plus_occupation": occ_p, "ceiling_B_minus_occupation": occ_m,
        "ceiling_gap_occupation": abs(occ_p - occ_m),
        "ceiling_A_plus_occupation": occ_A_p, "ceiling_A_minus_occupation": occ_A_m,
        "ceiling_gap_A_occupation": abs(occ_A_p - occ_A_m),
        "label_variance_occupation_plus": V_occ_p,
        "label_variance_occupation_minus": V_occ_m,
        "label_variance_occupation_abs_diff": abs(V_occ_p - V_occ_m),
        "cov_YA_theta_occupation_plus": cov_occ_p,
        "cov_YA_theta_occupation_minus": cov_occ_m,
        "cov_YA_theta_occupation_abs_diff": abs(cov_occ_p - cov_occ_m),
        "occupation_gap_ratio_B_over_A": (abs(occ_p - occ_m)
                                          / max(abs(occ_A_p - occ_A_m), 1e-300)),
        "discrete_vs_linear_ceiling_max_err": max(abs(mean_p - I_plus),
                                                  abs(mean_m - I_minus)),
    }


# ==========================================================================
# S2 (b): the sharp p = 3 / p >= 4 threshold
# ==========================================================================
def s2_threshold() -> dict:
    table = {}
    for p in range(3, 9):
        J = stationary_identification_jacobian(p, obs_index=0)
        rank = int(np.linalg.matrix_rank(J, tol=1e-10))
        A = np.zeros((1, p)); A[0, 0] = 1.0
        dirs = nonidentified_directions(A, np.full(p, 1.0 / p),
                                        stationary=True, unit_diagonal=True)
        cb = counting_bound(p, 1, stationary=True)
        table[str(p)] = {"jacobian_rank": rank, "jacobian_shape": list(J.shape),
                         "n_invisible_directions": int(len(dirs)),
                         "free_parameters": cb["free_parameters"],
                         "constraints": cb["constraints"],
                         "deficiency": cb["deficiency"]}
        rec("S2b", f"p={p}", "n_invisible_directions", "stationary", p, float(len(dirs)))
        rec("S2b", f"p={p}", "jacobian_rank", "stationary", p, float(rank))
        print(f"  p={p}: rank(J)={rank}/{J.shape[0]}  free={cb['free_parameters']}"
              f"  constraints={cb['constraints']}  invisible directions={len(dirs)}")
    return {
        "table": table,
        "p3_locally_identified": bool(table["3"]["jacobian_rank"] == 2
                                      and table["3"]["n_invisible_directions"] == 0),
        "p4_has_invisible_direction": bool(table["4"]["n_invisible_directions"] >= 1),
        "threshold_is_sharp": bool(table["3"]["n_invisible_directions"] == 0
                                   and all(table[str(p)]["n_invisible_directions"] >= 1
                                           for p in range(4, 9))),
    }


# ==========================================================================
# S2 (c): genericity in higher dimension
# ==========================================================================
def random_correlation(p: int, rng) -> np.ndarray:
    """A generic unit-diagonal PD covariance (Wishart, rescaled)."""
    W = rng.standard_normal((p, p + 3))
    return to_correlation(W @ W.T / (p + 3))


def s2_genericity(rng) -> dict:
    inst_rows: list[dict] = []
    summary: dict = {}
    for p in S2_P_VALUES:
        for d in S2_D_OBS:
            cb = counting_bound(p, d, stationary=False)
            n_hit, gaps, epss, n_dirs = 0, [], [], []
            for _ in range(S2_INSTANCES):
                K0 = random_correlation(p, rng)
                idx = rng.choice(p, size=d, replace=False)
                A = np.eye(p)[idx]
                rest = np.setdiff1d(np.arange(p), idx)
                jdx = rng.choice(rest, size=min(d, rest.size), replace=False)
                B = np.eye(p)[jdx]
                h = np.full(p, 1.0 / p)

                dirs = nonidentified_directions(A, h, stationary=False,
                                                unit_diagonal=True)
                n_dirs.append(len(dirs))
                if len(dirs) == 0:
                    inst_rows.append({"p": p, "d_obs": d, "n_directions": 0,
                                      "eps": 0.0, "ceiling_gap": 0.0,
                                      "observed_discrepancy": 0.0,
                                      "I_plus": np.nan, "I_minus": np.nan})
                    continue
                n_hit += 1

                best = None
                for _t in range(S2_DIR_TRIALS):
                    coef = rng.standard_normal(len(dirs))
                    D = np.tensordot(coef, dirs, axes=1)
                    D = D / max(np.linalg.norm(D, 2), 1e-300)
                    eps = max_psd_step(K0, D)
                    if eps <= 0.0:
                        continue
                    Kp, Km = K0 + eps * D, K0 - eps * D
                    gap = abs(linear_ceiling(Kp, B, h) - linear_ceiling(Km, B, h))
                    if best is None or gap > best[0]:
                        best = (gap, eps, Kp, Km)
                if best is None:
                    continue
                gap, eps, Kp, Km = best
                disc = observed_discrepancy(Kp, Km, A, h)
                gaps.append(gap); epss.append(eps)
                inst_rows.append({
                    "p": p, "d_obs": d, "n_directions": int(len(dirs)),
                    "eps": eps, "ceiling_gap": gap, "observed_discrepancy": disc,
                    "I_plus": linear_ceiling(Kp, B, h),
                    "I_minus": linear_ceiling(Km, B, h)})

            g = np.asarray(gaps, dtype=float)
            key = f"p{p}_d{d}"
            summary[key] = {
                "p": p, "d_obs": d, "n_instances": S2_INSTANCES,
                "n_with_invisible_direction": n_hit,
                "fraction_with_invisible_direction": n_hit / S2_INSTANCES,
                "mean_n_directions": float(np.mean(n_dirs)),
                "free_parameters": cb["free_parameters"],
                "constraints": cb["constraints"],
                "deficiency": cb["deficiency"],
                "gap_min": float(g.min()), "gap_q25": float(np.quantile(g, 0.25)),
                "gap_median": float(np.median(g)),
                "gap_q75": float(np.quantile(g, 0.75)),
                "gap_max": float(g.max()), "gap_mean": float(g.mean()),
                "eps_median": float(np.median(epss)),
                "max_observed_discrepancy": float(
                    max(r["observed_discrepancy"] for r in inst_rows
                        if r["p"] == p and r["d_obs"] == d)),
            }
            rec("S2c", key, "fraction_with_invisible_direction", "random", p,
                n_hit / S2_INSTANCES, f"d_obs={d}")
            rec("S2c", key, "ceiling_gap_median", "random", p,
                float(np.median(g)), f"d_obs={d}")
            print(f"  p={p:2d} d={d}: {n_hit}/{S2_INSTANCES} admit invisible directions"
                  f" (mean dim {np.mean(n_dirs):.1f}, deficiency {cb['deficiency']});"
                  f" gap median {np.median(g):.4f}"
                  f" [{np.quantile(g, .25):.4f}, {np.quantile(g, .75):.4f}]"
                  f" max {g.max():.4f}")

    save_csv(inst_rows, "s1_s2_random_instances")
    return {"summary": summary, "instances": inst_rows}


# ==========================================================================
# Figures
# ==========================================================================
def figure_identifiability(plt, mini: dict) -> None:
    # Two columns: the two correlation functions, and the counterfactual value
    # of protocol B under each.  An earlier draft carried a third panel with the
    # distribution of ceiling gaps over the random higher-dimensional instances;
    # those numbers are in results/s1_s2_random_instances.csv and in the S2c
    # block of the JSON, and the figure is deliberately kept to the minimal
    # example that a reader can check by hand.
    fig = plt.figure(figsize=(4.6, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 0.42], wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    # ---- panel (a): the two correlation functions -----------------------
    lags = np.arange(4)
    rp, rm = np.asarray(mini["rho_plus"]), np.asarray(mini["rho_minus"])
    ax_a.axhline(0.0, color="0.8", lw=0.6, zorder=0)
    ax_a.plot(lags, np.asarray(mini["K0_rho"]), marker=".", color="0.55", ls=":",
              lw=1.0, label=r"$\rho_{0}(u)=e^{-u/\tau}$")
    ax_a.plot(lags, rp, marker="o", color=PALETTE[0], ls="-",
              label=r"$\rho_{+}$  ($K_0+\varepsilon\Delta$)")
    ax_a.plot(lags, rm, marker="s", color=PALETTE[1], ls="--",
              label=r"$\rho_{-}$  ($K_0-\varepsilon\Delta$)")
    ax_a.set_xlabel("lag $u$")
    ax_a.set_ylabel(r"correlation $\rho(u)$")
    ax_a.set_xticks(lags)
    ax_a.set_ylim(-0.62, 1.16)
    ax_a.set_yticks([-0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax_a.legend(loc="upper right", handlelength=1.9, labelspacing=0.25,
                borderpad=0.2)
    ax_a.text(
        0.015, 0.02,
        "observed functionals identical:\n"
        r"$\mathrm{Var}(Y_A)=%.6f$" % mini["var_Y_A_plus"] + "\n"
        r"$\mathrm{Cov}(Y_A,\Theta)=%.6f$" % mini["cov_Y_A_theta_plus"] + "\n"
        r"$\mathrm{Var}(\Theta)=%.6f$" % mini["var_theta_plus"] + "\n"
        r"max discrepancy $=%.0e$" % max(
            mini["max_abs_discrepancy_identified_functionals"], 1.1e-16),
        transform=ax_a.transAxes, fontsize=6.0, va="bottom", ha="left",
        linespacing=1.35)

    # ---- the counterfactual protocol values of protocol B ----------------------
    # Only the mean target is plotted.  The stationary four-point construction
    # of `thm:minimal` is proved for the mean-target benchmark; the occupation
    # target is reached only by the separate permutation construction of
    # `prop:permutation`, a different instance, so putting it in this figure
    # would suggest the four-point witness proves more than it does.
    vals = [mini["ceiling_B_plus_mean"], mini["ceiling_B_minus_mean"]]
    ax_bar.bar([0, 1], vals, color=[PALETTE[0], PALETTE[1]], width=0.66)
    for x, v in zip([0, 1], vals):
        ax_bar.text(x, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=6.2)
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels([r"$K_+$", r"$K_-$"])
    ax_bar.set_xlim(-0.65, 1.65)
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_bar.set_ylabel(r"counterfactual protocol value $\mathcal{I}(B;K)$")
    ax_bar.set_xlabel("mean target", fontsize=6.2)

    fig.tight_layout()
    save_figure(fig, "fig_identifiability")
    plt.close(fig)


def figure_equal_budget(plt, eb: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    r = eb["curve_rows"]
    N = np.array([x["N"] for x in r], dtype=float)

    ax.plot(N, [x["same_continuous"] for x in r], color="0.45", ls=":", marker=None,
            label=r"same time ($D=1,M=N$), exact")
    ax.errorbar(N, [x["same_mc"] for x in r],
                yerr=[x["same_mc_ci95"] for x in r], fmt="+", color="0.45",
                ms=5, capsize=2, lw=0.9, label="same time, Monte Carlo")
    ax.plot(N, [x["dispersed_continuous"] for x in r], color=PALETTE[0], ls="-",
            label=r"dispersed ($D=N,M=1$), exact")
    ax.errorbar(N, [x["dispersed_mc"] for x in r],
                yerr=[x["dispersed_mc_ci95"] for x in r], fmt="o", color=PALETTE[0],
                ms=3.4, capsize=2, lw=0.9, mfc="white",
                label="dispersed, Monte Carlo")

    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(n) for n in BUDGETS])
    ax.set_xlabel(r"raw-segment budget $N = D\,M$")
    ax.set_ylabel(r"protocol value $I_g(S)$")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, "fig_equal_budget")
    plt.close(fig)


# ==========================================================================
def main() -> None:
    t_start = time.perf_counter()
    plt = setup_matplotlib()
    rng = np.random.default_rng(SEED)

    print("\n=== S1a: one-snapshot explainability (T/tau = 14, nu^2 = 0.2) ===")
    with Timer("S1a"):
        snap = s1_snapshot(rng)

    print("\n=== S1b: equal-budget D-vs-M curve (T/tau = 20, alpha = 0) ===")
    with Timer("S1b"):
        eb = s1_equal_budget(rng)

    print("\n=== S1c: OU closed form A_0 = tau log(2)/2 ===")
    ou = s1_ou_closed_form()

    print("\n=== S2a: minimal stationary counterexample ===")
    mini = s2_minimal_example()

    print("\n=== S2b: sharp p = 3 / p >= 4 threshold ===")
    thr = s2_threshold()

    print("\n=== S2c: genericity over random higher-dimensional instances ===")
    with Timer("S2c"):
        generic = s2_genericity(rng)

    print("\n=== figures ===")
    figure_identifiability(plt, mini)
    figure_equal_budget(plt, eb)

    runtime = time.perf_counter() - t_start
    save_csv(rows, "s1_s2_regression_identifiability")

    # ---- discrepancy audit ------------------------------------------------
    disc = {
        "S1a_max_abs_continuous_vs_conference": max(
            snap["per_alpha"][f"{a:g}"]["abs_diff_continuous_vs_conference"]
            for a in ALPHAS),
        "S1a_max_abs_discrete_vs_continuous": max(
            snap["per_alpha"][f"{a:g}"]["abs_diff_discrete_vs_continuous"]
            for a in ALPHAS),
        "S1a_max_abs_mc_vs_discrete": max(
            snap["per_alpha"][f"{a:g}"]["abs_diff_mc_vs_discrete"] for a in ALPHAS),
        "S1a_max_z_mc_vs_discrete": max(
            snap["per_alpha"][f"{a:g}"]["z_mc_vs_discrete"] for a in ALPHAS),
        "S1a_mc_within_ci_all_alphas": all(
            snap["per_alpha"][f"{a:g}"]["mc_within_ci_of_discrete"] for a in ALPHAS),
        "S1b_same_time_continuous_vs_conference_N64":
            eb["N64"]["same_time_abs_diff_continuous_vs_conference"],
        "S1b_dispersed_continuous_vs_conference_N64":
            eb["N64"]["dispersed_abs_diff_continuous_vs_conference"],
        "S1b_max_abs_discrete_vs_continuous": max(
            max(abs(x["same_discrete"] - x["same_continuous"]),
                abs(x["dispersed_discrete"] - x["dispersed_continuous"]))
            for x in eb["curve_rows"]),
        "S1b_max_abs_mc_vs_discrete": max(
            max(abs(x["same_mc"] - x["same_discrete"]),
                abs(x["dispersed_mc"] - x["dispersed_discrete"]))
            for x in eb["curve_rows"]),
        "S1b_max_mc_ci95": max(max(x["same_mc_ci95"], x["dispersed_mc_ci95"])
                               for x in eb["curve_rows"]),
        "S1b_max_z_mc_vs_discrete": max(
            max(abs(x["same_mc"] - x["same_discrete"]) / max(x["same_mc_ci95"] / 1.96, 1e-300),
                abs(x["dispersed_mc"] - x["dispersed_discrete"])
                / max(x["dispersed_mc_ci95"] / 1.96, 1e-300))
            for x in eb["curve_rows"]),
        "S1c_A0_abs_diff": ou["abs_diff"],
        "S2a_max_abs_discrepancy_identified_functionals":
            mini["max_abs_discrepancy_identified_functionals"],
        "S2c_max_observed_discrepancy": max(
            v["max_observed_discrepancy"] for v in generic["summary"].values()),
    }

    headline = {
        "seed": SEED,
        "monte_carlo_objects_per_rep": MC_N,
        "monte_carlo_reps": MC_REPS,
        "discrete_grid_p": P_GRID,
        "continuous_quadrature_grid": CONT_GRID,
        "s1a_snapshot": snap,
        "s1b_equal_budget": eb,
        "s1c_ou_closed_form": ou,
        "s2a_minimal_example": mini,
        "s2b_threshold": thr,
        "s2c_genericity": {"summary": generic["summary"],
                           "n_instances_per_cell": S2_INSTANCES,
                           "n_direction_trials": S2_DIR_TRIALS},
        "discrepancy_audit": disc,
        "runtime_seconds": runtime,
        "environment": environment_record(),
    }
    save_json(headline, "s1_s2_regression_identifiability")

    print("\n=== discrepancy audit ===")
    for k, v in disc.items():
        print(f"  {k:58s} {v}")
    print(f"\ntotal runtime {runtime:.1f}s")


if __name__ == "__main__":
    main()
