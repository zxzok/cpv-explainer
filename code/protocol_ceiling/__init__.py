"""protocol_ceiling: protocol-limited learning of temporal aggregates.

Identifiability, calibration-based estimation, and label-aware observation
design for labels of the form ``Theta = sum_j omega_j g(Z_j)`` observed through
a sparse, noisy temporal protocol.

Public entry points
-------------------
``fit_covariance``                 estimate ``K`` from dense calibration data
``estimate_protocol_ceiling``      plug-in ``I_hat_g(S)``
``bootstrap_protocol_ceiling``     object-level percentile interval
``evaluate_protocol``              exact ceiling / risk report for a protocol
``select_protocol_greedy``         label-aware greedy design
``select_protocol_robust``         lower-confidence-bound design
``nonidentified_directions``       basis of counterfactually invisible ``Delta``
``minimal_stationary_example``     the sharp four-point counterexample
"""

from .covariance import (Action, TimeGrid, allocation_protocol, bin_midpoints,
                         candidate_actions,
                         dispersed_protocol, kernel_matrix, low_rank_approx,
                         make_kernel, project_psd, recency_weight, sample_paths,
                         same_time_protocol, stationary_toeplitz, to_correlation,
                         trait_state_correlation, uniform_grid)
from .design import (DesignResult, design_imse, design_kernel_quadrature,
                     design_mutual_information, design_random, design_same_time,
                     design_uniform, find_submodularity_violation, objective,
                     swap_local_search,
                     nonlinear_ratio_lower_bound, refine_protocol_continuous,
                     select_protocol_exhaustive, select_protocol_greedy,
                     select_protocol_robust, submodularity_ratio_certificate)
from .diagnostics import (LEARNERS, RiskDecomposition, r2_score,
                          within_between_r2)
from .estimation import (CovarianceFit, effective_rank, estimate_ceiling_family,
                         estimate_protocol_ceiling, fit_covariance,
                         selection_regret_bound, trait_ceiling,
                         trait_ceiling_interval, trait_share_interval,
                         uniform_error_bound)
from .identifiability import (certify, counting_bound, linear_ceiling,
                              max_psd_step, minimal_stationary_example,
                              nonidentified_directions, observed_discrepancy,
                              stationary_identification_jacobian)
from .risk import (CeilingReport, ProtocolState, bayes_risk, ceiling_utilization,
                   evaluate_protocol, explained_covariance, explained_variance,
                   label_variance, protocol_ceiling, residual_covariance,
                   trait_state_split)
from .transforms import (HermiteLabel, LabelFunctional, MeanLabel, SquareLabel,
                         ThresholdLabel, TwoSidedLabel, hermite_coefficients,
                         indicator_hermite_coefficients, make_label, sigmoid_label)
from .resolution import (ProtocolClass, ResolutionSelection,
                         bootstrap_uniform_error, nested_classes,
                         resolution_adaptive_select, theorem_bound,
                         uniform_error)
from .values import (ProtocolValues, best_linear_value,
                     best_linear_value_from_moments, gaussian_bayes_value,
                     linear_value_of_protocol)
from .uncertainty import (BootstrapResult, bootstrap_covariances,
                          bootstrap_protocol_ceiling, coverage,
                          lower_confidence_bound)

__version__ = "0.1.0"

__all__ = [name for name in dir() if not name.startswith("_")]
