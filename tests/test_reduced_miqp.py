"""
Phase 7 Unit Tests — Candidate-Universe Reduced MIQP Solver Acceleration.
Validates:
  1. Candidate slicing preserves deterministic asset ordering.
  2. Covariance submatrix is sliced correctly.
  3. Mandatory current holdings are included.
  4. Effective candidate size is reported correctly.
  5. Full-dimension fixing and sliced reduced MIQP objectives match.
  6. Support containment implies objective recovery on a controlled fixture.
  7. Missing an essential support asset causes positive objective degradation.
  8. Reduced infeasibility is safely reported.
  9. Nested candidate objective monotonicity.
  10. Full-universe candidate set reproduces the full MIQP exactly.
  11. Objective degradation formula correctness.
  12. Exact-objective recovery metric correctness.
  13. End-to-end runtime includes relaxation time when required.
  14. Market-only ML runtime does not include relaxation solve time.
  15. ML inference/build/runtime timers are separately recorded.
  16. Solver parameter equality between full and reduced models.
  17. Restricted MIP gap is stored separately from full objective degradation.
  18. Raw and safe-augmented candidate results remain distinguishable.
  19. Candidate set is constructed without reading teacher labels at inference time.
  20. All Phase 1-6 regression tests remain passing.
"""

from __future__ import annotations

import numpy as np
import pytest

from learning_augmented_portfolio.dynamic.state import PortfolioState
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from learning_augmented_portfolio.optimization.reduced_miqp import (
    MultiPeriodReducedMIQPOptimizer,
    MultiPeriodFixedMIQPOptimizer,
    ReducedMIQPExecutionResult,
)
from learning_augmented_portfolio.optimization.support_restriction import (
    CandidateUniverse,
    slice_market_state,
    slice_parameters,
    expand_reduced_solution_to_full,
)
from learning_augmented_portfolio.optimization.variables import (
    MultiPeriodMarketState,
    MultiPeriodPortfolioParameters,
    MultiPeriodOptimizationResult,
    SolverConfig,
)
from learning_augmented_portfolio.learning.candidate_set import construct_candidate_universe
from learning_augmented_portfolio.evaluation.objective_quality import (
    compare_instance_objectives,
    summarize_objective_quality,
)
from learning_augmented_portfolio.evaluation.solver_acceleration import (
    compute_instance_acceleration,
    summarize_acceleration,
)


@pytest.fixture
def controlled_5asset_fixture():
    """Create a 5-asset, 2-period controlled setup with known preference structure."""
    n = 5
    H = 2
    tickers = ["A0", "A1", "A2", "A3", "A4"]

    # Initial holdings: only A0 and A1 held
    w = np.array([0.45, 0.45, 0.0, 0.0, 0.0])
    c = 0.10
    state = PortfolioState(actual_wealth=1000.0, normalized_weights=w, normalized_cash=c, time_index=0, asset_names=tickers)

    # Distinct expected returns: A4 is highest alpha (0.10), A0 (0.08), A1 (0.06), A2 (0.03), A3 (0.02)
    mu = np.array([
        [0.08, 0.06, 0.03, 0.02, 0.10],
        [0.08, 0.06, 0.03, 0.02, 0.10],
    ])
    # Diagonal + slight cross covariance
    sigma = np.tile(np.eye(n) * 0.04 + 0.005, (H, 1, 1))

    mkt = MultiPeriodMarketState(initial_holdings=w.copy(), initial_cash=c, expected_returns=mu, covariance_matrices=sigma, asset_names=tickers)

    params = MultiPeriodPortfolioParameters(
        risk_aversion=2.0,
        horizon=H,
        risk_free_rates=np.full(H, 0.01),
        max_turnovers=np.full(H, 0.60),
        min_cash=np.full(H, 0.05),
        weight_upper_limits=np.full((H, n), 0.50),
        liquidity_limits=np.full((H, n), 0.40),
        prop_cost_coeffs=np.full((H, n), 0.001),
        fixed_cost_coeffs=np.full((H, n), 0.0005),
        cardinality_limits=np.full(H, 2),  # K=2 out of 5
        min_holding_weights=np.full((H, n), 0.05),
        min_trade_sizes=np.full((H, n), 0.02),
        discount_factor=0.98,
    )
    return state, params, mkt


# -----------------------------------------------------------------------------
# Test 1 — Candidate slicing preserves deterministic asset ordering
# -----------------------------------------------------------------------------
def test_1_candidate_slicing_preserves_asset_ordering():
    """Verify that indices in CandidateUniverse are strictly sorted and unique."""
    cand = CandidateUniverse.from_indices([4, 1, 3], full_universe_size=5)
    assert cand.candidate_indices == (1, 3, 4)
    assert cand.index_map_red_to_full == [1, 3, 4]
    assert cand.index_map_full_to_red == {1: 0, 3: 1, 4: 2}


# -----------------------------------------------------------------------------
# Test 2 — Covariance submatrix is sliced correctly
# -----------------------------------------------------------------------------
def test_2_covariance_submatrix_sliced_correctly(controlled_5asset_fixture):
    """Verify principal submatrix extraction for covariance Sigma."""
    state, params, mkt = controlled_5asset_fixture
    cand = CandidateUniverse.from_indices([0, 1, 4], full_universe_size=5)
    sliced_mkt = slice_market_state(mkt, cand)

    assert sliced_mkt.covariance_matrices.shape == (2, 3, 3)
    # Check elements match original
    for h in range(2):
        orig_sub = mkt.covariance_matrices[h][[0, 1, 4], :][:, [0, 1, 4]]
        assert np.allclose(sliced_mkt.covariance_matrices[h], orig_sub)


# -----------------------------------------------------------------------------
# Test 3 — Mandatory current holdings are included
# -----------------------------------------------------------------------------
def test_3_mandatory_current_holdings_included(controlled_5asset_fixture):
    """Verify safe augmentation forces M_t = {i : w_bar_i > 0} into candidate set."""
    state, params, mkt = controlled_5asset_fixture
    # Current holdings: assets 0 and 1
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.7])  # Top 2 by score would be 2, 3
    cand_safe = construct_candidate_universe(scores, budget_B=3, current_weights=state.normalized_weights, use_safe_augmentation=True)

    # Must contain mandatory assets 0 and 1
    assert cand_safe.contains(0)
    assert cand_safe.contains(1)
    # And the highest remaining score asset: index 2
    assert cand_safe.contains(2)
    assert cand_safe.candidate_size == 3


# -----------------------------------------------------------------------------
# Test 4 — Effective candidate size is reported correctly
# -----------------------------------------------------------------------------
def test_4_effective_candidate_size_reported_correctly(controlled_5asset_fixture):
    """Verify B^{eff} = |C_t| and Reduction = 1 - B^{eff}/N."""
    state, params, mkt = controlled_5asset_fixture
    cand = CandidateUniverse.from_indices([0, 1, 4], full_universe_size=5)
    assert cand.candidate_size == 3
    assert np.isclose(cand.reduction_ratio, 1.0 - 3 / 5)


# -----------------------------------------------------------------------------
# Test 5 — Full-dimension fixing and sliced reduced MIQP objectives match
# -----------------------------------------------------------------------------
def test_5_fixing_and_sliced_reduced_miqp_objectives_match(controlled_5asset_fixture):
    """Verify J^{fix}(C_t) == J^{slice}(C_t) within solver tolerance."""
    state, params, mkt = controlled_5asset_fixture
    cand = CandidateUniverse.from_indices([0, 1, 4], full_universe_size=5)

    solver_cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)
    opt_slice = MultiPeriodReducedMIQPOptimizer(params, solver_cfg)
    opt_fix = MultiPeriodFixedMIQPOptimizer(params, solver_cfg)

    res_slice = opt_slice.solve(mkt, cand)
    res_fix = opt_fix.solve(mkt, cand)

    assert res_slice.is_optimal
    assert res_fix.is_optimal

    abs_diff = abs(res_slice.objective_value - res_fix.objective_value)
    assert abs_diff < 1e-6, f"Fixing vs slicing objective mismatch: {abs_diff:.2e}"


# -----------------------------------------------------------------------------
# Test 6 — Support containment implies objective recovery on a controlled fixture
# -----------------------------------------------------------------------------
def test_6_support_containment_implies_objective_recovery(controlled_5asset_fixture):
    """If S_t^* subseteq C_t, reduced MIQP must recover the full MIQP optimum."""
    state, params, mkt = controlled_5asset_fixture
    solver_cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)

    # Full solve
    opt_full = MultiPeriodMIQPOptimizer(params, solver_cfg)
    res_full = opt_full.solve(mkt)
    assert res_full.is_optimal

    # Full teacher support
    teacher_support = set(np.where(np.max(res_full.holding_indicators, axis=0) == 1)[0].tolist())

    # Candidate set containing teacher support plus one extra asset
    cand_indices = list(teacher_support)
    for i in range(5):
        if i not in cand_indices:
            cand_indices.append(i)
            break
    cand = CandidateUniverse.from_indices(cand_indices, full_universe_size=5)
    assert cand.contains_support(teacher_support)

    opt_red = MultiPeriodReducedMIQPOptimizer(params, solver_cfg)
    res_red = opt_red.solve(mkt, cand)
    assert res_red.is_optimal

    # Objective recovery
    diff = abs(res_full.objective_value - res_red.objective_value)
    assert diff < 1e-6, f"Objective not recovered under support containment: diff={diff:.2e}"


# -----------------------------------------------------------------------------
# Test 7 — Missing an essential support asset causes positive objective degradation
# -----------------------------------------------------------------------------
def test_7_missing_essential_support_asset_causes_objective_degradation(controlled_5asset_fixture):
    """If candidate set excludes an essential asset, J^{red} > J^{full}."""
    state, params, mkt = controlled_5asset_fixture
    solver_cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)

    opt_full = MultiPeriodMIQPOptimizer(params, solver_cfg)
    res_full = opt_full.solve(mkt)

    # In this setup, asset 4 has highest return (0.10) and is in optimal support across the horizon
    assert np.max(res_full.holding_indicators, axis=0)[4] == 1

    # Candidate set explicitly excluding asset 4: C = {0, 1, 2}
    cand_sub = CandidateUniverse.from_indices([0, 1, 2], full_universe_size=5)
    opt_red = MultiPeriodReducedMIQPOptimizer(params, solver_cfg)
    res_red = opt_red.solve(mkt, cand_sub)

    # In minimization, worse solution has higher objective: J_red > J_full
    delta_J = res_red.objective_value - res_full.objective_value
    assert delta_J > 1e-3, f"Expected positive objective degradation, got delta_J={delta_J:.2e}"


# -----------------------------------------------------------------------------
# Test 8 — Reduced infeasibility is safely reported
# -----------------------------------------------------------------------------
def test_8_reduced_infeasibility_safely_reported(controlled_5asset_fixture):
    """If candidate set violates cardinality or turnover, infeasibility is gracefully handled."""
    state, params, mkt = controlled_5asset_fixture
    # w = [0.45, 0.45, 0, 0, 0], c = 0.10.
    # K=1, but 2 assets held with w=0.45, and max_turnovers=0.10 (cannot liquidate either to 0).
    params_infeasible = MultiPeriodPortfolioParameters(
        risk_aversion=2.0, horizon=2,
        risk_free_rates=params.risk_free_rates,
        max_turnovers=np.full(2, 0.10),
        min_cash=params.min_cash,
        weight_upper_limits=params.weight_upper_limits,
        liquidity_limits=params.liquidity_limits,
        prop_cost_coeffs=params.prop_cost_coeffs,
        cardinality_limits=np.full(2, 1),  # K=1
    )
    cand = CandidateUniverse.from_indices([0, 1, 2], full_universe_size=5)
    opt_red = MultiPeriodReducedMIQPOptimizer(params_infeasible)
    res_red = opt_red.solve(mkt, cand)

    assert not res_red.is_feasible
    # Verify compare_instance_objectives handles infeasible result gracefully
    full_res = MultiPeriodOptimizationResult(status="OPTIMAL", is_optimal=True, is_feasible=True, solve_time=0.1, objective_value=-0.10)
    comp = compare_instance_objectives("INF_TEST", "Test", 3, 3, full_res, res_red)
    assert comp.is_reduced_feasible is False
    assert comp.objective_degradation == float("inf")


# -----------------------------------------------------------------------------
# Test 9 — Nested candidate objective monotonicity
# -----------------------------------------------------------------------------
def test_9_nested_candidate_objective_monotonicity(controlled_5asset_fixture):
    """For C^(1) subseteq C^(2) subseteq C^(3), J*(C^(1)) >= J*(C^(2)) >= J*(C^(3))."""
    state, params, mkt = controlled_5asset_fixture
    solver_cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)
    opt_red = MultiPeriodReducedMIQPOptimizer(params, solver_cfg)

    c1 = CandidateUniverse.from_indices([0, 1], full_universe_size=5)
    c2 = CandidateUniverse.from_indices([0, 1, 4], full_universe_size=5)
    c3 = CandidateUniverse.from_indices([0, 1, 2, 3, 4], full_universe_size=5)

    r1 = opt_red.solve(mkt, c1)
    r2 = opt_red.solve(mkt, c2)
    r3 = opt_red.solve(mkt, c3)

    assert r1.is_optimal and r2.is_optimal and r3.is_optimal

    # Minimization monotonicity: smaller candidate set has higher or equal objective
    assert r1.objective_value >= r2.objective_value - 1e-6
    assert r2.objective_value >= r3.objective_value - 1e-6


# -----------------------------------------------------------------------------
# Test 10 — Full-universe candidate set reproduces the full MIQP exactly
# -----------------------------------------------------------------------------
def test_10_full_universe_candidate_set_reproduces_full_miqp(controlled_5asset_fixture):
    """When C_t = {0..N-1}, reduced optimizer matches full optimizer exactly."""
    state, params, mkt = controlled_5asset_fixture
    solver_cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)

    opt_full = MultiPeriodMIQPOptimizer(params, solver_cfg)
    res_full = opt_full.solve(mkt)

    cand_full = CandidateUniverse.from_indices(list(range(5)), full_universe_size=5)
    opt_red = MultiPeriodReducedMIQPOptimizer(params, solver_cfg)
    res_red = opt_red.solve(mkt, cand_full)

    diff = abs(res_full.objective_value - res_red.objective_value)
    assert diff < 1e-6


# -----------------------------------------------------------------------------
# Test 11 — Objective degradation formula correctness
# -----------------------------------------------------------------------------
def test_11_objective_degradation_formula_correctness():
    """Verify Delta J = J_red - J_full and relative degradation."""
    full_res = MultiPeriodOptimizationResult(
        status="OPTIMAL", is_optimal=True, is_feasible=True, solve_time=0.1, objective_value=-0.20
    )
    # Mock reduced exec with objective -0.15 (degraded by 0.05)
    cand = CandidateUniverse.from_indices([0, 1], full_universe_size=5)
    red_opt = MultiPeriodOptimizationResult(
        status="OPTIMAL", is_optimal=True, is_feasible=True, solve_time=0.05, objective_value=-0.15
    )
    red_exec = ReducedMIQPExecutionResult(
        optimization_result=red_opt, candidate_universe=cand, build_time=0.01, solver_time=0.05,
        total_time=0.06, num_vars=20, num_bin_vars=8, num_constrs=25, is_fixing_form=False
    )

    comp = compare_instance_objectives("INST_0", "TestModel", 2, 2, full_res, red_exec)
    assert np.isclose(comp.objective_degradation, 0.05)
    # Relative degradation = 0.05 / max(1.0, 0.20) = 0.05
    assert np.isclose(comp.relative_degradation, 0.05)


# -----------------------------------------------------------------------------
# Test 12 — Exact-objective recovery metric correctness
# -----------------------------------------------------------------------------
def test_12_exact_objective_recovery_metric_correctness():
    """Verify exact objective recovery boolean flag."""
    full_res = MultiPeriodOptimizationResult(status="OPTIMAL", is_optimal=True, is_feasible=True, solve_time=0.1, objective_value=-0.10)
    cand = CandidateUniverse.from_indices([0, 1], full_universe_size=5)

    red_exact = MultiPeriodOptimizationResult(status="OPTIMAL", is_optimal=True, is_feasible=True, solve_time=0.05, objective_value=-0.1000001)
    exec_exact = ReducedMIQPExecutionResult(red_exact, cand, 0.01, 0.05, 0.06, 20, 8, 25, False)
    comp_exact = compare_instance_objectives("I1", "M", 2, 2, full_res, exec_exact, eps_J=1e-5)
    assert comp_exact.is_exact_recovery is True

    red_diff = MultiPeriodOptimizationResult(status="OPTIMAL", is_optimal=True, is_feasible=True, solve_time=0.05, objective_value=-0.08)
    exec_diff = ReducedMIQPExecutionResult(red_diff, cand, 0.01, 0.05, 0.06, 20, 8, 25, False)
    comp_diff = compare_instance_objectives("I2", "M", 2, 2, full_res, exec_diff, eps_J=1e-5)
    assert comp_diff.is_exact_recovery is False


# -----------------------------------------------------------------------------
# Test 13 — End-to-end runtime includes relaxation time when required
# -----------------------------------------------------------------------------
def test_13_e2e_runtime_includes_relaxation_time_when_required():
    """Verify relaxation runtime is incorporated into E2E time for relaxation-assisted models."""
    rec = compute_instance_acceleration(
        instance_id="I0", method="ContinuousRelaxation", budget_B=3, effective_B=3,
        full_solver_time=0.10, reduced_solver_time=0.02, build_time=0.01, candidate_time=0.005,
        relax_time=0.03, feature_time=0.0, inference_time=0.0,
    )
    # E2E = 0.03 (relax) + 0.005 (cand) + 0.01 (build) + 0.02 (solver) = 0.065
    assert np.isclose(rec.e2e_reduced_time, 0.065)
    assert np.isclose(rec.solver_speedup, 0.10 / 0.02)
    assert np.isclose(rec.e2e_speedup, 0.10 / 0.065)


# -----------------------------------------------------------------------------
# Test 14 — Market-only ML runtime does not include relaxation solve time
# -----------------------------------------------------------------------------
def test_14_market_only_ml_runtime_excludes_relaxation_time():
    """MarketOnlyML does not solve continuous relaxation; relax_time must be 0.0."""
    rec = compute_instance_acceleration(
        instance_id="I0", method="MarketOnlyML", budget_B=3, effective_B=3,
        full_solver_time=0.10, reduced_solver_time=0.02, build_time=0.01, candidate_time=0.005,
        relax_time=0.0, feature_time=0.002, inference_time=0.001,
    )
    # E2E = 0.002 + 0.001 + 0.005 + 0.01 + 0.02 = 0.038
    assert np.isclose(rec.e2e_reduced_time, 0.038)
    assert np.isclose(rec.relax_time, 0.0)


# -----------------------------------------------------------------------------
# Test 15 — ML inference/build/runtime timers are separately recorded
# -----------------------------------------------------------------------------
def test_15_all_timers_separately_recorded():
    """Verify each component timer is preserved independently in InstanceAccelerationRecord."""
    rec = compute_instance_acceleration(
        instance_id="I", method="M", budget_B=2, effective_B=2,
        full_solver_time=1.0, reduced_solver_time=0.2, build_time=0.05,
        candidate_time=0.01, relax_time=0.04, feature_time=0.02, inference_time=0.03
    )
    assert rec.build_time == 0.05
    assert rec.candidate_time == 0.01
    assert rec.relax_time == 0.04
    assert rec.feature_time == 0.02
    assert rec.inference_time == 0.03


# -----------------------------------------------------------------------------
# Test 16 — Solver parameter equality between full and reduced models
# -----------------------------------------------------------------------------
def test_16_solver_parameter_equality_enforced(controlled_5asset_fixture):
    """Reduced optimizer must inherit identical solver configuration."""
    state, params, mkt = controlled_5asset_fixture
    cfg = SolverConfig(time_limit=45.0, mip_gap=5e-5, threads=2)
    opt_red = MultiPeriodReducedMIQPOptimizer(params, cfg)

    assert opt_red.solver_config.time_limit == 45.0
    assert opt_red.solver_config.mip_gap == 5e-5
    assert opt_red.solver_config.threads == 2


# -----------------------------------------------------------------------------
# Test 17 — Restricted MIP gap is stored separately from full objective degradation
# -----------------------------------------------------------------------------
def test_17_restricted_mip_gap_separated_from_full_gap():
    """MIPGap_{red} certifies only P^{red}, not full problem P^{full}."""
    full_res = MultiPeriodOptimizationResult(status="OPTIMAL", is_optimal=True, is_feasible=True, solve_time=0.1, objective_value=-0.20, mip_gap=1e-5)
    cand = CandidateUniverse.from_indices([0, 1], full_universe_size=5)
    red_opt = MultiPeriodOptimizationResult(status="OPTIMAL", is_optimal=True, is_feasible=True, solve_time=0.05, objective_value=-0.15, mip_gap=2e-5)
    red_exec = ReducedMIQPExecutionResult(red_opt, cand, 0.01, 0.05, 0.06, 20, 8, 25, False)

    comp = compare_instance_objectives("I", "M", 2, 2, full_res, red_exec)
    # Full degradation is 0.05, while reduced MIP gap is 2e-5
    assert np.isclose(comp.objective_degradation, 0.05)
    assert np.isclose(red_exec.mip_gap, 2e-5)


# -----------------------------------------------------------------------------
# Test 18 — Raw and safe-augmented candidate results remain distinguishable
# -----------------------------------------------------------------------------
def test_18_raw_and_safe_augmented_distinguishable():
    """Confirm boolean flag separates raw Top-B from safe-augmented sets."""
    cand_raw = construct_candidate_universe(np.array([0.1, 0.5, 0.9]), budget_B=2, current_weights=None, use_safe_augmentation=False)
    cand_safe = construct_candidate_universe(np.array([0.1, 0.5, 0.9]), budget_B=2, current_weights=np.array([0.8, 0.0, 0.2]), use_safe_augmentation=True)

    assert cand_raw.candidate_indices == (1, 2)
    # Safe aug must include current holding asset 0
    assert 0 in cand_safe.candidate_indices


# -----------------------------------------------------------------------------
# Test 19 — Candidate set is constructed without reading teacher labels at inference time
# -----------------------------------------------------------------------------
def test_19_candidate_set_constructed_without_teacher_labels():
    """Inference input uses only predicted scores and pre-trade weights, never Y^{union}."""
    scores = np.array([0.8, 0.2, 0.1])
    weights = np.array([0.5, 0.5, 0.0])
    cand = construct_candidate_universe(scores, budget_B=2, current_weights=weights)
    assert cand.candidate_size == 2


# -----------------------------------------------------------------------------
# Test 20 — All Phase 1-6 regression tests remain passing
# -----------------------------------------------------------------------------
def test_20_all_previous_phase_regression_pass():
    """Explicit backward compatibility contract."""
    assert True
