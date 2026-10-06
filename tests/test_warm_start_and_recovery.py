"""
Comprehensive Unit Tests for Phase 8:
  - Candidate-set Semantics Audit (Semantics B Fixed Total Budget)
  - Primal MIP Warm Starts (W0, W1, W2, W3) and Feasibility Repair
  - Full-Problem Continuous Relaxation Lower Bounds (LB)
  - Solution Lifting and Independent Constraint Verification
  - Mathematically Certified Upper Bounds (UB) and Certificates (Gap^{cert})
  - Nested Recovery Loop with Eventual Full-Universe Fallback
  - Separation of Fast Mode vs Certified Mode and Runtime Accounting
"""

from __future__ import annotations

import time
import pytest
import numpy as np

from learning_augmented_portfolio.dynamic.state import PortfolioState
from learning_augmented_portfolio.optimization.variables import (
    MultiPeriodMarketState,
    MultiPeriodPortfolioParameters,
    MultiPeriodOptimizationResult,
    SolverConfig,
)
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from learning_augmented_portfolio.optimization.reduced_miqp import MultiPeriodReducedMIQPOptimizer
from learning_augmented_portfolio.optimization.relaxation import MultiPeriodRelaxedQPOptimizer
from learning_augmented_portfolio.optimization.repair import ContinuousRepairOptimizer
from learning_augmented_portfolio.optimization.support_restriction import CandidateUniverse
from learning_augmented_portfolio.learning.candidate_set import construct_candidate_universe
from learning_augmented_portfolio.learning.warm_start import (
    WarmStartType,
    MIPStartData,
    build_previous_solution_warm_start,
    inject_warm_start_into_model,
)
from learning_augmented_portfolio.recovery.candidate_sets import generate_nested_candidate_sequence
from learning_augmented_portfolio.recovery.feasibility import FullProblemFeasibilityChecker
from learning_augmented_portfolio.recovery.certificate import (
    FullProblemCertificate,
    compute_full_problem_certificate,
)
from learning_augmented_portfolio.recovery.recovery_loop import (
    NestedRecoveryOptimizer,
    RecoveryLoopResult,
)
from learning_augmented_portfolio.dataset.instance import (
    OptimizationInstance,
    OptimizationInstanceMetadata,
)


def make_test_instance(instance_id: str, state: PortfolioState, mkt: MultiPeriodMarketState, params: MultiPeriodPortfolioParameters) -> OptimizationInstance:
    meta = OptimizationInstanceMetadata(
        instance_id=instance_id,
        calendar_date="2026-09-28",
        time_index=0,
        split="test",
        universe_id="TEST",
        n_assets=mkt.n_assets,
        horizon=params.horizon,
        dataset_version="1.0",
        config_hash="00000000",
        git_commit=None,
        gurobi_version="12.0",
        python_version="3.13",
        solver_params={},
        random_seed=42,
        solver_status="OPTIMAL",
        label_quality="SOLVER_CERTIFIED",
        is_trainable_label=True,
        objective=-0.20,
        best_bound=-0.20,
        mip_gap=0.0,
        runtime=0.1,
        work=0.01,
        node_count=1,
        sol_count=1,
        used_fallback=False,
    )
    return OptimizationInstance(metadata=meta, actual_state=state, params=params, market_state=mkt)



@pytest.fixture
def controlled_5asset_setup():
    """Create a 5-asset, 2-period controlled setup with known preference structure."""
    n = 5
    H = 2
    tickers = ["A0", "A1", "A2", "A3", "A4"]

    # Initial holdings: only A0 and A1 held
    w = np.array([0.45, 0.45, 0.0, 0.0, 0.0])
    c = 0.10
    state = PortfolioState(actual_wealth=1000.0, normalized_weights=w, normalized_cash=c, time_index=0, asset_names=tickers)

    # A4 has highest return (0.10), A0 (0.08), A1 (0.06), A2 (0.03), A3 (0.02)
    mu = np.array([
        [0.08, 0.06, 0.03, 0.02, 0.10],
        [0.08, 0.06, 0.03, 0.02, 0.10],
    ])
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


# =============================================================================
# CANDIDATE SEMANTICS AUDIT TESTS (Tests 1 - 4)
# =============================================================================

def test_1_candidate_semantics_audit_matches_definition(controlled_5asset_setup):
    """Verify Semantics B: C_t(B) = M_t cup Top_{B - |M_t|}(p_t | i not in M_t)."""
    state, params, mkt = controlled_5asset_setup
    # Held: {0, 1} (|M_t| = 2). Scores rank: 4 (0.9), 3 (0.8), 2 (0.7), 1 (0.2), 0 (0.1)
    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    cand = construct_candidate_universe(scores, budget_B=3, current_weights=state.normalized_weights, semantics="fixed_budget")

    # M_t={0, 1} takes 2 slots. 1 slot remains for top score not in M_t, which is index 4
    assert cand.contains(0)
    assert cand.contains(1)
    assert cand.contains(4)
    assert not cand.contains(3)
    assert cand.candidate_size == 3


def test_2_mandatory_constrained_top_b_has_exact_budget(controlled_5asset_setup):
    """Verify |C_t(B)| == B exactly under Semantics B."""
    state, params, mkt = controlled_5asset_setup
    scores = np.array([0.5, 0.1, 0.9, 0.2, 0.8])
    for B in [2, 3, 4, 5]:
        cand = construct_candidate_universe(scores, budget_B=B, current_weights=state.normalized_weights, semantics="fixed_budget")
        assert cand.candidate_size == B


def test_3_candidate_sets_are_strictly_nested_as_b_increases(controlled_5asset_setup):
    """Verify C_t(B_1) subset C_t(B_2) subset ... subset C_t(B_m)."""
    state, params, mkt = controlled_5asset_setup
    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    c_seq = generate_nested_candidate_sequence(scores, state.normalized_weights, budgets=[2, 3, 4, 5])
    assert len(c_seq) == 4
    for k in range(3):
        assert set(c_seq[k].candidate_indices).issubset(set(c_seq[k + 1].candidate_indices))
        assert c_seq[k].candidate_size < c_seq[k + 1].candidate_size


def test_4_budget_smaller_than_mandatory_set_is_rejected(controlled_5asset_setup):
    """Verify B < |M_t| raises ValueError under Semantics B."""
    state, params, mkt = controlled_5asset_setup
    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    # |M_t| = 2 (assets 0 and 1). Budget B=1 is strictly less than 2
    with pytest.raises(ValueError, match="strictly less than mandatory set size"):
        construct_candidate_universe(scores, budget_B=1, current_weights=state.normalized_weights, semantics="fixed_budget")


# =============================================================================
# WARM START TESTS (Tests 5 - 10)
# =============================================================================

def test_5_previous_solution_horizon_shift_is_correct(controlled_5asset_setup):
    """Verify W1 shifts horizon forward and completes final period."""
    state, params, mkt = controlled_5asset_setup
    # Mock previous solution for H=2: period 0 has [0, 1], period 1 has [0, 4]
    mock_y = np.array([[1, 1, 0, 0, 0], [1, 0, 0, 0, 1]])
    mock_z = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 1]])
    mock_res = MultiPeriodOptimizationResult(
        status="OPTIMAL", is_optimal=True, is_feasible=True, solve_time=0.1,
        holding_indicators=mock_y, trade_indicators=mock_z,
        post_trade_holdings=mock_y.astype(float) * 0.4,
        post_trade_cash=np.array([0.1, 0.1]),
    )
    w1 = build_previous_solution_warm_start(mock_res, horizon=2, n_assets=5)
    assert w1.start_type == WarmStartType.PREVIOUS_SOLUTION
    assert w1.y_start.shape == (2, 5)
    # Period 0 of new warm start takes Period 1 of old solution: [1, 0, 0, 0, 1]
    assert np.array_equal(w1.y_start[0], mock_y[1])
    # Period 1 repeats last holding
    assert np.array_equal(w1.y_start[1], mock_y[1])
    assert w1.is_complete is True


def test_6_ml_discrete_start_dimensions_and_binary():
    """Verify ML discrete start container maintains valid dimensions and binary values."""
    start = MIPStartData(
        start_type=WarmStartType.ML_DISCRETE,
        y_start=np.array([[1, 0, 0], [0, 1, 0]]),
        z_start=np.array([[1, 0, 0], [0, 1, 0]]),
        construction_time=0.002,
    )
    assert start.y_start.shape == (2, 3)
    assert np.all(np.isin(start.y_start, [0, 1]))


def test_7_continuous_repair_returns_feasible_start(controlled_5asset_setup):
    """Verify W3 continuous QP repair returns certified feasible primal solution."""
    state, params, mkt = controlled_5asset_setup
    repairer = ContinuousRepairOptimizer(params)

    # Propose binary holding for assets 0 and 1 (currently held, no trade needed)
    prop_y = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]])
    prop_z = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

    res = repairer.repair(mkt, prop_y, prop_z)
    assert res.is_feasible is True
    assert res.repaired_solution is not None
    assert res.repaired_solution.is_feasible is True
    assert res.objective_value is not None


def test_8_infeasible_proposed_starts_are_safely_detected(controlled_5asset_setup):
    """Verify that contradictory or impossible binary patterns report is_feasible=False."""
    state, params, mkt = controlled_5asset_setup

    # Impossible min cash reserve 2.0 while initial wealth is 1.0
    params_infeasible = MultiPeriodPortfolioParameters(
        risk_aversion=2.0, horizon=2,
        risk_free_rates=params.risk_free_rates,
        max_turnovers=params.max_turnovers,
        min_cash=np.full(2, 2.0),
        weight_upper_limits=params.weight_upper_limits,
        liquidity_limits=params.liquidity_limits,
        prop_cost_coeffs=params.prop_cost_coeffs,
        cardinality_limits=np.full(2, 2),
    )
    repairer = ContinuousRepairOptimizer(params_infeasible)
    prop_y = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]])
    prop_z = np.zeros((2, 5), dtype=int)
    res = repairer.repair(mkt, prop_y, prop_z)
    assert res.is_feasible is False
    assert res.repaired_solution is None


def test_9_warm_start_does_not_modify_feasible_set_or_objective(controlled_5asset_setup):
    """Verify that injecting a warm start produces the EXACT same objective as cold solve."""
    state, params, mkt = controlled_5asset_setup
    cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)
    opt = MultiPeriodMIQPOptimizer(params, cfg)

    # 1. Cold solve
    res_cold = opt.solve(mkt)

    # 2. Warm solve with incumbent from cold solve
    warm = MIPStartData(
        start_type=WarmStartType.ML_REPAIRED,
        y_start=res_cold.holding_indicators,
        z_start=res_cold.trade_indicators,
        x_start=res_cold.post_trade_holdings,
        b_start=res_cold.post_trade_cash,
        is_complete=True,
    )
    res_warm = opt.solve(mkt, warm_start=warm)

    assert abs(res_cold.objective_value - res_warm.objective_value) < 1e-6


def test_10_start_timing_metrics_are_correctly_recorded():
    """Verify construction and repair time attributes are persisted."""
    start = MIPStartData(
        start_type=WarmStartType.ML_REPAIRED,
        y_start=np.zeros((2, 2), dtype=int),
        construction_time=0.005,
        repair_time=0.012,
    )
    assert np.isclose(start.construction_time, 0.005)
    assert np.isclose(start.repair_time, 0.012)


# =============================================================================
# CERTIFICATION TESTS (Tests 11 - 16)
# =============================================================================

def test_11_full_continuous_relaxation_satisfies_lb_le_j_star(controlled_5asset_setup):
    """Verify LB_t = J^{relax} <= J* (Theorem 1 Lower Bound)."""
    state, params, mkt = controlled_5asset_setup
    cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)

    opt_miqp = MultiPeriodMIQPOptimizer(params, cfg)
    res_miqp = opt_miqp.solve(mkt)
    j_star = res_miqp.objective_value

    opt_relax = MultiPeriodRelaxedQPOptimizer(params, cfg)
    res_relax = opt_relax.solve(mkt)
    lb = res_relax.objective_value

    assert lb <= j_star + 1e-6, f"Relaxation bound violated: LB={lb:.6f} > J*={j_star:.6f}"


def test_12_lifted_solution_checked_against_original_constraints(controlled_5asset_setup):
    """Verify FullProblemFeasibilityChecker validates all physical constraints."""
    state, params, mkt = controlled_5asset_setup
    cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)
    opt_full = MultiPeriodMIQPOptimizer(params, cfg)
    res_full = opt_full.solve(mkt)

    checker = FullProblemFeasibilityChecker(params, mkt)
    audit = checker.audit_solution(res_full)
    assert audit.is_feasible is True
    assert audit.max_constraint_violation < 1e-4
    assert audit.objective_value is not None
    assert abs(audit.objective_value - res_full.objective_value) < 1e-6


def test_13_only_full_feasible_solutions_define_ub(controlled_5asset_setup):
    """Verify that an infeasible lifted solution is rejected and yields UB = None."""
    state, params, mkt = controlled_5asset_setup
    checker = FullProblemFeasibilityChecker(params, mkt)

    # Infeasible dummy result: violates budget
    infeasible_res = MultiPeriodOptimizationResult(
        status="OPTIMAL", is_optimal=True, is_feasible=True, solve_time=0.1,
        post_trade_holdings=np.ones((2, 5)),  # sum(x) = 5.0 >> 1.0!
        pre_trade_holdings=np.zeros((2, 5)),
        trades=np.ones((2, 5)),
        absolute_trades=np.ones((2, 5)),
        pre_trade_cash=np.zeros(2),
        post_trade_cash=np.zeros(2),
        transaction_costs=np.zeros(2),
        holding_indicators=np.ones((2, 5), dtype=int),
        trade_indicators=np.ones((2, 5), dtype=int),
    )
    audit = checker.audit_solution(infeasible_res)
    assert audit.is_feasible is False
    assert len(audit.violations) > 0

    cert = compute_full_problem_certificate("TEST_INF", lower_bound=-0.2, upper_bound=audit.objective_value, reduced_mip_gap=0.01, candidate_budget=3, candidate_size=3)
    assert cert.is_certified is False
    assert cert.upper_bound is None


def test_14_certificate_satisfies_lb_le_j_star_le_ub(controlled_5asset_setup):
    """Verify LB <= J* <= UB and non-negativity of certificate gap."""
    state, params, mkt = controlled_5asset_setup
    cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)

    opt_miqp = MultiPeriodMIQPOptimizer(params, cfg)
    j_star = opt_miqp.solve(mkt).objective_value

    opt_relax = MultiPeriodRelaxedQPOptimizer(params, cfg)
    lb = opt_relax.solve(mkt).objective_value

    # Suboptimal candidate: C = {0, 1, 2} (missing alpha asset 4)
    cand_sub = CandidateUniverse.from_indices([0, 1, 2], full_universe_size=5)
    opt_red = MultiPeriodReducedMIQPOptimizer(params, cfg)
    red_exec = opt_red.solve(mkt, cand_sub, expand_solution=True)
    ub = red_exec.optimization_result.objective_value

    # With realistic epsilon_cert = 0.05, gap <= epsilon_cert -> relaxation_certified True
    cert_wide = compute_full_problem_certificate("TEST_BOUNDS", lower_bound=lb, upper_bound=ub, reduced_mip_gap=1e-4, candidate_budget=3, candidate_size=3, epsilon_cert=0.05)
    assert cert_wide.relaxation_certified is True
    assert cert_wide.is_certified is True
    assert lb <= j_star + 1e-6
    assert j_star <= ub + 1e-6
    assert cert_wide.relaxation_certificate_gap >= 0.0

    # With strict epsilon_cert = 1e-4, gap > epsilon_cert -> relaxation_certified False (correct semantics!)
    cert_strict = compute_full_problem_certificate("TEST_BOUNDS", lower_bound=lb, upper_bound=ub, reduced_mip_gap=1e-4, candidate_budget=3, candidate_size=3, epsilon_cert=1e-4)
    assert cert_strict.relaxation_certified is False
    assert cert_strict.is_certified is False


def test_15_certificate_gap_upper_bounds_true_degradation(controlled_5asset_setup):
    """Verify J(x_hat) - J* <= UB - LB."""
    state, params, mkt = controlled_5asset_setup
    cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)

    j_star = MultiPeriodMIQPOptimizer(params, cfg).solve(mkt).objective_value
    lb = MultiPeriodRelaxedQPOptimizer(params, cfg).solve(mkt).objective_value

    cand_sub = CandidateUniverse.from_indices([0, 1, 2], full_universe_size=5)
    ub = MultiPeriodReducedMIQPOptimizer(params, cfg).solve(mkt, cand_sub).objective_value

    true_degradation = ub - j_star
    cert_gap = ub - lb
    assert true_degradation <= cert_gap + 1e-6


def test_16_reduced_mip_gap_separated_from_certificate_gap():
    """Verify reduced MIP gap is explicitly preserved as a separate field."""
    cert = compute_full_problem_certificate(
        instance_id="SEPARATION_TEST",
        lower_bound=-0.30,
        upper_bound=-0.25,
        reduced_mip_gap=0.0001,
        candidate_budget=3,
        candidate_size=3,
    )
    assert cert.reduced_mip_gap == 0.0001
    assert np.isclose(cert.absolute_gap, 0.05)


def test_16b_relaxation_vs_full_mip_certificate_semantics():
    """
    Mandatory Pre-Phase-9 Regression Test:
    Verify strict distinction between Relaxation-Certified and Full-MIP Solver Certified.
    """
    from learning_augmented_portfolio.recovery.certificate import CertificateSource

    # Case 1: UB - LB^{relax} = 0.003177 > epsilon_{cert} = 1e-4 -> NOT relaxation certified!
    cert_fail = compute_full_problem_certificate(
        instance_id="FAIL_RELAX_TEST",
        lower_bound=-0.253177,
        upper_bound=-0.250000,
        reduced_mip_gap=1e-5,
        candidate_budget=3,
        candidate_size=3,
        epsilon_cert=1e-4,
        is_full_universe=False,
    )
    assert cert_fail.relaxation_certified is False
    assert cert_fail.is_certified is False
    assert cert_fail.full_mip_certified is False
    assert cert_fail.any_valid_certificate is False
    assert cert_fail.certificate_source == CertificateSource.NONE

    # Case 2: Final full universe B=N solve with solver B&B bound -> FULL_MIP_BOUND certified!
    cert_mip = compute_full_problem_certificate(
        instance_id="MIP_BOUND_TEST",
        lower_bound=-0.253177,       # Relaxation gap is still 0.003177 > 1e-4
        upper_bound=-0.250000,
        reduced_mip_gap=1e-5,
        candidate_budget=10,
        candidate_size=10,
        epsilon_cert=1e-4,
        full_mip_bound=-0.250002,    # MIP bound gap is 2e-6 <= 1e-4!
        full_mip_gap=1e-5,
        epsilon_mip=1e-4,
        is_full_universe=True,
    )
    assert cert_mip.relaxation_certified is False
    assert cert_mip.full_mip_certified is True
    assert cert_mip.any_valid_certificate is True
    assert cert_mip.certificate_source == CertificateSource.FULL_MIP_BOUND
    assert cert_mip.lower_bound_best == -0.250002

    # Case 3: Tight relaxation bound -> FULL_RELAXATION certified!
    cert_relax = compute_full_problem_certificate(
        instance_id="RELAX_SUCCESS_TEST",
        lower_bound=-0.250030,
        upper_bound=-0.250000,       # Gap is 3e-5 <= 1e-4
        reduced_mip_gap=1e-5,
        candidate_budget=5,
        candidate_size=5,
        epsilon_cert=1e-4,
        is_full_universe=False,
    )
    assert cert_relax.relaxation_certified is True
    assert cert_relax.full_mip_certified is False
    assert cert_relax.any_valid_certificate is True
    assert cert_relax.certificate_source == CertificateSource.FULL_RELAXATION


def test_16c_warm_start_reporting_semantics():
    """
    Mandatory Pre-Phase-9 Regression Test:
    Verify WarmStartEvaluationRecord correctly represents N/A feasibility for NoStart
    and distinguishes partial vs complete feasible starts.
    """
    from learning_augmented_portfolio.evaluation.warm_start_metrics import (
        WarmStartEvaluationRecord,
        summarize_warm_start_evaluations,
    )

    rec_w0 = WarmStartEvaluationRecord(
        instance_id="inst_0",
        start_type="W0_NoStart",
        construction_time=0.0,
        repair_time=0.0,
        partial_start_provided=False,
        complete_start_feasible=None,   # N/A!
        repair_success_rate=None,
        first_feasible_time=0.005,
        delta_t_first=0.0,
        first_incumbent_obj=-0.20,
        final_runtime=0.1,
        work=0.01,
        node_count=10,
        final_objective=-0.25,
        best_bound=-0.25,
        mip_gap=0.0,
    )
    assert rec_w0.complete_start_feasible is None
    assert rec_w0.partial_start_provided is False

    summary = summarize_warm_start_evaluations([rec_w0])
    assert np.isnan(summary["CompleteFeasibleRate"].iloc[0])
    assert summary["PartialStartProvidedRate"].iloc[0] == 0.0


# =============================================================================
# RECOVERY TESTS (Tests 17 - 22)
# =============================================================================

def test_17_candidate_expansion_occurs_after_certificate_failure(controlled_5asset_setup):
    """Verify recovery enlarges candidate set when certificate gap exceeds tolerance."""
    state, params, mkt = controlled_5asset_setup
    cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)

    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    inst = make_test_instance("EXP_TEST", state, mkt, params)

    # Tight tolerance forces recovery from B=2 to B=5
    recovery_opt = NestedRecoveryOptimizer(params, cfg, epsilon_cert=1e-4)
    res = recovery_opt.solve(inst, scores=scores, budgets=[2, 3, 5], fast_mode=False)

    assert res.stages_executed > 1
    assert res.final_budget > 2


def test_18_recovery_objective_is_monotone_non_increasing(controlled_5asset_setup):
    """Verify J*(C^(1)) >= J*(C^(2)) >= ... >= J*(C^(m)) across recovery stages."""
    state, params, mkt = controlled_5asset_setup
    cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)
    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])

    inst = make_test_instance("MONOTONE_TEST", state, mkt, params)

    recovery_opt = NestedRecoveryOptimizer(params, cfg, epsilon_cert=1e-6)
    res = recovery_opt.solve(inst, scores=scores, budgets=[2, 3, 5], fast_mode=False)

    objectives = [st.objective_value for st in res.stage_telemetry if st.objective_value is not None]
    for k in range(len(objectives) - 1):
        # In minimization, larger candidate set has equal or better (lower) objective
        assert objectives[k] >= objectives[k + 1] - 1e-6


def test_19_full_universe_final_stage_reproduces_full_problem(controlled_5asset_setup):
    """Verify that final recovery stage with B=N reproduces full MIQP exactly."""
    state, params, mkt = controlled_5asset_setup
    cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)
    j_full = MultiPeriodMIQPOptimizer(params, cfg).solve(mkt).objective_value

    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    inst = make_test_instance("FULL_REPRO_TEST", state, mkt, params)

    # Force full recovery by asking for impossible epsilon
    recovery_opt = NestedRecoveryOptimizer(params, cfg, epsilon_cert=1e-9)
    res = recovery_opt.solve(inst, scores=scores, budgets=[2, 5], fast_mode=False)

    assert res.full_fallback_occurred is True
    assert abs(res.final_objective - j_full) < 1e-6


def test_20_infeasible_early_stage_expands_rather_than_crashes(controlled_5asset_setup):
    """Verify that if an early stage is infeasible, recovery advances to next stage gracefully."""
    state, params, mkt = controlled_5asset_setup
    cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)

    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    inst = make_test_instance("INF_RECOVERY_TEST", state, mkt, params)

    recovery_opt = NestedRecoveryOptimizer(params, cfg, epsilon_cert=1e-4)
    # Stages execute without crashing
    res = recovery_opt.solve(inst, scores=scores, budgets=[2, 3, 5], fast_mode=False)
    assert res.final_solution is not None


def test_21_previous_stage_solution_can_initialize_next_stage(controlled_5asset_setup):
    """Verify inter-stage warm starting injects stage j incumbent into stage j+1."""
    state, params, mkt = controlled_5asset_setup
    cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)

    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    inst = make_test_instance("INTERSTAGE_WS_TEST", state, mkt, params)

    recovery_opt = NestedRecoveryOptimizer(params, cfg, epsilon_cert=1e-6, use_recovery_warm_starts=True)
    res = recovery_opt.solve(inst, scores=scores, budgets=[2, 3, 5], fast_mode=False)
    assert res.stages_executed >= 2


def test_22_stopping_occurs_exactly_when_tolerance_is_met(controlled_5asset_setup):
    """Verify recovery terminates immediately when certificate gap <= epsilon."""
    state, params, mkt = controlled_5asset_setup
    cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)

    # If epsilon is large (e.g. 1.0), stage 0 should satisfy it immediately
    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    inst = make_test_instance("EARLY_STOP_TEST", state, mkt, params)

    recovery_opt = NestedRecoveryOptimizer(params, cfg, epsilon_cert=1.0)
    res = recovery_opt.solve(inst, scores=scores, budgets=[2, 3, 5], fast_mode=False)
    assert res.stages_executed == 1
    assert res.full_fallback_occurred is False


# =============================================================================
# RUNTIME ACCOUNTING & OPERATING MODES (Tests 23 - 26)
# =============================================================================

def test_23_relaxation_cost_is_reused_without_double_counting(controlled_5asset_setup):
    """Verify passing precomputed_lower_bound skips relaxation solver call."""
    state, params, mkt = controlled_5asset_setup
    cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)

    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    inst = make_test_instance("REUSE_TEST", state, mkt, params)

    recovery_opt = NestedRecoveryOptimizer(params, cfg, epsilon_cert=1.0)
    res = recovery_opt.solve(inst, scores=scores, budgets=[3, 5], precomputed_lower_bound=-0.35, precomputed_relaxation_time=0.015)
    assert res.lower_bound == -0.35
    assert res.relaxation_time == 0.015


def test_24_fast_mode_excludes_certificate_overhead(controlled_5asset_setup):
    """Verify Fast Mode runs single stage with zero certificate overhead."""
    state, params, mkt = controlled_5asset_setup
    cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)

    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    inst = make_test_instance("FAST_MODE_TEST", state, mkt, params)

    recovery_opt = NestedRecoveryOptimizer(params, cfg)
    res = recovery_opt.solve(inst, scores=scores, budgets=[3, 5], fast_mode=True)
    assert res.fast_mode is True
    assert res.stages_executed == 1
    assert res.is_certified is False
    assert res.lower_bound is None


def test_25_certified_mode_includes_certificate_and_verifies_bounds(controlled_5asset_setup):
    """Verify Certified Mode produces valid lower and upper bounds."""
    state, params, mkt = controlled_5asset_setup
    cfg = SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1)

    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    inst = make_test_instance("CERT_MODE_TEST", state, mkt, params)

    recovery_opt = NestedRecoveryOptimizer(params, cfg, epsilon_cert=1e-4)
    res = recovery_opt.solve(inst, scores=scores, budgets=[3, 5], fast_mode=False)
    assert res.fast_mode is False
    assert res.lower_bound is not None
    assert res.upper_bound is not None
    assert res.final_certificate_gap is not None
    assert res.final_certificate_gap >= 0.0


def test_26_all_phase_1_to_7_regression_pass():
    """Smoke test ensuring importability and consistency of all prior phase modules."""
    from learning_augmented_portfolio.optimization.single_period_qp import SinglePeriodQPOptimizer
    from learning_augmented_portfolio.optimization.single_period_miqp import SinglePeriodMIQPOptimizer
    from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
    from learning_augmented_portfolio.dynamic.simulator import RollingHorizonSimulator
    from learning_augmented_portfolio.dataset.generator import HistoricalDatasetGenerator
    from learning_augmented_portfolio.learning.active_set import build_logistic_regression_predictor
    from learning_augmented_portfolio.optimization.reduced_miqp import MultiPeriodReducedMIQPOptimizer

    assert SinglePeriodQPOptimizer is not None
    assert SinglePeriodMIQPOptimizer is not None
    assert MultiPeriodMIQPOptimizer is not None
    assert RollingHorizonSimulator is not None
    assert HistoricalDatasetGenerator is not None
    assert build_logistic_regression_predictor is not None
    assert MultiPeriodReducedMIQPOptimizer is not None
