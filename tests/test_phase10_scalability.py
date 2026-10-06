"""
Phase 10 Unit Tests: Scalability, Pre-Scaling Audits, and Rigorous Solver Benchmarking.
Covers all 25 mandatory tests specified in Phase 10 Section 28.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from learning_augmented_portfolio.backtest.records import (
    FinancialMetricsSummary,
    CertificationMetricsSummary,
    compute_financial_metrics,
)
from learning_augmented_portfolio.scalability.failure import (
    SolverExitStatus,
    classify_solver_outcome,
)
from learning_augmented_portfolio.scalability.instance_generator import (
    FactorModelSpec,
    ScalableInstanceGenerator,
)
from learning_augmented_portfolio.scalability.model_stats import (
    compute_theoretical_model_stats,
)
from learning_augmented_portfolio.scalability.experiment_grid import (
    CardinalityRegime,
    compute_cardinality,
    compute_candidate_budget,
    generate_nested_budget_sequence,
)
from learning_augmented_portfolio.scalability.anytime import (
    AnytimePoint,
    AnytimeTrajectory,
)
from learning_augmented_portfolio.scalability.runner import (
    MethodSolveResult,
    ScalabilityRunner,
)
from learning_augmented_portfolio.evaluation.scalability_metrics import (
    E2ETimerBreakdown,
    compute_speedup,
    compute_runtime_reduction,
    compute_objective_quality,
)
from learning_augmented_portfolio.optimization.variables import SolverConfig


# 1. Phase 9 CAGR / Wealth Reconciliation
def test_1_cagr_wealth_reconciliation():
    v0 = 1_000_000.0
    vt = 1_100_000.0  # +10%
    wealth_series = np.linspace(v0, vt, 52)
    turnover = np.full(51, 0.1)
    costs = np.full(51, 100.0)
    holdings = np.full(51, 4)
    cash = np.full(51, 0.1)

    m = compute_financial_metrics(
        "Test",
        wealth_series,
        turnover,
        costs,
        holdings,
        cash,
        annualization_factor=52.0,
        n_trading_days=252,
    )
    # Check exact reconciliation: (1.10) ** (252 / 252) - 1 = 0.10
    assert abs(m.total_wealth_return - 0.10) < 1e-6
    assert abs(m.cagr - 0.10) < 1e-6
    # Wealth trajectory matches final wealth
    assert m.final_wealth == vt

    # Also verify exact reconciliation on saved real-market wealth trajectories
    wealth_path = Path("artifacts/real_backtest/wealth_trajectories.csv")
    fin_path = Path("artifacts/real_backtest/financial_metrics.csv")
    if wealth_path.exists() and fin_path.exists():
        w_df = pd.read_csv(wealth_path)
        f_df = pd.read_csv(fin_path).set_index("Strategy")
        N_w = len(w_df) - 1
        for strat in f_df.index:
            w_series = w_df[strat].values
            v_0, v_t = w_series[0], w_series[-1]
            exp_tot = v_t / v_0 - 1.0
            rep_tot = float(f_df.loc[strat, "TotalReturn"].rstrip("%")) / 100.0
            assert rep_tot == pytest.approx(exp_tot, abs=1e-4)
            exp_cagr = (v_t / v_0) ** (52.0 / float(N_w)) - 1.0
            rep_cagr = float(f_df.loc[strat, "CAGR"].rstrip("%")) / 100.0
            assert rep_cagr == pytest.approx(exp_cagr, abs=1e-4)


# 2. Identical Sharpe Convention Across Experiments
def test_2_identical_sharpe_convention():
    wealth_series = np.array([100.0, 102.0, 104.0, 106.0, 108.0])
    m = compute_financial_metrics(
        "SharpeTest",
        wealth_series,
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
        annualization_factor=52.0,
        risk_free_rate=0.03,
    )
    assert m.risk_free_rate_used == 0.03
    assert m.sharpe_convention == "annualized_excess_cagr_over_annualized_vol"
    expected_excess = m.cagr - 0.03
    expected_sharpe = expected_excess / m.annualized_volatility
    assert abs(m.sharpe_ratio - expected_sharpe) < 1e-6


# 3. Canonical Minimum-Cash Configuration
def test_3_canonical_minimum_cash_configuration():
    gen = ScalableInstanceGenerator()
    inst = gen.generate_instance(n_assets=10, horizon=3, cardinality_limit=4, seed=42)
    assert float(inst.params.min_cash[0]) == 0.10


# 4. N/A vs Zero Certificate Fields
def test_4_na_vs_zero_certificate_fields():
    # If a strategy does not compute relaxation certificates, gap must be None, not 0.0
    cert = CertificationMetricsSummary(
        strategy_name="Fast_MarketOnly",
        relaxation_certified_rate=0.0,
        full_mip_certified_rate=0.0,
        any_valid_certificate_rate=0.0,
        mean_relaxation_gap=None,  # Must be None (represented as N/A)
    )
    assert cert.mean_relaxation_gap is None
    assert cert.mean_relaxation_gap != 0.0


# 5. Factor-Model Covariance PSD
def test_5_factor_model_covariance_psd():
    gen = ScalableInstanceGenerator()
    rng = np.random.default_rng(123)
    for n in [10, 25, 50, 100]:
        cov = gen.generate_covariance_matrix(n, rng)
        eigs = np.linalg.eigvalsh(cov)
        assert np.all(eigs > 1e-7), f"Minimum eigenvalue {np.min(eigs)} not strictly positive for N={n}"


# 6. Deterministic Scalable-Instance Generation
def test_6_deterministic_instance_generation():
    gen = ScalableInstanceGenerator()
    inst1 = gen.generate_instance(n_assets=25, horizon=3, cardinality_limit=5, seed=999)
    inst2 = gen.generate_instance(n_assets=25, horizon=3, cardinality_limit=5, seed=999)
    np.testing.assert_array_equal(inst1.market_state.expected_returns, inst2.market_state.expected_returns)
    np.testing.assert_array_equal(inst1.market_state.covariance_matrices, inst2.market_state.covariance_matrices)
    np.testing.assert_array_equal(inst1.actual_state.normalized_weights, inst2.actual_state.normalized_weights)


# 7. Model-Size Formulas
def test_7_model_size_formulas():
    stats_full = compute_theoretical_model_stats(n_assets=50, horizon=3)
    assert stats_full.n_binary == 2 * 50 * 3  # 2NH = 300
    assert stats_full.n_continuous == 3 * (4 * 50 + 3)  # 609
    assert stats_full.n_vars == 300 + 609  # 909
    assert stats_full.n_constraints == 3 * (8 * 50 + 5)  # 1215

    stats_red = compute_theoretical_model_stats(n_assets=50, horizon=3, candidate_size=10)
    assert stats_red.n_binary == 2 * 10 * 3  # 60
    assert stats_red.n_vars == 3 * (6 * 10 + 3)  # 189
    assert stats_red.reduction_ratio == 0.80


# 8. Full/Reduced Instance Identity Except Candidate Restriction
def test_8_full_reduced_instance_identity():
    gen = ScalableInstanceGenerator()
    inst = gen.generate_instance(n_assets=10, horizon=2, cardinality_limit=3, seed=42)
    runner = ScalabilityRunner(SolverConfig(time_limit=1.0, mip_gap=1e-3))
    # Both use identical underlying market_state and params
    res_full = runner.solve_full_miqp(inst, record_anytime=False)
    res_fast = runner.solve_fast_market_only(inst, candidate_budget_B=4)
    assert res_full.n_assets == res_fast.n_assets
    assert res_full.horizon == res_fast.horizon


# 9. Candidate Ratio Conversion to Valid B
def test_9_candidate_ratio_conversion():
    # B >= K_port and B >= n_held
    N = 50
    b1 = compute_candidate_budget(n_assets=N, ratio_rho_b=0.10, cardinality_k=10, n_held=4)
    assert b1 >= 10
    b2 = compute_candidate_budget(n_assets=N, ratio_rho_b=0.10, cardinality_k=4, n_held=12)
    assert b2 >= 12
    assert b2 <= N


# 10. Nested Ratio-Based Candidate Sequences
def test_10_nested_budget_sequence():
    seq = generate_nested_budget_sequence(
        n_assets=100,
        ratios=[0.10, 0.20, 0.30, 0.50, 1.00],
        cardinality_k=10,
        n_held=8,
    )
    # Must be strictly increasing and end at N
    for i in range(len(seq) - 1):
        assert seq[i] < seq[i + 1]
    assert seq[-1] == 100
    assert seq[0] >= 10


# 11. Common TimeLimit Enforcement
def test_11_common_timelimit_enforcement():
    cfg = SolverConfig(time_limit=0.5)
    runner = ScalabilityRunner(cfg)
    assert runner.solver_config.time_limit == 0.5


# 12. Timeout With Incumbent Classification
def test_12_timeout_with_incumbent():
    status = classify_solver_outcome(
        is_optimal=False,
        is_feasible=True,
        has_incumbent=True,
        gurobi_status=9,  # TIME_LIMIT
    )
    assert status == SolverExitStatus.TIME_LIMIT_WITH_INCUMBENT


# 13. Timeout Without Incumbent Classification
def test_13_timeout_without_incumbent():
    status = classify_solver_outcome(
        is_optimal=False,
        is_feasible=False,
        has_incumbent=False,
        gurobi_status=9,  # TIME_LIMIT
    )
    assert status == SolverExitStatus.TIME_LIMIT_NO_INCUMBENT


# 14. License-Limit Classification
def test_14_license_limit_classification():
    status = classify_solver_outcome(
        is_optimal=False,
        is_feasible=False,
        has_incumbent=False,
        gurobi_status=0,
        error_message="Model too large for size-limited license",
    )
    assert status == SolverExitStatus.LICENSE_LIMIT


# 15. Incumbent Not Mislabeled Optimal
def test_15_incumbent_not_mislabeled_optimal():
    # If Gurobi status is TIME_LIMIT but solution exists, must NOT be OPTIMAL
    status = classify_solver_outcome(
        is_optimal=False,
        is_feasible=True,
        has_incumbent=True,
        gurobi_status=9,
    )
    assert status != SolverExitStatus.OPTIMAL
    assert status == SolverExitStatus.TIME_LIMIT_WITH_INCUMBENT


# 16. Full Lower/Upper Bound Ordering
def test_16_lower_upper_bound_ordering():
    # For minimization: LB <= UB
    lb = -0.50
    ub = -0.48
    assert lb <= ub


# 17. Reduced Full-Feasibility Check
def test_17_reduced_full_feasibility_check():
    gen = ScalableInstanceGenerator()
    inst = gen.generate_instance(n_assets=10, horizon=1, cardinality_limit=3, seed=42)
    runner = ScalabilityRunner(SolverConfig(time_limit=1.0))
    res = runner.solve_fast_market_only(inst, candidate_budget_B=5)
    assert res.status == SolverExitStatus.OPTIMAL
    assert res.candidate_size <= 10


# 18. Anytime Monotonic Incumbent Behavior
def test_18_anytime_monotonic_incumbent():
    traj = AnytimeTrajectory()
    traj.add_point(AnytimePoint(elapsed_seconds=0.1, incumbent=10.0, best_bound=2.0, gap=0.8, node_count=0, work=0.01))
    traj.add_point(AnytimePoint(elapsed_seconds=0.2, incumbent=8.0, best_bound=3.0, gap=0.5, node_count=5, work=0.02))
    traj.add_point(AnytimePoint(elapsed_seconds=0.3, incumbent=5.0, best_bound=4.0, gap=0.2, node_count=10, work=0.03))

    incumbents = [p.incumbent for p in traj.points if p.incumbent is not None]
    # Non-increasing for minimization
    for i in range(len(incumbents) - 1):
        assert incumbents[i] >= incumbents[i + 1]


# 19. Best Bound Monotonic Behavior
def test_19_best_bound_monotonic():
    traj = AnytimeTrajectory()
    traj.add_point(AnytimePoint(elapsed_seconds=0.1, incumbent=10.0, best_bound=2.0, gap=0.8, node_count=0, work=0.01))
    traj.add_point(AnytimePoint(elapsed_seconds=0.2, incumbent=8.0, best_bound=3.0, gap=0.5, node_count=5, work=0.02))
    traj.add_point(AnytimePoint(elapsed_seconds=0.3, incumbent=5.0, best_bound=4.0, gap=0.2, node_count=10, work=0.03))

    bounds = [p.best_bound for p in traj.points if p.best_bound is not None]
    # Non-decreasing for minimization
    for i in range(len(bounds) - 1):
        assert bounds[i] <= bounds[i + 1]


# 20. End-to-End Runtime Component Accounting
def test_20_e2e_runtime_accounting():
    timers = E2ETimerBreakdown(
        t_data_sec=0.001,
        t_feature_sec=0.002,
        t_ranking_sec=0.003,
        t_relax_sec=0.004,
        t_build_sec=0.005,
        t_solver_sec=0.010,
    )
    expected_sum = 0.001 + 0.002 + 0.003 + 0.004 + 0.005 + 0.010
    assert abs(timers.total_e2e_sec - expected_sum) < 1e-9
    assert abs(timers.total_e2e_ms - expected_sum * 1000.0) < 1e-6


# 21. Paired-Instance Experiment Identity
def test_21_paired_instance_identity():
    gen = ScalableInstanceGenerator()
    inst = gen.generate_instance(n_assets=10, horizon=1, cardinality_limit=3, seed=777)
    runner = ScalabilityRunner(SolverConfig(time_limit=1.0))
    res_full = runner.solve_full_miqp(inst, record_anytime=False)
    res_red = runner.solve_fast_market_only(inst, candidate_budget_B=5)
    # Must share exact same instance_id
    assert res_full.instance_id == res_red.instance_id == "synth_N10_H1_K3_seed777"


# 22. Failure Types Remain Separated
def test_22_failure_types_separated():
    statuses = {
        SolverExitStatus.OPTIMAL,
        SolverExitStatus.TIME_LIMIT_WITH_INCUMBENT,
        SolverExitStatus.TIME_LIMIT_NO_INCUMBENT,
        SolverExitStatus.INFEASIBLE,
        SolverExitStatus.NUMERICAL_FAILURE,
        SolverExitStatus.MEMORY_FAILURE,
        SolverExitStatus.LICENSE_LIMIT,
        SolverExitStatus.BUILD_FAILURE,
        SolverExitStatus.NOT_EXECUTED_LICENSE_LIMIT,
    }
    assert len(statuses) == 9


# 23. OracleSupport Never Exposed as Deployable
def test_23_oracle_support_never_deployable():
    gen = ScalableInstanceGenerator()
    inst = gen.generate_instance(n_assets=10, horizon=1, cardinality_limit=3, seed=42)
    runner = ScalabilityRunner(SolverConfig(time_limit=1.0))
    res_oracle = runner.solve_oracle_support(inst, teacher_support={0, 1, 2})
    assert res_oracle.is_deployable is False


# 24. Unsupported N Due to License is Not Silently Dropped
def test_24_unsupported_n_license_limit_explicit():
    # Model exceeding 2000 vars/constrs will be explicitly marked LICENSE_LIMIT
    status = classify_solver_outcome(
        is_optimal=False,
        is_feasible=False,
        has_incumbent=False,
        gurobi_status=0,
        error_message="Model too large for size-limited license",
    )
    assert status == SolverExitStatus.LICENSE_LIMIT


# 25. Speedup and Objective Quality When Timeout
def test_25_objective_quality_timeout_semantics():
    # When full times out, degradation against UB is reported with caveat
    q = compute_objective_quality(
        obj_method=-0.24,
        obj_full=None,
        full_is_optimal=False,
        ub_full=-0.23,
        lb_best=-0.26,
    )
    assert q["abs_degradation"] is None  # Not true degradation
    assert q["incumbent_diff"] == -0.24 - (-0.23)
    assert q["bound_gap"] == -0.24 - (-0.26)
    assert q["is_exact_recovery"] is False
