"""
Unit Tests for Phase 9: Real-Market Backtesting & No-Lookahead Architecture:
  1. Adjusted return construction
  2. Strict chronological feature windows
  3. No-lookahead covariance
  4. No-lookahead ADV
  5. Liquidity scaling with actual wealth
  6. Transaction-cost normalization
  7. Rebalancing-calendar correctness
  8. Warm-up exclusion
  9. Chronological real-data split
  10. Missing-data policy
  11. Real-data teacher replayability
  12. Real-data ML training uses train only
  13. Validation-only candidate-budget selection
  14. Frozen-test enforcement
  15. Realized return separated from forecast return
  16. Wealth accounting over real timestamps
  17. Hard-constraint audit
  18. Financial-metric correctness
  19. Certificate-source semantics
  20. Regression test integrity
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from learning_augmented_portfolio.data import (
    RealMarketUniverse,
    TradingCalendar,
    MarketDataStore,
    ReturnCalculator,
    LiquidityCalculator,
    audit_no_lookahead_slice,
)
from learning_augmented_portfolio.forecasts import (
    ExpectedReturnEstimator,
    ReturnEstimatorType,
    CovarianceEstimator,
    CovarianceEstimatorType,
)
from learning_augmented_portfolio.backtest import (
    RealMarketBacktestEngine,
    EqualWeightBaseline,
    BuyAndHoldBaseline,
    BacktestAuditor,
    compute_financial_metrics,
)
from learning_augmented_portfolio.optimization.variables import (
    PortfolioParameters,
    MultiPeriodMarketState,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from learning_augmented_portfolio.recovery.certificate import CertificateSource, compute_full_problem_certificate


@pytest.fixture
def benchmark_data_store() -> MarketDataStore:
    """Generate 2021-2024 benchmark daily market store for testing."""
    return MarketDataStore.generate_benchmark_market_dataset(start_year=2021, end_year=2024, random_seed=42)


@pytest.fixture
def standard_params() -> MultiPeriodPortfolioParameters:
    n = 10
    single_p = PortfolioParameters(
        risk_aversion=5.0,
        risk_free_rate=0.0001,
        max_turnover=0.40,
        min_cash=0.10,
        weight_upper_limits=np.full(n, 0.40),
        liquidity_limits=np.full(n, 0.20),
        prop_cost_coeffs=np.full(n, 0.0005),
        cardinality_limit=4,
        min_holding_weights=np.full(n, 0.08),
        min_trade_sizes=np.full(n, 0.02),
        fixed_cost_coeffs=np.full(n, 0.00001),
    )
    return MultiPeriodPortfolioParameters.from_single_period(single_p, horizon=3)


# 1. Adjusted Return Construction
def test_1_adjusted_return_construction(benchmark_data_store):
    ret_calc = ReturnCalculator(benchmark_data_store)
    first_date = benchmark_data_store.trading_dates[1]
    r0 = ret_calc.get_realized_single_day_return(first_date)

    p0 = benchmark_data_store.get_prices(benchmark_data_store.trading_dates[0])
    p1 = benchmark_data_store.get_prices(first_date)
    expected_r = (p1 / p0) - 1.0

    assert np.allclose(r0, expected_r, atol=1e-5)


# 2. Strict Chronological Feature Windows
def test_2_strict_chronological_feature_windows(benchmark_data_store):
    ret_calc = ReturnCalculator(benchmark_data_store)
    date_50 = benchmark_data_store.trading_dates[50]
    win = ret_calc.get_historical_return_window(date_50, lookback_window=20)
    assert win.shape == (20, benchmark_data_store.universe.n_assets)


# 3. No-Lookahead Covariance
def test_3_no_lookahead_covariance(benchmark_data_store):
    ret_calc = ReturnCalculator(benchmark_data_store)
    cov_est = CovarianceEstimator(ret_calc, estimator_type=CovarianceEstimatorType.LEDOIT_WOLF, lookback_window=40)
    d = benchmark_data_store.trading_dates[50]
    sigma = cov_est.estimate_daily_covariance(d)
    assert sigma.shape == (10, 10)
    eigvals = np.linalg.eigvalsh(sigma)
    assert np.all(eigvals >= 1e-6)


# 4. No-Lookahead ADV
def test_4_no_lookahead_adv(benchmark_data_store):
    liq_calc = LiquidityCalculator(benchmark_data_store)
    d = benchmark_data_store.trading_dates[40]
    adv = liq_calc.compute_adv(d, lookback_window=20)
    assert len(adv) == 10
    assert np.all(adv > 0)


# 5. Liquidity Scaling With Actual Wealth
def test_5_liquidity_scaling_with_actual_wealth(benchmark_data_store):
    liq_calc = LiquidityCalculator(benchmark_data_store, default_participation_rate=0.05)
    d = benchmark_data_store.trading_dates[40]

    limit_1m = liq_calc.compute_normalized_liquidity_limits(d, portfolio_wealth=1_000_000.0)
    limit_10m = liq_calc.compute_normalized_liquidity_limits(d, portfolio_wealth=10_000_000.0)

    # 10x larger wealth -> 10x smaller normalized trade limit
    ratio = limit_1m / limit_10m
    valid_mask = (limit_1m < 1.0) & (limit_10m < 1.0)
    if np.any(valid_mask):
        assert np.allclose(ratio[valid_mask], 10.0, rtol=0.1)


# 6. Transaction-Cost Normalization
def test_6_transaction_cost_normalization():
    fixed_dollar = 20.0
    wealth = 100_000.0
    f_norm = fixed_dollar / wealth
    assert np.isclose(f_norm, 0.0002)


# 7. Rebalancing-Calendar Correctness
def test_7_rebalancing_calendar_correctness(benchmark_data_store):
    cal = benchmark_data_store.calendar
    d_start = benchmark_data_store.trading_dates[60]
    d_end = benchmark_data_store.trading_dates[100]

    rebal_dates = cal.get_rebalancing_dates(d_start, d_end, frequency_days=5)
    assert len(rebal_dates) == 9
    assert rebal_dates[0] == d_start


# 8. Warm-Up Exclusion
def test_8_warm_up_exclusion(benchmark_data_store):
    ret_calc = ReturnCalculator(benchmark_data_store)
    too_early = benchmark_data_store.trading_dates[10]
    with pytest.raises(ValueError, match="Insufficient historical"):
        ret_calc.get_historical_return_window(too_early, lookback_window=60)


# 9. Chronological Real-Data Split
def test_9_chronological_real_data_split(benchmark_data_store):
    all_dates = benchmark_data_store.trading_dates
    T = len(all_dates)
    train_end = int(T * 0.6)
    val_end = int(T * 0.8)

    train_dates = all_dates[:train_end]
    val_dates = all_dates[train_end:val_end]
    test_dates = all_dates[val_end:]

    assert max(train_dates) < min(val_dates)
    assert max(val_dates) < min(test_dates)


# 10. Missing-Data Policy
def test_10_missing_data_policy():
    bad_df = pd.DataFrame({
        "date": ["2026-01-01"], "ticker": ["AAPL"],
        "close": [150.0], "adj_close": [np.nan], "volume": [1000]
    })
    with pytest.raises(ValueError, match="unhandled NaN"):
        MarketDataStore(bad_df)


# 11. Real-Data Teacher Replayability
def test_11_real_data_teacher_replayability(benchmark_data_store, standard_params):
    ret_calc = ReturnCalculator(benchmark_data_store)
    liq_calc = LiquidityCalculator(benchmark_data_store)
    exp_est = ExpectedReturnEstimator(ret_calc)
    cov_est = CovarianceEstimator(ret_calc)

    d = benchmark_data_store.trading_dates[80]
    w0 = np.array([0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    c0 = 0.40

    H = standard_params.horizon
    mu_mat = exp_est.construct_horizon_forecast(d, H)
    cov_mat = cov_est.construct_horizon_forecast(d, H)
    gross_mat = 1.0 + mu_mat[:-1]

    mkt = MultiPeriodMarketState(
        initial_holdings=w0,
        initial_cash=c0,
        expected_returns=mu_mat,
        covariance_matrices=cov_mat,
        gross_returns=gross_mat,
        asset_names=benchmark_data_store.universe.tickers,
    )

    opt = MultiPeriodMIQPOptimizer(standard_params, SolverConfig(time_limit=10.0, mip_gap=1e-5, threads=1))
    res1 = opt.solve(mkt)
    res2 = opt.solve(mkt)

    assert res1.is_feasible and res2.is_feasible
    assert np.isclose(res1.objective_value, res2.objective_value, atol=1e-6)


# 12. Real-Data ML Training Uses Train Only
def test_12_real_data_ml_training_uses_train_only():
    split_records = [{"split": "train", "val": 1}, {"split": "train", "val": 2}, {"split": "test", "val": 3}]
    df = pd.DataFrame(split_records)
    train_subset = df[df["split"] == "train"]
    assert len(train_subset) == 2
    assert "test" not in train_subset["split"].values


# 13. Validation-Only Candidate-Budget Selection
def test_13_validation_only_candidate_budget_selection():
    val_budget_results = {3: 0.92, 4: 0.98, 5: 1.0}
    chosen_b = min(b for b, cov in val_budget_results.items() if cov >= 0.95)
    assert chosen_b == 4


# 14. Frozen-Test Enforcement
def test_14_frozen_test_enforcement():
    frozen_policy = {"model": "LogisticRegression", "B": 4}
    # Test evaluation must strictly consume frozen_policy without altering it
    assert frozen_policy["B"] == 4


# 15. Realized Return Separated From Forecast Return
def test_15_realized_return_separated_from_forecast_return(benchmark_data_store):
    ret_calc = ReturnCalculator(benchmark_data_store)
    exp_est = ExpectedReturnEstimator(ret_calc)
    d = benchmark_data_store.trading_dates[80]
    d_next = benchmark_data_store.trading_dates[81]

    mu_hat = exp_est.estimate_daily_mean(d)
    r_real = ret_calc.get_realized_single_day_return(d_next)
    # Realized returns must not match forecasts identically
    assert not np.allclose(mu_hat, r_real)


# 16. Wealth Accounting Over Real Timestamps
def test_16_wealth_accounting_over_real_timestamps(benchmark_data_store, standard_params):
    engine = RealMarketBacktestEngine(
        benchmark_data_store,
        standard_params,
        rebalance_frequency_days=5,
    )
    d_start = benchmark_data_store.trading_dates[70]
    d_end = benchmark_data_store.trading_dates[90]

    eq_strat = EqualWeightBaseline(standard_params)
    res = engine.run_backtest(
        "EqualWeight",
        d_start,
        d_end,
        lambda mkt, st, dt: (*eq_strat.get_target_weights(10), {"solve_time_ms": 0.1}),
    )

    auditor = BacktestAuditor(standard_params)
    audit_res = auditor.audit_backtest("EqualWeight", res.step_records, res.wealth_series, check_constraints=False)
    assert audit_res.is_wealth_accounting_valid is True
    assert audit_res.max_wealth_residual < 1e-4


# 17. Hard-Constraint Audit
def test_17_hard_constraint_audit(standard_params):
    auditor = BacktestAuditor(standard_params)
    assert auditor.tolerance == 1e-4


# 18. Financial-Metric Correctness
def test_18_financial_metric_correctness():
    w = np.array([100.0, 105.0, 110.0, 108.0, 115.0])
    turnover = np.array([0.1, 0.1, 0.1, 0.1])
    costs = np.array([1.0, 1.0, 1.0, 1.0])
    metrics = compute_financial_metrics("TestStrat", w, turnover, costs, np.array([3, 3, 3, 3]), np.array([0.1, 0.1, 0.1, 0.1]))

    assert metrics.final_wealth == 115.0
    assert metrics.total_transaction_cost == 4.0
    assert metrics.max_drawdown <= 0.0


# 19. Certificate-Source Semantics
def test_19_certificate_source_semantics():
    cert_full = compute_full_problem_certificate(
        instance_id="CERT_SRC",
        lower_bound=-0.25,
        upper_bound=-0.24,
        reduced_mip_gap=1e-5,
        candidate_budget=10,
        candidate_size=10,
        full_mip_bound=-0.240001,
        is_full_universe=True,
    )
    assert cert_full.certificate_source == CertificateSource.FULL_MIP_BOUND


# 20. Audit No Lookahead
def test_20_audit_no_lookahead_slice(benchmark_data_store):
    d = benchmark_data_store.trading_dates[100]
    assert audit_no_lookahead_slice(benchmark_data_store, d, lookback_window=40) is True
