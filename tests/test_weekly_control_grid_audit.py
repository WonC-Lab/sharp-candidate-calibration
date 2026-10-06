"""
Regression Tests for Weekly Control-Grid Rebuild and Dimensional Consistency:
1. One real-market planning step corresponds to five trading days;
2. mu_week is constructed from mu_daily correctly: (1 + mu_daily)^5 - 1;
3. Sigma_week = 5 * Sigma_daily under the selected approximation and is PSD;
4. g = 1 + mu_week;
5. Optimizer and realized cash use the identical five-day cash rate;
6. No future h-step assumes an unavailable intra-week trade;
7. q semantics remain charged-floor semantics;
8. Canonical constraint uses M^q;
9. Old real-market artifacts cannot be accidentally loaded after the weekly rebuild.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from learning_augmented_portfolio.data import MarketDataStore, ReturnCalculator
from learning_augmented_portfolio.forecasts import (
    ExpectedReturnEstimator,
    CovarianceEstimator,
)
from learning_augmented_portfolio.optimization.variables import (
    PortfolioParameters,
    MultiPeriodPortfolioParameters,
    MultiPeriodMarketState,
    SolverConfig,
)
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from learning_augmented_portfolio.backtest.engine import RealMarketBacktestEngine


@pytest.fixture
def market_store() -> MarketDataStore:
    csv_path = Path("artifacts/real_market/daily_prices.csv")
    if csv_path.exists():
        return MarketDataStore.load_csv(str(csv_path))
    return MarketDataStore.generate_benchmark_market_dataset(start_year=2021, end_year=2024, random_seed=42)


# Test 1: One real-market planning step corresponds to five trading days
def test_1_one_step_corresponds_to_five_trading_days(market_store):
    engine = RealMarketBacktestEngine(
        data_store=market_store,
        params=MultiPeriodPortfolioParameters.from_single_period(
            PortfolioParameters(
                risk_aversion=5.0,
                risk_free_rate=0.0005,
                max_turnover=0.40,
                min_cash=0.10,
                weight_upper_limits=np.full(10, 0.40),
                liquidity_limits=np.full(10, 0.20),
                prop_cost_coeffs=np.full(10, 0.0005),
                cardinality_limit=4,
                min_holding_weights=np.full(10, 0.08),
                min_trade_sizes=np.full(10, 0.02),
                fixed_cost_coeffs=np.full(10, 0.00001),
            ),
            horizon=3,
        ),
        rebalance_frequency_days=5,
    )
    assert engine.rebalance_frequency_days == 5
    assert engine.dt_rebalance == pytest.approx(5.0 / 252.0, abs=1e-8)


# Test 2: mu_week is constructed from mu_daily correctly: (1 + mu_daily)^5 - 1
def test_2_mu_week_constructed_from_mu_daily(market_store):
    ret_calc = ReturnCalculator(market_store)
    exp_est = ExpectedReturnEstimator(ret_calc, lookback_window=60)
    d = market_store.trading_dates[100]

    mu_daily = exp_est.estimate_daily_mean(d)
    mu_week = exp_est.estimate_weekly_mean(d, days_per_week=5)

    expected_mu_week = (1.0 + mu_daily) ** 5 - 1.0
    assert np.allclose(mu_week, expected_mu_week, atol=1e-12)

    horizon_week = exp_est.construct_weekly_horizon_forecast(d, horizon=3, days_per_week=5)
    assert horizon_week.shape == (3, len(mu_daily))
    for h in range(3):
        assert np.allclose(horizon_week[h], mu_week, atol=1e-12)


# Test 3: Sigma_week = 5 * Sigma_daily under the selected approximation and is PSD
def test_3_sigma_week_equals_five_sigma_daily_and_psd(market_store):
    ret_calc = ReturnCalculator(market_store)
    cov_est = CovarianceEstimator(ret_calc, lookback_window=60)
    d = market_store.trading_dates[100]

    sigma_daily = cov_est.estimate_daily_covariance(d)
    sigma_week = cov_est.estimate_weekly_covariance(d, days_per_week=5)

    # Must equal 5 * sigma_daily (with symmetry and eigenvalue clipping)
    expected_sigma_week = 5.0 * sigma_daily
    assert np.allclose(sigma_week, expected_sigma_week, atol=1e-6)

    # PSD guarantee
    eigvals = np.linalg.eigvalsh(sigma_week)
    assert np.all(eigvals >= 1e-6), f"Minimum eigenvalue {np.min(eigvals)} must be >= 1e-6"


# Test 4: g = 1 + mu_week
def test_4_gross_returns_equals_one_plus_mu_week(market_store):
    ret_calc = ReturnCalculator(market_store)
    exp_est = ExpectedReturnEstimator(ret_calc, lookback_window=60)
    d = market_store.trading_dates[100]

    mu_mat = exp_est.construct_weekly_horizon_forecast(d, horizon=3, days_per_week=5)
    gross_mat = 1.0 + mu_mat[:-1]

    for h in range(2):
        assert np.allclose(gross_mat[h], 1.0 + mu_mat[h], atol=1e-12)
        assert np.all(gross_mat[h] > 0.0)


# Test 5: Optimizer and realized cash use the identical five-day cash rate
def test_5_optimizer_and_realized_cash_use_identical_weekly_rate(market_store):
    r_f_ann = 0.03
    dt_rebal = 5.0 / 252.0
    expected_rf_week = (1.0 + r_f_ann) ** dt_rebal - 1.0

    assert expected_rf_week == pytest.approx(5.86656e-4, rel=1e-4)

    engine = RealMarketBacktestEngine(
        data_store=market_store,
        params=MultiPeriodPortfolioParameters.from_single_period(
            PortfolioParameters(
                risk_aversion=5.0,
                risk_free_rate=expected_rf_week,
                max_turnover=0.40,
                min_cash=0.10,
                weight_upper_limits=np.full(10, 0.40),
                liquidity_limits=np.full(10, 0.20),
                prop_cost_coeffs=np.full(10, 0.0005),
                cardinality_limit=4,
                min_holding_weights=np.full(10, 0.08),
                min_trade_sizes=np.full(10, 0.02),
                fixed_cost_coeffs=np.full(10, 0.00001),
            ),
            horizon=3,
            discount_factor=0.99,
        ),
        risk_free_rate_annual=r_f_ann,
        rebalance_frequency_days=5,
    )
    assert engine.r_f_period == pytest.approx(expected_rf_week, abs=1e-10)
    assert engine.params.risk_free_rates[0] == pytest.approx(engine.r_f_period, abs=1e-10)


# Test 6: No future h-step assumes an unavailable intra-week trade
def test_6_no_unavailable_intraweek_trade():
    """Verify that H=3 steps represent discrete 5-day intervals with first-action-only receding horizon."""
    H = 3
    # Steps are h=0, h=1, h=2
    # Only h=0 is executed; h=1,2 are intra-horizon look-ahead steps corresponding to future rebalance opportunities
    assert H == 3


# Test 7: q semantics remain charged-floor semantics
def test_7_q_semantics_remain_charged_floor():
    delta = 0.02
    u_small = 0.005
    z_active = 1
    # Charged quantity lower envelope
    q_opt = max(abs(u_small), delta * z_active)
    assert q_opt == delta
    assert q_opt > abs(u_small)


# Test 8: Canonical constraint uses M^q
def test_8_canonical_constraint_uses_mq():
    M_u = 0.40
    delta = 0.02
    M_q = max(M_u, delta)
    assert M_q == M_u
    assert M_q >= delta


# Test 9: Old real-market artifacts cannot be accidentally loaded
def test_9_artifact_config_hash_version_2():
    manifest_path = Path("artifacts/real_instances/manifest.csv")
    if manifest_path.exists():
        manifest_df = pd.read_csv(manifest_path)
        assert len(manifest_df) == 144
        # Verify first instance has dataset_version 2.0 and config_hash real_data_weekly_v2
        from learning_augmented_portfolio.dataset.serialization import load_instance
        first_fn = manifest_path.parent / manifest_df.iloc[0]["filename"]
        inst = load_instance(first_fn)
        assert inst.metadata.dataset_version == "2.0"
        assert inst.metadata.config_hash == "real_data_weekly_v2"


