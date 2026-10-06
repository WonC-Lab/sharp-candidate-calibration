"""
Unit tests for Phase 1: Convex Single-Period Portfolio Optimization.
"""

import numpy as np
import pytest

from learning_augmented_portfolio.optimization import (
    MarketState,
    OptimizationResult,
    PortfolioParameters,
    SinglePeriodQPOptimizer,
    SolverConfig,
)


@pytest.fixture
def base_2asset_instance():
    """Provides a canonical 2-asset market state and portfolio parameters."""
    n = 2
    mu = np.array([0.12, 0.04])
    Sigma = np.array([[0.04, 0.00], [0.00, 0.04]])
    w_bar = np.array([0.10, 0.10])
    c_bar = 0.80

    state = MarketState(
        expected_returns=mu,
        covariance_matrix=Sigma,
        current_weights=w_bar,
        current_cash=c_bar,
        asset_names=["Asset_A", "Asset_B"],
    )

    params = PortfolioParameters(
        risk_aversion=2.0,
        risk_free_rate=0.01,
        max_turnover=0.30,
        min_cash=0.10,
        weight_upper_limits=np.array([0.60, 0.60]),
        liquidity_limits=np.array([0.30, 0.30]),
        prop_cost_coeffs=np.array([0.002, 0.002]),
        sector_indices={"Sector_1": [0], "Sector_2": [1]},
        sector_bounds={"Sector_1": (0.0, 0.50), "Sector_2": (0.0, 0.50)},
    )
    return state, params


def test_1_absolute_trade_linearization_tightness(base_2asset_instance):
    """
    Test 1: Verify that at optimum, q_i == |u_i| holds tightly within numerical tolerance.
    """
    state, params = base_2asset_instance
    optimizer = SinglePeriodQPOptimizer(params)
    res = optimizer.solve(state)

    assert res.is_optimal
    assert res.trades is not None
    assert res.absolute_trades is not None
    np.testing.assert_allclose(res.absolute_trades, np.abs(res.trades), atol=1e-7)


def test_2_cash_and_budget_conservation(base_2asset_instance):
    """
    Test 2: Verify self-financing identity:
    1^T w + c + kappa == 1.0 and c == c_bar - 1^T u - kappa
    """
    state, params = base_2asset_instance
    optimizer = SinglePeriodQPOptimizer(params)
    res = optimizer.solve(state)

    assert res.is_optimal
    total_wealth = np.sum(res.post_trade_weights) + res.post_trade_cash + res.transaction_cost
    assert np.isclose(total_wealth, 1.0, atol=1e-7)

    expected_cash = state.current_cash - np.sum(res.trades) - res.transaction_cost
    assert np.isclose(res.post_trade_cash, expected_cash, atol=1e-7)


def test_3_liquidity_limits_enforced(base_2asset_instance):
    """
    Test 3: Verify q_i <= L_i holds for all assets.
    """
    state, params = base_2asset_instance
    optimizer = SinglePeriodQPOptimizer(params)
    res = optimizer.solve(state)

    assert res.is_optimal
    assert np.all(res.absolute_trades <= params.liquidity_limits + 1e-8)


def test_4_turnover_limit_enforced(base_2asset_instance):
    """
    Test 4: Verify sum_i q_i <= tau_max holds.
    """
    state, params = base_2asset_instance
    optimizer = SinglePeriodQPOptimizer(params)
    res = optimizer.solve(state)

    assert res.is_optimal
    total_turnover = np.sum(res.absolute_trades)
    assert total_turnover <= params.max_turnover + 1e-8


def test_5_sector_exposure_bounds(base_2asset_instance):
    """
    Test 5: Verify E_g^min <= sum_{i in S_g} w_i <= E_g^max for all sectors.
    """
    state, params = base_2asset_instance
    optimizer = SinglePeriodQPOptimizer(params)
    res = optimizer.solve(state)

    assert res.is_optimal
    for sector, indices in params.sector_indices.items():
        e_min, e_max = params.sector_bounds[sector]
        sector_weight = np.sum(res.post_trade_weights[indices])
        assert sector_weight >= e_min - 1e-8
        assert sector_weight <= e_max + 1e-8


def test_6_min_cash_reserve(base_2asset_instance):
    """
    Test 6: Verify c >= c_min holds.
    """
    state, params = base_2asset_instance
    optimizer = SinglePeriodQPOptimizer(params)
    res = optimizer.solve(state)

    assert res.is_optimal
    assert res.post_trade_cash >= params.min_cash - 1e-8


def test_7_no_trade_region_under_high_costs(base_2asset_instance):
    """
    Test 7: Under prohibitively high transaction costs (a_i = 10.0),
    the optimal solution should be not to trade: u* == 0, w* == w_bar.
    """
    state, params = base_2asset_instance
    high_cost_params = PortfolioParameters(
        risk_aversion=params.risk_aversion,
        risk_free_rate=params.risk_free_rate,
        max_turnover=params.max_turnover,
        min_cash=params.min_cash,
        weight_upper_limits=params.weight_upper_limits,
        liquidity_limits=params.liquidity_limits,
        prop_cost_coeffs=np.array([10.0, 10.0]),  # Prohibitive cost
        sector_indices=params.sector_indices,
        sector_bounds=params.sector_bounds,
    )
    optimizer = SinglePeriodQPOptimizer(high_cost_params)
    res = optimizer.solve(state)

    assert res.is_optimal
    np.testing.assert_allclose(res.trades, np.zeros(2), atol=1e-6)
    np.testing.assert_allclose(res.post_trade_weights, state.current_weights, atol=1e-6)
    assert np.isclose(res.transaction_cost, 0.0, atol=1e-6)


def test_8_zero_transaction_costs_unconstrained_recovery():
    """
    Test 8: With zero transaction costs and unbinding bounds:
    (a) Interior cash regime (c* > c_min): w* = (1/gamma) * Sigma^{-1} (mu - r_f 1)
    (b) Fully invested regime (c* = 0): w* = (1/gamma) * Sigma^{-1} (mu - lambda* 1)
    """
    gamma = 2.0
    r_f = 0.01

    # -------------------------------------------------------------------------
    # Case (a): Interior cash regime where c* > 0 strictly
    # -------------------------------------------------------------------------
    mu_a = np.array([0.03, 0.02])
    Sigma_a = np.diag([0.05, 0.05])
    excess_mu_a = mu_a - r_f * np.ones(2)
    w_analytical_a = np.linalg.inv(gamma * Sigma_a) @ excess_mu_a  # [0.20, 0.10]
    expected_c_a = 1.0 - np.sum(w_analytical_a)  # 0.70

    state_a = MarketState(
        expected_returns=mu_a,
        covariance_matrix=Sigma_a,
        current_weights=np.array([0.15, 0.15]),
        current_cash=0.70,
    )
    params_a = PortfolioParameters(
        risk_aversion=gamma,
        risk_free_rate=r_f,
        max_turnover=5.0,
        min_cash=0.0,
        weight_upper_limits=np.array([2.0, 2.0]),
        liquidity_limits=np.array([5.0, 5.0]),
        prop_cost_coeffs=np.array([0.0, 0.0]),
    )
    res_a = SinglePeriodQPOptimizer(params_a).solve(state_a)
    assert res_a.is_optimal
    np.testing.assert_allclose(res_a.post_trade_weights, w_analytical_a, atol=1e-5)
    assert np.isclose(res_a.post_trade_cash, expected_c_a, atol=1e-5)

    # -------------------------------------------------------------------------
    # Case (b): Fully invested budget-binding regime (c* = 0)
    # -------------------------------------------------------------------------
    mu_b = np.array([0.08, 0.06])
    Sigma_b = np.diag([0.02, 0.03])
    inv_Sigma_b = np.diag([50.0, 100.0 / 3.0])
    ones = np.ones(2)
    # lambda* = (1^T Sigma^{-1} mu - gamma) / (1^T Sigma^{-1} 1)
    lambda_star = (ones @ inv_Sigma_b @ mu_b - gamma) / (ones @ inv_Sigma_b @ ones)
    w_analytical_b = (1.0 / gamma) * inv_Sigma_b @ (mu_b - lambda_star * ones)  # [0.80, 0.20]

    state_b = MarketState(
        expected_returns=mu_b,
        covariance_matrix=Sigma_b,
        current_weights=np.array([0.5, 0.5]),
        current_cash=0.0,
    )
    params_b = PortfolioParameters(
        risk_aversion=gamma,
        risk_free_rate=r_f,
        max_turnover=5.0,
        min_cash=0.0,
        weight_upper_limits=np.array([2.0, 2.0]),
        liquidity_limits=np.array([5.0, 5.0]),
        prop_cost_coeffs=np.array([0.0, 0.0]),
    )
    res_b = SinglePeriodQPOptimizer(params_b).solve(state_b)
    assert res_b.is_optimal
    np.testing.assert_allclose(res_b.post_trade_weights, w_analytical_b, atol=1e-5)
    assert np.isclose(res_b.post_trade_cash, 0.0, atol=1e-5)


def test_9_infeasibility_detection():
    """
    Test 9: Incompatible constraints (e.g. c_min = 0.8 and sum of sector mins = 0.5 > 1 - c_min)
    must cause the solver to return INFEASIBLE cleanly.
    """
    n = 2
    state = MarketState(
        expected_returns=np.array([0.10, 0.05]),
        covariance_matrix=np.diag([0.04, 0.04]),
        current_weights=np.array([0.10, 0.10]),
        current_cash=0.80,
    )

    # Incompatible: c_min = 0.8, but Sector 1 requires w_0 >= 0.3 and Sector 2 requires w_1 >= 0.3
    # sum w + c >= 0.3 + 0.3 + 0.8 = 1.4 > 1.0
    params = PortfolioParameters(
        risk_aversion=2.0,
        risk_free_rate=0.01,
        max_turnover=1.0,
        min_cash=0.80,
        weight_upper_limits=np.array([1.0, 1.0]),
        liquidity_limits=np.array([1.0, 1.0]),
        prop_cost_coeffs=np.array([0.001, 0.001]),
        sector_indices={"Sec1": [0], "Sec2": [1]},
        sector_bounds={"Sec1": (0.30, 1.0), "Sec2": (0.30, 1.0)},
    )

    optimizer = SinglePeriodQPOptimizer(params)
    res = optimizer.solve(state)

    assert not res.is_optimal
    assert not res.is_feasible
    assert res.status == "INFEASIBLE"


def test_10_psd_validation_and_rejection():
    """
    Test 10: Non-PSD covariance matrix (negative eigenvalue) must be rejected
    by MarketState.validate with a ValueError.
    """
    n = 2
    mu = np.array([0.10, 0.05])
    # Indefinite matrix with eigenvalues: lambda_1 > 0, lambda_2 < 0
    non_psd_cov = np.array([[0.04, 0.10], [0.10, 0.04]])

    state = MarketState(
        expected_returns=mu,
        covariance_matrix=non_psd_cov,
        current_weights=np.array([0.5, 0.5]),
        current_cash=0.0,
    )

    with pytest.raises(ValueError, match="positive semi-definite"):
        state.validate()
