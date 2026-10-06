"""
Comprehensive 12-point unit test suite for Phase 2: Single-Period MIQP.
"""

import numpy as np
import pytest

from learning_augmented_portfolio.optimization import (
    ExhaustiveDiscreteOracle,
    MarketState,
    PortfolioParameters,
    SinglePeriodMIQPOptimizer,
    SinglePeriodQPOptimizer,
    SolverConfig,
    compute_tight_big_m,
)


@pytest.fixture
def base_3asset_miqp_instance():
    """Provides a canonical 3-asset market state and Phase 2 portfolio parameters."""
    n = 3
    mu = np.array([0.15, 0.10, 0.03])
    Sigma = np.diag([0.04, 0.03, 0.02])
    w_bar = np.array([0.10, 0.10, 0.10])
    c_bar = 0.70

    state = MarketState(
        expected_returns=mu,
        covariance_matrix=Sigma,
        current_weights=w_bar,
        current_cash=c_bar,
        asset_names=["HighGrowth", "MediumGrowth", "LowYield"],
    )

    params = PortfolioParameters(
        risk_aversion=2.0,
        risk_free_rate=0.01,
        max_turnover=0.50,
        min_cash=0.10,
        weight_upper_limits=np.array([0.50, 0.50, 0.50]),
        liquidity_limits=np.array([0.40, 0.40, 0.40]),
        prop_cost_coeffs=np.array([0.001, 0.001, 0.001]),
        # Phase 2 Discrete parameters
        cardinality_limit=2,  # Must choose at most 2 out of 3
        min_holding_weights=np.array([0.15, 0.15, 0.15]),  # ell_i
        min_trade_sizes=np.array([0.05, 0.05, 0.05]),      # delta_i
        fixed_cost_coeffs=np.array([0.0005, 0.0005, 0.0005]),  # f_i
    )
    return state, params


def test_1_holding_indicator_logic(base_3asset_miqp_instance):
    """
    Test 1: Verify y_i = 0 => w_i = 0 (holding indicator logic).
    """
    state, params = base_3asset_miqp_instance
    res = SinglePeriodMIQPOptimizer(params).solve(state)

    assert res.is_optimal
    assert res.holding_indicators is not None
    assert res.post_trade_weights is not None

    for i in range(state.n_assets):
        if res.holding_indicators[i] == 0:
            assert np.isclose(res.post_trade_weights[i], 0.0, atol=1e-8)


def test_2_minimum_holding(base_3asset_miqp_instance):
    """
    Test 2: Verify y_i = 1 => w_i >= ell_i (minimum holding threshold).
    """
    state, params = base_3asset_miqp_instance
    res = SinglePeriodMIQPOptimizer(params).solve(state)

    assert res.is_optimal
    for i in range(state.n_assets):
        if res.holding_indicators[i] == 1:
            assert res.post_trade_weights[i] >= params.min_holding_weights[i] - 1e-8


def test_3_trade_indicator_logic(base_3asset_miqp_instance):
    """
    Test 3: Verify z_i = 0 => q_i = 0 => u_i = 0 (trade indicator logic).
    """
    state, params = base_3asset_miqp_instance
    res = SinglePeriodMIQPOptimizer(params).solve(state)

    assert res.is_optimal
    assert res.trade_indicators is not None
    for i in range(state.n_assets):
        if res.trade_indicators[i] == 0:
            assert np.isclose(res.absolute_trades[i], 0.0, atol=1e-8)
            assert np.isclose(res.trades[i], 0.0, atol=1e-8)


def test_4_minimum_trade_size(base_3asset_miqp_instance):
    """
    Test 4: Verify z_i = 1 => q_i >= delta_i (minimum trade size threshold).
    """
    state, params = base_3asset_miqp_instance
    res = SinglePeriodMIQPOptimizer(params).solve(state)

    assert res.is_optimal
    for i in range(state.n_assets):
        if res.trade_indicators[i] == 1:
            assert res.absolute_trades[i] >= params.min_trade_sizes[i] - 1e-8


def test_5_cardinality(base_3asset_miqp_instance):
    """
    Test 5: Verify sum_i y_i <= K (cardinality limit).
    """
    state, params = base_3asset_miqp_instance
    res = SinglePeriodMIQPOptimizer(params).solve(state)

    assert res.is_optimal
    total_held = np.sum(res.holding_indicators)
    assert total_held <= params.cardinality_limit


def test_6_fixed_plus_proportional_costs(base_3asset_miqp_instance):
    """
    Test 6: Verify numerically that kappa == sum_i (a_i * q_i + f_i * z_i).
    """
    state, params = base_3asset_miqp_instance
    res = SinglePeriodMIQPOptimizer(params).solve(state)

    assert res.is_optimal
    expected_kappa = np.sum(
        params.prop_cost_coeffs * res.absolute_trades + params.fixed_cost_coeffs * res.trade_indicators
    )
    assert np.isclose(res.transaction_cost, expected_kappa, atol=1e-7)


def test_7_fixed_cost_no_trade_region():
    """
    Test 7: Show that sufficiently high fixed execution charges f_i suppress marginal trades.
    """
    n = 2
    mu = np.array([0.08, 0.079])  # Tiny return difference
    Sigma = np.diag([0.04, 0.04])
    w_bar = np.array([0.20, 0.20])
    c_bar = 0.60

    state = MarketState(
        expected_returns=mu,
        covariance_matrix=Sigma,
        current_weights=w_bar,
        current_cash=c_bar,
    )

    # When fixed charge is large (0.05 = 5% of portfolio wealth), trading for a 0.1% return difference is prohibited
    params = PortfolioParameters(
        risk_aversion=2.0,
        risk_free_rate=0.01,
        max_turnover=0.50,
        min_cash=0.10,
        weight_upper_limits=np.array([0.50, 0.50]),
        liquidity_limits=np.array([0.30, 0.30]),
        prop_cost_coeffs=np.array([0.001, 0.001]),
        cardinality_limit=2,
        min_holding_weights=np.array([0.10, 0.10]),
        min_trade_sizes=np.array([0.01, 0.01]),
        fixed_cost_coeffs=np.array([0.05, 0.05]),  # High fixed charge
    )

    res = SinglePeriodMIQPOptimizer(params).solve(state)
    assert res.is_optimal
    # All trade indicators should be 0 (no trade)
    np.testing.assert_array_equal(res.trade_indicators, [0, 0])
    np.testing.assert_allclose(res.trades, [0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(res.post_trade_weights, w_bar, atol=1e-8)


def test_8_tight_big_m_validity():
    """
    Test 8: Verify tight Big-M bounds calculation on extreme buy and sell cases.
    """
    w_bar = np.array([0.40, 0.00, 0.70])
    U = np.array([0.60, 0.50, 0.50])  # Note: Asset 2 is over-limit (w_bar > U)
    L = np.array([0.25, 0.30, 0.40])
    tau_max = 0.35

    m_sell, m_buy, m_symmetric = compute_tight_big_m(w_bar, U, L, tau_max)

    # Asset 0: w_bar=0.40, U=0.60 => M_sell = min(0.40, 0.25, 0.35) = 0.25
    #                                M_buy  = min(0.20, 0.25, 0.35) = 0.20
    #                                M_sym  = 0.25
    assert np.isclose(m_sell[0], 0.25)
    assert np.isclose(m_buy[0], 0.20)
    assert np.isclose(m_symmetric[0], 0.25)

    # Asset 1: w_bar=0.00, U=0.50 => M_sell = min(0.00, 0.30, 0.35) = 0.00
    #                                M_buy  = min(0.50, 0.30, 0.35) = 0.30
    #                                M_sym  = 0.30
    assert np.isclose(m_sell[1], 0.00)
    assert np.isclose(m_buy[1], 0.30)
    assert np.isclose(m_symmetric[1], 0.30)

    # Asset 2: w_bar=0.70 > U=0.50 => M_sell = min(0.70, 0.40, 0.35) = 0.35
    #                                  M_buy  = min(max(0, -0.20), 0.40, 0.35) = 0.00
    #                                  M_sym  = 0.35
    assert np.isclose(m_sell[2], 0.35)
    assert np.isclose(m_buy[2], 0.00)
    assert np.isclose(m_symmetric[2], 0.35)


def test_9_invalid_big_m_minimum_trade_rejection():
    """
    Test 9: Reject configuration where minimum trade size delta_i exceeds tight Big-M bound M_i.
    """
    state = MarketState(
        expected_returns=np.array([0.10, 0.05]),
        covariance_matrix=np.diag([0.04, 0.04]),
        current_weights=np.array([0.05, 0.05]),
        current_cash=0.90,
    )

    # Asset 0: U=0.10, w_bar=0.05 => M_buy = 0.05, M_sell = 0.05 => M_sym = 0.05
    # But delta_0 = 0.10 > 0.05 => Infeasible to trade!
    params = PortfolioParameters(
        risk_aversion=2.0,
        risk_free_rate=0.01,
        max_turnover=0.50,
        min_cash=0.10,
        weight_upper_limits=np.array([0.10, 0.50]),
        liquidity_limits=np.array([0.50, 0.50]),
        prop_cost_coeffs=np.array([0.001, 0.001]),
        cardinality_limit=2,
        min_trade_sizes=np.array([0.10, 0.02]),  # delta_0 = 0.10 > M_0 = 0.05
    )

    optimizer = SinglePeriodMIQPOptimizer(params)
    with pytest.raises(ValueError, match="exceeds tight Big-M bound"):
        optimizer.solve(state)


def test_10_cardinality_driven_asset_selection():
    """
    Test 10: With K=1 out of 3 assets, optimizer must pick only the single most attractive asset.
    """
    n = 3
    mu = np.array([0.20, 0.05, 0.02])  # Asset 0 vastly superior
    Sigma = np.diag([0.04, 0.04, 0.04])
    w_bar = np.array([0.0, 0.0, 0.0])
    c_bar = 1.0

    state = MarketState(
        expected_returns=mu,
        covariance_matrix=Sigma,
        current_weights=w_bar,
        current_cash=c_bar,
    )

    params = PortfolioParameters(
        risk_aversion=2.0,
        risk_free_rate=0.01,
        max_turnover=0.80,
        min_cash=0.10,
        weight_upper_limits=np.array([0.60, 0.60, 0.60]),
        liquidity_limits=np.array([0.80, 0.80, 0.80]),
        prop_cost_coeffs=np.array([0.001, 0.001, 0.001]),
        cardinality_limit=1,  # Only 1 asset permitted
        min_holding_weights=np.array([0.10, 0.10, 0.10]),
        min_trade_sizes=np.array([0.05, 0.05, 0.05]),
        fixed_cost_coeffs=np.array([0.0005, 0.0005, 0.0005]),
    )

    res = SinglePeriodMIQPOptimizer(params).solve(state)
    assert res.is_optimal
    assert np.sum(res.holding_indicators) == 1
    # Asset 0 must be selected, assets 1 and 2 must be 0
    assert res.holding_indicators[0] == 1
    assert res.holding_indicators[1] == 0
    assert res.holding_indicators[2] == 0
    assert res.post_trade_weights[0] > 0.10


def test_11_infeasibility_detection():
    """
    Test 11: Incompatible cardinality and minimum holding constraints cause clean INFEASIBLE.
    Example: 3 assets, each has ell_i = 0.40. Min cash = 0.20.
    If K=3, sum w_i >= 3 * 0.40 = 1.20 > 1.0 - c_min.
    If sector lower bounds also force holdings, clean INFEASIBLE must be detected.
    """
    state = MarketState(
        expected_returns=np.array([0.10, 0.08]),
        covariance_matrix=np.diag([0.04, 0.04]),
        current_weights=np.array([0.10, 0.10]),
        current_cash=0.80,
    )

    # Incompatible: K=2, ell_0=0.60, ell_1=0.60, c_min=0.20 => sum w + c >= 1.40 > 1.0
    # AND sector lower bounds force w_0 >= 0.50 and w_1 >= 0.50
    params = PortfolioParameters(
        risk_aversion=2.0,
        risk_free_rate=0.01,
        max_turnover=1.0,
        min_cash=0.20,
        weight_upper_limits=np.array([1.0, 1.0]),
        liquidity_limits=np.array([1.0, 1.0]),
        prop_cost_coeffs=np.array([0.001, 0.001]),
        cardinality_limit=2,
        min_holding_weights=np.array([0.60, 0.60]),
        sector_indices={"S1": [0], "S2": [1]},
        sector_bounds={"S1": (0.50, 1.0), "S2": (0.50, 1.0)},
    )

    res = SinglePeriodMIQPOptimizer(params).solve(state)
    assert not res.is_optimal
    assert not res.is_feasible
    assert res.status == "INFEASIBLE"


def test_12_phase1_regression_limit():
    """
    Test 12: Deliberate degenerate regression limit:
    f_i = 0, ell_i = 0, delta_i = 0, K = n, allow_degenerate = True.
    Phase 2 MIQP must reproduce Phase 1 QP optimum to high numerical tolerance.
    """
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
    )

    # Phase 1 parameters
    params_qp = PortfolioParameters(
        risk_aversion=2.0,
        risk_free_rate=0.01,
        max_turnover=0.30,
        min_cash=0.10,
        weight_upper_limits=np.array([0.60, 0.60]),
        liquidity_limits=np.array([0.30, 0.30]),
        prop_cost_coeffs=np.array([0.002, 0.002]),
        sector_indices={"Sec1": [0], "Sec2": [1]},
        sector_bounds={"Sec1": (0.0, 0.50), "Sec2": (0.0, 0.50)},
    )
    res_qp = SinglePeriodQPOptimizer(params_qp).solve(state)

    # Degenerate Phase 2 parameters
    params_miqp_deg = PortfolioParameters(
        risk_aversion=2.0,
        risk_free_rate=0.01,
        max_turnover=0.30,
        min_cash=0.10,
        weight_upper_limits=np.array([0.60, 0.60]),
        liquidity_limits=np.array([0.30, 0.30]),
        prop_cost_coeffs=np.array([0.002, 0.002]),
        sector_indices={"Sec1": [0], "Sec2": [1]},
        sector_bounds={"Sec1": (0.0, 0.50), "Sec2": (0.0, 0.50)},
        cardinality_limit=n,
        min_holding_weights=np.zeros(n),
        min_trade_sizes=np.zeros(n),
        fixed_cost_coeffs=np.zeros(n),
        allow_degenerate=True,
    )
    res_miqp = SinglePeriodMIQPOptimizer(params_miqp_deg).solve(state)

    assert res_qp.is_optimal
    assert res_miqp.is_optimal
    # Objective value equality
    assert np.isclose(res_qp.objective_value, res_miqp.objective_value, atol=1e-6)
    # Decision variables equality
    np.testing.assert_allclose(res_qp.post_trade_weights, res_miqp.post_trade_weights, atol=1e-5)
    np.testing.assert_allclose(res_qp.trades, res_miqp.trades, atol=1e-5)
    assert np.isclose(res_qp.post_trade_cash, res_miqp.post_trade_cash, atol=1e-5)
    assert np.isclose(res_qp.transaction_cost, res_miqp.transaction_cost, atol=1e-5)
