"""
Comprehensive 14-point unit test suite for Phase 3: Deterministic Multi-Period MIQP.
"""

import numpy as np
import pytest

from learning_augmented_portfolio.optimization import (
    MarketState,
    MultiPeriodMarketState,
    MultiPeriodMIQPOptimizer,
    MultiPeriodPortfolioParameters,
    PortfolioParameters,
    SinglePeriodMIQPOptimizer,
    SolverConfig,
)


@pytest.fixture
def base_multi_period_instance():
    """Provides a canonical 3-period, 3-asset multi-period market state and portfolio parameters."""
    H = 3
    n = 3
    asset_names = ["Asset_A", "Asset_B", "Asset_C"]

    # Initial state at h=0
    x0_minus = np.array([0.15, 0.15, 0.10])
    b0_minus = 0.60

    # Expected returns per horizon step
    mu = np.array([
        [0.12, 0.08, 0.04],
        [0.10, 0.07, 0.04],
        [0.09, 0.06, 0.03],
    ])

    # Covariance matrices
    sigma_base = np.diag([0.04, 0.03, 0.02])
    covariances = np.tile(sigma_base, (H, 1, 1))

    # Gross returns for transitions (H-1 = 2 steps)
    # Asset A grows 10% in step 0, Asset B grows 5%, Asset C flat
    gross_returns = np.array([
        [1.10, 1.05, 1.00],
        [1.08, 1.04, 1.00],
    ])

    state = MultiPeriodMarketState(
        initial_holdings=x0_minus,
        initial_cash=b0_minus,
        expected_returns=mu,
        covariance_matrices=covariances,
        gross_returns=gross_returns,
        asset_names=asset_names,
    )

    params = MultiPeriodPortfolioParameters(
        risk_aversion=2.0,
        horizon=H,
        risk_free_rates=np.array([0.01, 0.01, 0.01]),
        max_turnovers=np.array([0.40, 0.40, 0.40]),
        min_cash=np.array([0.10, 0.10, 0.10]),
        weight_upper_limits=np.full((H, n), 0.50),
        liquidity_limits=np.full((H, n), 0.30),
        prop_cost_coeffs=np.full((H, n), 0.001),
        discount_factor=0.98,
        cardinality_limits=np.array([2, 2, 2]),
        min_holding_weights=np.full((H, n), 0.10),
        min_trade_sizes=np.full((H, n), 0.02),
        fixed_cost_coeffs=np.full((H, n), 0.0005),
        sector_indices={"Tech": [0, 1], "Defensive": [2]},
        sector_bounds={"Tech": (0.0, 0.70), "Defensive": (0.0, 0.50)},
    )
    return state, params


def test_1_h1_exact_reduction():
    """
    Test 1: When H = 1, Phase 3 Multi-Period MIQP must reduce EXACTLY to Phase 2 MIQP.
    Verify objective, x_0 vs w, u_0 vs u, q_0 vs q, b_0 vs c, kappa_0 vs kappa, y_0, z_0.
    """
    n = 3
    mu = np.array([0.14, 0.09, 0.03])
    Sigma = np.diag([0.04, 0.03, 0.02])
    w_bar = np.array([0.10, 0.10, 0.10])
    c_bar = 0.70

    state_p2 = MarketState(
        expected_returns=mu,
        covariance_matrix=Sigma,
        current_weights=w_bar,
        current_cash=c_bar,
    )
    params_p2 = PortfolioParameters(
        risk_aversion=2.5,
        risk_free_rate=0.015,
        max_turnover=0.35,
        min_cash=0.15,
        weight_upper_limits=np.array([0.50, 0.50, 0.50]),
        liquidity_limits=np.array([0.25, 0.25, 0.25]),
        prop_cost_coeffs=np.array([0.001, 0.001, 0.001]),
        cardinality_limit=2,
        min_holding_weights=np.array([0.10, 0.10, 0.10]),
        min_trade_sizes=np.array([0.03, 0.03, 0.03]),
        fixed_cost_coeffs=np.array([0.0004, 0.0004, 0.0004]),
    )

    # Solve Phase 2
    res_p2 = SinglePeriodMIQPOptimizer(params_p2).solve(state_p2)
    assert res_p2.is_optimal

    # Build H=1 Phase 3 instance
    params_p3 = MultiPeriodPortfolioParameters.from_single_period(params_p2, horizon=1)
    state_p3 = MultiPeriodMarketState.from_single_period(state_p2, horizon=1)

    # Solve Phase 3
    res_p3 = MultiPeriodMIQPOptimizer(params_p3).solve(state_p3)
    assert res_p3.is_optimal

    # Exact comparison
    assert np.isclose(res_p2.objective_value, res_p3.objective_value, atol=1e-7)
    np.testing.assert_allclose(res_p2.post_trade_weights, res_p3.post_trade_holdings[0], atol=1e-6)
    np.testing.assert_allclose(res_p2.trades, res_p3.trades[0], atol=1e-6)
    np.testing.assert_allclose(res_p2.absolute_trades, res_p3.absolute_trades[0], atol=1e-6)
    assert np.isclose(res_p2.post_trade_cash, res_p3.post_trade_cash[0], atol=1e-6)
    assert np.isclose(res_p2.transaction_cost, res_p3.transaction_costs[0], atol=1e-6)
    np.testing.assert_array_equal(res_p2.holding_indicators, res_p3.holding_indicators[0])
    np.testing.assert_array_equal(res_p2.trade_indicators, res_p3.trade_indicators[0])


def test_2_trade_state_identity(base_multi_period_instance):
    """
    Test 2: Verify u_h == x_h - x_h^- holds across all h in {0, ..., H-1}.
    """
    state, params = base_multi_period_instance
    res = MultiPeriodMIQPOptimizer(params).solve(state)
    assert res.is_optimal

    H = params.horizon
    for h in range(H):
        expected_u = res.post_trade_holdings[h] - res.pre_trade_holdings[h]
        np.testing.assert_allclose(res.trades[h], expected_u, atol=1e-7)


def test_3_absolute_trade_tightness(base_multi_period_instance):
    """
    Test 3: Verify q_{i,h} == |u_{i,h}| holds for all h and assets.
    """
    state, params = base_multi_period_instance
    res = MultiPeriodMIQPOptimizer(params).solve(state)
    assert res.is_optimal

    H = params.horizon
    for h in range(H):
        np.testing.assert_allclose(res.absolute_trades[h], np.abs(res.trades[h]), atol=1e-7)


def test_4_self_financing_identity(base_multi_period_instance):
    """
    Test 4: Verify W_h^+ == W_h^- - kappa_h for all h.
    """
    state, params = base_multi_period_instance
    res = MultiPeriodMIQPOptimizer(params).solve(state)
    assert res.is_optimal

    H = params.horizon
    for h in range(H):
        w_minus = np.sum(res.pre_trade_holdings[h]) + res.pre_trade_cash[h]
        w_plus = np.sum(res.post_trade_holdings[h]) + res.post_trade_cash[h]
        assert np.isclose(w_plus, w_minus - res.transaction_costs[h], atol=1e-7)


def test_5_risky_holding_transition(base_multi_period_instance):
    """
    Test 5: Verify deterministic transition x_{h+1}^- == D_h * x_h for h = 0, ..., H-2.
    """
    state, params = base_multi_period_instance
    res = MultiPeriodMIQPOptimizer(params).solve(state)
    assert res.is_optimal

    H = params.horizon
    for h in range(H - 1):
        expected_x_next_minus = state.gross_returns[h] * res.post_trade_holdings[h]
        np.testing.assert_allclose(res.pre_trade_holdings[h + 1], expected_x_next_minus, atol=1e-7)


def test_6_cash_transition(base_multi_period_instance):
    """
    Test 6: Verify deterministic cash transition b_{h+1}^- == (1 + r_{f,h}) * b_h.
    """
    state, params = base_multi_period_instance
    res = MultiPeriodMIQPOptimizer(params).solve(state)
    assert res.is_optimal

    H = params.horizon
    for h in range(H - 1):
        rf_factor = 1.0 + params.risk_free_rates[h]
        expected_b_next_minus = rf_factor * res.post_trade_cash[h]
        assert np.isclose(res.pre_trade_cash[h + 1], expected_b_next_minus, atol=1e-7)


def test_7_dynamic_big_m_validity(base_multi_period_instance):
    """
    Test 7: Verify all trades satisfy q_{i,h} <= M_{i,h} under dynamic Big-M bounds.
    """
    state, params = base_multi_period_instance
    res = MultiPeriodMIQPOptimizer(params).solve(state)
    assert res.is_optimal

    m_symmetric = res.metadata["dynamic_big_m"]
    H = params.horizon
    for h in range(H):
        assert np.all(res.absolute_trades[h] <= m_symmetric[h] + 1e-7)


def test_8_per_period_cardinality(base_multi_period_instance):
    """
    Test 8: Verify sum_i y_{i,h} <= K_h holds for every period h independently.
    """
    state, params = base_multi_period_instance
    res = MultiPeriodMIQPOptimizer(params).solve(state)
    assert res.is_optimal

    H = params.horizon
    for h in range(H):
        total_held_h = np.sum(res.holding_indicators[h])
        assert total_held_h <= params.cardinality_limits[h]


def test_9_per_period_liquidity_and_turnover(base_multi_period_instance):
    """
    Test 9: Verify q_{i,h} <= L_{i,h} and sum_i q_{i,h} <= tau_h^max for all h.
    """
    state, params = base_multi_period_instance
    res = MultiPeriodMIQPOptimizer(params).solve(state)
    assert res.is_optimal

    H = params.horizon
    for h in range(H):
        assert np.all(res.absolute_trades[h] <= params.liquidity_limits[h] + 1e-7)
        assert np.sum(res.absolute_trades[h]) <= params.max_turnovers[h] + 1e-7


def test_10_sector_and_cash_constraints_across_horizon(base_multi_period_instance):
    """
    Test 10: Verify sector bounds and cash reserve b_h >= b_h^min hold across all h.
    """
    state, params = base_multi_period_instance
    res = MultiPeriodMIQPOptimizer(params).solve(state)
    assert res.is_optimal

    H = params.horizon
    for h in range(H):
        assert res.post_trade_cash[h] >= params.min_cash[h] - 1e-7
        for sector, idxs in params.sector_indices.items():
            e_min, e_max = params.sector_bounds[sector]
            sec_weight = np.sum(res.post_trade_holdings[h, idxs])
            assert sec_weight >= e_min - 1e-7
            assert sec_weight <= e_max + 1e-7


def test_11_zero_return_dynamic_consistency():
    """
    Test 11: Set gross returns g_h = 1.0 (0% return) and r_f = 0.
    Holdings propagate consistently when no rebalancing is needed.
    """
    H = 2
    n = 2
    state = MultiPeriodMarketState(
        initial_holdings=np.array([0.20, 0.20]),
        initial_cash=0.60,
        expected_returns=np.full((H, n), 0.05),
        covariance_matrices=np.tile(np.diag([0.04, 0.04]), (H, 1, 1)),
        gross_returns=np.ones((H - 1, n)),  # flat returns
    )
    # High fixed costs suppress unnecessary trading
    params = MultiPeriodPortfolioParameters(
        risk_aversion=2.0,
        horizon=H,
        risk_free_rates=np.zeros(H),
        max_turnovers=np.full(H, 0.30),
        min_cash=np.full(H, 0.10),
        weight_upper_limits=np.full((H, n), 0.50),
        liquidity_limits=np.full((H, n), 0.30),
        prop_cost_coeffs=np.full((H, n), 0.001),
        fixed_cost_coeffs=np.full((H, n), 0.05),  # High fixed charge
        cardinality_limits=np.full(H, 2),
    )

    res = MultiPeriodMIQPOptimizer(params).solve(state)
    assert res.is_optimal
    # In step 1, pre-trade holding equals step 0 post-trade holding
    np.testing.assert_allclose(res.pre_trade_holdings[1], res.post_trade_holdings[0], atol=1e-7)


def test_12_forced_future_rebalancing():
    """
    Test 12: Asset 0 experiences massive growth (g_{0,0} = 1.50) so x_{0,1}^- = 1.5 * x_{0,0}.
    If U_{0,1} = 0.30, the model is FORCED to execute a sell trade at h=1 (u_{0,1} < 0).
    """
    H = 2
    n = 2
    state = MultiPeriodMarketState(
        initial_holdings=np.array([0.25, 0.00]),
        initial_cash=0.75,
        expected_returns=np.array([[0.10, 0.05], [0.10, 0.05]]),
        covariance_matrices=np.tile(np.diag([0.04, 0.04]), (H, 1, 1)),
        gross_returns=np.array([[1.50, 1.00]]),  # 50% growth in Asset 0
    )
    params = MultiPeriodPortfolioParameters(
        risk_aversion=2.0,
        horizon=H,
        risk_free_rates=np.zeros(H),
        max_turnovers=np.full(H, 0.50),
        min_cash=np.full(H, 0.05),
        # Upper limit in h=1 is strictly 0.30
        weight_upper_limits=np.array([[0.40, 0.40], [0.30, 0.40]]),
        liquidity_limits=np.full((H, n), 0.40),
        prop_cost_coeffs=np.full((H, n), 0.001),
        min_holding_weights=np.full((H, n), 0.05),
    )

    res = MultiPeriodMIQPOptimizer(params).solve(state)
    assert res.is_optimal
    # Pre-trade holding at h=1
    x_1_minus = res.pre_trade_holdings[1, 0]
    # Post-trade holding at h=1 must respect U_{0,1} = 0.30
    x_1 = res.post_trade_holdings[1, 0]
    assert x_1 <= 0.30 + 1e-7
    # Forced sell trade occurred
    assert res.trades[1, 0] < -1e-4


def test_13_multi_period_infeasibility_detection():
    """
    Test 13: Multi-period dynamic infeasibility (liquidity trap).
    Initial holding in Asset 0 is 0.40.
    At h=1, regulatory upper limit tightens to U_{0,1} = 0.10.
    However, liquidity limits are L_{0,0} = 0.10 and L_{0,1} = 0.10.
    Even selling maximum allowable liquidity at h=0 (leaving 0.30)
    and at h=1 (leaving 0.20), the portfolio cannot reach U_{0,1} = 0.10.
    Ensures Gurobi safely detects and reports INFEASIBLE.
    """
    H = 2
    n = 2
    state = MultiPeriodMarketState(
        initial_holdings=np.array([0.40, 0.00]),
        initial_cash=0.60,
        expected_returns=np.full((H, n), 0.05),
        covariance_matrices=np.tile(np.diag([0.04, 0.04]), (H, 1, 1)),
        gross_returns=np.ones((H - 1, n)),  # flat returns
    )
    params = MultiPeriodPortfolioParameters(
        risk_aversion=2.0,
        horizon=H,
        risk_free_rates=np.zeros(H),
        max_turnovers=np.full(H, 0.50),
        min_cash=np.full(H, 0.05),
        # Tightened limit at h=1: 0.10
        weight_upper_limits=np.array([[0.50, 0.50], [0.10, 0.50]]),
        # Liquidity limit per period is only 0.10
        liquidity_limits=np.array([[0.10, 0.50], [0.10, 0.50]]),
        prop_cost_coeffs=np.full((H, n), 0.001),
    )

    res = MultiPeriodMIQPOptimizer(params).solve(state)
    assert not res.is_optimal
    assert not res.is_feasible
    assert res.status == "INFEASIBLE"


def test_14_psd_validation_across_horizon():
    """
    Test 14: If any Sigma_h has a negative eigenvalue, reject with ValueError.
    """
    H = 2
    n = 2
    non_psd_cov = np.array([[0.04, 0.10], [0.10, 0.04]])
    covariances = np.array([np.diag([0.04, 0.04]), non_psd_cov])

    state = MultiPeriodMarketState(
        initial_holdings=np.array([0.10, 0.10]),
        initial_cash=0.80,
        expected_returns=np.full((H, n), 0.05),
        covariance_matrices=covariances,
        gross_returns=np.ones((H - 1, n)),
    )

    with pytest.raises(ValueError, match="is not PSD"):
        state.validate()
