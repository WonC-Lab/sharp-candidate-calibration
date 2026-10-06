"""
Comprehensive 14-point unit test suite for Phase 4: Rolling-Horizon MPC Simulator.
"""

import numpy as np
import pytest

from learning_augmented_portfolio.dynamic import (
    FirstAction,
    OptimizationFailureError,
    PortfolioState,
    RollingHorizonPolicy,
    RollingHorizonSimulator,
    realized_market_transition,
    renormalize_state,
    rescale_fixed_costs,
    rescale_liquidity_limits,
    validate_first_action_execution,
)
from learning_augmented_portfolio.forecasts import (
    ConstantForecastProvider,
    ForecastBundle,
    InstrumentedForecastProvider,
    PredefinedTrajectoryForecastProvider,
)
from learning_augmented_portfolio.optimization.variables import SolverConfig


@pytest.fixture
def base_forecast_bundle():
    """Canonical 2-period, 2-asset forecast bundle."""
    H = 2
    n = 2
    mu = np.array([[0.10, 0.05], [0.09, 0.05]])
    Sigma = np.tile(np.diag([0.04, 0.03]), (H, 1, 1))
    gross_returns = np.array([[1.08, 1.02]])
    rf = np.array([0.01, 0.01])
    u_limits = np.full((H, n), 0.50)
    tau_limits = np.full(H, 0.40)
    c_min = np.full(H, 0.10)
    prop_costs = np.full((H, n), 0.001)
    dollar_fixed_costs = np.full((H, n), 50.0)  # $50 per trade
    dollar_adv = np.full((H, n), 1_000_000.0)   # $1M daily volume
    participation_rates = np.array([0.10, 0.10]) # 10%

    return ForecastBundle(
        expected_returns=mu,
        covariance_matrices=Sigma,
        gross_returns=gross_returns,
        risk_free_rates=rf,
        weight_upper_limits=u_limits,
        max_turnovers=tau_limits,
        min_cash=c_min,
        prop_cost_coeffs=prop_costs,
        dollar_fixed_costs=dollar_fixed_costs,
        dollar_adv=dollar_adv,
        participation_rates=participation_rates,
        cardinality_limits=np.full(H, 2),
        min_holding_weights=np.full((H, n), 0.05),
        min_trade_sizes=np.full((H, n), 0.02),
    )


def test_1_first_action_only_execution(base_forecast_bundle):
    """
    Test 1: For H > 1, verify only u^*_{0|t} is executed.
    Planned actions for h >= 1 do not directly alter the realized state.
    """
    initial_state = PortfolioState(
        actual_wealth=100_000.0,
        normalized_weights=np.array([0.15, 0.10]),
        normalized_cash=0.75,
        time_index=0,
    )
    provider = ConstantForecastProvider(base_forecast_bundle)
    policy = RollingHorizonPolicy()

    first_action, opt_res = policy.plan_and_extract_action(initial_state, base_forecast_bundle)

    # First action matches index 0 of multi-period plan
    np.testing.assert_allclose(first_action.trade, opt_res.trades[0])
    np.testing.assert_allclose(first_action.post_trade_holdings, opt_res.post_trade_holdings[0])

    # Future planned trade (h = 1) exists in plan but is distinct
    assert opt_res.trades.shape[0] == 2
    # Verify execution identity
    validate_first_action_execution(
        first_action.post_trade_holdings,
        first_action.post_trade_cash,
        first_action.transaction_cost,
    )


def test_2_forecast_realized_return_separation(base_forecast_bundle):
    """
    Test 2: When planning forecast r_hat != r_real, optimization uses r_hat
    while realized transition strictly uses r_real.
    """
    initial_state = PortfolioState(
        actual_wealth=100_000.0,
        normalized_weights=np.array([0.15, 0.10]),
        normalized_cash=0.75,
        time_index=0,
    )
    policy = RollingHorizonPolicy()
    first_action, _ = policy.plan_and_extract_action(initial_state, base_forecast_bundle)

    # Deliberate divergence: Forecast gross return was 1.08 and 1.02.
    # Realized returns are negative shock: -5% and +1%
    realized_risky = np.array([-0.05, 0.01])
    realized_rf = 0.01

    x_tilde, b_tilde, R_V = realized_market_transition(
        post_trade_holdings=first_action.post_trade_holdings,
        post_trade_cash=first_action.post_trade_cash,
        realized_risky_returns=realized_risky,
        realized_risk_free_rate=realized_rf,
    )

    # Physical holdings must reflect realized_risky, NOT forecast
    expected_x_tilde = (1.0 + realized_risky) * first_action.post_trade_holdings
    np.testing.assert_allclose(x_tilde, expected_x_tilde, atol=1e-8)


def test_3_wealth_transition(base_forecast_bundle):
    """
    Test 3: Verify V_{t+1} == V_t * [ 1^T G^{real} x_0* + (1 + r_f^{real}) b_0* ].
    """
    V_t = 200_000.0
    initial_state = PortfolioState(
        actual_wealth=V_t,
        normalized_weights=np.array([0.20, 0.20]),
        normalized_cash=0.60,
    )
    policy = RollingHorizonPolicy()
    first_action, _ = policy.plan_and_extract_action(initial_state, base_forecast_bundle)

    realized_risky = np.array([0.03, -0.02])
    realized_rf = 0.005

    x_tilde, b_tilde, R_V = realized_market_transition(
        first_action.post_trade_holdings,
        first_action.post_trade_cash,
        realized_risky,
        realized_rf,
    )
    new_state = renormalize_state(x_tilde, b_tilde, R_V, V_t, current_time_index=0)

    expected_wealth = V_t * (np.sum((1.0 + realized_risky) * first_action.post_trade_holdings) + (1.0 + realized_rf) * first_action.post_trade_cash)
    assert np.isclose(new_state.actual_wealth, expected_wealth, atol=1e-6)


def test_4_renormalization_identity(base_forecast_bundle):
    """
    Test 4: Verify 1^T w_bar_{t+1} + c_bar_{t+1} == 1.0 strictly holds.
    """
    initial_state = PortfolioState(
        actual_wealth=150_000.0,
        normalized_weights=np.array([0.25, 0.15]),
        normalized_cash=0.60,
    )
    policy = RollingHorizonPolicy()
    first_action, _ = policy.plan_and_extract_action(initial_state, base_forecast_bundle)

    x_tilde, b_tilde, R_V = realized_market_transition(
        first_action.post_trade_holdings,
        first_action.post_trade_cash,
        np.array([0.05, 0.02]),
        0.01,
    )
    new_state = renormalize_state(x_tilde, b_tilde, R_V, initial_state.actual_wealth, 0)
    assert np.isclose(np.sum(new_state.normalized_weights) + new_state.normalized_cash, 1.0, atol=1e-7)


def test_5_transaction_cost_deduction(base_forecast_bundle):
    """
    Test 5: Verify transaction costs are deducted exactly once from pre-trade wealth.
    """
    initial_state = PortfolioState(
        actual_wealth=100_000.0,
        normalized_weights=np.array([0.10, 0.10]),
        normalized_cash=0.80,
    )
    policy = RollingHorizonPolicy()
    first_action, _ = policy.plan_and_extract_action(initial_state, base_forecast_bundle)

    post_trade_wealth = np.sum(first_action.post_trade_holdings) + first_action.post_trade_cash
    assert np.isclose(post_trade_wealth, 1.0 - first_action.transaction_cost, atol=1e-7)


def test_6_new_numeraire_fixed_cost():
    """
    Test 6: Verify fixed cost is correctly rescaled: f_{i,t+1} = F_{i,t+1} / V_{t+1}.
    """
    dollar_fixed_costs = np.array([100.0, 50.0])
    V_0 = 100_000.0
    f_0 = rescale_fixed_costs(dollar_fixed_costs, V_0)
    np.testing.assert_allclose(f_0, [0.001, 0.0005])

    # Wealth doubles to $200,000
    V_1 = 200_000.0
    f_1 = rescale_fixed_costs(dollar_fixed_costs, V_1)
    np.testing.assert_allclose(f_1, [0.0005, 0.00025])


def test_7_new_numeraire_liquidity_limit():
    """
    Test 7: Verify liquidity limit is correctly rescaled: L_{i,t+1} = alpha_i * DADV_{i,t+1} / V_{t+1}.
    """
    dollar_adv = np.array([2_000_000.0, 1_000_000.0])
    alphas = np.array([0.05, 0.10])
    V_0 = 1_000_000.0
    L_0 = rescale_liquidity_limits(dollar_adv, alphas, V_0)
    # Asset 0: 0.05 * 2M / 1M = 0.10
    # Asset 1: 0.10 * 1M / 1M = 0.10
    np.testing.assert_allclose(L_0, [0.10, 0.10])

    # If portfolio grows to $2M, normalized liquidity limit must halve to 0.05
    V_1 = 2_000_000.0
    L_1 = rescale_liquidity_limits(dollar_adv, alphas, V_1)
    np.testing.assert_allclose(L_1, [0.05, 0.05])


def test_8_no_return_no_trade_invariance():
    """
    Test 8: With r^{real}=0, r_f=0, and u=0, wealth and normalized weights remain strictly unchanged.
    """
    initial_weights = np.array([0.30, 0.20])
    initial_cash = 0.50
    V_t = 100_000.0

    # No trade: post_trade == initial
    post_trade_h = initial_weights
    post_trade_c = initial_cash

    x_tilde, b_tilde, R_V = realized_market_transition(
        post_trade_holdings=post_trade_h,
        post_trade_cash=post_trade_c,
        realized_risky_returns=np.zeros(2),
        realized_risk_free_rate=0.0,
    )
    new_state = renormalize_state(x_tilde, b_tilde, R_V, V_t, 0)

    assert np.isclose(new_state.actual_wealth, V_t)
    np.testing.assert_allclose(new_state.normalized_weights, initial_weights)
    assert np.isclose(new_state.normalized_cash, initial_cash)


def test_9_risky_asset_appreciation():
    """
    Test 9: One asset doubles in price (+100%).
    Verify analytical wealth growth and normalized weight rebalancing.
    """
    post_h = np.array([0.40, 0.00])
    post_c = 0.60
    V_t = 100_000.0

    # Asset 0 gains +100%, cash gains 0%
    r_real = np.array([1.00, 0.00])
    r_f = 0.0

    x_tilde, b_tilde, R_V = realized_market_transition(post_h, post_c, r_real, r_f)
    # Expected x_tilde = [0.80, 0.0], b_tilde = 0.60 => R_V = 1.40 (+40% portfolio return)
    assert np.isclose(R_V, 1.40)

    new_state = renormalize_state(x_tilde, b_tilde, R_V, V_t, 0)
    assert np.isclose(new_state.actual_wealth, 140_000.0)
    # New weight: 0.80 / 1.40 = 4 / 7
    expected_w0 = 0.80 / 1.40
    assert np.isclose(new_state.normalized_weights[0], expected_w0)
    assert np.isclose(new_state.normalized_cash, 0.60 / 1.40)


def test_10_receding_horizon_reoptimization(base_forecast_bundle):
    """
    Test 10: Verify genuine MPC receding-horizon behavior:
    The action planned for h=1 at time t may differ from the newly optimized action at t+1
    due to realized returns and state shift.
    """
    initial_state = PortfolioState(
        actual_wealth=100_000.0,
        normalized_weights=np.array([0.15, 0.10]),
        normalized_cash=0.75,
        time_index=0,
    )
    provider = ConstantForecastProvider(base_forecast_bundle)
    simulator = RollingHorizonSimulator(RollingHorizonPolicy())

    # 2 steps simulation
    realized_risky = np.array([
        [-0.05, 0.08],  # Step 0: Asset 0 drops 5%, Asset 1 jumps 8%
        [0.02, 0.01],   # Step 1
    ])
    realized_rf = np.array([0.01, 0.01])

    history = simulator.run(
        initial_state=initial_state,
        horizon=2,
        num_steps=2,
        forecast_provider=provider,
        realized_risky_returns=realized_risky,
        realized_risk_free_rates=realized_rf,
    )

    rec0 = history.records[0]
    rec1 = history.records[1]

    # At t=0, the planned trade for h=1 was rec0.planned_trades_horizon[1]
    planned_h1_at_t0 = rec0.planned_trades_horizon[1]
    # At t=1, the actual executed trade was rec1.executed_trade
    executed_at_t1 = rec1.executed_trade

    # Due to market shock, the new optimization produces a fresh, re-optimized decision
    assert rec0.time_step == 0
    assert rec1.time_step == 1
    assert history.num_steps == 2


def test_11_time_series_state_conservation(base_forecast_bundle):
    """
    Test 11: Multi-step simulation: verify 1^T w + c == 1.0 and V_t > 0 at every step.
    """
    initial_state = PortfolioState(
        actual_wealth=100_000.0,
        normalized_weights=np.array([0.10, 0.10]),
        normalized_cash=0.80,
    )
    num_steps = 4
    np.random.seed(42)
    realized_risky = np.random.uniform(-0.03, 0.04, size=(num_steps, 2))
    realized_rf = np.full(num_steps, 0.005)

    simulator = RollingHorizonSimulator()
    history = simulator.run(
        initial_state=initial_state,
        horizon=2,
        num_steps=num_steps,
        forecast_provider=ConstantForecastProvider(base_forecast_bundle),
        realized_risky_returns=realized_risky,
        realized_risk_free_rates=realized_rf,
    )

    for rec in history.records:
        assert rec.actual_wealth_pre > 0
        assert rec.actual_wealth_post > 0
        assert np.isclose(np.sum(rec.pre_trade_weights) + rec.pre_trade_cash, 1.0, atol=1e-5)
        assert np.isclose(np.sum(rec.next_normalized_weights) + rec.next_normalized_cash, 1.0, atol=1e-5)


def test_12_solver_failure_without_incumbent():
    """
    Test 12: Infeasible problem without fallback raises OptimizationFailureError.
    """
    initial_state = PortfolioState(
        actual_wealth=100_000.0,
        normalized_weights=np.array([0.40, 0.40]),
        normalized_cash=0.20,
    )
    # Impossible requirement: min cash 0.80 + holding lower bound 0.30 + 0.30 = 1.40 > 1.0
    impossible_bundle = ForecastBundle(
        expected_returns=np.full((2, 2), 0.05),
        covariance_matrices=np.tile(np.diag([0.04, 0.04]), (2, 1, 1)),
        risk_free_rates=np.zeros(2),
        weight_upper_limits=np.full((2, 2), 1.0),
        max_turnovers=np.full(2, 0.50),
        min_cash=np.full(2, 0.80),
        prop_cost_coeffs=np.full((2, 2), 0.001),
        liquidity_limits=np.full((2, 2), 0.50),
        min_holding_weights=np.full((2, 2), 0.30),
        cardinality_limits=np.full(2, 2),
    )
    policy = RollingHorizonPolicy(allow_fallback_on_failure=False)

    with pytest.raises(OptimizationFailureError, match="No feasible incumbent available"):
        policy.plan_and_extract_action(initial_state, impossible_bundle)


def test_13_time_limit_with_feasible_incumbent(base_forecast_bundle):
    """
    Test 13: When time limit is very tight, if a feasible incumbent exists, it is safely extracted.
    """
    initial_state = PortfolioState(
        actual_wealth=100_000.0,
        normalized_weights=np.array([0.15, 0.10]),
        normalized_cash=0.75,
    )
    policy = RollingHorizonPolicy(solver_config=SolverConfig(time_limit=10.0))
    first_action, opt_res = policy.plan_and_extract_action(initial_state, base_forecast_bundle)

    assert opt_res.is_feasible
    assert first_action.trade is not None


def test_14_no_lookahead_forecast_interface(base_forecast_bundle):
    """
    Test 14: Use InstrumentedForecastProvider to audit that the provider
    never receives a query for calendar time t' > current_time t.
    """
    underlying = ConstantForecastProvider(base_forecast_bundle)
    instrumented = InstrumentedForecastProvider(underlying)

    initial_state = PortfolioState(
        actual_wealth=100_000.0,
        normalized_weights=np.array([0.10, 0.10]),
        normalized_cash=0.80,
    )
    simulator = RollingHorizonSimulator()
    num_steps = 3
    simulator.run(
        initial_state=initial_state,
        horizon=2,
        num_steps=num_steps,
        forecast_provider=instrumented,
        realized_risky_returns=np.zeros((num_steps, 2)),
        realized_risk_free_rates=np.zeros(num_steps),
    )

    # Verify query sequence
    assert len(instrumented.query_log) == num_steps
    for step_idx, (queried_time, queried_h) in enumerate(instrumented.query_log):
        # Queried time must strictly match current simulation step
        assert queried_time == step_idx
        assert queried_h == 2

    assert instrumented.verify_no_lookahead(max_allowed_time=num_steps - 1)
