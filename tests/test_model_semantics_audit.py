"""
Regression Tests for Model Semantics Audit and Time-Scale Consistency.
Audits:
  1. Feasibility of micro-trades (0 < |u| < delta) under the implemented q/u/z constraints;
  2. Proof that q = max(|u|, delta*z) differs from |u| whenever 0 < |u| < delta;
  3. Synchronization between manuscript/manifest delta and executed experiment delta;
  4. Time-scale conversion of annual risk-free rate to weekly/period frequency;
  5. Mutual frequency consistency among expected returns, covariance, and transition returns.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import gurobipy as gp
from gurobipy import GRB

from learning_augmented_portfolio.optimization.variables import (
    PortfolioParameters,
    MultiPeriodPortfolioParameters,
    SolverConfig,
    MultiPeriodMarketState,
)
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from learning_augmented_portfolio.scalability.instance_generator import ScalableInstanceGenerator
from learning_augmented_portfolio.backtest.engine import RealMarketBacktestEngine
from learning_augmented_portfolio.data import MarketDataStore


def test_implemented_constraints_permit_micro_trades():
    """
    Test 1: Verify that the implemented constraints:
        q >= u
        q >= -u
        delta * z <= q <= M * z
        z in {0, 1}
    strictly permit 0 < |u| < delta (e.g. u=0.005, delta=0.01, z=1, q=0.01).
    A claimed model requiring |u| in {0} union [delta, M] would reject this.
    """
    u_val = 0.005
    delta_val = 0.01
    z_val = 1
    q_val = 0.01
    m_val = 0.50

    # 1. Direct mathematical verification of feasibility
    assert q_val >= u_val, "q >= u must hold"
    assert q_val >= -u_val, "q >= -u must hold"
    assert q_val >= delta_val * z_val, "q >= delta * z must hold"
    assert q_val <= m_val * z_val, "q <= M * z must hold"
    assert z_val in (0, 1), "z must be binary"

    # Micro-trade condition holds
    assert 0.0 < abs(u_val) < delta_val, "u is strictly inside (0, delta)"
    assert q_val != abs(u_val), "q cannot equal |u| when q is bounded by delta"

    # 2. Solver verification in a standalone MIP
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(env=env) as m:
            u = m.addVar(lb=-1.0, ub=1.0, name="u")
            q = m.addVar(lb=0.0, ub=1.0, name="q")
            z = m.addVar(vtype=GRB.BINARY, name="z")

            # Fix u to 0.005
            m.addConstr(u == u_val)
            # Implemented constraints
            m.addConstr(q - u >= 0.0)
            m.addConstr(q + u >= 0.0)
            m.addConstr(q >= delta_val * z)
            m.addConstr(q <= m_val * z)

            # Minimize transaction charge q
            m.setObjective(q, GRB.MINIMIZE)
            m.optimize()

            assert m.Status == GRB.OPTIMAL
            assert z.X == 1.0, "z must be 1 to permit non-zero trade u"
            assert q.X == pytest.approx(delta_val, abs=1e-8), "q must equal delta, not |u|"
            assert abs(q.X - abs(u.X)) == pytest.approx(delta_val - u_val, abs=1e-8)


def test_q_differs_from_abs_u_under_micro_trades():
    """
    Test 2: Verify that q = max(|u|, delta*z) holds exactly, and that
    q > |u| strictly occurs when |u| < delta. Any unconditional claim that q = |u|
    fails.
    """
    H = 1
    N = 2
    delta_val = 0.02
    u_cap = 0.40

    params = MultiPeriodPortfolioParameters(
        risk_aversion=5.0,
        horizon=H,
        risk_free_rates=np.array([0.0001]),
        max_turnovers=np.array([0.40]),
        min_cash=np.array([0.10]),
        weight_upper_limits=np.full((H, N), u_cap),
        liquidity_limits=np.full((H, N), 0.30),
        prop_cost_coeffs=np.full((H, N), 0.0005),
        discount_factor=1.0,
        sector_indices={},
        sector_bounds={},
        cardinality_limits=np.array([2]),
        min_holding_weights=np.full((H, N), 0.05),
        min_trade_sizes=np.full((H, N), delta_val),
        fixed_cost_coeffs=np.full((H, N), 0.00001),
        allow_degenerate=True,
    )

    # State where asset 0 is at 0.20 and cash is 0.80.
    # Set mu and covariance such that optimal unconstrained x0 is ~ 0.205, inducing micro-trade u0 ~ 0.005 < delta=0.02
    cov = np.eye(N) * 0.04
    # With gamma=5, Sigma=0.04*I, marginal return 0.04125 gives optimal weight around 0.205
    mu = np.array([[0.04125, 0.01]])
    gross = np.ones((1, N))

    mkt = MultiPeriodMarketState(
        initial_holdings=np.array([0.20, 0.0]),
        initial_cash=0.80,
        expected_returns=mu,
        covariance_matrices=cov[np.newaxis, :, :],
        gross_returns=gross,
        asset_names=["A0", "A1"],
    )

    opt = MultiPeriodMIQPOptimizer(params, SolverConfig(time_limit=5.0, mip_gap=1e-5, threads=1))
    res = opt.solve(mkt)

    assert res.is_feasible
    u_sol = res.trades[0]
    q_sol = res.absolute_trades[0]
    z_sol = res.trade_indicators[0]

    # Check the mathematical envelope condition for every asset
    for i in range(N):
        expected_q = max(abs(u_sol[i]), delta_val * z_sol[i])
        assert q_sol[i] == pytest.approx(expected_q, abs=1e-6)

        if 1e-5 < abs(u_sol[i]) < delta_val - 1e-5:
            # Micro-trade present: q must strictly exceed |u|
            assert q_sol[i] > abs(u_sol[i]) + 1e-4
            assert q_sol[i] == pytest.approx(delta_val, abs=1e-6)


def test_appendix_delta_matches_executed_delta():
    """
    Test 3: Verify that the delta values declared in the manuscript and reproducibility manifest
    match the executed code configurations:
      - 0.01 for synthetic scalability instances (Table B, Table C, ScalableInstanceGenerator);
      - 0.02 for the dynamic benchmark dataset and 2024 real-market backtest.
    """
    manifest_path = Path("artifacts/final_reproducibility_manifest.json")
    assert manifest_path.exists(), "Reproducibility manifest must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Manifest canonical value
    canonical_delta = data["canonical_portfolio_parameters"]["canonical_min_trade_delta_i"]
    assert canonical_delta == 0.01, f"Manifest canonical delta must be 0.01, got {canonical_delta}"

    # 2. ScalableInstanceGenerator delta
    gen = ScalableInstanceGenerator()
    inst_synth = gen.generate_instance(n_assets=10, horizon=3, cardinality_limit=3, seed=42)
    assert inst_synth.params.min_trade_sizes is not None
    assert np.allclose(inst_synth.params.min_trade_sizes, 0.01), "Scalability instances must use delta = 0.01"

    # 3. Dynamic dataset saved instances delta
    sample_npz = Path("artifacts/instances/train/instance_SP500_SUB5_0000.npz")
    if sample_npz.exists():
        npz_data = np.load(sample_npz)
        assert np.allclose(npz_data["min_trade_sizes"], 0.02), "Dynamic dataset instances must use delta = 0.02"

    # 4. Real market training instances delta
    real_sample_npz = Path("artifacts/real_instances/instance_real_train_000_2021-03-29.npz")
    if real_sample_npz.exists():
        real_data = np.load(real_sample_npz)
        assert np.allclose(real_data["min_trade_sizes"], 0.02), "Real market instances must use delta = 0.02"


def test_annual_rf_not_used_directly_as_weekly_return():
    """
    Test 4: Verify that annual risk-free rate (e.g. 2% or 3%) is properly converted
    to periodic frequency and NEVER passed directly as a weekly step rate (e.g., > 1%).
    """
    # 1. Backtest engine period rate conversion
    store = MarketDataStore.load_csv("artifacts/real_market/daily_prices.csv")
    single_p = PortfolioParameters(
        risk_aversion=5.0,
        risk_free_rate=0.0001,
        max_turnover=0.40,
        min_cash=0.10,
        weight_upper_limits=np.full(10, 0.40),
        liquidity_limits=np.full(10, 0.20),
        prop_cost_coeffs=np.full(10, 0.0005),
        cardinality_limit=4,
        min_holding_weights=np.full(10, 0.08),
        min_trade_sizes=np.full(10, 0.02),
        fixed_cost_coeffs=np.full(10, 0.00001),
    )
    base_params = MultiPeriodPortfolioParameters.from_single_period(single_p, horizon=3)

    engine = RealMarketBacktestEngine(
        data_store=store,
        params=base_params,
        rebalance_frequency_days=5,
        risk_free_rate_annual=0.03,
    )

    # 5-day compounding rate must be around 0.03 * (5/252) ~ 0.000595, NOT 0.03
    expected_rf_5d = (1.0 + 0.03) ** (5.0 / 252.0) - 1.0
    assert engine.r_f_period == pytest.approx(expected_rf_5d, rel=1e-4)
    assert engine.r_f_period < 0.001, "Weekly r_f must be < 0.1%, never annual 3%"

    # 2. Scalable instance generator rate conversion
    gen = ScalableInstanceGenerator()
    inst = gen.generate_instance(n_assets=10, horizon=3, cardinality_limit=3, seed=42)
    rf_synth = inst.params.risk_free_rates[0]
    dt_synth = 5.0 / 252.0
    assert rf_synth == pytest.approx(0.03 * dt_synth, rel=1e-4)
    assert rf_synth < 0.001, "Scaled r_f must be on periodic scale, never raw annual"


def test_time_scale_consistency_returns_covariance_transitions():
    """
    Test 5: Verify frequency consistency among expected returns, covariance matrices,
    and transition gross returns.
    """
    # 1. Scalable instance generator consistency
    gen = ScalableInstanceGenerator()
    inst = gen.generate_instance(n_assets=10, horizon=3, cardinality_limit=3, seed=42)
    mkt = inst.market_state

    # Daily factor vol is scaled by sqrt(5/252), so period variance is O(dt)
    dt = 5.0 / 252.0
    mean_diag_cov = np.mean(np.diagonal(mkt.covariance_matrices[0]))
    # Annualized vol typically ~ 20-30% -> annual var ~ 0.04-0.09 -> weekly var ~ 0.0008-0.0018
    assert 0.0001 < mean_diag_cov < 0.01, f"Weekly variance must be O(dt), got {mean_diag_cov}"

    # Mean return is base_annual * dt ~ 0.08 * (5/252) ~ 0.0016
    mean_mu = np.mean(np.abs(mkt.expected_returns[0]))
    assert 0.0001 < mean_mu < 0.02, f"Weekly expected return must be O(dt), got {mean_mu}"

    # Gross return definition: g = 1 + mu
    expected_gross = 1.0 + mkt.expected_returns[:-1]
    assert np.allclose(mkt.gross_returns, expected_gross, atol=1e-8), "Gross returns must strictly equal 1 + mu"
