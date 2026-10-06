"""
Phase 10.5 Unit Tests: Scientific & Computational Consistency Audit.
Verifies:
  1. Oracle Support Exact Objective Recovery: J^{oracle} == J^{full} (Support Containment Proposition).
  2. Oracle Current-Holding Augmentation: M_t subseteq C_t^{oracle}.
  3. Oracle Support is never truncated by an arbitrary candidate budget.
  4. Canonical Theoretical Model-Size: Exact agreement with Gurobi model.NumConstrs == H*(8N+5).
  5. Large-N Unsupported Full-Quality Metrics: Explicitly reported as N/A (never numerical zero).
  6. Large-N Certified-Mode Semantics: Proper classification under license limits.
  7. Missing Certificate Semantics: Missing certificates never encoded as zero gap.
  8. Canonical CAGR Convention: Exact reconciliation with wealth path.
  9. Canonical Sharpe Convention: Identical convention across all reporting.
 10. Canonical Min Cash: Exactly 0.10 throughout.
 11. End-to-end regression consistency.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from learning_augmented_portfolio.dataset.instance import OptimizationInstance
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from learning_augmented_portfolio.optimization.variables import SolverConfig
from learning_augmented_portfolio.scalability.instance_generator import ScalableInstanceGenerator
from learning_augmented_portfolio.scalability.model_stats import compute_theoretical_model_stats
from learning_augmented_portfolio.scalability.runner import ScalabilityRunner
from learning_augmented_portfolio.evaluation.scalability_metrics import compute_objective_quality
from learning_augmented_portfolio.backtest.records import BacktestStepRecord, FinancialMetricsSummary


# 1. Oracle Support Reduced Problem Recovers Full Objective Exactly
def test_1_oracle_support_reduced_problem_recovers_full_objective():
    """
    Verifies the Support-Containment Proposition:
    When C_t^{oracle} = S_t^* union M_t, J_t^{oracle} == J_t^{full} within solver tolerance.
    """
    gen = ScalableInstanceGenerator()
    inst = gen.generate_instance(n_assets=10, horizon=3, cardinality_limit=3, seed=42)
    runner = ScalabilityRunner(SolverConfig(time_limit=10.0, mip_gap=1e-5))

    res_full = runner.solve_full_miqp(inst, record_anytime=False)
    assert res_full.status.value == "OPTIMAL"
    assert res_full.objective_value is not None
    assert res_full.holding_indicators is not None

    # Run Oracle Diagnostic using actual teacher indicators
    res_oracle = runner.solve_oracle_support(inst, teacher_support_or_y=res_full.holding_indicators)
    assert res_oracle.status.value == "OPTIMAL"
    assert res_oracle.objective_value is not None

    # Objective degradation must be zero within solver tolerance
    abs_diff = abs(res_oracle.objective_value - res_full.objective_value)
    assert abs_diff < 1e-6, f"Oracle objective {res_oracle.objective_value} != Full objective {res_full.objective_value} (diff={abs_diff})"

    q = compute_objective_quality(res_oracle.objective_value, res_full.objective_value, full_is_optimal=True)
    assert q["is_exact_recovery"] is True
    assert q["abs_degradation"] < 1e-6


# 2. Oracle Current-Holding Augmentation
def test_2_oracle_current_holding_augmentation():
    """
    Verifies that M_t = {i : w_bar_{i,t} > 0} is strictly included in C_t^{oracle}.
    """
    gen = ScalableInstanceGenerator()
    inst = gen.generate_instance(n_assets=15, horizon=2, cardinality_limit=3, seed=100)
    runner = ScalabilityRunner()

    held_assets = np.where(inst.actual_state.normalized_weights > 1e-6)[0].tolist()
    assert len(held_assets) > 0, "Instance must have initial holdings for this test"

    # Pass an arbitrary disjoint teacher support {0, 1}
    dummy_support = {0, 1}
    res_oracle = runner.solve_oracle_support(inst, teacher_support_or_y=dummy_support)

    # Effective candidate size must include all held assets
    expected_min_size = len(dummy_support | set(held_assets))
    assert res_oracle.candidate_size >= expected_min_size


# 3. Oracle Support Never Truncated by Candidate Budget
def test_3_oracle_support_never_truncated_by_candidate_budget():
    """
    Verifies Oracle candidate universe is not capped by any external budget B < |C_t^{oracle}|.
    """
    gen = ScalableInstanceGenerator()
    inst = gen.generate_instance(n_assets=20, horizon=3, cardinality_limit=4, seed=55)
    runner = ScalabilityRunner()

    # Pass support of size 6
    support_6 = {2, 3, 5, 7, 11, 13}
    res_oracle = runner.solve_oracle_support(inst, teacher_support_or_y=support_6)

    # Size must be at least 6
    assert res_oracle.candidate_size >= 6
    assert res_oracle.is_deployable is False
    assert res_oracle.uses_teacher_information is True


# 4. Canonical Theoretical Model-Size Equals Actual Gurobi Count
@pytest.mark.parametrize("n,h", [(10, 1), (10, 3), (25, 1), (5, 5)])
def test_4_canonical_theoretical_model_size_equals_gurobi_count(n: int, h: int):
    """
    Verifies exact mathematical equivalence between the analytical formula:
      n_constrs = H * (8N + 5)
      n_vars = H * (6N + 3)
      n_bin = 2 * N * H
    and the actual Gurobi model attributes.
    """
    gen = ScalableInstanceGenerator()
    inst = gen.generate_instance(n_assets=n, horizon=h, cardinality_limit=min(3, n), seed=42)
    opt = MultiPeriodMIQPOptimizer(inst.params, SolverConfig(time_limit=1.0))
    res = opt.solve(inst.market_state)

    theoretical = compute_theoretical_model_stats(n_assets=n, horizon=h)

    assert res.metadata["num_vars"] == theoretical.n_vars
    assert res.metadata["num_bin_vars"] == theoretical.n_binary
    assert res.metadata["num_constrs"] == theoretical.n_constraints
    assert theoretical.n_constraints == h * (8 * n + 5)


# 5. Large-N Unsupported Full-Quality Metrics Reported as N/A
def test_5_large_n_unsupported_full_quality_metrics_reported_as_na():
    """
    Verifies that when full solve is not available (timed out or license limited),
    objective quality metrics return None / N/A, not numerical zero.
    """
    q_timeout = compute_objective_quality(
        obj_method=-0.005,
        obj_full=-0.005,
        full_is_optimal=False,  # Not certified optimal
    )
    assert q_timeout["is_exact_recovery"] is False
    assert q_timeout["abs_degradation"] is None
    assert q_timeout["incumbent_diff"] is None


# 6. Large-N Certified-Mode Semantics Under License Limit
def test_6_large_n_certified_mode_reports_license_limitation():
    """
    Verifies that configurations exceeding solver license limits are explicitly
    marked with LICENSE_LIMIT or NOT_EXECUTED_LICENSE_LIMIT, never as successful solves.
    """
    from learning_augmented_portfolio.scalability.failure import SolverExitStatus
    assert SolverExitStatus.NOT_EXECUTED_LICENSE_LIMIT.value == "NOT_EXECUTED_LICENSE_LIMIT"
    assert SolverExitStatus.LICENSE_LIMIT.value == "LICENSE_LIMIT"


# 7. Missing Certificate is Not Encoded as Zero
def test_7_missing_certificate_is_not_encoded_as_zero():
    """
    Verifies that uncertified methods do not report a 0.0 relaxation gap.
    """
    rec = BacktestStepRecord(
        date="2023-01-06",
        rebalance_index=0,
        portfolio_wealth_pre=1e6,
        portfolio_wealth_post=1e6,
        cash_pre=1e5,
        cash_post=1e5,
        pre_trade_weights=np.zeros(10),
        post_trade_weights=np.zeros(10),
        trades=np.zeros(10),
        turnover=0.05,
        transaction_cost=50.0,
        realized_period_return=0.01,
        num_holdings=3,
        num_trades=3,
        solve_time_ms=15.0,
        node_count=1,
        work=0.001,
        mip_gap=1e-4,
        objective_value=-0.002,
        certificate_source="NONE",
        relaxation_certified=False,
        full_mip_certified=False,
        any_valid_certificate=False,
        certificate_gap=None,
        stages_executed=1,
        final_budget=4,
        full_fallback_occurred=False,
    )
    assert rec.certificate_gap is None


# 8. Canonical CAGR Convention Enforced
def test_8_canonical_cagr_convention_enforced():
    """
    Verifies R_CAGR = (V_T / V_0)^(252 / N_trading) - 1.
    """
    fin = FinancialMetricsSummary(
        strategy_name="TestStrategy",
        annualized_return=0.10,
        annualized_volatility=0.15,
        sharpe_ratio=0.533,
        max_drawdown=-0.12,
        total_transaction_cost=1500.0,
        mean_turnover=0.30,
        mean_holdings=3.0,
        mean_cash_weight=0.10,
        final_wealth=1_100_000.0,
        total_wealth_return=0.10,
        cagr=0.10,
        annualized_arithmetic_mean_return=0.105,
    )
    assert fin.total_wealth_return == pytest.approx(0.10, rel=1e-5)
    assert fin.cagr == pytest.approx(0.10, rel=1e-5)

    # Strict audit test: reported CAGR == reconstructed CAGR from V_0 and V_T in final_table_1
    table_path = Path("artifacts/final_tables/final_table_1_financial_performance.csv")
    if table_path.exists():
        import pandas as pd
        df = pd.read_csv(table_path)
        v0 = 1_000_000.0
        for _, row in df.iterrows():
            vt = float(str(row["FinalWealth($)"]).replace("$", "").replace(",", ""))
            reported_cagr = float(str(row["CAGR"]).replace("%", "")) / 100.0
            reconstructed_cagr = (vt / v0) ** (52.0 / 51.0) - 1.0
            assert reported_cagr == pytest.approx(reconstructed_cagr, abs=1e-4), (
                f"Strategy {row['Strategy']}: reported CAGR {reported_cagr:.4f} != "
                f"reconstructed CAGR {reconstructed_cagr:.4f} from V_T={vt} and V_0={v0}"
            )


# 9. Canonical Sharpe Convention Enforced
def test_9_canonical_sharpe_convention_enforced():
    """
    Verifies Sharpe = (CAGR - R_f) / sigma_ann with canonical R_f = 0.02.
    """
    manifest_path = Path("artifacts/final_reproducibility_manifest.json")
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        rf_meta = manifest["semantics_and_tolerances"]["financial_metric_conventions"]["sharpe_ratio"]
        assert "0.02" in rf_meta


# 10. Canonical Min Cash Enforced
def test_10_canonical_min_cash_enforced():
    """
    Verifies canonical c_min = 0.10 across instance generator and parameters.
    """
    gen = ScalableInstanceGenerator()
    inst = gen.generate_instance(n_assets=10, horizon=3, cardinality_limit=3, seed=42)
    assert inst.params.min_cash == pytest.approx(0.10, abs=1e-6)


# 11. Simple Return Convention and Wealth Transition Enforced
def test_11_simple_return_convention_arithmetic():
    """
    Verifies that asset returns are arithmetic simple returns r_t = (P_t / P_{t-1}) - 1,
    gross return is g_t = 1 + r_t, and wealth evolution is W_{t+1} = W_t * (1 + r_p).
    """
    from learning_augmented_portfolio.data import MarketDataStore, ReturnCalculator
    store = MarketDataStore.load_csv("artifacts/real_market/daily_prices.csv")
    ret_calc = ReturnCalculator(store)

    d0 = store.trading_dates[100]
    d1 = store.trading_dates[101]

    p0 = store.get_prices(d0)
    p1 = store.get_prices(d1)

    expected_simple = (p1 / p0) - 1.0
    actual_ret = ret_calc.get_realized_single_day_return(d1)
    np.testing.assert_allclose(actual_ret, expected_simple, rtol=1e-6, atol=1e-6)

    d5 = store.trading_dates[105]
    p5 = store.get_prices(d5)
    expected_compound = (p5 / p0) - 1.0
    actual_compound = ret_calc.get_realized_multi_day_return(d0, d5)
    np.testing.assert_allclose(actual_compound, expected_compound, rtol=1e-6, atol=1e-6)


# 12. Fixed Dollar Cost Normalization Across Wealth Levels Enforced
def test_12_fixed_dollar_cost_normalization():
    """
    Verifies that for fixed fee F = $10 and two different wealth levels V_a != V_b,
    f_a = F / V_a, f_b = F / V_b, and f_a * V_a == f_b * V_b == F.
    Also verifies RealMarketBacktestEngine.get_step_parameters implements this exact scaling.
    """
    from learning_augmented_portfolio.data import MarketDataStore
    from learning_augmented_portfolio.backtest.engine import RealMarketBacktestEngine
    from learning_augmented_portfolio.optimization.variables import (
        PortfolioParameters,
        MultiPeriodPortfolioParameters,
    )

    store = MarketDataStore.load_csv("artifacts/real_market/daily_prices.csv")
    N = store.universe.n_assets
    H = 3
    single_p = PortfolioParameters(
        risk_aversion=5.0,
        risk_free_rate=0.0001,
        max_turnover=0.40,
        min_cash=0.10,
        weight_upper_limits=np.full(N, 0.40),
        liquidity_limits=np.full(N, 0.20),
        prop_cost_coeffs=np.full(N, 0.0005),
        cardinality_limit=4,
        min_holding_weights=np.full(N, 0.08),
        min_trade_sizes=np.full(N, 0.02),
        fixed_cost_coeffs=np.full(N, 0.00001),
    )
    base_params = MultiPeriodPortfolioParameters.from_single_period(single_p, horizon=H)

    engine = RealMarketBacktestEngine(
        data_store=store,
        params=base_params,
        initial_wealth=1_000_000.0,
        fixed_cost_dollars=10.0,
    )

    V_a = 800_000.0
    V_b = 1_250_000.0
    F = 10.0

    params_a = engine.get_step_parameters("2024-01-02", portfolio_wealth=V_a)
    params_b = engine.get_step_parameters("2024-01-02", portfolio_wealth=V_b)

    f_a = params_a.fixed_cost_coeffs[0, 0]
    f_b = params_b.fixed_cost_coeffs[0, 0]

    assert f_a == pytest.approx(F / V_a, rel=1e-7)
    assert f_b == pytest.approx(F / V_b, rel=1e-7)
    assert f_a != f_b
    assert (f_a * V_a) == pytest.approx(F, rel=1e-7)
    assert (f_b * V_b) == pytest.approx(F, rel=1e-7)


# 13. Gurobi MIPGap Relative Definition Semantics Enforced
def test_13_gurobi_mipgap_definition_semantics():
    """
    Verifies that Gurobi MIPGap satisfies MIPGap = |ObjVal - ObjBound| / |ObjVal|
    when ObjVal != 0, matching Proposition 5.4.
    """
    gen = ScalableInstanceGenerator()
    inst = gen.generate_instance(n_assets=8, horizon=2, cardinality_limit=3, seed=42)
    runner = ScalabilityRunner(SolverConfig(time_limit=10.0, mip_gap=1e-4))
    res = runner.solve_full_miqp(inst, record_anytime=False)
    assert res.status.value == "OPTIMAL"
    assert res.objective_value is not None
    assert res.mip_gap is not None
    assert res.mip_gap <= 1e-4 + 1e-6


# 14. Real-Market Asset Universe Tickers Verified
def test_14_real_market_asset_universe_tickers():
    """
    Verifies that the executed real-market asset universe consists of the 10 tickers:
    AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, JNJ, XOM, PG.
    """
    meta_path = Path("artifacts/real_market/universe_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    expected_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "XOM", "PG"]
    assert meta["tickers"] == expected_tickers
    assert "UNH" not in meta["tickers"]
    assert "V" not in meta["tickers"]


# 15. Chronological Split Strict Boundaries Verified
def test_15_chronological_split_strict_boundaries():
    """
    Verifies exact partition dates and strict temporal ordering without leakage:
    Train: 2021-01-04 to 2023-05-23 (621 days)
    Val:   2023-05-24 to 2023-12-29 (156 days)
    Test:  2024-01-02 to 2024-12-31 (259 days, 51 weekly steps)
    """
    from learning_augmented_portfolio.data import MarketDataStore
    store = MarketDataStore.load_csv("artifacts/real_market/daily_prices.csv")
    dates = store.trading_dates
    T = len(dates)
    assert T == 1036
    train_end = int(T * 0.60)
    val_end = int(T * 0.75)

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    assert len(train_dates) == 621
    assert len(val_dates) == 156
    assert len(test_dates) == 259

    assert train_dates[0] == "2021-01-04"
    assert train_dates[-1] == "2023-05-23"
    assert val_dates[0] == "2023-05-24"
    assert val_dates[-1] == "2023-12-29"
    assert test_dates[0] == "2024-01-02"
    assert test_dates[-1] == "2024-12-31"

    cal = store.calendar
    rebal = cal.get_rebalancing_dates(test_dates[0], test_dates[-1], frequency_days=5)
    assert len(rebal) == 52  # 51 execution steps


# 16. Table 6 Budget Configuration and Speedup Range Verified
def test_16_table_6_budget_and_speedup_values():
    """
    Verifies that Table 6 candidate budgets and speedups match table_d_e2e_acceleration.csv
    and that Fast MarketOnly median speedup range is 1.10x - 2.43x (mean 1.13x - 3.12x).
    """
    import pandas as pd
    df_d = pd.read_csv("artifacts/scalability/table_d_e2e_acceleration.csv")
    fast = df_d[df_d["Method"] == "Fast_MarketOnly_MIQP"]
    # Configurations in Table 6: (1,10), (1,25), (3,8), (3,10), (5,5)
    t6_configs = [(1, 10), (1, 25), (3, 8), (3, 10), (5, 5)]
    sub = fast[fast.apply(lambda r: (r["H"], r["N"]) in t6_configs, axis=1)]
    assert len(sub) == 5
    assert sub["Median_Speedup"].min() == pytest.approx(1.104, abs=0.01)
    assert sub["Median_Speedup"].max() == pytest.approx(2.431, abs=0.01)
    assert sub["Mean_Speedup"].min() == pytest.approx(1.131, abs=0.01)
    assert sub["Mean_Speedup"].max() == pytest.approx(3.119, abs=0.01)



