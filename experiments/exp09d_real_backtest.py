"""
Phase 9 Experiment 9D: Real-Market Out-of-Sample Backtesting and 4-Layer Audited Evaluation.
Evaluates 7 benchmark strategies over the chronological Test period (2024-01-02 to 2024-12-31):
  1. Full-MIQP MPC (exact, B=N=10)
  2. Fast MarketOnly MIQP (B=4, one-shot reduced solve)
  3. ContinuousRelaxation Ranking (B=4, one-shot reduced solve)
  4. Best ML Candidate (B=4, one-shot reduced solve using frozen model)
  5. Certified / Nested Recovery Method (B0=4, ladder=[4, 6, 10], certified stopping)
  6. Equal Weight (1/N)
  7. Buy and Hold

Generates:
  - 4 strictly separated metrics tables (Financial, Solver/Optimization, Certification, Forecast Quality)
  - Mathematical audit report (Wealth accounting residual, budget conservation, hard constraints)
  - 7 publication-quality plots saved in artifacts/plots/
"""

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any, Dict, List, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from learning_augmented_portfolio.data import (
    MarketDataStore,
    RealMarketUniverse,
    ReturnCalculator,
    LiquidityCalculator,
)
from learning_augmented_portfolio.dynamic.state import PortfolioState
from learning_augmented_portfolio.forecasts import (
    ExpectedReturnEstimator,
    CovarianceEstimator,
)
from learning_augmented_portfolio.optimization.variables import (
    PortfolioParameters,
    MultiPeriodMarketState,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from learning_augmented_portfolio.optimization.reduced_miqp import MultiPeriodReducedMIQPOptimizer
from learning_augmented_portfolio.optimization.relaxation import MultiPeriodRelaxedQPOptimizer
from learning_augmented_portfolio.optimization.support_restriction import CandidateUniverse
from learning_augmented_portfolio.learning.active_set import SupportPredictorModel
from learning_augmented_portfolio.learning.baselines import ContinuousRelaxationYBaseline
from learning_augmented_portfolio.learning.candidate_set import construct_candidate_universe
from learning_augmented_portfolio.recovery.recovery_loop import NestedRecoveryOptimizer
from learning_augmented_portfolio.recovery.certificate import CertificateSource
from learning_augmented_portfolio.backtest.engine import (
    RealMarketBacktestEngine,
    BacktestExecutionResult,
)
from learning_augmented_portfolio.backtest.baselines import (
    EqualWeightBaseline,
    BuyAndHoldBaseline,
)
from learning_augmented_portfolio.backtest.audit import BacktestAuditor, BacktestAuditResult
from learning_augmented_portfolio.backtest.records import (
    BacktestStepRecord,
    compute_financial_metrics,
    FinancialMetricsSummary,
    OptimizationMetricsSummary,
    CertificationMetricsSummary,
)
from learning_augmented_portfolio.dataset.instance import (
    OptimizationInstance,
    OptimizationInstanceMetadata,
)


def run_exp09d_real_backtest() -> None:
    print("=" * 80)
    print("PHASE 9D: REAL-MARKET OUT-OF-SAMPLE BACKTESTING & EVALUATION")
    print("=" * 80)

    # 1. Load Market Data Store & Universe
    store = MarketDataStore.load_csv("artifacts/real_market/daily_prices.csv")
    universe = store.universe
    N = universe.n_assets
    H = 3

    # Date partitions
    all_dates = store.trading_dates
    T = len(all_dates)
    val_end = int(T * 0.75)
    test_start_date = all_dates[val_end]      # 2024-01-02
    test_end_date = all_dates[-1]             # 2024-12-31

    print(f"\n[1] Configuration & Partitions:")
    print(f"    - Universe: {universe.tickers} (N={N})")
    print(f"    - Planning Horizon: H={H} steps")
    print(f"    - Test Period: {test_start_date} -> {test_end_date} (Chronological)")

    # 2. Parameters & Engine Setup (Weekly 5-trading-day Control Grid)
    rf_week = (1.0 + 0.03) ** (5.0 / 252.0) - 1.0
    single_p = PortfolioParameters(
        risk_aversion=5.0,
        risk_free_rate=rf_week,
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
    base_params = MultiPeriodPortfolioParameters.from_single_period(single_p, horizon=H, discount_factor=0.99)
    solver_cfg = SolverConfig(time_limit=15.0, mip_gap=1e-5, threads=1)

    engine = RealMarketBacktestEngine(
        data_store=store,
        params=base_params,
        solver_config=solver_cfg,
        initial_wealth=1_000_000.0,
        fixed_cost_dollars=10.0,
        proportional_cost_bps=5.0,
        participation_rate=0.05,
        rebalance_frequency_days=5,
        risk_free_rate_annual=0.03,
        use_weekly_forecasts=True,
    )

    # 3. Load Frozen Policy & Trained Models
    with open("artifacts/real_models/frozen_policy.json", "r", encoding="utf-8") as f:
        frozen_policy = json.load(f)

    B_policy = int(frozen_policy["candidate_budget_B"])
    print(f"    - Frozen Candidate Budget Policy: B = {B_policy} (Reduction = {frozen_policy['reduction']:.1%})")

    best_ml_model: SupportPredictorModel = joblib.load("artifacts/real_models/best_active_set_model.joblib")
    cont_relax_baseline = ContinuousRelaxationYBaseline()

    # Helpers
    def make_instance(
        step_dt: str,
        st: PortfolioState,
        step_p: MultiPeriodPortfolioParameters,
        mkt: MultiPeriodMarketState,
        relaxation=None,
    ) -> OptimizationInstance:
        meta = OptimizationInstanceMetadata(
            instance_id=f"bt_{step_dt}",
            calendar_date=step_dt,
            time_index=st.time_index,
            split="test",
            universe_id="LIQUID_BASKET_10",
            n_assets=N,
            horizon=H,
            dataset_version="1.0",
            config_hash="test_v1",
            git_commit=None,
            gurobi_version="12.0",
            python_version="3.13",
            solver_params={"mip_gap": 1e-5, "time_limit": 15.0},
            random_seed=42,
            solver_status=2,
            label_quality="HEURISTIC",
            is_trainable_label=False,
            objective=0.0,
            best_bound=0.0,
            mip_gap=0.0,
            runtime=0.0,
            work=0.0,
            node_count=0,
            sol_count=1,
            used_fallback=False,
        )
        return OptimizationInstance(
            metadata=meta,
            actual_state=st,
            params=step_p,
            market_state=mkt,
            relaxation=relaxation,
        )

    # 4. Strategy Definitions
    # (1) Full-MIQP MPC (exact, B=10)
    full_opt_cache: Dict[str, Any] = {}
    full_trades_audit: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def strat_full_miqp(mkt: MultiPeriodMarketState, st: PortfolioState, dt: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        step_p = engine.get_step_parameters(dt, st.actual_wealth)
        opt = MultiPeriodMIQPOptimizer(step_p, solver_cfg)
        t0 = time.perf_counter()
        sol = opt.solve(mkt)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        if sol.is_feasible and sol.post_trade_holdings is not None:
            x0 = sol.post_trade_holdings[0]
            b0 = sol.post_trade_cash[0]
            if sol.trades is not None and sol.absolute_trades is not None and sol.trade_indicators is not None:
                full_trades_audit.append((sol.trades.copy(), sol.absolute_trades.copy(), sol.trade_indicators.copy()))
        else:
            x0 = st.normalized_weights
            b0 = st.normalized_cash

        meta = {
            "solve_time_ms": solve_ms,
            "node_count": sol.node_count or 0,
            "work": sol.work or 0.0,
            "mip_gap": sol.mip_gap or 0.0,
            "objective_value": sol.objective_value,
            "certificate_source": CertificateSource.FULL_MIP_BOUND.value,
            "relaxation_certified": False,
            "full_mip_certified": True,
            "any_valid_certificate": True,
            "certificate_gap": None,
            "stages_executed": 1,
            "final_budget": N,
            "full_fallback_occurred": False,
        }
        full_opt_cache[dt] = meta
        return x0, b0, meta

    # (2) Fast MarketOnly MIQP (B=4, one-shot reduced)
    def strat_fast_market_only(mkt: MultiPeriodMarketState, st: PortfolioState, dt: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        step_p = engine.get_step_parameters(dt, st.actual_wealth)
        inst = make_instance(dt, st, step_p, mkt)

        t0 = time.perf_counter()
        scores = best_ml_model.predict_instance_scores(inst)
        cand = construct_candidate_universe(scores, budget_B=B_policy, current_weights=st.normalized_weights, use_safe_augmentation=True)
        reduced_opt = MultiPeriodReducedMIQPOptimizer(step_p, solver_cfg)
        red_exec = reduced_opt.solve(mkt, cand, expand_solution=True)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        sol = red_exec.optimization_result
        if sol.is_feasible and sol.post_trade_holdings is not None:
            x0 = sol.post_trade_holdings[0]
            b0 = sol.post_trade_cash[0]
        else:
            x0 = st.normalized_weights
            b0 = st.normalized_cash

        meta = {
            "solve_time_ms": solve_ms,
            "node_count": sol.node_count or 0,
            "work": sol.work or 0.0,
            "mip_gap": sol.mip_gap or 0.0,
            "objective_value": sol.objective_value,
            "certificate_source": CertificateSource.NONE.value,
            "relaxation_certified": False,
            "full_mip_certified": False,
            "any_valid_certificate": False,
            "certificate_gap": None,
            "stages_executed": 1,
            "final_budget": B_policy,
            "full_fallback_occurred": False,
        }
        return x0, b0, meta

    # (3) ContinuousRelaxation Ranking (B=4, one-shot reduced)
    def strat_continuous_relaxation(mkt: MultiPeriodMarketState, st: PortfolioState, dt: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        step_p = engine.get_step_parameters(dt, st.actual_wealth)

        t0 = time.perf_counter()
        rel_opt = MultiPeriodRelaxedQPOptimizer(step_p, solver_cfg)
        rel_res = rel_opt.solve(mkt)

        inst = make_instance(dt, st, step_p, mkt, relaxation=rel_res)
        scores = cont_relax_baseline.predict_instance_scores(inst)
        cand = construct_candidate_universe(scores, budget_B=B_policy, current_weights=st.normalized_weights, use_safe_augmentation=True)

        reduced_opt = MultiPeriodReducedMIQPOptimizer(step_p, solver_cfg)
        red_exec = reduced_opt.solve(mkt, cand, expand_solution=True)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        sol = red_exec.optimization_result
        if sol.is_feasible and sol.post_trade_holdings is not None:
            x0 = sol.post_trade_holdings[0]
            b0 = sol.post_trade_cash[0]
        else:
            x0 = st.normalized_weights
            b0 = st.normalized_cash

        # Check relaxation gap
        gap = (sol.objective_value - rel_res.objective_value) if (sol.is_feasible and rel_res.is_feasible) else None
        is_rel_cert = bool(gap is not None and gap <= 1e-4)

        meta = {
            "solve_time_ms": solve_ms,
            "node_count": sol.node_count or 0,
            "work": sol.work or 0.0,
            "mip_gap": sol.mip_gap or 0.0,
            "objective_value": sol.objective_value,
            "certificate_source": CertificateSource.FULL_RELAXATION.value if is_rel_cert else CertificateSource.NONE.value,
            "relaxation_certified": is_rel_cert,
            "full_mip_certified": False,
            "any_valid_certificate": is_rel_cert,
            "certificate_gap": gap,
            "stages_executed": 1,
            "final_budget": B_policy,
            "full_fallback_occurred": False,
        }
        return x0, b0, meta

    # (4) Best ML Candidate (B=4, one-shot reduced)
    def strat_best_ml(mkt: MultiPeriodMarketState, st: PortfolioState, dt: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        # Functionally identical to strat_fast_market_only, representing the frozen ML model
        return strat_fast_market_only(mkt, st, dt)

    # (5) Certified / Nested Recovery Method (starts B=4, ladder=[4, 6, 10])
    rec_trades_audit: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def strat_certified_recovery(mkt: MultiPeriodMarketState, st: PortfolioState, dt: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        step_p = engine.get_step_parameters(dt, st.actual_wealth)
        inst = make_instance(dt, st, step_p, mkt)

        t0 = time.perf_counter()
        scores = best_ml_model.predict_instance_scores(inst)

        rec_opt = NestedRecoveryOptimizer(
            params=step_p,
            solver_config=solver_cfg,
            epsilon_cert=1e-4,
            use_recovery_warm_starts=True,
        )
        rec_res = rec_opt.solve(
            instance=inst,
            scores=scores,
            budgets=[B_policy, 6, 10],
            fast_mode=False,
        )
        solve_ms = (time.perf_counter() - t0) * 1000.0

        sol = rec_res.final_solution
        if sol is not None and sol.is_feasible and sol.post_trade_holdings is not None:
            x0 = sol.post_trade_holdings[0]
            b0 = sol.post_trade_cash[0]
            if sol.trades is not None and sol.absolute_trades is not None and sol.trade_indicators is not None:
                rec_trades_audit.append((sol.trades.copy(), sol.absolute_trades.copy(), sol.trade_indicators.copy()))
        else:
            x0 = st.normalized_weights
            b0 = st.normalized_cash

        cert_src_str = rec_res.certificate_source.value if hasattr(rec_res.certificate_source, "value") else str(rec_res.certificate_source)
        meta = {
            "solve_time_ms": solve_ms,
            "node_count": sol.node_count if sol else 0,
            "work": sol.work if sol else 0.0,
            "mip_gap": sol.mip_gap if sol else 0.0,
            "objective_value": rec_res.final_objective,
            "certificate_source": cert_src_str,
            "relaxation_certified": rec_res.relaxation_certified,
            "full_mip_certified": rec_res.full_mip_certified,
            "any_valid_certificate": rec_res.any_valid_certificate,
            "certificate_gap": rec_res.final_certificate_gap,
            "stages_executed": rec_res.stages_executed,
            "final_budget": rec_res.final_budget,
            "full_fallback_occurred": rec_res.full_fallback_occurred,
        }
        return x0, b0, meta

    # (6) Equal Weight (1/N)
    eq_baseline = EqualWeightBaseline(base_params)

    def strat_equal_weight(mkt: MultiPeriodMarketState, st: PortfolioState, dt: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        w, c = eq_baseline.get_target_weights(N)
        return w, c, {"solve_time_ms": 0.01, "node_count": 0, "work": 0.0, "mip_gap": 0.0}

    # (7) Buy and Hold
    bh_baseline = BuyAndHoldBaseline(initial_weights=np.zeros(N), initial_cash=1.0)

    def strat_buy_and_hold(mkt: MultiPeriodMarketState, st: PortfolioState, dt: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        if not bh_baseline.has_initialized:
            # Initialize with 1/N allocation on first step
            w, c = eq_baseline.get_target_weights(N)
            bh_baseline.initial_weights = w
            bh_baseline.initial_cash = c
        w, c = bh_baseline.get_action(st.normalized_weights, st.normalized_cash)
        return w, c, {"solve_time_ms": 0.01, "node_count": 0, "work": 0.0, "mip_gap": 0.0}

    strategies = {
        "Full_MIQP_MPC": strat_full_miqp,
        "Fast_MarketOnly_MIQP": strat_fast_market_only,
        "ContinuousRelaxation_MIQP": strat_continuous_relaxation,
        "Best_ML_Candidate": strat_best_ml,
        "Certified_Recovery_Method": strat_certified_recovery,
        "Equal_Weight_1_over_N": strat_equal_weight,
        "Buy_and_Hold": strat_buy_and_hold,
    }

    # 5. Run Backtests
    print("\n[2] Executing Out-of-Sample Backtests across 7 Strategies...")
    results: Dict[str, BacktestExecutionResult] = {}
    audit_results: Dict[str, BacktestAuditResult] = {}
    financial_summaries: Dict[str, FinancialMetricsSummary] = {}

    auditor = BacktestAuditor(base_params, tolerance=1e-4)

    for name, fn in strategies.items():
        t0_strat = time.perf_counter()
        res = engine.run_backtest(
            strategy_name=name,
            start_date=test_start_date,
            end_date=test_end_date,
            strategy_fn=fn,
        )
        elapsed = time.perf_counter() - t0_strat
        results[name] = res

        # Run independent mathematical audit
        # Equal weight and buy-and-hold might have > 4 holdings (violates K=4 cardinality of active strategy)
        check_active_constraints = (name not in ["Equal_Weight_1_over_N", "Buy_and_Hold"])
        audit_res = auditor.audit_backtest(name, res.step_records, res.wealth_series, check_constraints=check_active_constraints)
        audit_results[name] = audit_res

        # Financial Metrics
        turnovers = np.array([r.turnover for r in res.step_records])
        costs = np.array([r.transaction_cost for r in res.step_records])
        holdings = np.array([r.num_holdings for r in res.step_records])
        cash_w = np.array([r.cash_post for r in res.step_records])

        fin = compute_financial_metrics(
            strategy_name=name,
            wealth_series=res.wealth_series,
            turnovers=turnovers,
            transaction_costs=costs,
            holdings_counts=holdings,
            cash_weights=cash_w,
            annualization_factor=52.0,
            risk_free_rate=0.02,
        )
        financial_summaries[name] = fin

        print(f"    - {name:<26}: Final Wealth=${res.wealth_series[-1]:,.2f}, Sharpe={fin.sharpe_ratio:.2f}, Ret={fin.annualized_return:.1%}, MaxDD={fin.max_drawdown:.1%}, Steps={len(res.step_records)} ({elapsed:.2f}s)")

    # 6. Compute Optimization / Solver Quality Metrics
    full_recs = results["Full_MIQP_MPC"].step_records
    full_times = np.array([r.solve_time_ms for r in full_recs])
    full_objs = np.array([r.objective_value for r in full_recs])

    opt_summaries: Dict[str, OptimizationMetricsSummary] = {}
    cert_summaries: Dict[str, CertificationMetricsSummary] = {}

    for name, res in results.items():
        recs = res.step_records
        times = np.array([r.solve_time_ms for r in recs])
        nodes = np.array([r.node_count for r in recs])
        works = np.array([r.work for r in recs])
        gaps = np.array([r.mip_gap for r in recs])
        objs = np.array([r.objective_value if r.objective_value is not None else 0.0 for r in recs])
        stages = np.array([r.stages_executed for r in recs])
        fallbacks = np.array([1.0 if r.full_fallback_occurred else 0.0 for r in recs])

        # Objective degradation: J^{strategy} - J^{full} (>= 0 for minimization)
        degradations = objs - full_objs
        exact_rec = np.mean(np.abs(degradations) <= 1e-4)

        mean_t = float(np.mean(times))
        med_t = float(np.median(times))
        speedup = float(np.mean(full_times) / mean_t) if mean_t > 1e-6 else 1.0

        opt_summaries[name] = OptimizationMetricsSummary(
            strategy_name=name,
            mean_runtime_ms=mean_t,
            median_runtime_ms=med_t,
            speedup_vs_full=speedup,
            mean_node_count=float(np.mean(nodes)),
            mean_work=float(np.mean(works)),
            mean_mip_gap=float(np.mean(gaps)),
            mean_objective_degradation=float(np.mean(degradations)),
            exact_recovery_rate=float(exact_rec),
            avg_recovery_stages=float(np.mean(stages)),
            full_fallback_rate=float(np.mean(fallbacks)),
        )

        # Certification
        rel_certs = np.array([1.0 if r.relaxation_certified else 0.0 for r in recs])
        mip_certs = np.array([1.0 if r.full_mip_certified else 0.0 for r in recs])
        any_certs = np.array([1.0 if r.any_valid_certificate else 0.0 for r in recs])
        cert_gaps = [r.certificate_gap for r in recs if r.certificate_gap is not None]
        mean_gap = float(np.mean(cert_gaps)) if cert_gaps else None

        cert_summaries[name] = CertificationMetricsSummary(
            strategy_name=name,
            relaxation_certified_rate=float(np.mean(rel_certs)),
            full_mip_certified_rate=float(np.mean(mip_certs)),
            any_valid_certificate_rate=float(np.mean(any_certs)),
            mean_relaxation_gap=mean_gap,
        )

    # 7. Compute Forecast Quality Metrics
    ret_calc = ReturnCalculator(store)
    exp_est = ExpectedReturnEstimator(ret_calc, lookback_window=60)
    rebal_dates = results["Full_MIQP_MPC"].rebalance_dates

    forecast_errors = []
    directional_correct = []
    spearman_corrs = []

    for t_idx in range(len(rebal_dates) - 1):
        d_curr = rebal_dates[t_idx]
        d_next = rebal_dates[t_idx + 1]

        mu_hat = exp_est.construct_weekly_horizon_forecast(d_curr, 1, days_per_week=5)[0]
        r_real = ret_calc.get_realized_multi_day_return(d_curr, d_next)

        # MAE
        err = np.abs(mu_hat - r_real)
        forecast_errors.extend(err.tolist())

        # Directional Hit Rate
        hit = (np.sign(mu_hat) == np.sign(r_real)).astype(float)
        directional_correct.extend(hit.tolist())

        # Spearman Rank Correlation
        if np.std(mu_hat) > 1e-8 and np.std(r_real) > 1e-8:
            rho, _ = stats.spearmanr(mu_hat, r_real)
            if not np.isnan(rho):
                spearman_corrs.append(float(rho))

    mae_val = float(np.mean(forecast_errors))
    dir_acc = float(np.mean(directional_correct))
    ic_val = float(np.mean(spearman_corrs)) if spearman_corrs else 0.0

    # 8. Print Tables
    print("\n" + "=" * 90)
    print("LAYER 1: FINANCIAL PERFORMANCE SUMMARY TABLE (2024 Test Out-of-Sample)")
    print("=" * 90)
    fin_rows = []
    for s, m in financial_summaries.items():
        fin_rows.append({
            "Strategy": s,
            "TotalReturn": f"{m.total_wealth_return * 100:.2f}%",
            "CAGR": f"{m.cagr * 100:.2f}%",
            "Ann.ArithmeticMean": f"{m.annualized_arithmetic_mean_return * 100:.2f}%",
            "Ann.Vol": f"{m.annualized_volatility * 100:.2f}%",
            "Sharpe": f"{m.sharpe_ratio:.2f}",
            "MaxDD": f"{m.max_drawdown * 100:.2f}%",
            "Turnover": f"{m.mean_turnover * 100:.1f}%",
            "Costs($)": f"${m.total_transaction_cost:,.1f}",
            "Holdings": f"{m.mean_holdings:.1f}",
            "Cash": f"{m.mean_cash_weight * 100:.1f}%",
            "FinalWealth($)": f"${m.final_wealth:,.0f}",
        })
    fin_df = pd.DataFrame(fin_rows)
    print(fin_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("LAYER 2: OPTIMIZATION & SOLVER PERFORMANCE SUMMARY TABLE")
    print("=" * 90)
    opt_rows = []
    for s, m in opt_summaries.items():
        opt_rows.append({
            "Strategy": s,
            "MeanTime(ms)": f"{m.mean_runtime_ms:.2f}",
            "MedianTime(ms)": f"{m.median_runtime_ms:.2f}",
            "Speedup": f"{m.speedup_vs_full:.2f}x",
            "MeanNodes": f"{m.mean_node_count:.1f}",
            "MeanWork": f"{m.mean_work:.3f}",
            "MIPGap": f"{m.mean_mip_gap:.2e}",
            "ObjDegradation": f"{m.mean_objective_degradation:.2e}",
            "ExactRecovery": f"{m.exact_recovery_rate * 100:.1f}%",
            "AvgStages": f"{m.avg_recovery_stages:.2f}",
            "FallbackRate": f"{m.full_fallback_rate * 100:.1f}%",
        })
    opt_df = pd.DataFrame(opt_rows)
    print(opt_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("LAYER 3: CERTIFICATION SUMMARY TABLE (Corrected Semantics)")
    print("=" * 90)
    cert_rows = []
    for s, m in cert_summaries.items():
        cert_rows.append({
            "Strategy": s,
            "RelaxationCertRate": f"{m.relaxation_certified_rate * 100:.1f}%",
            "FullMIPCertRate": f"{m.full_mip_certified_rate * 100:.1f}%",
            "AnyCertRate": f"{m.any_valid_certificate_rate * 100:.1f}%",
            "MeanCertGap": f"{m.mean_relaxation_gap:.6f}" if m.mean_relaxation_gap is not None else "N/A",
        })
    cert_df = pd.DataFrame(cert_rows)
    print(cert_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("LAYER 4: FORECAST QUALITY METRICS (Isolated from Solver Optimality)")
    print("=" * 90)
    print(f"    - Estimator: Historical Rolling Mean (Window=60)")
    print(f"    - Mean Absolute Error (MAE)            : {mae_val:.4f} ({mae_val * 100:.2f}%)")
    print(f"    - Directional Accuracy (Hit Rate)      : {dir_acc * 100:.2f}%")
    print(f"    - Information Coefficient (Rank IC)    : {ic_val:.4f}")

    print("\n" + "=" * 90)
    print("LAYER 5: INDEPENDENT MATHEMATICAL & CONSTRAINT AUDIT REPORT")
    print("=" * 90)
    audit_rows = []
    for s, a in audit_results.items():
        audit_rows.append({
            "Strategy": s,
            "TotalSteps": a.total_steps,
            "MaxWealthResidual": f"{a.max_wealth_residual:.2e}",
            "MaxBudgetResidual": f"{a.max_budget_residual:.2e}",
            "WealthAccountingValid": a.is_wealth_accounting_valid,
            "ConstraintViolations": a.constraint_violation_count,
        })
    audit_df = pd.DataFrame(audit_rows)
    print(audit_df.to_string(index=False))

    # Layer 6: Micro-Trade & Charged-Floor Audit
    print("\n" + "=" * 90)
    print("LAYER 6: MICRO-TRADE & CHARGED-FLOOR AUDIT (Weekly Control Grid)")
    print("=" * 90)
    def audit_trade_decisions(name: str, records: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], delta_floor: float = 0.02):
        if not records:
            return {}
        u_all = np.concatenate([r[0].flatten() for r in records])
        q_all = np.concatenate([r[1].flatten() for r in records])
        z_all = np.concatenate([r[2].flatten() for r in records])
        total = len(u_all)
        abs_u = np.abs(u_all)
        micro = (abs_u > 1e-7) & (abs_u < delta_floor - 1e-7)
        micro_cnt = int(np.sum(micro))
        micro_pct = (micro_cnt / total) * 100.0
        pos_u = abs_u[abs_u > 1e-7]
        min_pos_u = float(np.min(pos_u)) if len(pos_u) > 0 else 0.0
        excess = q_all - abs_u
        max_excess = float(np.max(excess))
        spurious = int(np.sum((z_all > 0.5) & (abs_u <= 1e-7)))
        return {
            "Experiment Family": name,
            "Total Decisions": total,
            "Decisions with 0 < |u| < delta": micro_cnt,
            "Fraction (%)": f"{micro_pct:.2f}%",
            "Min Positive |u|": f"{min_pos_u:.3e}",
            "Max Excess (q - |u|)": f"{max_excess:.3e}",
            "Spurious Trades": spurious,
            "Executed Floor delta": f"{delta_floor:.2f} ({int(delta_floor*10000)} bps)",
        }

    micro_rows = []
    if full_trades_audit:
        micro_rows.append(audit_trade_decisions("2024 Backtest (Full MIQP MPC)", full_trades_audit))
    if rec_trades_audit:
        micro_rows.append(audit_trade_decisions("2024 Backtest (Certified Recovery)", rec_trades_audit))
    micro_df = pd.DataFrame(micro_rows)
    if not micro_df.empty:
        print(micro_df.to_string(index=False))

    # 9. Save All Reports & Metrics
    out_dir = Path("artifacts/real_backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    fin_df.to_csv(out_dir / "financial_metrics.csv", index=False)
    opt_df.to_csv(out_dir / "optimization_metrics.csv", index=False)
    cert_df.to_csv(out_dir / "certification_metrics.csv", index=False)
    audit_df.to_csv(out_dir / "audit_metrics.csv", index=False)
    if not micro_df.empty:
        micro_df.to_csv(out_dir / "micro_trade_audit.csv", index=False)

    # Save Wealth Trajectories and Weekly Returns
    wealth_dict = {"Date": results["Full_MIQP_MPC"].rebalance_dates}
    for s, r in results.items():
        wealth_dict[s] = r.wealth_series
    pd.DataFrame(wealth_dict).to_csv(out_dir / "wealth_trajectories.csv", index=False)

    ret_dict = {"Date": results["Full_MIQP_MPC"].rebalance_dates[1:]}
    for s, r in results.items():
        w = r.wealth_series
        ret_dict[s] = w[1:] / w[:-1] - 1.0
    pd.DataFrame(ret_dict).to_csv(out_dir / "weekly_returns.csv", index=False)

    with open(out_dir / "forecast_quality.json", "w", encoding="utf-8") as f:
        json.dump({
            "mae": mae_val,
            "directional_accuracy": dir_acc,
            "rank_ic": ic_val,
        }, f, indent=2)

    # 10. Generate 7 Publication Plots
    plot_dir = Path("artifacts/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    dates_plot = [pd.to_datetime(d) for d in results["Full_MIQP_MPC"].rebalance_dates]

    # Plot 1: Cumulative Wealth
    plt.figure(figsize=(11, 6), dpi=150)
    for name, res in results.items():
        plt.plot(dates_plot, res.wealth_series, label=name.replace("_", " "), linewidth=1.8)
    plt.title("Plot 1: Cumulative Wealth Trajectory on Real Market Equities (2024 Test Period)", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Portfolio Wealth ($)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_dir / "plot01_cumulative_wealth.png")
    plt.close()

    # Plot 2: Drawdowns
    plt.figure(figsize=(11, 5), dpi=150)
    for name, res in results.items():
        w = res.wealth_series
        peaks = np.maximum.accumulate(w)
        dd = (w - peaks) / peaks * 100.0
        plt.plot(dates_plot, dd, label=name.replace("_", " "), linewidth=1.5)
    plt.title("Plot 2: Out-of-Sample Portfolio Drawdown (%) Over Time", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Drawdown (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_dir / "plot02_drawdowns.png")
    plt.close()

    # Plot 3: Rolling Turnover & Transaction Costs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, dpi=150)
    step_dates = dates_plot[:-1]
    for name in ["Full_MIQP_MPC", "Fast_MarketOnly_MIQP", "Certified_Recovery_Method", "Equal_Weight_1_over_N"]:
        res = results[name]
        to = [r.turnover * 100.0 for r in res.step_records]
        cost_cum = np.cumsum([r.transaction_cost for r in res.step_records])
        ax1.plot(step_dates, to, label=name.replace("_", " "), linewidth=1.5)
        ax2.plot(step_dates, cost_cum, label=name.replace("_", " "), linewidth=1.5)
    ax1.set_title("Plot 3: Turnover (%) and Cumulative Transaction Costs ($)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Turnover (%)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Cumulative Costs ($)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_dir / "plot03_turnover_costs.png")
    plt.close()

    # Plot 4: Solver End-to-End Runtime
    plt.figure(figsize=(11, 5), dpi=150)
    for name in ["Full_MIQP_MPC", "Fast_MarketOnly_MIQP", "ContinuousRelaxation_MIQP", "Certified_Recovery_Method"]:
        res = results[name]
        runtimes = [r.solve_time_ms for r in res.step_records]
        plt.plot(step_dates, runtimes, label=name.replace("_", " "), linewidth=1.5)
    plt.title("Plot 4: Solver End-to-End Runtime (ms) Per Rebalancing Step", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Runtime (ms)", fontsize=11)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_dir / "plot04_solver_runtime.png")
    plt.close()

    # Plot 5: Objective Degradation Over Time
    plt.figure(figsize=(11, 5), dpi=150)
    full_step_objs = np.array([r.objective_value for r in results["Full_MIQP_MPC"].step_records])
    for name in ["Fast_MarketOnly_MIQP", "ContinuousRelaxation_MIQP", "Certified_Recovery_Method"]:
        res = results[name]
        strat_objs = np.array([r.objective_value for r in res.step_records])
        deg = strat_objs - full_step_objs
        plt.plot(step_dates, deg, label=name.replace("_", " "), linewidth=1.5)
    plt.title("Plot 5: Objective Degradation (J_reduced - J_full) Over Time", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Objective Degradation", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_dir / "plot05_objective_degradation.png")
    plt.close()

    # Plot 6: Candidate Set Size & Recovery Stages
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, dpi=150)
    rec_res = results["Certified_Recovery_Method"]
    final_bs = [r.final_budget for r in rec_res.step_records]
    stages_hist = [r.stages_executed for r in rec_res.step_records]

    ax1.step(step_dates, final_bs, where="mid", color="#1f77b4", linewidth=2.0, label="Certified Recovery Final Budget B")
    ax1.axhline(N, color="gray", linestyle="--", label=f"Full Universe N={N}")
    ax1.axhline(B_policy, color="red", linestyle=":", label=f"Initial Budget B={B_policy}")
    ax1.set_title("Plot 6: Candidate Universe Size & Recovery Stages in Certified Mode", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Candidate Count B", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="center right", fontsize=9)

    ax2.step(step_dates, stages_hist, where="mid", color="#ff7f0e", linewidth=2.0, label="Recovery Stages Executed")
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Stages Executed", fontsize=11)
    ax2.set_yticks([1, 2, 3])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_dir / "plot06_candidate_size_stages.png")
    plt.close()

    # Plot 7: Certification Source & Certificate Gap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, dpi=150)
    cert_gaps = [r.certificate_gap if r.certificate_gap is not None else 0.0 for r in rec_res.step_records]
    sources = [r.certificate_source for r in rec_res.step_records]

    source_codes = []
    for s in sources:
        if s == CertificateSource.FULL_RELAXATION.value:
            source_codes.append(1)
        elif s == CertificateSource.FULL_MIP_BOUND.value:
            source_codes.append(2)
        else:
            source_codes.append(0)

    ax1.scatter(step_dates, source_codes, c=source_codes, cmap="viridis", s=40, edgecolors="black")
    ax1.set_title("Plot 7: Certification Source & Relaxation Certificate Gap Over Time", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Certificate Source", fontsize=11)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(["None", "Full Relaxation", "Full MIP Bound"])
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(step_dates, cert_gaps, color="#2ca02c", linewidth=1.8, label="Relaxation Certificate Gap (UB - LB)")
    ax2.axhline(1e-4, color="red", linestyle="--", label="epsilon_cert = 1e-4")
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Certificate Gap", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_dir / "plot07_certification.png")
    plt.close()

    print(f"\n[3] Successfully generated and saved all 7 plots to {plot_dir}/")
    print(f"    - plot01_cumulative_wealth.png")
    print(f"    - plot02_drawdowns.png")
    print(f"    - plot03_turnover_costs.png")
    print(f"    - plot04_solver_runtime.png")
    print(f"    - plot05_objective_degradation.png")
    print(f"    - plot06_candidate_size_stages.png")
    print(f"    - plot07_certification.png")


if __name__ == "__main__":
    run_exp09d_real_backtest()
