"""
Phase 9 Experiment 9E: Comprehensive Real-Market Sensitivity Analyses.
Analyzes performance robustness across 3 critical dimensions:
  1. Transaction Costs: Low (2 bps, $5), Base (5 bps, $10), High (15 bps, $30)
  2. Risk Aversion: gamma in {2.0, 5.0, 10.0}
  3. Candidate Budget: B in {3, 4, 6, 8, 10}

Compares Full MIQP, Fast MarketOnly MIQP, and Certified Recovery on the 2024 Test Period.
"""

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

from learning_augmented_portfolio.data import MarketDataStore
from learning_augmented_portfolio.dynamic.state import PortfolioState
from learning_augmented_portfolio.optimization.variables import (
    PortfolioParameters,
    MultiPeriodMarketState,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from learning_augmented_portfolio.optimization.reduced_miqp import MultiPeriodReducedMIQPOptimizer
from learning_augmented_portfolio.learning.active_set import SupportPredictorModel
from learning_augmented_portfolio.learning.candidate_set import construct_candidate_universe
from learning_augmented_portfolio.recovery.recovery_loop import NestedRecoveryOptimizer
from learning_augmented_portfolio.recovery.certificate import CertificateSource
from learning_augmented_portfolio.backtest.engine import RealMarketBacktestEngine
from learning_augmented_portfolio.backtest.records import compute_financial_metrics
from learning_augmented_portfolio.dataset.instance import (
    OptimizationInstance,
    OptimizationInstanceMetadata,
)


def run_exp09e_real_sensitivity() -> None:
    print("=" * 80)
    print("PHASE 9E: REAL-MARKET SENSITIVITY STUDIES (Costs, Risk Aversion, Budget B)")
    print("=" * 80)

    store = MarketDataStore.load_csv("artifacts/real_market/daily_prices.csv")
    universe = store.universe
    N = universe.n_assets
    H = 3

    all_dates = store.trading_dates
    T = len(all_dates)
    val_end = int(T * 0.75)
    test_start_date = all_dates[val_end]
    test_end_date = all_dates[-1]

    best_ml_model: SupportPredictorModel = joblib.load("artifacts/real_models/best_active_set_model.joblib")
    solver_cfg = SolverConfig(time_limit=15.0, mip_gap=1e-5, threads=1)

    def evaluate_configuration(
        gamma: float,
        prop_bps: float,
        fixed_cost: float,
        budget_B: int,
    ) -> Dict[str, Any]:
        rf_week = (1.0 + 0.03) ** (5.0 / 252.0) - 1.0
        single_p = PortfolioParameters(
            risk_aversion=gamma,
            risk_free_rate=rf_week,
            max_turnover=0.40,
            min_cash=0.10,
            weight_upper_limits=np.full(N, 0.40),
            liquidity_limits=np.full(N, 0.20),
            prop_cost_coeffs=np.full(N, prop_bps * 1e-4),
            cardinality_limit=4,
            min_holding_weights=np.full(N, 0.08),
            min_trade_sizes=np.full(N, 0.02),
            fixed_cost_coeffs=np.full(N, fixed_cost / 1_000_000.0),
        )
        base_params = MultiPeriodPortfolioParameters.from_single_period(single_p, horizon=H, discount_factor=0.99)

        engine = RealMarketBacktestEngine(
            data_store=store,
            params=base_params,
            solver_config=solver_cfg,
            initial_wealth=1_000_000.0,
            fixed_cost_dollars=fixed_cost,
            proportional_cost_bps=prop_bps,
            participation_rate=0.05,
            rebalance_frequency_days=5,
            risk_free_rate_annual=0.03,
            use_weekly_forecasts=True,
        )

        def make_instance(step_dt, st, step_p, mkt):
            meta = OptimizationInstanceMetadata(
                instance_id=f"sens_{step_dt}",
                calendar_date=step_dt,
                time_index=st.time_index,
                split="test",
                universe_id="LIQUID_BASKET_10",
                n_assets=N,
                horizon=H,
                dataset_version="1.0",
                config_hash="sens",
                git_commit=None,
                gurobi_version="12.0",
                python_version="3.13",
                solver_params={},
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
            return OptimizationInstance(metadata=meta, actual_state=st, params=step_p, market_state=mkt)

        # Full MIQP
        def strat_full(mkt, st, dt):
            step_p = engine.get_step_parameters(dt, st.actual_wealth)
            opt = MultiPeriodMIQPOptimizer(step_p, solver_cfg)
            t0 = time.perf_counter()
            sol = opt.solve(mkt)
            t_ms = (time.perf_counter() - t0) * 1000.0
            x0 = sol.post_trade_holdings[0] if (sol.is_feasible and sol.post_trade_holdings is not None) else st.normalized_weights
            b0 = sol.post_trade_cash[0] if (sol.is_feasible and sol.post_trade_cash is not None) else st.normalized_cash
            return x0, b0, {"solve_time_ms": t_ms, "objective_value": sol.objective_value}

        # Fast MarketOnly MIQP
        def strat_fast(mkt, st, dt):
            step_p = engine.get_step_parameters(dt, st.actual_wealth)
            inst = make_instance(dt, st, step_p, mkt)
            t0 = time.perf_counter()
            scores = best_ml_model.predict_instance_scores(inst)
            n_held = int(np.sum(st.normalized_weights > 1e-6))
            eff_B = max(budget_B, n_held)
            cand = construct_candidate_universe(scores, budget_B=eff_B, current_weights=st.normalized_weights, use_safe_augmentation=True)
            red_opt = MultiPeriodReducedMIQPOptimizer(step_p, solver_cfg)
            red_exec = red_opt.solve(mkt, cand, expand_solution=True)
            t_ms = (time.perf_counter() - t0) * 1000.0
            sol = red_exec.optimization_result
            x0 = sol.post_trade_holdings[0] if (sol.is_feasible and sol.post_trade_holdings is not None) else st.normalized_weights
            b0 = sol.post_trade_cash[0] if (sol.is_feasible and sol.post_trade_cash is not None) else st.normalized_cash
            return x0, b0, {"solve_time_ms": t_ms, "objective_value": sol.objective_value}

        # Certified Recovery
        def strat_rec(mkt, st, dt):
            step_p = engine.get_step_parameters(dt, st.actual_wealth)
            inst = make_instance(dt, st, step_p, mkt)
            t0 = time.perf_counter()
            scores = best_ml_model.predict_instance_scores(inst)
            rec_opt = NestedRecoveryOptimizer(step_p, solver_cfg, epsilon_cert=1e-4)
            n_held = int(np.sum(st.normalized_weights > 1e-6))
            eff_B = max(budget_B, n_held)
            ladder = sorted(list(set([eff_B, min(eff_B + 2, 10), 10])))
            rec_res = rec_opt.solve(inst, scores, budgets=ladder, fast_mode=False)
            t_ms = (time.perf_counter() - t0) * 1000.0
            sol = rec_res.final_solution
            x0 = sol.post_trade_holdings[0] if (sol is not None and sol.is_feasible and sol.post_trade_holdings is not None) else st.normalized_weights
            b0 = sol.post_trade_cash[0] if (sol is not None and sol.is_feasible and sol.post_trade_cash is not None) else st.normalized_cash
            return x0, b0, {
                "solve_time_ms": t_ms,
                "objective_value": rec_res.final_objective,
                "stages_executed": rec_res.stages_executed,
                "any_valid_certificate": rec_res.any_valid_certificate,
                "full_fallback_occurred": rec_res.full_fallback_occurred,
            }

        res_full = engine.run_backtest("Full_MIQP", test_start_date, test_end_date, strat_full)
        res_fast = engine.run_backtest("Fast_MIQP", test_start_date, test_end_date, strat_fast)
        res_rec = engine.run_backtest("Certified_Recovery", test_start_date, test_end_date, strat_rec)

        full_objs = np.array([r.objective_value for r in res_full.step_records])
        fast_objs = np.array([r.objective_value for r in res_fast.step_records])
        rec_objs = np.array([r.objective_value for r in res_rec.step_records])

        fast_deg = np.mean(fast_objs - full_objs)
        rec_deg = np.mean(rec_objs - full_objs)

        t_full = np.mean([r.solve_time_ms for r in res_full.step_records])
        t_fast = np.mean([r.solve_time_ms for r in res_fast.step_records])
        t_rec = np.mean([r.solve_time_ms for r in res_rec.step_records])

        fin_full = compute_financial_metrics("Full", res_full.wealth_series, np.array([r.turnover for r in res_full.step_records]), np.array([r.transaction_cost for r in res_full.step_records]), np.zeros(len(res_full.step_records)), np.zeros(len(res_full.step_records)))
        fin_fast = compute_financial_metrics("Fast", res_fast.wealth_series, np.array([r.turnover for r in res_fast.step_records]), np.array([r.transaction_cost for r in res_fast.step_records]), np.zeros(len(res_fast.step_records)), np.zeros(len(res_fast.step_records)))
        fin_rec = compute_financial_metrics("Rec", res_rec.wealth_series, np.array([r.turnover for r in res_rec.step_records]), np.array([r.transaction_cost for r in res_rec.step_records]), np.zeros(len(res_rec.step_records)), np.zeros(len(res_rec.step_records)))

        return {
            "full_ret": fin_full.annualized_return,
            "full_sharpe": fin_full.sharpe_ratio,
            "full_time_ms": t_full,
            "full_costs": fin_full.total_transaction_cost,
            "fast_ret": fin_fast.annualized_return,
            "fast_sharpe": fin_fast.sharpe_ratio,
            "fast_time_ms": t_fast,
            "fast_speedup": t_full / t_fast if t_fast > 0 else 1.0,
            "fast_costs": fin_fast.total_transaction_cost,
            "fast_obj_deg": fast_deg,
            "fast_exact_rec": np.mean(np.abs(fast_objs - full_objs) <= 1e-4),
            "rec_ret": fin_rec.annualized_return,
            "rec_sharpe": fin_rec.sharpe_ratio,
            "rec_time_ms": t_rec,
            "rec_speedup": t_full / t_rec if t_rec > 0 else 1.0,
            "rec_costs": fin_rec.total_transaction_cost,
            "rec_obj_deg": rec_deg,
            "rec_exact_rec": np.mean(np.abs(rec_objs - full_objs) <= 1e-4),
            "rec_stages": np.mean([r.stages_executed for r in res_rec.step_records]),
            "rec_cert_rate": np.mean([1.0 if r.any_valid_certificate else 0.0 for r in res_rec.step_records]),
        }

    # 1. Study 1: Cost Sensitivity
    print("\n[1] Running Transaction Cost Sensitivity (Low: 2bps/$5, Base: 5bps/$10, High: 15bps/$30)...")
    cost_configs = [
        ("Low (2bps/$5)", 2.0, 5.0),
        ("Base (5bps/$10)", 5.0, 10.0),
        ("High (15bps/$30)", 15.0, 30.0),
    ]
    cost_rows = []
    for label, prop, fix in cost_configs:
        out = evaluate_configuration(gamma=5.0, prop_bps=prop, fixed_cost=fix, budget_B=4)
        cost_rows.append({
            "CostRegime": label,
            "FullRet": f"{out['full_ret']*100:.1f}%",
            "FullSharpe": f"{out['full_sharpe']:.2f}",
            "FullCosts": f"${out['full_costs']:,.0f}",
            "FastRet": f"{out['fast_ret']*100:.1f}%",
            "FastSharpe": f"{out['fast_sharpe']:.2f}",
            "FastCosts": f"${out['fast_costs']:,.0f}",
            "FastSpeedup": f"{out['fast_speedup']:.2f}x",
            "RecRet": f"{out['rec_ret']*100:.1f}%",
            "RecSharpe": f"{out['rec_sharpe']:.2f}",
            "RecStages": f"{out['rec_stages']:.2f}",
            "RecCertRate": f"{out['rec_cert_rate']*100:.1f}%",
        })
    cost_df = pd.DataFrame(cost_rows)
    print("\nCOST SENSITIVITY TABLE:")
    print(cost_df.to_string(index=False))

    # 2. Study 2: Risk Aversion Sensitivity
    print("\n[2] Running Risk Aversion Sensitivity (gamma in {2.0, 5.0, 10.0})...")
    gamma_configs = [2.0, 5.0, 10.0]
    gamma_rows = []
    for g in gamma_configs:
        out = evaluate_configuration(gamma=g, prop_bps=5.0, fixed_cost=10.0, budget_B=4)
        gamma_rows.append({
            "RiskAversion(gamma)": g,
            "FullRet": f"{out['full_ret']*100:.1f}%",
            "FullSharpe": f"{out['full_sharpe']:.2f}",
            "FastRet": f"{out['fast_ret']*100:.1f}%",
            "FastSharpe": f"{out['fast_sharpe']:.2f}",
            "FastSpeedup": f"{out['fast_speedup']:.2f}x",
            "FastObjDeg": f"{out['fast_obj_deg']:.2e}",
            "RecRet": f"{out['rec_ret']*100:.1f}%",
            "RecSharpe": f"{out['rec_sharpe']:.2f}",
            "RecStages": f"{out['rec_stages']:.2f}",
            "RecCertRate": f"{out['rec_cert_rate']*100:.1f}%",
        })
    gamma_df = pd.DataFrame(gamma_rows)
    print("\nRISK AVERSION SENSITIVITY TABLE:")
    print(gamma_df.to_string(index=False))

    # 3. Study 3: Candidate Budget Sensitivity
    print("\n[3] Running Candidate Budget Sensitivity (B in {3, 4, 6, 8, 10})...")
    budget_configs = [3, 4, 6, 8, 10]
    budget_rows = []
    for b in budget_configs:
        out = evaluate_configuration(gamma=5.0, prop_bps=5.0, fixed_cost=10.0, budget_B=b)
        budget_rows.append({
            "Budget(B)": b,
            "Reduction": f"{1.0 - b/10.0:.0%}",
            "FastTime(ms)": f"{out['fast_time_ms']:.2f}",
            "FastSpeedup": f"{out['fast_speedup']:.2f}x",
            "FastExactRec": f"{out['fast_exact_rec']*100:.1f}%",
            "FastObjDeg": f"{out['fast_obj_deg']:.2e}",
            "FastRet": f"{out['fast_ret']*100:.1f}%",
            "FastSharpe": f"{out['fast_sharpe']:.2f}",
            "RecStages": f"{out['rec_stages']:.2f}",
            "RecCertRate": f"{out['rec_cert_rate']*100:.1f}%",
        })
    budget_df = pd.DataFrame(budget_rows)
    print("\nCANDIDATE BUDGET SENSITIVITY TABLE:")
    print(budget_df.to_string(index=False))

    # Save CSVs
    out_dir = Path("artifacts/real_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_df.to_csv(out_dir / "cost_sensitivity.csv", index=False)
    gamma_df.to_csv(out_dir / "gamma_sensitivity.csv", index=False)
    budget_df.to_csv(out_dir / "budget_sensitivity.csv", index=False)
    print(f"\n[4] Saved all sensitivity summaries to {out_dir}/")


if __name__ == "__main__":
    run_exp09e_real_sensitivity()
