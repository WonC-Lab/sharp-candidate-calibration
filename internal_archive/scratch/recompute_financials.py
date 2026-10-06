import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from learning_augmented_portfolio.data import MarketDataStore
from learning_augmented_portfolio.optimization.variables import (
    PortfolioParameters,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from learning_augmented_portfolio.optimization.reduced_miqp import MultiPeriodReducedMIQPOptimizer
from learning_augmented_portfolio.learning.active_set import SupportPredictorModel
from learning_augmented_portfolio.learning.baselines import ContinuousRelaxationYBaseline
from learning_augmented_portfolio.learning.candidate_set import construct_candidate_universe
from learning_augmented_portfolio.recovery.recovery_loop import NestedRecoveryOptimizer
from learning_augmented_portfolio.backtest.engine import RealMarketBacktestEngine
from learning_augmented_portfolio.backtest.baselines import EqualWeightBaseline, BuyAndHoldBaseline

store = MarketDataStore.load_csv("artifacts/real_market/daily_prices.csv")
universe = store.universe
N = universe.n_assets
H = 3
all_dates = store.trading_dates
val_end = int(len(all_dates) * 0.75)
test_start_date = all_dates[val_end]
test_end_date = all_dates[-1]

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

model: SupportPredictorModel = joblib.load("artifacts/real_models/best_active_set_model.joblib")

def strat_full_miqp(mkt, st, dt):
    opt = MultiPeriodMIQPOptimizer(base_params, solver_cfg)
    sol = opt.optimize(st, mkt)
    return sol.first_step_weights, sol.first_step_cash, sol.metadata

def strat_fast_market_only(mkt, st, dt):
    cands = construct_candidate_universe(st, method="heuristic", universe_size=N, max_candidates=4)
    opt = MultiPeriodReducedMIQPOptimizer(base_params, solver_cfg)
    sol = opt.optimize(st, mkt, cands)
    return sol.first_step_weights, sol.first_step_cash, sol.metadata

def strat_continuous_relaxation(mkt, st, dt):
    baseline = ContinuousRelaxationYBaseline()
    cands = baseline.select_candidates(st, mkt, base_params, target_budget=4)
    opt = MultiPeriodReducedMIQPOptimizer(base_params, solver_cfg)
    sol = opt.optimize(st, mkt, cands)
    return sol.first_step_weights, sol.first_step_cash, sol.metadata

def strat_best_ml(mkt, st, dt):
    cands = model.predict_candidates(st, mkt, target_budget=4)
    opt = MultiPeriodReducedMIQPOptimizer(base_params, solver_cfg)
    sol = opt.optimize(st, mkt, cands)
    return sol.first_step_weights, sol.first_step_cash, sol.metadata

def strat_certified_recovery(mkt, st, dt):
    nested_opt = NestedRecoveryOptimizer(
        base_params=base_params,
        model=model,
        candidate_budgets=[4, 6, 10],
        epsilon_cert=1e-4,
        solver_config=solver_cfg,
    )
    rec_res = nested_opt.optimize(st, mkt)
    sol = rec_res.final_solution
    meta = dict(sol.metadata)
    meta.update({
        "certificate_type": rec_res.certificate.cert_type.value,
        "certificate_gap": rec_res.certificate.gap,
        "certificate_source": rec_res.certificate.source.value,
        "stages_executed": rec_res.stages_executed,
        "final_budget": rec_res.final_budget,
        "full_fallback_occurred": rec_res.full_fallback_occurred,
    })
    return sol.first_step_weights, sol.first_step_cash, meta

eq_baseline = EqualWeightBaseline(base_params)
def strat_equal_weight(mkt, st, dt):
    w, c = eq_baseline.get_target_weights(N)
    return w, c, {"solve_time_ms": 0.01, "node_count": 0, "work": 0.0, "mip_gap": 0.0}

bh_baseline = BuyAndHoldBaseline(initial_weights=np.zeros(N), initial_cash=1.0)
def strat_buy_and_hold(mkt, st, dt):
    if not bh_baseline.has_initialized:
        w, c = eq_baseline.get_target_weights(N)
        bh_baseline.initial_weights = w
        bh_baseline.initial_cash = c
    w, c = bh_baseline.get_action(st.normalized_weights, st.normalized_cash)
    return w, c, {"solve_time_ms": 0.01, "node_count": 0, "work": 0.0, "mip_gap": 0.0}

strategies = {
    "Full MIQP MPC": strat_full_miqp,
    "Fast MarketOnly MIQP": strat_fast_market_only,
    "ContinuousRelaxation MIQP": strat_continuous_relaxation,
    "Best ML Candidate": strat_best_ml,
    "Certified Recovery": strat_certified_recovery,
    "Equal Weight (1/N)": strat_equal_weight,
    "Buy and Hold": strat_buy_and_hold,
}

res_map = {}
for name, fn in strategies.items():
    res_map[name] = engine.run_backtest(name, test_start_date, test_end_date, fn)

print("\n" + "="*110)
print("EXACT FINANCIAL RECOMPUTATION FROM 51-WEEK WEALTH TRAJECTORIES")
print("="*110)

wealth_df_dict = {"Date": res_map["Full MIQP MPC"].rebalance_dates}
weekly_ret_dict = {"Date": res_map["Full MIQP MPC"].rebalance_dates[1:]}

table_rows = []
for name, res in res_map.items():
    w = res.wealth_series
    wealth_df_dict[name] = w
    v0, vt = w[0], w[-1]
    tot_ret = vt / v0 - 1.0
    r_w = w[1:] / w[:-1] - 1.0
    weekly_ret_dict[name] = r_w
    N_w = len(r_w) # 51
    
    cagr = (vt / v0) ** (52.0 / 51.0) - 1.0
    ann_arith = np.mean(r_w) * 52.0
    ann_vol = np.std(r_w, ddof=1) * np.sqrt(52.0)
    sharpe_cagr_02 = (cagr - 0.02) / ann_vol
    rf_w_02 = 0.02 / 52.0
    sharpe_weekly_02 = np.sqrt(52.0) * np.mean(r_w - rf_w_02) / np.std(r_w, ddof=1)
    sharpe_weekly_03 = np.sqrt(52.0) * np.mean(r_w - rf_week) / np.std(r_w, ddof=1)
    peaks = np.maximum.accumulate(w)
    max_dd = np.min((w - peaks) / peaks)
    
    table_rows.append({
        "Strategy": name,
        "FinalWealth": vt,
        "TotRet": tot_ret,
        "CAGR": cagr,
        "AnnArith": ann_arith,
        "AnnVol": ann_vol,
        "Sharpe_CAGR_02": sharpe_cagr_02,
        "Sharpe_Wk_02": sharpe_weekly_02,
        "Sharpe_Wk_03": sharpe_weekly_03,
        "MaxDD": max_dd
    })

pd.DataFrame(wealth_df_dict).to_csv("artifacts/real_backtest/wealth_trajectories.csv", index=False)
pd.DataFrame(weekly_ret_dict).to_csv("artifacts/real_backtest/weekly_returns.csv", index=False)
print("Saved artifacts/real_backtest/wealth_trajectories.csv and weekly_returns.csv")

for r in table_rows:
    print(f"{r['Strategy']:<26} | TotRet={r['TotRet']*100:6.2f}% | CAGR={r['CAGR']*100:5.2f}% | Arith={r['AnnArith']*100:5.2f}% | Vol={r['AnnVol']*100:5.2f}% | Sh_CAGR_02={r['Sharpe_CAGR_02']:4.2f} | Sh_Wk_02={r['Sharpe_Wk_02']:4.2f} | Sh_Wk_03={r['Sharpe_Wk_03']:4.2f} | MaxDD={r['MaxDD']*100:6.2f}% | FinalWealth=${r['FinalWealth']:11,.2f}")
