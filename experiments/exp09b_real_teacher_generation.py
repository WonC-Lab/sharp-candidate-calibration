"""
Phase 9 Experiment 9B: Real-Data Teacher Generation.
Constructs chronological OptimizationInstances across Train and Validation splits:
  - Generates full-MIQP teacher solutions and continuous relaxations
  - Extracts teacher active-set labels Y^{union}, Y^{first}, Z^{union}, Z^{first}
  - Replays and serializes into artifacts/real_instances/manifest.csv
"""

from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd

from learning_augmented_portfolio.data import (
    RealMarketUniverse,
    TradingCalendar,
    MarketDataStore,
    ReturnCalculator,
    LiquidityCalculator,
)
from learning_augmented_portfolio.dynamic.state import PortfolioState
from learning_augmented_portfolio.forecasts import (
    ExpectedReturnEstimator,
    ReturnEstimatorType,
    CovarianceEstimator,
    CovarianceEstimatorType,
)
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from learning_augmented_portfolio.optimization.relaxation import MultiPeriodRelaxedQPOptimizer
from learning_augmented_portfolio.optimization.variables import (
    PortfolioParameters,
    MultiPeriodMarketState,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)
from learning_augmented_portfolio.dataset.instance import (
    OptimizationInstance,
    OptimizationInstanceMetadata,
    OptimizationLabels,
)
from learning_augmented_portfolio.dataset.serialization import save_instance


def run_exp09b_real_teacher_generation() -> None:
    print("=" * 80)
    print("PHASE 9B: REAL-DATA TEACHER INSTANCE GENERATION")
    print("=" * 80)

    store = MarketDataStore.load_csv("artifacts/real_market/daily_prices.csv")
    universe = store.universe
    N = universe.n_assets
    H = 3

    ret_calc = ReturnCalculator(store)
    liq_calc = LiquidityCalculator(store, default_participation_rate=0.05)
    exp_est = ExpectedReturnEstimator(ret_calc, lookback_window=60)
    cov_est = CovarianceEstimator(ret_calc, lookback_window=60)

    # Base parameters with weekly control grid
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

    cal = store.calendar
    all_dates = store.trading_dates
    T = len(all_dates)
    train_end = int(T * 0.60)
    val_end = int(T * 0.75)

    # Weekly rebalancing starting after warm-up (day 60)
    rebal_dates = cal.get_rebalancing_dates(all_dates[60], all_dates[val_end - 1], frequency_days=5)

    out_dir = Path("artifacts/real_instances")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1] Generating teacher instances on {len(rebal_dates)} rebalance dates...")
    manifest_rows = []

    curr_wealth = 1_000_000.0
    curr_w = np.zeros(N)
    curr_cash = 1.0

    for idx, d in enumerate(rebal_dates):
        split = "train" if d < all_dates[train_end] else "validation"
        inst_id = f"real_{split}_{idx:03d}_{d}"

        # 1. State
        state = PortfolioState(
            actual_wealth=curr_wealth,
            normalized_weights=curr_w.copy(),
            normalized_cash=curr_cash,
            time_index=idx,
            asset_names=universe.tickers,
        )

        # 2. Rescaled params for this step
        norm_liq = liq_calc.compute_normalized_liquidity_limits(d, curr_wealth)
        norm_f = np.full(N, 10.0 / curr_wealth)
        norm_a = np.full(N, 0.0005)

        step_params = MultiPeriodPortfolioParameters(
            risk_aversion=5.0,
            horizon=H,
            risk_free_rates=base_params.risk_free_rates,
            max_turnovers=base_params.max_turnovers,
            min_cash=base_params.min_cash,
            weight_upper_limits=base_params.weight_upper_limits,
            liquidity_limits=np.tile(norm_liq, (H, 1)),
            prop_cost_coeffs=np.tile(norm_a, (H, 1)),
            discount_factor=0.99,
            sector_indices=base_params.sector_indices,
            sector_bounds=base_params.sector_bounds,
            cardinality_limits=base_params.cardinality_limits,
            min_holding_weights=base_params.min_holding_weights,
            min_trade_sizes=base_params.min_trade_sizes,
            fixed_cost_coeffs=np.tile(norm_f, (H, 1)),
        )

        # 3. Forecasts (Weekly 5-trading-day planning unit)
        mu_mat = exp_est.construct_weekly_horizon_forecast(d, H, days_per_week=5)
        cov_mat = cov_est.construct_weekly_horizon_forecast(d, H, days_per_week=5)
        gross_mat = 1.0 + mu_mat[:-1]

        mkt = MultiPeriodMarketState(
            initial_holdings=curr_w.copy(),
            initial_cash=curr_cash,
            expected_returns=mu_mat,
            covariance_matrices=cov_mat,
            gross_returns=gross_mat,
            asset_names=universe.tickers,
        )

        # 4. Solve Teacher MIQP
        miqp_opt = MultiPeriodMIQPOptimizer(step_params, solver_cfg)
        sol = miqp_opt.solve(mkt)

        # 5. Solve Continuous Relaxation
        rel_opt = MultiPeriodRelaxedQPOptimizer(step_params, solver_cfg)
        rel_sol = rel_opt.solve(mkt)

        # 6. Extract Labels
        if sol.is_feasible and sol.holding_indicators is not None:
            y_union = (np.sum(sol.holding_indicators, axis=0) > 0).astype(int)
            y_first = sol.holding_indicators[0].astype(int)
            z_union = (np.sum(sol.trade_indicators, axis=0) > 0).astype(int)
            z_first = sol.trade_indicators[0].astype(int)
            labels = OptimizationLabels(
                y_first=y_first,
                y_union=y_union,
                z_first=z_first,
                z_union=z_union,
            )
        else:
            labels = None

        meta = OptimizationInstanceMetadata(
            instance_id=inst_id,
            calendar_date=d,
            time_index=idx,
            split=split,
            universe_id="LIQUID_BASKET_10",
            n_assets=N,
            horizon=H,
            dataset_version="2.0",
            config_hash="real_data_weekly_v2",
            git_commit=None,
            gurobi_version="12.0",
            python_version="3.13",
            solver_params={"mip_gap": 1e-5, "time_limit": 15.0},
            random_seed=42,
            solver_status=sol.status,
            label_quality="SOLVER_CERTIFIED" if sol.is_optimal else "HEURISTIC",
            is_trainable_label=sol.is_optimal,
            objective=sol.objective_value,
            best_bound=sol.best_bound,
            mip_gap=sol.mip_gap,
            runtime=sol.solve_time,
            work=sol.work or 0.0,
            node_count=sol.node_count or 0,
            sol_count=sol.sol_count or 1,
            used_fallback=False,
        )

        inst = OptimizationInstance(
            metadata=meta,
            actual_state=state,
            params=step_params,
            market_state=mkt,
            solution=sol,
            relaxation=rel_sol,
            labels=labels,
        )

        # Save individual instance file
        npz_path = save_instance(inst, out_dir)

        manifest_rows.append({
            "instance_id": inst_id,
            "filename": npz_path.name,
            "calendar_date": d,
            "split": split,
            "is_trainable": sol.is_optimal,
            "objective": sol.objective_value,
            "runtime": sol.solve_time,
            "node_count": sol.node_count or 0,
        })

        # Advance state to next rebalance date using realized return
        if idx < len(rebal_dates) - 1:
            d_next = rebal_dates[idx + 1]
            x0 = sol.post_trade_holdings[0] if sol.post_trade_holdings is not None else curr_w
            b0 = sol.post_trade_cash[0] if sol.post_trade_cash is not None else curr_cash

            trades = x0 - curr_w
            cost = float(np.sum(norm_a * np.abs(trades) + norm_f * (np.abs(trades) > 0.01)))
            wealth_post_cost = curr_wealth * (1.0 - cost)

            p_ret = ret_calc.get_realized_multi_day_return(d, d_next)
            growth = float(np.sum(x0 * (1.0 + p_ret)) + b0 * (1.0 + rf_week))
            curr_wealth = wealth_post_cost * growth
            curr_w = (x0 * (1.0 + p_ret)) / growth
            curr_cash = (b0 * (1.0 + rf_week)) / growth

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = out_dir / "manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"\n[2] Generated and saved {len(manifest_rows)} instances to {out_dir}")
    print(f"    - Train instances     : {len(manifest_df[manifest_df['split'] == 'train'])}")
    print(f"    - Validation instances: {len(manifest_df[manifest_df['split'] == 'validation'])}")
    print(f"    - Optimal solution rate: {manifest_df['is_trainable'].mean() * 100:.1f}%")
    print(f"    - Mean teacher solve time: {manifest_df['runtime'].mean() * 1000:.2f} ms")


if __name__ == "__main__":
    run_exp09b_real_teacher_generation()
