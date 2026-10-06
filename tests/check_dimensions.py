import numpy as np
import gurobipy as gp
from learning_augmented_portfolio.optimization.variables import MultiPeriodPortfolioParameters, MultiPeriodMarketState, SolverConfig
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer

def check_dims(n, H):
    params = MultiPeriodPortfolioParameters(
        risk_aversion=1.0,
        horizon=H,
        risk_free_rates=np.full(H, 0.0005),
        max_turnovers=np.full(H, 0.5),
        min_cash=np.full(H, 0.01),
        weight_upper_limits=np.full((H, n), 0.3),
        liquidity_limits=np.full((H, n), 0.2),
        prop_cost_coeffs=np.full((H, n), 0.001),
        fixed_cost_coeffs=np.full((H, n), 0.0001),
        min_holding_weights=np.full((H, n), 0.02),
        min_trade_sizes=np.full((H, n), 0.01),
        cardinality_limits=np.full(H, min(n, 10)),
        discount_factor=0.99
    )
    init_h = np.full(n, 0.9 / n)
    init_c = 0.1
    state = MultiPeriodMarketState(
        initial_holdings=init_h,
        initial_cash=init_c,
        expected_returns=np.full((H, n), 0.001),
        covariance_matrices=np.array([np.eye(n)*0.04 for _ in range(H)]),
        gross_returns=np.ones((H-1, n)) if H > 1 else None,
        asset_names=[f"A{i}" for i in range(n)]
    )
    opt = MultiPeriodMIQPOptimizer(params, SolverConfig(output_flag=False))
    res = opt.solve(state)
    m = res.metadata
    print(f"N={n}, H={H}: NumVars={m['num_vars']}, NumBinVars={m['num_bin_vars']}, NumConstrs={m['num_constrs']}")

for n, H in [(5, 1), (10, 1), (10, 3)]:
    check_dims(n, H)
