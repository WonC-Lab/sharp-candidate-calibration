import numpy as np
import gurobipy as gp
from collections import Counter
from learning_augmented_portfolio.optimization.variables import MultiPeriodPortfolioParameters, MultiPeriodMarketState, SolverConfig
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer

def run_inspect(n, H):
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

    orig_opt = gp.Model.optimize
    constr_names = []
    num_qnzs = 0
    def my_opt(self, callback=None):
        nonlocal constr_names, num_qnzs
        self.update()
        constr_names = [c.ConstrName for c in self.getConstrs()]
        num_qnzs = self.NumQNZs
        return orig_opt(self, callback)
    gp.Model.optimize = my_opt
    try:
        opt = MultiPeriodMIQPOptimizer(params, SolverConfig(output_flag=False))
        opt.solve(state)
    finally:
        gp.Model.optimize = orig_opt

    prefix_counts = Counter()
    for name in constr_names:
        # group by pattern
        if "trade_def" in name:
            prefix_counts["trade_def (x - u == x^-)"] += 1
        elif "self_financing" in name:
            prefix_counts["self_financing (budget)"] += 1
        elif "transaction_cost" in name:
            prefix_counts["transaction_cost_def"] += 1
        elif "abs_trade_pos" in name:
            prefix_counts["abs_trade_pos (q - u >= 0)"] += 1
        elif "abs_trade_neg" in name:
            prefix_counts["abs_trade_neg (q + u >= 0)"] += 1
        elif "holding_ub" in name:
            prefix_counts["holding_ub (x <= U*y)"] += 1
        elif "holding_lb" in name:
            prefix_counts["holding_lb (x >= ell*y)"] += 1
        elif "cardinality" in name:
            prefix_counts["cardinality (sum y <= K)"] += 1
        elif "trade_ub" in name:
            prefix_counts["trade_ub_big_m (q <= M*z)"] += 1
        elif "trade_lb" in name:
            prefix_counts["trade_lb (q >= delta*z)"] += 1
        elif "turnover" in name:
            prefix_counts["turnover_limit (sum q <= tau)"] += 1
        elif "init_holdings" in name:
            prefix_counts["init_holdings (x_0^- == init)"] += 1
        elif "init_cash" in name:
            prefix_counts["init_cash (b_0^- == init)"] += 1
        elif "holding_transition" in name:
            prefix_counts["holding_transition (x_h+1^- == g*x_h)"] += 1
        elif "cash_transition" in name:
            prefix_counts["cash_transition (b_h+1^- == (1+rf)*b_h)"] += 1
        else:
            prefix_counts[name] += 1

    print(f"=== N={n}, H={H} (Total Constrs={len(constr_names)}, NumQNZs={num_qnzs}) ===")
    for k, v in sorted(prefix_counts.items()):
        print(f"  {k}: {v}")

for n, H in [(5, 1), (10, 1), (10, 3)]:
    run_inspect(n, H)
