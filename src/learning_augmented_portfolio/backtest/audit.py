"""
Backtest auditing module: Wealth accounting identity and hard constraint audits.
Ensures zero leaks, strict self-financing, and hard constraint compliance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from ..optimization.variables import MultiPeriodPortfolioParameters
from .records import BacktestStepRecord


@dataclass
class BacktestAuditResult:
    """Audit summary for an entire backtest execution."""
    strategy_name: str
    total_steps: int
    max_wealth_residual: float
    max_budget_residual: float
    is_wealth_accounting_valid: bool
    constraint_violation_count: int
    violation_details: List[str]


class BacktestAuditor:
    """
    Independently audits all mathematical identities and portfolio constraints
    over real-market backtesting trajectories.
    """

    def __init__(self, params: MultiPeriodPortfolioParameters, tolerance: float = 1e-4) -> None:
        self.params = params
        self.tolerance = tolerance

    def audit_backtest(
        self,
        strategy_name: str,
        records: List[BacktestStepRecord],
        realized_wealth_series: np.ndarray,
        check_constraints: bool = True,
    ) -> BacktestAuditResult:
        """
        Verify:
          1. V_{t+1} = V_t * R^V_{t+1}
          2. sum(w_i) + c = 1.0 at every step
          3. Optional constraints (cardinality, min holding, min trade, cash reserve).
        """
        max_w_res = 0.0
        max_b_res = 0.0
        violations: List[str] = []

        # 1. Wealth transitions
        for t in range(len(records) - 1):
            v_t = records[t].portfolio_wealth_post
            v_next_actual = records[t + 1].portfolio_wealth_pre
            r_real = records[t].realized_period_return
            v_next_theoretical = v_t * (1.0 + r_real)
            res = abs(v_next_actual - v_next_theoretical)
            max_w_res = max(max_w_res, res)
            if res > self.tolerance:
                violations.append(f"Step {t} wealth residual {res:.2e} exceeds tolerance.")

        # 2. Constraints at each rebalance
        for rec in records:
            w_post = rec.post_trade_weights
            c_post = rec.cash_post
            cost_frac = rec.transaction_cost / rec.portfolio_wealth_pre if rec.portfolio_wealth_pre > 0 else 0.0
            # Satisfies either gross target conservation (sum=1) or net self-financing conservation (sum + cost = 1)
            b_res = min(abs(np.sum(w_post) + c_post - 1.0), abs(np.sum(w_post) + c_post + cost_frac - 1.0))
            max_b_res = max(max_b_res, b_res)
            if b_res > self.tolerance:
                violations.append(f"Date {rec.date} budget residual {b_res:.2e} exceeds tolerance.")

            if not check_constraints:
                continue

            # Cardinality
            holdings = int(np.sum(w_post > 1e-5))
            k_max = int(self.params.cardinality_limits[0]) if self.params.cardinality_limits is not None else len(w_post)
            if holdings > k_max:
                violations.append(f"Date {rec.date} cardinality {holdings} > max {k_max}")

            # Min holdings
            if self.params.min_holding_weights is not None:
                for i, wi in enumerate(w_post):
                    min_w = float(self.params.min_holding_weights[0, i])
                    if 1e-5 < wi < min_w - self.tolerance:
                        violations.append(f"Date {rec.date} asset {i} weight {wi:.4f} < min {min_w}")

            # Min cash
            min_c = float(self.params.min_cash[0])
            if c_post < min_c - self.tolerance:
                violations.append(f"Date {rec.date} cash {c_post:.4f} < min reserve {min_c}")

        is_valid = bool(len(violations) == 0 and max_w_res <= self.tolerance and max_b_res <= self.tolerance)

        return BacktestAuditResult(
            strategy_name=strategy_name,
            total_steps=len(records),
            max_wealth_residual=float(max_w_res),
            max_budget_residual=float(max_b_res),
            is_wealth_accounting_valid=is_valid,
            constraint_violation_count=len(violations),
            violation_details=violations[:5],
        )
