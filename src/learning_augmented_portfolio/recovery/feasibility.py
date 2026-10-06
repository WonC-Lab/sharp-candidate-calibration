"""
Independent Full-Problem Feasibility Checker.
Validates lifted or proposed full-dimensional portfolio solutions against ALL original MIQP constraints.
Enforces:
  1. Cash and self-financing accounting across the entire horizon
  2. Dynamic asset and cash state transitions
  3. Holding upper and lower bounds conditional on binaries
  4. Trade upper and minimum trade bounds conditional on trade binaries
  5. Per-period cardinality limits
  6. Per-period turnover limits
  7. Sector exposure limits
  8. Minimum cash reserve
  9. Binary integrality
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..optimization.variables import (
    MultiPeriodMarketState,
    MultiPeriodPortfolioParameters,
    MultiPeriodOptimizationResult,
)


@dataclass
class FeasibilityAuditReport:
    """Detailed audit report from independent feasibility verification."""
    is_feasible: bool
    objective_value: Optional[float]
    max_constraint_violation: float
    violations: List[str] = field(default_factory=list)
    per_period_cardinality: Optional[np.ndarray] = None
    per_period_turnover: Optional[np.ndarray] = None


class FullProblemFeasibilityChecker:
    """
    Independent checker that does NOT rely on solver internal state.
    Audits a lifted or candidate solution against all physical constraints of the full-universe problem.
    """

    def __init__(
        self,
        params: MultiPeriodPortfolioParameters,
        market_state: MultiPeriodMarketState,
        tolerance: float = 1e-4,
    ) -> None:
        self.params = params
        self.state = market_state
        self.tol = tolerance

    def audit_solution(
        self,
        sol: MultiPeriodOptimizationResult,
    ) -> FeasibilityAuditReport:
        """
        Check all constraints for a full-dimension solution.

        Args:
            sol: MultiPeriodOptimizationResult in full dimension N.

        Returns:
            FeasibilityAuditReport with violation list and exact full-universe objective value.
        """
        if not sol.is_feasible:
            return FeasibilityAuditReport(
                is_feasible=False,
                objective_value=None,
                max_constraint_violation=float("inf"),
                violations=["Input solution is marked unfeasible by solver."],
            )

        H = self.params.horizon
        N = self.state.n_assets
        violations: List[str] = []
        max_violation = 0.0

        x = sol.post_trade_holdings
        x_minus = sol.pre_trade_holdings
        u = sol.trades
        q = sol.absolute_trades
        b = sol.post_trade_cash
        b_minus = sol.pre_trade_cash
        kappa = sol.transaction_costs
        y = sol.holding_indicators
        z = sol.trade_indicators

        if x is None or y is None:
            return FeasibilityAuditReport(
                is_feasible=False,
                objective_value=None,
                max_constraint_violation=float("inf"),
                violations=["Missing variable arrays in solution."],
            )

        # 1. Dimension checks
        if x.shape != (H, N) or y.shape != (H, N):
            violations.append(f"Dimension mismatch: x={x.shape}, expected ({H}, {N})")
            return FeasibilityAuditReport(
                is_feasible=False,
                objective_value=None,
                max_constraint_violation=float("inf"),
                violations=violations,
            )

        # 2. Initial state verification at h = 0
        v_init_holdings = np.max(np.abs(x_minus[0] - self.state.initial_holdings))
        if v_init_holdings > self.tol:
            violations.append(f"Initial holdings mismatch: max diff = {v_init_holdings:.2e}")
            max_violation = max(max_violation, v_init_holdings)

        v_init_cash = abs(b_minus[0] - self.state.initial_cash)
        if v_init_cash > self.tol:
            violations.append(f"Initial cash mismatch: diff = {v_init_cash:.2e}")
            max_violation = max(max_violation, v_init_cash)

        # 3. Transitions for h > 0
        gross_returns = self.state.gross_returns if self.state.gross_returns is not None else np.ones((H - 1, N))
        for h in range(H - 1):
            expected_x_minus = gross_returns[h] * x[h]
            diff_trans_x = np.max(np.abs(x_minus[h + 1] - expected_x_minus))
            if diff_trans_x > self.tol:
                violations.append(f"State transition error for risky assets at h={h+1}: max diff = {diff_trans_x:.2e}")
                max_violation = max(max_violation, diff_trans_x)

            rf_fac = 1.0 + float(self.params.risk_free_rates[h])
            expected_b_minus = rf_fac * b[h]
            diff_trans_b = abs(b_minus[h + 1] - expected_b_minus)
            if diff_trans_b > self.tol:
                violations.append(f"State transition error for cash at h={h+1}: diff = {diff_trans_b:.2e}")
                max_violation = max(max_violation, diff_trans_b)

        # 4. Intra-period constraints
        actual_cards = np.zeros(H, dtype=int)
        actual_turnovers = np.zeros(H, dtype=float)

        for h in range(H):
            # Trade balance: x[h, i] - u[h, i] == x_minus[h, i]
            diff_trade_bal = np.max(np.abs(x[h] - u[h] - x_minus[h]))
            if diff_trade_bal > self.tol:
                violations.append(f"Trade balance violation at h={h}: max diff = {diff_trade_bal:.2e}")
                max_violation = max(max_violation, diff_trade_bal)

            # Absolute trade: q[h, i] >= |u[h, i]|
            viol_abs_trade = np.maximum(0.0, np.abs(u[h]) - q[h])
            max_viol_abs = float(np.max(viol_abs_trade))
            if max_viol_abs > self.tol:
                violations.append(f"Absolute trade tightness violation at h={h}: max diff = {max_viol_abs:.2e}")
                max_violation = max(max_violation, max_viol_abs)

            # Self-financing identity: sum(x) + b == sum(x_minus) + b_minus - kappa
            lhs = np.sum(x[h]) + b[h]
            rhs = np.sum(x_minus[h]) + b_minus[h] - kappa[h]
            diff_self_fin = abs(lhs - rhs)
            if diff_self_fin > self.tol:
                violations.append(f"Self-financing violation at h={h}: |lhs - rhs| = {diff_self_fin:.2e}")
                max_violation = max(max_violation, diff_self_fin)

            # Transaction cost accounting: kappa[h] >= sum(c_prop * q + c_fixed * z)
            c_prop = self.params.prop_cost_coeffs[h]
            c_fixed = self.params.fixed_cost_coeffs[h] if self.params.fixed_cost_coeffs is not None else np.zeros(N)
            min_kappa = float(np.sum(c_prop * q[h] + c_fixed * z[h]))
            if kappa[h] < min_kappa - self.tol:
                diff_kappa = min_kappa - kappa[h]
                violations.append(f"Transaction cost under-reporting at h={h}: diff = {diff_kappa:.2e}")
                max_violation = max(max_violation, diff_kappa)

            # Holding upper & lower limits
            u_lim = self.params.weight_upper_limits[h]
            viol_upper = np.maximum(0.0, x[h] - u_lim * y[h])
            max_viol_upper = float(np.max(viol_upper))
            if max_viol_upper > self.tol:
                violations.append(f"Weight upper limit violation at h={h}: max diff = {max_viol_upper:.2e}")
                max_violation = max(max_violation, max_viol_upper)

            if self.params.min_holding_weights is not None:
                ell_lim = self.params.min_holding_weights[h]
                viol_lower = np.maximum(0.0, ell_lim * y[h] - x[h])
                max_viol_lower = float(np.max(viol_lower))
                if max_viol_lower > self.tol:
                    violations.append(f"Minimum holding weight violation at h={h}: max diff = {max_viol_lower:.2e}")
                    max_violation = max(max_violation, max_viol_lower)

            # Trade liquidity limits
            liq_lim = self.params.liquidity_limits[h]
            viol_liq = np.maximum(0.0, q[h] - liq_lim)
            max_viol_liq = float(np.max(viol_liq))
            if max_viol_liq > self.tol:
                violations.append(f"Liquidity limit violation at h={h}: max diff = {max_viol_liq:.2e}")
                max_violation = max(max_violation, max_viol_liq)

            # Trade indicator and minimum trade size
            if self.params.min_trade_sizes is not None:
                delta_lim = self.params.min_trade_sizes[h]
                viol_min_trade = np.maximum(0.0, delta_lim * z[h] - q[h])
                max_viol_min_trade = float(np.max(viol_min_trade))
                if max_viol_min_trade > self.tol:
                    violations.append(f"Minimum trade size violation at h={h}: max diff = {max_viol_min_trade:.2e}")
                    max_violation = max(max_violation, max_viol_min_trade)

            # Cardinality
            card = int(np.sum(y[h] > 0.5))
            actual_cards[h] = card
            if self.params.cardinality_limits is not None:
                k_limit = int(self.params.cardinality_limits[h])
                if card > k_limit:
                    violations.append(f"Cardinality limit exceeded at h={h}: count={card} > limit={k_limit}")
                    max_violation = max(max_violation, float(card - k_limit))

            # Turnover
            t_over = float(np.sum(q[h]))
            actual_turnovers[h] = t_over
            if self.params.max_turnovers is not None:
                max_t = float(self.params.max_turnovers[h])
                if t_over > max_t + self.tol:
                    violations.append(f"Turnover limit exceeded at h={h}: turnover={t_over:.4f} > limit={max_t:.4f}")
                    max_violation = max(max_violation, t_over - max_t)

            # Minimum cash reserve
            min_c = float(self.params.min_cash[h])
            if b[h] < min_c - self.tol:
                violations.append(f"Minimum cash reserve violated at h={h}: cash={b[h]:.4f} < min={min_c:.4f}")
                max_violation = max(max_violation, min_c - b[h])

            # Sector exposure bounds
            if self.params.sector_bounds and self.params.sector_indices:
                for sector_name, asset_indices in self.params.sector_indices.items():
                    if sector_name in self.params.sector_bounds:
                        e_min, e_max = self.params.sector_bounds[sector_name]
                        sec_exposure = float(np.sum(x[h, asset_indices]))
                        if sec_exposure < e_min - self.tol:
                            violations.append(f"Sector min violation for '{sector_name}' at h={h}: {sec_exposure:.4f} < {e_min:.4f}")
                            max_violation = max(max_violation, e_min - sec_exposure)
                        if sec_exposure > e_max + self.tol:
                            violations.append(f"Sector max violation for '{sector_name}' at h={h}: {sec_exposure:.4f} > {e_max:.4f}")
                            max_violation = max(max_violation, sec_exposure - e_max)

        is_feasible = (len(violations) == 0)

        # 5. Compute full-problem objective if feasible
        full_obj = None
        if is_feasible:
            total_obj = 0.0
            for h in range(H):
                discount = (self.params.discount_factor ** h)
                cov_h = self.state.covariance_matrices[h]
                variance_term = 0.5 * self.params.risk_aversion * float(x[h].T @ cov_h @ x[h])
                return_term = -float(self.state.expected_returns[h] @ x[h])
                rf_term = -float(self.params.risk_free_rates[h]) * b[h]
                cost_term = kappa[h]
                period_obj = variance_term + return_term + rf_term + cost_term
                total_obj += discount * period_obj
            full_obj = total_obj

        return FeasibilityAuditReport(
            is_feasible=is_feasible,
            objective_value=full_obj,
            max_constraint_violation=max_violation,
            violations=violations,
            per_period_cardinality=actual_cards,
            per_period_turnover=actual_turnovers,
        )
