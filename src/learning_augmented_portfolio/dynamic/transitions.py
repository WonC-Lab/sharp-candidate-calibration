"""
Pure accounting and state transition functions for Rolling-Horizon MPC.
Strictly decoupled from optimization solvers and Gurobi interfaces.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from .state import PortfolioState


def validate_first_action_execution(
    post_trade_holdings: np.ndarray,
    post_trade_cash: float,
    transaction_cost: float,
) -> None:
    """
    Verify that executing the first planned action satisfies the normalized post-trade budget identity:
        1^T x^*_{0|t} + b^*_{0|t} == 1.0 - kappa^*_{0|t}
    """
    post_trade_sum = float(np.sum(post_trade_holdings) + post_trade_cash)
    expected_sum = float(1.0 - transaction_cost)
    if not np.isclose(post_trade_sum, expected_sum, atol=1e-5):
        raise ValueError(
            f"Execution accounting mismatch: 1^T x_0 + b_0 = {post_trade_sum:.6f} != 1 - kappa ({expected_sum:.6f})"
        )


def realized_market_transition(
    post_trade_holdings: np.ndarray,
    post_trade_cash: float,
    realized_risky_returns: np.ndarray,
    realized_risk_free_rate: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Propagate post-trade holdings through realized market returns between dates t and t+1.
    All outputs in this step are measured in units of the old numeraire V_t.

    Formulas:
        x_tilde^-_{t+1} = (1 + r^{real}_{t+1}) * x^*_{0|t}
        b_tilde^-_{t+1} = (1 + r^{real}_{f,t+1}) * b^*_{0|t}
        R^V_{t+1}       = 1^T x_tilde^-_{t+1} + b_tilde^-_{t+1}

    Args:
        post_trade_holdings: x^*_{0|t} in R_+^n.
        post_trade_cash: b^*_{0|t} in R.
        realized_risky_returns: r^{real}_{t+1} in R^n (must satisfy 1 + r_i >= 0).
        realized_risk_free_rate: r^{real}_{f,t+1} in R.

    Returns:
        Tuple of (x_tilde_minus, b_tilde_minus, wealth_growth_factor R^V).
    """
    n = len(post_trade_holdings)
    if realized_risky_returns.shape != (n,):
        raise ValueError(
            f"realized_risky_returns shape {realized_risky_returns.shape} != ({n},)"
        )
    if np.any(1.0 + realized_risky_returns < 0):
        raise ValueError("Realized gross returns 1 + r^{real} must be non-negative (limited liability)")

    gross_risky = 1.0 + realized_risky_returns
    gross_cash = 1.0 + realized_risk_free_rate

    x_tilde_minus = gross_risky * post_trade_holdings
    b_tilde_minus = float(gross_cash * post_trade_cash)

    wealth_growth_factor = float(np.sum(x_tilde_minus) + b_tilde_minus)
    if wealth_growth_factor <= 0:
        raise ValueError(
            f"Terminal economic bankruptcy: realized wealth growth factor R^V = {wealth_growth_factor:.6f} <= 0"
        )

    return x_tilde_minus, b_tilde_minus, wealth_growth_factor


def renormalize_state(
    x_tilde_minus: np.ndarray,
    b_tilde_minus: float,
    wealth_growth_factor: float,
    current_wealth: float,
    current_time_index: int,
) -> PortfolioState:
    """
    Renormalize the portfolio state by the new wealth numeraire V_{t+1} = V_t * R^V_{t+1}.

    Formulas:
        V_{t+1}       = V_t * R^V_{t+1}
        w_bar_{t+1}   = x_tilde^-_{t+1} / R^V_{t+1}
        c_bar_{t+1}   = b_tilde^-_{t+1} / R^V_{t+1}
        Identity: 1^T w_bar_{t+1} + c_bar_{t+1} == 1.0

    Returns:
        New PortfolioState at time t+1.
    """
    new_wealth = float(current_wealth * wealth_growth_factor)
    new_normalized_weights = x_tilde_minus / wealth_growth_factor
    new_normalized_cash = float(b_tilde_minus / wealth_growth_factor)

    return PortfolioState(
        actual_wealth=new_wealth,
        normalized_weights=new_normalized_weights,
        normalized_cash=new_normalized_cash,
        time_index=current_time_index + 1,
    )


def rescale_fixed_costs(
    dollar_fixed_costs: np.ndarray,
    actual_wealth: float,
) -> np.ndarray:
    """
    Rescale physical dollar fixed transaction charges into current wealth numeraire units:
        f_{i,t} = F_{i,t} / V_t
    """
    if actual_wealth <= 0:
        raise ValueError(f"actual_wealth must be positive, got {actual_wealth}")
    return dollar_fixed_costs / actual_wealth


def rescale_liquidity_limits(
    dollar_adv: np.ndarray,
    participation_rates: np.ndarray,
    actual_wealth: float,
) -> np.ndarray:
    """
    Rescale physical market dollar ADV and participation rates into current wealth numeraire units:
        L_{i,t} = (alpha_i * DADV_{i,t}) / V_t
    """
    if actual_wealth <= 0:
        raise ValueError(f"actual_wealth must be positive, got {actual_wealth}")
    return (participation_rates * dollar_adv) / actual_wealth
