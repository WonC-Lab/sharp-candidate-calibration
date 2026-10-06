"""
Tight Big-M computation and validation for portfolio rebalancing.
Supports both single-period (Phase 2) and dynamic multi-period (Phase 3) models.

Mathematical Formulation:
    Single-period (h = 0):
        M_{i,0}^{sell} = min( max(0, x_{i,0}^-), L_{i,0}, tau_0^max )
        M_{i,0}^{buy}  = min( max(0, U_{i,0} - x_{i,0}^-), L_{i,0}, tau_0^max )
        M_{i,0}        = max( M_{i,0}^{sell}, M_{i,0}^{buy} )

    Multi-period future steps (h >= 1):
        x_{i,h}^- <= x_bar_{i,h}^- = g_{i,h-1} * U_{i,h-1}
        M_{i,h}^{sell} = min( x_bar_{i,h}^-, L_{i,h}, tau_h^max )
        M_{i,h}^{buy}  = min( U_{i,h}, L_{i,h}, tau_h^max )
        M_{i,h}        = max( M_{i,h}^{sell}, M_{i,h}^{buy} )
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np


def compute_tight_big_m(
    current_weights: np.ndarray,
    weight_upper_limits: np.ndarray,
    liquidity_limits: np.ndarray,
    max_turnover: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mathematically tight, non-arbitrary Big-M upper bounds on trade sizes for single period.

    Args:
        current_weights: w_bar in R^n, pre-trade risky weights.
        weight_upper_limits: U in R^n, post-trade holding limits.
        liquidity_limits: L in R^n, per-asset trading limits.
        max_turnover: tau_max >= 0, maximum allowable total portfolio turnover.

    Returns:
        Tuple of (M_sell, M_buy, M_symmetric), all in R^n.
    """
    n = len(current_weights)
    if not (len(weight_upper_limits) == len(liquidity_limits) == n):
        raise ValueError("All input arrays must have the same length")
    if max_turnover < 0:
        raise ValueError(f"max_turnover must be non-negative, got {max_turnover}")

    max_sell_holding = np.maximum(0.0, current_weights)
    m_sell = np.minimum(max_sell_holding, np.minimum(liquidity_limits, max_turnover))

    max_buy_room = np.maximum(0.0, weight_upper_limits - current_weights)
    m_buy = np.minimum(max_buy_room, np.minimum(liquidity_limits, max_turnover))

    m_symmetric = np.maximum(m_sell, m_buy)
    return m_sell, m_buy, m_symmetric


def compute_dynamic_tight_big_m(
    initial_holdings: np.ndarray,
    weight_upper_limits: np.ndarray,
    liquidity_limits: np.ndarray,
    max_turnovers: np.ndarray,
    gross_returns: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute dynamic tight Big-M bounds across a multi-period planning horizon h = 0, ..., H-1.

    Args:
        initial_holdings: x_0^- in R^n, pre-trade risky holdings at h=0.
        weight_upper_limits: U of shape (H, n), holding caps.
        liquidity_limits: L of shape (H, n), per-period trading limits.
        max_turnovers: tau^max of shape (H,), per-period turnover limits.
        gross_returns: g of shape (H-1, n) or None, where g_{h-1} = 1 + r_hat_{h-1} >= 0.

    Returns:
        Tuple of (M_sell, M_buy, M_symmetric), all of shape (H, n).
    """
    H, n = weight_upper_limits.shape
    if initial_holdings.shape != (n,):
        raise ValueError(f"initial_holdings shape {initial_holdings.shape} != ({n},)")
    if liquidity_limits.shape != (H, n):
        raise ValueError(f"liquidity_limits shape {liquidity_limits.shape} != ({H}, {n})")
    if max_turnovers.shape != (H,):
        raise ValueError(f"max_turnovers shape {max_turnovers.shape} != ({H},)")

    if H > 1:
        if gross_returns is None:
            # Default to flat gross returns g = 1.0 (0% return) if not provided
            g = np.ones((H - 1, n))
        else:
            if gross_returns.shape != (H - 1, n):
                raise ValueError(f"gross_returns shape {gross_returns.shape} != ({H - 1}, {n})")
            if np.any(gross_returns < 0):
                raise ValueError("gross_returns must be non-negative (limited liability)")
            g = gross_returns
    else:
        g = np.empty((0, n))

    m_sell = np.zeros((H, n))
    m_buy = np.zeros((H, n))
    m_symmetric = np.zeros((H, n))

    # h = 0: Known deterministic pre-trade holdings x_0^-
    m_s0, m_b0, m_sym0 = compute_tight_big_m(
        current_weights=initial_holdings,
        weight_upper_limits=weight_upper_limits[0],
        liquidity_limits=liquidity_limits[0],
        max_turnover=float(max_turnovers[0]),
    )
    m_sell[0] = m_s0
    m_buy[0] = m_b0
    m_symmetric[0] = m_sym0

    # h >= 1: Endogenous pre-trade holdings x_h^- <= g_{h-1} * U_{h-1}
    for h in range(1, H):
        tau_h = float(max_turnovers[h])
        l_h = liquidity_limits[h]
        u_h = weight_upper_limits[h]

        # Valid deterministic upper bound on pre-trade holding x_{i,h}^-
        x_bar_prev_cap = g[h - 1] * weight_upper_limits[h - 1]

        m_sell[h] = np.minimum(x_bar_prev_cap, np.minimum(l_h, tau_h))
        m_buy[h] = np.minimum(u_h, np.minimum(l_h, tau_h))
        m_symmetric[h] = np.maximum(m_sell[h], m_buy[h])

    return m_sell, m_buy, m_symmetric


def validate_big_m(
    m_symmetric: np.ndarray,
    min_trade_sizes: Optional[np.ndarray],
    asset_names: Optional[List[str]] = None,
) -> None:
    """
    Validate that minimum trade size delta_i does not exceed maximum feasible trade size M_i.
    Works for both 1D (single period) and 2D (multi-period) arrays.
    """
    if min_trade_sizes is None:
        return

    if m_symmetric.ndim == 1:
        n = len(m_symmetric)
        for i in range(n):
            delta_i = min_trade_sizes[i]
            m_i = m_symmetric[i]
            if delta_i > m_i + 1e-9:
                name = asset_names[i] if asset_names and i < len(asset_names) else f"Asset_{i}"
                raise ValueError(
                    f"Infeasible trade configuration for {name}: minimum trade size delta_{i}={delta_i:.6f} "
                    f"exceeds tight Big-M bound M_{i}={m_i:.6f} (M_i < delta_i)"
                )
    elif m_symmetric.ndim == 2:
        H, n = m_symmetric.shape
        delta_arr = min_trade_sizes if min_trade_sizes.ndim == 2 else np.tile(min_trade_sizes, (H, 1))
        for h in range(H):
            for i in range(n):
                delta_hi = delta_arr[h, i]
                m_hi = m_symmetric[h, i]
                if delta_hi > m_hi + 1e-9:
                    name = asset_names[i] if asset_names and i < len(asset_names) else f"Asset_{i}"
                    raise ValueError(
                        f"Infeasible trade configuration at horizon h={h} for {name}: "
                        f"minimum trade size delta_{h},{i}={delta_hi:.6f} "
                        f"exceeds dynamic Big-M bound M_{h},{i}={m_hi:.6f} (M_{h},{i} < delta_{h},{i})"
                    )
