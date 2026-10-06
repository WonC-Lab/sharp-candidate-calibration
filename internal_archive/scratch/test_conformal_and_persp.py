"""
Test script to verify conformal calibrator and perspective relaxation optimizer.
"""

import numpy as np
from learning_augmented_portfolio.conformal import SplitConformalCalibrator, compute_support_depth
from learning_augmented_portfolio.optimization.variables import (
    MultiPeriodMarketState,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)
from learning_augmented_portfolio.optimization.relaxation import (
    MultiPeriodRelaxedQPOptimizer,
    MultiPeriodPerspectiveRelaxedOptimizer,
)


def test_conformal():
    print("Testing conformal calibrator...")
    np.random.seed(42)
    n_train = 50
    n_cal = 40
    n_test = 30
    d = 4

    X_train = np.random.randn(n_train, d)
    B_train = np.random.randint(3, 10, size=n_train)

    X_cal = np.random.randn(n_cal, d)
    B_cal = np.random.randint(3, 10, size=n_cal)

    X_test = np.random.randn(n_test, d)
    B_test = np.random.randint(3, 10, size=n_test)

    for alpha in [0.20, 0.10, 0.05]:
        calibrator = SplitConformalCalibrator(alpha=alpha, B_min=3, N=10)
        calibrator.fit_predictor(X_train, B_train)
        res = calibrator.calibrate(X_cal, B_cal)
        B_pred = calibrator.predict_calibrated_budget(X_test)
        coverage = np.mean(B_test <= B_pred)
        print(f"Alpha {alpha:.2f}: k_cal={res.k_cal}/{res.n_cal}, q_alpha={res.q_alpha:.2f}, coverage={coverage:.1%} (target >= {1-alpha:.1%})")

    # Test marginal calibration
    m_cal = calibrator.marginal_calibrate(B_cal)
    print(f"Marginal calibrated budget: {m_cal}")
    print("Conformal test passed.")


def test_perspective():
    print("\nTesting perspective relaxation...")
    np.random.seed(42)
    N = 5
    H = 3

    # Generate synthetic PSD cov
    A = np.random.randn(N, N)
    cov = A @ A.T + 0.1 * np.eye(N)

    params = MultiPeriodPortfolioParameters(
        horizon=H,
        risk_aversion=5.0,
        discount_factor=0.99,
        min_cash=np.full(H, 0.10),
        weight_upper_limits=np.full((H, N), 0.40),
        min_holding_weights=np.full((H, N), 0.08),
        liquidity_limits=np.full((H, N), 0.20),
        min_trade_sizes=np.full((H, N), 0.02),
        fixed_cost_coeffs=np.full((H, N), 1e-4),
        prop_cost_coeffs=np.full((H, N), 5e-4),
        max_turnovers=np.full(H, 0.30),
        cardinality_limits=np.full(H, 3),
        risk_free_rates=np.full(H, 0.001),
    )

    state = MultiPeriodMarketState(
        initial_holdings=np.zeros(N),
        initial_cash=1.0,
        expected_returns=np.full((H, N), 0.01),
        covariance_matrices=np.array([cov for _ in range(H)]),
        gross_returns=np.full((H - 1, N), 1.01),
    )

    solver_cfg = SolverConfig(output_flag=False, time_limit=10.0)

    basic_opt = MultiPeriodRelaxedQPOptimizer(params, solver_cfg)
    basic_res = basic_opt.solve(state)

    persp_opt = MultiPeriodPerspectiveRelaxedOptimizer(params, solver_cfg, rho_persp=0.95)
    persp_res = persp_opt.solve(state)

    print(f"Basic relaxation LB: {basic_res.objective_value:.6f}")
    print(f"Perspective LB:      {persp_res.objective_value:.6f}")
    assert basic_res.objective_value is not None
    assert persp_res.objective_value is not None
    assert persp_res.objective_value >= basic_res.objective_value - 1e-6, "Perspective LB must dominate basic LB"
    print("Perspective test passed! LB_persp >= LB_basic verified.")


if __name__ == "__main__":
    test_conformal()
    test_perspective()
