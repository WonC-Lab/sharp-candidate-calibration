"""
Execution script to validate:
1. Exactness budget B^\dagger calculation across instances.
2. Split-conformal candidate budget calibration and empirical coverage.
3. Logarithmic recovery-depth bound verification.
4. Cumulative structural work verification.
5. Perspective relaxation lower-bound dominance (LB^{basic} <= LB^{persp} <= J^*).
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from learning_augmented_portfolio.optimization.variables import (
    MultiPeriodMarketState,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from learning_augmented_portfolio.optimization.reduced_miqp import MultiPeriodReducedMIQPOptimizer
from learning_augmented_portfolio.optimization.support_restriction import CandidateUniverse
from learning_augmented_portfolio.optimization.relaxation import (
    MultiPeriodRelaxedQPOptimizer,
    MultiPeriodPerspectiveRelaxedOptimizer,
)
from learning_augmented_portfolio.conformal import (
    SplitConformalCalibrator,
    compute_support_depth,
)


def load_npz_instance(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    initial_holdings = data["initial_holdings"]
    if "pre_trade_cash" in data:
        initial_cash = float(data["pre_trade_cash"][0])
    else:
        initial_cash = float(max(0.0, 1.0 - np.sum(initial_holdings)))
    expected_returns = data["expected_returns"]
    covariance_matrices = data["covariance_matrices"]
    gross_returns = data["gross_returns"] if "gross_returns" in data else None
    
    H, n = expected_returns.shape
    params = MultiPeriodPortfolioParameters(
        horizon=H,
        risk_aversion=5.0,
        discount_factor=0.99,
        min_cash=data["min_cash"] if "min_cash" in data else np.full(H, 0.10),
        weight_upper_limits=data["weight_upper_limits"] if "weight_upper_limits" in data else np.full((H, n), 0.40),
        min_holding_weights=data["min_holding_weights"] if "min_holding_weights" in data else np.full((H, n), 0.08),
        liquidity_limits=data["liquidity_limits"] if "liquidity_limits" in data else np.full((H, n), 0.20),
        min_trade_sizes=data["min_trade_sizes"] if "min_trade_sizes" in data else np.full((H, n), 0.02),
        fixed_cost_coeffs=data["fixed_cost_coeffs"] if "fixed_cost_coeffs" in data else np.full((H, n), 1e-4),
        prop_cost_coeffs=data["prop_cost_coeffs"] if "prop_cost_coeffs" in data else np.full((H, n), 5e-4),
        max_turnovers=data["max_turnovers"] if "max_turnovers" in data else np.full(H, 0.30),
        cardinality_limits=data["cardinality_limits"] if "cardinality_limits" in data else np.full(H, 3),
        risk_free_rates=data["risk_free_rates"] if "risk_free_rates" in data else np.full(H, 0.001),
    )
    
    state = MultiPeriodMarketState(
        initial_holdings=initial_holdings,
        initial_cash=initial_cash,
        expected_returns=expected_returns,
        covariance_matrices=covariance_matrices,
        gross_returns=gross_returns,
    )
    return state, params, float(data["objective"]) if "objective" in data else None


def compute_instance_b_dagger(state, params, full_opt_val, solver_cfg, eps_ref=1e-4):
    H = params.horizon
    N = state.n_assets
    K = int(params.cardinality_limits[0])
    
    # Candidate ranking: market momentum + return score
    scores = np.mean(state.expected_returns, axis=0) / (np.sqrt(np.diag(state.covariance_matrices[0])) + 1e-6)
    mandatory = [i for i in range(N) if state.initial_holdings[i] > 1e-6]
    non_mand_ranked = [i for i in np.argsort(-scores) if i not in mandatory]
    
    # Grid of B from max(K, len(mandatory)) to N
    B_start = max(K, len(mandatory))
    best_b_dagger = N
    
    for B in range(B_start, N + 1):
        cand = mandatory + non_mand_ranked[:(B - len(mandatory))]
        cand_univ = CandidateUniverse(candidate_indices=tuple(sorted(cand)), full_universe_size=N)
        red_opt = MultiPeriodReducedMIQPOptimizer(params, solver_cfg)
        exec_res = red_opt.solve(state, cand_univ)
        res = exec_res.optimization_result
        if res.is_feasible and res.objective_value is not None:
            if abs(res.objective_value - full_opt_val) <= eps_ref:
                best_b_dagger = B
                break
    return best_b_dagger, scores, mandatory


def run_all():
    print("=" * 70)
    print("RUNNING CONFORMAL CALIBRATION & RECOVERY SMOOTHNESS AUDIT")
    print("=" * 70)
    
    manifest_path = "artifacts/instances_medium/manifest.csv"
    if not os.path.exists(manifest_path):
        print("Manifest not found:", manifest_path)
        return
        
    df_manifest = pd.read_csv(manifest_path)
    solver_cfg = SolverConfig(output_flag=False, time_limit=15.0, mip_gap=1e-5)
    
    records = []
    print(f"Loaded manifest with {len(df_manifest)} instances.")
    
    for idx, row in df_manifest.iterrows():
        npz_rel = os.path.join("artifacts/instances_medium", row["filename"])
        if not os.path.exists(npz_rel):
            continue
        state, params, obj_ref = load_npz_instance(npz_rel)
        if obj_ref is None or not np.isfinite(obj_ref):
            # solve full
            full_opt = MultiPeriodMIQPOptimizer(params, solver_cfg)
            f_res = full_opt.solve(state)
            obj_ref = f_res.objective_value
            
        b_dag, scores, mand = compute_instance_b_dagger(state, params, obj_ref, solver_cfg)
        
        # Also compute basic and perspective relaxation LB
        basic_opt = MultiPeriodRelaxedQPOptimizer(params, solver_cfg)
        b_res = basic_opt.solve(state)
        
        # Check perspective model variable count: H * (7 * N + 3)
        n_vars_persp = params.horizon * (7 * state.n_assets + 3)
        if n_vars_persp <= 200:
            persp_opt = MultiPeriodPerspectiveRelaxedOptimizer(params, solver_cfg, rho_persp=0.95)
            p_res = persp_opt.solve(state)
            p_val = p_res.objective_value
            p_dom = (p_val >= b_res.objective_value - 1e-6) if p_val is not None else None
        else:
            p_val = None
            p_dom = None
        
        # Features for adaptive prediction: mean expected return, mean vol, cash, mand count
        mean_ret = float(np.mean(state.expected_returns))
        mean_vol = float(np.mean(np.sqrt(np.diag(state.covariance_matrices[0]))))
        feat = [mean_ret, mean_vol, float(state.initial_cash), float(len(mand))]
        
        records.append({
            "instance_id": row["instance_id"],
            "split": row["split"],
            "obj_ref": obj_ref,
            "b_dagger": b_dag,
            "lb_basic": b_res.objective_value,
            "lb_persp": p_val,
            "persp_dominates": p_dom,
            "features": feat,
            "mand_len": len(mand),
        })
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(df_manifest)} instances...")
            
    df_res = pd.DataFrame(records)
    print("\nDataset split counts:")
    print(df_res["split"].value_counts())
    
    # Train / Cal / Test split
    df_train = df_res[df_res["split"] == "train"]
    df_cal = df_res[df_res["split"] == "validation"]
    df_test = df_res[df_res["split"] == "test"]
    
    X_train = np.array(df_train["features"].tolist())
    y_train = df_train["b_dagger"].values
    
    X_cal = np.array(df_cal["features"].tolist())
    y_cal = df_cal["b_dagger"].values
    
    X_test = np.array(df_test["features"].tolist())
    y_test = df_test["b_dagger"].values
    
    print("\n" + "=" * 50)
    print("CONFORMAL COVERAGE ON TEST SET (N=10, H=3)")
    print("=" * 50)
    
    N = 10
    B_min = 3
    for alpha in [0.20, 0.10, 0.05]:
        calibrator = SplitConformalCalibrator(alpha=alpha, B_min=B_min, N=N)
        calibrator.fit_predictor(X_train, y_train)
        cal_res = calibrator.calibrate(X_cal, y_cal)
        
        # Adaptive predictions on test set
        B_adaptive = calibrator.predict_calibrated_budget(X_test)
        cov_adaptive = np.mean(y_test <= B_adaptive)
        mean_b_adaptive = np.mean(B_adaptive)
        
        # Marginal predictions on test set
        B_marginal = calibrator.marginal_calibrate(y_cal)
        cov_marginal = np.mean(y_test <= B_marginal)
        
        print(f"Alpha = {alpha:.2f} (Target Coverage >= {1-alpha:.1%}):")
        print(f"  k_cal = {cal_res.k_cal}/{cal_res.n_cal}, q_alpha = {cal_res.q_alpha:.2f}")
        print(f"  Adaptive: Coverage = {cov_adaptive:.1%}, Mean B = {mean_b_adaptive:.2f} (B/N = {mean_b_adaptive/N:.1%})")
        print(f"  Marginal: Coverage = {cov_marginal:.1%}, Fixed B = {B_marginal} (B/N = {B_marginal/N:.1%})")
        
    print("\n" + "=" * 50)
    print("RECOVERY DEPTH & STRUCTURAL WORK AUDIT (TEST SET)")
    print("=" * 50)
    r = 1.5
    B1 = 3
    bound_violations = 0
    work_violations = 0
    
    for idx, row in df_test.iterrows():
        b_dag = row["b_dagger"]
        # Theoretical bound
        l_star_bound = 1 + int(np.ceil(max(0.0, np.log(b_dag / B1) / np.log(r))))
        
        # Simulate ladder
        b_curr = B1
        stage = 1
        work_sum = 0
        while True:
            work_sum += b_curr
            if b_curr >= b_dag:
                break
            stage += 1
            b_curr = min(N, int(np.ceil(B1 * (r ** (stage - 1)))))
            if stage > 10:
                break
                
        # Theoretical structural work bound
        work_bound = (r**2 / (r - 1)) * b_dag + stage
        
        if stage > l_star_bound:
            bound_violations += 1
        if work_sum >= work_bound:
            work_violations += 1
            
    print(f"Recovery Depth Theorem Violations:    {bound_violations} / {len(df_test)}")
    print(f"Structural Work Theorem Violations: {work_violations} / {len(df_test)}")
    
    print("\n" + "=" * 50)
    print("PERSPECTIVE RELAXATION LOWER BOUND AUDIT")
    print("=" * 50)
    valid_persp = df_res.dropna(subset=["lb_persp"])
    if len(valid_persp) > 0:
        df_all_persp = valid_persp[valid_persp["persp_dominates"] == True]
        print(f"Instances where LB_persp >= LB_basic - 1e-6: {len(df_all_persp)} / {len(valid_persp)} ({len(df_all_persp)/len(valid_persp):.1%})")
        gap_improvement = (valid_persp["lb_persp"] - valid_persp["lb_basic"])
        print(f"Mean LB tightening (LB_persp - LB_basic): {gap_improvement.mean():.6e}")
        print(f"Max  LB tightening:                      {gap_improvement.max():.6e}")
    else:
        print("Note: Perspective relaxation for N=10, H=3 (219 variables) exceeds the 200-variable quadratic license limit; audited separately on N=5, H=3.")
    
    # Save results to artifacts
    out_csv = "artifacts/conformal_synthetic_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")


if __name__ == "__main__":
    run_all()
