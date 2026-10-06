"""
Audit perspective relaxation lower-bound dominance on N=5, H=3 instances.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from scratch.run_conformal_validation import load_npz_instance
from learning_augmented_portfolio.optimization.relaxation import (
    MultiPeriodRelaxedQPOptimizer,
    MultiPeriodPerspectiveRelaxedOptimizer,
)
from learning_augmented_portfolio.optimization.variables import SolverConfig


def audit_perspective_n5():
    manifest_path = "artifacts/instances/manifest.csv"
    df = pd.read_csv(manifest_path)
    solver_cfg = SolverConfig(output_flag=False, time_limit=15.0)
    
    records = []
    print(f"Auditing perspective relaxation on {len(df)} N=5 instances...")
    
    for idx, row in df.iterrows():
        npz_rel = os.path.join("artifacts/instances", row["filename"])
        if not os.path.exists(npz_rel):
            continue
        state, params, obj_ref = load_npz_instance(npz_rel)
        
        basic_opt = MultiPeriodRelaxedQPOptimizer(params, solver_cfg)
        b_res = basic_opt.solve(state)
        
        persp_opt = MultiPeriodPerspectiveRelaxedOptimizer(params, solver_cfg, rho_persp=0.95)
        p_res = persp_opt.solve(state)
        
        lb_b = b_res.objective_value
        lb_p = p_res.objective_value
        
        records.append({
            "instance_id": row["instance_id"],
            "split": row["split"],
            "obj_ref": obj_ref,
            "lb_basic": lb_b,
            "lb_persp": lb_p,
            "tightening": lb_p - lb_b,
            "dominates": (lb_p >= lb_b - 1e-6),
            "cert_basic": (obj_ref - lb_b <= 1e-4) if obj_ref is not None else False,
            "cert_persp": (obj_ref - lb_p <= 1e-4) if obj_ref is not None else False,
        })
        
    df_p = pd.DataFrame(records)
    print(f"\nTotal N=5 instances processed: {len(df_p)}")
    print(f"Dominance Rate (LB_persp >= LB_basic): {df_p['dominates'].mean():.1%}")
    print(f"Mean LB Tightening:                    {df_p['tightening'].mean():.6e}")
    print(f"Max  LB Tightening:                    {df_p['tightening'].max():.6e}")
    print(f"Basic Relaxation Certificate Rate:     {df_p['cert_basic'].mean():.1%}")
    print(f"Persp Relaxation Certificate Rate:     {df_p['cert_persp'].mean():.1%}")
    
    out_csv = "artifacts/perspective_audit_n5.csv"
    df_p.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")


if __name__ == "__main__":
    audit_perspective_n5()
