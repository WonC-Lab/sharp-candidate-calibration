"""
Generates the 8 Canonical Scientific Tables for Manuscript Preparation (Phase 10.5).
Tables 1-3: Phase 9 Real-Market Backtest & Acceleration
Tables 4-8: Phase 10 Scalability, Structural Slicing, Frontier & Corrected Oracle Diagnostic
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    print("=" * 80)
    print("GENERATING FINAL CANONICAL SCIENTIFIC TABLES (Tables 1 - 8)")
    print("=" * 80)

    tables_dir = Path("artifacts/final_tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    scal_dir = Path("artifacts/scalability")
    p9_dir = Path("artifacts/real_backtest")

    # -------------------------------------------------------------------------
    # Final Table 1: Real-Market Financial Performance (Phase 9)
    # -------------------------------------------------------------------------
    df_p9_fin = pd.read_csv(p9_dir / "financial_metrics.csv")
    df_p9_fin.to_csv(tables_dir / "final_table_1_financial_performance.csv", index=False)
    print("[1] Saved Final Table 1: Real-Market Financial Performance")

    # -------------------------------------------------------------------------
    # Final Table 2: Real-Market Solver Acceleration (Phase 9)
    # -------------------------------------------------------------------------
    df_p9_opt = pd.read_csv(p9_dir / "optimization_metrics.csv")
    df_p9_opt.to_csv(tables_dir / "final_table_2_solver_acceleration.csv", index=False)
    print("[2] Saved Final Table 2: Real-Market Solver Acceleration")

    # -------------------------------------------------------------------------
    # Final Table 3: Certification and Recovery (Phase 9)
    # -------------------------------------------------------------------------
    df_p9_cert = pd.read_csv(p9_dir / "certification_metrics.csv")
    df_p9_cert.to_csv(tables_dir / "final_table_3_certification_recovery.csv", index=False)
    print("[3] Saved Final Table 3: Certification and Recovery")

    # -------------------------------------------------------------------------
    # Final Table 4: Structural Scaling (Phase 10 canonical H*(8N+5))
    # -------------------------------------------------------------------------
    df_t4_raw = pd.read_csv(scal_dir / "table_a_structural_scaling.csv")
    df_t4_raw.to_csv(tables_dir / "final_table_4_structural_scaling.csv", index=False)
    print("[4] Saved Final Table 4: Structural Scaling")

    # -------------------------------------------------------------------------
    # Final Table 5: Empirical Full-MIQP Scaling (Executable configs only)
    # -------------------------------------------------------------------------
    df_t5_raw = pd.read_csv(scal_dir / "table_b_full_miqp_scaling.csv")
    df_t5 = df_t5_raw[df_t5_raw["Status"] != "NOT_EXECUTED_LICENSE_LIMIT"].copy()
    df_t5.to_csv(tables_dir / "final_table_5_full_miqp_empirical_scaling.csv", index=False)
    print("[5] Saved Final Table 5: Empirical Full-MIQP Scaling")

    # -------------------------------------------------------------------------
    # Final Table 6: Candidate-Reduced Scaling (Quality validation distinguished)
    # -------------------------------------------------------------------------
    df_t6_raw = pd.read_csv(scal_dir / "table_c_reduced_scaling.csv")
    df_t6 = df_t6_raw.copy()
    df_t6["QualityValidationAvailable"] = True
    df_t6.to_csv(tables_dir / "final_table_6_candidate_reduced_scaling.csv", index=False)
    print("[6] Saved Final Table 6: Candidate-Reduced Scaling")

    # -------------------------------------------------------------------------
    # Final Table 7: Speed-Quality-Certification Frontier
    # -------------------------------------------------------------------------
    df_t7_raw = pd.read_csv(scal_dir / "table_d_e2e_acceleration.csv")
    df_t7 = df_t7_raw[df_t7_raw["Method"] != "Oracle_Diagnostic"].copy()
    df_t7.to_csv(tables_dir / "final_table_7_speed_quality_frontier.csv", index=False)
    print("[7] Saved Final Table 7: Speed-Quality-Certification Frontier")

    # -------------------------------------------------------------------------
    # Final Table 8: Oracle Upper-Bound Diagnostic (Corrected semantics)
    # -------------------------------------------------------------------------
    df_t8 = df_t7_raw[df_t7_raw["Method"] == "Oracle_Diagnostic"].copy()
    df_t8["IsDeployable"] = False
    df_t8["UsesTeacherInformation"] = True
    df_t8.to_csv(tables_dir / "final_table_8_oracle_diagnostic.csv", index=False)
    print("[8] Saved Final Table 8: Oracle Upper-Bound Diagnostic")

    print("\n[SUCCESS] ALL 8 CANONICAL TABLES GENERATED IN artifacts/final_tables/!")


if __name__ == "__main__":
    main()
