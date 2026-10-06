"""
Phase 10 Experiment 10B: Structural & Model-Size Analytical Scaling (Table A).
Evaluates exact mathematical dimensions across the full (N, H) Cartesian grid:
  - N in {10, 25, 50, 100, 250, 500}
  - H in {1, 3, 5}
  - Full vs Physically Sliced Reduced MIQP (rho_B in {0.10, 0.20, 0.30, 0.50})
Verifies:
  - Binary count: n_bin = 2 * N * H (Full) vs 2 * B * H (Reduced)
  - Constraint count: n_constrs = H * (8 * N + 5)
  - Combinatorial assignment space: 2^(2NH) vs 2^(2BH) (combinatorial bound description)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from learning_augmented_portfolio.scalability.model_stats import compute_theoretical_model_stats
from learning_augmented_portfolio.scalability.experiment_grid import (
    CardinalityRegime,
    compute_cardinality,
    compute_candidate_budget,
)


def run_exp10b_structural_scaling() -> None:
    print("=" * 80)
    print("PHASE 10B: STRUCTURAL & MODEL-SIZE SCALING (Table A)")
    print("=" * 80)

    grid_n = [10, 25, 50, 100, 250, 500]
    grid_h = [1, 3, 5]
    ratios_b = [0.10, 0.20, 0.30, 0.50]

    rows = []

    for H in grid_h:
        for N in grid_n:
            K_port = compute_cardinality(N, CardinalityRegime.PROPORTIONAL, proportional_rho_k=0.10)
            full_stats = compute_theoretical_model_stats(N, H)

            # 1. Full Model Row
            rows.append({
                "N": N,
                "H": H,
                "ModelForm": "Full_MIQP",
                "K_port": K_port,
                "Candidate_B": N,
                "CandidateRatio": "100%",
                "NumVars": full_stats.n_vars,
                "ContinuousVars": full_stats.n_continuous,
                "BinaryVars": full_stats.n_binary,
                "Constraints": full_stats.n_constraints,
                "QuadTerms": full_stats.n_quadratic_terms,
                "BinarySearchSpaceBound": f"2^{full_stats.n_binary}",
            })

            # 2. Reduced Model Rows for each candidate ratio
            for rho_b in ratios_b:
                B = compute_candidate_budget(N, rho_b, cardinality_k=K_port, n_held=0)
                red_stats = compute_theoretical_model_stats(N, H, candidate_size=B)
                rows.append({
                    "N": N,
                    "H": H,
                    "ModelForm": f"Reduced_rho{rho_b:.2f}",
                    "K_port": K_port,
                    "Candidate_B": B,
                    "CandidateRatio": f"{rho_b * 100:.0f}%",
                    "NumVars": red_stats.n_vars,
                    "ContinuousVars": red_stats.n_continuous,
                    "BinaryVars": red_stats.n_binary,
                    "Constraints": red_stats.n_constraints,
                    "QuadTerms": red_stats.n_quadratic_terms,
                    "BinarySearchSpaceBound": f"2^{red_stats.n_binary}",
                })

    df = pd.DataFrame(rows)
    out_dir = Path("artifacts/scalability")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "table_a_structural_scaling.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n[1] Generated Table A ({len(df)} configurations) -> {csv_path}")
    print("\nSAMPLE ROWS (N in {50, 100, 500}, H=3):")
    sample_df = df[(df["H"] == 3) & (df["N"].isin([50, 100, 500])) & (df["ModelForm"].isin(["Full_MIQP", "Reduced_rho0.20"]))]
    print(sample_df[["N", "H", "ModelForm", "Candidate_B", "NumVars", "BinaryVars", "Constraints", "QuadTerms"]].to_string(index=False))


if __name__ == "__main__":
    run_exp10b_structural_scaling()
