"""
Phase 10G: Scalability Summary & Primary Publication Plots.
Loads Tables A-F and generates 8 publication-grade scalability plots:
  - Plot 1: N vs Median Full MIQP Runtime (separate H curves)
  - Plot 2: N vs E2E Speedup for each candidate ratio
  - Plot 3: Candidate Reduction vs Objective Quality
  - Plot 4: Anytime incumbent/bound curves
  - Plot 5: N vs Time to First Feasible
  - Plot 6: N vs Gurobi Work
  - Plot 7: N vs Certificate Success Rate
  - Plot 8: (N, H) Heatmap of median speedup & solver feasibility
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
    })


def main():
    print("=" * 80)
    print("PHASE 10G: SCALABILITY SUMMARY & PLOT GENERATION")
    print("=" * 80)

    setup_plot_style()
    artifacts_dir = Path("artifacts/scalability")
    plots_dir = Path("artifacts/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df_table_a = pd.read_csv(artifacts_dir / "table_a_structural_scaling.csv")
    df_table_b = pd.read_csv(artifacts_dir / "table_b_full_miqp_scaling.csv")
    df_table_c = pd.read_csv(artifacts_dir / "table_c_reduced_scaling.csv")
    df_table_d = pd.read_csv(artifacts_dir / "table_d_e2e_acceleration.csv")
    df_table_e = pd.read_csv(artifacts_dir / "table_e_anytime_quality.csv")
    df_table_f = pd.read_csv(artifacts_dir / "table_f_certification_scaling.csv")

    with open(artifacts_dir / "anytime_trajectories.json", "r") as f:
        anytime_trajs = json.load(f)

    # Clean CandidateRatio in Table C
    df_table_c["CandidateRatio_num"] = (
        df_table_c["CandidateRatio"].astype(str).str.rstrip("%").astype(float) / 100.0
    )

    # -------------------------------------------------------------------------
    # Plot 1: N vs Median Full MIQP Runtime (separate H curves)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    h_values = sorted(df_table_b["H"].unique())
    colors = {1: "#1f77b4", 3: "#ff7f0e", 5: "#2ca02c"}
    markers = {1: "o", 3: "s", 5: "^"}

    for h in h_values:
        sub = df_table_b[(df_table_b["H"] == h) & (df_table_b["Status"] != "NOT_EXECUTED_LICENSE_LIMIT")].sort_values("N")
        if not sub.empty:
            plt.plot(
                sub["N"],
                sub["MedianTime_ms"] / 1000.0,
                marker=markers.get(h, "o"),
                color=colors.get(h, "blue"),
                label=f"Horizon H={h} (Executed)",
            )
        # Indicate license limit threshold
        limit_sub = df_table_b[(df_table_b["H"] == h) & (df_table_b["Status"] == "NOT_EXECUTED_LICENSE_LIMIT")].sort_values("N")
        if not limit_sub.empty:
            max_exec_n = sub["N"].max() if not sub.empty else 0
            first_limit_n = limit_sub["N"].min()
            plt.axvline(
                x=first_limit_n,
                color=colors.get(h, "blue"),
                linestyle=":",
                alpha=0.6,
                label=f"H={h} License Cap (N>{max_exec_n})",
            )

    plt.xlabel("Universe Size N")
    plt.ylabel("Median Solver Runtime (seconds)")
    plt.title("Plot 1: Full MIQP Runtime vs Problem Scale (N, H)")
    plt.yscale("log")
    plt.legend(frameon=True)
    plt.tight_layout()
    p1_path = plots_dir / "plot1_full_miqp_runtime_vs_N.png"
    plt.savefig(p1_path, dpi=300)
    plt.close()
    print(f"[Plot 1] Saved to {p1_path}")

    # -------------------------------------------------------------------------
    # Plot 2: N vs E2E Speedup for each candidate ratio
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    h_focus = 1
    sub_c = df_table_c[(df_table_c["H"] == h_focus) & (df_table_c["Method"] == "Fast_MarketOnly_MIQP")].sort_values("N")
    
    # Merge with Table B to compute exact speedup
    merged_c = sub_c.merge(df_table_b[df_table_b["H"] == h_focus][["N", "MedianTime_ms"]], on="N")
    merged_c["Speedup"] = merged_c["MedianTime_ms"] / merged_c["Median_E2E_ms"]

    ratios = sorted([float(x) for x in merged_c["CandidateRatio_num"].unique()])
    palette = [plt.cm.viridis(val) for val in np.linspace(0.1, 0.9, len(ratios))]

    for idx, r in enumerate(ratios):
        r_sub = merged_c[np.isclose(merged_c["CandidateRatio_num"], r)].sort_values("N")
        if not r_sub.empty:
            plt.plot(
                r_sub["N"],
                r_sub["Speedup"],
                marker="o",
                color=palette[idx],
                label=f"Ratio rho_B = {float(r):.2f}",
            )

    plt.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Parity (Speedup=1.0x)")
    plt.xlabel("Universe Size N (H=1)")
    plt.ylabel("End-to-End Speedup (T_full / T_method)")
    plt.title("Plot 2: End-to-End Speedup vs Universe Size N")
    plt.legend(frameon=True)
    plt.tight_layout()
    p2_path = plots_dir / "plot2_e2e_speedup_vs_N.png"
    plt.savefig(p2_path, dpi=300)
    plt.close()
    print(f"[Plot 2] Saved to {p2_path}")

    # -------------------------------------------------------------------------
    # Plot 3: Candidate Reduction vs Objective Quality
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    methods_plot3 = ["Fast_MarketOnly_MIQP", "ContinuousRelaxation_MIQP", "Certified_Recovery"]
    markers_p3 = {"Fast_MarketOnly_MIQP": "o", "ContinuousRelaxation_MIQP": "s", "Certified_Recovery": "D"}
    colors_p3 = {"Fast_MarketOnly_MIQP": "#1f77b4", "ContinuousRelaxation_MIQP": "#ff7f0e", "Certified_Recovery": "#2ca02c"}

    for m in methods_plot3:
        m_sub = df_table_c[(df_table_c["Method"] == m) & (df_table_c["H"] == 3) & (df_table_c["N"] == 10)].sort_values("CandidateRatio_num")
        if not m_sub.empty:
            red_pct = (1.0 - m_sub["CandidateRatio_num"]) * 100.0
            plt.plot(
                red_pct,
                m_sub["Mean_ObjDegradation"],
                marker=markers_p3.get(m, "o"),
                color=colors_p3.get(m, "black"),
                label=m,
            )

    plt.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Candidate Dimension Reduction (%)")
    plt.ylabel("Objective Degradation (Delta J = J_method - J_full)")
    plt.title("Plot 3: Solution Quality vs Dimension Reduction (H=3, N=10)")
    plt.legend(frameon=True)
    plt.tight_layout()
    p3_path = plots_dir / "plot3_candidate_reduction_vs_quality.png"
    plt.savefig(p3_path, dpi=300)
    plt.close()
    print(f"[Plot 3] Saved to {p3_path}")

    # -------------------------------------------------------------------------
    # Plot 4: Anytime incumbent/bound curves
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    key_prefix = "H3_N10_"
    avail_keys = [k for k in anytime_trajs.keys() if k.startswith(key_prefix)]

    colors_p4 = {"full": "#1f77b4", "market_only": "#2ca02c", "relaxation": "#ff7f0e"}
    styles_p4 = {"full": "-", "market_only": "--", "relaxation": "-."}

    for k in avail_keys:
        m_tag = k.replace(key_prefix, "")
        traj = anytime_trajs[k]
        if traj:
            df_traj = pd.DataFrame(traj)
            valid_ub = df_traj[df_traj["ub"].notnull()]
            if not valid_ub.empty:
                plt.step(
                    valid_ub["t"],
                    valid_ub["ub"],
                    where="post",
                    color=colors_p4.get(m_tag, "black"),
                    linestyle=styles_p4.get(m_tag, "-"),
                    label=f"{m_tag.capitalize()} Incumbent UB(t)",
                )

    plt.xlabel("Elapsed Wall-Clock Time (seconds)")
    plt.ylabel("Incumbent Objective Value")
    plt.title("Plot 4: Anytime Incumbent Trajectory UB(t) (H=3, N=10)")
    plt.legend(frameon=True)
    plt.tight_layout()
    p4_path = plots_dir / "plot4_anytime_incumbent_curves.png"
    plt.savefig(p4_path, dpi=300)
    plt.close()
    print(f"[Plot 4] Saved to {p4_path}")

    # -------------------------------------------------------------------------
    # Plot 5: N vs Time to First Feasible
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    methods_p5 = ["full", "market_only", "relaxation"]
    bar_data = []
    labels_p5 = []
    
    sub_e_1s = df_table_e[df_table_e["TimeBudget_sec"] == 1.0]
    for (h, n), grp in sub_e_1s.groupby(["H", "N"]):
        for m in methods_p5:
            row = grp[grp["Method"] == m]
            if not row.empty:
                val = row["Median_TimeToFirstFeasible"].values[0]
                # Default to median e2e time if callback first-feasible is sub-millisecond
                if np.isnan(val) or val <= 0:
                    val = row["Mean_E2E_Time"].values[0]
                bar_data.append(val * 1000.0)  # in ms
                labels_p5.append(f"H{h}N{n}\n{m}")

    plt.bar(range(len(bar_data)), bar_data, color="#3470a3", alpha=0.85)
    plt.xticks(range(len(labels_p5)), labels_p5, rotation=35, ha="right", fontsize=9)
    plt.ylabel("Time to First Feasible Solution (ms)")
    plt.title("Plot 5: Time to First Feasible Incumbent across Methods")
    plt.tight_layout()
    p5_path = plots_dir / "plot5_time_to_first_feasible.png"
    plt.savefig(p5_path, dpi=300)
    plt.close()
    print(f"[Plot 5] Saved to {p5_path}")

    # -------------------------------------------------------------------------
    # Plot 6: N vs Gurobi Work
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    for h in [1, 3]:
        sub_b_h = df_table_b[(df_table_b["H"] == h) & (df_table_b["Status"] != "NOT_EXECUTED_LICENSE_LIMIT")].sort_values("N")
        if not sub_b_h.empty:
            plt.plot(
                sub_b_h["N"],
                sub_b_h["MedianWork"],
                marker="o",
                label=f"Full MIQP (H={h})",
            )
        sub_c_h = df_table_c[(df_table_c["H"] == h) & (df_table_c["Method"] == "Fast_MarketOnly_MIQP")].sort_values("N")
        if not sub_c_h.empty:
            # Pick smallest candidate ratio
            min_r = sub_c_h["CandidateRatio_num"].min()
            sub_c_min = sub_c_h[sub_c_h["CandidateRatio_num"] == min_r]
            plt.plot(
                sub_c_min["N"],
                sub_c_min["Median_Work"],
                marker="s",
                linestyle="--",
                label=f"Reduced rho={min_r:.2f} (H={h})",
            )

    plt.xlabel("Universe Size N")
    plt.ylabel("Gurobi Work (Normalized Solver Effort)")
    plt.title("Plot 6: Solver Work Scaling (Full vs Reduced)")
    plt.legend(frameon=True)
    plt.tight_layout()
    p6_path = plots_dir / "plot6_gurobi_work_scaling.png"
    plt.savefig(p6_path, dpi=300)
    plt.close()
    print(f"[Plot 6] Saved to {p6_path}")

    # -------------------------------------------------------------------------
    # Plot 7: N vs Certificate Success Rate
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    for h in sorted(df_table_f["H"].unique()):
        sub_f_h = df_table_f[df_table_f["H"] == h].sort_values("N")
        plt.plot(
            sub_f_h["N"],
            sub_f_h["RelaxationCertifiedRate"] * 100.0,
            marker="o",
            label=f"H={h}",
        )

    plt.xlabel("Universe Size N")
    plt.ylabel("Relaxation Certificate Success Rate (%)")
    plt.title("Plot 7: Continuous Relaxation Certification Success Rate")
    plt.ylim(-5, 105)
    plt.legend(frameon=True)
    plt.tight_layout()
    p7_path = plots_dir / "plot7_certificate_success_rate.png"
    plt.savefig(p7_path, dpi=300)
    plt.close()
    print(f"[Plot 7] Saved to {p7_path}")

    # -------------------------------------------------------------------------
    # Plot 8: (N, H) Heatmap of median speedup / solver feasibility
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    pivot_speedup = df_table_d[df_table_d["Method"] == "Fast_MarketOnly_MIQP"].pivot(
        index="H", columns="N", values="Median_Speedup"
    )
    
    im = plt.imshow(pivot_speedup.values, cmap="YlGnBu", aspect="auto")
    plt.colorbar(im, label="Median End-to-End Speedup (x)")
    
    # Tick labels
    plt.xticks(range(len(pivot_speedup.columns)), pivot_speedup.columns)
    plt.yticks(range(len(pivot_speedup.index)), pivot_speedup.index)
    
    # Annotate numbers
    for i in range(len(pivot_speedup.index)):
        for j in range(len(pivot_speedup.columns)):
            val = pivot_speedup.values[i, j]
            if not np.isnan(val):
                text_col = "white" if val > 2.0 else "black"
                plt.text(j, i, f"{val:.2f}x", ha="center", va="center", color=text_col, fontweight="bold")

    plt.title("Plot 8: Speedup Heatmap across Problem Scale (N, H)")
    plt.xlabel("Universe Size N")
    plt.ylabel("Horizon H")
    plt.tight_layout()
    p8_path = plots_dir / "plot8_speedup_heatmap.png"
    plt.savefig(p8_path, dpi=300)
    plt.close()
    print(f"[Plot 8] Saved to {p8_path}")

    print("\n[SUCCESS] All 8 publication plots generated successfully in artifacts/plots/!")


if __name__ == "__main__":
    main()
