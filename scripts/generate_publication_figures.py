"""
Generate publication-quality figures for the portfolio MIQP paper:
- Figure 2: Speed-Quality-Certification Frontier
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Set professional publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'lines.linewidth': 1.8,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

out_dir = Path("artifacts/plots")
out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Data Extraction: Controlled Synthetic (H=3, N=10) & Real Market (H=3, N=10)
# -------------------------------------------------------------
# Data from Table 6 / Table 8 / Table 9
data = [
    # Synthetic Multi-Period (H=3, N=10)
    {"Env": "Synthetic (H=3, N=10)", "Method": "Fast MarketOnly", "Speedup": 3.12, "MatchRate": 73.3, "Mode": "Fast"},
    {"Env": "Synthetic (H=3, N=10)", "Method": "ContinuousRelaxation", "Speedup": 1.80, "MatchRate": 86.7, "Mode": "Relaxation-Assisted"},
    {"Env": "Synthetic (H=3, N=10)", "Method": "Certified Recovery", "Speedup": 0.98, "MatchRate": 100.0, "Mode": "Certified Recovery"},
    {"Env": "Synthetic (H=3, N=10)", "Method": "Full MIQP Reference", "Speedup": 1.00, "MatchRate": 100.0, "Mode": "Reference"},

    # Synthetic Multi-Period (H=3, N=8)
    {"Env": "Synthetic (H=3, N=8)", "Method": "Fast MarketOnly", "Speedup": 1.50, "MatchRate": 73.3, "Mode": "Fast"},
    {"Env": "Synthetic (H=3, N=8)", "Method": "ContinuousRelaxation", "Speedup": 0.95, "MatchRate": 86.7, "Mode": "Relaxation-Assisted"},
    {"Env": "Synthetic (H=3, N=8)", "Method": "Certified Recovery", "Speedup": 0.58, "MatchRate": 100.0, "Mode": "Certified Recovery"},

    # Real-Market 2024 Weekly MPC (H=3, N=10)
    {"Env": "Real Market (H=3, N=10)", "Method": "Fast MarketOnly / ML", "Speedup": 1.73, "MatchRate": 70.6, "Mode": "Fast"},
    {"Env": "Real Market (H=3, N=10)", "Method": "ContinuousRelaxation", "Speedup": 1.25, "MatchRate": 74.5, "Mode": "Relaxation-Assisted"},
    {"Env": "Real Market (H=3, N=10)", "Method": "Certified Recovery", "Speedup": 0.59, "MatchRate": 100.0, "Mode": "Certified Recovery"},
    {"Env": "Real Market (H=3, N=10)", "Method": "Full MIQP Reference", "Speedup": 1.00, "MatchRate": 100.0, "Mode": "Reference"},
]

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(8.5, 5.2))

# Color palette and markers for the three operating profiles
palette = {
    "Fast": ("#D95F02", "o"),                 # Bright orange circle
    "Relaxation-Assisted": ("#7570B3", "s"),  # Purple square
    "Certified Recovery": ("#1B9E77", "^"),   # Green triangle
    "Reference": ("#333333", "D"),            # Dark grey diamond
}

markers = {
    "Synthetic (H=3, N=10)": "filled",
    "Real Market (H=3, N=10)": "open",
    "Synthetic (H=3, N=8)": "alpha",
}

# Plot points
plotted_labels = set()
for _, row in df.iterrows():
    mode = row["Mode"]
    env = row["Env"]
    color, base_marker = palette[mode]

    label = mode if mode not in plotted_labels else None
    if label:
        plotted_labels.add(mode)

    if env == "Real Market (H=3, N=10)":
        ax.scatter(row["Speedup"], row["MatchRate"], color=color, marker=base_marker, s=130,
                   edgecolor='black', linewidth=1.5, zorder=5, label=label)
        ax.annotate(f"Real: {row['Method']}", (row["Speedup"], row["MatchRate"]),
                    textcoords="offset points", xytext=(8, -5 if row["Speedup"] < 1.0 else 5),
                    fontsize=8.5, alpha=0.85)
    elif env == "Synthetic (H=3, N=10)":
        ax.scatter(row["Speedup"], row["MatchRate"], color=color, marker=base_marker, s=110,
                   edgecolor='black', linewidth=1.0, alpha=0.9, zorder=4, label=label)
        ax.annotate(f"Synth: {row['Method']}", (row["Speedup"], row["MatchRate"]),
                    textcoords="offset points", xytext=(8, 6 if row["MatchRate"] < 95 else -12),
                    fontsize=8.5, alpha=0.85)
    else:
        ax.scatter(row["Speedup"], row["MatchRate"], color=color, marker=base_marker, s=70,
                   alpha=0.5, edgecolor=color, zorder=3)

# Add reference line at Speedup = 1.0
ax.axvline(1.0, color='gray', linestyle=':', linewidth=1.2, zorder=1, label="Unreduced Baseline (1.0x)")

# Region shading / annotations
ax.axhspan(98, 102, color='#1B9E77', alpha=0.08, label="Certified 100% Quality Region")

# Set limits and labels
ax.set_xlim(0.4, 3.4)
ax.set_ylim(65, 105)
ax.set_xlabel("Mean Solver Speedup Relative to Full MIQP (Higher = Faster)", fontweight='bold')
ax.set_ylabel("Reference-Objective Match Rate (%) (Higher = Better)", fontweight='bold')
ax.set_title("The Speed–Quality–Certification Frontier in Repeated Portfolio MIQPs", pad=12, fontweight='bold')

# Annotations for operating regimes
ax.text(2.6, 68, "Fast Regimes\n(High acceleration,\nmild degradation)", 
        ha='center', va='bottom', fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="#FFF2EB", ec="#D95F02", alpha=0.8))

ax.text(0.65, 93, "Certified Regimes\n(100% match,\nsolver fallback overhead)", 
        ha='center', va='top', fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="#EBF5F0", ec="#1B9E77", alpha=0.8))

ax.grid(True)
ax.legend(loc='lower left', framealpha=0.9, edgecolor='#ccc')

plt.tight_layout()
fig.savefig(out_dir / "figure2_speed_quality_frontier.pdf")
fig.savefig(out_dir / "figure2_speed_quality_frontier.png")
plt.close(fig)

print("Figure 2 successfully generated in artifacts/plots!")
