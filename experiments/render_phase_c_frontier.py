"""Render the locked Phase C three-panel figure from existing audit CSVs only."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


root = Path(__file__).resolve().parents[1]
data = root / "artifacts" / "k20_replication"
cal = pd.read_csv(data / "k20_summary.csv")
b5 = pd.read_csv(data / "b5" / "cell_summary.csv")
out = root / "manuscript" / "plots" / "fig_phase_c_certified_frontier.png"
out.parent.mkdir(parents=True, exist_ok=True)

fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.8), constrained_layout=True)
colors = {8: "#4c78a8", 20: "#f58518", 30: "#54a24b", 50: "#b279a2"}
for N in (8, 20, 30, 50):
    part = cal[cal.N == N].sort_values("alpha", ascending=False)
    axs[0].plot(1 - part.alpha, part.mean_initial_B_over_N, "o-", label=f"N={N}", color=colors[N])
axs[0].set(xlabel="Nominal confidence $1-\\alpha$", ylabel="Initial candidate ratio $B_1/N$", ylim=(0.65, 1.05), title="A  Calibrated compression")
axs[0].legend(frameon=False, fontsize=8)

for alpha, color in ((0.20, "#4c78a8"), (0.10, "#f58518"), (0.05, "#54a24b")):
    part = b5[(b5.pipeline == "D_CAND_WARM") & np.isclose(b5.alpha, alpha)].sort_values("N")
    y = part.ratio_of_means_speedup.to_numpy()
    error = np.vstack((y - part.speedup_ci_low.to_numpy(), part.speedup_ci_high.to_numpy() - y))
    axs[1].errorbar(part.N, y, yerr=error, marker="o", capsize=3, label=f"$\\alpha={alpha:.2f}$", color=color)
axs[1].axhline(1, color="black", linestyle="--", linewidth=1)
axs[1].set(xlabel="Facilities $N$", ylabel="Cold/direct-warm time ratio", ylim=(0.3, 1.12), title="B  Certified wall-clock ratio")
axs[1].legend(frameon=False, fontsize=8)

part = b5[np.isclose(b5.alpha, 0.20)].pivot(index="N", columns="pipeline", values="mean_full_nodes").loc[[8, 20, 30, 50]]
xs = np.arange(4)
axs[2].bar(xs - 0.17, part["A_COLD"], width=0.34, label="Cold full", color="#4c78a8")
axs[2].bar(xs + 0.17, part["D_CAND_WARM"], width=0.34, label="Warm full", color="#f58518")
axs[2].set(xticks=xs, xticklabels=[8, 20, 30, 50], xlabel="Facilities $N$", ylabel="Mean full-MIP nodes", title="C  Search effort ($\\alpha=.20$)")
axs[2].legend(frameon=False, fontsize=8)
fig.savefig(out, dpi=220)
