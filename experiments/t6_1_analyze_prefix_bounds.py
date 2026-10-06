"""Pre-specified T6.1 diagnostic of the 80 locked-instance prefix-bound records."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "t6_1_full_prefix_reanalysis"
RAW = OUT / "prefix_raw.csv"
MANIFEST = OUT / "manifest.json"
TOLERANCES = (0.0, 1e-4, 1e-3)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def envelopes(lower: list[float], upper: list[float]) -> tuple[list[float], list[float]]:
    a = list(lower)
    c = list(upper)
    for i in range(len(a) - 2, -1, -1):
        a[i] = max(a[i], a[i + 1])
    for i in range(1, len(c)):
        c[i] = min(c[i], c[i - 1])
    return a, c


def identified_set(budgets: list[int], lower: list[float], upper: list[float], tau: float):
    a, c = envelopes(lower, upper)
    if any(x > y for x, y in zip(a, c)):
        return None, a, c
    s_lo, s_hi = a[-1], c[-1]
    result = []
    if max(s_lo, a[0] - tau) <= s_hi:
        result.append(budgets[0])
    for i in range(1, len(budgets)):
        x = max(s_lo, a[i] - tau)
        if x <= s_hi and x < c[i - 1] - tau:
            result.append(budgets[i])
    return result, a, c


def is_noncontiguous(values: list[int]) -> bool:
    return len(values) > 1 and any(y != x + 1 for x, y in zip(values, values[1:]))


def analyze_group(frame: pd.DataFrame):
    frame = frame.sort_values("B").reset_index(drop=True)
    n = int(frame.N.iloc[0])
    b0 = int(frame.b0.iloc[0])
    budgets = list(range(b0, n + 1))
    if frame.B.astype(int).tolist() != budgets:
        raise RuntimeError(f"Missing or repeated prefix at seed {frame.generation_seed.iloc[0]}")
    if frame.instance_sha256.nunique() != 1 or frame.full_new_ub.nunique() != 1:
        raise RuntimeError("Inconsistent per-instance fingerprint or full record")
    lower = frame.candidate_lb.astype(float).tolist()
    upper = frame.candidate_ub.astype(float).tolist()
    full_lb = lower[-1]
    full_ub = upper[-1]
    all_a, all_c = envelopes(lower, upper)
    max_envelope_conflict = max(0.0, *(x-y for x, y in zip(all_a, all_c)))
    # Full information is available at stage zero; earlier prefix records are
    # revealed in ascending B order. This is prefix-bound, not early-time, ID.
    output = []
    for tau in TOLERANCES:
        final_set, final_a, final_c = identified_set(budgets, lower, upper, tau)
        first_count = None
        first_budget = None
        for count in range(len(budgets)):
            partial_l = [lower[i] if i < count or i == len(budgets)-1 else -math.inf
                         for i in range(len(budgets))]
            partial_u = [upper[i] if i < count or i == len(budgets)-1 else math.inf
                         for i in range(len(budgets))]
            partial_set, _, _ = identified_set(budgets, partial_l, partial_u, tau)
            if partial_set is not None and len(partial_set) == 1:
                first_count = count
                first_budget = budgets[count-1] if count else None
                break
        if final_set is None or len(final_set) != 1:
            first_count = None
            first_budget = None
        row = {
            "N": n, "M": int(frame.M.iloc[0]), "test_index": int(frame.test_index.iloc[0]),
            "generation_seed": int(frame.generation_seed.iloc[0]), "b0": b0,
            "num_feasible_prefixes": len(budgets), "tau": tau,
            "identified_set": json.dumps(final_set) if final_set is not None else "INCOMPATIBLE_NUMERIC_INTERVALS",
            "set_size": len(final_set) if final_set is not None else -1,
            "singleton_budget": final_set[0] if final_set is not None and len(final_set) == 1 else None,
            "noncontiguous": is_noncontiguous(final_set) if final_set is not None else False,
            "first_singleton_candidate_count": first_count,
            "first_singleton_candidate_budget": first_budget,
            "first_singleton_before_all_prefixes": first_count is not None and first_count < len(budgets)-1,
            "full_lb": full_lb, "full_ub": full_ub,
            "full_absolute_gap": max(0.0, full_ub-full_lb),
            "max_envelope_conflict": max_envelope_conflict,
            "max_prefix_absolute_gap": float(frame.absolute_gap.max()),
            "nonoptimal_status_count": int((frame.status != "OPTIMAL").sum()),
        }
        output.append(row)
    return frame, output


def make_figure(frame: pd.DataFrame, diag: pd.DataFrame, selection_note: str) -> None:
    frame = frame.sort_values("B")
    budgets = frame.B.astype(int).to_numpy()
    n = int(frame.N.iloc[0])
    seed = int(frame.generation_seed.iloc[0])
    lb = frame.candidate_lb.astype(float).to_numpy()
    ub = frame.candidate_ub.astype(float).to_numpy()
    full_lb, full_ub = float(lb[-1]), float(ub[-1])
    gap_lo = np.maximum(0.0, lb-full_ub)
    gap_hi = np.maximum(0.0, ub-full_lb)
    gap_lo[-1] = gap_hi[-1] = 0.0
    a, c = envelopes(lb.tolist(), ub.tolist())
    fig, panels = plt.subplot_mosaic([["A", "A"], ["B", "C"]],
                                     figsize=(7.0, 5.5), constrained_layout=True)
    ax = [panels["A"], panels["B"], panels["C"]]
    ax[0].fill_between(budgets, gap_lo, gap_hi, color="#98b7cf", alpha=0.7,
                       label="bound-justified gap interval")
    ax[0].plot(budgets, ub-full_ub, color="#174a6e", lw=1.4, label="incumbent difference")
    ax[0].set(xlabel="Candidate budget B", ylabel="Gap to full optimum", title="A  Candidate gap profile")
    ax[0].legend(fontsize=7, loc="upper right")
    sub = diag[diag.generation_seed == seed].sort_values("tau")
    x = np.arange(len(sub))
    for j, (_, row) in enumerate(sub.iterrows()):
        qset = json.loads(row.identified_set) if row.set_size >= 0 else []
        if qset:
            ax[1].plot([j, j], [min(qset), max(qset)], color="#b57219", lw=3)
            ax[1].scatter([j] * len(qset), qset, color="#a83f32", s=20, zorder=3)
    ax[1].set_xticks(x, ["0", "1e-4", "1e-3"])
    ax[1].set(xlabel="Prespecified tolerance tau", ylabel="Identified Q budgets",
              title="B  Quality-budget identification", ylim=(int(budgets[0])-1, n+1))
    center = full_ub
    ax[2].plot(budgets, np.asarray(a)-center, color="#2c8b6a", label="lower envelope a_B")
    ax[2].plot(budgets, np.asarray(c)-center, color="#9c4d71", label="upper envelope c_B")
    ax[2].axhline(1e-4, color="#555", ls="--", lw=0.8, label="tau=1e-4")
    selected = sub[np.isclose(sub.tau, 1e-4)].iloc[0]
    qset = json.loads(selected.identified_set) if selected.set_size >= 0 else []
    for budget in qset:
        ax[2].axvline(budget, color="#e8b44b", alpha=0.35, lw=4)
    ax[2].set(xlabel="Candidate budget B", ylabel="Envelope minus full incumbent",
              title="C  Solver-bound envelopes")
    ax[2].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Locked CFLP instance N={n}, seed={seed}; {selection_note}", fontsize=9)
    fig.savefig(OUT / "theory_evidence.png", dpi=220)
    plt.close(fig)


def main() -> None:
    if not RAW.exists() or not MANIFEST.exists():
        raise RuntimeError("Full-prefix raw data and manifest must be complete")
    source_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sha(RAW) != source_manifest["raw_sha256"]:
        raise RuntimeError("Raw output hash mismatch")
    raw = pd.read_csv(RAW)
    grouped = list(raw.groupby(["N", "generation_seed"], sort=True))
    if len(grouped) != 80:
        raise RuntimeError(f"Expected 80 distinct instances, got {len(grouped)}")
    diagnostics = []
    frame_by_seed = {}
    for (_, seed), group in grouped:
        frame, out = analyze_group(group)
        frame_by_seed[int(seed)] = frame
        diagnostics.extend(out)
    diag = pd.DataFrame(diagnostics)
    diag.to_csv(OUT / "identification_by_instance.csv", index=False)
    summary = []
    for tau, group in diag.groupby("tau", sort=True):
        summary.append({
            "tau": float(tau), "unique_instances": len(group),
            "singleton": int((group.set_size == 1).sum()),
            "multiple": int((group.set_size > 1).sum()),
            "incompatible_numeric": int((group.set_size < 0).sum()),
            "noncontiguous": int(group.noncontiguous.sum()),
            "first_singleton_before_all": int(group.first_singleton_before_all_prefixes.sum()),
            "no_singleton_even_after_all": int((group.set_size != 1).sum()),
            "nonoptimal_prefix_rows": int(group.nonoptimal_status_count.sum()),
            "numeric_conflict_above_1e_7": int((group.max_envelope_conflict > 1e-7).sum()),
        })
    pd.DataFrame(summary).to_csv(OUT / "identification_summary.csv", index=False)
    selection = diag[np.isclose(diag.tau, 1e-4) & (diag.set_size == 1)].copy()
    if len(selection):
        selection["ratio"] = selection.singleton_budget / selection.N
        selection = selection.sort_values(["ratio", "generation_seed"])
        chosen = selection.iloc[(len(selection)-1)//2]
        selection_note = "lower median reported-bound singleton Q(1e-4)/N"
    else:
        estimates = []
        for seed, frame in frame_by_seed.items():
            b = frame.B.astype(int).to_numpy()
            u = frame.candidate_ub.astype(float).to_numpy()
            q = int(b[np.flatnonzero(u-u[-1] <= 1e-4)[0]])
            estimates.append((q/int(frame.N.iloc[0]), seed))
        estimates.sort()
        _, seed = estimates[(len(estimates)-1)//2]
        chosen = {"generation_seed": seed}
        selection_note = "lower median incumbent-based Q proxy; not certified"
    chosen_seed = int(chosen["generation_seed"])
    make_figure(frame_by_seed[chosen_seed], diag, selection_note)
    audit = {
        "analysis_plan_sha256": source_manifest["analysis_plan_sha256"],
        "raw_sha256": sha(RAW),
        "analysis_code_sha256": sha(Path(__file__)),
        "diagnostic_sha256": sha(OUT / "identification_by_instance.csv"),
        "summary_sha256": sha(OUT / "identification_summary.csv"),
        "figure_sha256": sha(OUT / "theory_evidence.png"),
        "figure_selection_seed": chosen_seed,
        "figure_selection_rule": selection_note,
        "figure_selection_pool_size": len(selection),
        "diagnostic_label": "prefix-bound identification, not early-time solver identification",
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"FIGURE seed={chosen_seed} rule={selection_note}")


if __name__ == "__main__":
    main()
