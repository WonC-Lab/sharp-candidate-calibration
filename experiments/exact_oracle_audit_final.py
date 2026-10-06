"""Final integer-CFLP exact-oracle audit matching the manuscript protocol."""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learning_augmented_portfolio.benchmarks.exact_oracle_cflp import (  # noqa: E402
    build_exact_candidate_family,
    generate_integer_cflp_instances,
    solve_exact_integer_cflp,
)
from learning_augmented_portfolio.conformal.conformal_budget import calibrate_marginal_budget  # noqa: E402


OUT = ROOT / "artifacts" / "k20_replication"
INSTANCE_PATH = OUT / "exact_oracle_instance_labels.csv"
RAW_PATH = OUT / "exact_oracle_test_results.csv"
SUMMARY_PATH = OUT / "exact_oracle_summary.csv"
N = 10
M = 15
N_CAL = 20
N_TEST = 20
REPLICATIONS = 30
ALPHAS = (0.20, 0.10, 0.05)
BASE_SEED = 1234
BOOTSTRAP_SEED = 20_261_006
BOOTSTRAP_RESAMPLES = 100_000


def append_row(path: Path, row: Dict[str, Any]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def label_instance(instance: Any) -> Dict[str, Any]:
    family, b_min, ranking = build_exact_candidate_family(instance)
    full = solve_exact_integer_cflp(instance, candidate_set=None, time_limit=5.0)
    b_dagger = None
    candidate_failures: List[int] = []
    if full["exact_certified"]:
        exact_objective = int(full["exact_integer_objective"])
        for B in range(b_min, N + 1):
            restricted = solve_exact_integer_cflp(instance, candidate_set=family[B], time_limit=3.0)
            if not restricted["exact_certified"]:
                candidate_failures.append(B)
                continue
            if int(restricted["exact_integer_objective"]) == exact_objective:
                b_dagger = B
                break
    return {
        "B_min": b_min,
        "B_dagger_exact": b_dagger,
        "full_exact_certified": bool(full["exact_certified"]),
        "full_exact_objective": full.get("exact_integer_objective"),
        "full_upper_bound": full.get("upper_bound"),
        "full_lower_bound": full.get("lower_bound"),
        "full_gap": full.get("gap"),
        "candidate_failures_before_exact": json.dumps(candidate_failures),
        "ranking": json.dumps([int(i) for i in ranking]),
        "label_status": "OK" if full["exact_certified"] and b_dagger is not None else "UNCERTIFIED_RECORDED",
    }


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (INSTANCE_PATH, RAW_PATH, SUMMARY_PATH):
        if path.exists():
            path.unlink()
    rng = np.random.default_rng(BASE_SEED)
    rep_rows: List[Dict[str, Any]] = []
    for replication in range(1, REPLICATIONS + 1):
        replication_seed = int(rng.integers(1, 10_000_000))
        instances = generate_integer_cflp_instances(N_CAL + N_TEST, N=N, M=M, seed=replication_seed)
        labels: List[Dict[str, Any]] = []
        for index, instance in enumerate(instances):
            split = "calibration" if index < N_CAL else "test"
            split_index = index if split == "calibration" else index - N_CAL
            labeled = label_instance(instance)
            row = {
                "replication": replication,
                "replication_seed": replication_seed,
                "split": split,
                "split_index": split_index,
                "instance_id": instance.instance_id,
                "N": N,
                "M": M,
                **labeled,
            }
            append_row(INSTANCE_PATH, row)
            labels.append(row)
        failures = [row for row in labels if row["label_status"] != "OK"]
        if failures:
            raise RuntimeError(
                f"Replication {replication} has {len(failures)} uncertified exact-oracle labels; retained in {INSTANCE_PATH}."
            )
        cal_labels = [int(row["B_dagger_exact"]) for row in labels if row["split"] == "calibration"]
        test_labels = [row for row in labels if row["split"] == "test"]
        for alpha in ALPHAS:
            calibrated = calibrate_marginal_budget(
                cal_labels,
                alpha=alpha,
                N=N,
                B_min_new=0,
                target_type="exact_oracle",
            )
            hits = []
            budgets = []
            for row in test_labels:
                budget = min(N, max(int(row["B_min"]), int(calibrated["B_alpha"])))
                hit = int(row["B_dagger_exact"]) <= budget
                out = {
                    "replication": replication,
                    "replication_seed": replication_seed,
                    "alpha": alpha,
                    "nominal_coverage": 1.0 - alpha,
                    "n_cal": N_CAL,
                    "n_test": N_TEST,
                    "conformal_k": calibrated["k_alpha"],
                    "conformal_quantile": calibrated["quantile"],
                    "test_index": row["split_index"],
                    "instance_id": row["instance_id"],
                    "B_min": row["B_min"],
                    "B_dagger_exact": row["B_dagger_exact"],
                    "B_alpha": budget,
                    "exactness_hit": hit,
                }
                append_row(RAW_PATH, out)
                hits.append(hit)
                budgets.append(budget)
            rep_rows.append({
                "replication": replication,
                "alpha": alpha,
                "coverage": float(np.mean(hits)),
                "mean_B_over_N": float(np.mean(budgets)) / N,
            })

    reps = pd.DataFrame(rep_rows)
    boot_rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries = []
    for alpha in ALPHAS:
        group = reps[np.isclose(reps["alpha"], alpha)]
        coverage = group["coverage"].to_numpy(float)
        indices = boot_rng.integers(0, len(coverage), size=(BOOTSTRAP_RESAMPLES, len(coverage)))
        boot_means = coverage[indices].mean(axis=1)
        summaries.append({
            "N": N,
            "M": M,
            "replications": REPLICATIONS,
            "n_cal": N_CAL,
            "n_test": N_TEST,
            "alpha": alpha,
            "nominal_coverage": 1.0 - alpha,
            "mean_exact_coverage": float(coverage.mean()),
            "coverage_sd": float(coverage.std(ddof=1)),
            "coverage_se": float(coverage.std(ddof=1) / math.sqrt(len(coverage))),
            "bootstrap_mean_ci_low": float(np.percentile(boot_means, 2.5)),
            "bootstrap_mean_ci_high": float(np.percentile(boot_means, 97.5)),
            "replication_q05": float(np.percentile(coverage, 5)),
            "replication_q50": float(np.percentile(coverage, 50)),
            "replication_q95": float(np.percentile(coverage, 95)),
            "mean_candidate_ratio": float(group["mean_B_over_N"].mean()),
            "uncertified_label_count": 0,
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
