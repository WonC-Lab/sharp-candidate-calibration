"""T6.1: expose every feasible prefix on the 80 locked Phase-B test instances.

This script never overwrites Phase B/B.5 outputs or a completed T6.1 result.
The analysis plan is T6_1_ANALYSIS_PLAN.md, written before inspecting output.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import final_empirical_strengthening_k20 as phase_b  # noqa: E402
from learning_augmented_portfolio.benchmarks.repeated_cflp import (  # noqa: E402
    build_cflp_candidate_family,
)

OUT = ROOT / "artifacts" / "t6_1_full_prefix_reanalysis"
RAW = OUT / "prefix_raw.csv"
PART = OUT / "prefix_raw.csv.part"
MANIFEST = OUT / "manifest.json"
PHASE_B_RAW = ROOT / "artifacts" / "k20_replication" / "k20_raw_results.csv"
EXPECTED_PHASE_B_SHA = "7e0f87beb202c6d1335124951df760c9e5a930a4ffd0d3b05a2fcd98a9a85263"

FIELDS = [
    "N", "M", "test_index", "generation_seed", "instance_id", "instance_sha256",
    "B", "b0", "ranked_active", "excluded_fixed_count", "pre_presolve_num_vars_code",
    "pre_presolve_num_constrs_code", "status", "status_code", "sol_count",
    "candidate_ub", "candidate_lb", "absolute_gap", "solver_runtime_seconds",
    "wall_seconds", "mip_gap", "node_count", "full_original_ub",
    "full_original_lb", "full_original_status", "full_original_lp_lb",
    "full_new_ub", "full_new_lb", "full_new_status",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instance_hash(instance) -> str:
    digest = hashlib.sha256()
    for arr in (instance.fixed_costs, instance.capacities, instance.demands, instance.transport_costs):
        values = np.asarray(arr, dtype="<f8", order="C")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def locked_instances() -> list[tuple[int, int, int, int, pd.Series]]:
    if sha(PHASE_B_RAW) != EXPECTED_PHASE_B_SHA:
        raise RuntimeError("Phase B raw hash changed; refusing re-analysis")
    old = pd.read_csv(PHASE_B_RAW)
    if len(old) != 240:
        raise RuntimeError(f"Expected 240 alpha rows, got {len(old)}")
    result = []
    for (n, idx, seed), group in old.groupby(["N", "test_index", "generation_seed"], sort=True):
        if len(group) != 3 or set(np.round(group.alpha.to_numpy(float), 2)) != {0.05, 0.10, 0.20}:
            raise RuntimeError(f"Incorrect alpha reuse: {(n, idx, seed)}")
        for field in ("M", "instance_id", "B_min", "total_capacity", "total_demand", "ranking",
                      "reference_solver_bound", "reference_solver_incumbent", "lp_valid_bound"):
            if group[field].nunique(dropna=False) != 1:
                raise RuntimeError(f"Alpha rows disagree on {field}: {(n, idx, seed)}")
        row = group.iloc[0]
        if int(seed) != phase_b.TEST_SEED_BASE[int(n)] + int(idx):
            raise RuntimeError(f"Locked seed mismatch: {(n, idx, seed)}")
        result.append((int(n), int(row.M), int(idx), int(seed), row))
    if len(result) != 80 or {n for n, _, _, _, _ in result} != {8, 20, 30, 50}:
        raise RuntimeError("The deduplicated frame is not the locked 80-instance sample")
    return result


def main() -> None:
    if gp.gurobi.version()[:3] != (13, 0, 3):
        raise RuntimeError(f"Expected locked Gurobi 13.0.3, found {gp.gurobi.version()}")
    if RAW.exists() or PART.exists() or MANIFEST.exists():
        raise RuntimeError("T6.1 output already exists; refusing overwrite")
    instances = locked_instances()
    OUT.mkdir(parents=True, exist_ok=True)
    completed = []
    with PART.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for ordinal, (n, m, idx, seed, saved) in enumerate(instances, 1):
            instance, regenerated_seed = phase_b.make_test_instance(n, m, idx)
            family, b0, ranking = build_cflp_candidate_family(instance, ranking_method="efficiency")
            if regenerated_seed != seed or instance.instance_id != saved.instance_id:
                raise RuntimeError(f"Instance identity mismatch at {n}/{idx}")
            if b0 != int(saved.B_min) or list(map(int, ranking)) != json.loads(saved["ranking"]):
                raise RuntimeError(f"Candidate family mismatch at {n}/{idx}")
            if not np.isclose(float(instance.capacities.sum()), float(saved.total_capacity), atol=1e-9):
                raise RuntimeError(f"Capacity mismatch at {n}/{idx}")
            if not np.isclose(float(instance.total_demand), float(saved.total_demand), atol=1e-9):
                raise RuntimeError(f"Demand mismatch at {n}/{idx}")
            fingerprint = instance_hash(instance)
            prefix = []
            for budget in range(b0, n + 1):
                active = family[budget]
                solved = phase_b.solve_cflp_audited(
                    instance,
                    candidate_set=active,
                    time_limit=phase_b.FULL_TIME_LIMIT if budget == n else phase_b.INITIAL_CANDIDATE_TIME_LIMIT,
                    mip_gap=phase_b.MIP_GAP,
                )
                if not solved["feasible"] or not np.isfinite(solved["bound"]):
                    raise RuntimeError(f"No finite primal-dual record at {n}/{idx}/B={budget}: {solved['status']}")
                if budget == n:
                    full = solved
                prefix.append((budget, solved))
            if not np.isclose(full["incumbent"], float(saved.reference_solver_incumbent), atol=1e-6, rtol=0):
                raise RuntimeError(f"New full objective differs from locked full at {n}/{idx}")
            for budget, solved in prefix:
                writer.writerow({
                    "N": n, "M": m, "test_index": idx, "generation_seed": seed,
                    "instance_id": instance.instance_id, "instance_sha256": fingerprint,
                    "B": budget, "b0": b0, "ranked_active": json.dumps([int(i) for i in ranking[:budget]]),
                    "excluded_fixed_count": n - budget,
                    "pre_presolve_num_vars_code": n * (m + 1),
                    "pre_presolve_num_constrs_code": m + budget,
                    "status": solved["status"], "status_code": solved["status_code"],
                    "sol_count": solved["sol_count"], "candidate_ub": solved["incumbent"],
                    "candidate_lb": solved["bound"],
                    "absolute_gap": max(0.0, solved["incumbent"] - solved["bound"]),
                    "solver_runtime_seconds": solved["solver_runtime_seconds"],
                    "wall_seconds": solved["wall_seconds"], "mip_gap": solved["mip_gap"],
                    "node_count": solved["node_count"],
                    "full_original_ub": float(saved.reference_solver_incumbent),
                    "full_original_lb": float(saved.reference_solver_bound),
                    "full_original_status": saved.reference_solver_status,
                    "full_original_lp_lb": float(saved.lp_valid_bound),
                    "full_new_ub": full["incumbent"], "full_new_lb": full["bound"],
                    "full_new_status": full["status"],
                })
            stream.flush()
            completed.append({"N": n, "M": m, "test_index": idx, "generation_seed": seed,
                              "instance_sha256": fingerprint, "b0": b0, "prefix_count": len(prefix)})
            print(f"[{ordinal}/80] N={n} index={idx} seed={seed} b0={b0} prefixes={len(prefix)}", flush=True)
    os.replace(PART, RAW)
    manifest = {
        "phase": "T6.1 locked-instance full-prefix re-analysis",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_phase_b_raw_sha256": EXPECTED_PHASE_B_SHA,
        "raw_sha256": sha(RAW),
        "code_sha256": {
            str(path.relative_to(ROOT)): sha(path) for path in (
                Path(__file__), ROOT / "experiments" / "final_empirical_strengthening_k20.py",
                ROOT / "src" / "learning_augmented_portfolio" / "benchmarks" / "repeated_cflp.py",
            )
        },
        "analysis_plan_sha256": sha(ROOT / "T6_1_ANALYSIS_PLAN.md"),
        "solver_version": gp.gurobi.version(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "parameters": {
            "generator": "Phase B make_test_instance / generate_iid_cflp_instances",
            "ranking": "efficiency",
            "N_M_scales": phase_b.SCALES,
            "MIPGap": phase_b.MIP_GAP,
            "candidate_TimeLimit": phase_b.INITIAL_CANDIDATE_TIME_LIMIT,
            "full_TimeLimit": phase_b.FULL_TIME_LIMIT,
            "Threads": "AUTO (not explicitly set)",
            "Presolve": "AUTO (not explicitly set)",
            "model": "Phase B full pre-presolve array, zero-fixed excluded blocks",
            "quality_tolerances": [0.0, 1e-4, 1e-3],
        },
        "unique_instances": completed,
        "row_count": int(sum(item["prefix_count"] for item in completed)),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"COMPLETE rows={manifest['row_count']} sha256={manifest['raw_sha256']}", flush=True)


if __name__ == "__main__":
    main()
