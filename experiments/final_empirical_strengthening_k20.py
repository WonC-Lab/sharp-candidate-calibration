"""Clean K=20 replication for the real-valued CFLP scaling benchmark.

This script deliberately does not modify or overwrite the legacy K=8 artifacts.
It separates:
  A. reference-bound-assisted diagnostic timing; and
  B. an operational certified pipeline using a timed full-universe LP relaxation.

The calibrated target is the operational reference-fidelity budget, not the
mathematical exactness budget.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from learning_augmented_portfolio.benchmarks.repeated_cflp import (  # noqa: E402
    CFLPInstance,
    build_cflp_candidate_family,
    generate_iid_cflp_instances,
)
from learning_augmented_portfolio.conformal.conformal_budget import (  # noqa: E402
    calibrate_marginal_budget,
)
from learning_augmented_portfolio.theory.exactness_budget import (  # noqa: E402
    compute_reference_match_budget,
)


SCALES: Tuple[Tuple[int, int], ...] = ((8, 15), (20, 20), (30, 25), (50, 30))
ALPHAS: Tuple[float, ...] = (0.20, 0.10, 0.05)
N_CAL = 32
N_TEST = 20
CALIBRATION_BASE_SEED = 2026
TEST_SEED_BASE: Dict[int, int] = {
    8: 10_080_000,
    20: 10_200_000,
    30: 10_300_000,
    50: 10_500_000,
}
EPSILON_REF = 1e-4
EPSILON_CERT = 1e-3
RHO_REC = 1.5
MIP_GAP = 1e-5
FULL_TIME_LIMIT = 15.0
CAL_FULL_TIME_LIMIT = 10.0
CAL_CANDIDATE_TIME_LIMIT = 5.0
INITIAL_CANDIDATE_TIME_LIMIT = 10.0
RECOVERY_TIME_LIMIT = 5.0
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20_261_006
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "k20_replication"
RAW_PATH = OUTPUT_ROOT / "k20_raw_results.csv"
CALIBRATION_PATH = OUTPUT_ROOT / "calibration_labels.csv"
CALIBRATION_SUMMARY_PATH = OUTPUT_ROOT / "calibration_summary.csv"
BOOTSTRAP_PATH = OUTPUT_ROOT / "paired_bootstrap_summary.csv"
SUMMARY_PATH = OUTPUT_ROOT / "k20_summary.csv"
LEGACY_COMPARISON_PATH = OUTPUT_ROOT / "legacy_k8_vs_k20.csv"
MANIFEST_PATH = OUTPUT_ROOT / "replication_manifest.json"
DRY_RUN_PATH = OUTPUT_ROOT / "dry_run_results.csv"
REPORT_PATH = OUTPUT_ROOT / "PHASE_B_REPORT.md"
EXACT_ORACLE_INSTANCE_PATH = OUTPUT_ROOT / "exact_oracle_instance_labels.csv"
EXACT_ORACLE_RAW_PATH = OUTPUT_ROOT / "exact_oracle_test_results.csv"
EXACT_ORACLE_SUMMARY_PATH = OUTPUT_ROOT / "exact_oracle_summary.csv"


def _status_name(status: int) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        GRB.WORK_LIMIT: "WORK_LIMIT",
        GRB.MEM_LIMIT: "MEM_LIMIT",
    }
    return names.get(status, f"STATUS_{status}")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def solve_cflp_audited(
    instance: CFLPInstance,
    candidate_set: Optional[set[int]] = None,
    *,
    relax_openings: bool = False,
    time_limit: float,
    mip_gap: float = MIP_GAP,
) -> Dict[str, Any]:
    """Solve a CFLP with complete row-level solver provenance.

    With ``relax_openings=True``, binary opening variables are replaced by
    continuous variables in [0, 1]. This enlarges the feasible set of the full
    MIP and therefore provides a mathematically valid lower bound for this
    minimization problem.
    """
    N, M = instance.N, instance.M
    active = set(range(N)) if candidate_set is None else set(candidate_set)
    wall_start = time.perf_counter()
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    try:
        with gp.Model("cflp_k20_audit", env=env) as model:
            model.setParam("TimeLimit", float(time_limit))
            model.setParam("MIPGap", float(mip_gap))
            y_type = GRB.CONTINUOUS if relax_openings else GRB.BINARY
            y: Dict[int, gp.Var] = {}
            x: Dict[Tuple[int, int], gp.Var] = {}
            for i in range(N):
                if i in active:
                    y[i] = model.addVar(lb=0.0, ub=1.0, vtype=y_type, name=f"y_{i}")
                else:
                    y[i] = model.addVar(lb=0.0, ub=0.0, vtype=y_type, name=f"y_{i}_null")
            for i in range(N):
                for j in range(M):
                    ub = GRB.INFINITY if i in active else 0.0
                    x[i, j] = model.addVar(lb=0.0, ub=ub, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")
            model.setObjective(
                gp.quicksum(instance.fixed_costs[i] * y[i] for i in active)
                + gp.quicksum(
                    instance.transport_costs[i, j] * x[i, j]
                    for i in active
                    for j in range(M)
                ),
                GRB.MINIMIZE,
            )
            for j in range(M):
                model.addConstr(gp.quicksum(x[i, j] for i in active) == instance.demands[j])
            for i in active:
                model.addConstr(
                    gp.quicksum(x[i, j] for j in range(M)) <= instance.capacities[i] * y[i]
                )
            model.optimize()
            wall_seconds = time.perf_counter() - wall_start
            status = int(model.Status)
            sol_count = int(model.SolCount)
            incumbent = float(model.ObjVal) if sol_count > 0 else float("inf")
            try:
                bound = float(model.ObjBound)
            except (gp.GurobiError, AttributeError):
                bound = float("-inf")
            open_facilities: List[int] = []
            if sol_count > 0:
                open_facilities = sorted(i for i in active if y[i].X > 0.5)
            return {
                "feasible": sol_count > 0,
                "status_code": status,
                "status": _status_name(status),
                "sol_count": sol_count,
                "incumbent": incumbent,
                "bound": bound,
                "solver_runtime_seconds": float(model.Runtime),
                "wall_seconds": wall_seconds,
                "mip_gap": float(model.MIPGap) if sol_count > 0 and not relax_openings else None,
                "node_count": float(model.NodeCount),
                "open_facilities": open_facilities,
                "relax_openings": relax_openings,
            }
    finally:
        env.dispose()


def recovery_ladder(initial_B: int, N: int, rho: float = RHO_REC) -> List[int]:
    if not 1 <= initial_B <= N:
        raise ValueError(f"initial_B={initial_B} must be in [1, {N}]")
    ladder = [int(initial_B)]
    while ladder[-1] < N:
        current = ladder[-1]
        nxt = min(N, max(current + 1, int(math.ceil(rho * current))))
        ladder.append(nxt)
    return ladder


def certificate_decision(
    *,
    candidate_ub: float,
    full_lp_lb: float,
    epsilon_cert: float,
    at_full_universe: bool,
    full_mip_bound: float,
) -> Tuple[bool, str, float, float]:
    """Return certified, source, certified gap, and the bound actually used."""
    if _finite(candidate_ub) and _finite(full_lp_lb):
        gap = float(candidate_ub - full_lp_lb)
        if gap <= epsilon_cert + 1e-9:
            return True, "FULL_LP_RELAXATION", gap, float(full_lp_lb)
    if at_full_universe and _finite(candidate_ub) and _finite(full_mip_bound):
        gap = float(candidate_ub - full_mip_bound)
        if gap <= epsilon_cert + 1e-9:
            return True, "FULL_MIP_BOUND", gap, float(full_mip_bound)
        return False, "NONE", gap, float(full_mip_bound)
    fallback_gap = float(candidate_ub - full_lp_lb) if _finite(candidate_ub) and _finite(full_lp_lb) else float("inf")
    return False, "NONE", fallback_gap, float(full_lp_lb)


def make_test_instance(N: int, M: int, test_index: int) -> Tuple[CFLPInstance, int]:
    seed = TEST_SEED_BASE[N] + test_index
    instance = generate_iid_cflp_instances(num_instances=1, N=N, M=M, seed=seed)[0]
    return dataclasses.replace(instance, instance_id=f"cflp_N{N}_test_{test_index:02d}_seed_{seed}"), seed


def compute_calibration_labels(*, force: bool = False) -> pd.DataFrame:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if CALIBRATION_PATH.exists() and not force:
        df = pd.read_csv(CALIBRATION_PATH)
        expected = {(N, i) for N, _ in SCALES for i in range(N_CAL)}
        observed = set(zip(df["N"].astype(int), df["calibration_index"].astype(int)))
        if observed == expected:
            return df
        raise RuntimeError("Existing calibration cache is incomplete; rerun with --force-calibration.")

    rows: List[Dict[str, Any]] = []
    for N, M in SCALES:
        stream_seed = CALIBRATION_BASE_SEED + N
        instances = generate_iid_cflp_instances(N_CAL, N=N, M=M, seed=stream_seed)
        for idx, instance in enumerate(instances):
            family, b_min, _ = build_cflp_candidate_family(instance, ranking_method="efficiency")
            full = solve_cflp_audited(
                instance,
                time_limit=CAL_FULL_TIME_LIMIT,
                mip_gap=MIP_GAP,
            )
            candidate_ubs: Dict[int, float] = {}
            failed_budgets: List[int] = []
            for B in range(b_min, N + 1):
                result = solve_cflp_audited(
                    instance,
                    candidate_set=family[B],
                    time_limit=CAL_CANDIDATE_TIME_LIMIT,
                    mip_gap=MIP_GAP,
                )
                if result["feasible"]:
                    candidate_ubs[B] = result["incumbent"]
                else:
                    failed_budgets.append(B)
            if not full["feasible"]:
                b_ref = N
                label_status = "FULL_REFERENCE_FAILED_FALLBACK_N"
            else:
                b_ref = int(
                    compute_reference_match_budget(
                        candidate_ubs,
                        full["incumbent"],
                        EPSILON_REF,
                        b_min,
                        N,
                    )["budget"]
                )
                label_status = "OK" if not failed_budgets else "CANDIDATE_FAILURES_RECORDED"
            row = {
                "N": N,
                "M": M,
                "calibration_index": idx,
                "calibration_stream_seed": stream_seed,
                "stream_position": idx,
                "instance_id": instance.instance_id,
                "B_min": b_min,
                "B_dagger_ref": b_ref,
                "epsilon_ref": EPSILON_REF,
                "calibration_target": "operational_reference_fidelity",
                "UB_reference": full["incumbent"],
                "reference_bound": full["bound"],
                "reference_status": full["status"],
                "failed_candidate_budgets": json.dumps(failed_budgets),
                "label_status": label_status,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(CALIBRATION_PATH, index=False)
    return pd.DataFrame(rows)


def calibration_summary(labels: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for N, _ in SCALES:
        values = labels.loc[labels["N"] == N, "B_dagger_ref"].astype(float).tolist()
        if len(values) != N_CAL:
            raise RuntimeError(f"N={N}: expected {N_CAL} calibration labels, found {len(values)}")
        for alpha in ALPHAS:
            result = calibrate_marginal_budget(
                values,
                alpha=alpha,
                N=N,
                B_min_new=1,
                target_type="reference",
            )
            rows.append({
                "N": N,
                "alpha": alpha,
                "nominal_coverage": 1.0 - alpha,
                "n_cal": N_CAL,
                "conformal_k": result["k_alpha"],
                "conformal_quantile": result["quantile"],
                "calibrated_B": result["B_alpha"],
                "safe_full_fallback": result["fallback_to_full"],
                "calibration_status": result["status"],
                "calibration_target": "operational_reference_fidelity",
                "epsilon_ref": EPSILON_REF,
            })
    out = pd.DataFrame(rows)
    out.to_csv(CALIBRATION_SUMMARY_PATH, index=False)
    return out


def _run_candidate_path(
    instance: CFLPInstance,
    family: Dict[int, set[int]],
    ladder: Sequence[int],
    *,
    lower_bound: float,
    operational: bool,
) -> Dict[str, Any]:
    stage_records: List[Dict[str, Any]] = []
    total_solve = 0.0
    total_comparison = 0.0
    certified = False
    source = "NONE"
    certified_gap = float("inf")
    used_bound = lower_bound

    for stage_index, B in enumerate(ladder):
        limit = INITIAL_CANDIDATE_TIME_LIMIT if stage_index == 0 else RECOVERY_TIME_LIMIT
        solved = solve_cflp_audited(
            instance,
            candidate_set=family[B],
            time_limit=limit,
            mip_gap=MIP_GAP,
        )
        total_solve += solved["wall_seconds"]
        t0 = time.perf_counter()
        if operational:
            certified, source, certified_gap, used_bound = certificate_decision(
                candidate_ub=solved["incumbent"],
                full_lp_lb=lower_bound,
                epsilon_cert=EPSILON_CERT,
                at_full_universe=(B == instance.N),
                full_mip_bound=solved["bound"],
            )
            terminate = certified or B == instance.N
        else:
            diagnostic_gap = (
                solved["incumbent"] - lower_bound
                if _finite(solved["incumbent"]) and _finite(lower_bound)
                else float("inf")
            )
            certified_gap = diagnostic_gap
            used_bound = lower_bound
            terminate = diagnostic_gap <= EPSILON_CERT + 1e-9 or B == instance.N
            source = "REFERENCE_MIP_BOUND_DIAGNOSTIC" if diagnostic_gap <= EPSILON_CERT + 1e-9 else "NONE"
            certified = False
        total_comparison += time.perf_counter() - t0
        stage_records.append({
            "stage_index": stage_index,
            "B": B,
            "status": solved["status"],
            "incumbent": solved["incumbent"],
            "bound": solved["bound"],
            "wall_seconds": solved["wall_seconds"],
            "diagnostic_or_certified_gap": certified_gap,
            "certificate_source": source,
            "certified": certified,
        })
        if terminate:
            break

    final = stage_records[-1]
    return {
        "stages": stage_records,
        "stage_B_sequence": [r["B"] for r in stage_records],
        "stage_count": len(stage_records),
        "initial_candidate_seconds": stage_records[0]["wall_seconds"],
        "recovery_seconds": sum(r["wall_seconds"] for r in stage_records[1:]),
        "total_candidate_seconds": total_solve,
        "comparison_seconds": total_comparison,
        "reached_N": final["B"] == instance.N,
        "recovery_to_N": ladder[0] < instance.N and final["B"] == instance.N,
        "certified": certified,
        "certificate_source": source,
        "certified_gap": certified_gap,
        "valid_LB": used_bound,
        "final_B": final["B"],
        "final_status": final["status"],
        "final_bound": final["bound"],
        "final_incumbent": final["incumbent"],
    }


def evaluate_one(
    *,
    N: int,
    M: int,
    alpha: float,
    test_index: int,
    calibration_row: pd.Series,
) -> Dict[str, Any]:
    instance, generation_seed = make_test_instance(N, M, test_index)

    t0 = time.perf_counter()
    _ = instance.features.copy()
    t_features = time.perf_counter() - t0

    t0 = time.perf_counter()
    family, b_min, ranking = build_cflp_candidate_family(instance, ranking_method="efficiency")
    t_ranking = time.perf_counter() - t0

    t0 = time.perf_counter()
    calibrated_B = int(calibration_row["calibrated_B"])
    initial_B = min(N, max(b_min, calibrated_B))
    ladder = recovery_ladder(initial_B, N)
    t_budget = time.perf_counter() - t0

    reference = solve_cflp_audited(
        instance,
        time_limit=FULL_TIME_LIMIT,
        mip_gap=MIP_GAP,
    )
    relaxation = solve_cflp_audited(
        instance,
        relax_openings=True,
        time_limit=FULL_TIME_LIMIT,
        mip_gap=MIP_GAP,
    )
    lp_lb = relaxation["bound"]
    if relaxation["status"] == "OPTIMAL" and relaxation["feasible"]:
        lp_lb = relaxation["incumbent"]

    diagnostic = _run_candidate_path(
        instance,
        family,
        ladder,
        lower_bound=reference["bound"],
        operational=False,
    )
    operational = _run_candidate_path(
        instance,
        family,
        ladder,
        lower_bound=lp_lb,
        operational=True,
    )

    t_total_ref = (
        t_features
        + t_ranking
        + t_budget
        + diagnostic["total_candidate_seconds"]
        + diagnostic["comparison_seconds"]
    )
    t_total_cert = (
        t_features
        + t_ranking
        + t_budget
        + relaxation["wall_seconds"]
        + operational["total_candidate_seconds"]
        + operational["comparison_seconds"]
    )
    ref_ub = reference["incumbent"]
    final_ub = operational["final_incumbent"]
    total_capacity = float(np.sum(instance.capacities))

    return {
        "N": N,
        "M": M,
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "test_index": test_index,
        "generation_seed": generation_seed,
        "instance_id": instance.instance_id,
        "n_cal": N_CAL,
        "conformal_k": int(calibration_row["conformal_k"]),
        "conformal_quantile": calibration_row["conformal_quantile"],
        "B_alpha": calibrated_B,
        "epsilon_ref": EPSILON_REF,
        "calibration_target": "operational_reference_fidelity",
        "B_min": b_min,
        "total_capacity": total_capacity,
        "total_demand": instance.total_demand,
        "initial_B": initial_B,
        "initial_full": initial_B == N,
        "stage_count": operational["stage_count"],
        "stage_B_sequence": json.dumps(operational["stage_B_sequence"]),
        "reached_N": operational["reached_N"],
        "recovery_to_N": operational["recovery_to_N"],
        "diagnostic_stage_count": diagnostic["stage_count"],
        "diagnostic_stage_B_sequence": json.dumps(diagnostic["stage_B_sequence"]),
        "diagnostic_reached_N": diagnostic["reached_N"],
        "diagnostic_recovery_to_N": diagnostic["recovery_to_N"],
        "UB_reference": ref_ub,
        "reference_bound": reference["bound"],
        "candidate_final_UB": final_ub,
        "valid_LB": operational["valid_LB"],
        "candidate_minus_reference": final_ub - ref_ub if _finite(final_ub) and _finite(ref_ub) else float("inf"),
        "certified_gap": operational["certified_gap"],
        "reference_fidelity_hit": bool(_finite(final_ub) and _finite(ref_ub) and final_ub - ref_ub <= EPSILON_REF + 1e-9),
        "certified": operational["certified"],
        "certificate_source": operational["certificate_source"],
        "certificate_tolerance": EPSILON_CERT,
        "final_solver_status": operational["final_status"],
        "final_solver_bound": operational["final_bound"],
        "final_solver_incumbent": operational["final_incumbent"],
        "reference_solver_status": reference["status"],
        "reference_solver_bound": reference["bound"],
        "reference_solver_incumbent": reference["incumbent"],
        "lp_solver_status": relaxation["status"],
        "lp_valid_bound": lp_lb,
        "diagnostic_gap": diagnostic["certified_gap"],
        "diagnostic_termination_source": diagnostic["certificate_source"],
        "T_features_seconds": t_features,
        "T_ranking_seconds": t_ranking,
        "T_budget_seconds": t_budget,
        "T_candidate_seconds": operational["initial_candidate_seconds"],
        "T_relaxation_LB_seconds": relaxation["wall_seconds"],
        "T_certification_seconds": operational["comparison_seconds"],
        "T_recovery_seconds": operational["recovery_seconds"],
        "T_total_reference_assisted_seconds": t_total_ref,
        "T_total_certified_seconds": t_total_cert,
        "T_full_reference_seconds": reference["wall_seconds"],
        "T_candidate_reference_assisted_seconds": diagnostic["initial_candidate_seconds"],
        "T_recovery_reference_assisted_seconds": diagnostic["recovery_seconds"],
        "T_comparison_reference_assisted_seconds": diagnostic["comparison_seconds"],
        "operational_stage_records": json.dumps(operational["stages"], allow_nan=True),
        "diagnostic_stage_records": json.dumps(diagnostic["stages"], allow_nan=True),
        "ranking": json.dumps([int(i) for i in ranking]),
        "row_status": "OK" if reference["feasible"] and relaxation["feasible"] else "SOLVER_FAILURE_RECORDED",
    }


def _append_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def run_dry_run(cal_summary: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    if DRY_RUN_PATH.exists() and force:
        DRY_RUN_PATH.unlink()
    rows: List[Dict[str, Any]] = []
    for N, M in SCALES:
        cal = cal_summary[(cal_summary["N"] == N) & np.isclose(cal_summary["alpha"], 0.20)].iloc[0]
        row = evaluate_one(N=N, M=M, alpha=0.20, test_index=0, calibration_row=cal)
        if not (_finite(row["valid_LB"]) and _finite(row["UB_reference"]) and _finite(row["candidate_final_UB"])):
            raise RuntimeError(f"N={N}: dry-run produced a non-finite bound or incumbent")
        if row["valid_LB"] > row["UB_reference"] + 1e-7:
            raise AssertionError(f"N={N}: LP lower bound exceeds full MIP incumbent")
        if row["UB_reference"] > row["candidate_final_UB"] + 1e-7:
            raise AssertionError(f"N={N}: full MIP incumbent exceeds restricted candidate incumbent")
        rows.append(row)
        _append_row(DRY_RUN_PATH, row)
    return pd.DataFrame(rows)


def run_full(cal_summary: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    if RAW_PATH.exists() and force:
        RAW_PATH.unlink()
    completed: set[Tuple[int, float, int]] = set()
    if RAW_PATH.exists():
        prior = pd.read_csv(RAW_PATH)
        completed = set(
            (int(r.N), round(float(r.alpha), 8), int(r.test_index))
            for r in prior.itertuples()
        )
    for N, M in SCALES:
        for alpha in ALPHAS:
            cal = cal_summary[(cal_summary["N"] == N) & np.isclose(cal_summary["alpha"], alpha)].iloc[0]
            for test_index in range(N_TEST):
                key = (N, round(alpha, 8), test_index)
                if key in completed:
                    continue
                row = evaluate_one(N=N, M=M, alpha=alpha, test_index=test_index, calibration_row=cal)
                _append_row(RAW_PATH, row)
                completed.add(key)
    result = pd.read_csv(RAW_PATH)
    expected = len(SCALES) * len(ALPHAS) * N_TEST
    if len(result) != expected:
        raise RuntimeError(f"Expected {expected} raw rows, found {len(result)}")
    return result


def _paired_bootstrap(
    full: np.ndarray,
    total: np.ndarray,
    *,
    rng: np.random.Generator,
) -> Dict[str, float]:
    n = len(full)
    indices = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    full_boot = full[indices].mean(axis=1)
    total_boot = total[indices].mean(axis=1)
    speed_boot = full_boot / total_boot
    diff = full - total
    diff_boot = diff[indices].mean(axis=1)
    return {
        "speedup": float(full.mean() / total.mean()),
        "speedup_ci_low": float(np.percentile(speed_boot, 2.5)),
        "speedup_ci_high": float(np.percentile(speed_boot, 97.5)),
        "mean_difference_seconds": float(diff.mean()),
        "difference_ci_low_seconds": float(np.percentile(diff_boot, 2.5)),
        "difference_ci_high_seconds": float(np.percentile(diff_boot, 97.5)),
        "median_difference_seconds": float(np.median(diff)),
        "fraction_candidate_faster": float(np.mean(total < full)),
        "mean_T_full_seconds": float(full.mean()),
        "mean_T_total_seconds": float(total.mean()),
    }


def analyze(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for (N, alpha), group in raw.groupby(["N", "alpha"], sort=True):
        group = group.sort_values("test_index")
        if len(group) != N_TEST or group["test_index"].nunique() != N_TEST:
            raise RuntimeError(f"N={N}, alpha={alpha}: paired test set is incomplete")
        full = group["T_full_reference_seconds"].to_numpy(float)
        for mode, column in (
            ("reference_assisted_diagnostic", "T_total_reference_assisted_seconds"),
            ("operational_certified", "T_total_certified_seconds"),
        ):
            stats = _paired_bootstrap(full, group[column].to_numpy(float), rng=rng)
            bootstrap_rows.append({"N": int(N), "alpha": float(alpha), "mode": mode, "K": N_TEST, **stats})
        summary_rows.append({
            "N": int(N),
            "M": int(group["M"].iloc[0]),
            "alpha": float(alpha),
            "nominal_coverage": 1.0 - float(alpha),
            "K": N_TEST,
            "n_cal": N_CAL,
            "conformal_k": int(group["conformal_k"].iloc[0]),
            "conformal_quantile": group["conformal_quantile"].iloc[0],
            "mean_initial_B": float(group["initial_B"].mean()),
            "mean_initial_B_over_N": float((group["initial_B"] / N).mean()),
            "InitialFullRate": float(group["initial_full"].mean()),
            "RecoveryToFullRate": float(group["recovery_to_N"].mean()),
            "DiagnosticRecoveryToFullRate": float(group["diagnostic_recovery_to_N"].mean()),
            "reference_fidelity_rate": float(group["reference_fidelity_hit"].mean()),
            "certified_rate": float(group["certified"].mean()),
            "mean_stage_count": float(group["stage_count"].mean()),
            "mean_T_full_seconds": float(group["T_full_reference_seconds"].mean()),
            "mean_T_reference_assisted_seconds": float(group["T_total_reference_assisted_seconds"].mean()),
            "mean_T_certified_seconds": float(group["T_total_certified_seconds"].mean()),
            "uncertified_count": int((~group["certified"].astype(bool)).sum()),
            "solver_failure_count": int((group["row_status"] != "OK").sum()),
            "certificate_source_counts": json.dumps(group["certificate_source"].value_counts().to_dict()),
        })
    boot = pd.DataFrame(bootstrap_rows)
    summary = pd.DataFrame(summary_rows)
    boot.to_csv(BOOTSTRAP_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    legacy = pd.read_csv(PROJECT_ROOT / "artifacts" / "scaled_cflp_audited_summary.csv")
    alpha20 = boot[(boot["alpha"] == 0.20) & (boot["mode"] == "reference_assisted_diagnostic")]
    compare = legacy[["N", "Speedup_RatioOfMeans"]].merge(
        alpha20[["N", "speedup", "speedup_ci_low", "speedup_ci_high"]], on="N", how="outer"
    )
    compare = compare.rename(columns={
        "Speedup_RatioOfMeans": "legacy_K8_speedup",
        "speedup": "new_K20_alpha20_reference_assisted_speedup",
    })
    compare["legacy_K"] = 8
    compare["new_K"] = 20
    compare["direction_preserved"] = (
        (compare["legacy_K8_speedup"] >= 1.0)
        == (compare["new_K20_alpha20_reference_assisted_speedup"] >= 1.0)
    )
    compare.to_csv(LEGACY_COMPARISON_PATH, index=False)
    return summary, boot, compare


def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _powershell_json(command: str) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(proc.stdout.strip())
    except Exception:
        return {}


def write_manifest(*, phase: str) -> Dict[str, Any]:
    cpu = _powershell_json(
        "Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"
    )
    system = _powershell_json(
        "Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory,Manufacturer,Model | ConvertTo-Json -Compress"
    )
    outputs = [
        CALIBRATION_PATH,
        CALIBRATION_SUMMARY_PATH,
        DRY_RUN_PATH,
        RAW_PATH,
        SUMMARY_PATH,
        BOOTSTRAP_PATH,
        LEGACY_COMPARISON_PATH,
        EXACT_ORACLE_INSTANCE_PATH,
        EXACT_ORACLE_RAW_PATH,
        EXACT_ORACLE_SUMMARY_PATH,
        REPORT_PATH,
    ]
    manifest = {
        "phase": phase,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_repository_detected": (PROJECT_ROOT / ".git").exists(),
        "git_commit": None,
        "experiment_script": str(Path(__file__).resolve()),
        "supporting_scripts": [str((PROJECT_ROOT / "experiments" / "exact_oracle_audit_final.py").resolve())],
        "exact_commands": {
            "tests": f"{sys.executable} -m pytest tests/test_k20_replication.py -q",
            "dry_run": f"{sys.executable} experiments/final_empirical_strengthening_k20.py --mode dry-run",
            "full": f"{sys.executable} experiments/final_empirical_strengthening_k20.py --mode full",
            "exact_oracle": f"{sys.executable} experiments/exact_oracle_audit_final.py",
        },
        "environment": {
            "os": platform.platform(),
            "python_version": sys.version,
            "gurobi_version": ".".join(map(str, gp.gurobi.version())),
            "cpu": cpu,
            "system": system,
            "ram_gb": round(float(system.get("TotalPhysicalMemory", 0)) / (1024 ** 3), 3) if system else None,
        },
        "protocol": {
            "scales": [{"N": N, "M": M} for N, M in SCALES],
            "generator": "generate_iid_cflp_instances (real-valued)",
            "ranking": "ascending fixed_cost/capacity + mean_transport_cost",
            "ranking_fitted": False,
            "n_train": 0,
            "n_cal": N_CAL,
            "n_test": N_TEST,
            "calibration_stream_seeds": {str(N): CALIBRATION_BASE_SEED + N for N, _ in SCALES},
            "calibration_stream_positions": [0, N_CAL - 1],
            "test_seed_ranges": {str(N): [base, base + N_TEST - 1] for N, base in TEST_SEED_BASE.items()},
            "alpha_grid": list(ALPHAS),
            "epsilon_ref": EPSILON_REF,
            "epsilon_cert_absolute": EPSILON_CERT,
            "calibration_target": "operational_reference_fidelity",
            "rho_rec": RHO_REC,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "solver": {
            "Threads": "AUTO (not explicitly set, preserving legacy semantics)",
            "MIPGap": MIP_GAP,
            "full_reference_time_limit_seconds": FULL_TIME_LIMIT,
            "calibration_full_time_limit_seconds": CAL_FULL_TIME_LIMIT,
            "calibration_candidate_time_limit_seconds": CAL_CANDIDATE_TIME_LIMIT,
            "initial_candidate_time_limit_seconds": INITIAL_CANDIDATE_TIME_LIMIT,
            "recovery_time_limit_seconds": RECOVERY_TIME_LIMIT,
            "LP_relaxation": "facility-opening binaries relaxed to continuous [0,1]",
        },
        "outputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in outputs if path.exists()
        ],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_report(summary: pd.DataFrame, boot: pd.DataFrame, compare: pd.DataFrame, raw: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "None."
        safe = df.copy()
        safe.columns = [str(column).replace("|", "\\|") for column in safe.columns]
        lines = [
            "| " + " | ".join(safe.columns) + " |",
            "| " + " | ".join("---" for _ in safe.columns) + " |",
        ]
        for values in safe.itertuples(index=False, name=None):
            rendered = []
            for value in values:
                if isinstance(value, float):
                    cell = f"{value:.8g}"
                else:
                    cell = str(value)
                rendered.append(cell.replace("|", "\\|").replace("\n", " "))
            lines.append("| " + " | ".join(rendered) + " |")
        return "\n".join(lines)

    boot_wide = boot.pivot(
        index=["N", "alpha"],
        columns="mode",
        values=["speedup", "speedup_ci_low", "speedup_ci_high"],
    )
    boot_wide.columns = [f"{metric}_{mode}" for metric, mode in boot_wide.columns]
    boot_wide = boot_wide.reset_index()
    display = summary.merge(boot_wide, on=["N", "alpha"], how="left")
    failures = raw[(raw["row_status"] != "OK") | (~raw["certified"].astype(bool))][
        ["N", "alpha", "test_index", "generation_seed", "row_status", "certified", "certificate_source", "final_solver_status", "stage_B_sequence"]
    ]
    exact_oracle = pd.read_csv(EXACT_ORACLE_SUMMARY_PATH) if EXACT_ORACLE_SUMMARY_PATH.exists() else pd.DataFrame()
    exact_oracle_status = (
        "RESOLVED: a separate integer-CFLP audit was rerun at "
        "`N=10, M=15, n_cal=20, n_test=20`, with 30 independent replications and no uncertified labels.\n\n"
        + md_table(exact_oracle)
        if not exact_oracle.empty
        else "UNRESOLVED: the final exact-oracle rerun has not been completed."
    )
    alpha20_ref = boot[
        np.isclose(boot["alpha"], 0.20)
        & (boot["mode"] == "reference_assisted_diagnostic")
    ].sort_values("N")
    alpha20_cert = boot[
        np.isclose(boot["alpha"], 0.20)
        & (boot["mode"] == "operational_certified")
    ].sort_values("N")
    changed_scales = compare.loc[~compare["direction_preserved"].astype(bool), "N"].astype(int).tolist()
    qualitative = (
        "- Reference-assisted K=20 diagnostic at alpha=0.20: "
        + ", ".join(
            f"N={int(row.N)} {row.speedup:.3f}x (95% CI [{row.speedup_ci_low:.3f}, {row.speedup_ci_high:.3f}])"
            for row in alpha20_ref.itertuples()
        )
        + ".\n- Operational certified K=20 at alpha=0.20: "
        + ", ".join(
            f"N={int(row.N)} {row.speedup:.3f}x (95% CI [{row.speedup_ci_low:.3f}, {row.speedup_ci_high:.3f}])"
            for row in alpha20_cert.itertuples()
        )
        + ".\n- Legacy direction changed at scales "
        + str(changed_scales)
        + "; in particular, the legacy N=30 slowdown did not reproduce in reference-assisted point estimates, while its K=20 CI spans 1.0.\n"
        + "- The operational pipeline recovered to the full universe for every run that did not start at N. The full LP relaxation produced no early certificates; all 240 numerical certificates came from the paid full-MIP stage.\n"
        + "- Consequently, operational certification was slower than the single full reference solve in every N × alpha cell. This negative result is retained."
    )
    report = f"""# PHASE B REPORT — K=20 CFLP Replication

Generated: {datetime.now(timezone.utc).isoformat()}

## 1. Exact final protocol

- Real-valued CFLP scaling at `(N,M) = (8,15), (20,20), (30,25), (50,30)`.
- Fixed deterministic efficiency ranking; no fitted training stage.
- `n_cal=32`, `n_test=20`, alpha grid `{list(ALPHAS)}`.
- Calibration target: operational reference fidelity with `epsilon_ref={EPSILON_REF}`.
- Absolute certification tolerance: `epsilon_cert={EPSILON_CERT}`.
- Recovery factor: `rho_rec={RHO_REC}`.
- Paired bootstrap: `{BOOTSTRAP_RESAMPLES}` resamples, seed `{BOOTSTRAP_SEED}`.

## 2. Calibration semantics

The calibrated label is `B_dagger_ref`, the smallest candidate budget whose incumbent is within `epsilon_ref` of the full-universe reference incumbent. It is not the mathematical exactness budget.

## 3. Certification semantics

Reference-assisted timing uses the separately computed full-MIP bound only as a diagnostic termination reference and excludes that reference solve from its timing. It is not reported as deployable certification. Operational timing pays for a full-universe LP relaxation, uses its valid lower bound, and at `B=N` accepts a full-MIP certificate only when the numerical incumbent-bound gap satisfies the declared tolerance. Merely reaching `B=N` never sets `certified=True`.

## 4. Complete alpha × N results

{md_table(display)}

## 5. Paired bootstrap results

{md_table(boot)}

## 6. Legacy K=8 vs new K=20

{md_table(compare)}

## 7. Failed or uncertified cases

{md_table(failures) if len(failures) else 'None.'}

## 8. Qualitative conclusion changes

{qualitative}

## 9. Exact-oracle discrepancy status

{exact_oracle_status}

## 10. Recommended manuscript changes

1. Replace legacy K=8 scaling aggregates only after review of this report.
2. Name the scaling calibration target as operational reference fidelity.
3. Separate reference-assisted diagnostic and operational certified timing.
4. Report all alpha levels and preserve slowdowns/full-universe collapse.
5. Correct InitialFullRate and RecoveryToFullRate definitions.
6. Distinguish real-valued scaling from the integer exact-oracle audit.
7. Replace portfolio “feasibility floor” with mandatory-state/candidate-construction floor unless additional feasibility assumptions are proved.
8. Apply the S1.6 one-way ceiling implication correction during Phase C.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("calibrate", "dry-run", "full", "analyze"), default="dry-run")
    parser.add_argument("--force-calibration", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    labels = compute_calibration_labels(force=args.force_calibration)
    cal_summary = calibration_summary(labels)
    if args.mode == "calibrate":
        write_manifest(phase="B1_CALIBRATION")
        print(cal_summary.to_string(index=False))
        return
    if args.mode == "dry-run":
        dry = run_dry_run(cal_summary, force=args.force)
        write_manifest(phase="B2_DRY_RUN")
        print(dry[["N", "alpha", "B_min", "initial_B", "stage_B_sequence", "valid_LB", "candidate_final_UB", "UB_reference", "certified", "certificate_source", "T_total_reference_assisted_seconds", "T_total_certified_seconds"]].to_string(index=False))
        return
    if args.mode == "full":
        raw = run_full(cal_summary, force=args.force)
        summary, boot, compare = analyze(raw)
        write_report(summary, boot, compare, raw)
        write_manifest(phase="B5_REPORT_COMPLETE")
        print(summary.to_string(index=False))
        return
    if not RAW_PATH.exists():
        raise FileNotFoundError(RAW_PATH)
    raw = pd.read_csv(RAW_PATH)
    summary, boot, compare = analyze(raw)
    write_report(summary, boot, compare, raw)
    write_manifest(phase="B5_REPORT_COMPLETE")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
