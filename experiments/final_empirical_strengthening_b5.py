"""Phase B.5: preregistered K=20 warm-started operational certification audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import final_empirical_strengthening_k20 as b  # noqa: E402
from learning_augmented_portfolio.benchmarks.repeated_cflp import build_cflp_candidate_family  # noqa: E402

OUT = ROOT / "artifacts" / "k20_replication" / "b5"
REPEATS = 3
THREADS = 1
REL_GAP = 0.0
ABS_GAP = b.EPSILON_CERT
TOL = 1e-5
PIPELINES = ("A_COLD", "B_CAND_LP_COLD", "C_CAND_LP_WARM", "D_CAND_WARM")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status_name(status: int) -> str:
    return b._status_name(status)


def validate_start(instance, y_values, x_values, active):
    n, m = instance.N, instance.M
    if y_values.shape != (n,) or x_values.shape != (n, m):
        raise ValueError("Wrong start dimension")
    if not (np.isfinite(y_values).all() and np.isfinite(x_values).all()):
        raise ValueError("Nonfinite MIP start")
    if np.max(np.abs(y_values - np.rint(y_values))) > TOL:
        raise ValueError("Nonintegral start y")
    if np.min(y_values) < -TOL or np.max(y_values) > 1 + TOL:
        raise ValueError("Opening bound violated")
    if np.min(x_values) < -TOL:
        raise ValueError("Negative shipment")
    inactive = sorted(set(range(n)) - set(active))
    if inactive and (np.max(np.abs(y_values[inactive])) > TOL or np.max(np.abs(x_values[inactive, :])) > TOL):
        raise ValueError("Excluded block not zero")
    demand_error = float(np.max(np.abs(x_values.sum(axis=0) - instance.demands)))
    capacity_violation = float(np.max(x_values.sum(axis=1) - instance.capacities * y_values))
    if demand_error > TOL or capacity_violation > TOL:
        raise ValueError(f"Infeasible start: demand={demand_error}, capacity={capacity_violation}")
    objective = float(instance.fixed_costs @ y_values + np.sum(instance.transport_costs * x_values))
    return {"demand_error": demand_error, "capacity_violation": capacity_violation, "objective": objective}


def solve(instance, *, active=None, lp=False, start=None, limit=15.0):
    n, m = instance.N, instance.M
    active = set(range(n)) if active is None else set(active)
    if start is not None:
        check = validate_start(instance, start[0], start[1], active)
    else:
        check = None
    wall0 = time.perf_counter()
    messages = []
    first_incumbent = None
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model("cflp_b5", env=env) as model:
            model.Params.OutputFlag = 1
            model.Params.LogToConsole = 0
            model.Params.Threads = THREADS
            model.Params.TimeLimit = float(limit)
            if not lp:
                model.Params.MIPGap = REL_GAP
                model.Params.MIPGapAbs = ABS_GAP
            y = {}
            x = {}
            for i in range(n):
                y[i] = model.addVar(lb=0, ub=1 if i in active else 0,
                                    vtype=GRB.CONTINUOUS if lp else GRB.BINARY, name=f"y_{i}")
            for i in range(n):
                for j in range(m):
                    x[i, j] = model.addVar(lb=0, ub=GRB.INFINITY if i in active else 0,
                                           vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")
            model.setObjective(
                gp.quicksum(float(instance.fixed_costs[i]) * y[i] for i in range(n))
                + gp.quicksum(float(instance.transport_costs[i, j]) * x[i, j]
                              for i in range(n) for j in range(m)), GRB.MINIMIZE)
            for j in range(m):
                model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == float(instance.demands[j]))
            for i in range(n):
                model.addConstr(gp.quicksum(x[i, j] for j in range(m)) <= float(instance.capacities[i]) * y[i])
            if start is not None:
                for i in range(n):
                    y[i].Start = float(start[0][i])
                    for j in range(m):
                        x[i, j].Start = float(start[1][i, j])

            def cb(mod, where):
                nonlocal first_incumbent
                if where == GRB.Callback.MESSAGE:
                    messages.append(mod.cbGet(GRB.Callback.MSG_STRING))
                elif where == GRB.Callback.MIPSOL and first_incumbent is None:
                    first_incumbent = {
                        "objective": float(mod.cbGet(GRB.Callback.MIPSOL_OBJ)),
                        "runtime": float(mod.cbGet(GRB.Callback.RUNTIME)),
                    }

            model.optimize(cb if not lp else None)
            elapsed = time.perf_counter() - wall0
            status = int(model.Status)
            sol_count = int(model.SolCount)
            ub = float(model.ObjVal) if sol_count else math.inf
            lb = float(model.ObjBound) if not lp else (ub if status == GRB.OPTIMAL else -math.inf)
            yy = np.array([y[i].X for i in range(n)]) if sol_count else None
            xx = np.array([[x[i, j].X for j in range(m)] for i in range(n)]) if sol_count else None
            log = "".join(messages)
            accepted = None
            if start is not None:
                if "Loaded user MIP start with objective" in log or "User MIP start produced solution with objective" in log:
                    accepted = True
                elif "User MIP start did not produce a new incumbent solution" in log or "User MIP start violates" in log:
                    accepted = False
            root_match = re.search(r"Root relaxation: objective\s+([+\-\d.eE]+)", log)
            root_obj = float(root_match.group(1)) if root_match else math.nan
            abs_gap = max(0.0, ub - lb) if math.isfinite(ub) and math.isfinite(lb) else math.inf
            certified = bool(sol_count and abs_gap <= ABS_GAP + 1e-7)
            if start is not None and accepted is None:
                raise RuntimeError("Gurobi start disposition not identifiable from messages: " + log[:2000])
            return {
                "status": status_name(status), "status_code": status, "ub": ub, "lb": lb,
                "abs_gap": abs_gap, "certified": certified, "termination_by_threshold": bool(certified and status == GRB.OPTIMAL and abs_gap > 1e-7),
                "wall_seconds": elapsed, "solver_seconds": float(model.Runtime), "nodes": float(model.NodeCount),
                "root_obj": root_obj, "first_incumbent_obj": first_incumbent["objective"] if first_incumbent else math.nan,
                "first_incumbent_seconds": first_incumbent["runtime"] if first_incumbent else math.nan,
                "start_accepted": accepted, "start_objective": check["objective"] if check else math.nan,
                "start_demand_error": check["demand_error"] if check else math.nan,
                "start_capacity_violation": check["capacity_violation"] if check else math.nan,
                "y": yy, "x": xx, "log": log,
            }


def evaluate(N, M, alpha, test_index, repeat, calibrated_B):
    instance, seed = b.make_test_instance(N, M, test_index)
    t0 = time.perf_counter()
    _ = instance.features.copy()
    t_features = time.perf_counter() - t0
    t0 = time.perf_counter()
    family, b_min, _ = build_cflp_candidate_family(instance, ranking_method="efficiency")
    t_ranking = time.perf_counter() - t0
    t0 = time.perf_counter()
    initial_B = max(b_min, calibrated_B)
    ladder = b.recovery_ladder(initial_B, N)
    t_budget = time.perf_counter() - t0
    setup = t_features + t_ranking + t_budget
    # Rotate the two independent full solves' order across repetitions.
    cold = None
    candidate = None
    candidate_B = initial_B
    if repeat % 2 == 0:
        cold = solve(instance, limit=b.FULL_TIME_LIMIT)
        candidate = solve(instance, active=family[candidate_B], limit=b.INITIAL_CANDIDATE_TIME_LIMIT if candidate_B < N else b.FULL_TIME_LIMIT)
    else:
        candidate = solve(instance, active=family[candidate_B], limit=b.INITIAL_CANDIDATE_TIME_LIMIT if candidate_B < N else b.FULL_TIME_LIMIT)
        cold = solve(instance, limit=b.FULL_TIME_LIMIT)
    if candidate["y"] is None:
        raise RuntimeError("Candidate MIP has no incumbent")
    start = (candidate["y"], candidate["x"])
    start_check = validate_start(instance, *start, family[candidate_B])
    if abs(start_check["objective"] - candidate["ub"]) > 1e-5:
        raise RuntimeError("Candidate vector objective differs from Gurobi objective")
    lp = solve(instance, lp=True, limit=b.FULL_TIME_LIMIT)
    if lp["status"] != "OPTIMAL":
        raise RuntimeError("Full LP lower bound unavailable")
    lp_certificate = candidate["ub"] - lp["ub"] <= ABS_GAP + 1e-7
    if lp_certificate:
        raise RuntimeError("Unexpected early LP certificate; pipeline accounting requires review")
    # At B=N the candidate solve itself is the full-universe certification solve.
    fallback = {}
    if candidate_B < N:
        orders = ["cold", "warm_c", "warm_d"]
        orders = orders[repeat % 3:] + orders[:repeat % 3]
        for name in orders:
            fallback[name] = solve(instance, start=start if name != "cold" else None, limit=b.FULL_TIME_LIMIT)
    else:
        if not candidate["certified"]:
            raise RuntimeError("B=N candidate full solve not certified")
        fallback = {"cold": candidate, "warm_c": candidate, "warm_d": candidate}
    rows = []
    for name in PIPELINES:
        if name == "A_COLD":
            full = cold
            tc, tl, tf = 0.0, 0.0, cold["wall_seconds"]
            stages, source, accepted = 0, "FULL_MIP_BOUND", None
        else:
            tc = candidate["wall_seconds"]
            tl = lp["wall_seconds"] if name != "D_CAND_WARM" else 0.0
            key = {"B_CAND_LP_COLD": "cold", "C_CAND_LP_WARM": "warm_c", "D_CAND_WARM": "warm_d"}[name]
            full = fallback[key]
            tf = full["wall_seconds"] if candidate_B < N else 0.0
            stages = 1 + int(candidate_B < N)
            source = "FULL_MIP_BOUND" if candidate_B < N else "INITIAL_FULL_MIP_BOUND"
            accepted = full["start_accepted"] if candidate_B < N and name in ("C_CAND_LP_WARM", "D_CAND_WARM") else None
        total = (setup if name != "A_COLD" else 0.0) + tc + tl + tf
        if not full["certified"]:
            raise RuntimeError(f"Uncertified path {N},{alpha},{test_index},{repeat},{name}: {full['status']} {full['abs_gap']}")
        rows.append({
            "N": N, "M": M, "alpha": alpha, "test_index": test_index, "repeat": repeat,
            "generation_seed": seed, "pipeline": name, "B_alpha": calibrated_B, "B_min": b_min,
            "initial_B": initial_B, "initial_full": candidate_B == N, "candidate_stages": stages,
            "T_setup": setup if name != "A_COLD" else 0.0, "T_candidate": tc, "T_LP": tl,
            "T_full_certification": tf, "T_total": total, "certificate_source": source,
            "certified": full["certified"], "full_status": full["status"],
            "full_ub": full["ub"], "full_lb": full["lb"], "full_abs_gap": full["abs_gap"],
            "full_nodes": full["nodes"], "full_root_relax_obj": full["root_obj"],
            "full_first_incumbent_obj": full["first_incumbent_obj"],
            "full_first_incumbent_seconds": full["first_incumbent_seconds"],
            "full_termination_by_threshold": full["termination_by_threshold"],
            "start_accepted": accepted, "start_objective": full["start_objective"] if accepted is not None else math.nan,
            "start_demand_error": full["start_demand_error"] if accepted is not None else math.nan,
            "start_capacity_violation": full["start_capacity_violation"] if accepted is not None else math.nan,
            "candidate_ub": candidate["ub"] if name != "A_COLD" else math.nan,
            "lp_lb": lp["ub"] if name in ("B_CAND_LP_COLD", "C_CAND_LP_WARM") else math.nan,
        })
    return rows


def bootstrap(cell, baseline, rng):
    # Paired resampling of 20 instances; all pipeline measurements share the same indices.
    x = cell["T_total"].to_numpy(float)
    y = baseline["T_total"].to_numpy(float)
    ix = rng.integers(0, len(x), size=(b.BOOTSTRAP_RESAMPLES, len(x)))
    speed = y[ix].mean(axis=1) / x[ix].mean(axis=1)
    diff = (y[ix] - x[ix]).mean(axis=1)
    return np.quantile(speed, [0.025, 0.975]), np.quantile(diff, [0.025, 0.975])


def summarize(raw):
    median = raw.groupby(["N", "alpha", "test_index", "pipeline"], as_index=False).agg({
        **{k: "median" for k in ("T_setup", "T_candidate", "T_LP", "T_full_certification", "T_total", "full_nodes", "full_root_relax_obj", "full_first_incumbent_obj", "full_first_incumbent_seconds", "full_abs_gap", "candidate_stages")},
        "certified": "min", "start_accepted": "mean", "certificate_source": "first", "initial_full": "first",
    })
    rng = np.random.default_rng(b.BOOTSTRAP_SEED)
    rows = []
    for (N, alpha), group in median.groupby(["N", "alpha"], sort=True):
        base = group[group.pipeline == "A_COLD"].sort_values("test_index")
        for pipeline in PIPELINES:
            cell = group[group.pipeline == pipeline].sort_values("test_index")
            if not np.array_equal(base.test_index, cell.test_index):
                raise RuntimeError("Paired test indices differ")
            ci_speed, ci_diff = bootstrap(cell, base, rng)
            delta = base.T_total.to_numpy() - cell.T_total.to_numpy()
            rows.append({
                "N": N, "alpha": alpha, "pipeline": pipeline,
                "mean_total": cell.T_total.mean(), "median_total": cell.T_total.median(),
                "ratio_of_means_speedup": base.T_total.mean() / cell.T_total.mean(),
                "speedup_ci_low": ci_speed[0], "speedup_ci_high": ci_speed[1],
                "mean_paired_saved_seconds": delta.mean(), "median_paired_saved_seconds": np.median(delta),
                "saved_ci_low": ci_diff[0], "saved_ci_high": ci_diff[1],
                "fraction_faster": np.mean(delta > 0),
                "mean_T_candidate": cell.T_candidate.mean(), "mean_T_LP": cell.T_LP.mean(),
                "mean_T_full_certification": cell.T_full_certification.mean(),
                "median_full_nodes": cell.full_nodes.median(), "mean_full_nodes": cell.full_nodes.mean(),
                "mean_candidate_stages": cell.candidate_stages.mean(),
                "certificate_rate": cell.certified.mean(),
                "certificate_sources": ";".join(sorted(cell.certificate_source.unique())),
                "start_acceptance_rate": cell.start_accepted.mean() if cell.start_accepted.notna().any() else math.nan,
                "mean_first_incumbent_seconds": cell.full_first_incumbent_seconds.mean(),
                "mean_root_relax_obj": cell.full_root_relax_obj.mean(),
                "mean_final_abs_gap": cell.full_abs_gap.mean(),
            })
    return median, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cal = pd.read_csv(b.CALIBRATION_SUMMARY_PATH)
    if len(cal) != len(b.SCALES) * len(b.ALPHAS):
        raise RuntimeError("Calibration table incomplete")
    rows = []
    scales = b.SCALES[:1] if args.smoke else b.SCALES
    alphas = b.ALPHAS[:1] if args.smoke else b.ALPHAS
    n_test = 1 if args.smoke else b.N_TEST
    repeats = 1 if args.smoke else REPEATS
    for N, M in scales:
        for alpha in alphas:
            selected = cal[(cal.N == N) & np.isclose(cal.alpha, alpha)]
            if len(selected) != 1:
                raise RuntimeError("Calibration row missing")
            budget = int(selected.iloc[0].calibrated_B)
            for test_index in range(n_test):
                for repeat in range(repeats):
                    rows.extend(evaluate(N, M, alpha, test_index, repeat, budget))
                if not args.smoke:
                    print(f"complete N={N} alpha={alpha} test={test_index + 1}/{n_test}", flush=True)
    raw = pd.DataFrame(rows)
    path = OUT / ("smoke.csv" if args.smoke else "raw_repetitions.csv")
    raw.to_csv(path, index=False)
    if args.smoke:
        print(raw[["pipeline", "T_total", "certified", "start_accepted"]].to_string(index=False))
        return
    med, summary = summarize(raw)
    med.to_csv(OUT / "instance_medians.csv", index=False)
    summary.to_csv(OUT / "cell_summary.csv", index=False)
    manifest = {
        "gurobi_version": gp.gurobi.version(), "python": sys.version, "platform": platform.platform(),
        "repeats": REPEATS, "threads": THREADS, "relative_gap": REL_GAP, "absolute_gap": ABS_GAP,
        "full_time_limit": b.FULL_TIME_LIMIT, "candidate_time_limit": b.INITIAL_CANDIDATE_TIME_LIMIT,
        "bootstrap_resamples": b.BOOTSTRAP_RESAMPLES, "bootstrap_seed": b.BOOTSTRAP_SEED,
        "calibration_sha256": sha256(b.CALIBRATION_SUMMARY_PATH), "phase_b_raw_sha256": sha256(b.RAW_PATH),
        "raw_sha256": sha256(OUT / "raw_repetitions.csv"),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
