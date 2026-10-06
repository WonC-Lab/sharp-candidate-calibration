"""Exact-objective integer CFLP benchmark and oracle calibration.

Mathematical Principle:
In a complete bipartite transportation network where:
- facility fixed costs f_i are integers,
- unit transportation costs c_ij are integers,
- customer demands d_j are integers,
- facility capacities C_i are integers,
the minimum cost network flow polyhedron for any fixed binary decision y in {0, 1}^N
is totally unimodular. Hence, all extreme points are integral (x_ij in Z), and the
total objective J(y, x) is strictly integer-valued for every feasible solution.

Theorem (Integer Optimality Certificate):
If a solver produces an integer-feasible incumbent with upper bound UB in Z and a valid
dual/continuous lower bound LB such that:
    UB - LB < 1 - delta (delta > 0),
then because J(y, x) in Z for all feasible points, no distinct feasible objective can satisfy
J < UB. Thus:
    UB = J^* exactly.

This certifies the exact mathematical oracle target B^dagger_C(theta) without numerical ambiguity.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
import math
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from ..theory.exactness_budget import compute_family_feasibility_floor_cflp
from ..conformal.conformal_budget import calibrate_marginal_budget, calibrate_adaptive_budget


@dataclass
class IntegerCFLPInstance:
    """CFLP instance with strictly integer data ensuring integral optimal objectives."""
    instance_id: str
    N: int
    M: int
    fixed_costs: np.ndarray       # shape (N,) integer int64
    capacities: np.ndarray        # shape (N,) integer int64
    demands: np.ndarray           # shape (M,) integer int64
    transport_costs: np.ndarray   # shape (N, M) integer int64
    total_demand: int             # integer
    features: np.ndarray          # continuous instance features for ranking / predictor


def generate_integer_cflp_instances(
    num_instances: int = 50,
    N: int = 10,
    M: int = 20,
    seed: int = 42
) -> List[IntegerCFLPInstance]:
    """Generate instances with strictly integral parameters."""
    rng = np.random.default_rng(seed)
    instances = []
    
    for idx in range(num_instances):
        inst_id = f"cflp_exact_{idx:04d}"
        
        # Strictly integer costs and capacities
        fixed_costs = rng.integers(150, 450, size=N, dtype=np.int64)
        demands = rng.integers(10, 30, size=M, dtype=np.int64)
        tot_d = int(np.sum(demands))
        
        # Ensure total capacity exceeds total demand
        mean_dem_per_fac = tot_d / max(1, N // 3)
        capacities = rng.integers(int(mean_dem_per_fac * 0.8), int(mean_dem_per_fac * 1.6), size=N, dtype=np.int64)
        while np.sum(capacities) < tot_d * 1.5:
            capacities += rng.integers(10, 30, size=N, dtype=np.int64)
            
        transport_costs = rng.integers(5, 50, size=(N, M), dtype=np.int64)
        
        # Instance features for ranking / prediction
        feats = np.array([
            float(np.mean(fixed_costs)),
            float(np.std(fixed_costs)),
            float(np.mean(capacities)),
            float(np.std(capacities)),
            float(tot_d),
            float(np.mean(transport_costs)),
            float(np.sum(capacities)) / float(tot_d),
            float(N),
            float(M)
        ], dtype=float)
        
        inst = IntegerCFLPInstance(
            instance_id=inst_id,
            N=N,
            M=M,
            fixed_costs=fixed_costs,
            capacities=capacities,
            demands=demands,
            transport_costs=transport_costs,
            total_demand=tot_d,
            features=feats
        )
        instances.append(inst)
        
    return instances


def solve_exact_integer_cflp(
    instance: IntegerCFLPInstance,
    candidate_set: Optional[Set[int]] = None,
    time_limit: float = 10.0,
    safety_margin: float = 0.05
) -> Dict[str, Any]:
    """Solve CFLP and verify exact integer optimality via UB - LB < 1 - safety_margin."""
    N = instance.N
    M = instance.M
    active_facilities = sorted(list(candidate_set)) if candidate_set is not None else list(range(N))
    
    # Capacity check
    tot_active_cap = sum(int(instance.capacities[i]) for i in active_facilities)
    if tot_active_cap < instance.total_demand:
        return {
            "feasible": False,
            "exact_certified": False,
            "upper_bound": float("inf"),
            "lower_bound": float("-inf"),
            "exact_integer_objective": None,
            "gap": float("inf")
        }
        
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model("Exact_Integer_CFLP", env=env) as m:
            m.Params.TimeLimit = time_limit
            m.Params.MIPGap = 1e-6
            m.Params.IntegralityFocus = 1  # prioritize integer arithmetic
            m.Params.FeasibilityTol = 1e-9
            m.Params.OptimalityTol = 1e-9
            
            y = m.addVars(active_facilities, vtype=GRB.BINARY, name="y")
            x = m.addVars(active_facilities, M, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
            
            # Objective: strictly integer coefficients
            obj = gp.quicksum(int(instance.fixed_costs[i]) * y[i] for i in active_facilities) + \
                  gp.quicksum(int(instance.transport_costs[i, j]) * x[i, j] for i in active_facilities for j in range(M))
            m.setObjective(obj, GRB.MINIMIZE)
            
            # Demand satisfaction: sum_i x_ij == d_j
            for j in range(M):
                m.addConstr(gp.quicksum(x[i, j] for i in active_facilities) == int(instance.demands[j]))
                
            # Capacity constraints: sum_j x_ij <= C_i * y_i
            for i in active_facilities:
                m.addConstr(gp.quicksum(x[i, j] for j in range(M)) <= int(instance.capacities[i]) * y[i])
                
            start_t = time.perf_counter()
            m.optimize()
            sol_time = time.perf_counter() - start_t
            
            if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
                ub = m.ObjVal
                lb = m.ObjBound
                gap = ub - lb
                
                # Check integer exactness certification criterion: UB - LB < 1 - safety_margin
                is_exact = (gap < (1.0 - safety_margin))
                int_obj = int(round(ub)) if is_exact else None
                
                open_facs = {i for i in active_facilities if y[i].X > 0.5}
                return {
                    "feasible": True,
                    "exact_certified": is_exact,
                    "upper_bound": ub,
                    "lower_bound": lb,
                    "exact_integer_objective": int_obj,
                    "gap": gap,
                    "active_support": open_facs,
                    "solve_time": sol_time
                }
            return {
                "feasible": False,
                "exact_certified": False,
                "upper_bound": float("inf"),
                "lower_bound": float("-inf"),
                "exact_integer_objective": None,
                "gap": float("inf"),
                "solve_time": sol_time
            }


def build_exact_candidate_family(
    instance: IntegerCFLPInstance
) -> Tuple[Dict[int, Set[int]], int, List[int]]:
    """Build efficiency ranking and feasibility floor for integer CFLP."""
    scores = instance.fixed_costs / instance.capacities + np.mean(instance.transport_costs, axis=1)
    ranking = list(np.argsort(scores))
    B_min = compute_family_feasibility_floor_cflp(
        instance.capacities, instance.demands, ranking, float(instance.total_demand)
    )
    family = {B: set(ranking[:B]) for B in range(B_min, instance.N + 1)}
    return family, B_min, ranking


def evaluate_oracle_exactness_conformal_study(
    M_replications: int = 50,
    n_train: int = 15,
    n_cal: int = 60,
    n_test: int = 15,
    N: int = 10,
    M: int = 20,
    alphas: list = [0.20, 0.10, 0.05],
    seed: int = 777
) -> Dict[str, Any]:
    """Execute direct empirical audit of the exact conformal theorem on certified integer optima."""
    rng = np.random.default_rng(seed)
    total_inst = n_train + n_cal + n_test
    
    rep_records = []
    
    for rep in range(1, M_replications + 1):
        rep_seed = int(rng.integers(1, 1000000))
        instances = generate_integer_cflp_instances(num_instances=total_inst, N=N, M=M, seed=rep_seed)
        
        cal_insts = instances[n_train:n_train + n_cal]
        test_insts = instances[n_train + n_cal:]
        
        def compute_oracle_b_dagger(inst):
            fam, b_min, rank = build_exact_candidate_family(inst)
            full_sol = solve_exact_integer_cflp(inst, candidate_set=None, time_limit=5.0)
            if not full_sol["exact_certified"]:
                return None, None
            j_star = full_sol["exact_integer_objective"]
            
            b_dagger = inst.N
            for B in range(b_min, inst.N + 1):
                sub_sol = solve_exact_integer_cflp(inst, candidate_set=fam[B], time_limit=3.0)
                if sub_sol["exact_certified"] and sub_sol["exact_integer_objective"] == j_star:
                    b_dagger = B
                    break
            return b_dagger, b_min
            
        cal_b_daggers = []
        for inst in cal_insts:
            b_dag, b_min = compute_oracle_b_dagger(inst)
            if b_dag is not None:
                cal_b_daggers.append(b_dag)
                
        if len(cal_b_daggers) < 40:
            continue
            
        test_data = []
        for inst in test_insts:
            b_dag, b_min = compute_oracle_b_dagger(inst)
            if b_dag is not None:
                test_data.append((b_dag, b_min))
                
        if len(test_data) == 0:
            continue
            
        for alpha in alphas:
            cal_res = calibrate_marginal_budget(cal_b_daggers, alpha=alpha, N=N, target_type="exact_oracle")
            q_alpha = cal_res["quantile"]
            
            hits = 0
            b_alpha_list = []
            for b_dag, b_min in test_data:
                b_alpha = min(N, max(b_min, int(q_alpha)))
                if b_dag <= b_alpha:
                    hits += 1
                b_alpha_list.append(b_alpha)
                
            cov_rate = hits / len(test_data)
            rep_records.append({
                "replication": rep,
                "alpha": alpha,
                "nominal_coverage": 1.0 - alpha,
                "exact_oracle_coverage": cov_rate,
                "mean_B_alpha": float(np.mean(b_alpha_list)),
                "candidate_ratio": float(np.mean(b_alpha_list)) / N
            })
            
    df = pd.DataFrame(rep_records)
    summary = []
    for alpha in alphas:
        sub = df[df["alpha"] == alpha]
        covs = sub["exact_oracle_coverage"].values
        summary.append({
            "alpha": alpha,
            "nominal_coverage": 1.0 - alpha,
            "exact_oracle_mean_coverage": float(np.mean(covs)),
            "exact_oracle_se": float(np.std(covs, ddof=1) / math.sqrt(len(covs))),
            "ci_95_lower": float(np.percentile(covs, 2.5)),
            "ci_95_upper": float(np.percentile(covs, 97.5)),
            "mean_candidate_ratio": float(sub["candidate_ratio"].mean()),
            "replications": len(sub)
        })
    return pd.DataFrame(summary)
