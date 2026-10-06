"""Capacitated Facility Location Problem (CFLP) generator, solvers, and candidate families."""
import dataclasses
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from ..theory.exactness_budget import compute_family_feasibility_floor_cflp

@dataclasses.dataclass
class CFLPInstance:
    """Represents a Capacitated Facility Location Problem instance theta."""
    instance_id: str
    N: int  # Number of facilities
    M: int  # Number of customers
    fixed_costs: np.ndarray      # shape (N,)
    capacities: np.ndarray       # shape (N,)
    demands: np.ndarray          # shape (M,)
    transport_costs: np.ndarray  # shape (N, M)
    total_demand: float
    features: np.ndarray         # Summary feature vector for ML budget prediction

def generate_iid_cflp_instances(
    num_instances: int = 100,
    N: int = 10,
    M: int = 20,
    seed: int = 42
) -> List[CFLPInstance]:
    """Generate independent, identically distributed (IID) CFLP instances for Class A experiments."""
    rng = np.random.default_rng(seed)
    instances = []
    
    for k in range(num_instances):
        # Facility coordinates in [0, 1]^2
        fac_coords = rng.uniform(0.0, 1.0, size=(N, 2))
        cust_coords = rng.uniform(0.0, 1.0, size=(M, 2))
        
        # Euclidean distances as transport costs
        dists = np.linalg.norm(fac_coords[:, None, :] - cust_coords[None, :, :], axis=-1)
        transport_costs = dists * 10.0
        
        # Fixed costs in [50, 200]
        fixed_costs = rng.uniform(50.0, 200.0, size=N)
        
        # Customer demands in [5, 25]
        demands = rng.uniform(5.0, 25.0, size=M)
        total_demand = float(np.sum(demands))
        
        # Facility capacities such that average total capacity is ~ 2.5 * total_demand
        mean_cap = (2.5 * total_demand) / N
        capacities = rng.uniform(0.6 * mean_cap, 1.4 * mean_cap, size=N)
        
        # Features for ML budget predictor: [N, M, total_demand, mean_cap, mean_fixed_cost, mean_dist]
        features = np.array([
            float(N),
            float(M),
            total_demand,
            float(np.mean(capacities)),
            float(np.mean(fixed_costs)),
            float(np.mean(transport_costs)),
            float(np.std(fixed_costs)),
            float(np.std(capacities))
        ], dtype=float)
        
        inst = CFLPInstance(
            instance_id=f"cflp_iid_{k:04d}",
            N=N,
            M=M,
            fixed_costs=fixed_costs,
            capacities=capacities,
            demands=demands,
            transport_costs=transport_costs,
            total_demand=total_demand,
            features=features
        )
        instances.append(inst)
        
    return instances

def generate_ar1_cflp_sequence(
    T: int = 50,
    N: int = 10,
    M: int = 20,
    rho_d: float = 0.85,
    rho_f: float = 0.90,
    seed: int = 123
) -> List[CFLPInstance]:
    """Generate temporally dependent AR(1) CFLP sequence for Class B stress testing."""
    rng = np.random.default_rng(seed)
    
    # Base network locations
    fac_coords = rng.uniform(0.0, 1.0, size=(N, 2))
    cust_coords = rng.uniform(0.0, 1.0, size=(M, 2))
    base_dists = np.linalg.norm(fac_coords[:, None, :] - cust_coords[None, :, :], axis=-1)
    
    bar_d = rng.uniform(10.0, 20.0, size=M)
    bar_f = rng.uniform(80.0, 150.0, size=N)
    mean_cap = (2.5 * float(np.sum(bar_d))) / N
    capacities = rng.uniform(0.7 * mean_cap, 1.3 * mean_cap, size=N)
    
    curr_d = bar_d.copy()
    curr_f = bar_f.copy()
    curr_scale = 10.0
    
    instances = []
    for t in range(T):
        # AR(1) innovation
        curr_d = np.clip(rho_d * curr_d + (1.0 - rho_d) * bar_d + rng.normal(0, 2.0, size=M), 3.0, 40.0)
        curr_f = np.clip(rho_f * curr_f + (1.0 - rho_f) * bar_f + rng.normal(0, 10.0, size=N), 30.0, 300.0)
        curr_scale = np.clip(0.9 * curr_scale + 0.1 * 10.0 + rng.normal(0, 0.5), 5.0, 20.0)
        
        t_costs = base_dists * curr_scale
        tot_d = float(np.sum(curr_d))
        
        feats = np.array([
            float(N), float(M), tot_d, float(np.mean(capacities)),
            float(np.mean(curr_f)), float(np.mean(t_costs)),
            float(np.std(curr_f)), float(np.std(capacities))
        ], dtype=float)
        
        inst = CFLPInstance(
            instance_id=f"cflp_ar1_t{t:03d}",
            N=N,
            M=M,
            fixed_costs=curr_f.copy(),
            capacities=capacities.copy(),
            demands=curr_d.copy(),
            transport_costs=t_costs.copy(),
            total_demand=tot_d,
            features=feats
        )
        instances.append(inst)
        
    return instances

def build_cflp_candidate_family(
    instance: CFLPInstance,
    ranking_method: str = "efficiency"
) -> Tuple[Dict[int, Set[int]], int, List[int]]:
    """Construct nested candidate family and compute B_{min, C}(theta).
    
    Returns:
        candidate_family: Dict mapping B -> Set of facility indices.
        B_min_C: Minimal feasible budget for this specific candidate family.
        ranking: Ordered list of facility indices.
    """
    N = instance.N
    if ranking_method == "efficiency":
        # Score s_i = f_i / C_i + mean(c_ij)
        scores = instance.fixed_costs / instance.capacities + np.mean(instance.transport_costs, axis=1)
        ranking = list(np.argsort(scores))
    elif ranking_method == "reverse":
        scores = instance.fixed_costs / instance.capacities + np.mean(instance.transport_costs, axis=1)
        ranking = list(np.argsort(scores)[::-1])
    elif ranking_method == "random":
        # Fixed pseudo-random ranking based on instance_id hash
        seed = abs(hash(instance.instance_id)) % (2**31)
        rng = np.random.default_rng(seed)
        ranking = list(rng.permutation(N))
    else:
        raise ValueError(f"Unknown ranking_method: {ranking_method}")
        
    B_min_C = compute_family_feasibility_floor_cflp(
        instance.capacities, instance.demands, ranking, instance.total_demand
    )
    
    candidate_family = {}
    for B in range(B_min_C, N + 1):
        candidate_family[B] = set(ranking[:B])
        
    return candidate_family, B_min_C, ranking

def solve_cflp(
    instance: CFLPInstance,
    candidate_set: Optional[Set[int]] = None,
    time_limit: float = 10.0,
    mip_gap: float = 1e-5
) -> Dict[str, Any]:
    """Solve CFLP or candidate-restricted subproblem using Gurobi."""
    N = instance.N
    M = instance.M
    
    active_facilities = set(range(N)) if candidate_set is None else candidate_set
    
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    
    with gp.Model("cflp", env=env) as model:
        model.setParam("TimeLimit", time_limit)
        model.setParam("MIPGap", mip_gap)
        
        # Variables
        y = {}
        for i in range(N):
            if i in active_facilities:
                y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
            else:
                y[i] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.BINARY, name=f"y_{i}_null")
                
        x = {}
        for i in range(N):
            for j in range(M):
                if i in active_facilities:
                    x[i, j] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")
                else:
                    x[i, j] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}_null")
                    
        model.update()
        
        # Objective: sum f_i y_i + sum c_ij x_ij
        obj_expr = gp.quicksum(instance.fixed_costs[i] * y[i] for i in active_facilities) + \
                   gp.quicksum(instance.transport_costs[i, j] * x[i, j] for i in active_facilities for j in range(M))
        model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # Demand satisfaction constraints
        for j in range(M):
            model.addConstr(
                gp.quicksum(x[i, j] for i in active_facilities) == instance.demands[j],
                name=f"demand_{j}"
            )
            
        # Capacity constraints
        for i in active_facilities:
            model.addConstr(
                gp.quicksum(x[i, j] for j in range(M)) <= instance.capacities[i] * y[i],
                name=f"capacity_{i}"
            )
            
        model.optimize()
        
        status = model.Status
        runtime_ms = model.Runtime * 1000.0
        
        if status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
            ub = model.ObjVal
            lb = model.ObjBound
            abs_gap = ub - lb
            rel_gap = abs_gap / max(1.0, abs(ub))
            
            # Active open facilities
            open_facs = {i for i in active_facilities if y[i].X > 0.5}
            
            return {
                "feasible": True,
                "status": status,
                "upper_bound": ub,
                "lower_bound": lb,
                "abs_gap": abs_gap,
                "rel_gap": rel_gap,
                "active_support": open_facs,
                "runtime_ms": runtime_ms
            }
        else:
            return {
                "feasible": False,
                "status": status,
                "upper_bound": float("inf"),
                "lower_bound": float("-inf"),
                "abs_gap": float("inf"),
                "rel_gap": float("inf"),
                "active_support": set(),
                "runtime_ms": runtime_ms
            }
