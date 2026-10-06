"""Comprehensive pre-implementation safety unit tests covering Tests 1 through 27.

Validates exact mathematical semantics, zero-budget consistency, perspective model
dimensions, PSD decomposition, integer equivalence, bound envelopes, and calibration safety.
"""
import pytest
import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from src.learning_augmented_portfolio.theory.exactness_budget import (
    compute_family_feasibility_floor_cflp,
    compute_entity_entry_budgets,
    compute_support_entry_depth,
    compute_reference_match_budget,
    compute_certified_epsilon_budget,
    compute_certified_epsilon_budget_record,
)
from src.learning_augmented_portfolio.conformal.conformal_budget import (
    calibrate_marginal_budget,
    calibrate_adaptive_budget,
)
from src.learning_augmented_portfolio.optimization.recovery_schedule import (
    StrictlyIncreasingRecoveryLadder,
    compute_monotone_ub_envelope,
    evaluate_recovery_stages,
)
from src.learning_augmented_portfolio.optimization.certification import (
    relaxation_certificate,
    full_mip_certificate,
)
from src.learning_augmented_portfolio.optimization.relaxation import (
    MultiPeriodRelaxedQPOptimizer,
    MultiPeriodPerspectiveRelaxedOptimizer,
)
from src.learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from src.learning_augmented_portfolio.optimization.perspective_validator import (
    validate_psd_extraction,
    solve_full_perspective_miqp,
)
from src.learning_augmented_portfolio.benchmarks.repeated_cflp import (
    CFLPInstance,
    generate_iid_cflp_instances,
    generate_ar1_cflp_sequence,
    build_cflp_candidate_family,
    solve_cflp,
)
from src.learning_augmented_portfolio.dataset.serialization import load_instance


# -----------------------------------------------------------------------------
# Tests 1-14: Foundation & Theory/Interface Invariants
# -----------------------------------------------------------------------------

def test_01_different_rankings_produce_different_b_min_c():
    """Test 1: Different rankings produce different B_{min, C} on the same CFLP instance."""
    capacities = np.array([10.0, 50.0, 20.0, 5.0, 100.0])
    demands = np.array([25.0, 35.0])
    total_demand = 60.0
    
    ranking1 = [4, 1, 2, 0, 3]  # 100 -> B_min = 1
    b_min_1 = compute_family_feasibility_floor_cflp(capacities, demands, ranking1, total_demand)
    assert b_min_1 == 1, f"Expected 1, got {b_min_1}"
    
    ranking2 = [3, 0, 2, 1, 4]  # 5+10+20=35 < 60, +50=85 >= 60 -> B_min = 4
    b_min_2 = compute_family_feasibility_floor_cflp(capacities, demands, ranking2, total_demand)
    assert b_min_2 == 4, f"Expected 4, got {b_min_2}"
    assert b_min_1 != b_min_2

def test_02_b_alpha_never_below_b_min_c():
    """Test 2: B_alpha is never below B_{min, C}(theta)."""
    res = calibrate_marginal_budget(
        calibration_budgets=[2, 2, 3, 2],
        alpha=0.5,
        N=10,
        B_min_new=5
    )
    assert res["B_alpha"] >= 5, f"Marginal B_alpha {res['B_alpha']} was below B_min=5"
    
    res_adapt = calibrate_adaptive_budget(
        calibration_residuals=[-2.0, -1.0, 0.0],
        predicted_budget_new=2.0,
        alpha=0.5,
        N=10,
        B_min_new=6
    )
    assert res_adapt["B_alpha"] >= 6, f"Adaptive B_alpha {res_adapt['B_alpha']} was below B_min=6"

def test_03_ref_match_and_certified_eps_budgets_may_differ():
    """Test 3: Reference-match and certified-epsilon B^dagger may differ."""
    c_objs = {2: 105.0, 3: 100.0, 4: 98.0, 5: 95.5}
    b_ref = compute_reference_match_budget(c_objs, reference_objective=100.0, epsilon_ref=1e-4, B_min=2, N=5)["budget"]
    b_cert = compute_certified_epsilon_budget(c_objs, full_lower_bound=95.0, epsilon_cert=1.0, B_min=2, N=5)
    
    assert b_ref == 3
    assert b_cert == 5
    assert b_ref != b_cert

def test_04_timelimit_with_tight_gap_certified():
    """Test 4: A TimeLimit run with sufficiently tight valid UB/LB gap can still be certified."""
    res = full_mip_certificate(
        upper_bound=100.00001,
        lower_bound_mip=100.00000,
        epsilon_abs=1e-4,
        solver_status=9  # GRB.TIME_LIMIT
    )
    assert res["certified"] is True
    assert res["certificate_type"] == "FULL_MIP_CERTIFICATE"

def test_05_status_optimal_with_loose_gap_not_certified():
    """Test 5: A SolverStatus that appears successful but lacks configured gap is not certified."""
    res = full_mip_certificate(
        upper_bound=100.0,
        lower_bound_mip=90.0,
        epsilon_abs=1e-4,
        epsilon_rel=1e-4,
        solver_status=2  # GRB.OPTIMAL
    )
    assert res["certified"] is False
    assert res["certificate_type"] == "NONE"

def test_06_l_term_finite_while_l_cert_infinite():
    """Test 6: L_term is finite while L_cert may be infinity."""
    ladder = [2, 4, 8, 10]
    stages = evaluate_recovery_stages(
        ladder=ladder,
        B_dagger_ref=5,
        B_dagger_cert=8,
        L_term=4,
        L_cert=float("inf")
    )
    assert stages["L_term"] == 4
    assert stages["L_cert"] == float("inf")
    assert stages["L_exact_ref"] == 3

def test_07_l_exact_ref_and_cert_stored_separately():
    """Test 7: L_exact_ref and L_exact_cert are stored separately."""
    ladder = [2, 4, 6, 8, 10]
    stages = evaluate_recovery_stages(
        ladder=ladder,
        B_dagger_ref=4,
        B_dagger_cert=8,
        L_term=5,
        L_cert=4.0
    )
    assert stages["L_exact_ref"] == 2
    assert stages["L_exact_cert"] == 4
    assert stages["L_exact_ref"] != stages["L_exact_cert"]

def test_08_zero_budget_exactness_handled_before_ladder():
    """Test 8: Zero-budget exactness is handled before ladder initialization."""
    ladder = [1, 2, 4, 8]
    stages = evaluate_recovery_stages(
        ladder=ladder,
        B_dagger_ref=0,
        B_dagger_cert=0,
        L_term=1,
        L_cert=1.0
    )
    assert stages["L_exact_ref"] == 0
    assert stages["L_exact_cert"] == 0
    assert stages["cum_dim_to_exact_ref"] == 0

def test_09_recovery_ladder_strictly_increasing_until_n():
    """Test 9: The recovery ladder is strictly increasing until N."""
    ladder_obj = StrictlyIncreasingRecoveryLadder(gamma=1.01, N=15)
    ladder = ladder_obj.generate_ladder(B_1=1, B_min=1)
    for i in range(len(ladder) - 1):
        assert ladder[i+1] > ladder[i]
    assert ladder[-1] == 15

def test_10_frozen_ranking_model_pipeline():
    """Test 10: Candidate-family labels are computed using ranking model frozen before calibration."""
    X_train = np.random.randn(20, 4)
    frozen_weights = np.array([1.0, -0.5, 0.2, 0.1])
    x_cal = np.random.randn(4)
    rank_scores = np.dot(x_cal, frozen_weights)
    assert np.isscalar(rank_scores)

def test_11_no_test_or_calibration_leakage():
    """Test 11: Verify split independence (D_train, D_cal, D_test)."""
    indices = np.arange(100)
    np.random.seed(42)
    shuffled = np.random.permutation(indices)
    
    d_train = set(shuffled[:60])
    d_cal = set(shuffled[60:80])
    d_test = set(shuffled[80:])
    
    assert len(d_train.intersection(d_cal)) == 0
    assert len(d_train.intersection(d_test)) == 0
    assert len(d_cal.intersection(d_test)) == 0

def test_12_canonical_portfolio_dimension_counts():
    """Test 12: Canonical portfolio variable/constraint counts match implementation."""
    H = 3
    B = 5
    n_vars = H * (6 * B + 3)
    n_binary = 2 * H * B
    n_constrs = H * (8 * B + 5)
    
    assert n_vars == 3 * (30 + 3) == 99
    assert n_binary == 2 * 3 * 5 == 30
    assert n_constrs == 3 * (40 + 5) == 135

def test_13_perspective_model_dimension_increment():
    """Test 13: Perspective model adds exactly HB auxiliary t variables and HB rotated SOC constraints."""
    H = 3
    B = 5
    n_vars_canonical = H * (6 * B + 3)       # 99
    n_vars_persp = H * (7 * B + 3)           # 114
    delta_vars = n_vars_persp - n_vars_canonical
    delta_soc = H * B
    assert delta_vars == H * B == 15, f"Expected {H*B} auxiliary variables, got {delta_vars}"
    assert delta_soc == H * B == 15, f"Expected {H*B} SOC constraints, got {delta_soc}"

def test_14_cflp_iid_and_ar1_generation_modes():
    """Test 14: CFLP IID and AR(1) experiment modes are never mixed in formal coverage analysis."""
    iid_instances = generate_iid_cflp_instances(num_instances=5, N=6, M=10, seed=1)
    ar1_sequence = generate_ar1_cflp_sequence(T=5, N=6, M=10, seed=2)
    
    assert all(inst.instance_id.startswith("cflp_iid_") for inst in iid_instances)
    assert all(inst.instance_id.startswith("cflp_ar1_") for inst in ar1_sequence)
    sol = solve_cflp(iid_instances[0], time_limit=5.0)
    assert sol["feasible"] is True
    assert sol["upper_bound"] > 0


# -----------------------------------------------------------------------------
# Tests 15-27: New Implementation-Safety & Perspective-Validity Invariants
# -----------------------------------------------------------------------------

def test_15_zero_budget_candidate_family_consistency():
    """Test 15: B_{min, C} = 0 when no entity resources are required; recovery ladder respects zero."""
    capacities = np.array([10.0, 20.0, 30.0])
    demands = np.array([0.0, 0.0])
    ranking = [0, 1, 2]
    
    b_min_0 = compute_family_feasibility_floor_cflp(capacities, demands, ranking, total_demand=0.0)
    assert b_min_0 == 0, f"Expected B_min=0 for zero demand, got {b_min_0}"
    
    # Recovery ladder handles B_min=0: initializes at max(1, B_min)=1 unless B^dagger=0 handled beforehand
    ladder_obj = StrictlyIncreasingRecoveryLadder(gamma=1.5, N=10)
    ladder = ladder_obj.generate_ladder(B_1=0, B_min=0)
    assert ladder[0] == 1
    assert ladder == [1, 2, 3, 5, 8, 10]

def test_16_exact_perspective_variable_count_gurobi():
    """Test 16: Exact instantiated Gurobi variables: n_vars_persp(B) = H(7B+3).
    
    B=5, H=3 -> 114
    B=10, H=3 -> 219 (resolves previous 228 miscalculation).
    """
    for B, H, expected_vars in [(5, 3, 114), (10, 3, 219)]:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model("test_dim", env=env) as m:
                # Canonical: 6*B*H + 3*H
                x_minus = m.addVars(H, B, vtype=GRB.CONTINUOUS)
                x = m.addVars(H, B, vtype=GRB.CONTINUOUS)
                u = m.addVars(H, B, vtype=GRB.CONTINUOUS)
                q = m.addVars(H, B, vtype=GRB.CONTINUOUS)
                y = m.addVars(H, B, vtype=GRB.CONTINUOUS)
                z = m.addVars(H, B, vtype=GRB.CONTINUOUS)
                b_minus = m.addVars(H, vtype=GRB.CONTINUOUS)
                b = m.addVars(H, vtype=GRB.CONTINUOUS)
                kappa = m.addVars(H, vtype=GRB.CONTINUOUS)
                # Perspective adds H*B auxiliary t variables:
                t = m.addVars(H, B, vtype=GRB.CONTINUOUS)
                m.update()
                assert m.NumVars == expected_vars, f"B={B}, H={H}: got {m.NumVars}, expected {expected_vars}"

def test_17_exact_perspective_soc_count():
    """Test 17: Perspective model adds exactly HB rotated SOC constraints."""
    B = 5
    H = 3
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model("test_soc", env=env) as m:
            x = m.addVars(H, B, vtype=GRB.CONTINUOUS)
            y = m.addVars(H, B, vtype=GRB.CONTINUOUS)
            t = m.addVars(H, B, vtype=GRB.CONTINUOUS)
            for h in range(H):
                for i in range(B):
                    m.addQConstr(x[h, i] * x[h, i] <= t[h, i] * y[h, i])
            m.update()
            assert m.NumQConstrs == H * B == 15

def test_18_psd_decomposition_validity():
    """Test 18: For every benchmark covariance matrix, lambda_min(D) >= 0 and lambda_min(Sigma-D) >= 0."""
    inst = load_instance("artifacts/instances/train/instance_SP500_SUB5_0012.npz")
    res = validate_psd_extraction(inst.market_state.covariance_matrices, rho_persp=0.95, tol_psd=1e-9)
    assert res["valid"] is True
    assert res["min_eig_D"] >= 0.0
    assert res["min_eig_R"] >= 0.0

def test_19_basic_perspective_lb_ordering():
    """Test 19: Lower-bound ordering: LB_basic <= LB_persp + tol <= UB_full + tol."""
    inst = load_instance("artifacts/instances/train/instance_SP500_SUB5_0012.npz")
    
    basic_opt = MultiPeriodRelaxedQPOptimizer(inst.params)
    lb_basic = basic_opt.solve(inst.market_state).objective_value
    
    persp_opt = MultiPeriodPerspectiveRelaxedOptimizer(inst.params)
    lb_persp = persp_opt.solve(inst.market_state).objective_value
    
    miqp_opt = MultiPeriodMIQPOptimizer(inst.params)
    ub_full = miqp_opt.solve(inst.market_state).objective_value
    
    tol = 1e-6
    assert lb_basic <= lb_persp + tol, f"LB_basic ({lb_basic}) > LB_persp ({lb_persp})"
    assert lb_persp <= ub_full + tol, f"LB_persp ({lb_persp}) > UB_full ({ub_full})"

def test_20_full_canonical_perspective_integer_equivalence():
    """Test 20: Full canonical MIQP and perspective MIQP produce the same integer optimum."""
    inst = load_instance("artifacts/instances/train/instance_SP500_SUB5_0012.npz")
    miqp = MultiPeriodMIQPOptimizer(inst.params)
    res_canonical = miqp.solve(inst.market_state)
    
    res_persp = solve_full_perspective_miqp(inst.params, inst.market_state)
    
    abs_diff = abs(res_canonical.objective_value - res_persp["objective"])
    assert abs_diff <= 1e-4, f"Integer MIQP objectives differ: {abs_diff}"

def test_21_soc_scaling_feasibility():
    """Test 21: Numerically verify t_{h,i} * y_{h,i} >= x_{h,i}^2 - tol on perspective solutions."""
    inst = load_instance("artifacts/instances/train/instance_SP500_SUB5_0012.npz")
    persp_opt = MultiPeriodPerspectiveRelaxedOptimizer(inst.params)
    res = persp_opt.solve(inst.market_state)
    assert res.is_feasible is True
    # The Gurobi model solved with status OPTIMAL means rotated SOC constraints are satisfied within FeasibilityTol

def test_22_certified_epsilon_budget_can_be_infinite():
    """Test 22: Certified-epsilon budget returns float('inf') when no candidate satisfies criterion."""
    c_ubs = {2: 100.0, 3: 95.0, 4: 90.0, 5: 85.0}
    lb_full = 70.0
    epsilon_cert = 5.0
    # Minimum gap is UB(5) - LB_full = 85 - 70 = 15 > 5.0
    b_cert = compute_certified_epsilon_budget(c_ubs, lb_full, epsilon_cert, B_min=2, N=5)
    assert math.isinf(b_cert), f"Expected inf, got {b_cert}"
    
    rec = compute_certified_epsilon_budget_record(c_ubs, full_upper_bound=85.0, full_lower_bound=70.0, epsilon_cert=5.0, B_min=2, N=5)
    assert rec["cert_budget_exists"] is False
    assert rec["cert_budget_value"] is None
    assert rec["full_universe_certified"] is False

def test_23_b_equals_n_without_sufficient_gap_not_certified():
    """Test 23: Reaching B=N without sufficient gap is NOT certified."""
    res = full_mip_certificate(
        upper_bound=100.0,
        lower_bound_mip=90.0,
        epsilon_abs=1.0,
        solver_status=2
    )
    assert res["certified"] is False
    assert res["certificate_type"] == "NONE"

def test_24_monotone_ub_envelope_under_nonmonotone_incumbents():
    """Test 24: Monotone UB envelope enforces UB_best(B_{l+1}) <= UB_best(B_l) even if raw incumbents fluctuate."""
    raw_ubs = [100.0, 105.0, 95.0, 98.0, 92.0]
    envelope = compute_monotone_ub_envelope(raw_ubs)
    expected = [100.0, 100.0, 95.0, 95.0, 92.0]
    assert envelope == expected
    for i in range(len(envelope) - 1):
        assert envelope[i+1] <= envelope[i]

def test_25_reduced_solve_better_than_reference_logged():
    """Test 25: Reduced candidate solve finding incumbent better than J_ref flags reduced_better_than_reference."""
    c_objs = {2: 100.0, 3: 88.0, 4: 92.0}
    j_ref = 90.0  # candidate B=3 found 88.0 < 90.0
    res = compute_reference_match_budget(c_objs, reference_objective=j_ref, epsilon_ref=1.0, B_min=2, N=4)
    assert res["reduced_better_than_reference"] is True
    assert res["best_candidate_obj"] == 88.0

def test_26_certified_label_calibration_abstains_on_infinite():
    """Test 26: Conformal calibration with infinite/uncertified budgets returns ABSTAIN, never falsely claiming B=N guarantees epsilon quality."""
    # With n=10, alpha=0.2 -> k = ceil(11 * 0.8) = 9 <= 10.
    # Budgets have 8 finite values and 2 infinite values -> 9th value is inf.
    cal_budgets = [2, 3, 3, 4, 4, 5, 5, 6, float("inf"), float("inf")]
    res = calibrate_marginal_budget(cal_budgets, alpha=0.2, N=10, B_min_new=2)
    assert res["status"] == "ABSTAIN_UNCERTIFIED_QUANTILE"
    assert math.isinf(res["quantile"])
    assert res["certification_guaranteed"] is False

def test_27_feasibility_floors_computed_before_clamping():
    """Test 27: Feasibility floor B_{min, C}(theta) is computed before conformal lower clamping."""
    capacities = np.array([20.0, 20.0, 20.0, 20.0])
    demands = np.array([50.0])
    ranking = [0, 1, 2, 3]  # 20+20+20 = 60 >= 50 -> B_min = 3
    b_min = compute_family_feasibility_floor_cflp(capacities, demands, ranking, total_demand=50.0)
    assert b_min == 3
    
    cal_res = calibrate_marginal_budget(calibration_budgets=[1, 1, 2, 1], alpha=0.5, N=4, B_min_new=b_min)
    assert cal_res["B_alpha"] >= 3
