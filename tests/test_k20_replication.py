import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "final_empirical_strengthening_k20",
    ROOT / "experiments" / "final_empirical_strengthening_k20.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_recovery_ladder_is_strict_and_reaches_n():
    ladder = MODULE.recovery_ladder(3, 8, rho=1.5)
    assert ladder == [3, 5, 8]
    assert all(a < b for a, b in zip(ladder, ladder[1:]))


def test_b_equals_n_is_not_automatic_certificate():
    certified, source, gap, bound = MODULE.certificate_decision(
        candidate_ub=10.0,
        full_lp_lb=0.0,
        epsilon_cert=1.0,
        at_full_universe=True,
        full_mip_bound=5.0,
    )
    assert certified is False
    assert source == "NONE"
    assert gap == 5.0
    assert bound == 5.0


def test_full_mip_numerical_bound_can_certify_at_n():
    certified, source, gap, _ = MODULE.certificate_decision(
        candidate_ub=10.0,
        full_lp_lb=0.0,
        epsilon_cert=1e-3,
        at_full_universe=True,
        full_mip_bound=9.9995,
    )
    assert certified is True
    assert source == "FULL_MIP_BOUND"
    assert gap <= 1e-3


def test_lp_bound_full_objective_candidate_ub_sandwich():
    inst = MODULE.generate_iid_cflp_instances(1, N=8, M=15, seed=123456)[0]
    family, b_min, _ = MODULE.build_cflp_candidate_family(inst, ranking_method="efficiency")
    lp = MODULE.solve_cflp_audited(inst, relax_openings=True, time_limit=5.0)
    full = MODULE.solve_cflp_audited(inst, time_limit=5.0)
    candidate = MODULE.solve_cflp_audited(inst, family[b_min], time_limit=5.0)
    assert lp["feasible"] and full["feasible"] and candidate["feasible"]
    lp_lb = lp["incumbent"] if lp["status"] == "OPTIMAL" else lp["bound"]
    assert lp_lb <= full["incumbent"] + 1e-7
    assert full["incumbent"] <= candidate["incumbent"] + 1e-7


def test_test_seeds_are_disjoint_and_deterministic():
    all_seeds = []
    for N, _ in MODULE.SCALES:
        observed = [MODULE.make_test_instance(N, dict(MODULE.SCALES)[N], k)[1] for k in range(MODULE.N_TEST)]
        expected = list(range(MODULE.TEST_SEED_BASE[N], MODULE.TEST_SEED_BASE[N] + MODULE.N_TEST))
        assert observed == expected
        all_seeds.extend(observed)
    assert len(all_seeds) == len(set(all_seeds))
