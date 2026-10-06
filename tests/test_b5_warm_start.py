import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import final_empirical_strengthening_b5 as b5
from learning_augmented_portfolio.benchmarks.repeated_cflp import build_cflp_candidate_family


def test_valid_candidate_full_vector_and_solver_receipt():
    inst, _ = b5.b.make_test_instance(8, 15, 0)
    family, _, _ = build_cflp_candidate_family(inst, ranking_method="efficiency")
    cand = b5.solve(inst, active=family[7], limit=10)
    check = b5.validate_start(inst, cand["y"], cand["x"], family[7])
    assert abs(check["objective"] - cand["ub"]) < 1e-5
    warm = b5.solve(inst, start=(cand["y"], cand["x"]), limit=15)
    assert warm["start_accepted"] is True
    assert warm["certified"]
    assert warm["abs_gap"] <= b5.ABS_GAP


def test_invalid_excluded_block_is_not_repaired():
    inst, _ = b5.b.make_test_instance(8, 15, 0)
    family, _, _ = build_cflp_candidate_family(inst, ranking_method="efficiency")
    cand = b5.solve(inst, active=family[7], limit=10)
    y = cand["y"].copy()
    x = cand["x"].copy()
    excluded = next(iter(set(range(8)) - family[7]))
    y[excluded] = 1.0
    with pytest.raises(ValueError, match="Excluded block"):
        b5.validate_start(inst, y, x, family[7])


def test_absolute_gap_and_root_relaxation_invariant():
    inst, _ = b5.b.make_test_instance(8, 15, 1)
    family, _, _ = build_cflp_candidate_family(inst, ranking_method="efficiency")
    cand = b5.solve(inst, active=family[7], limit=10)
    cold = b5.solve(inst, limit=15)
    warm = b5.solve(inst, start=(cand["y"], cand["x"]), limit=15)
    assert cold["certified"] and warm["certified"]
    assert cold["abs_gap"] <= b5.ABS_GAP and warm["abs_gap"] <= b5.ABS_GAP
    assert np.isclose(cold["root_obj"], warm["root_obj"], atol=1e-8)
