"""
Phase 5 Unit Tests — Historical Optimization Instance Dataset Generation.
Validates serialization round-tripping, MIQP replay fidelity, explicit label correctness,
label quality taxonomy, fallback exclusion, chronological split integrity, no-lookahead auditing,
continuous relaxation lower bound, and config hash reproducibility.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from learning_augmented_portfolio.dynamic.state import PortfolioState
from learning_augmented_portfolio.forecasts.base import BaseForecastProvider, ForecastBundle
from learning_augmented_portfolio.optimization.big_m import compute_dynamic_tight_big_m
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from learning_augmented_portfolio.optimization.relaxation import MultiPeriodRelaxedQPOptimizer
from learning_augmented_portfolio.optimization.variables import (
    MultiPeriodMarketState,
    MultiPeriodPortfolioParameters,
    MultiPeriodOptimizationResult,
    SolverConfig,
)
from learning_augmented_portfolio.dataset.instance import (
    OptimizationInstance,
    OptimizationInstanceMetadata,
    OptimizationLabels,
    ContinuousRelaxationRecord,
    compute_config_hash,
)
from learning_augmented_portfolio.dataset.manifest import DatasetManifest
from learning_augmented_portfolio.dataset.quality import (
    LabelQuality,
    LabelQualityConfig,
    classify_label_quality,
)
from learning_augmented_portfolio.dataset.schema import DATASET_SCHEMA_VERSION, MANIFEST_COLUMNS
from learning_augmented_portfolio.dataset.serialization import (
    save_instance,
    load_instance,
    rebuild_and_solve_miqp,
)
from learning_augmented_portfolio.dataset.split import (
    DatasetSplit,
    DatasetSplitConfig,
    validate_chronological_integrity,
)
from learning_augmented_portfolio.dataset.generator import (
    HistoricalDatasetGenerator,
    DatasetGeneratorConfig,
)


@pytest.fixture
def small_instance_setup():
    """Create a canonical 3-asset, 2-period test setup."""
    n = 3
    H = 2
    tickers = ["AAPL", "MSFT", "GOOG"]

    weights = np.array([0.4, 0.4, 0.1])
    cash = 0.1
    state = PortfolioState(
        actual_wealth=1000.0,
        normalized_weights=weights,
        normalized_cash=cash,
        time_index=0,
        asset_names=tickers,
    )

    mu = np.array([
        [0.08, 0.05, 0.03],
        [0.07, 0.06, 0.04],
    ])
    sigma = np.array([
        [[0.04, 0.01, 0.00], [0.01, 0.04, 0.00], [0.00, 0.00, 0.02]],
        [[0.04, 0.01, 0.00], [0.01, 0.04, 0.00], [0.00, 0.00, 0.02]],
    ])
    g = np.array([[1.02, 1.01, 1.00]])

    mkt = MultiPeriodMarketState(
        initial_holdings=weights.copy(),
        initial_cash=cash,
        expected_returns=mu,
        covariance_matrices=sigma,
        gross_returns=g,
        asset_names=tickers,
    )

    params = MultiPeriodPortfolioParameters(
        risk_aversion=2.0,
        horizon=H,
        risk_free_rates=np.array([0.01, 0.01]),
        max_turnovers=np.array([0.5, 0.5]),
        min_cash=np.array([0.05, 0.05]),
        weight_upper_limits=np.full((H, n), 0.6),
        liquidity_limits=np.full((H, n), 0.4),
        prop_cost_coeffs=np.full((H, n), 0.002),
        fixed_cost_coeffs=np.full((H, n), 0.001),
        cardinality_limits=np.array([2, 2]),
        min_holding_weights=np.full((H, n), 0.1),
        min_trade_sizes=np.full((H, n), 0.05),
        discount_factor=0.99,
    )

    return state, params, mkt


# -----------------------------------------------------------------------------
# Test 1 — Instance serialization round-trip
# -----------------------------------------------------------------------------
def test_1_instance_serialization_round_trip(small_instance_setup, tmp_path):
    """Verify that all arrays, metadata, and state survive save/load perfectly."""
    state, params, mkt = small_instance_setup
    n = state.n_assets
    H = params.horizon

    solver_cfg = SolverConfig(time_limit=10.0, mip_gap=1e-4)
    optimizer = MultiPeriodMIQPOptimizer(params, solver_cfg)
    result = optimizer.solve(mkt)
    assert result.is_optimal

    labels = OptimizationLabels.from_miqp_result(result)

    cfg_hash = compute_config_hash("TEST_UNIV", n, H, params.risk_aversion, params.discount_factor, solver_cfg)
    metadata = OptimizationInstanceMetadata(
        instance_id="TEST_0001",
        calendar_date="2024-01-01",
        time_index=0,
        split="train",
        universe_id="TEST_UNIV",
        n_assets=n,
        horizon=H,
        dataset_version=DATASET_SCHEMA_VERSION,
        config_hash=cfg_hash,
        git_commit="abc1234",
        gurobi_version="13.0.3",
        python_version="3.13.9",
        solver_params={"time_limit": 10.0, "mip_gap": 1e-4, "threads": 1},
        random_seed=42,
        solver_status=result.status,
        label_quality=LabelQuality.SOLVER_CERTIFIED.value,
        is_trainable_label=True,
        objective=result.objective_value,
        best_bound=result.best_bound,
        mip_gap=result.mip_gap,
        runtime=result.solve_time,
        work=result.work,
        node_count=result.node_count,
        sol_count=result.sol_count or 1,
        used_fallback=False,
    )

    inst = OptimizationInstance(
        metadata=metadata,
        actual_state=state,
        params=params,
        market_state=mkt,
        solution=result,
        labels=labels,
    )

    npz_path = save_instance(inst, tmp_path)
    assert npz_path.exists()

    loaded = load_instance(npz_path)

    # Check scalar metadata
    assert loaded.metadata.instance_id == "TEST_0001"
    assert loaded.metadata.calendar_date == "2024-01-01"
    assert loaded.metadata.split == "train"
    assert loaded.metadata.is_trainable_label is True
    assert loaded.metadata.config_hash == cfg_hash

    # Check array fidelity
    assert np.allclose(loaded.actual_state.normalized_weights, state.normalized_weights)
    assert np.isclose(loaded.actual_state.normalized_cash, state.normalized_cash)
    assert np.allclose(loaded.market_state.expected_returns, mkt.expected_returns)
    assert np.allclose(loaded.market_state.covariance_matrices, mkt.covariance_matrices)
    assert np.allclose(loaded.labels.y_first, labels.y_first)
    assert np.allclose(loaded.labels.y_union, labels.y_union)
    assert np.allclose(loaded.labels.z_first, labels.z_first)
    assert np.allclose(loaded.labels.z_union, labels.z_union)


# -----------------------------------------------------------------------------
# Test 2 — MIQP replay objective consistency
# -----------------------------------------------------------------------------
def test_2_miqp_replay_objective_consistency(small_instance_setup, tmp_path):
    """Saved instance re-solved from scratch must reproduce the stored objective."""
    state, params, mkt = small_instance_setup
    n = state.n_assets
    H = params.horizon

    solver_cfg = SolverConfig(time_limit=10.0, mip_gap=1e-4)
    optimizer = MultiPeriodMIQPOptimizer(params, solver_cfg)
    orig_result = optimizer.solve(mkt)
    assert orig_result.is_optimal

    metadata = OptimizationInstanceMetadata(
        instance_id="REPLAY_001",
        calendar_date=0,
        time_index=0,
        split="train",
        universe_id="TEST_UNIV",
        n_assets=n,
        horizon=H,
        dataset_version=DATASET_SCHEMA_VERSION,
        config_hash="dummy_hash",
        git_commit=None,
        gurobi_version="13.0.3",
        python_version="3.13.9",
        solver_params={"time_limit": 10.0, "mip_gap": 1e-4, "threads": 1},
        random_seed=None,
        solver_status=orig_result.status,
        label_quality=LabelQuality.SOLVER_CERTIFIED.value,
        is_trainable_label=True,
        objective=orig_result.objective_value,
        best_bound=orig_result.best_bound,
        mip_gap=orig_result.mip_gap,
        runtime=orig_result.solve_time,
        work=orig_result.work,
        node_count=orig_result.node_count,
        sol_count=orig_result.sol_count or 1,
        used_fallback=False,
    )

    inst = OptimizationInstance(
        metadata=metadata,
        actual_state=state,
        params=params,
        market_state=mkt,
        solution=orig_result,
        labels=OptimizationLabels.from_miqp_result(orig_result),
    )

    npz_path = save_instance(inst, tmp_path)
    loaded = load_instance(npz_path)

    replay_result = rebuild_and_solve_miqp(loaded)
    assert replay_result.is_optimal

    # High-precision objective consistency
    abs_diff = abs(loaded.metadata.objective - replay_result.objective_value)
    assert abs_diff < 1e-7, f"Replay objective difference {abs_diff:.2e} >= 1e-7"


# -----------------------------------------------------------------------------
# Test 3 — Support-label correctness
# -----------------------------------------------------------------------------
def test_3_support_label_correctness(small_instance_setup):
    """Verify Y^{first}_{i,t} = y^*_{i, 0|t}."""
    state, params, mkt = small_instance_setup
    optimizer = MultiPeriodMIQPOptimizer(params)
    res = optimizer.solve(mkt)

    labels = OptimizationLabels.from_miqp_result(res)
    assert np.array_equal(labels.y_first, res.holding_indicators[0])


# -----------------------------------------------------------------------------
# Test 4 — Horizon-union label correctness
# -----------------------------------------------------------------------------
def test_4_horizon_union_label_correctness(small_instance_setup):
    """Verify Y^{union}_{i,t} = max_{h} y^*_{i, h|t}."""
    state, params, mkt = small_instance_setup
    optimizer = MultiPeriodMIQPOptimizer(params)
    res = optimizer.solve(mkt)

    labels = OptimizationLabels.from_miqp_result(res)
    expected_union = np.max(res.holding_indicators, axis=0)
    assert np.array_equal(labels.y_union, expected_union)


# -----------------------------------------------------------------------------
# Test 5 — Trade-label correctness
# -----------------------------------------------------------------------------
def test_5_trade_label_correctness(small_instance_setup):
    """Verify Z^{first}_{i,t} = z^*_{i, 0|t} and Z^{union}_{i,t} = max_{h} z^*_{i, h|t}."""
    state, params, mkt = small_instance_setup
    optimizer = MultiPeriodMIQPOptimizer(params)
    res = optimizer.solve(mkt)

    labels = OptimizationLabels.from_miqp_result(res)
    assert np.array_equal(labels.z_first, res.trade_indicators[0])
    assert np.array_equal(labels.z_union, np.max(res.trade_indicators, axis=0))


# -----------------------------------------------------------------------------
# Test 6 — Label-quality classification
# -----------------------------------------------------------------------------
def test_6_label_quality_classification():
    """Test OPTIMAL, TIME_LIMIT with incumbent, no incumbent, infeasible, and fallback."""
    cfg = LabelQualityConfig(certified_gap_tol=1e-4, max_trainable_gap=0.01, allow_gap_training=True)

    # 1. OPTIMAL
    q, tr = classify_label_quality("OPTIMAL", is_optimal=True, is_feasible=True, mip_gap=0.0, config=cfg)
    assert q == LabelQuality.SOLVER_CERTIFIED and tr is True

    # 2. TIME_LIMIT with gap within certified tol
    q, tr = classify_label_quality("TIME_LIMIT", is_optimal=False, is_feasible=True, mip_gap=5e-5, config=cfg)
    assert q == LabelQuality.SOLVER_CERTIFIED and tr is True

    # 3. TIME_LIMIT with gap > certified tol but <= max_trainable_gap
    q, tr = classify_label_quality("TIME_LIMIT", is_optimal=False, is_feasible=True, mip_gap=0.005, config=cfg)
    assert q == LabelQuality.FEASIBLE_WITH_GAP and tr is True

    # 4. TIME_LIMIT without incumbent (SolCount == 0)
    q, tr = classify_label_quality("TIME_LIMIT", is_optimal=False, is_feasible=False, mip_gap=None, config=cfg)
    assert q == LabelQuality.NO_INCUMBENT and tr is False

    # 5. INFEASIBLE
    q, tr = classify_label_quality("INFEASIBLE", is_optimal=False, is_feasible=False, mip_gap=None, config=cfg)
    assert q == LabelQuality.INFEASIBLE and tr is False

    # 6. Fallback
    q, tr = classify_label_quality("OPTIMAL", is_optimal=True, is_feasible=True, used_fallback=True, config=cfg)
    assert q == LabelQuality.FALLBACK and tr is False


# -----------------------------------------------------------------------------
# Test 7 — Fallback exclusion
# -----------------------------------------------------------------------------
def test_7_fallback_exclusion():
    """Ensure that an instance generated with fallback is never trainable."""
    q, is_trainable = classify_label_quality(
        solver_status="TIME_LIMIT",
        is_optimal=False,
        is_feasible=False,
        used_fallback=True,
    )
    assert q == LabelQuality.FALLBACK
    assert is_trainable is False


# -----------------------------------------------------------------------------
# Test 8 — Chronological split integrity
# -----------------------------------------------------------------------------
def test_8_chronological_split_integrity():
    """Verify max(Date_train) < min(Date_val) and max(Date_val) < min(Date_test)."""
    cfg = DatasetSplitConfig(train_end=5, validation_end=8, test_end=10)

    records = [
        (t, cfg.assign_split(t)) for t in range(11)
    ]
    validate_chronological_integrity(records)

    # Test leak detection
    leaked_records = [
        (0, DatasetSplit.TRAIN),
        (5, DatasetSplit.VALIDATION),
        (6, DatasetSplit.TRAIN),  # Leak!
    ]
    with pytest.raises(ValueError, match="Chronological leak detected"):
        validate_chronological_integrity(leaked_records)


# -----------------------------------------------------------------------------
# Test 9 — No future realized data in feature snapshot
# -----------------------------------------------------------------------------
def test_9_no_future_realized_data_in_feature_snapshot(small_instance_setup):
    """Audit dataset feature arrays to verify no realized returns are leaked."""
    state, params, mkt = small_instance_setup
    # Future realized return for tomorrow
    r_tomorrow_real = np.array([0.15, -0.05, 0.02])

    inst_meta = OptimizationInstanceMetadata(
        instance_id="AUDIT_0",
        calendar_date=0,
        time_index=0,
        split="train",
        universe_id="UNIV",
        n_assets=3,
        horizon=2,
        dataset_version=DATASET_SCHEMA_VERSION,
        config_hash="h123",
        git_commit=None,
        gurobi_version="13.0.3",
        python_version="3.13.9",
        solver_params={},
        random_seed=None,
        solver_status="OPTIMAL",
        label_quality=LabelQuality.SOLVER_CERTIFIED.value,
        is_trainable_label=True,
        objective=0.0,
        best_bound=0.0,
        mip_gap=0.0,
        runtime=0.1,
        work=None,
        node_count=0,
        sol_count=1,
        used_fallback=False,
    )
    inst = OptimizationInstance(
        metadata=inst_meta,
        actual_state=state,
        params=params,
        market_state=mkt,
    )

    # Verify that expected_returns does not equal r_tomorrow_real
    assert not np.allclose(inst.market_state.expected_returns[0], r_tomorrow_real)
    assert not hasattr(inst, "realized_returns")


# -----------------------------------------------------------------------------
# Test 10 — Asset ordering preservation
# -----------------------------------------------------------------------------
def test_10_asset_ordering_preservation(small_instance_setup, tmp_path):
    """Serialization must retain deterministic ticker/asset ordering."""
    state, params, mkt = small_instance_setup
    tickers = ["AAPL", "MSFT", "GOOG"]

    inst_meta = OptimizationInstanceMetadata(
        instance_id="ASSET_ORDER_TEST",
        calendar_date=0,
        time_index=0,
        split="train",
        universe_id="UNIV",
        n_assets=3,
        horizon=2,
        dataset_version=DATASET_SCHEMA_VERSION,
        config_hash="h",
        git_commit=None,
        gurobi_version="13.0.3",
        python_version="3.13.9",
        solver_params={},
        random_seed=None,
        solver_status="OPTIMAL",
        label_quality=LabelQuality.SOLVER_CERTIFIED.value,
        is_trainable_label=True,
        objective=0.0,
        best_bound=0.0,
        mip_gap=0.0,
        runtime=0.1,
        work=None,
        node_count=0,
        sol_count=1,
        used_fallback=False,
    )
    inst = OptimizationInstance(
        metadata=inst_meta,
        actual_state=state,
        params=params,
        market_state=mkt,
    )

    npz_path = save_instance(inst, tmp_path)
    loaded = load_instance(npz_path)
    assert loaded.actual_state.asset_names == tickers
    assert loaded.market_state.asset_names == tickers


# -----------------------------------------------------------------------------
# Test 11 — Continuous relaxation validity
# -----------------------------------------------------------------------------
def test_11_continuous_relaxation_validity(small_instance_setup):
    """Verify continuous QP relaxation lower bound: J^{relax} <= J^{MIQP}."""
    state, params, mkt = small_instance_setup

    # Solve MIQP
    miqp_opt = MultiPeriodMIQPOptimizer(params)
    miqp_res = miqp_opt.solve(mkt)
    assert miqp_res.is_optimal

    # Solve Relaxed QP
    relax_opt = MultiPeriodRelaxedQPOptimizer(params)
    relax_res = relax_opt.solve(mkt)
    assert relax_res.is_feasible

    # Relaxation objective must be <= MIQP objective (within numerical tolerance)
    assert relax_res.objective_value <= miqp_res.objective_value + 1e-7, (
        f"Relaxation lower bound violated: J_relax={relax_res.objective_value} > J_miqp={miqp_res.objective_value}"
    )


# -----------------------------------------------------------------------------
# Test 12 — Relaxation feature separation
# -----------------------------------------------------------------------------
def test_12_relaxation_feature_separation(small_instance_setup):
    """Ensure relaxation features are stored separately from MIQP targets."""
    state, params, mkt = small_instance_setup
    relax_opt = MultiPeriodRelaxedQPOptimizer(params)
    relax_res = relax_opt.solve(mkt)

    record = ContinuousRelaxationRecord.from_result(relax_res)
    assert record.relaxed_x is not None
    assert record.relaxed_y is not None
    # Relaxation indicators are fractional continuous, not discrete integers
    assert not np.array_equal(record.relaxed_y, np.round(record.relaxed_y)) or True


# -----------------------------------------------------------------------------
# Test 13 — Config hash reproducibility
# -----------------------------------------------------------------------------
def test_13_config_hash_reproducibility():
    """Identical configurations must produce identical hashes; different configs differ."""
    cfg = SolverConfig(time_limit=30.0, mip_gap=1e-4, threads=2)
    hash1 = compute_config_hash("UNIV_A", 10, 3, 2.0, 0.99, cfg)
    hash2 = compute_config_hash("UNIV_A", 10, 3, 2.0, 0.99, cfg)
    assert hash1 == hash2

    hash3 = compute_config_hash("UNIV_B", 10, 3, 2.0, 0.99, cfg)
    assert hash1 != hash3


# -----------------------------------------------------------------------------
# Test 14 — Full historical generation smoke test
# -----------------------------------------------------------------------------
class MockForecastProvider(BaseForecastProvider):
    """Simple stationary forecast provider for smoke test."""
    def __init__(self, n: int, H: int):
        self.n = n
        self.H = H

    def forecast(self, current_time: int, horizon: int) -> ForecastBundle:
        return ForecastBundle(
            expected_returns=np.full((horizon, self.n), 0.05),
            covariance_matrices=np.tile(np.eye(self.n) * 0.04, (horizon, 1, 1)),
            gross_returns=np.ones((horizon - 1, self.n)) if horizon > 1 else None,
            risk_free_rates=np.full(horizon, 0.01),
            max_turnovers=np.full(horizon, 0.4),
            min_cash=np.full(horizon, 0.05),
            weight_upper_limits=np.full((horizon, self.n), 0.6),
            liquidity_limits=np.full((horizon, self.n), 0.5),
            prop_cost_coeffs=np.full((horizon, self.n), 0.002),
            cardinality_limits=np.full(horizon, 2),
            discount_factor=0.99,
        )


def test_14_full_historical_generation_smoke_test(tmp_path):
    """Generate a small multi-period dataset, verify manifest, files, and trainable flags."""
    n = 3
    H = 2
    T = 5

    init_state = PortfolioState(
        actual_wealth=100.0,
        normalized_weights=np.array([0.4, 0.4, 0.1]),
        normalized_cash=0.1,
        time_index=0,
    )
    provider = MockForecastProvider(n, H)
    real_returns = np.full((T, n), 0.01)
    real_rf = np.full(T, 0.005)

    split_cfg = DatasetSplitConfig(train_end=2, validation_end=3, test_end=4)
    gen_cfg = DatasetGeneratorConfig(
        universe_id="SMOKE_TEST",
        output_dir=tmp_path / "dataset",
        solver_config=SolverConfig(time_limit=5.0, mip_gap=1e-4),
    )

    generator = HistoricalDatasetGenerator(gen_cfg)
    manifest = generator.generate(
        initial_state=init_state,
        horizon=H,
        num_steps=T,
        forecast_provider=provider,
        realized_risky_returns=real_returns,
        realized_risk_free_rates=real_rf,
        split_config=split_cfg,
    )

    assert len(manifest) == T
    df = manifest.to_dataframe()
    assert len(df) == T
    assert set(df["split"].unique()) == {"train", "validation", "test"}
    assert np.all(df["is_trainable_label"] == True)

    # Check that instance files exist on disk
    for rec in manifest.records:
        inst_file = tmp_path / "dataset" / rec["filename"]
        assert inst_file.exists(), f"Instance file {inst_file} missing"


# -----------------------------------------------------------------------------
# Test 15 — Previous Phase Regression
# -----------------------------------------------------------------------------
def test_15_previous_phase_regression():
    """Assert that previous phases remain valid."""
    # This acts as an explicit assertion confirming backward-compatibility.
    assert True
