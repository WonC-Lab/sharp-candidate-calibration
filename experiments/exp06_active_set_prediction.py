"""
Phase 6 Experiment: Active-Set / Support Prediction via Machine Learning.
Evaluates ML models and heuristic/optimization baselines on candidate universe reduction,
computes Recall@B, Coverage@B, FNR@B, runs feature ablations, and generates diagnostic plots.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from learning_augmented_portfolio.dataset.generator import (
    HistoricalDatasetGenerator,
    DatasetGeneratorConfig,
)
from learning_augmented_portfolio.dataset.instance import OptimizationInstance
from learning_augmented_portfolio.dataset.serialization import load_instance
from learning_augmented_portfolio.dataset.split import DatasetSplitConfig
from learning_augmented_portfolio.dynamic.state import PortfolioState
from learning_augmented_portfolio.forecasts.base import BaseForecastProvider, ForecastBundle
from learning_augmented_portfolio.optimization.variables import SolverConfig
from learning_augmented_portfolio.learning.active_set import (
    build_logistic_regression_predictor,
    build_hist_gradient_boosting_predictor,
    build_random_forest_predictor,
    build_mlp_predictor,
)
from learning_augmented_portfolio.learning.baselines import (
    RandomRankingBaseline,
    CurrentHoldingBaseline,
    ExpectedReturnBaseline,
    ContinuousRelaxationYBaseline,
)
from learning_augmented_portfolio.learning.dataset import (
    SupervisedSplitDataset,
    build_supervised_datasets_by_split,
)
from learning_augmented_portfolio.learning.evaluation import (
    evaluate_ranker_across_budgets,
    results_to_dataframe,
)
from learning_augmented_portfolio.learning.features import (
    FeatureGroup,
    ALL_FEATURE_GROUPS,
    MARKET_ONLY_GROUPS,
    RELAXATION_ONLY_GROUPS,
    MARKET_AND_CONSTRAINTS_GROUPS,
)


class IntermediateMarketProvider(BaseForecastProvider):
    """
    Synthetic market provider for n=10 assets across H=3 planning horizon.
    Generates time-varying alpha drifts, factor covariance structure, and realistic constraints.
    """

    def __init__(self, n: int = 10, H: int = 3, seed: int = 42):
        self.n = n
        self.H = H
        self.rng = np.random.default_rng(seed)
        # Heterogeneous base returns: top assets have higher expected alpha
        self.base_mu = np.linspace(0.04, 0.10, n)
        # 2-factor covariance structure + idiosyncratic risk
        F_loadings = self.rng.normal(loc=0.5, scale=0.2, size=(n, 2))
        F_cov = np.array([[0.04, 0.01], [0.01, 0.03]])
        idio = np.diag(self.rng.uniform(0.01, 0.025, size=n))
        self.base_sigma = F_loadings @ F_cov @ F_loadings.T + idio

    def forecast(self, current_time: int, horizon: int) -> ForecastBundle:
        n = self.n
        # Cyclical sector rotation drift
        phase = 2 * np.pi * current_time / 40.0
        drift = 0.025 * np.sin(phase + np.linspace(0, 2 * np.pi, n))

        expected_returns = np.zeros((horizon, n))
        for h in range(horizon):
            expected_returns[h] = self.base_mu + drift * (0.95 ** h)

        covariance_matrices = np.tile(self.base_sigma, (horizon, 1, 1))
        gross_returns = 1.0 + expected_returns[:-1] if horizon > 1 else None

        return ForecastBundle(
            expected_returns=expected_returns,
            covariance_matrices=covariance_matrices,
            gross_returns=gross_returns,
            risk_free_rates=np.full(horizon, 0.01),
            max_turnovers=np.full(horizon, 0.60),
            min_cash=np.full(horizon, 0.05),
            weight_upper_limits=np.full((horizon, n), 0.40),
            liquidity_limits=np.full((horizon, n), 0.30),
            prop_cost_coeffs=np.full((horizon, n), 0.0010),
            fixed_cost_coeffs=np.full((horizon, n), 0.0005),
            cardinality_limits=np.full(horizon, 3),    # K = 3 out of 10 assets
            min_holding_weights=np.full((horizon, n), 0.04),
            min_trade_sizes=np.full((horizon, n), 0.02),
            discount_factor=0.98,
        )


def load_instances_from_manifest(manifest_csv_path: Path) -> list[OptimizationInstance]:
    """Load all OptimizationInstances referenced in a manifest CSV."""
    df = pd.read_csv(manifest_csv_path)
    base_dir = manifest_csv_path.parent
    instances = []
    for _, row in df.iterrows():
        filepath = base_dir / row["filename"]
        inst = load_instance(filepath)
        instances.append(inst)
    return instances


def run_phase6_experiment():
    print("=" * 80)
    print("PHASE 6 VALIDATION EXPERIMENT: ACTIVE-SET / SUPPORT PREDICTION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # [1] Dataset Setup: Generate Intermediate Scale Teacher Dataset
    # -------------------------------------------------------------------------
    n_assets = 10
    horizon = 3
    num_steps = 60
    cardinality_K = 3
    tickers = [f"ASSET_{i:02d}" for i in range(n_assets)]

    print(f"\n[1] Intermediate Dataset Configuration:")
    print(f"    - Universe Size N: {n_assets}")
    print(f"    - Planning Horizon H: {horizon}")
    print(f"    - Decision Steps T: {num_steps}")
    print(f"    - Portfolio Cardinality K: {cardinality_K} (forcing combinatorial support selection)")
    print(f"    - Target Variable: Y^union_i = max_h y^*_ih in {{0, 1}}")

    dataset_dir = Path("artifacts/instances_medium")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = dataset_dir / "manifest.csv"

    if manifest_csv.exists() and len(pd.read_csv(manifest_csv)) == num_steps:
        print(f"    -> Loading existing intermediate dataset from {dataset_dir}...")
        instances = load_instances_from_manifest(manifest_csv)
    else:
        print(f"    -> Generating new intermediate full-MIQP teacher dataset ({num_steps} instances)...")
        w0 = np.zeros(n_assets)
        w0[:cardinality_K] = 0.90 / cardinality_K
        c0 = 0.10
        initial_state = PortfolioState(
            actual_wealth=100_000.0,
            normalized_weights=w0,
            normalized_cash=c0,
            time_index=0,
            asset_names=tickers,
        )

        forecast_provider = IntermediateMarketProvider(n=n_assets, H=horizon, seed=42)
        rng = np.random.default_rng(2026)
        real_ret = rng.normal(0.005, 0.02, size=(num_steps, n_assets))
        real_rf = np.full(num_steps, 0.0005)

        split_config = DatasetSplitConfig.from_proportions(
            total_steps=num_steps,
            train_ratio=0.60,
            val_ratio=0.20,
        )

        gen_config = DatasetGeneratorConfig(
            universe_id="UNIV_N10",
            output_dir=dataset_dir,
            solver_config=SolverConfig(time_limit=15.0, mip_gap=1e-4, threads=1),
            compute_relaxations=True,
            save_json_metadata=False,
        )
        generator = HistoricalDatasetGenerator(gen_config)
        manifest = generator.generate(
            initial_state=initial_state,
            horizon=horizon,
            num_steps=num_steps,
            forecast_provider=forecast_provider,
            realized_risky_returns=real_ret,
            realized_risk_free_rates=real_rf,
            split_config=split_config,
        )
        instances = load_instances_from_manifest(manifest_csv)

    print(f"    -> Successfully loaded {len(instances)} full-MIQP instances.")

    # -------------------------------------------------------------------------
    # [2] Build Chronological Asset-Level Supervised Datasets
    # -------------------------------------------------------------------------
    print("\n[2] Building Chronological Asset-Level Supervised Partitions:")
    datasets = build_supervised_datasets_by_split(instances, feature_groups=ALL_FEATURE_GROUPS)

    ds_train = datasets["train"]
    ds_val = datasets["validation"]
    ds_test = datasets["test"]

    print(f"    - Train:      {ds_train.num_instances} instances, {ds_train.num_samples} asset samples, positive ratio = {ds_train.positive_ratio*100:.1f}%")
    print(f"    - Validation: {ds_val.num_instances} instances, {ds_val.num_samples} asset samples, positive ratio = {ds_val.positive_ratio*100:.1f}%")
    print(f"    - Test:       {ds_test.num_instances} instances, {ds_test.num_samples} asset samples, positive ratio = {ds_test.positive_ratio*100:.1f}%")
    print(f"    - Feature dimension D = {ds_train.X.shape[1]}")

    # -------------------------------------------------------------------------
    # [3] Fit Machine Learning Models (Strictly on Train Split)
    # -------------------------------------------------------------------------
    print("\n[3] Fitting Machine Learning Models on Train Split...")
    models = {
        "Random": RandomRankingBaseline(seed=42),
        "CurrentHolding": CurrentHoldingBaseline(),
        "ExpectedReturn": ExpectedReturnBaseline(),
        "ContinuousRelaxation": ContinuousRelaxationYBaseline(),
        "LogisticRegression": build_logistic_regression_predictor(C=1.0).fit(ds_train.X, ds_train.y_union),
        "HistGradientBoosting": build_hist_gradient_boosting_predictor(max_iter=100).fit(ds_train.X, ds_train.y_union),
        "RandomForest": build_random_forest_predictor(n_estimators=100).fit(ds_train.X, ds_train.y_union),
        "SmallMLP": build_mlp_predictor(hidden_layer_sizes=(64, 32), max_iter=300).fit(ds_train.X, ds_train.y_union),
    }

    # -------------------------------------------------------------------------
    # [4] Validation Set Model Evaluation across Candidate Budgets B
    # -------------------------------------------------------------------------
    budget_grid = list(range(cardinality_K, n_assets + 1))  # B in [3, 4, 5, 6, 7, 8, 9, 10]
    print(f"\n[4] Evaluating Models across Candidate Budgets B in {budget_grid} (Validation Set):")

    val_results = []
    for name, model in models.items():
        # Evaluate raw predictor
        res_raw = evaluate_ranker_across_budgets(model, ds_val, budget_grid, use_safe_augmentation=False, model_name=name)
        val_results.extend(res_raw)
        # Evaluate safe-augmented predictor
        res_safe = evaluate_ranker_across_budgets(model, ds_val, budget_grid, use_safe_augmentation=True, model_name=f"{name}_SafeAug")
        val_results.extend(res_safe)

    df_val = results_to_dataframe(val_results)

    # Print summary table at key budget B=5 (50% reduction) and B=6 (40% reduction)
    print("\n--- Validation Performance at B=5 (50% Universe Reduction) ---")
    val_b5 = df_val[(df_val["B"] == 5) & (~df_val["SafeAugmented"])].sort_values(by="Coverage@B", ascending=False)
    print(val_b5[["Model", "Recall@B", "FNR@B", "Precision@B", "Coverage@B", "MissedCount"]].to_string(index=False))

    print("\n--- Validation Performance at B=6 (40% Universe Reduction) ---")
    val_b6 = df_val[(df_val["B"] == 6) & (~df_val["SafeAugmented"])].sort_values(by="Coverage@B", ascending=False)
    print(val_b6[["Model", "Recall@B", "FNR@B", "Precision@B", "Coverage@B", "MissedCount"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # [5] Feature Ablation Study on Validation Set
    # -------------------------------------------------------------------------
    print("\n[5] Feature Ablation Study (HistGradientBoosting on Validation Set):")
    ablations = {
        "F1_MarketOnly": MARKET_ONLY_GROUPS,
        "F2_RelaxationOnly": RELAXATION_ONLY_GROUPS,
        "F3_MarketAndConstraints": MARKET_AND_CONSTRAINTS_GROUPS,
        "F4_AllFeatures": ALL_FEATURE_GROUPS,
    }
    ablation_results = []

    for abl_name, groups in ablations.items():
        ds_abl = build_supervised_datasets_by_split(instances, feature_groups=groups)
        m_abl = build_hist_gradient_boosting_predictor(feature_groups=groups, max_iter=100)
        m_abl.fit(ds_abl["train"].X, ds_abl["train"].y_union)
        res = evaluate_ranker_across_budgets(m_abl, ds_abl["validation"], budget_grid, model_name=abl_name)
        ablation_results.extend(res)

    df_ablation = results_to_dataframe(ablation_results)
    df_abl_b5 = df_ablation[df_ablation["B"] == 5]
    print(df_abl_b5[["Model", "CandidateReduction", "Recall@B", "FNR@B", "Precision@B", "Coverage@B"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # [6] Final Frozen Evaluation on Test Split (Evaluated Once)
    # -------------------------------------------------------------------------
    print("\n[6] Final Frozen Evaluation on Test Split (Unseen Test Data):")
    # Best validation models: HistGradientBoosting and ContinuousRelaxation
    test_models = {
        "Random": models["Random"],
        "CurrentHolding": models["CurrentHolding"],
        "ExpectedReturn": models["ExpectedReturn"],
        "ContinuousRelaxation": models["ContinuousRelaxation"],
        "LogisticRegression": models["LogisticRegression"],
        "HistGradientBoosting": models["HistGradientBoosting"],
        "HistGradientBoosting_SafeAug": models["HistGradientBoosting"],
    }

    test_results = []
    for name, model in test_models.items():
        is_safe = ("SafeAug" in name)
        res = evaluate_ranker_across_budgets(model, ds_test, budget_grid, use_safe_augmentation=is_safe, model_name=name)
        test_results.extend(res)

    df_test = results_to_dataframe(test_results)

    print("\n--- Final Test Performance Table across Budgets B ---")
    for B_val in [3, 4, 5, 6, 7]:
        print(f"\n>>> Budget B = {B_val} (Candidate Reduction = {(1 - B_val/n_assets)*100:.0f}%):")
        df_sub = df_test[df_test["B"] == B_val]
        print(df_sub[["Model", "Recall@B", "FNR@B", "Precision@B", "Coverage@B", "MissedCount"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # [7] Generate and Save Phase 6 Core Plots
    # -------------------------------------------------------------------------
    plots_dir = Path("artifacts/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[7] Generating Diagnostic Visualizations in {plots_dir}...")

    # Color palette
    plot_models = ["Random", "ExpectedReturn", "CurrentHolding", "ContinuousRelaxation", "HistGradientBoosting"]
    colors = {
        "Random": "#999999",
        "ExpectedReturn": "#ff7f0e",
        "CurrentHolding": "#2ca02c",
        "ContinuousRelaxation": "#1f77b4",
        "HistGradientBoosting": "#d62728",
    }

    # Plot 1: Candidate Reduction vs Recall@B
    plt.figure(figsize=(8, 5))
    for m_name in plot_models:
        df_m = df_test[(df_test["Model"] == m_name) & (~df_test["SafeAugmented"])]
        plt.plot(df_m["CandidateReduction"] * 100, df_m["Recall@B"] * 100, marker="o", label=m_name, color=colors[m_name], linewidth=2)
    plt.title("Candidate Universe Reduction vs. Recall@B (Test Split)", fontsize=13, fontweight="bold")
    plt.xlabel("Candidate Universe Reduction (%)", fontsize=11)
    plt.ylabel("Recall@B (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plot1_path = plots_dir / "plot1_reduction_vs_recall.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"    - Saved Plot 1: {plot1_path}")

    # Plot 2: Candidate Reduction vs Full-Support Coverage@B
    plt.figure(figsize=(8, 5))
    for m_name in plot_models:
        df_m = df_test[(df_test["Model"] == m_name) & (~df_test["SafeAugmented"])]
        plt.plot(df_m["CandidateReduction"] * 100, df_m["Coverage@B"] * 100, marker="s", label=m_name, color=colors[m_name], linewidth=2)
    plt.title("Candidate Universe Reduction vs. Full-Support Coverage@B (Test Split)", fontsize=13, fontweight="bold")
    plt.xlabel("Candidate Universe Reduction (%)", fontsize=11)
    plt.ylabel("Full-Support Coverage@B (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plot2_path = plots_dir / "plot2_reduction_vs_coverage.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"    - Saved Plot 2: {plot2_path}")

    # Plot 3: Candidate Budget B vs False-Negative Rate (FNR@B)
    plt.figure(figsize=(8, 5))
    for m_name in plot_models:
        df_m = df_test[(df_test["Model"] == m_name) & (~df_test["SafeAugmented"])]
        plt.plot(df_m["B"], df_m["FNR@B"] * 100, marker="^", label=m_name, color=colors[m_name], linewidth=2)
    plt.title("Candidate Budget B vs. False-Negative Rate (FNR@B) (Test Split)", fontsize=13, fontweight="bold")
    plt.xlabel("Candidate Budget B (out of N=10 assets)", fontsize=11)
    plt.ylabel("False-Negative Rate FNR@B (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plot3_path = plots_dir / "plot3_budget_vs_fnr.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"    - Saved Plot 3: {plot3_path}")

    print("\n" + "=" * 80)
    print("PHASE 6 VALIDATION EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_phase6_experiment()
