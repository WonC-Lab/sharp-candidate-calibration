"""
Phase 9 Experiment 9C: Real-Market Active-Set Training & Validation Selection.
- Trains support predictors Y^{union} strictly on real-market Train instances (113 instances).
- Evaluates Support Coverage, Recall, and Reduction across candidate budgets B on Validation split (31 instances).
- Selects best model and frozen budget B policy, storing to artifacts/real_models/frozen_policy.json.
"""

from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from learning_augmented_portfolio.dataset.instance import OptimizationInstance
from learning_augmented_portfolio.dataset.manifest import DatasetManifest
from learning_augmented_portfolio.dataset.serialization import load_instance
from learning_augmented_portfolio.learning.active_set import (
    build_logistic_regression_predictor,
    build_hist_gradient_boosting_predictor,
)
from learning_augmented_portfolio.learning.dataset import build_supervised_datasets_by_split
from learning_augmented_portfolio.learning.features import MARKET_ONLY_GROUPS, ALL_FEATURE_GROUPS
from learning_augmented_portfolio.learning.candidate_set import construct_candidate_universe


def run_exp09c_real_active_set_training() -> None:
    print("=" * 80)
    print("PHASE 9C: REAL-DATA ACTIVE-SET TRAINING & MODEL SELECTION")
    print("=" * 80)

    manifest_path = Path("artifacts/real_instances/manifest.csv")
    manifest_df = pd.read_csv(manifest_path)
    base_dir = manifest_path.parent

    train_rows = manifest_df[manifest_df["split"] == "train"]
    val_rows = manifest_df[manifest_df["split"] == "validation"]

    print(f"\n[1] Loading real-data instances: Train ({len(train_rows)}), Validation ({len(val_rows)})...")
    train_instances = [load_instance(base_dir / row["filename"]) for _, row in train_rows.iterrows()]
    val_instances = [load_instance(base_dir / row["filename"]) for _, row in val_rows.iterrows()]

    # Extract asset-level tabular datasets
    print("\n[2] Building asset-level tabular datasets...")
    train_ds_mo = build_supervised_datasets_by_split(train_instances, feature_groups=MARKET_ONLY_GROUPS)["train"]
    train_ds_ff = build_supervised_datasets_by_split(train_instances, feature_groups=ALL_FEATURE_GROUPS)["train"]

    val_ds_mo = build_supervised_datasets_by_split(val_instances, feature_groups=MARKET_ONLY_GROUPS)["validation"]
    val_ds_ff = build_supervised_datasets_by_split(val_instances, feature_groups=ALL_FEATURE_GROUPS)["validation"]

    print(f"    Train samples: {train_ds_mo.num_samples}, Positive ratio (Y_union=1): {train_ds_mo.positive_ratio:.3f}")
    print(f"    Validation samples: {val_ds_mo.num_samples}, Positive ratio: {val_ds_mo.positive_ratio:.3f}")

    # Train Models
    print("\n[3] Training ML Support Predictors on Train split only...")
    mo_lr = build_logistic_regression_predictor(feature_groups=MARKET_ONLY_GROUPS, random_state=42)
    mo_lr.fit(train_ds_mo.X, train_ds_mo.y_union)

    ff_lr = build_logistic_regression_predictor(feature_groups=ALL_FEATURE_GROUPS, random_state=42)
    ff_lr.fit(train_ds_ff.X, train_ds_ff.y_union)

    mo_hgb = build_hist_gradient_boosting_predictor(feature_groups=MARKET_ONLY_GROUPS, random_state=42)
    mo_hgb.fit(train_ds_mo.X, train_ds_mo.y_union)

    models = {
        "MarketOnly_LogisticRegression": mo_lr,
        "FullFeature_LogisticRegression": ff_lr,
        "MarketOnly_HistGradientBoosting": mo_hgb,
    }

    # Evaluate across budget grid on Validation instances
    print("\n[4] Evaluating Candidate Budget Policy B in {3, 4, 5, 6, 8, 10} on Validation split...")
    budget_grid = [3, 4, 5, 6, 8, 10]
    val_records = []

    for name, model in models.items():
        for B in budget_grid:
            coverages = []
            recalls = []

            for inst in val_instances:
                scores = model.predict_instance_scores(inst)
                try:
                    cand = construct_candidate_universe(
                        scores=scores,
                        budget_B=B,
                        current_weights=inst.actual_state.normalized_weights,
                        use_safe_augmentation=True,
                    )
                    selected_set = set(cand.candidate_indices)
                except ValueError:
                    selected_set = set()
                teacher_support = set(np.where(inst.labels.y_union == 1)[0])

                # Support coverage: 1 if all teacher support assets included
                cov = 1.0 if teacher_support.issubset(selected_set) else 0.0
                coverages.append(cov)

                # Recall
                rec = len(teacher_support.intersection(selected_set)) / max(1, len(teacher_support))
                recalls.append(rec)

            val_records.append({
                "model": name,
                "B": B,
                "reduction": 1.0 - B / 10.0,
                "coverage": float(np.mean(coverages)),
                "recall": float(np.mean(recalls)),
            })

    val_df = pd.DataFrame(val_records)
    print("\n" + "=" * 80)
    print("VALIDATION CANDIDATE SELECTION EVALUATION TABLE")
    print("=" * 80)
    print(val_df.to_string(index=False))

    # Selection Policy: Minimum budget B achieving >= 95% validation coverage
    candidates = val_df[val_df["coverage"] >= 0.95]
    if not candidates.empty:
        best_row = candidates.sort_values(by=["B", "recall"], ascending=[True, False]).iloc[0]
    else:
        best_row = val_df.sort_values(by=["coverage", "recall"], ascending=[False, False]).iloc[0]

    selected_model_name = str(best_row["model"])
    selected_B = int(best_row["B"])

    print(f"\n[5] Selected Frozen Policy on Validation:")
    print(f"    - Model : {selected_model_name}")
    print(f"    - Budget: B = {selected_B} (Candidate Reduction = {1.0 - selected_B/10.0:.1%})")
    print(f"    - Validation Coverage: {best_row['coverage'] * 100:.1f}%, Recall: {best_row['recall'] * 100:.1f}%")

    # Save models and frozen policy
    out_dir = Path("artifacts/real_models")
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(models[selected_model_name], out_dir / "best_active_set_model.joblib")
    with open(out_dir / "frozen_policy.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected_model": selected_model_name,
            "candidate_budget_B": selected_B,
            "reduction": 1.0 - selected_B / 10.0,
            "validation_coverage": float(best_row["coverage"]),
            "validation_recall": float(best_row["recall"]),
            "selection_criterion": "Minimal B with coverage >= 0.95 on real validation split",
        }, f, indent=2)

    print(f"    Saved frozen policy to {out_dir / 'frozen_policy.json'}")


if __name__ == "__main__":
    run_exp09c_real_active_set_training()
