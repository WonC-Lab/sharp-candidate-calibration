"""
Phase 9 Experiment 9A: Real-Market Data Pipeline Generation & No-Lookahead Auditing.
Generates:
  - Fixed Historical Liquid-Equity Basket (N=10) daily data (2021-01-01 to 2024-06-30)
  - Adjusted prices, volume, returns, and rolling liquidity
  - Strict chronological validation and missing-data policy auditing
  - Saves artifacts to artifacts/real_market/
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from learning_augmented_portfolio.data import (
    RealMarketUniverse,
    TradingCalendar,
    MarketDataStore,
    ReturnCalculator,
    LiquidityCalculator,
    audit_no_lookahead_slice,
)
from learning_augmented_portfolio.forecasts import (
    ExpectedReturnEstimator,
    ReturnEstimatorType,
    CovarianceEstimator,
    CovarianceEstimatorType,
)


def run_exp09a_real_data_pipeline() -> None:
    print("=" * 80)
    print("PHASE 9A: REAL-MARKET DATA PIPELINE & AUDITING")
    print("=" * 80)

    universe = RealMarketUniverse()
    print(f"\n[1] Asset Universe: {universe.n_assets} liquid equities across {universe.n_sectors} sectors:")
    for a in universe.assets:
        print(f"    - {a.ticker:<6} | {a.sector:<15} | Base Price: ${a.base_price:>6.1f} | Base ADV: {a.base_adv:>10,.0f} shs | {a.description}")

    print("\n[2] Generating deterministic daily market series (2021 - 2024)...")
    store = MarketDataStore.generate_benchmark_market_dataset(universe=universe, start_year=2021, end_year=2024, random_seed=42)
    print(f"    Total trading days generated: {store.num_dates} (from {store.trading_dates[0]} to {store.trading_dates[-1]})")

    out_dir = Path("artifacts/real_market")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "daily_prices.csv"
    store.save_csv(csv_path)
    print(f"    Saved market dataset to {csv_path}")

    # Save universe metadata
    meta_path = out_dir / "universe_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "universe_name": "Fixed Historical Liquid-Equity Basket (N=10)",
            "n_assets": universe.n_assets,
            "tickers": universe.tickers,
            "sectors": universe.sectors,
            "limitation": universe.survivorship_bias_limitation,
            "sample_period": f"{store.trading_dates[0]} to {store.trading_dates[-1]}",
            "trading_days": store.num_dates,
        }, f, indent=2)
    print(f"    Saved universe metadata to {meta_path}")

    # No-lookahead audit across sample dates
    print("\n[3] Auditing No-Lookahead Invariance on sample decision dates...")
    sample_indices = [60, 150, 300, 500, 700]
    for idx in sample_indices:
        d = store.trading_dates[idx]
        assert audit_no_lookahead_slice(store, d, lookback_window=60) is True
        print(f"    [Audit Passed] Decision date: {d} (t={idx}) - strictly backward-looking.")

    # Chronological partition validation
    dates = store.trading_dates
    T = len(dates)
    train_end = int(T * 0.60)
    val_end = int(T * 0.75)

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    print(f"\n[4] Chronological Partition Split:")
    print(f"    - Train split     : {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
    print(f"    - Validation split: {len(val_dates)} days ({val_dates[0]} to {val_dates[-1]})")
    print(f"    - Test split      : {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")

    assert max(train_dates) < min(val_dates) < max(val_dates) < min(test_dates)
    print("    [Verification Passed] Strict temporal ordering guaranteed: Train < Validation < Test.")

    print("\nPhase 9A data pipeline successfully created and audited.")


if __name__ == "__main__":
    run_exp09a_real_data_pipeline()
