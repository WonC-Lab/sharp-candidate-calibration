import pandas as pd
import numpy as np

# Load original
df = pd.read_csv("artifacts/final_tables/final_table_1_financial_performance.csv")
v0 = 1_000_000.0
rf = 0.02

# We update with clear canonical columns:
# Strategy, TotalReturn, CAGR, AnnArithmetic, AnnVol, Sharpe, MaxDD, Turnover, Costs($), Holdings, Cash, FinalWealth($)
updated_rows = []
for _, r in df.iterrows():
    strat = r["Strategy"]
    fw_str = r["FinalWealth($)"].replace("$", "").replace(",", "")
    vt = float(fw_str)
    tot_ret = vt / v0 - 1.0
    cagr_exact = (vt / v0) ** (52.0 / 51.0) - 1.0
    ann_arith = float(r["Ann.Return"].replace("%", "")) / 100.0
    ann_vol = float(r["Ann.Vol"].replace("%", "")) / 100.0
    sharpe_cagr = (cagr_exact - rf) / ann_vol
    
    updated_rows.append({
        "Strategy": strat,
        "TotalReturn": f"{tot_ret * 100:.2f}%",
        "CAGR": f"{cagr_exact * 100:.2f}%",
        "Ann.Arithmetic": f"{ann_arith * 100:.2f}%",
        "Ann.Vol": r["Ann.Vol"],
        "Sharpe": f"{sharpe_cagr:.2f}",
        "MaxDD": r["MaxDD"],
        "Turnover": r["Turnover"],
        "Costs($)": r["Costs($)"],
        "Holdings": r["Holdings"],
        "Cash": r["Cash"],
        "FinalWealth($)": r["FinalWealth($)"]
    })

out_df = pd.DataFrame(updated_rows)
out_df.to_csv("artifacts/final_tables/final_table_1_financial_performance.csv", index=False)
out_df.to_csv("artifacts/real_backtest/financial_metrics.csv", index=False)
print("Successfully updated final_table_1 and financial_metrics.csv with exact CAGR and Total Return!")
