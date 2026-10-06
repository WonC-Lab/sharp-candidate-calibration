import pandas as pd
import numpy as np

# Load original financial metrics
df = pd.read_csv("artifacts/final_tables/final_table_1_financial_performance.csv")
print("=== ORIGINAL TABLE ===")
print(df[["Strategy", "Ann.Return", "Ann.Vol", "Sharpe", "FinalWealth($)"]])

# Duration: 51 steps (weekly) in 2024. Exact duration in years: 51 / 52 = 0.980769 years
v0 = 1_000_000.0
rf = 0.02

rows = []
for idx, r in df.iterrows():
    strat = r["Strategy"]
    fw_str = r["FinalWealth($)"].replace("$", "").replace(",", "")
    vt = float(fw_str)
    tot_ret = vt / v0 - 1.0
    cagr_exact = (vt / v0) ** (52.0 / 51.0) - 1.0
    ann_arith = float(r["Ann.Return"].replace("%", "")) / 100.0
    ann_vol = float(r["Ann.Vol"].replace("%", "")) / 100.0
    sharpe_cagr = (cagr_exact - rf) / ann_vol
    rows.append({
        "Strategy": strat,
        "FinalWealth": vt,
        "TotalReturn": tot_ret,
        "CAGR": cagr_exact,
        "AnnArithmetic": ann_arith,
        "AnnVol": ann_vol,
        "Sharpe_CAGR": sharpe_cagr,
        "Sharpe_Orig": float(r["Sharpe"])
    })

res_df = pd.DataFrame(rows)
print("\n=== RECONSTRUCTED METRICS ===")
for _, r in res_df.iterrows():
    print(f"{r['Strategy']:<26}: Final=${r['FinalWealth']:,.0f} | TotRet={r['TotalReturn']*100:.2f}% | CAGR={r['CAGR']*100:.2f}% | AnnArith={r['AnnArithmetic']*100:.2f}% | Vol={r['AnnVol']*100:.2f}% | Sharpe(CAGR)={r['Sharpe_CAGR']:.2f} | OrigSharpe={r['Sharpe_Orig']:.2f}")
