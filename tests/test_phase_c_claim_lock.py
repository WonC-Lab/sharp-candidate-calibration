"""Static Phase C claim checks against locked audit artifacts; no solves are run."""
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MASTER = (ROOT / "manuscript" / "master.tex").read_text(encoding="utf-8")
DATA = ROOT / "artifacts" / "k20_replication"


def test_locked_supplement_grid():
    k20 = pd.read_csv(DATA / "k20_summary.csv")
    b5 = pd.read_csv(DATA / "b5" / "cell_summary.csv")
    lines = re.findall(r"^(8|20|30|50)&\.(20|10|05)&(.+?)\\\\$", MASTER, re.MULTILINE)
    assert len(lines) == 12
    for ns, als, rest in lines:
        n, alpha = int(ns), int(als) / 100
        vals = rest.split("&")
        assert len(vals) == 9
        calibration = k20[(k20.N == n) & (k20.alpha.round(2) == alpha)].iloc[0]
        assert abs(float(vals[0]) - calibration.mean_initial_B_over_N) < 0.0006
        assert abs(float(vals[1].replace("\\%", "")) / 100 - calibration.InitialFullRate) < 1e-9
        assert abs(float(vals[2].replace("\\%", "")) / 100 - calibration.RecoveryToFullRate) < 1e-9
        for offset, pipeline in enumerate(("A_COLD", "B_CAND_LP_COLD", "C_CAND_LP_WARM", "D_CAND_WARM"), start=3):
            row = b5[(b5.N == n) & (b5.alpha.round(2) == alpha) & (b5.pipeline == pipeline)].iloc[0]
            assert abs(float(vals[offset]) - row.mean_total) < 0.00006
        direct = b5[(b5.N == n) & (b5.alpha.round(2) == alpha) & (b5.pipeline == "D_CAND_WARM")].iloc[0]
        assert abs(float(vals[7]) - direct.ratio_of_means_speedup) < 0.0006
        ci = [float(x) for x in vals[8].strip("[]").split(",")]
        assert abs(ci[0] - direct.speedup_ci_low) < 0.0006
        assert abs(ci[1] - direct.speedup_ci_high) < 0.0006


def test_locked_exact_oracle_and_no_legacy_primary_claims():
    oracle = pd.read_csv(DATA / "exact_oracle_summary.csv")
    for alpha, printed in ((0.20, "84.83"), (0.10, "96.50"), (0.05, "98.83")):
        rate = oracle.loc[oracle.alpha.round(2) == alpha, "mean_exact_coverage"].iloc[0] * 100
        assert abs(rate - float(printed)) < 0.006
        assert printed in MASTER
    for stale in ("1.180", "1.082", "0.852", "1.49"):
        assert stale not in MASTER
