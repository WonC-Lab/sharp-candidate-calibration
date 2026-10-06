"""Export the locked T6.1 Figure 1 as vector PDF without rerunning solvers.

The archived source plotting function and verified saved data are used verbatim.
Only Matplotlib's save destination/format is redirected for this packaging export.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "t6_1_full_prefix_reanalysis"
TARGET = ROOT / "COAP_SUBMISSION" / "figure1.pdf"
sys.path.insert(0, str(ROOT / "experiments"))
import t6_1_analyze_prefix_bounds as locked_figure  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


raw_manifest = json.loads((ART / "manifest.json").read_text(encoding="utf-8"))
figure_manifest = json.loads((ART / "analysis_manifest.json").read_text(encoding="utf-8"))
assert sha(ART / "prefix_raw.csv") == raw_manifest["raw_sha256"]
assert sha(ART / "identification_by_instance.csv") == figure_manifest["diagnostic_sha256"]
seed = figure_manifest["figure_selection_seed"]
assert seed == 10080015

frame = pd.read_csv(ART / "prefix_raw.csv")
frame = frame[frame.generation_seed == seed].copy()
diagnostics = pd.read_csv(ART / "identification_by_instance.csv")

original_savefig = matplotlib.figure.Figure.savefig


def vector_savefig(self, _filename, *args, **kwargs):
    kwargs.pop("dpi", None)
    return original_savefig(self, TARGET, *args, format="pdf", **kwargs)


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.figure.Figure.savefig = vector_savefig
try:
    locked_figure.make_figure(frame, diagnostics, figure_manifest["figure_selection_rule"])
finally:
    matplotlib.figure.Figure.savefig = original_savefig

print(f"Exported {TARGET}; SHA-256 {sha(TARGET)}")
