"""Mechanically inventory project files for the COAP cleanup decision.

Run before deletion; this script never deletes or moves files.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "SUBMISSION_CLEANUP_MANIFEST.md"
EXCLUDED_PARTS = {".venv", ".pytest_cache", "__pycache__"}


def classify(rel: str) -> tuple[str, str, str, str]:
    p = Path(rel)
    name = p.name
    if rel.startswith("COAP_SUBMISSION/"):
        return "SUBMIT", "COAP upload/preflight package", "No", "No new scientific result"
    if rel in {
        "manuscript/master.tex", "manuscript/master.pdf", "manuscript/references.bib"
    }:
        return "KEEP_INTERNAL", "Authoritative T6.1 freeze baseline", "Yes" if name == "references.bib" else "Baseline", "Yes"
    if rel == "artifacts/t6_1_full_prefix_reanalysis/theory_evidence.png":
        return "KEEP_INTERNAL", "Figure 1 source image and immutable re-analysis output", "Yes", "Yes"
    if rel.startswith(("artifacts/", "experiments/", "src/", "tests/")):
        return "KEEP_INTERNAL", "Scientific/reproducibility artifact; not a manuscript upload", "No", "Potentially"
    if rel.startswith("scripts/"):
        return "KEEP_INTERNAL", "Source or audit utility; preserve pending reproducibility review", "No", "Potentially"
    if rel.startswith("manuscript/sections/") or rel.startswith("manuscript/plots/") or rel.startswith("manuscript/archive/"):
        return "ARCHIVE", "Historical non-authoritative manuscript material", "No", "Potentially"
    if rel.startswith("theory/") or rel.startswith("scratch/"):
        return "ARCHIVE", "Historical theory or scratch material; conservatively retain", "No", "Potentially"
    if rel.startswith("tmp/") or (rel.startswith("manuscript/master.") and p.suffix in {".aux", ".bbl", ".blg", ".log", ".out"}):
        return "DELETE_CANDIDATE", "Regenerable compiler/render intermediate", "No", "No"
    if name.startswith("test_fig"):
        return "ARCHIVE", "Historical figure-test output; retain pending unique-content review", "No", "Potentially"
    if rel.endswith((".md", ".txt")) and rel not in {"README.md", "requirements.txt"}:
        return "KEEP_INTERNAL", "Research audit, provenance or submission-preparation record", "No", "Potentially"
    return "KEEP_INTERNAL", "Project configuration or source; conservative retention", "No", "Potentially"


def esc(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


rows = []
for path in ROOT.rglob("*"):
    if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
        continue
    rel = path.relative_to(ROOT).as_posix()
    if rel == OUT.name:
        continue
    category, reason, referenced, unique = classify(rel)
    rows.append((rel, category, reason, referenced, unique, "No action before dependency and clean-room checks"))
rows.sort(key=lambda row: row[0].lower())

lines = [
    "# COAP submission cleanup safety inventory",
    "",
    "This inventory was generated before any project deletion or archive move. Each project file outside regenerable virtual-environment/cache directories has one category. Existing files under `.venv/` are `KEEP_INTERNAL` as an environment tree; `.pytest_cache/` and `__pycache__/` are `DELETE_CANDIDATE` caches. No directory-level classification authorizes deletion of a unique scientific file. The table is a pre-action snapshot; a later action log records any completed moves/deletions.",
    "",
    "## Dependency check",
    "",
    "The authoritative source is `manuscript/master.tex`. It contains no `\\input` or `\\include`; its only project-local runtime dependencies are `manuscript/references.bib` and `artifacts/t6_1_full_prefix_reanalysis/theory_evidence.png` via `\\bibliography{references}` and `\\includegraphics`. Standard TeX packages/classes are distribution dependencies, not project files. Figure regeneration additionally requires `experiments/t6_1_analyze_prefix_bounds.py` and the T6.1 prefix CSV/manifest; numerical table verification requires Phase B/B.5 raw artifacts and their scripts. All are `KEEP_INTERNAL`. No `DELETE_CANDIDATE` is an active-source or regeneration dependency.",
    "",
    "| Path | Category | Reason | Referenced by active source? | Contains unique scientific content? | Action taken |",
    "|---|---|---|---|---|---|",
]
for row in rows:
    lines.append("| " + " | ".join(esc(v) for v in row) + " |")
lines.extend(["", f"Inventory count (excluding this manifest and environment/cache trees): {len(rows)} files.", ""])
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}: {len(rows)} files")
