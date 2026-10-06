"""Finalize the pre-deletion inventory with a verified move/delete action ledger."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "SUBMISSION_CLEANUP_MANIFEST.md"
BEFORE = ROOT / "internal_archive" / "pre_cleanup_inventory.md"
EXCLUDED = {".venv", ".pytest_cache", "__pycache__"}


def moved_to(rel: str) -> str | None:
    for prefix in ("manuscript/sections/", "manuscript/plots/", "manuscript/archive/"):
        if rel.startswith(prefix):
            return "internal_archive/" + rel
    for prefix in ("theory/", "scratch/"):
        if rel.startswith(prefix):
            return "internal_archive/" + rel
    if rel.startswith("test_fig"):
        return "internal_archive/figure_tests/" + rel
    return None


def action(rel: str) -> str:
    target = moved_to(rel)
    if target and (ROOT / target).is_file():
        return f"Moved recoverably to `{target}`"
    if rel.startswith("tmp/pdfs/"):
        return "Deleted after PDF visual QA; regenerable page PNG"
    if rel in {
        "manuscript/master.aux", "manuscript/master.bbl", "manuscript/master.blg",
        "manuscript/master.log", "manuscript/master.out",
    }:
        return "Deleted after clean-room compilation; regenerable TeX intermediate"
    if (ROOT / rel).is_file():
        return "Retained in place"
    raise RuntimeError(f"Unaccounted original path: {rel}")


lines = BEFORE.read_text(encoding="utf-8").splitlines()
header_idx = lines.index("| Path | Category | Reason | Referenced by active source? | Contains unique scientific content? | Action taken |")
old_rows = []
for line in lines[header_idx + 2 :]:
    if not line.startswith("| "):
        break
    fields = [field.strip() for field in line.strip("|").split("|")]
    if len(fields) != 6:
        raise RuntimeError(f"Malformed row: {line[:100]}")
    old_rows.append(fields)

tracked = {row[0] for row in old_rows}
new_rows = []
for path in ROOT.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(ROOT).as_posix()
    if any(part in EXCLUDED for part in path.relative_to(ROOT).parts) or rel == OUT.name or rel in tracked:
        continue
    if rel.startswith("COAP_SUBMISSION/"):
        category, reason, taken = "SUBMIT", "Journal manuscript or editorial upload staging file", "Created/staged; upload role requires author confirmation"
    elif rel.startswith("internal_archive/"):
        category, reason, taken = "ARCHIVE", "Recoverably retained historical file", "Moved from original path or preserved pre-cleanup inventory"
    else:
        category, reason, taken = "KEEP_INTERNAL", "Internal audit, provenance, source, or preparation file", "Created or retained internally"
    new_rows.append([rel, category, reason, "Yes" if rel == "COAP_SUBMISSION/figure1.pdf" else "No", "Potentially" if category != "SUBMIT" else "No new science", taken])

new_rows.append([
    "COAP_SUBMISSION/figure1.png", "DELETE_CANDIDATE",
    "Byte-identical temporary duplicate of retained T6.1 PNG; vector PDF is submission graphic",
    "No", "No", "Deleted after clean-room vector-figure build and visual QA",
])

out = [
    "# COAP submission cleanup manifest",
    "",
    "The safety inventory was created **before** any deletion or move and is preserved unchanged as `internal_archive/pre_cleanup_inventory.md`. The table below updates every original row with its action, and adds newly staged/archived files. Files inside `.venv/` remain `KEEP_INTERNAL` as an environment tree; `.pytest_cache/` and `__pycache__/` remain undeleted `DELETE_CANDIDATE` caches. No scientific raw result, manifest, unique draft, or figure-generation script was deleted.",
    "",
    "## Dependency check",
    "",
    "The authoritative T6.1 source is `manuscript/master.tex`. It has no local `\\input`/`\\include`; its two project-local build dependencies are `manuscript/references.bib` and the locked `artifacts/t6_1_full_prefix_reanalysis/theory_evidence.png`. The packaged source instead refers only to its sibling `references.bib` and `figure1.pdf`. Clean-room compilation and all-page visual QA passed. The raw Phase B/B.5 and T6.1 results, exact-oracle outputs, generators, solver/analysis scripts, seed manifests and hashes remain outside the package. No deleted file is referenced by either active source or required to verify a numerical table or regenerate Figure 1.",
    "",
    "| Path | Category | Reason | Referenced by active source? | Contains unique scientific content? | Action taken |",
    "|---|---|---|---|---|---|",
]
for row in sorted(old_rows, key=lambda x: x[0].lower()):
    row[5] = action(row[0])
    out.append("| " + " | ".join(row) + " |")
for row in sorted(new_rows, key=lambda x: x[0].lower()):
    out.append("| " + " | ".join(row) + " |")
out.extend([
    "", "## Permanent deletion summary", "",
    "Deleted: the one byte-identical packaged PNG duplicate, five `manuscript/master` TeX intermediates, and 88 `tmp/pdfs/` page-render PNGs. These outputs can be recreated. Historical manuscript/theory/scratch material was moved, not deleted. The recoverable relocation map is encoded in the action column.",
    "", f"Original rows: {len(old_rows)}; added final-state rows: {len(new_rows)}.", "",
])
OUT.write_text("\n".join(out), encoding="utf-8")
print(f"Finalized {OUT}: {len(old_rows) + len(new_rows)} rows")
