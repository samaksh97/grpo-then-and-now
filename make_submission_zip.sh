#!/usr/bin/env bash
# Build the NeurIPS 2026 Education Track submission ZIP.
#
# Track requirements addressed:
#   - Teaching materials as a single ZIP under 200 MB
#   - Source code included even though a repo is also linked
#   - Any individual asset over ~40 MB must be hosted externally (none here)
#
# Uses Python's zipfile so no `zip` binary is required.
# Usage:  ./make_submission_zip.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

python3 - <<'PY'
import os, zipfile

OUT = "neurips2026_education_grpo.zip"
INCLUDE_FILES = [
    "notebook/grpo_failure_lab.ipynb",
    "README.md",
    "GRPO_PRIMER.md",
    "SUBMISSION.md",
    "concept_summary.tex",
    "concept_summary_full.tex",
    "neurips_2026.sty",
    "concept_summary.pdf",      # optional
]
INCLUDE_DIRS = ["figures", "data", "solutions"]
SKIP = (".ipynb_checkpoints", "__pycache__", ".DS_Store", ".git")

def keep(path):
    return not any(s in path for s in SKIP)

members = []
for f in INCLUDE_FILES:
    if os.path.isfile(f):
        # Never ship a PDF that is older than its LaTeX source: it would be the
        # wrong (pre-trim) version and could blow the 2-page limit.
        if f == "concept_summary.pdf" and os.path.getmtime(f) < os.path.getmtime("concept_summary.tex"):
            print("ERROR: concept_summary.pdf is OLDER than concept_summary.tex (stale build).")
            print("       Recompile the 2-page summary and re-run; excluding it from the ZIP.")
            continue
        members.append(f)
    elif not f.endswith(".pdf"):
        print(f"WARNING: expected file missing: {f}")
    else:
        print("NOTE: concept_summary.pdf not found - compile concept_summary.tex and re-run.")

for d in INCLUDE_DIRS:
    for root, _, files in os.walk(d):
        for name in files:
            p = os.path.join(root, name)
            if keep(p):
                members.append(p)

# Warn about assets that would need external hosting
for p in members:
    mb = os.path.getsize(p) / 1e6
    if mb > 40:
        print(f"WARNING: {p} is {mb:.1f} MB (>40 MB) - host externally and link instead.")

if os.path.exists(OUT):
    os.remove(OUT)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(members):
        z.write(p)

size_mb = os.path.getsize(OUT) / 1e6
print(f"\nBuilt {OUT}: {len(members)} files, {size_mb:.2f} MB (limit 200 MB).")
assert size_mb < 200, "ZIP exceeds the 200 MB track limit"
PY
