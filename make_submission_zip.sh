#!/usr/bin/env bash
# Build the NeurIPS 2026 Education Track submission ZIP.
#
# Track requirements addressed:
#   - Teaching materials as a single ZIP under 200 MB
#   - Source code included even though a repo is also linked
#   - Any individual asset over ~40 MB must be hosted externally (none here)
#
# The slide decks are packaged as self-contained files: slides/build_standalone.py inlines
# KaTeX (JS, CSS and woff2 fonts) into each deck, so equations render even when a single
# HTML file is opened straight from the ZIP. Only KaTeX's LICENSE is shipped alongside.
#
# Uses Python's zipfile so no `zip` binary is required.
# Usage:  ./make_submission_zip.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
python3 slides/build_standalone.py "${STAGE}/slides"

STAGE="${STAGE}" python3 - <<'PY'
import glob, os, zipfile

OUT = "neurips2026_education_grpo.zip"
STAGE = os.environ["STAGE"]
REQUIRED_FILES = ["README.md", "concept_summary.pdf", "LICENSE", "LICENSE-CONTENT.md"]
DECKS = ["grpo_intro.html", "dapo_intro.html", "disco_intro.html", "sgrpo_intro.html"]
INCLUDE_PATTERNS = [
    "notebook/*.ipynb",
    "grpo_lab/*.py",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "Makefile",
    "tools/*.py",
    "tests/*.py",
    "slides/vendor/katex/LICENSE",
    "data/kernelbench/README.md",
    "data/kernelbench/tasks.json",
    "solutions/*.py",
    "INSTRUCTOR_NOTES.md",
    "slides/handouts/*.pdf",
    "figures/set/*.png",
    "figures/set/figures.json",
    "figures/import_figure_set.py",
    "figures/00a_rlvr_kernelbench_loop.png",
    "figures/00b_ppo_update_loop.png",
    "figures/generate_primer_diagrams.py",
    "figures/generate_neurips_figure.py",
    "figures/neurips_figure_simple.png",
    "figures/neurips_figure_simple.pdf",
    "poster/grpo_then_and_now_poster.pdf",
    "poster/poster.html",
    "poster/build_poster.py",
    "poster/qr_*.svg",
]
REJECT = (".ipynb_checkpoints", "__pycache__", ".DS_Store", "Thumbs.db",
          "Zone.Identifier", "PG$Secure", "sec.endpointdlp", ".git", ".zip")

def keep(path):
    return not any(part in path for part in REJECT) and not os.path.islink(path)

missing = [path for path in REQUIRED_FILES if not os.path.isfile(path)]
if missing:
    raise FileNotFoundError(f"Required submission files are missing: {missing}")
if os.path.getmtime("concept_summary.pdf") < os.path.getmtime("concept_summary.tex"):
    raise RuntimeError("concept_summary.pdf is older than concept_summary.tex; recompile it first")

members = list(REQUIRED_FILES)
for pattern in INCLUDE_PATTERNS:
    matches = [path for path in glob.glob(pattern) if os.path.isfile(path) and keep(path)]
    if not matches:
        raise FileNotFoundError(f"Submission pattern matched no files: {pattern}")
    members.extend(matches)
# (source path, path inside the ZIP); the decks come from the self-contained build
entries = [(p, p) for p in sorted(set(members))]
entries += [(os.path.join(STAGE, "slides", d), f"slides/{d}") for d in DECKS]
entries.sort(key=lambda e: e[1])

# Warn about assets that would need external hosting
for src, arc in entries:
    mb = os.path.getsize(src) / 1e6
    if mb > 40:
        print(f"WARNING: {arc} is {mb:.1f} MB (>40 MB) - host externally and link instead.")

if os.path.exists(OUT):
    os.remove(OUT)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for src, arc in entries:
        z.write(src, arc)

size_mb = os.path.getsize(OUT) / 1e6
largest = max(entries, key=lambda e: os.path.getsize(e[0]))
print(f"\nBuilt {OUT}: {len(entries)} files, {size_mb:.2f} MB (limit 200 MB); "
      f"largest file {largest[1]} ({os.path.getsize(largest[0]) / 1e6:.2f} MB).")
assert size_mb < 200, "ZIP exceeds the 200 MB track limit"
PY
