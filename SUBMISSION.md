# NeurIPS 2026 Education Track — Submission Compliance Checklist

Target venue: **NeurIPS 2026 Education Track**
Call: <https://neurips.cc/Conferences/2026/CallforEducationalResources>
Submit via: <https://openreview.net/group?id=NeurIPS.cc/2026/Education_Track>

| Milestone | Date |
|---|---|
| Submissions open | 3 July 2026 |
| **Submissions close** | **4 September 2026 (AoE)** |
| Notification | 28 September 2026 |

---

## Requirement checklist

| # | Requirement (from the call) | Status | Where |
|---|---|---|---|
| 1 | PDF **≤ 2 pages** | Done — 2 pages, recompiled 18 Nov 2026 with Tectonic 0.17 (XeTeX, the engine of the September PDF; the September source compiles to the same page layout). The 21 Oct `.tex` compiled to 3 pages, so the text was tightened | `concept_summary.tex`, `concept_summary.pdf` |
| 2 | Use the **"education, final"** template, **without the checklist** | Done — `\usepackage[education, final]{neurips_2026}`; no checklist block | `concept_summary.tex` |
| 3 | **Single-blind**: author names included | Done — authors listed | `concept_summary.tex` |
| 4 | State **the concept** | §1 "The Concept" | `concept_summary.tex` |
| 5 | State **leveling & prerequisite knowledge** | §2 — advanced undergrad/graduate; basic RL understanding required | `concept_summary.tex` |
| 6 | State **learning objectives & outcomes** | §3 — 6 objectives tagged with Bloom verbs (explain/apply/analyse/compare/experiment/evaluate) | `concept_summary.tex` |
| 7 | **≥3 linked papers** from 2022–2026 at leading venues | 13 linked references: DAPO, DisCO, S-GRPO @ NeurIPS'25; KernelBench @ ICML'25; BAPO @ ICLR'26; DeepSeekMath, DeepSeek-V3, DeepSeek-R1, Qwen3, PPO, Dr. GRPO, GSPO, and DCPO on arXiv | `concept_summary.tex` |
| 8 | **Teaching materials summary** | §5 | `concept_summary.tex` |
| 9 | Materials as **ZIP < 200 MB** | 27.8 MB (77 files; integrity verified, built 19 Nov 2026) | `make_submission_zip.sh` |
| 10 | No individual asset > ~40 MB (else host externally) | Largest packaged file ≈4.4 MB (notebook/02_failure_lab.ipynb) | script warns if violated |
| 11 | **Source code included** even though a repo is linked | Eight notebooks (including a method testbed), the `grpo_lab` package, four self-contained decks with PDF handouts, solutions, data, the curated figure set, the poster (PDF and source), tests and build scripts | `make_submission_zip.sh` |
| 12 | Prerequisites & objectives stated **in the materials too** | First cell of every notebook (reading time, runtime, prerequisites, Colab badge); title slide of each deck (with an acronym glossary) | `notebook/*.ipynb`, `slides/*.html` |
| 13 | Materials are **as self-contained as possible** | Each packaged deck embeds KaTeX's JS, CSS, and fonts, the shared deck code and the figures, so each deck works offline from a single file; notebooks embed their figures, fetch the repository when opened on Colab, and need no model weights, GPU, or API key | `slides/build_standalone.py`, `make_submission_zip.sh`, `notebook/` |
| 14 | **Original**, created specifically for this track | Yes | — |
| 15 | Concept is **emerging**, not an intro to established knowledge | GRPO failure modes + 2025 responses; the optional refresher notebook labels prerequisite material as scaffolding | `concept_summary.tex`, `notebook/00_pg_ppo_refresher.ipynb`, `README.md` |
| 16 | In-person presentation in Sydney; ≥1 author registers | Author action required | — |

---

## Before submitting

1. **Compile and confirm the page count after any edit** (Overleaf with XeLaTeX, or Tectonic: `tectonic -X compile concept_summary.tex`), and confirm it is still at most 2 pages with the required `[education, final]` template. The checked-in PDF matches the current `.tex` (18 Nov 2026).
2. **Save the PDF** as `concept_summary.pdf` in this folder.
3. **Build the ZIP:** `./make_submission_zip.sh` (it builds the self-contained decks with `slides/build_standalone.py` first).
4. **Re-run all notebooks** so saved outputs are current (`make notebooks`), and make sure `make test` passes. Figures come from the curated set (`make figures FIGSET=path/to/export.zip`, which also re-embeds them in the notebooks); the two refresher diagrams (00a, 00b) from `figures/generate_primer_diagrams.py`. Rebuild the handouts after changing a deck (`make handouts`).
5. **Publish the public repository:** run `tools/publish_github.sh`, review `build/github`, then `git -C build/github push origin main`. Enable GitHub Pages once (Settings → Pages → Source: GitHub Actions); CI then tests every push and publishes the decks and handouts.

## Deliberate scope statements (do not remove)

- **Data honesty.** KernelBench *task specifications* are real (Level-1 problem IDs `1_Square_matrix_multiplication_`, `23_Softmax`, `40_LayerNorm`); per-kernel *latency/speedup values are illustrative*, not from live `nvcc` runs. Stated in the notebook intro, §5 of the summary, and `data/kernelbench/README.md`.
- **Toy-to-scale gap.** Failures are demonstrated in micro-environments; the link to production LLM training is supported by citation (`02_failure_lab.ipynb` Appendix C), not reproduced.
