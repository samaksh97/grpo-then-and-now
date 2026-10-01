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
| 1 | PDF **≤ 2 pages** | Body ≈1.9 pp (911 words + table). **Verify by compiling** | `concept_summary.tex` |
| 2 | Use the **"education, final"** template, **without the checklist** | Done — `\usepackage[education, final]{neurips_2026}`; no checklist block | `concept_summary.tex` |
| 3 | **Single-blind**: author names included | Done — authors listed | `concept_summary.tex` |
| 4 | State **the concept** | §1 "The Concept" | `concept_summary.tex` |
| 5 | State **leveling & prerequisite knowledge** | §2 — advanced undergrad/graduate; no prior RL assumed | `concept_summary.tex` |
| 6 | State **learning objectives & outcomes** | §3 — 6 objectives tagged with Bloom verbs (understand/apply/analyse/evaluate/create) | `concept_summary.tex` |
| 7 | **≥3 linked papers** from 2022–2026 at leading venues | §4 — 7 papers (DAPO, DisCO, S-GRPO @ NeurIPS'25; KernelBench @ ICML'25; DeepSeekMath, DeepSeek-R1, Kevin) | `concept_summary.tex` |
| 8 | **Teaching materials summary** | §5 | `concept_summary.tex` |
| 9 | Materials as **ZIP < 200 MB** | ~4–5 MB | `make_submission_zip.sh` |
| 10 | No individual asset > ~40 MB (else host externally) | Largest file ≈0.3 MB | script warns if violated |
| 11 | **Source code included** even though a repo is linked | Notebook + solutions + data + `.sty` all in ZIP | `make_submission_zip.sh` |
| 12 | Prerequisites & objectives stated **in the materials too** | Notebook cell 0 | `notebook/grpo_failure_lab.ipynb` |
| 13 | **Original**, created specifically for this track | Yes | — |
| 14 | Concept is **emerging**, not an intro to established knowledge | GRPO failure modes + 2025 successors; Section 1 explicitly labelled as brief scaffolding | `concept_summary.tex`, notebook cell 0, `README.md` |
| 15 | In-person presentation in Sydney; ≥1 author registers | Author action required | — |

---

## Before submitting

1. **Compile and confirm the page count.** Upload `concept_summary.tex` + `neurips_2026.sty` to Overleaf and check it is ≤ 2 pages. If it overflows, trim in this order: (a) the paragraph after the takeaway box in §1, (b) the last bullet of §5, (c) drop the `ppo` reference.
2. **Save the PDF** as `concept_summary.pdf` in this folder.
3. **Build the ZIP:** `./make_submission_zip.sh`
4. **Re-run the notebook** so shipped outputs are current: all 29 code cells should pass with 0 errors and regenerate 15 figures.
5. If linking a public repo, mirror this folder to a public host (the current remote is internal GitLab and will not be reachable by reviewers).

## Deliberate scope statements (do not remove)

- **Data honesty.** KernelBench *task specifications* are real (Level-1 problem IDs `1_Square_matrix_multiplication_`, `23_Softmax`, `40_LayerNorm`); per-kernel *latency/speedup values are illustrative*, not from live `nvcc` runs. Stated in the notebook intro, §5 of the summary, and `data/kernelbench/README.md`.
- **Toy-to-scale gap.** Failures are demonstrated in micro-environments; the link to production LLM training is supported by citation (Section 7.5), not reproduced.
