# KernelBench Data — Provenance & Honest Scope

This folder contains the task specifications used by the GRPO Failure Lab notebook to ground
its micro-environments in a real-world, verifiable-reward domain: **GPU kernel generation**.

## What is real vs. illustrative

**Real (sourced from the public benchmark):**
- The **task specifications** reference actual problems from the public **KernelBench** dataset
  (Ouyang et al., ICML 2025): [`ScalingIntelligence/KernelBench`](https://huggingface.co/datasets/ScalingIntelligence/KernelBench).
  Each task records its real `kernelbench_problem_id`, `kernelbench_problem_name`, `level`, and a
  PyTorch reference module, e.g.:
  - `1_Square_matrix_multiplication_` (Level 1, problem 1)
  - `23_Softmax` (Level 1, problem 23)
  - `40_LayerNorm` (Level 1, problem 40)
- The **qualitative failure patterns** (naive global loads are slow; shared-memory tiling with
  coalesced access wins; oversized tiles overflow shared memory and fail to compile; races cause
  accuracy mismatches; warp-shuffle reductions beat multi-pass kernels) are all documented behaviors
  reported in KernelBench (Ouyang et al., 2025) and Kevin (Baronio et al., 2025).

**Illustrative (NOT measured here):**
- The specific **latency (µs), speedup multipliers, and pass/fail flags** on each candidate kernel
  are **illustrative values**, hand-set to reflect the qualitative patterns above. They were **not**
  produced by live `nvcc` compilation or GPU profiling in this notebook — the tutorial is intentionally
  CPU-only and dependency-light so it runs anywhere in seconds. Every task carries a `metrics_note`
  field stating this.

## Why this design

The pedagogical goal is to make GRPO's *learning dynamics* transparent and reproducible on any laptop,
using a realistic verifiable-reward task as the framing. Using illustrative-but-representative outcomes
lets learners run the full failure lab with zero GPU/`nvcc`/PyTorch dependencies.

## Running against the real benchmark (optional)

To reproduce these numbers for real, clone the official harness and run on a GPU:

```bash
git clone https://github.com/ScalingIntelligence/KernelBench
# generate candidate kernels, then:
uv run python scripts/eval_from_generations.py run_name=<run> level=1 num_gpu_devices=<n>
```

The harness compiles each generated kernel with `nvcc`, checks correctness with `torch.allclose`
against the PyTorch reference, and benchmarks runtime with `torch.cuda.Event` to compute the
`fast_p` speedup metric — the exact verifiable reward this tutorial models.

## Citations
- Ouyang, Guo, Arora, Zhang, Hu, Ré, Mirhoseini. *KernelBench: Can LLMs Write Efficient GPU Kernels?* ICML 2025.
- Baronio, Marsella, Pan, Guo, Alberti. *Kevin: Multi-Turn RL for Generating CUDA Kernels.* arXiv:2507.11948, 2025.
