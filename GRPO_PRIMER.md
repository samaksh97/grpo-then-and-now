# GRPO Primer: The Whole Thing, Simply

*Personal study notes. Read this before (or alongside) the notebook.*

---

## 1. The Setup: What is SFT and Why Do We Need RL?

### What is Supervised Fine-Tuning (SFT)?

The starting point for adapting pretrained large language models (LLMs) is **Supervised Fine-Tuning (SFT)**. 

In SFT, we collect high-quality demonstration pairs $(x, y)$, where $x$ is a task prompt and $y = (y_1, y_2, \dots, y_T)$ is an expert demonstration.

**Concrete Example (KernelBench / GPU Kernel Optimization):**
- **Prompt $x$:** `"Write an optimized CUDA kernel for 2D matrix multiplication C = A @ B of size 1024x1024."`
- **Demonstration $y$:** An expert-written CUDA kernel using a specific block size (`BLOCK_SIZE = 32`) and shared-memory tiling.

SFT trains the model by minimizing cross-entropy loss (maximizing token log-probabilities):

$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} \log \pi_\theta(y_t \mid y_{<t}, x)$$

SFT is purely **imitation learning** -- the model learns to reproduce the exact token sequence provided by the demonstrator.

---

### The Three Limitations of SFT

While SFT is effective for style adaptation and basic instruction following, it has fundamental ceilings when applied to mathematical reasoning and code generation:

1. **The Imitation Ceiling & Lack of Exploration:**
   The model can never exceed the capabilities of the training data. If human demonstrations only contain basic shared-memory tiling, SFT cannot discover novel, hardware-specific optimizations (such as warp shuffle intrinsics `__shfl_down_sync`, register tiling, or tensor core instructions) that were absent from the corpus.

2. **Rigid Token-Level Supervision vs. Equifinality:**
   In code and mathematics, there are often dozens of completely different token sequences that compute the exact same result (or run even faster). SFT penalizes any valid token sequence that deviates from the single reference string, forcing the model to memorize superficial syntax rather than problem-solving logic.

3. **Distribution Shift & Error Compounding:**
   At test time, the model generates autoregressively without guidance. If it emits a single suboptimal token, it enters an unvisited state space where errors compound rapidly.

---

### Where RL Enters: Exploration + Verifiable Outcomes

Reinforcement Learning (RL) changes the supervision from **how** to write (token imitation) to **what** is achieved (outcome evaluation).

Instead of "copy this exact solution string," RL says: "explore different candidate implementations; an automated verifier (`nvcc` compiler + test harness + runtime profiler) will execute and benchmark them." Successful strategies are reinforced; failing ones are suppressed.

The core policy gradient equation (Williams, 1992) is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ R \cdot \nabla_\theta \log \pi_\theta(a \mid s) \right]$$

In words: **push up the probability of actions that got high reward.** But if all rewards are positive ($R > 0$), every action gets pushed up, creating massive gradient variance.

---

## 2. The Baseline Trick

To fix the noise, subtract a **baseline** $b$ from the reward:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ (R - b) \cdot \nabla_\theta \log \pi_\theta(a \mid s) \right]$$

The term $(R - b)$ is the **advantage**:
- If $R > b$: Action was better than expected $\to$ **reinforce**.
- If $R < b$: Action was worse than expected $\to$ **suppress**.

**PPO's approach (Standard RLHF):** Train a separate neural network (the "critic" $V_\psi(s)$) to predict expected reward. PPO requires **4 models in GPU memory simultaneously**:
1. **Policy $\pi_\theta$ (Active, Trained):** The generator.
2. **Reference $\pi_{\text{ref}}$ (Frozen):** Base SFT model providing KL regularization.
3. **Reward Model $R_\phi$ (Frozen):** Neural preference scorer.
4. **Critic $V_\psi$ (Active, Trained):** Value network predicting baseline.

This 4-model overhead requires ~56 GB VRAM for a 7B model and couples two competing training loops.

---

## 3. GRPO: The Group *Is* the Baseline

Introduced in DeepSeekMath (Shao et al., 2024) and scaled in DeepSeek-R1 (2025), **Group Relative Policy Optimization (GRPO)** eliminates the Critic and the Reward Model:

1. **Deterministic Verifier replaces the Reward Model:** In code/math, run `nvcc` and unit tests directly.
2. **Group Mean replaces the Critic:** Sample $G$ responses $\{a_1, \dots, a_G\}$ for the same prompt, score them, and compute group statistics:

$$\bar{R} = \frac{1}{G}\sum_{i=1}^G R_i, \qquad \sigma = \text{std}(\{R_i\})$$

$$A_i = \frac{R_i - \bar{R}}{\sigma + \epsilon}$$

Think of it like **runners in a heat**: you are judged against the other runners in your heat, not an absolute clock.

**What is left:** Only 2 models in memory ($\pi_\theta, \pi_{\text{ref}}$), 1 training loop, and zero reward model training ($\mathbf{75\%}$ VRAM savings).

---

## 4. The Full Objective (Clipping, KL, and Averaging)

For token sequences, GRPO incorporates PPO's clipped surrogate loss and KL regularization:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min\left( r_{i,t}(\theta) A_i, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) A_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

where $r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} \mid a_{i,<t})}{\pi_{\text{old}}(a_{i,t} \mid a_{i,<t})}$ is the importance sampling ratio.

---

## 5. When GRPO Breaks (The Three Failure Modes)

GRPO breaks in three systematic ways, operating across different structural levels:

### Failure 1: Zero-Gradient Groups (Macro / Group Bottleneck)

- **What happens:** If a kernel task is too hard (all $G$ samples fail to compile) or too easy (all $G$ succeed), every sample gets the same reward. The standard deviation $\sigma = 0$, and the numerator $(R_i - \bar{R}) = 0$.
- **The consequence:** Advantage is zero for all $G$ candidates $\implies$ **zero gradient**. Training stalls and compute is wasted.
- **At scale:** At $95\%$ accuracy with $G=8$, $66\%$ of groups produce zero learning signal.

### Failure 2: Entropy Collapse (Micro / Token Clipping Bottleneck)

- **What happens:** Symmetric clipping $[1-\epsilon, 1+\epsilon]$ (e.g. $[0.8, 1.2]$) caps ratio increases equally, but has a 50x asymmetric effect on probability mass:
  - Common token ($p=0.85$): $p \times 1.2 = 1.02 \to \mathbf{+0.12}$ allowed probability mass increase.
  - Rare exploratory token ($p=0.01$): $p \times 1.2 = 0.012 \to \mathbf{+0.002}$ allowed increase.
- **The consequence:** Even when a novel exploratory pattern (e.g. warp shuffle `__shfl_down_sync`) yields high reward, it cannot grow fast enough before dominant standard templates saturate the distribution. Policy entropy crashes to zero, permanently trapping the model in suboptimal templates.

### Failure 3: Length Bias (Structural / Sequence Aggregation Bottleneck)

- **What happens:** GRPO normalizes token losses by response length $\frac{1}{|o_i|}$.
- **The consequence:** A concise 20-token stub gets **10x more per-token gradient** than a thorough 200-token kernel with shared-memory tiling and register caching, even if both receive the same reward ($R=1.0$). The model is penalized for multi-step reasoning.

---

## 6. DAPO: The Direct Successor (NeurIPS 2025)

**DAPO** (Decoupled clip And dynamic sampling Policy Optimization) directly fixes all three failure modes:

| DAPO Fix | Fixes Failure Mode | Algorithmic Mechanism |
|---|---|---|
| **1. Dynamic Sampling** | Zero-gradient groups (5.1) | Discard groups where $\sigma = 0$; resample until batch has reward variance. |
| **2. Clip-Higher** | Entropy collapse (5.2) | Decouple clipping bounds: $\text{clip}(r_t, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})$ with $\epsilon_{\text{high}} > \epsilon_{\text{low}}$ (e.g. $0.2 / 0.28$). Gives rare tokens room to grow. |
| **3. Token-Level Loss** | Length bias (5.3) | Aggregate loss across all tokens in the batch equally ($\frac{1}{\sum |o_i|}$) instead of per-sample. |
| **4. Overlong Reward Shaping** | Truncation noise | Soft linear penalty for responses exceeding length limits instead of abrupt zero reward. |

---

## 7. Beyond DAPO: Frontier Directions

- **DisCO (NeurIPS 2025):** Replaces group-relative normalization with a discriminative objective and adaptive Lagrange multiplier constraints, removing question-difficulty bias.
- **S-GRPO (NeurIPS 2025):** Introduces serial exit points and decaying rewards along a single reasoning chain, reducing inference tokens by 40-61%.

---

## 8. Summary: The Mental Model

```
SFT (Imitation Learning)  ──>  Limited by dataset demonstrations & rigid token matching
    │
    ▼
PPO / RLHF (1992, 2017)   ──>  4 models in VRAM (~56 GB), 2 training loops, learned critic baseline
    │
    ▼
GRPO (2024)               ──>  2 models in VRAM (~14 GB), deterministic verifier + group mean baseline
    │
    ├──> DAPO (2025)      ──>  Dynamic Sampling + Clip-Higher + Token-Level Loss
    ├──> DisCO (2025)     ──>  Discriminative constrained optimization
    └──> S-GRPO (2025)    ──>  Serial exit points for inference efficiency
```
