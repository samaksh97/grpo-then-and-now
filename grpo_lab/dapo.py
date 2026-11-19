"""The Dynamic Sampling run simulation from dapo_four_fixes.ipynb Section 2 (Yu et al., 2025, Sec. 3.2).

Synthetic prompt difficulties stand in for a policy that improves over training; each step rolls the G
responses of every prompt and applies DAPO's constraint 0 < #correct < G (Eq. 8). This is the data behind
figure F9.
"""
import numpy as np


def prompt_accuracies(skill, n_prompts, rng):
    """Per-prompt accuracy p, shape (n_prompts,). skill in [0, 1) drags the Beta(2, 5) difficulty
    distribution toward 1.0, standing in for a policy that is getting better."""
    base = rng.beta(2.0, 5.0, size=n_prompts)      # most prompts start hard
    return base + skill * (1.0 - base)


def is_informative(p_vec, group_size, rng):
    """Roll the G responses of each prompt and test DAPO's constraint 0 < #correct < G; shape (n_prompts,)."""
    n_correct = rng.binomial(group_size, p_vec)
    return (n_correct > 0) & (n_correct < group_size)


def effective_batch_run(prompt_batch=512, group_size=16, n_steps=60, seed=0):
    """Naive GRPO's effective batch over a run, and the generation Dynamic Sampling needs to stay full.

    Returns a dict of arrays over steps: 'naive_effective' (prompts contributing gradient),
    'cost_multiplier' (prompts sampled per informative prompt), and the relative gradient noise
    1/sqrt(n_eff) for both methods.
    """
    rng = np.random.default_rng(seed)
    steps = np.arange(0, n_steps)
    skills = 0.95 * steps / steps.max()            # the policy improves over the run
    naive_effective, cost_multiplier = [], []
    for skill in skills:
        keep = is_informative(prompt_accuracies(skill, prompt_batch, rng), group_size, rng)
        n_eff = int(keep.sum())
        naive_effective.append(n_eff)
        cost_multiplier.append(prompt_batch / max(n_eff, 1))   # sample ~batch / (share informative)
    naive_effective = np.array(naive_effective)
    dynamic_effective = np.full_like(naive_effective, prompt_batch)
    return {"steps": steps, "naive_effective": naive_effective, "dynamic_effective": dynamic_effective,
            "cost_multiplier": np.array(cost_multiplier),
            "naive_noise": 1.0 / np.sqrt(np.maximum(naive_effective, 1)),
            "dynamic_noise": 1.0 / np.sqrt(dynamic_effective)}
