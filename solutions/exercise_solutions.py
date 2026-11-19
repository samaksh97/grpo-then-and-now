"""Seeded reference solutions for the six core GRPO Failure Lab exercises."""

import numpy as np
import matplotlib.pyplot as plt


def softmax(logits):
    x = np.asarray(logits) - np.max(logits)
    e = np.exp(x)
    return e / e.sum()


def entropy(probs):
    probs = np.asarray(probs)
    probs = probs[probs > 1e-10]
    return -np.sum(probs * np.log(probs))


class BanditVerifier:
    """Notebook-compatible verifier API."""
    def __init__(self, rewards, correct_action):
        self.rewards = np.asarray(rewards, dtype=float)
        self.correct = correct_action

    def __call__(self, action):
        return self.rewards[action]


BANDIT_VERIFIER = BanditVerifier([0, 0, 0, 1, 0, 0, 0, 0], correct_action=3)


def bandit_grpo_step(logits, verifier, group_size, lr):
    """One update; p_correct is present even when the group has zero variance."""
    probs = softmax(logits)
    actions = np.random.choice(len(logits), size=group_size, p=probs)
    rewards = np.array([verifier(a) for a in actions])
    info = {
        "reward_mean": rewards.mean(),
        "entropy": entropy(probs),
        "p_correct": probs[verifier.correct],
        "generated_responses": group_size,
    }
    if rewards.std() < 1e-8:
        return logits, {**info, "zero_grad": True, "grad_norm": 0.0}
    advantages = (rewards - rewards.mean()) / rewards.std()
    grad = np.zeros_like(logits)
    for action, advantage in zip(actions, advantages):
        one_hot = np.zeros_like(logits)
        one_hot[action] = 1.0
        grad += advantage * (one_hot - probs) / group_size
    logits = logits + lr * grad
    new_probs = softmax(logits)
    return logits, {
        **info,
        "entropy": entropy(new_probs),
        "p_correct": new_probs[verifier.correct],
        "zero_grad": False,
        "grad_norm": np.linalg.norm(grad),
    }


def bandit_grpo_step_dynamic(logits, verifier, group_size, lr, max_attempts=5):
    """Finite-retry Dynamic Sampling with generated-response diagnostics."""
    probs = softmax(logits)
    for attempts in range(1, max_attempts + 1):
        actions = np.random.choice(len(logits), size=group_size, p=probs)
        rewards = np.array([verifier(a) for a in actions])
        if rewards.std() > 1e-8:
            break
    info = {
        "reward_mean": rewards.mean(),
        "entropy": entropy(probs),
        "p_correct": probs[verifier.correct],
        "attempts": attempts,
        "generated_responses": attempts * group_size,
    }
    if rewards.std() < 1e-8:
        return logits, {**info, "zero_grad": True, "grad_norm": 0.0}
    advantages = (rewards - rewards.mean()) / rewards.std()
    grad = np.zeros_like(logits)
    for action, advantage in zip(actions, advantages):
        one_hot = np.zeros_like(logits)
        one_hot[action] = 1.0
        grad += advantage * (one_hot - probs) / group_size
    logits = logits + lr * grad
    new_probs = softmax(logits)
    return logits, {
        **info,
        "entropy": entropy(new_probs),
        "p_correct": new_probs[verifier.correct],
        "zero_grad": False,
        "grad_norm": np.linalg.norm(grad),
    }


def solve_exercise_1():
    rewards = np.array([1, 0, 1, 1, 0, 0], dtype=float)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    print("Exercise 1:", advantages, "sum=", advantages.sum())
    return advantages


def solve_exercise_2():
    """Full seeded sweep using the >=0.90 pre-update convention and requested plot."""
    group_sizes = [4, 8, 16, 32, 64]
    results = {}
    for group_size in group_sizes:
        np.random.seed(42)
        logits = np.zeros(8)
        reached = None
        zero_count = 0
        for step in range(300):
            if reached is None and softmax(logits)[BANDIT_VERIFIER.correct] >= 0.90:
                reached = step
            logits, info = bandit_grpo_step(logits, BANDIT_VERIFIER, group_size, 0.5)
            zero_count += info["zero_grad"]
        results[group_size] = {
            "steps_to_90": reached,
            "zero_grad_fraction": zero_count / 300,
            "generated_responses": 300 * group_size,
        }

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(group_sizes, [results[g]["steps_to_90"] if results[g]["steps_to_90"] is not None else 300 for g in group_sizes], "o-")
    axes[0].set(title=r"First step with $P(correct)\geq0.90$", xlabel="G", ylabel="Step")
    axes[1].plot(group_sizes, [results[g]["zero_grad_fraction"] for g in group_sizes], "o-", color="tab:red")
    axes[1].set(title="Zero-gradient fraction", xlabel="G", ylabel="Fraction")
    plt.tight_layout()
    plt.close(fig)
    print("Exercise 2:", results)
    return results


def solve_exercise_3():
    """Matched requested-update comparison, including generation cost."""
    results = {}
    for name, step_fn in [("vanilla", bandit_grpo_step), ("dynamic", bandit_grpo_step_dynamic)]:
        np.random.seed(123)
        logits = np.zeros(8)
        zero_count = generated = 0
        for _ in range(100):
            logits, info = step_fn(logits, BANDIT_VERIFIER, 8, 0.5)
            zero_count += info["zero_grad"]
            generated += info["generated_responses"]
        results[name] = {
            "p_correct": softmax(logits)[BANDIT_VERIFIER.correct],
            "zero_updates": zero_count,
            "generated_responses": generated,
        }
    assert results["dynamic"]["generated_responses"] >= results["vanilla"]["generated_responses"]
    print("Exercise 3:", results)
    print("Finite retries reduce, but do not guarantee elimination of, zero-gradient updates.")
    return results


def fixed_rollout_clip_toy(eps_low, eps_high, epochs=30, lr=0.8):
    """Actual clipped-surrogate updates against one frozen behavior rollout."""
    old_logits = np.array([2.0, -1.0, -2.0, -2.0])
    old_probs = softmax(old_logits)
    actions = np.array([0, 0, 0, 0, 0, 1, 2, 3])
    rewards = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=float)
    advantages = (rewards - rewards.mean()) / rewards.std()
    logits = old_logits.copy()
    history = []
    for _ in range(epochs):
        probs = softmax(logits)
        ratios = probs[actions] / old_probs[actions]
        grad = np.zeros_like(logits)
        clipped = []
        for action, advantage, ratio in zip(actions, advantages, ratios):
            active = ((advantage >= 0 and ratio <= 1 + eps_high) or
                      (advantage < 0 and ratio >= 1 - eps_low))
            clipped.append(not active)
            if active:
                one_hot = np.zeros_like(logits)
                one_hot[action] = 1.0
                grad += advantage * ratio * (one_hot - probs) / len(actions)
        history.append({"ratios": ratios.copy(), "entropy": entropy(probs),
                        "clip_fraction": np.mean(clipped), "probs": probs.copy()})
        logits += lr * grad
    return history


def solve_exercise_4():
    configs = [("symmetric", 0.20, 0.20), ("DAPO", 0.20, 0.28), ("wide_toy", 0.20, 0.50)]
    results = {}
    for name, low, high in configs:
        history = fixed_rollout_clip_toy(low, high)
        results[name] = {
            "max_ratio": max(np.max(h["ratios"]) for h in history),
            "mean_clip_fraction": np.mean([h["clip_fraction"] for h in history]),
            "final_rare_probability": history[-1]["probs"][1],
        }
    assert all(np.allclose(fixed_rollout_clip_toy(lo, hi)[0]["ratios"], 1.0)
               for _, lo, hi in configs)
    print("Exercise 4:", results)
    return results


END = 9
CANDIDATES = [[], [0, 0, END], [2, 1, END], [1, 1, END], [2, 2, END]]


def vulnerable_verifier(sequence):
    return float(bool(sequence) and sequence[-1] == END)


def robust_verifier(sequence, target_value=3):
    """Interpret valid operations and test behavior rather than exact text."""
    if not sequence or sequence[-1] != END:
        return 0.0
    value = 1
    for operation in sequence[:-1]:
        if operation == 1:
            value += 1
        elif operation == 2:
            value *= 2
        else:
            return 0.0
    return float(value == target_value)


def train_candidate_policy(verifier, seed=7, steps=120):
    np.random.seed(seed)
    logits = np.zeros(len(CANDIDATES))
    trace = []
    for _ in range(steps):
        probs = softmax(logits)
        actions = np.random.choice(len(logits), 8, p=probs)
        rewards = np.array([verifier(CANDIDATES[a]) for a in actions])
        if rewards.std() > 1e-8:
            advantages = (rewards - rewards.mean()) / rewards.std()
            grad = np.zeros_like(logits)
            for action, advantage in zip(actions, advantages):
                one_hot = np.zeros_like(logits)
                one_hot[action] = 1.0
                grad += advantage * (one_hot - probs) / len(actions)
            logits += 0.25 * grad
        trace.append(softmax(logits).copy())
    return np.asarray(trace)


def solve_exercise_5():
    vulnerable_trace = train_candidate_policy(vulnerable_verifier)
    robust_trace = train_candidate_policy(robust_verifier)
    assert robust_verifier([]) == 0.0
    assert robust_verifier([0, 0, END]) == 0.0
    assert robust_verifier([2, 1, END]) == robust_verifier([1, 1, END]) == 1.0
    results = {
        "exploit_initial_probability": 1 / len(CANDIDATES),
        "exploit_final_vulnerable": vulnerable_trace[-1, 1],
        "exploit_final_robust": robust_trace[-1, 1],
    }
    print("Exercise 5 (finite-candidate toy):", results)
    return results


CANONICAL_SCENARIOS = [
    ("All rewards in most groups are identical on a hard task.",
     "Zero-gradient groups", "Dynamic Sampling; finite retries reduce failures at extra response cost"),
    ("Fixed-rollout positive-advantage ratios repeatedly hit the upper clip bound.",
     "Clipping-constrained exploration", "DAPO Clip-Higher"),
    ("Correct responses vary from 8 to 200 tokens under sample-level normalization.",
     "Variable-length weighting", "DAPO token-level normalization"),
    ("Responses enter the cache region or hit the truncation boundary.",
     "Overlong behavior", "Additive soft penalty; separately consider overlong filtering"),
    ("Perfect training reward hides failure on randomized held-out tests.",
     "Reward hacking", "Behavior/property-based adversarial verifier tests"),
]


def solve_exercise_6():
    print("Exercise 6:")
    for index, (symptom, diagnosis, response) in enumerate(CANONICAL_SCENARIOS, 1):
        print(f"  {index}. {symptom}\n     {diagnosis} -> {response}")
    return CANONICAL_SCENARIOS


if __name__ == "__main__":
    solve_exercise_1()
    solve_exercise_2()
    solve_exercise_3()
    solve_exercise_4()
    solve_exercise_5()
    solve_exercise_6()
    print("ALL 6 EXERCISE SOLUTIONS EXECUTED SUCCESSFULLY.")
