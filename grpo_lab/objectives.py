"""Group-relative advantages, clipped objectives and GRPO update steps for the toy environments.

References: DeepSeekMath (Shao et al., 2024): GRPO objective Eq. 3, KL estimator Eq. 4, outcome-supervision
advantage in Sec. 4.1.2. PPO (Schulman et al., 2017): clipped objective L^CLIP, Eq. 7. DAPO (Yu et al., 2025):
decoupled clip range Eq. 8.
"""
import numpy as np


def softmax(logits):
    """Numerically stable softmax over the last axis; shape (K,) -> (K,)."""
    x = logits - logits.max()
    e = np.exp(x)
    return e / e.sum()


def entropy(probs):
    """Shannon entropy in nats of a probability vector, shape (K,)."""
    p = probs[probs > 1e-10]
    return -np.sum(p * np.log(p))


def group_advantages(rewards, eps=1e-8):
    """A_i = (R_i - mean(R)) / (std(R) + eps) for one group; rewards shape (G,) -> (G,).

    DeepSeekMath Sec. 4.1.2 (outcome supervision). std is the population standard deviation (ddof=0).
    A group whose rewards are all equal gets all-zero advantages, so it contributes no gradient.
    """
    rewards = np.asarray(rewards, dtype=float)
    return (rewards - rewards.mean()) / (rewards.std() + eps)


def ppo_clip_objective(ratio, advantage, eps=0.2):
    """Per-sample PPO-Clip term min(r A, clip(r, 1 - eps, 1 + eps) A) (Schulman et al., 2017, Eq. 7)."""
    return np.minimum(ratio * advantage, np.clip(ratio, 1 - eps, 1 + eps) * advantage)


def ppo_clip_gradient(ratio, advantage, eps=0.2):
    """d/dr of ppo_clip_objective: A while the unclipped branch is active, 0 once the sample is clipped."""
    active = ratio < 1 + eps if advantage >= 0 else ratio > 1 - eps
    return np.where(active, advantage, 0.0)


def clip_is_active(advantage, ratio, eps_low, eps_high):
    """True while min(rA, clip(r, 1 - eps_low, 1 + eps_high) A) still has a gradient (DAPO Eq. 8).

    The objective goes flat once the ratio has moved past the bound in the direction the advantage favors:
    above 1 + eps_high for A >= 0, below 1 - eps_low for A < 0.
    """
    return (advantage >= 0 and ratio <= 1 + eps_high) or (advantage < 0 and ratio >= 1 - eps_low)


def bandit_grpo_step(logits, verifier, group_size, lr):
    """One Bandit-GRPO step on a softmax policy over K actions; logits shape (K,).

    Samples G actions, scores them, computes group-relative advantages and takes one gradient-ascent step.
    Returns (new_logits, info). A group with identical rewards returns the logits unchanged with
    info['zero_grad'] = True (failure mode 1). Mirrors the version written out in 01_grpo_core.ipynb.
    """
    probs = softmax(logits)
    actions = np.random.choice(len(logits), size=group_size, p=probs)
    rewards = np.array([verifier(a) for a in actions])
    base_info = {"reward_mean": rewards.mean(), "entropy": entropy(probs), "p_correct": probs[verifier.correct]}
    if rewards.std() < 1e-8:
        return logits, {**base_info, "grad_norm": 0.0, "zero_grad": True}
    advantages = (rewards - rewards.mean()) / rewards.std()
    # d log pi(a) / d logits = one_hot(a) - probs for a softmax policy
    grad = np.zeros_like(logits)
    for a, adv in zip(actions, advantages):
        one_hot = np.zeros_like(logits)
        one_hot[a] = 1.0
        grad += adv * (one_hot - probs)
    grad /= group_size
    logits = logits + lr * grad
    new_probs = softmax(logits)
    return logits, {"reward_mean": rewards.mean(), "entropy": entropy(new_probs),
                    "grad_norm": np.linalg.norm(grad), "zero_grad": False, "p_correct": new_probs[verifier.correct]}


def bandit_grpo_step_dynamic(logits, verifier, group_size, lr, max_attempts=5):
    """Bandit-GRPO with finite-retry Dynamic Sampling (DAPO Sec. 3.2); logits shape (K,).

    Resamples the whole group, up to max_attempts times, until its rewards differ. info['generated_responses']
    counts every sampled response, including discarded groups: that is the cost of the filter. Finite retries
    reduce zero-gradient updates but cannot guarantee their absence.
    """
    probs = softmax(logits)
    attempts = 0
    for _ in range(max_attempts):
        attempts += 1
        actions = np.random.choice(len(logits), size=group_size, p=probs)
        rewards = np.array([verifier(a) for a in actions])
        if rewards.std() > 1e-8:
            break
    info_base = {"reward_mean": rewards.mean(), "entropy": entropy(probs), "p_correct": probs[verifier.correct],
                 "attempts": attempts, "generated_responses": attempts * group_size}
    if rewards.std() < 1e-8:
        return logits, {**info_base, "zero_grad": True, "grad_norm": 0.0}
    advantages = (rewards - rewards.mean()) / rewards.std()
    grad = np.zeros_like(logits)
    for action, adv in zip(actions, advantages):
        one_hot = np.zeros_like(logits)
        one_hot[action] = 1.0
        grad += adv * (one_hot - probs) / group_size
    logits = logits + lr * grad
    new_probs = softmax(logits)
    return logits, {**info_base, "entropy": entropy(new_probs), "p_correct": new_probs[verifier.correct],
                    "zero_grad": False, "grad_norm": np.linalg.norm(grad)}


def zero_gradient_probability(p, group_size):
    """P(all G independent binary rewards are equal) = p^G + (1 - p)^G, with success probability p."""
    return p ** group_size + (1 - p) ** group_size


def measure_zero_grad_fraction(success_rate, group_size, n_trials=2000):
    """Monte Carlo estimate of the fraction of groups whose binary rewards are all equal."""
    zero_count = 0
    for _ in range(n_trials):
        rewards = (np.random.random(group_size) < success_rate).astype(float)
        if rewards.std() < 1e-8:
            zero_count += 1
    return zero_count / n_trials


def fixed_rollout_clip_toy(eps_low, eps_high, epochs=30, lr=0.8):
    """Optimize one frozen group on a 4-action softmax policy and record the actual PPO ratios.

    Action 0 is common (p ~ 0.92) and action 1 is rare (p ~ 0.05). The frozen group has five rewarded copies of
    action 0, the rewarded rare action 1 (index 5, the token tracked in the plots) and unrewarded actions 2 and 3.
    Ratios are taken against the frozen old policy, so all equal 1 at epoch 0 and clipping cannot bind yet.
    Returns one dict per epoch with 'probs' (4,), 'ratios' (8,) and 'clip_fraction'.
    """
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
        clipped = np.zeros(len(actions), dtype=bool)
        for j, (action, adv, ratio) in enumerate(zip(actions, advantages, ratios)):
            active = clip_is_active(adv, ratio, eps_low, eps_high)
            clipped[j] = not active
            if active:
                one_hot = np.zeros_like(logits)
                one_hot[action] = 1.0
                grad += adv * ratio * (one_hot - probs) / len(actions)
        history.append({"probs": probs.copy(), "ratios": ratios.copy(), "clip_fraction": clipped.mean()})
        logits += lr * grad
    return history


def token_weights(lengths, scheme, max_length=None):
    """Loss coefficient of each token of each response; lengths shape (G,) -> (G,).

    'sample'   1 / (G |o_i|)       GRPO's sample-level average (DeepSeekMath Eq. 3)
    'token'    1 / sum_j |o_j|     DAPO's token-level average (DAPO Eq. 8)
    'constant' 1 / (G max_length)  Dr. GRPO's constant normalizer
    """
    lengths = np.asarray(lengths, dtype=float)
    if scheme == "sample":
        return 1.0 / (len(lengths) * lengths)
    if scheme == "token":
        return np.full(len(lengths), 1.0 / lengths.sum())
    if scheme == "constant":
        return np.full(len(lengths), 1.0 / (len(lengths) * max_length))
    raise ValueError(f"unknown scheme {scheme!r}")


def overlong_penalty(length, l_max, l_cache):
    """DAPO soft overlong punishment (Eq. 13): 0 up to l_max - l_cache, linear to -1 at l_max, -1 beyond."""
    start = l_max - l_cache
    if length <= start:
        return 0.0
    if length >= l_max:
        return -1.0
    return -(length - start) / l_cache


def kl_gradient(probs, ref_probs):
    """Gradient of KL(pi || pi_ref) with respect to softmax logits: pi * (log pi - log pi_ref - KL); shape (K,)."""
    log_ratio = np.log(probs + 1e-12) - np.log(ref_probs + 1e-12)
    return probs * (log_ratio - np.sum(probs * log_ratio))


def sequence_grpo_step(policy, ref_policy, verifier, group_size, lr, clip_eps, kl_coeff,
                       token_level_loss=False, clip_higher=None, update_epochs=8):
    """Sample one group with pi_old, then take update_epochs clipped gradient steps on that fixed group.

    Implements DeepSeekMath Eq. 3 on the Level B toy: per-token ratios r_t = pi_theta / pi_old, the clipped
    surrogate with one group-relative advantage per sequence, and a KL penalty to pi_ref. The KL term uses the
    exact categorical KL rather than the Eq. 4 estimator because the toy can enumerate its vocabulary.

    token_level_loss: weight every token by 1 / sum_i |o_i| (DAPO) instead of 1 / (G |o_i|) (GRPO); the two
        coincide here because every sequence has the same length.
    clip_higher: upper clip epsilon (DAPO Clip-Higher); defaults to clip_eps, i.e. a symmetric range.
    Returns (policy, info) with the final-epoch clip fraction and largest ratio, and their per-epoch traces.
    """
    old_policy = policy.copy()
    upper_clip = clip_higher if clip_higher is not None else clip_eps
    sequences, old_log_probs, rewards = [], [], []
    for _ in range(group_size):
        seq, lps = old_policy.sample()
        sequences.append(seq)
        old_log_probs.append(lps)
        rewards.append(verifier(seq))
    rewards = np.asarray(rewards)
    base = {"reward_mean": rewards.mean(), "entropy": policy.get_entropy()}
    if rewards.std() < 1e-8:
        return policy, {**base, "zero_grad": True, "grad_norm": 0.0, "clip_fraction": 0.0, "max_ratio": 1.0}
    advantages = (rewards - rewards.mean()) / rewards.std()
    total_tokens = sum(len(seq) for seq in sequences)
    epoch_clip_fractions, epoch_max_ratios = [], []
    last_grad = np.zeros_like(policy.logits)
    for _ in range(update_epochs):
        grad = np.zeros_like(policy.logits)
        clipped, n_terms, max_ratio = 0, 0, 1.0
        for i, seq in enumerate(sequences):
            ratios = np.exp(policy.log_prob(seq) - old_log_probs[i])
            norm = 1.0 / total_tokens if token_level_loss else 1.0 / (group_size * len(seq))
            for t, (action, ratio) in enumerate(zip(seq, ratios)):
                active = clip_is_active(advantages[i], ratio, clip_eps, upper_clip)
                clipped += int(not active)
                n_terms += 1
                max_ratio = max(max_ratio, float(ratio))
                if active:
                    probs_t = softmax(policy.logits[t])
                    one_hot = np.zeros(policy.vocab_size)
                    one_hot[action] = 1.0
                    # d(r A) / d logits = A r (one_hot - probs), by the chain rule through log pi
                    grad[t] += norm * advantages[i] * ratio * (one_hot - probs_t)
        for t in range(policy.seq_length):
            grad[t] -= kl_coeff * kl_gradient(softmax(policy.logits[t]), softmax(ref_policy.logits[t]))
        policy.logits += lr * grad
        last_grad = grad
        epoch_clip_fractions.append(clipped / max(n_terms, 1))
        epoch_max_ratios.append(max_ratio)
    return policy, {**base, "entropy": policy.get_entropy(), "zero_grad": False,
                    "grad_norm": np.linalg.norm(last_grad), "n_correct": np.sum(rewards == 1.0),
                    "clip_fraction": epoch_clip_fractions[-1], "max_ratio": epoch_max_ratios[-1],
                    "epoch_clip_fractions": epoch_clip_fractions, "epoch_max_ratios": epoch_max_ratios}


def sigmoid(x):
    """Logistic function; a single token's probability when its logit is x."""
    return 1 / (1 + np.exp(-x))


def logit(p):
    """Inverse of sigmoid: log(p / (1 - p))."""
    return np.log(p / (1 - p))
