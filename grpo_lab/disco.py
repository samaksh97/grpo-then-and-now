"""The one-question DisCO toy from disco_three_fixes.ipynb Section 4 (Li et al., NeurIPS 2025).

A softmax policy over K = 6 answers: index 0 is correct, index 1 a plausible confuser, the rest filler.
`train` runs the outer loop (sample a group from pi_old, take `inner` updates under DisCO's squared-hinge
KL trust region) for any gradient function, so the notebook's own disco_grad / grpo_grad are what run.
"""
import numpy as np

from .objectives import softmax

K, CORRECT, CONFUSER = 6, 0, 1


def init_logits(p_correct, p_confuser):
    """Logits, shape (K,), for one question: the correct answer, a plausible confuser, filler elsewhere."""
    filler = (1 - p_correct - p_confuser) / (K - 2)
    return np.log(np.array([p_correct, p_confuser] + [filler] * (K - 2)))


def grad_logp(probs, o):
    """d log pi(o) / d theta = onehot(o) - pi, shape (K,)."""
    g = -probs.copy()
    g[o] += 1.0
    return g


def split(rollouts):
    """(positive rollouts, negative rollouts) of one group of answer indices."""
    return rollouts[rollouts == CORRECT], rollouts[rollouts != CORRECT]


def kl_old_new(p_old, theta):
    """KL(pi_old || pi_theta) for the categorical policy."""
    return float(np.sum(p_old * (np.log(p_old) - np.log(softmax(theta)))))


def train(grad_fn, p0, p_conf, steps=150, G=16, lr=0.1, delta=0.01, beta=200.0, inner=2, seed=0):
    """Outer loop with the squared-hinge trust region; grad_fn(theta, rollouts) returns the ascent direction.

    Returns per-step arrays (P(correct), KL(pi_old || pi_theta), P(confuser)).
    """
    rng = np.random.default_rng(seed)
    theta = init_logits(p0, p_conf)
    acc, kls, conf = [], [], []
    for _ in range(steps):
        p_old = softmax(theta)
        rollouts = rng.choice(K, size=G, p=p_old)
        for _ in range(inner):
            g = grad_fn(theta, rollouts)
            # squared-hinge trust region: silent inside, restoring force outside
            violation = max(kl_old_new(p_old, theta) - delta, 0.0)
            g = g - 2 * beta * violation * (softmax(theta) - p_old)
            theta = theta + lr * g
        probs = softmax(theta)
        acc.append(probs[CORRECT])
        conf.append(probs[CONFUSER])
        kls.append(kl_old_new(p_old, theta))
    return np.array(acc), np.array(kls), np.array(conf)


def steps_to(acc, target=0.9):
    """First step at which accuracy reaches target, or None if it never does."""
    i = int(np.argmax(acc >= target))
    return i if acc[i] >= target else None
