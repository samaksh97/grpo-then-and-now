"""Experiment runners for the failure lab: training loops and sweeps built from envs and objectives.

Each function seeds NumPy itself, so a notebook cell that calls it reproduces the published numbers.
"""
import numpy as np

from . import envs, objectives
from .objectives import softmax


def zero_grad_sweep(success_rates, group_sizes, n_trials=2000, seed=42):
    """Monte Carlo zero-gradient fraction for every (G, p); returns {G: list over success_rates}."""
    np.random.seed(seed)
    return {G: [objectives.measure_zero_grad_fraction(p, G, n_trials) for p in success_rates]
            for G in group_sizes}


def train_sequence_grpo(clip_eps_lower, clip_eps_upper, n_steps=100, token_level=False, seed=42,
                        group_size=16, update_epochs=20, cfg=None):
    """Train the shaped-reward Level B toy with fresh rollouts per batch; returns (history, policy)."""
    cfg = cfg or envs.Config()
    np.random.seed(seed)
    policy = envs.SequencePolicy(cfg.seq_length, cfg.seq_vocab_size)
    ref_policy = policy.copy()
    verifier = envs.MatchFractionVerifier(envs.SEQ_TARGET)
    history = []
    for _ in range(n_steps):
        policy, info = objectives.sequence_grpo_step(
            policy, ref_policy, verifier, group_size=group_size, lr=0.35, clip_eps=clip_eps_lower,
            kl_coeff=cfg.kl_coeff, token_level_loss=token_level, clip_higher=clip_eps_upper,
            update_epochs=update_epochs)
        history.append(info)
    return history, policy


def quartile_mass(n_batches=2000, group_size=16, seed=31415):
    """Mean objective mass of the shortest and longest quarter of responses in random variable-length batches.

    Lengths are uniform on 4..64 tokens. Returns {'sample': (short, long), 'token': (short, long)}.
    """
    rng = np.random.default_rng(seed)
    masses = {"sample": ([], []), "token": ([], [])}
    for _ in range(n_batches):
        lens = rng.integers(4, 65, size=group_size)
        order = np.argsort(lens)
        short, long = order[:group_size // 4], order[-group_size // 4:]
        for scheme, mass in [("sample", np.full(group_size, 1 / group_size)), ("token", lens / lens.sum())]:
            masses[scheme][0].append(mass[short].sum())
            masses[scheme][1].append(mass[long].sum())
    return {scheme: (np.mean(s), np.mean(l)) for scheme, (s, l) in masses.items()}


def dynamic_sampling_sweep(success_rates, n_reps=500, group_size=8, max_attempts=5, seed=2718):
    """Zero-gradient fraction with one attempt versus up to max_attempts, and the responses generated.

    Returns (vanilla_zero_fraction, dynamic_zero_fraction, responses_per_requested_update), one per rate.
    """
    np.random.seed(seed)
    vanilla, dynamic, cost = [], [], []
    for sr in success_rates:
        vz = dz = attempts_total = 0
        for _ in range(n_reps):
            for attempt in range(1, max_attempts + 1):
                rewards = (np.random.random(group_size) < sr).astype(float)
                if rewards.std() > 1e-8:
                    break
            attempts_total += attempt
            dz += rewards.std() < 1e-8
            one = (np.random.random(group_size) < sr).astype(float)
            vz += one.std() < 1e-8
        vanilla.append(vz / n_reps)
        dynamic.append(dz / n_reps)
        cost.append(attempts_total * group_size / n_reps)
    return vanilla, dynamic, cost


def reward_hacking_run(n_steps=300, cfg=None):
    """Train the Level B toy against envs.FlawedVerifier; returns (history, policy)."""
    cfg = cfg or envs.Config()
    np.random.seed(cfg.seed)
    flawed = envs.FlawedVerifier(envs.SEQ_TARGET)
    policy = envs.SequencePolicy(cfg.seq_length, cfg.seq_vocab_size)
    ref_policy = policy.copy()
    history = []
    for _ in range(n_steps):
        policy, info = objectives.sequence_grpo_step(policy, ref_policy, flawed, group_size=16, lr=cfg.seq_lr,
                                                     clip_eps=cfg.clip_eps, kl_coeff=cfg.kl_coeff)
        history.append(info)
    return history, policy


def drgrpo_toy(group_size=4, max_length=1000, lengths=(25, 50, 100, 200, 400, 800)):
    """Dr. GRPO's two normalization changes on a group with one correct and three incorrect responses.

    Returns a dict with the per-question weights in expectation over p, both advantages of an incorrect
    response, and its per-token weight at each length under GRPO (1/(G|o_i|)) and Dr. GRPO (1/(G L_max)).
    """
    p = np.linspace(0.001, 0.999, 400)
    rewards = np.array([1.0] + [0.0] * (group_size - 1))
    adv_grpo = (rewards - rewards.mean()) / rewards.std()
    adv_drgrpo = rewards - rewards.mean()
    lengths = np.array(lengths)
    return {"p": p,
            "weights": {r"GRPO: $\sqrt{p(1-p)}$": np.sqrt(p * (1 - p)), r"Dr. GRPO: $p(1-p)$": p * (1 - p),
                        "DisCO: constant": np.ones_like(p)},
            "adv_grpo": adv_grpo[1], "adv_drgrpo": adv_drgrpo[1], "lengths": lengths,
            "per_token_grpo": adv_grpo[1] / (group_size * lengths),
            "per_token_drgrpo": np.full(len(lengths), adv_drgrpo[1] / (group_size * max_length))}


def policy_table(policy):
    """Action probabilities per position, shape (seq_length, vocab_size)."""
    return np.array([softmax(policy.logits[t]) for t in range(policy.seq_length)])
