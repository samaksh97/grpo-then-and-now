"""Toy verifiable-reward environments used throughout the notebooks.

Level A (bandit): pick one of the 8 candidate kernels for KernelBench Level 1 problem 1. The reward is the
candidate's recorded outcome in data/kernelbench/tasks.json (real task specification, illustrative
metrics): 1 if it compiles, matches PyTorch and is more than 1.2x faster, else 0.

Level B (sequence): emit a 4-stage kernel pipeline one token per position. A verifier scores the sequence.

All randomness uses NumPy's global generator (np.random), so np.random.seed(42) reproduces the notebooks.
"""
import numpy as np

from .objectives import entropy, softmax

SEQ_VOCAB = ["load_coalesced", "sync_store", "alloc_shared", "busy_wait", "tile_compute", "assert_check"]
SEQ_TARGET = [2, 0, 4, 1]   # alloc_shared -> load_coalesced -> tile_compute -> sync_store


class Config:
    """Default hyperparameters shared by the experiments; change a value here to change it everywhere."""
    seed = 42
    # Level A: kernel-configuration bandit
    bandit_n_actions = 8
    bandit_correct_action = 3      # Tile 32x32 + shared memory, the only candidate with reward 1
    bandit_group_size = 8          # G, rollouts per prompt
    bandit_lr = 0.5
    bandit_n_steps = 200
    # Level B: 4-stage pipeline
    seq_vocab_size = len(SEQ_VOCAB)
    seq_length = len(SEQ_TARGET)
    seq_group_size = 16
    seq_lr = 0.1
    seq_n_steps = 300
    # GRPO
    clip_eps = 0.2                 # epsilon in clip(r, 1 - eps, 1 + eps)
    kl_coeff = 0.01                # beta, the KL penalty coefficient


def seq_label(seq):
    """Readable stage names for a sequence of token indices."""
    return " -> ".join(SEQ_VOCAB[t] for t in seq)


class BanditVerifier:
    """Reads each candidate kernel's recorded outcome (models compile + correctness + speedup checks).

    noise_prob flips the reward with that probability, to model a noisy verifier.
    """

    def __init__(self, candidates, correct_action=3, noise_prob=0.0):
        self.candidates = candidates
        self.correct = correct_action
        self.noise_prob = noise_prob

    def __call__(self, action):
        reward = float(self.candidates[action]["reward"])
        if np.random.random() < self.noise_prob:
            reward = 1.0 - reward
        return reward

    def success_rate(self, probs):
        """Expected reward of a policy with action probabilities probs, shape (K,)."""
        return probs[self.correct]


class SequenceVerifier:
    """Binary verifier: 1.0 if the sequence equals the target pipeline exactly, else 0.0."""

    def __init__(self, target_sequence):
        self.target = np.array(target_sequence)

    def __call__(self, sequence):
        return 1.0 if np.array_equal(sequence, self.target) else 0.0


class MatchFractionVerifier:
    """Shaped verifier for teaching: the fraction of pipeline positions that match the target."""

    def __init__(self, target_sequence):
        self.target = np.asarray(target_sequence)

    def __call__(self, sequence):
        sequence = np.asarray(sequence)
        return float(len(sequence) == len(self.target) and np.mean(sequence == self.target))


class FlawedVerifier:
    """A deliberately exploitable verifier for the reward-hacking demo.

    Intended behavior: reward 1 for the exact target pipeline. Bug: any pipeline that contains the stage
    busy_wait (token 3) also earns 0.8, so the policy can learn the shortcut instead of the task.
    """

    def __init__(self, target_sequence, shortcut_token=3, shortcut_reward=0.8):
        self.target = np.array(target_sequence)
        self.shortcut_token = shortcut_token
        self.shortcut_reward = shortcut_reward

    def __call__(self, sequence):
        if np.array_equal(sequence, self.target):
            return 1.0
        return self.shortcut_reward if self.shortcut_token in sequence else 0.0


class SequencePolicy:
    """Per-position categorical policy with logits of shape (seq_length, vocab_size).

    Unlike a real language model, a position does not condition on earlier tokens; this keeps per-token
    probabilities and ratios easy to inspect.
    """

    def __init__(self, seq_length, vocab_size):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.logits = np.zeros((seq_length, vocab_size))

    def sample(self):
        """Sample one sequence; return (tokens, per-token log-probs), both of shape (seq_length,)."""
        seq, log_probs = [], []
        for t in range(self.seq_length):
            probs = softmax(self.logits[t])
            action = np.random.choice(self.vocab_size, p=probs)
            seq.append(action)
            log_probs.append(np.log(probs[action] + 1e-10))
        return np.array(seq), np.array(log_probs)

    def log_prob(self, sequence):
        """Per-token log-probabilities of a given sequence under the current logits, shape (seq_length,)."""
        return np.array([np.log(softmax(self.logits[t])[sequence[t]] + 1e-10) for t in range(self.seq_length)])

    def get_entropy(self):
        """Mean per-position entropy (nats)."""
        return np.mean([entropy(softmax(self.logits[t])) for t in range(self.seq_length)])

    def copy(self):
        """A frozen copy, used as pi_old (behavior policy) or pi_ref (KL anchor)."""
        clone = SequencePolicy(self.seq_length, self.vocab_size)
        clone.logits = self.logits.copy()
        return clone


def describe_bandit(candidates, correct_action):
    """Print the Level A candidates with their illustrative latency, speedup and reward."""
    print(f"Level A: {len(candidates)} candidate kernels for KernelBench L1-P1 (square matmul)")
    for i, c in enumerate(candidates):
        marker = "  <-- target" if i == correct_action else ""
        print(f"  [{i}] {c['name']:<44} {c['latency_us']:>6.1f} us  {c['speedup']:>4.2f}x  "
              f"reward={c['reward']}{marker}")


def describe_sequence():
    """Print the Level B vocabulary and target pipeline."""
    print(f"Level B: {len(SEQ_TARGET)}-stage pipeline over {len(SEQ_VOCAB)} stages {SEQ_VOCAB}")
    print(f"  target: {seq_label(SEQ_TARGET)}")
