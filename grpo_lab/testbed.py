"""One configurable GRPO-family objective, so that every method in the guide is a setting of the same flags.

The toy ("think, then answer"): for prompt q the policy emits "think" tokens until it emits "answer"; the
answer is correct only after at least d_q think tokens, and a response that never answers is truncated at
max_len. This gives what the length and difficulty failure modes need: variable lengths, truncation, and
prompts of different difficulty. The policy is tabular: logits[q, t] is the logit of "answer" at position t.

Flags (Config) and the papers they come from:
    adv_norm          "std" (GRPO, DeepSeekMath Sec. 4.1.2) | "none" (Dr. GRPO: subtract the mean only)
    aggregation       "sample" 1/(N|o_i|) (GRPO, DeepSeekMath Eq. 3) | "token" 1/sum_j|o_j| (DAPO Eq. 8) |
                      "constant" 1/(N L_max) (Dr. GRPO); N = responses in the batch
    eps_low, eps_high clip range 1 - eps_low .. 1 + eps_high (DAPO's Clip-Higher raises eps_high)
    ratio_level       "token" | "sequence" (GSPO: one length-normalized ratio per response, clipped once)
    clip_mode         "drop" (PPO/GRPO: a clipped term has zero gradient) | "is_weight" (CISPO: clip the
                      importance weight, stop its gradient, keep every token's log-prob gradient)
    dynamic_sampling  resample groups whose rewards are all equal, up to max_attempts (DAPO Sec. 3.2)
    overlong          "none" | "filter" (truncated responses carry no loss) | "shape" (DAPO Eq. 13 penalty)
LIBRARY_SETTINGS maps each flag to the corresponding TRL, verl and OpenRLHF options.
"""
from dataclasses import dataclass, fields, replace

import numpy as np

from .objectives import overlong_penalty, sigmoid


@dataclass(frozen=True)
class Config:
    adv_norm: str = "std"
    aggregation: str = "sample"
    eps_low: float = 0.2
    eps_high: float = 0.2
    ratio_level: str = "token"
    clip_mode: str = "drop"
    dynamic_sampling: bool = False
    overlong: str = "none"
    # training settings shared by every method (not method choices)
    group_size: int = 8
    update_epochs: int = 4
    lr: float = 0.15
    max_attempts: int = 4


PRESETS = {
    "GRPO": Config(),
    "DAPO": Config(aggregation="token", eps_high=0.28, dynamic_sampling=True, overlong="shape"),
    "Dr. GRPO": Config(adv_norm="none", aggregation="constant"),
    "GSPO": Config(ratio_level="sequence", eps_low=3e-4, eps_high=4e-4),
    "CISPO": Config(aggregation="token", clip_mode="is_weight", eps_low=1.0, eps_high=4.0),
}
METHOD_FLAGS = ["adv_norm", "aggregation", "eps_low", "eps_high", "ratio_level", "clip_mode",
                "dynamic_sampling", "overlong"]


@dataclass(frozen=True)
class ThinkThenAnswer:
    difficulties: tuple = (1, 2, 3, 5, 7)   # d_q: think tokens needed before a correct answer
    max_len: int = 12                       # L_max: a response that never answers is truncated here
    cache: int = 4                          # L_cache: width of the overlong-shaping window
    init_logit: float = -1.0                # initial logit of "answer" at every position (p = 0.27)

    def init_logits(self):
        return np.full((len(self.difficulties), self.max_len), self.init_logit)

    def sample(self, logits, q, rng):
        """One response for prompt q: dict with actions (1 = answer), old log-probs, length, correctness."""
        actions, logps = [], []
        for t in range(self.max_len):
            p = sigmoid(logits[q, t])
            a = int(rng.random() < p)
            actions.append(a)
            logps.append(np.log(p if a else 1 - p))
            if a:
                break
        answered = actions[-1] == 1
        length = len(actions)
        return {"q": q, "actions": np.array(actions), "old_logp": np.array(logps), "length": length,
                "truncated": not answered, "correct": answered and (length - 1) >= self.difficulties[q]}

    def exact_metrics(self, logits):
        """Per prompt, from the policy itself: P(correct), expected length, P(truncated), mean token entropy."""
        out = {"acc": [], "length": [], "truncated": [], "entropy": []}
        for q, d in enumerate(self.difficulties):
            p = sigmoid(logits[q])
            reach = np.concatenate([[1.0], np.cumprod(1 - p)[:-1]])     # P(the response reaches position t)
            first_answer = reach * p                                      # P(the first answer is at position t)
            ent = -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))
            out["acc"].append(first_answer[d:].sum())
            out["length"].append((reach).sum())
            out["truncated"].append(np.prod(1 - p))
            out["entropy"].append((reach * ent).sum() / reach.sum())
        return {k: np.array(v) for k, v in out.items()}


def advantages(rewards, adv_norm):
    """Group-relative advantages; rewards shape (G,) -> (G,). An all-equal group gets zeros."""
    centred = rewards - rewards.mean()
    if adv_norm == "none":
        return centred
    std = rewards.std()
    return centred / std if std > 1e-8 else np.zeros_like(rewards)


def token_weights(batch, cfg, max_len):
    """Per-response loss weight of each of its tokens under cfg.aggregation (0 for filtered responses)."""
    n = len(batch)
    total = sum(r["length"] for r in batch)
    weights = []
    for r in batch:
        if cfg.overlong == "filter" and r["truncated"]:
            weights.append(0.0)
        elif cfg.aggregation == "sample":
            weights.append(1.0 / (n * r["length"]))
        elif cfg.aggregation == "token":
            weights.append(1.0 / total)
        else:
            weights.append(1.0 / (n * max_len))
    return weights


def objective(logits, batch, cfg, max_len):
    """Value and gradient (w.r.t. logits) of the configured clipped surrogate, plus the clipped fraction.

    Token level, drop:  sum_i w_i sum_t min(r_t A_i, clip(r_t) A_i)            (DeepSeekMath Eq. 3, DAPO Eq. 8)
    Token level, CISPO: sum_i w_i sum_t sg(clip(r_t)) A_i log pi_t             (MiniMax-M1)
    Sequence (GSPO):    sum_i (1/N) min(s_i A_i, clip(s_i) A_i), s_i = exp(mean_t log r_t)
    """
    value, grad, clipped, units = 0.0, np.zeros_like(logits), 0, 0
    lo, hi = 1 - cfg.eps_low, 1 + cfg.eps_high
    for r, w in zip(batch, token_weights(batch, cfg, max_len)):
        q, A, T = r["q"], r["adv"], r["length"]
        p = sigmoid(logits[q, :T])
        logp = np.where(r["actions"] == 1, np.log(p), np.log(1 - p))
        dlogp = r["actions"] - p                         # d log pi(a_t) / d logit_t for a Bernoulli policy
        if cfg.ratio_level == "sequence":
            s = np.exp(np.mean(logp - r["old_logp"]))
            active = (A >= 0 and s <= hi) or (A < 0 and s >= lo)
            value += min(s * A, np.clip(s, lo, hi) * A) / len(batch)
            if active:
                grad[q, :T] += A * s * dlogp / (T * len(batch))
            clipped, units = clipped + (not active), units + 1
            continue
        ratio = np.exp(logp - r["old_logp"])
        if cfg.clip_mode == "is_weight":
            weight = np.clip(ratio, lo, hi)              # treated as a constant: no gradient through it
            value += w * np.sum(weight * A * logp)
            grad[q, :T] += w * weight * A * dlogp
            clipped += int(np.sum(weight != ratio))
        else:
            active = np.where(A >= 0, ratio <= hi, ratio >= lo)
            value += w * np.sum(np.minimum(ratio * A, np.clip(ratio, lo, hi) * A))
            grad[q, :T] += w * np.where(active, A * ratio * dlogp, 0.0)
            clipped += int(np.sum(~active))
        units += T
    return value, grad, clipped / max(units, 1)


def collect(env, logits, cfg, rng):
    """Sample one group per prompt (Dynamic Sampling resamples all-equal groups); returns batch and stats."""
    batch, generated, zero_groups = [], 0, 0
    for q in range(len(env.difficulties)):
        for attempt in range(cfg.max_attempts if cfg.dynamic_sampling else 1):
            group = [env.sample(logits, q, rng) for _ in range(cfg.group_size)]
            generated += cfg.group_size
            rewards = np.array([float(g["correct"]) for g in group])
            if cfg.overlong == "shape":
                rewards += [overlong_penalty(g["length"], env.max_len, env.cache) for g in group]
            if rewards.std() > 1e-8:
                break
        zero_groups += rewards.std() <= 1e-8
        if cfg.dynamic_sampling and rewards.std() <= 1e-8:
            continue                                     # still uninformative after max_attempts: drop it
        for g, a in zip(group, advantages(rewards, cfg.adv_norm)):
            g["adv"] = a
        batch.extend(group)
    return batch, {"generated": generated, "zero_groups": zero_groups}


def run(cfg, env=ThinkThenAnswer(), steps=150, seed=0):
    """Train from the same initial policy and seed; returns per-step metrics (exact, from the policy)."""
    rng = np.random.default_rng(seed)
    logits = env.init_logits()
    hist = {k: [] for k in ["acc", "acc_by_prompt", "length", "truncated", "entropy", "generated",
                            "zero_groups", "clip_fraction"]}
    for _ in range(steps):
        m = env.exact_metrics(logits)
        batch, stats = collect(env, logits, cfg, rng)
        clip_fraction = 0.0
        for _ in range(cfg.update_epochs if batch else 0):
            _, grad, clip_fraction = objective(logits, batch, cfg, env.max_len)
            norm = np.linalg.norm(grad)
            # Normalized step: methods differ in gradient *scale* (Dr. GRPO drops the std and divides by
            # L_max), which real optimizers largely absorb; comparing directions keeps the comparison fair.
            if norm > 1e-12:
                logits = logits + cfg.lr * grad / norm
        hist["acc"].append(m["acc"].mean())
        hist["acc_by_prompt"].append(m["acc"])
        hist["length"].append(m["length"].mean())
        hist["truncated"].append(m["truncated"].mean())
        hist["entropy"].append(m["entropy"].mean())
        hist["generated"].append(stats["generated"])
        hist["zero_groups"].append(stats["zero_groups"])
        hist["clip_fraction"].append(clip_fraction)
    return {k: np.array(v) for k, v in hist.items()}


def compare(methods, env=ThinkThenAnswer(), steps=150, seed=0):
    """Run several presets (names) or Configs on the same seed; returns {name: history}."""
    configs = {m if isinstance(m, str) else f"custom {i}": PRESETS[m] if isinstance(m, str) else m
               for i, m in enumerate(methods)}
    return {name: run(cfg, env, steps, seed) for name, cfg in configs.items()}


def diff(a, b):
    """The method flags on which two configs (or preset names) differ: [(flag, value_a, value_b)]."""
    a, b = (PRESETS[x] if isinstance(x, str) else x for x in (a, b))
    return [(f, getattr(a, f), getattr(b, f)) for f in METHOD_FLAGS if getattr(a, f) != getattr(b, f)]


def presets_table():
    """Markdown table: one row per preset, one column per method flag."""
    rows = ["| method | " + " | ".join(METHOD_FLAGS) + " |", "|---" * (len(METHOD_FLAGS) + 1) + "|"]
    for name, cfg in PRESETS.items():
        rows.append(f"| {name} | " + " | ".join(str(getattr(cfg, f)) for f in METHOD_FLAGS) + " |")
    return "\n".join(rows)


# Verified against the TRL (main), verl (latest) and OpenRLHF (main) documentation in October 2026.
LIBRARY_SETTINGS = {
    "adv_norm": ("`scale_rewards=\"group\"` (default) / `\"none\"`",
                 "`algorithm.norm_adv_by_std_in_grpo=True` / `False`",
                 "`--algo.advantage.estimator group_norm` / `dr_grpo`"),
    "aggregation": ("`loss_type=\"grpo\"` / `\"dapo\"` (default) / `\"dr_grpo\"`",
                    "`actor.loss_agg_mode=\"seq-mean-token-mean\"` / `\"token-mean\"` (default) / "
                    "`\"seq-mean-token-sum-norm\"`",
                    "`--actor.loss_agg_mode seq-mean-token-mean` / `token-mean` (default) / —"),
    "eps_low, eps_high": ("`epsilon`, `epsilon_high`", "`actor.clip_ratio_low`, `actor.clip_ratio_high`",
                          "not listed in the README"),
    "ratio_level": ("`importance_sampling_level=\"token\"` / `\"sequence\"`",
                    "`actor.policy_loss.loss_mode=\"gspo\"`", "`--actor.policy_loss_type gspo`"),
    "clip_mode": ("`loss_type=\"cispo\"` (with `epsilon_high`)", "proposed in verl PR #2076; check your version",
                  "—"),
    "dynamic_sampling": ("no built-in flag", "`algorithm.filter_groups.enable=True` (DAPO recipe)",
                         "`--algo.dynamic_filtering_enable`"),
    "overlong = filter": ("`mask_truncated_completions=True`", "not implemented in the DAPO recipe", "—"),
    "overlong = shape": ("reward `trl.rewards.get_soft_overlong_punishment(max_completion_len, soft_punish_cache)`",
                         "`reward_model.overlong_buffer.enable=True` (+ `len`, `penalty_factor`)",
                         "`--reward.overlong_buffer_len`, `--reward.overlong_penalty_factor`"),
    "group_size": ("`num_generations`", "`actor_rollout_ref.rollout.n`", "`--rollout.n_samples_per_prompt`"),
    "update_epochs": ("`num_iterations`", "`actor.ppo_epochs`", "—"),
}


def library_table():
    """Markdown table mapping each testbed flag to TRL GRPOConfig, verl and OpenRLHF settings."""
    rows = ["| testbed flag | TRL `GRPOConfig` | verl (`actor_rollout_ref.` ...) | OpenRLHF |", "|---|---|---|---|"]
    rows += [f"| `{flag}` | {trl} | {verl} | {orl} |" for flag, (trl, verl, orl) in LIBRARY_SETTINGS.items()]
    return "\n".join(rows)


def with_flags(base, **flags):
    """A preset (name or Config) with some flags changed, e.g. with_flags("GRPO", eps_high=0.28)."""
    return replace(PRESETS[base] if isinstance(base, str) else base, **flags)


assert set(METHOD_FLAGS) <= {f.name for f in fields(Config)}
