"""Unit tests for grpo_lab: the numbers quoted in the decks, notebooks and figures must stay true."""
import numpy as np
import pytest

from grpo_lab import data, envs, objectives


def test_two_of_eight_group_matches_figure_f3():
    adv = objectives.group_advantages([1, 0, 0, 0, 0, 0, 1, 0])
    assert adv[0] == pytest.approx(1.7320508, abs=1e-6)
    assert adv[1] == pytest.approx(-0.5773503, abs=1e-6)
    assert adv.sum() == pytest.approx(0.0, abs=1e-9)


def test_identical_rewards_give_zero_advantages():
    assert np.allclose(objectives.group_advantages([1.0] * 8), 0.0)


@pytest.mark.parametrize("ratio, advantage, expected", [
    (1.10, +1.0, 1.0), (1.30, +1.0, 0.0),     # positive advantage: clipped above 1 + eps
    (0.90, -1.0, -1.0), (0.70, -1.0, 0.0),    # negative advantage: clipped below 1 - eps
])
def test_ppo_clip_gradient(ratio, advantage, expected):
    assert float(objectives.ppo_clip_gradient(np.array(ratio), advantage)) == expected


def test_clip_higher_only_moves_the_upper_bound():
    assert objectives.clip_is_active(+1.0, 1.25, eps_low=0.2, eps_high=0.28)
    assert not objectives.clip_is_active(+1.0, 1.25, eps_low=0.2, eps_high=0.2)
    assert not objectives.clip_is_active(-1.0, 0.75, eps_low=0.2, eps_high=0.28)


def test_bandit_grpo_learns_the_passing_kernel():
    cfg = envs.Config()
    verifier = envs.BanditVerifier(data.matmul_task()["candidates"], correct_action=cfg.bandit_correct_action)
    np.random.seed(cfg.seed)
    logits = np.zeros(cfg.bandit_n_actions)
    for _ in range(cfg.bandit_n_steps):
        logits, info = objectives.bandit_grpo_step(logits, verifier, cfg.bandit_group_size, cfg.bandit_lr)
    assert info["p_correct"] == pytest.approx(0.997, abs=5e-4)


def test_sequence_grpo_reproduces_notebook_numbers():
    cfg = envs.Config()
    np.random.seed(cfg.seed)
    policy = envs.SequencePolicy(cfg.seq_length, cfg.seq_vocab_size)
    ref, verifier = policy.copy(), envs.MatchFractionVerifier(envs.SEQ_TARGET)
    history = []
    for _ in range(120):
        policy, info = objectives.sequence_grpo_step(policy, ref, verifier, cfg.seq_group_size, 0.35,
                                                     cfg.clip_eps, cfg.kl_coeff, update_epochs=8)
        history.append(info)
    assert np.mean([h["clip_fraction"] > 0 for h in history]) == pytest.approx(0.383, abs=5e-4)
    target_prob = np.mean([objectives.softmax(policy.logits[t])[envs.SEQ_TARGET[t]] for t in range(cfg.seq_length)])
    assert target_prob == pytest.approx(0.993, abs=5e-4)
