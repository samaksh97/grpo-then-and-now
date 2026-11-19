"""The method testbed: presets differ only in their flags, gradients match finite differences, runs are
deterministic, and every flag is mapped to the training libraries."""
import numpy as np
import pytest

from grpo_lab import testbed
from grpo_lab.testbed import Config, ThinkThenAnswer


def test_presets_differ_only_in_the_expected_flags():
    assert {f for f, _, _ in testbed.diff("GRPO", "DAPO")} == {"aggregation", "eps_high", "dynamic_sampling", "overlong"}
    assert {f for f, _, _ in testbed.diff("GRPO", "Dr. GRPO")} == {"adv_norm", "aggregation"}
    assert {f for f, _, _ in testbed.diff("GRPO", "GSPO")} == {"ratio_level", "eps_low", "eps_high"}


def batch_and_logits(cfg, seed=3):
    env = ThinkThenAnswer()
    rng = np.random.default_rng(seed)
    logits = env.init_logits() + 0.3 * rng.standard_normal(env.init_logits().shape)
    batch, _ = testbed.collect(env, logits, cfg, rng)
    moved = logits + 0.02 * rng.standard_normal(logits.shape)   # ratios move away from 1, inside the clip range
    return env, batch, moved


@pytest.mark.parametrize("cfg", [Config(), Config(aggregation="token"), Config(adv_norm="none", aggregation="constant"),
                                 Config(ratio_level="sequence")], ids=["grpo", "token", "drgrpo", "sequence"])
def test_gradient_matches_finite_differences(cfg):
    env, batch, logits = batch_and_logits(cfg)
    _, grad, clipped = testbed.objective(logits, batch, cfg, env.max_len)
    assert clipped == 0
    h, numeric = 1e-6, np.zeros_like(logits)
    for idx in np.ndindex(logits.shape):
        up, down = logits.copy(), logits.copy()
        up[idx] += h
        down[idx] -= h
        numeric[idx] = (testbed.objective(up, batch, cfg, env.max_len)[0]
                        - testbed.objective(down, batch, cfg, env.max_len)[0]) / (2 * h)
    assert np.allclose(grad, numeric, atol=1e-6)


def test_cispo_equals_token_level_grpo_on_fresh_rollouts():
    cfg = testbed.PRESETS["CISPO"]
    env, batch, _ = batch_and_logits(cfg)
    logits = ThinkThenAnswer().init_logits()                  # ratios are exactly 1 before any update
    rng = np.random.default_rng(0)
    batch, _ = testbed.collect(env, logits, cfg, rng)
    g_cispo = testbed.objective(logits, batch, cfg, env.max_len)[1]
    g_token = testbed.objective(logits, batch, Config(aggregation="token"), env.max_len)[1]
    assert np.allclose(g_cispo, g_token)


def test_runs_are_deterministic_and_dynamic_sampling_costs_generation():
    a = testbed.run(testbed.PRESETS["DAPO"], steps=20, seed=1)
    b = testbed.run(testbed.PRESETS["DAPO"], steps=20, seed=1)
    assert np.array_equal(a["acc"], b["acc"])
    grpo = testbed.run(testbed.PRESETS["GRPO"], steps=20, seed=1)
    assert a["generated"].sum() > grpo["generated"].sum() == 20 * 5 * 8


def test_every_method_flag_has_a_library_mapping():
    mapped = " ".join(testbed.LIBRARY_SETTINGS)
    assert all(flag in mapped for flag in testbed.METHOD_FLAGS)
