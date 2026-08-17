import numpy as np

from rlx.exploration.epsilon_greedy import EpsilonGreedy


def test_epsilon_follows_the_documented_schedule(cfg, rng):
    e = EpsilonGreedy(cfg, rng)
    decay_end = int(cfg.epsilon_decay_frac * cfg.total_steps)   # 2000

    # np.isclose, not ==: the schedule is a float interpolation, so the endpoint
    # lands on 0.050000000000000044 rather than exactly 0.05.
    assert np.isclose(e.epsilon(0), cfg.epsilon_start)
    assert np.isclose(e.epsilon(decay_end), cfg.epsilon_end)
    assert np.isclose(e.epsilon(cfg.total_steps), cfg.epsilon_end)
    assert np.isclose(e.epsilon(decay_end // 2),
                      (cfg.epsilon_start + cfg.epsilon_end) / 2)


def test_epsilon_never_leaves_its_bounds(cfg, rng):
    e = EpsilonGreedy(cfg, rng)
    for step in range(0, cfg.total_steps + 1, 137):
        assert cfg.epsilon_end <= e.epsilon(step) <= cfg.epsilon_start


def test_acts_greedily_when_epsilon_has_decayed(cfg, rng, q_values, key):
    e = EpsilonGreedy(cfg, rng)
    e.cfg.epsilon_end = 0.0
    assert e.act(q_values, key, cfg.total_steps) == 3


def test_acts_almost_uniformly_at_the_very_start(cfg, rng, q_values, key):
    e = EpsilonGreedy(cfg, rng)
    counts = np.bincount([e.act(q_values, key, 0) for _ in range(3000)], minlength=7)
    assert counts.min() > 200, f"not uniform enough: {counts}"


def test_returns_a_valid_action_at_every_stage(cfg, rng, q_values, key):
    e = EpsilonGreedy(cfg, rng)
    for step in (0, 500, 2000, 9999):
        a = e.act(q_values, key, step)
        assert isinstance(a, int)
        assert 0 <= a < len(q_values)


def test_reports_epsilon_for_logging(cfg, rng, q_values, key):
    e = EpsilonGreedy(cfg, rng)
    e.act(q_values, key, 0)
    assert "epsilon" in e.stats()


def test_adds_no_intrinsic_bonus(cfg, rng, key):
    assert EpsilonGreedy(cfg, rng).intrinsic_bonus(key) == 0.0


def test_is_reproducible_for_a_fixed_rng_seed(cfg, q_values, key):
    def actions():
        e = EpsilonGreedy(cfg, np.random.default_rng(42))
        return [e.act(q_values, key, 100) for _ in range(50)]

    assert actions() == actions()
