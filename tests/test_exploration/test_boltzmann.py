import numpy as np

from rlx.exploration.boltzmann import Boltzmann


def test_temperature_follows_the_documented_schedule(cfg, rng):
    b = Boltzmann(cfg, rng)
    decay_end = int(cfg.tau_decay_frac * cfg.total_steps)   # 4000

    assert np.isclose(b.temperature(0), cfg.tau_start)
    assert np.isclose(b.temperature(decay_end), cfg.tau_end)
    assert np.isclose(b.temperature(cfg.total_steps), cfg.tau_end)


def test_temperature_decreases_monotonically(cfg, rng):
    b = Boltzmann(cfg, rng)
    taus = [b.temperature(s) for s in range(0, cfg.total_steps, 100)]
    assert all(a >= b_ for a, b_ in zip(taus, taus[1:]))


def test_low_temperature_is_almost_greedy(cfg, rng, q_values, key):
    b = Boltzmann(cfg, rng)
    b.cfg.tau_end = 1e-3
    counts = np.bincount([b.act(q_values, key, cfg.total_steps) for _ in range(500)],
                         minlength=7)
    assert counts.argmax() == 3
    assert counts[3] > 480


def test_high_temperature_is_close_to_uniform(cfg, rng, q_values, key):
    b = Boltzmann(cfg, rng)
    b.cfg.tau_start = 100.0
    counts = np.bincount([b.act(q_values, key, 0) for _ in range(3500)], minlength=7)
    assert counts.min() > 300, f"not uniform enough: {counts}"


def test_better_actions_are_sampled_more_often_than_worse_ones(cfg, rng, q_values, key):
    """This is the whole point of Boltzmann over epsilon-greedy."""
    b = Boltzmann(cfg, rng)
    counts = np.bincount([b.act(q_values, key, 0) for _ in range(5000)], minlength=7)
    # q_values[5] = 0.30 is rated higher than q_values[4] = 0.05
    assert counts[5] > counts[4]


def test_no_overflow_at_tiny_temperature(cfg, rng, key):
    """exp(Q/tau) overflows unless max(Q) is subtracted first."""
    b = Boltzmann(cfg, rng)
    b.cfg.tau_end = 1e-6
    q = np.array([100.0, 50.0, 0.0, -50.0, 1.0, 2.0, 3.0])
    for _ in range(20):
        a = b.act(q, key, cfg.total_steps)
        assert 0 <= a < 7


def test_probabilities_are_finite_and_sum_to_one(cfg, rng):
    b = Boltzmann(cfg, rng)
    p = b.probabilities(np.array([100.0, 50.0, 0.0, -50.0, 1.0, 2.0, 3.0]), tau=1e-6)
    assert np.all(np.isfinite(p))
    assert np.isclose(p.sum(), 1.0)


def test_reports_temperature_for_logging(cfg, rng, q_values, key):
    b = Boltzmann(cfg, rng)
    b.act(q_values, key, 0)
    assert "temperature" in b.stats()


def test_adds_no_intrinsic_bonus(cfg, rng, key):
    assert Boltzmann(cfg, rng).intrinsic_bonus(key) == 0.0
