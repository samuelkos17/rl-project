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
    # Pin the temperature rather than inheriting the config default, like the two
    # tests above do. The q_values fixture spans 0.85, ~250x the ~0.003 gaps a
    # real network produces, so at the configured tau_start the softmax saturates
    # onto the argmax and this property becomes untestable on these numbers.
    b.cfg.tau_start = 1.0
    counts = np.bincount([b.act(q_values, key, 0) for _ in range(5000)], minlength=7)
    # q_values[5] = 0.30 is rated higher than q_values[4] = 0.05
    assert counts[5] > counts[4]


def test_configured_schedule_explores_then_commits_at_realistic_q_gaps(cfg, rng, key):
    """Regression test for the bug that made Boltzmann uniform-random.

    tau only means anything relative to the size of Q differences. With
    tau_end=0.05 against real gaps of ~0.003, Boltzmann picked its favourite
    action 15% of the time versus 14.3% for a coin flip -- it never exploited,
    for the entire run, and no test noticed. These Q-values are the measured
    scale (gap 0.0034, spread 0.0206); see docs/decision_log.md.
    """
    q = np.array([0.0, 0.004, 0.002, 0.0206, 0.001, 0.017, 0.010])
    uniform = 1.0 / len(q)
    b = Boltzmann(cfg, rng)

    p_start = b.probabilities(q, b.temperature(0))[3]
    p_end = b.probabilities(q, b.temperature(cfg.total_steps))[3]

    assert p_start > 1.5 * uniform, f"starts indistinguishable from random: {p_start:.3f}"
    assert p_end > 0.8, f"never commits to its favourite action: {p_end:.3f}"
    assert p_end > p_start, "must become greedier over training, not less"


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
