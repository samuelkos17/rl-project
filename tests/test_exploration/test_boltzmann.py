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


# Real Q-values from a freshly initialised QNetwork on the Empty-5 first
# observation (torch seed 41), shifted so the smallest is 0 -- a constant shift
# is a no-op for softmax. Its best-vs-second gap is 0.0208, matching the measured
# init-scale mean of 0.0206 over 6 instances x 15 seeds.
Q_INIT = np.array([0.1839, 0.0729, 0.0585, 0.2014, 0.2363, 0.2156, 0.0])
# The same vector rescaled to the measured TRAINED gap, 0.0030 (6 instances x
# 2 seeds x 160k steps). The Q-scale shrinks ~7x over a run, which is why the
# two ends of the tau schedule must be checked against different arrays.
Q_TRAINED = Q_INIT * (0.0030 / 0.02076)
BEST = 4  # argmax of both


def test_configured_schedule_explores_early_and_commits_late(cfg, rng, key):
    """Regression test for BOTH ways the tau schedule has been mis-scaled.

    tau only means anything relative to the size of Q differences, and that
    size is not constant: a random network spreads its Q-values ~7x wider than
    a trained one. Each end of the schedule is therefore checked against the
    scale that is actually present when that end is in force.

    Two real bugs this rejects, both shipped, neither caught by the tests that
    existed at the time:
      tau_end=0.05   -> p(best) 0.15 vs 0.143 for a coin flip. Boltzmann was
                        uniform-random for all 400k steps, on every instance.
      tau_start=0.01 -> p(best) 0.86 at step 0. Boltzmann committed to a
                        randomly initialised preference and never left; the
                        pilot showed it visiting 4 of ~36 states on Empty-5,
                        i.e. spinning in place for 20k steps.
    See docs/decision_log.md.
    """
    uniform = 1.0 / len(Q_INIT)
    b = Boltzmann(cfg, rng)

    p_start = b.probabilities(Q_INIT, b.temperature(0))[BEST]
    p_end = b.probabilities(Q_TRAINED, b.temperature(cfg.total_steps))[BEST]

    # Early: better than a coin flip -- that is Boltzmann's whole claim over
    # epsilon-greedy -- but nowhere near committed.
    assert p_start > uniform, f"starts indistinguishable from random: {p_start:.3f}"
    assert p_start < 0.5, f"already committed at step 0: {p_start:.3f}"
    # Late: actually exploits what it learned.
    assert p_end > 0.9, f"never commits to its favourite action: {p_end:.3f}"
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
