import numpy as np

from rlx.exploration.count_based import CountBased


def test_first_visit_gives_the_full_bonus(cfg, rng, key):
    c = CountBased(cfg, rng)
    c.observe(key)
    assert np.isclose(c.intrinsic_bonus(key), cfg.count_beta)


def test_bonus_shrinks_as_one_over_sqrt_n(cfg, rng, key):
    c = CountBased(cfg, rng)
    for _ in range(100):
        c.observe(key)
    assert np.isclose(c.intrinsic_bonus(key), cfg.count_beta / 10.0)


def test_bonus_decreases_monotonically_with_visits(cfg, rng, key):
    c = CountBased(cfg, rng)
    bonuses = []
    for _ in range(50):
        c.observe(key)
        bonuses.append(c.intrinsic_bonus(key))
    assert all(a >= b for a, b in zip(bonuses, bonuses[1:]))


def test_unseen_key_is_never_infinite(cfg, rng):
    """A key with zero visits must not divide by zero."""
    c = CountBased(cfg, rng)
    b = c.intrinsic_bonus(999)
    assert np.isfinite(b)
    assert b > 0


def test_distinct_keys_are_counted_separately(cfg, rng):
    c = CountBased(cfg, rng)
    for _ in range(100):
        c.observe(1)
    c.observe(2)
    assert c.intrinsic_bonus(2) > c.intrinsic_bonus(1)


def test_bonus_is_small_relative_to_maze_reward(cfg, rng, key):
    """MiniGrid returns live in [0, 1]. A bonus near 1.0 would drown the task."""
    c = CountBased(cfg, rng)
    c.observe(key)
    assert c.intrinsic_bonus(key) < 0.1


def test_acts_greedily_most_of_the_time(cfg, rng, q_values, key):
    c = CountBased(cfg, rng)
    counts = np.bincount([c.act(q_values, key, 0) for _ in range(2000)], minlength=7)
    assert counts[3] > 1500          # greedy roughly (1 - 0.05) of the time
    assert counts[3] < 2000          # but not always -- the floor is real


def test_epsilon_does_not_decay(cfg, rng, q_values, key):
    """Unlike epsilon-greedy, this one holds a constant small epsilon."""
    c = CountBased(cfg, rng)
    c.act(q_values, key, 0)
    early = c.stats()["epsilon"]
    c.act(q_values, key, cfg.total_steps)
    assert c.stats()["epsilon"] == early


def test_reports_mean_bonus_for_logging(cfg, rng, q_values, key):
    c = CountBased(cfg, rng)
    c.observe(key)
    c.act(q_values, key, 0)
    assert "mean_bonus" in c.stats()
    assert "epsilon" in c.stats()


def test_act_does_not_count(cfg, rng, q_values, key):
    """Only observe() counts. act() must leave the table untouched.

    The Explorer contract calls observe() once per environment step, after
    acting. If act() also incremented, every state would be counted twice, so
    every bonus would be too small by a factor of sqrt(2) and distinct_keys
    would be inflated. Verified 2026-08-20 that adding the extra increment to
    act() passes all 37 other exploration tests.
    """
    c = CountBased(cfg, rng)
    for step in range(10):
        c.act(q_values, key, step)
    assert dict(c.counts) == {}, "act() incremented the count table"
    assert c.stats()["distinct_keys"] == 0.0

    # and the first observe() must still yield the full, undivided bonus
    c.observe(key)
    assert np.isclose(c.intrinsic_bonus(key), cfg.count_beta)


def test_intrinsic_bonus_does_not_create_entries(cfg, rng, key):
    """Asking about an unseen key must not make it count as seen.

    self.counts is a defaultdict, so a bare self.counts[key] read inserts a
    zero entry and distinct_keys starts counting keys merely asked about.
    """
    c = CountBased(cfg, rng)
    c.intrinsic_bonus(b"never-observed")
    assert c.stats()["distinct_keys"] == 0.0
