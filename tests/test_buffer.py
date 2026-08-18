import numpy as np

from rlx.buffer import ReplayBuffer


def _obs(v):
    return np.full((7, 7, 3), v, dtype=np.uint8)


def test_length_grows_then_caps_at_capacity():
    buf = ReplayBuffer(capacity=3, rng=np.random.default_rng(0))
    assert len(buf) == 0
    for i in range(5):
        buf.add(_obs(i), i % 7, float(i), _obs(i + 1), False)
    assert len(buf) == 3


def test_sample_returns_correctly_shaped_arrays():
    buf = ReplayBuffer(capacity=100, rng=np.random.default_rng(0))
    for i in range(50):
        buf.add(_obs(i), i % 7, 1.0, _obs(i + 1), i % 10 == 0)
    obs, act, rew, next_obs, done = buf.sample(8)
    assert obs.shape == (8, 7, 7, 3)
    assert next_obs.shape == (8, 7, 7, 3)
    assert act.shape == (8,)
    assert rew.shape == (8,)
    assert done.shape == (8,)


def test_oldest_entries_are_overwritten_first():
    buf = ReplayBuffer(capacity=2, rng=np.random.default_rng(0))
    buf.add(_obs(1), 0, 1.0, _obs(1), False)
    buf.add(_obs(2), 0, 2.0, _obs(2), False)
    buf.add(_obs(3), 0, 3.0, _obs(3), False)
    _, _, rew, _, _ = buf.sample(50)
    assert 1.0 not in set(rew.tolist())


def test_same_rng_seed_gives_the_same_sample():
    def draw():
        buf = ReplayBuffer(capacity=100, rng=np.random.default_rng(7))
        for i in range(50):
            buf.add(_obs(i), i % 7, float(i), _obs(i + 1), False)
        return buf.sample(8)[2]

    assert np.array_equal(draw(), draw())


def test_transitions_keep_their_fields_together():
    """A shuffled field would silently train the agent on mismatched data."""
    buf = ReplayBuffer(capacity=10, rng=np.random.default_rng(0))
    for i in range(10):
        buf.add(_obs(i), i % 7, float(i), _obs(i + 100), i % 2 == 0)

    obs, act, rew, next_obs, done = buf.sample(200)
    for o, a, r, n, d in zip(obs, act, rew, next_obs, done):
        i = int(o[0, 0, 0])                 # the value we stored as the marker
        assert a == i % 7
        assert r == float(i)
        assert int(n[0, 0, 0]) == i + 100
        assert bool(d) == (i % 2 == 0)


def test_never_samples_unwritten_slots():
    """Sampling from a buffer with 3 of 1000 slots filled must not return zeros."""
    buf = ReplayBuffer(capacity=1000, rng=np.random.default_rng(0))
    for i in (1, 2, 3):
        buf.add(_obs(i), 0, float(i), _obs(i), False)
    _, _, rew, _, _ = buf.sample(500)
    assert set(rew.tolist()) <= {1.0, 2.0, 3.0}


def test_observations_are_stored_as_uint8():
    """float32 would quadruple memory for no benefit -- 100k entries is ~15 MB."""
    buf = ReplayBuffer(capacity=10, rng=np.random.default_rng(0))
    buf.add(_obs(1), 0, 1.0, _obs(1), False)
    assert buf.sample(1)[0].dtype == np.uint8


def test_stored_values_are_not_aliased_to_the_caller_array():
    """If the buffer keeps a reference, the training loop mutating obs would
    silently rewrite history."""
    buf = ReplayBuffer(capacity=10, rng=np.random.default_rng(0))
    obs = _obs(5)
    buf.add(obs, 0, 1.0, _obs(6), False)
    obs[:] = 99
    assert int(buf.sample(1)[0][0, 0, 0, 0]) == 5
