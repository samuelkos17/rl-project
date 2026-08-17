import numpy as np
import pytest

from rlx.config import RunConfig

N_ACTIONS = 7


@pytest.fixture
def cfg():
    """A config with a short, round total_steps so schedules are easy to check."""
    return RunConfig(env_id="Empty-5", strategy="epsilon_greedy", seed=0,
                     total_steps=10_000)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def q_values():
    """Fake Q-values with a single clear winner at index 3."""
    q = np.array([0.1, 0.2, 0.15, 0.9, 0.05, 0.3, 0.25])
    assert int(np.argmax(q)) == 3
    return q


@pytest.fixture
def key():
    """A stand-in for the real count key, which is obs.tobytes()."""
    return b"fake-observation-bytes"
