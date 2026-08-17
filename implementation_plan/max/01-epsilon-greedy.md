# Task 1 — Epsilon-greedy (and your test fixtures)

The baseline every other strategy is compared against. Also the task where you
set up the fake data all your other tasks use, so you never wait on Samuel.

**Files:**
- Create: `src/rlx/exploration/epsilon_greedy.py`
- Create: `tests/test_exploration/__init__.py`, `tests/test_exploration/conftest.py`
- Test: `tests/test_exploration/test_epsilon_greedy.py`

**Interfaces:**
- Consumes: `Explorer` (Samuel task 1), `RunConfig` (Samuel task 1).
- Produces: `EpsilonGreedy(cfg, rng)`, already wired into `make_explorer`.

**Config fields you read:** `epsilon_start` (1.0), `epsilon_end` (0.05),
`epsilon_decay_frac` (0.2), `total_steps`.

---

## What epsilon-greedy is

With probability epsilon, ignore what the network thinks and pick a completely
random action. Otherwise pick the action with the highest Q-value.

Epsilon starts at 1.0 (act entirely at random, since the network knows nothing
yet) and decays linearly to 0.05 over the **first 20%** of training, then stays
at 0.05 forever. The floor is there so the agent never becomes fully
deterministic and stuck.

---

- [ ] **Step 1: Create the shared test fixtures**

Create `tests/test_exploration/__init__.py` (empty) and
`tests/test_exploration/conftest.py`. Every later task uses these.

```python
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
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_exploration/test_epsilon_greedy.py`:

```python
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
```

- [ ] **Step 3: Run and watch them fail**

```bash
pytest tests/test_exploration/test_epsilon_greedy.py -v
```

Expected: `ModuleNotFoundError: No module named 'rlx.exploration.epsilon_greedy'`.

- [ ] **Step 4: Write the implementation**

Create `src/rlx/exploration/epsilon_greedy.py`:

```python
"""Epsilon-greedy: the baseline. Act randomly with probability epsilon,
otherwise act greedily. Epsilon decays linearly then holds at a floor."""

import numpy as np

from rlx.config import RunConfig
from rlx.exploration.base import Explorer


class EpsilonGreedy(Explorer):
    def __init__(self, cfg: RunConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self._decay_steps = max(1, int(cfg.epsilon_decay_frac * cfg.total_steps))
        self._epsilon = cfg.epsilon_start

    def epsilon(self, step: int) -> float:
        """Linear decay from epsilon_start to epsilon_end, then constant."""
        frac = min(1.0, step / self._decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, q_values: np.ndarray, count_key, step: int) -> int:
        self._epsilon = self.epsilon(step)
        if self.rng.random() < self._epsilon:
            return int(self.rng.integers(len(q_values)))
        return int(np.argmax(q_values))

    def stats(self) -> dict[str, float]:
        return {"epsilon": self._epsilon}
```

- [ ] **Step 5: Run the tests**

```bash
pytest tests/test_exploration/test_epsilon_greedy.py -v
```

Expected: 8 passed.

If `test_acts_almost_uniformly_at_the_very_start` fails, check that the random
branch draws from **all** actions, not from the non-greedy ones only. Both are
defensible definitions of epsilon-greedy, but ours is the standard one and the
test encodes it.

- [ ] **Step 6: Check it is reachable through the factory**

```bash
python -c "
import numpy as np
from rlx.config import RunConfig
from rlx.exploration import make_explorer
e = make_explorer('epsilon_greedy', RunConfig(env_id='Empty-5', strategy='epsilon_greedy', seed=0), np.random.default_rng(0))
print(type(e).__name__, e.act(np.array([0.1,0.9,0.2,0,0,0,0]), 1, 10**9))
"
```

Expected: `EpsilonGreedy` and an action index.

- [ ] **Step 7: Log and commit**

Append to `docs/decision_log.md`, in plain language:

```markdown
## 2026-08-18 — Epsilon-greedy baseline implemented

**Status:** Active

**What changed:** Added our baseline exploration strategy. With probability
"epsilon" the agent throws away what it has learned and picks a random action;
otherwise it does what it thinks is best. Epsilon starts at 1.0 (everything
random, because at the start the agent knows nothing) and drops in a straight
line to 0.05 over the first fifth of training.

**Why:** It is the standard baseline that every other strategy gets compared
against, and it is deliberately the dumbest of the four.

**What it means for the results:** The 0.05 floor means the agent always keeps a
little randomness rather than becoming completely predictable. If a smarter
strategy cannot beat this one, that is a genuinely interesting finding, not a
bug.
```

```bash
git add src/rlx/exploration/epsilon_greedy.py tests/test_exploration/ docs/decision_log.md
git commit -m "feat: epsilon-greedy exploration strategy"
```
