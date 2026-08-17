# Task 2 — Boltzmann (softmax) exploration

**Files:**
- Create: `src/rlx/exploration/boltzmann.py`
- Test: `tests/test_exploration/test_boltzmann.py`

**Interfaces:**
- Consumes: `Explorer`, `RunConfig`, the fixtures from task 1.
- Produces: `Boltzmann(cfg, rng)`.

**Config fields you read:** `tau_start` (1.0), `tau_end` (0.05),
`tau_decay_frac` (0.4), `total_steps`.

---

## What Boltzmann exploration is

Epsilon-greedy's random choices are *uniformly* random — when it explores, an
action the network rates as terrible is just as likely as one it rates as
second-best. Boltzmann fixes that: it samples every action with probability
proportional to `exp(Q / tau)`, so better-looking actions get picked more often
even while exploring.

`tau` is the **temperature**. High temperature flattens the distribution toward
uniform (lots of exploration). Low temperature sharpens it toward pure greedy.

Our schedule (the professor asked us to state it explicitly): **exponential**
decay from 1.0 to 0.05 over the first **40%** of training, constant after.

```
tau(t) = max(tau_end, tau_start * (tau_end / tau_start) ** (t / (tau_decay_frac * total_steps)))
```

Exponential rather than linear because temperature acts multiplicatively on the
Q-value scale — halving it repeatedly is the natural step, not subtracting a
constant.

**The numerical trap:** `exp(0.9 / 0.05)` is `exp(18)`, and at smaller `tau` this
overflows to `inf`, giving `nan` probabilities. Always subtract `max(Q)` before
exponentiating. This changes nothing mathematically (the constant cancels in the
normalisation) and it is the difference between working and silently producing
`nan`. There is a test for it.

---

- [x] **Step 1: Write the failing tests**

Create `tests/test_exploration/test_boltzmann.py`:

```python
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
```

- [x] **Step 2: Run and watch them fail**

```bash
pytest tests/test_exploration/test_boltzmann.py -v
```

Expected: `ModuleNotFoundError`.

- [x] **Step 3: Write the implementation**

Create `src/rlx/exploration/boltzmann.py`:

```python
"""Boltzmann (softmax) exploration: sample actions in proportion to
exp(Q / tau), with tau decaying exponentially."""

import numpy as np

from rlx.config import RunConfig
from rlx.exploration.base import Explorer


class Boltzmann(Explorer):
    def __init__(self, cfg: RunConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self._decay_steps = max(1, int(cfg.tau_decay_frac * cfg.total_steps))
        self._tau = cfg.tau_start

    def temperature(self, step: int) -> float:
        """Exponential decay from tau_start to tau_end, then constant."""
        frac = min(1.0, step / self._decay_steps)
        tau = self.cfg.tau_start * (self.cfg.tau_end / self.cfg.tau_start) ** frac
        return max(self.cfg.tau_end, tau)

    def probabilities(self, q_values: np.ndarray, tau: float) -> np.ndarray:
        """Softmax over Q/tau. Subtracting the max prevents overflow."""
        scaled = q_values / max(tau, 1e-8)
        shifted = np.exp(scaled - scaled.max())
        return shifted / shifted.sum()

    def act(self, q_values: np.ndarray, count_key, step: int) -> int:
        self._tau = self.temperature(step)
        p = self.probabilities(q_values, self._tau)
        return int(self.rng.choice(len(q_values), p=p))

    def stats(self) -> dict[str, float]:
        return {"temperature": self._tau}
```

- [x] **Step 4: Run the tests**

```bash
pytest tests/test_exploration/test_boltzmann.py -v
```

Expected: 9 passed.

If `test_no_overflow_at_tiny_temperature` fails with `nan` or
`probabilities do not sum to 1`, the max-subtraction is missing or is being done
after the exponential instead of before.

- [x] **Step 5: Look at the actual distribution once**

A sanity check you should read with your own eyes, not just assert on:

```bash
python -c "
import numpy as np
from rlx.config import RunConfig
from rlx.exploration.boltzmann import Boltzmann
b = Boltzmann(RunConfig(env_id='Empty-5', strategy='boltzmann', seed=0, total_steps=10000), np.random.default_rng(0))
q = np.array([0.1, 0.2, 0.15, 0.9, 0.05, 0.3, 0.25])
for tau in (1.0, 0.5, 0.2, 0.05):
    print(f'tau={tau:<5} p={np.round(b.probabilities(q, tau), 3)}')
"
```

At `tau=1.0` the probabilities should be fairly flat; at `tau=0.05` almost all
mass should sit on index 3. If that progression does not appear, the temperature
is being applied the wrong way round (multiplied instead of divided).

- [ ] **Step 6: Log and commit**

Append to `docs/decision_log.md`:

```markdown
## 2026-08-18 — Boltzmann exploration implemented

**Status:** Active

**What changed:** Added the second strategy. Instead of "best action, or a
completely random one", it picks each action with a probability based on how
good the agent thinks it is. A second-best action gets chosen fairly often; an
action the agent rates as useless almost never does.

The "temperature" controls how picky it is: high temperature means nearly random,
low temperature means nearly always the best action. We start at 1.0 and shrink
it to 0.05 over the first 40% of training. Our professor specifically asked us to
write this schedule down, so it is in the spec too.

**One implementation detail worth recording:** the maths involves raising e to
the power of (Q divided by temperature). At low temperatures that number gets
astronomically large and the computer gives up and returns "infinity", which
poisons everything downstream. The standard fix is subtracting the largest
Q-value first, which changes nothing mathematically but keeps the numbers small.
We have a test for it.

**What it means for the results:** In principle this should explore more
sensibly than epsilon-greedy, because its random choices are informed rather than
uniform. Whether that actually helps is one of the things we are measuring.
```

```bash
git add src/rlx/exploration/boltzmann.py tests/test_exploration/test_boltzmann.py docs/decision_log.md
git commit -m "feat: Boltzmann exploration strategy"
```
