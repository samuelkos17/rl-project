# Task 3 — Count-based exploration bonus

**Files:**
- Create: `src/rlx/exploration/count_based.py`
- Test: `tests/test_exploration/test_count_based.py`

**Interfaces:**
- Consumes: `Explorer`, `RunConfig`, task 1 fixtures.
- Produces: `CountBased(cfg, rng)`.

**Config fields you read:** `count_beta` (0.05), `count_epsilon` (0.05).

---

## What count-based exploration is

Keep a tally of how many times the agent has seen each situation. Then pay it a
small extra reward for being somewhere unfamiliar:

```
bonus = beta / sqrt(N(key))
```

First visit gives `beta / 1 = 0.05`. Hundredth visit gives `beta / 10 = 0.005`.
So rare situations pay well and familiar ones pay almost nothing. This is an
**intrinsic reward** — reward the agent gives itself, on top of the real reward
from the maze.

Action selection is epsilon-greedy with a **fixed small** epsilon of 0.05. The
exploration pressure is meant to come from the bonus, not from randomness, but a
small floor stops the agent getting stuck in a deterministic loop.

## The two things that must be right

**1. The bonus never reaches a reported score.** The training loop adds it to the
reward stored in the replay buffer, and nowhere else. Evaluation is greedy on the
maze's real reward only. Your class just returns the number; the loop handles the
rest. Do not add it anywhere yourself.

**2. You count observations, not positions.** `count_key` is
`obs.tobytes()` — the raw bytes of what the agent *sees*, not where it *is*.

This is a deliberate deviation from what our professor suggested, and the reason
matters: `(x, y, direction)` is information the agent never receives. Handing it
to this one strategy would mean comparing a strategy with privileged information
against three without it, and the controlled comparison — the entire point of the
project — would collapse. See `docs/decision_log.md`, entry
"Count-based bonus counts observations, not true positions".

The honest cost, which belongs in the report: two different spots in a maze can
look identical through a 7x7 window (**perceptual aliasing**), so counting views
is blurrier than counting positions and the bonus is somewhat weaker.

**Your class never needs to know any of this** — it just counts whatever key it
is handed. But you should understand it, because you may be asked in the exam.

---

- [ ] **Step 1: Write the failing tests**

Create `tests/test_exploration/test_count_based.py`:

```python
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
```

- [ ] **Step 2: Run and watch them fail**

```bash
pytest tests/test_exploration/test_count_based.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Create `src/rlx/exploration/count_based.py`:

```python
"""Count-based exploration: pay the agent a shrinking bonus for visiting
situations it has seen rarely.

The counted key is the agent's OWN observation as raw bytes, never its true
(x, y, direction). See CLAUDE.md section 8.
"""

from collections import defaultdict

import numpy as np

from rlx.config import RunConfig
from rlx.exploration.base import Explorer


class CountBased(Explorer):
    def __init__(self, cfg: RunConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        # Keys are raw observation bytes handed in by the training loop.
        self.counts: dict[bytes, int] = defaultdict(int)
        self._recent_bonuses: list[float] = []

    def act(self, q_values: np.ndarray, count_key, step: int) -> int:
        if self.rng.random() < self.cfg.count_epsilon:
            return int(self.rng.integers(len(q_values)))
        return int(np.argmax(q_values))

    def observe(self, count_key) -> None:
        self.counts[count_key] += 1

    def intrinsic_bonus(self, count_key) -> float:
        # max(count, 1) keeps an unseen key finite instead of dividing by zero.
        bonus = self.cfg.count_beta / np.sqrt(max(self.counts[count_key], 1))
        self._recent_bonuses.append(bonus)
        if len(self._recent_bonuses) > 1000:
            del self._recent_bonuses[:-1000]
        return float(bonus)

    def stats(self) -> dict[str, float]:
        return {
            "epsilon": self.cfg.count_epsilon,
            "mean_bonus": float(np.mean(self._recent_bonuses)) if self._recent_bonuses else 0.0,
            "distinct_keys": float(len(self.counts)),
        }
```

Note `defaultdict(int)` means reading an unseen key inserts it with count 0 —
which is why `max(..., 1)` is there rather than relying on the count being at
least 1.

- [ ] **Step 4: Run the tests**

```bash
pytest tests/test_exploration/test_count_based.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Check the bonus scale against real reward**

`beta = 0.05` is provisional. MiniGrid gives a return between 0 and 1 for solving
the maze, so if the total intrinsic bonus collected over an episode swamps that,
the agent will chase novelty and ignore the goal.

```bash
python -c "
import numpy as np
from rlx.config import RunConfig
from rlx.exploration.count_based import CountBased
cfg = RunConfig(env_id='Empty-5', strategy='count_based', seed=0)
c = CountBased(cfg, np.random.default_rng(0))
total = 0.0
for step in range(300):          # a plausible episode length
    k = step // 3                # roughly 100 distinct views per episode
    c.observe(k)
    total += c.intrinsic_bonus(k)
print(f'total intrinsic bonus over one early episode: {total:.3f}')
print('maze reward for solving: about 0.9')
"
```

If the printed total is far above ~1.0, the bonus dominates the real task.
**Do not silently change `beta`** — report the number to Samuel and Daniel, agree
a value, then change `count_beta` in `config.py` and write a decision-log entry
with the measured number in it.

`mean_bonus` is logged in `metrics.csv` for exactly this reason: on integration
day you can compare it against real return in actual runs.

- [ ] **Step 6: Log and commit**

Append to `docs/decision_log.md`:

```markdown
## 2026-08-19 — Count-based exploration implemented

**Status:** Active

**What changed:** Added the third strategy. It keeps a tally of how often the
agent has seen each situation, and pays a small bonus for being somewhere
unfamiliar — a lot on the first visit, almost nothing on the hundredth. The agent
mostly acts greedily, with a constant 5% chance of a random action so it cannot
get stuck repeating itself forever.

**Why the bonus is small (0.05):** solving the maze is worth about 0.9. If the
novelty bonus were anywhere near that, the agent would happily wander forever
collecting novelty and never bother finding the goal. We log the average bonus
during every run so we can check this did not happen.

**What it means for the results:** This is the only one of the four strategies
that changes the *reward*, not just the *action choice*. The bonus only ever
affects what the agent learns from — every score we report is the maze's real
reward, with the bonus switched off.
```

```bash
git add src/rlx/exploration/count_based.py tests/test_exploration/test_count_based.py docs/decision_log.md
git commit -m "feat: count-based exploration bonus"
```
