# Workstream B — Exploration strategies (Max)

> **For Claude sessions:** Read `CLAUDE.md`, then
> `docs/specs/2026-08-17-exploration-comparison-design.md` §5, then this file,
> then the numbered task file you are on. Work one task at a time, in order.

**Goal:** Four exploration strategies behind one shared interface, so the
training loop can swap them by changing a single string.

**Architecture:** Each strategy is one small file implementing the frozen
`Explorer` interface from `src/rlx/exploration/base.py`. Strategies are pure —
they receive Q-values and a count key, and return an action or a bonus. They
never touch the environment, the replay buffer, or the training loop.

**Tech Stack:** Python 3.11, numpy, torch (only for NoisyNets), pytest.

**Spec:** `docs/specs/2026-08-17-exploration-comparison-design.md`

---

## Your interface (frozen — do not change it)

```python
class Explorer(ABC):
    uses_noisy_net: bool = False

    @abstractmethod
    def act(self, q_values: np.ndarray, count_key: Hashable, step: int) -> int: ...

    def intrinsic_bonus(self, count_key: Hashable) -> float: return 0.0
    def observe(self, count_key: Hashable) -> None: pass
    def stats(self) -> dict[str, float]: return {}
```

Every strategy takes `(cfg: RunConfig, rng: np.random.Generator)` in its
constructor. Use `self.rng` for all randomness — never `np.random.*` directly, or
runs stop being reproducible.

## Tasks in order

| # | File | Strategy | Depends on |
|---|---|---|---|
| 1 | `01-epsilon-greedy.md` | epsilon-greedy baseline + your test fixtures | Samuel task 1 |
| 2 | `02-boltzmann.md` | Boltzmann / softmax | task 1 |
| 3 | `03-count-based.md` | count-based intrinsic bonus | task 1 |
| 4 | `04-noisy-nets.md` | NoisyNets | Samuel task 4 (`QNetwork`) |
| 5 | `05-writeups.md` | plain-language explanation of each strategy for the report | tasks 1–4 |

**You are not blocked by Samuel after his task 1.** You develop against fake
Q-value arrays. You never need a working DQN, a working environment, or a GPU.
Task 4 is the one exception — it needs `QNetwork` to exist — and even that can be
written against the placeholder `NoisyLinear` he ships in his task 4.

## The one thing you must not get wrong

`count_key` is the **agent's own 7x7x3 observation, as raw bytes**. It is *not* the
agent's `(x, y, direction)` position. The agent never sees its position.

If the count-based strategy counted true positions, it would be running on
information the other three strategies do not have, and the whole comparison
would be invalid. This is the point our professor pressed hardest on. See
`CLAUDE.md` §8.

## Definition of done

```bash
pytest tests/test_exploration/ -v
```

all green, and:

```python
from rlx.exploration import make_explorer, STRATEGIES
# all four names construct without error
```

## What you must not do

- Do not tune hyperparameters to make your strategy win. They come from the
  config and are fixed across everything.
- Do not add a fifth strategy.
- Do not let a strategy read the environment or the true state.
- Do not use the global numpy RNG.
