# Task 1 — Scaffold and frozen contracts

**This task blocks Max and Daniel. Do it first, merge it to `main`, and tell
them it is merged.** It is mechanical: mostly copy-paste. Target: 30 minutes.

**Files:**
- Create: `pyproject.toml`, `requirements.txt`
- Create: `src/rlx/__init__.py`, `src/rlx/config.py`
- Create: `src/rlx/exploration/__init__.py`, `src/rlx/exploration/base.py`
- Create: `tests/__init__.py`, `tests/test_config.py`, `tests/test_explorer_contract.py`
- Create: `configs/main.yaml`

**Interfaces:**
- Consumes: nothing.
- Produces: `RunConfig` (all field names), `Explorer` ABC,
  `make_explorer(name, cfg, rng)`, `ENV_IDS`, `difficulty_index(env_id)`.
  **Max and Daniel both code against these. They are frozen.**

---

- [ ] **Step 1: Create the conda environment**

```bash
conda create -n rl python=3.11 -y
conda activate rl
```

Do **not** use the base Anaconda 3.13 environment. `rliable` and parts of the
MiniGrid stack lag the newest Python, and a dependency fight costs a day we do
not have.

- [ ] **Step 2: Write `requirements.txt`**

```
torch>=2.0
gymnasium>=0.29
minigrid>=2.3
numpy>=1.24
pandas>=2.0
scipy>=1.10
matplotlib>=3.7
rliable>=1.0.8
pyyaml>=6.0
pytest>=7.4
tabulate>=0.9
```

`tabulate` is not decoration: `pandas.DataFrame.to_markdown()` raises
`ImportError` without it, and Daniel's report generator is built entirely on
`to_markdown`.

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "rlx"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Install the dependencies**

```bash
pip install -r requirements.txt
```

`pip install -e .` comes later, in step 7 — `src/rlx/` does not exist yet, so
running it now would install an empty package.

Expected: no errors. **If `rliable` or `minigrid` fails to install, stop and
report it — do not substitute a different package.** Record the failure in
`docs/decision_log.md`.

**Then check what torch you actually got:**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints something ending in `+cpu` with `False`, pip gave you a CPU-only
build — which is the default on Windows. That is fine for now, but **task 2's
CPU-vs-GPU benchmark cannot measure a GPU with this build.** See the
`docs/decision_log.md` entry "pip installed a CPU-only torch" and decide there.

- [ ] **Step 5: Write the failing test for `RunConfig`**

Create `tests/test_config.py`:

```python
from pathlib import Path
from rlx.config import RunConfig


def test_run_dir_is_nested_by_env_strategy_seed():
    cfg = RunConfig(env_id="DoorKey-5", strategy="epsilon_greedy", seed=3)
    assert cfg.run_dir == Path("results") / "DoorKey-5" / "epsilon_greedy" / "seed3"


def test_defaults_match_the_spec():
    cfg = RunConfig(env_id="Empty-5", strategy="epsilon_greedy", seed=0)
    assert cfg.buffer_size == 100_000
    assert cfg.batch_size == 32
    assert cfg.gamma == 0.99
    assert cfg.train_freq == 4
    assert cfg.eval_episodes == 1
```

- [ ] **Step 6: Run it and watch it fail**

```bash
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'rlx.config'`.

- [ ] **Step 7: Write `src/rlx/config.py`**

Create `src/rlx/__init__.py` (empty), then `src/rlx/config.py`:

```python
"""Run configuration. FROZEN CONTRACT -- field names are depended on by all
three workstreams. See CLAUDE.md section 6."""

from dataclasses import dataclass, asdict
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    env_id: str
    strategy: str
    seed: int

    # --- fixed across every strategy and every environment ---
    total_steps: int = 400_000
    buffer_size: int = 100_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    gamma: float = 0.99
    target_update: int = 1_000
    learning_starts: int = 1_000
    train_freq: int = 4
    grad_clip: float = 10.0

    # --- evaluation ---
    # 1, not 10: the layout is pinned per run and evaluation is greedy, so
    # MiniGrid is fully deterministic and 10 episodes would be identical.
    eval_every: int = 5_000
    eval_episodes: int = 1

    # --- logging ---
    # 10k, not 20k: the early-coverage window is the first 20% of training, so
    # at 20k only 4 snapshots would land inside it -- a 4-point trapezoid for the
    # project's main predictor. 10k gives 8 points. Storage is ~1 KB per
    # snapshot, so the resolution is effectively free.
    snapshot_every: int = 10_000

    # --- strategy hyperparameters (only the relevant ones are read) ---
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_frac: float = 0.2
    tau_start: float = 1.0
    tau_end: float = 0.05
    tau_decay_frac: float = 0.4
    count_beta: float = 0.05
    count_epsilon: float = 0.05
    noisy_sigma0: float = 0.5

    # --- execution ---
    device: str = "cpu"
    results_root: str = "results"

    @property
    def run_dir(self) -> Path:
        return Path(self.results_root) / self.env_id / self.strategy / f"seed{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)


def load_base_config(path: Path) -> dict:
    """Read the shared defaults block from a sweep YAML file."""
    with open(path) as f:
        return yaml.safe_load(f).get("defaults", {})
```

- [ ] **Step 8: Install the package, then run the test and watch it pass**

Now that `src/rlx/` exists, the editable install has something to find:

```bash
pip install -e .
pytest tests/test_config.py -v
```

Expected: 2 passed.

- [ ] **Step 9: Write the failing test for the `Explorer` contract**

Create `tests/test_explorer_contract.py`:

```python
import numpy as np
import pytest

from rlx.exploration.base import Explorer


class _Dummy(Explorer):
    def act(self, q_values, count_key, step):
        return int(np.argmax(q_values))


def test_explorer_cannot_be_instantiated_without_act():
    with pytest.raises(TypeError):
        Explorer()


def test_default_bonus_is_zero_and_observe_is_a_noop():
    e = _Dummy()
    assert e.intrinsic_bonus(("k",)) == 0.0
    assert e.observe(("k",)) is None
    assert e.stats() == {}
    assert e.uses_noisy_net is False


def test_act_returns_an_int_action():
    e = _Dummy()
    assert e.act(np.array([0.1, 0.9, 0.3]), ("k",), 0) == 1
```

- [ ] **Step 10: Run it and watch it fail**

```bash
pytest tests/test_explorer_contract.py -v
```

Expected: `ModuleNotFoundError: No module named 'rlx.exploration'`.

- [ ] **Step 11: Write `src/rlx/exploration/base.py`**

```python
"""The exploration strategy interface. FROZEN CONTRACT -- Max owns the
implementations, but the interface is agreed by all three. See CLAUDE.md
section 6.

`count_key` is the agent's OWN 7x7x3 observation, as raw bytes. It is NOT the
privileged (x, y, direction) state. See CLAUDE.md section 8.
"""

from abc import ABC, abstractmethod
from typing import Hashable

import numpy as np


class Explorer(ABC):
    #: True if this strategy needs the Q-network head built from NoisyLinear.
    uses_noisy_net: bool = False

    @abstractmethod
    def act(self, q_values: np.ndarray, count_key: Hashable, step: int) -> int:
        """Choose an action given Q-values for the current observation."""

    def intrinsic_bonus(self, count_key: Hashable) -> float:
        """Extra reward added to the transition stored in the replay buffer.

        Never enters the evaluation return. Default: no bonus.
        """
        return 0.0

    def observe(self, count_key: Hashable) -> None:
        """Called once per environment step, after acting. Default: no-op."""

    def stats(self) -> dict[str, float]:
        """Scalars to log this step (epsilon, temperature, mean bonus)."""
        return {}
```

- [ ] **Step 12: Write `src/rlx/exploration/__init__.py`**

The factory lives here so `base.py` stays free of imports of its own
implementations.

```python
"""Exploration strategies. Implementations are owned by Max (workstream B)."""

import numpy as np

from rlx.config import RunConfig
from rlx.exploration.base import Explorer

STRATEGIES = ("epsilon_greedy", "boltzmann", "count_based", "noisy")


def make_explorer(name: str, cfg: RunConfig, rng: np.random.Generator) -> Explorer:
    """Build the named exploration strategy."""
    if name == "epsilon_greedy":
        from rlx.exploration.epsilon_greedy import EpsilonGreedy
        return EpsilonGreedy(cfg, rng)
    if name == "boltzmann":
        from rlx.exploration.boltzmann import Boltzmann
        return Boltzmann(cfg, rng)
    if name == "count_based":
        from rlx.exploration.count_based import CountBased
        return CountBased(cfg, rng)
    if name == "noisy":
        from rlx.exploration.noisy import NoisyExplorer
        return NoisyExplorer(cfg, rng)
    raise ValueError(f"unknown strategy {name!r}, expected one of {STRATEGIES}")
```

The imports are inside the branches on purpose: Max's four modules do not exist
yet, and this lets the contract be merged before they land.

- [ ] **Step 13: Run the contract test and watch it pass**

```bash
pytest tests/test_explorer_contract.py -v
```

Expected: 3 passed.

- [ ] **Step 14: Write the environment naming contract**

`ENV_IDS` and `difficulty_index` are pure string conventions with no MiniGrid
dependency, and **Daniel's `aggregate.py` imports `difficulty_index`**. They ship
here, in the frozen-contract task, so he is not blocked waiting on your task 3.
Task 3 fills in the rest of this module.

Create `src/rlx/envs.py`:

```python
"""Environment naming and difficulty. FROZEN CONTRACT -- Daniel's analysis
imports difficulty_index. Task 3 adds the factory and grid BFS to this module."""

ENV_IDS = (
    "Empty-5", "Empty-8", "Empty-16",
    "DoorKey-5", "DoorKey-6", "DoorKey-7", "DoorKey-8", "DoorKey-10",
    "MultiRoom-N2", "MultiRoom-N3", "MultiRoom-N4", "MultiRoom-N5", "MultiRoom-N6",
)


def difficulty_index(env_id: str) -> int:
    """Grid size for Empty/DoorKey, room count for MultiRoom."""
    return int(env_id.split("-")[1].lstrip("N"))
```

Add `tests/test_env_naming.py`:

```python
from rlx.envs import ENV_IDS, difficulty_index


def test_there_are_thirteen_instances():
    assert len(ENV_IDS) == 13


def test_difficulty_increases_within_a_family():
    assert difficulty_index("DoorKey-5") < difficulty_index("DoorKey-10")
    assert difficulty_index("MultiRoom-N2") < difficulty_index("MultiRoom-N6")


def test_every_env_id_has_a_difficulty():
    assert all(difficulty_index(e) > 0 for e in ENV_IDS)
```

```bash
pytest tests/test_env_naming.py -v
```

Expected: 3 passed.

- [ ] **Step 15: Write `configs/main.yaml`**

The full experiment matrix. `sweep.py` (Task 6) reads this.

```yaml
# The full 260-run matrix: 13 environment instances x 4 strategies x 5 seeds.
defaults:
  total_steps: 400000     # provisional -- confirmed by Task 2's benchmark
  device: cpu             # provisional -- confirmed by Task 2's benchmark
  results_root: results

env_ids:
  - Empty-5
  - Empty-8
  - Empty-16
  - DoorKey-5
  - DoorKey-6
  - DoorKey-7
  - DoorKey-8
  - DoorKey-10
  - MultiRoom-N2
  - MultiRoom-N3
  - MultiRoom-N4
  - MultiRoom-N5
  - MultiRoom-N6

strategies:
  - epsilon_greedy
  - boltzmann
  - count_based
  - noisy

seeds: [0, 1, 2, 3, 4]
```

- [ ] **Step 16: Create the empty directories the other two need**

```bash
mkdir -p src/rlx/analysis report/figures
touch src/rlx/analysis/__init__.py
```

- [ ] **Step 17: Run the whole suite**

```bash
pytest -v
```

Expected: 8 passed. Read the output. If anything failed, fix it and re-run before
continuing.

- [ ] **Step 18: Log the change**

Append to `docs/decision_log.md`, in plain language for the team:

```markdown
## 2026-08-17 — Package skeleton and the three frozen interfaces

**Status:** Active

**What changed:** Created the `rlx` package, the run configuration, and the
exploration-strategy interface. These three things are now "frozen", meaning we
agreed not to change them without telling each other.

**Why:** All three of us are writing code at the same time. If we each invented
our own idea of what a config looks like, nothing would fit together on
integration day. Agreeing on the shapes first means we can work independently.

**What it means for the results:** Nothing. Plumbing.
```

- [ ] **Step 19: Commit and merge**

```bash
git add -A
git commit -m "feat: package scaffold and frozen contracts"
```

Open a PR to `main`, merge it, and **tell Max and Daniel it is merged.** They
are blocked until this is on `main`.
