# Task 1 — Visitation logging and the result writer

**Samuel's training loop imports this.** Do it first, merge it early, tell him.

**Files:**
- Create: `src/rlx/logging.py`
- Test: `tests/test_logging.py`

**Interfaces:**
- Consumes: `RunConfig` (Samuel task 1).
- Produces: `RunLogger` with exactly the six members listed in
  `00-overview.md`. **Frozen** — his training loop calls all of them.

---

## What this does

Two jobs, one class:

1. **Count where the agent goes.** A `(W, H, 4)` integer array, one counter per
   `(x, y, direction)`. Bumped once per environment step. Snapshotted
   periodically so we can see coverage *grow over time*, not just its final
   value.
2. **Write the result directory** in the exact format from `CLAUDE.md` §5.

**Why snapshots and not just the final array:** our central hypothesis is about
*early* coverage predicting *final* performance. That needs coverage as a
function of time, which means periodic snapshots.

**Why the directory is written atomically:** the sweep runner skips any run whose
directory exists. If a crashed run left a half-written directory behind, the
sweep would skip it forever and we would silently lose a data point. So we write
to `seed3.partial/` and rename to `seed3/` only on success. **A directory that
exists is a run that finished.**

Sizes are tiny: `16 x 16 x 4 = 1024` integers per snapshot, ~40 snapshots per run.

---

- [ ] **Step 1: Write the failing tests**

Create `tests/test_logging.py`:

```python
import json

import numpy as np
import pandas as pd
import pytest

from rlx.config import RunConfig
from rlx.logging import RunLogger


@pytest.fixture
def cfg(tmp_path):
    return RunConfig(env_id="Empty-5", strategy="epsilon_greedy", seed=0,
                     results_root=str(tmp_path), snapshot_every=100)


def test_distinct_states_counts_unique_position_direction_triples(cfg):
    log = RunLogger(cfg, width=5, height=5)
    assert log.distinct_states() == 0
    log.record_visit(1, 1, 0)
    log.record_visit(1, 1, 0)          # same triple again
    assert log.distinct_states() == 1
    log.record_visit(1, 1, 1)          # same cell, different direction
    assert log.distinct_states() == 2
    log.record_visit(2, 1, 0)
    assert log.distinct_states() == 3


def test_counts_accumulate_per_triple(cfg):
    log = RunLogger(cfg, width=5, height=5)
    for _ in range(7):
        log.record_visit(2, 3, 1)
    log.snapshot(100)
    log.finalize({"completed": True})
    data = np.load(cfg.run_dir / "visitation.npz")
    assert data["counts"][0][2, 3, 1] == 7


def test_snapshots_are_cumulative_and_ordered(cfg):
    log = RunLogger(cfg, width=5, height=5)
    log.record_visit(1, 1, 0)
    log.snapshot(100)
    log.record_visit(2, 2, 0)
    log.snapshot(200)
    log.finalize({"completed": True})

    data = np.load(cfg.run_dir / "visitation.npz")
    assert list(data["steps"]) == [100, 200]
    assert data["counts"].shape == (2, 5, 5, 4)
    assert data["counts"][0].sum() == 1        # cumulative, not per-interval
    assert data["counts"][1].sum() == 2


def test_metrics_csv_has_one_row_per_log_step(cfg):
    log = RunLogger(cfg, width=5, height=5)
    log.log_step(0, eval_return_mean=0.0, distinct_states=1)
    log.log_step(500, eval_return_mean=0.5, distinct_states=9)
    log.finalize({"completed": True})

    df = pd.read_csv(cfg.run_dir / "metrics.csv")
    assert len(df) == 2
    assert list(df["step"]) == [0, 500]
    assert df["eval_return_mean"].iloc[-1] == 0.5


def test_varying_scalar_keys_do_not_break_the_csv(cfg):
    """epsilon_greedy logs 'epsilon', count_based logs 'mean_bonus'."""
    log = RunLogger(cfg, width=5, height=5)
    log.log_step(0, eval_return_mean=0.0, epsilon=1.0)
    log.log_step(1, eval_return_mean=0.1, mean_bonus=0.05)
    log.finalize({"completed": True})

    df = pd.read_csv(cfg.run_dir / "metrics.csv")
    assert "epsilon" in df.columns and "mean_bonus" in df.columns
    assert pd.isna(df["mean_bonus"].iloc[0])


def test_finalize_writes_all_four_files_with_valid_content(cfg):
    log = RunLogger(cfg, width=5, height=5)
    log.log_step(0, eval_return_mean=0.0)
    log.snapshot(100)
    log.finalize({"git_sha": "abc123", "completed": True})

    for name in ("config.json", "metrics.csv", "visitation.npz", "meta.json"):
        assert (cfg.run_dir / name).exists(), name

    saved = json.loads((cfg.run_dir / "config.json").read_text())
    assert saved["env_id"] == "Empty-5"
    assert saved["seed"] == 0

    meta = json.loads((cfg.run_dir / "meta.json").read_text())
    assert meta["git_sha"] == "abc123"
    assert meta["completed"] is True


def test_nothing_is_visible_at_the_final_path_until_finalize(cfg):
    log = RunLogger(cfg, width=5, height=5)
    log.log_step(0, eval_return_mean=0.0)
    log.snapshot(100)
    assert not cfg.run_dir.exists()
    log.finalize({"completed": True})
    assert cfg.run_dir.exists()


def test_no_partial_directory_survives_finalize(cfg):
    log = RunLogger(cfg, width=5, height=5)
    log.snapshot(100)
    log.finalize({"completed": True})
    assert not list(cfg.run_dir.parent.glob("*.partial"))
```

- [ ] **Step 2: Run and watch them fail**

```bash
pytest tests/test_logging.py -v
```

Expected: `ModuleNotFoundError: No module named 'rlx.logging'`.

- [ ] **Step 3: Write `src/rlx/logging.py`**

```python
"""Visitation logging and the result-directory writer.

We log the agent's TRUE (x, y, direction). That is privileged information the
agent never receives -- it is measurement, not learning. See CLAUDE.md section 8.

Only raw counts are written. Every coverage metric is derived later in
rlx.analysis.coverage, so metric definitions can change without re-running
experiments.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from rlx.config import RunConfig

N_DIRECTIONS = 4


class RunLogger:
    """Accumulates visit counts and metric rows, then writes one result directory."""

    def __init__(self, cfg: RunConfig, width: int, height: int):
        self.cfg = cfg
        self.counts = np.zeros((width, height, N_DIRECTIONS), dtype=np.int32)
        self._snapshot_steps: list[int] = []
        self._snapshots: list[np.ndarray] = []
        self._rows: list[dict] = []
        self._partial = cfg.run_dir.with_name(cfg.run_dir.name + ".partial")

    def record_visit(self, x: int, y: int, direction: int) -> None:
        self.counts[x, y, direction] += 1

    def distinct_states(self) -> int:
        """Number of distinct (x, y, direction) triples seen at least once."""
        return int((self.counts > 0).sum())

    def log_step(self, step: int, **scalars: float) -> None:
        self._rows.append({"step": step, **scalars})

    def snapshot(self, step: int) -> None:
        """Store a copy of the cumulative count array at this step."""
        self._snapshot_steps.append(step)
        self._snapshots.append(self.counts.copy())

    def finalize(self, meta: dict) -> None:
        """Write the result directory atomically: build .partial, then rename."""
        if self._partial.exists():
            shutil.rmtree(self._partial)
        self._partial.mkdir(parents=True)

        (self._partial / "config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))
        (self._partial / "meta.json").write_text(json.dumps(meta, indent=2))
        pd.DataFrame(self._rows).to_csv(self._partial / "metrics.csv", index=False)
        np.savez_compressed(
            self._partial / "visitation.npz",
            steps=np.array(self._snapshot_steps, dtype=np.int64),
            counts=(np.stack(self._snapshots) if self._snapshots
                    else np.zeros((0, *self.counts.shape), dtype=np.int32)),
        )

        if self.cfg.run_dir.exists():
            shutil.rmtree(self.cfg.run_dir)
        self._partial.rename(self.cfg.run_dir)
```

`pd.DataFrame(self._rows)` handles the varying-keys case for free: missing keys
become `NaN` columns, which is exactly what the test expects.

- [ ] **Step 4: Run the tests**

```bash
pytest tests/test_logging.py -v
```

Expected: 8 passed.

If `test_snapshots_are_cumulative_and_ordered` fails with both snapshots equal,
`self.counts.copy()` was written as `self.counts` — every snapshot would then be
a view of the same array and they would all show the final state.

- [ ] **Step 5: Merge and tell Samuel**

```bash
git add src/rlx/logging.py tests/test_logging.py
git commit -m "feat: visitation logger and atomic result writer"
```

Open the PR, merge it, and **tell Samuel it is on main** — his training loop
imports `RunLogger` and he is stubbing it until yours lands.

- [ ] **Step 6: Log the change**

Append to `docs/decision_log.md`:

```markdown
## 2026-08-18 — Visit logging, and why results are written atomically

**Status:** Active

**What changed:** During a run we now record where the agent actually is —
its x, y position and which of the 4 directions it faces — and count how often it
has been in each of those situations. We save a snapshot of those counts every
10,000 steps.

**Why snapshots rather than one final total:** our main question is whether
exploring a lot *early* predicts doing well *later*. That needs coverage measured
over time, not just at the end.

**Why we compute no coverage percentages during the run:** we only save raw
counts. Every actual metric is worked out afterwards from the saved files. That
means if we decide on the last day that our definition of coverage was wrong, we
fix the analysis instead of re-running 260 experiments.

**Why the result folder is written in a funny way:** a run first writes to a
folder ending in `.partial`, and only renames it to the real name once it has
finished successfully. The sweep runner skips any run whose folder already
exists, so if a crashed run left half a folder behind, that experiment would be
skipped forever and we would quietly lose a data point without noticing. This way,
a folder existing always means a run that genuinely finished.

**What it means for the results:** Nothing scientific. It stops us losing data.
```

```bash
git add docs/decision_log.md && git commit -m "docs: log visitation logging decisions"
```
