# Task 2 — Results aggregation and synthetic data

After this task you are **independent of everyone**. The synthetic generator
produces fake result directories in the real format, so tasks 3–5 need no real
experiments.

**Files:**
- Create: `src/rlx/analysis/aggregate.py`
- Create: `scripts/make_synthetic_results.py`
- Test: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: the results format (`CLAUDE.md` §5), `ENV_IDS` / `difficulty_index`
  (Samuel task 3).
- Produces:
  - `load_run(run_dir) -> RunResult` (fields: `env_id, strategy, seed, metrics, steps, counts, config`)
  - `load_all(results_root) -> list[RunResult]`
  - `to_dataframe(runs) -> pd.DataFrame` — one row per run
  - `final_return(metrics, n_tail=5) -> float`

---

- [ ] **Step 1: Write the synthetic results generator**

Create `scripts/make_synthetic_results.py`. This is your development fixture for
the rest of the week — build it first.

```python
"""Generate fake result directories in the real format, for developing analysis
before any real experiment exists.

Run:  python scripts/make_synthetic_results.py --out results_synthetic
"""

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ENV_IDS = ["Empty-5", "Empty-8", "DoorKey-5", "DoorKey-8", "MultiRoom-N2", "MultiRoom-N4"]
STRATEGIES = ["epsilon_greedy", "boltzmann", "count_based", "noisy"]
SEEDS = [0, 1, 2, 3, 4]

TOTAL_STEPS = 400_000
EVAL_EVERY = 5_000
SNAPSHOT_EVERY = 10_000

#: Baked-in ground truth: strategies that explore more early also score higher,
#: and the effect is stronger on harder mazes. Task 4's analysis must recover it.
EXPLORE_RATE = {"epsilon_greedy": 1.0, "boltzmann": 1.15, "count_based": 1.4, "noisy": 1.25}


def _difficulty(env_id: str) -> float:
    return {"Empty": 0.2, "DoorKey": 0.6, "MultiRoom": 1.0}[env_id.split("-")[0]]


def _stable_seed(env_id: str, strategy: str, seed: int) -> int:
    """Deterministic across processes.

    Do NOT use hash((env_id, strategy, seed)): Python salts string hashing per
    process, so the 'same' synthetic dataset would differ between runs and you
    could not reproduce a plot you were looking at yesterday.
    """
    return zlib.crc32(f"{env_id}|{strategy}|{seed}".encode())


def make_run(out_root: Path, env_id: str, strategy: str, seed: int) -> None:
    rng = np.random.default_rng(_stable_seed(env_id, strategy, seed))
    w = h = 12
    difficulty = _difficulty(env_id)
    rate = EXPLORE_RATE[strategy] * rng.normal(1.0, 0.12)

    eval_steps = np.arange(0, TOTAL_STEPS, EVAL_EVERY)
    coverage = 1.0 - np.exp(-rate * eval_steps / (60_000 * (1 + 2 * difficulty)))
    early_auc = coverage[: len(coverage) // 5].mean()

    solved = rng.random() < np.clip(1.3 * early_auc - difficulty * 0.7, 0.02, 0.97)
    ceiling = (0.9 if solved else 0.0) * rng.uniform(0.9, 1.0)
    returns = ceiling / (1 + np.exp(-(eval_steps - TOTAL_STEPS * 0.4) / 40_000))

    metrics = pd.DataFrame({
        "step": eval_steps,
        "eval_return_mean": returns,
        "eval_return_std": np.zeros_like(returns),
        "train_return_mean": returns * 0.8,
        "episodes": (eval_steps / 200).astype(int),
        "distinct_states": (coverage * w * h * 4).astype(int),
        "loss": rng.random(len(eval_steps)) * 0.1,
        "epsilon": np.linspace(1.0, 0.05, len(eval_steps)),
    })

    snap_steps = np.arange(SNAPSHOT_EVERY, TOTAL_STEPS + 1, SNAPSHOT_EVERY)
    counts = np.zeros((len(snap_steps), w, h, 4), dtype=np.int32)
    order = rng.permutation(w * h * 4)
    for i, step in enumerate(snap_steps):
        frac = 1.0 - np.exp(-rate * step / (60_000 * (1 + 2 * difficulty)))
        visited = order[: int(frac * len(order))]
        flat = np.zeros(w * h * 4, dtype=np.int32)
        flat[visited] = rng.integers(1, 40, size=len(visited))
        counts[i] = flat.reshape(w, h, 4)

    run_dir = out_root / env_id / strategy / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    np.savez_compressed(run_dir / "visitation.npz", steps=snap_steps, counts=counts)
    (run_dir / "config.json").write_text(json.dumps(
        {"env_id": env_id, "strategy": strategy, "seed": seed,
         "total_steps": TOTAL_STEPS, "snapshot_every": SNAPSHOT_EVERY}, indent=2))
    (run_dir / "meta.json").write_text(json.dumps(
        {"git_sha": "synthetic", "hostname": "synthetic",
         "device": "cpu", "wall_time_s": 0.0, "completed": True}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results_synthetic"))
    args = p.parse_args()

    n = 0
    for env_id in ENV_IDS:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                make_run(args.out, env_id, strategy, seed)
                n += 1
    print(f"wrote {n} synthetic runs to {args.out}")
```

- [ ] **Step 2: Generate the synthetic data and look at it**

```bash
python scripts/make_synthetic_results.py --out results_synthetic
```

Expected: `wrote 120 synthetic runs`.

**Read one of the files to check the format is right:**

```bash
python -c "
import pandas as pd, numpy as np
df = pd.read_csv('results_synthetic/DoorKey-8/count_based/seed0/metrics.csv')
print(df.head())
d = np.load('results_synthetic/DoorKey-8/count_based/seed0/visitation.npz')
print('steps', d['steps'][:5], 'counts', d['counts'].shape)
"
```

Add `results_synthetic/` to `.gitignore`.

- [ ] **Step 3: Write the failing tests**

Create `tests/test_aggregate.py`:

```python
import numpy as np
import pandas as pd
import pytest

from rlx.analysis.aggregate import final_return, load_all, load_run, to_dataframe


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    import subprocess, sys
    out = tmp_path_factory.mktemp("synth")
    subprocess.run([sys.executable, "scripts/make_synthetic_results.py", "--out", str(out)],
                   check=True)
    return load_all(out)


def test_all_synthetic_runs_load(runs):
    assert len(runs) == 120


def test_identity_is_parsed_from_the_directory_path(runs):
    r = next(r for r in runs if r.env_id == "DoorKey-8"
             and r.strategy == "count_based" and r.seed == 0)
    assert isinstance(r.metrics, pd.DataFrame)
    assert r.counts.ndim == 4
    assert len(r.steps) == r.counts.shape[0]


def test_dataframe_has_one_row_per_run_with_the_expected_columns(runs):
    df = to_dataframe(runs)
    assert len(df) == len(runs)
    for col in ("env_id", "strategy", "seed", "final_return", "difficulty"):
        assert col in df.columns


def test_difficulty_increases_within_a_family(runs):
    df = to_dataframe(runs)
    d = df.groupby("env_id")["difficulty"].first()
    assert d["DoorKey-5"] < d["DoorKey-8"]


def test_final_return_averages_the_tail_not_the_last_point():
    metrics = pd.DataFrame({"step": range(10), "eval_return_mean": [0] * 5 + [1.0] * 5})
    assert final_return(metrics, n_tail=5) == 1.0
    assert final_return(metrics, n_tail=10) == 0.5


def test_incomplete_runs_are_skipped(tmp_path):
    (tmp_path / "Empty-5" / "epsilon_greedy" / "seed0").mkdir(parents=True)
    assert load_all(tmp_path) == []
```

- [ ] **Step 4: Run and watch them fail**

```bash
pytest tests/test_aggregate.py -v
```

- [ ] **Step 5: Write `src/rlx/analysis/aggregate.py`**

```python
"""Load result directories into memory and flatten them into a tidy table."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rlx.envs import difficulty_index

REQUIRED_FILES = ("metrics.csv", "visitation.npz", "config.json")


@dataclass
class RunResult:
    env_id: str
    strategy: str
    seed: int
    metrics: pd.DataFrame
    steps: np.ndarray
    counts: np.ndarray          # (T, W, H, 4) cumulative visit counts
    config: dict


def load_run(run_dir: Path) -> RunResult:
    """Load one result directory. The identity comes from the path."""
    data = np.load(run_dir / "visitation.npz")
    return RunResult(
        env_id=run_dir.parent.parent.name,
        strategy=run_dir.parent.name,
        seed=int(run_dir.name.removeprefix("seed")),
        metrics=pd.read_csv(run_dir / "metrics.csv"),
        steps=data["steps"],
        counts=data["counts"],
        config=json.loads((run_dir / "config.json").read_text()),
    )


def load_all(results_root: Path) -> list[RunResult]:
    """Load every complete run under results_root, sorted for determinism."""
    runs = []
    for run_dir in sorted(Path(results_root).glob("*/*/seed*")):
        if all((run_dir / f).exists() for f in REQUIRED_FILES):
            runs.append(load_run(run_dir))
    return runs


def final_return(metrics: pd.DataFrame, n_tail: int = 5) -> float:
    """Mean evaluation return over the last n_tail evaluation points.

    Averaging the tail rather than taking the final point reduces the effect of
    one noisy evaluation.
    """
    return float(metrics["eval_return_mean"].tail(n_tail).mean())


def to_dataframe(runs: list[RunResult]) -> pd.DataFrame:
    """One row per run: identity, difficulty, and final return."""
    return pd.DataFrame([
        {
            "env_id": r.env_id,
            "family": r.env_id.split("-")[0],
            "difficulty": difficulty_index(r.env_id),
            "strategy": r.strategy,
            "seed": r.seed,
            "final_return": final_return(r.metrics),
        }
        for r in runs
    ])
```

- [ ] **Step 6: Run the tests**

```bash
pytest tests/test_aggregate.py -v
```

Expected: 6 passed.

- [ ] **Step 7: Log the change**

Append to `docs/decision_log.md`, in plain language:

```markdown
## 2026-08-18 — We generate fake results to build the analysis against

**Status:** Active

**What changed:** Wrote a script that produces fake experiment folders in exactly
the same format as real ones. All of the analysis code is developed and tested
against these before a single real experiment exists.

**Why:** Three of us are working at once and the real results do not arrive until
the 21st. Without fake data, the analysis work could not start until then, and we
would be debugging plots and statistics on the last two days with no slack.

**The useful part:** the fake data has a known answer built into it — strategies
that explore more early are *made* to score better, and more so on harder mazes.
So when the statistics in task 4 run, we already know what number they should
produce. If they produce something else, the analysis is broken, and we find that
out on the 18th instead of the 22nd.

**What it means for the results:** Nothing goes into the report from this. It is
scaffolding, and the fake folders are gitignored.
```

- [ ] **Step 8: Commit**

```bash
git add src/rlx/analysis/aggregate.py scripts/make_synthetic_results.py tests/test_aggregate.py .gitignore docs/decision_log.md
git commit -m "feat: results aggregation and synthetic data generator"
```
