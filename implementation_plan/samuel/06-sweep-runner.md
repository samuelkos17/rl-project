# Task 6 — Sharded parallel sweep runner

Runs the 260-run matrix across three machines with no coordination between them.

**Files:**
- Create: `src/rlx/sweep.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: `RunConfig` (Task 1), `run_training` (Task 5), `configs/main.yaml` (Task 1).
- Produces: `expand_matrix(spec) -> list[RunConfig]`, `select_shard(configs, i, n)`,
  and the `python -m rlx.sweep` CLI.

**The design in one sentence:** the matrix is a deterministic ordered list, each
machine takes every n-th entry, and any run whose result directory already exists
is skipped. No shared state, no locks, no possibility of conflict.

---

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sweep.py`:

```python
from rlx.sweep import expand_matrix, pending_runs, select_shard

SPEC = {
    "defaults": {"total_steps": 1000, "device": "cpu"},
    "env_ids": ["Empty-5", "DoorKey-5"],
    "strategies": ["epsilon_greedy", "noisy"],
    "seeds": [0, 1],
}


def test_matrix_is_the_full_cross_product():
    assert len(expand_matrix(SPEC)) == 2 * 2 * 2


def test_defaults_are_applied_to_every_config():
    assert all(c.total_steps == 1000 for c in expand_matrix(SPEC))


def test_matrix_order_is_deterministic():
    a = [(c.env_id, c.strategy, c.seed) for c in expand_matrix(SPEC)]
    b = [(c.env_id, c.strategy, c.seed) for c in expand_matrix(SPEC)]
    assert a == b


def test_shards_partition_the_matrix_exactly_once():
    all_configs = expand_matrix(SPEC)
    covered = []
    for i in range(3):
        covered.extend(select_shard(all_configs, i, 3))
    keys = sorted((c.env_id, c.strategy, c.seed) for c in covered)
    expected = sorted((c.env_id, c.strategy, c.seed) for c in all_configs)
    assert keys == expected


def test_shards_are_roughly_balanced():
    all_configs = expand_matrix(SPEC)
    sizes = [len(select_shard(all_configs, i, 3)) for i in range(3)]
    assert max(sizes) - min(sizes) <= 1


def test_completed_runs_are_skipped(tmp_path):
    spec = {**SPEC, "defaults": {**SPEC["defaults"], "results_root": str(tmp_path)}}
    configs = expand_matrix(spec)
    done = configs[0]
    done.run_dir.mkdir(parents=True)

    remaining = pending_runs(configs)
    assert len(remaining) == len(configs) - 1
    assert done.run_dir not in [c.run_dir for c in remaining]
```

- [ ] **Step 2: Run and watch them fail**

```bash
pytest tests/test_sweep.py -v
```

Expected: `ModuleNotFoundError: No module named 'rlx.sweep'`.

- [ ] **Step 3: Write `src/rlx/sweep.py`**

```python
"""Run the experiment matrix in parallel, sharded across machines.

Each machine runs:   python -m rlx.sweep --config configs/main.yaml --shard i/3 --workers 8

Shards partition a deterministically ordered matrix, so the three machines never
need to talk to each other. Any run whose result directory already exists is
skipped, which makes a sweep resumable after a crash and safe to relaunch.
"""

import argparse
import concurrent.futures as cf
import traceback
from pathlib import Path

import yaml

from rlx.config import RunConfig
from rlx.train import run_training


def expand_matrix(spec: dict) -> list[RunConfig]:
    """Cross product of env_ids x strategies x seeds, in a fixed order."""
    defaults = spec.get("defaults", {})
    return [
        RunConfig(env_id=env_id, strategy=strategy, seed=seed, **defaults)
        for env_id in spec["env_ids"]
        for strategy in spec["strategies"]
        for seed in spec["seeds"]
    ]


def select_shard(configs: list[RunConfig], index: int, total: int) -> list[RunConfig]:
    """Every total-th config starting at index. Balanced to within one run."""
    if not 0 <= index < total:
        raise ValueError(f"shard index {index} out of range for {total} shards")
    return configs[index::total]


def pending_runs(configs: list[RunConfig]) -> list[RunConfig]:
    """Drop configs whose result directory already exists.

    Result directories are written atomically by RunLogger.finalize, so an
    existing directory always means a finished run.
    """
    return [c for c in configs if not c.run_dir.exists()]


def _run_one(cfg: RunConfig) -> str:
    try:
        run_training(cfg)
        return f"ok    {cfg.env_id:<14} {cfg.strategy:<15} seed{cfg.seed}"
    except Exception:
        return (f"FAIL  {cfg.env_id:<14} {cfg.strategy:<15} seed{cfg.seed}\n"
                + traceback.format_exc())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--shard", default="0/1", help="i/n, e.g. 0/3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        spec = yaml.safe_load(f)

    index, total = (int(v) for v in args.shard.split("/"))
    configs = pending_runs(select_shard(expand_matrix(spec), index, total))

    print(f"shard {index}/{total}: {len(configs)} runs pending, {args.workers} workers")
    if args.dry_run:
        for c in configs:
            print(f"  {c.env_id:<14} {c.strategy:<15} seed{c.seed}")
        return

    failures = 0
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, line in enumerate(pool.map(_run_one, configs), start=1):
            print(f"[{i}/{len(configs)}] {line}", flush=True)
            failures += line.startswith("FAIL")

    print(f"\nfinished: {len(configs) - failures} ok, {failures} failed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

```bash
pytest tests/test_sweep.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Check the matrix without running anything**

```bash
python -m rlx.sweep --config configs/main.yaml --shard 0/3 --dry-run
```

Expected: `shard 0/3: 87 runs pending` (260 split three ways gives 87, 87, 86)
and a list of triples. If the count is not 260 in total, `configs/main.yaml` is
wrong — fix the config, not the code.

- [ ] **Step 6: Time a small real sweep**

```bash
python -m rlx.sweep --config configs/pilot.yaml --shard 0/1 --workers 8
```

First create `configs/pilot.yaml` — the same shape as `main.yaml` but tiny, for
integration day:

```yaml
# Pilot sweep: cheap end-to-end check of the whole pipeline before the real run.
defaults:
  total_steps: 20000
  device: cpu
  results_root: results_pilot

env_ids: [Empty-5, DoorKey-5]
strategies: [epsilon_greedy, boltzmann, count_based, noisy]
seeds: [0, 1]
```

16 runs. **Time it, and read the per-worker throughput.** Multiply out to
estimate the full sweep. If the projection exceeds 6 hours, reduce `total_steps`
in `main.yaml` now rather than discovering it overnight.

- [ ] **Step 7: Check the pilot results actually learned something**

```bash
python -c "
import pandas as pd, glob
for f in sorted(glob.glob('results_pilot/*/*/*/metrics.csv')):
    df = pd.read_csv(f)
    print(f'{f:<60} final_eval={df.eval_return_mean.iloc[-1]:.3f}')
"
```

On `Empty-5` every strategy should end well above 0. If one strategy is flat at
0.0 while the others learn, that strategy is broken — tell Max, with the numbers.

- [ ] **Step 8: Log and commit**

Append a `docs/decision_log.md` entry with the measured pilot timing and the
projected full-sweep wall clock, in plain language, plus the final `total_steps`
if you changed it.

```bash
git add src/rlx/sweep.py configs/pilot.yaml tests/test_sweep.py docs/decision_log.md
git commit -m "feat: sharded parallel sweep runner"
```

- [ ] **Step 9: Launch the real sweep**

One command per machine, run them at the same time:

```bash
python -m rlx.sweep --config configs/main.yaml --shard 0/3 --workers 8
```

Machine 2 uses `--shard 1/3`, machine 3 uses `--shard 2/3`. When all three
finish, collect the `results/` directories onto one machine by copying — the
directory trees are disjoint, so copying cannot conflict.
