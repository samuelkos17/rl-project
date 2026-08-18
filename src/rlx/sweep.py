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
    if total < 1:
        raise ValueError(f"shard count must be >= 1, got {total}")
    if not 0 <= index < total:
        raise ValueError(f"shard index {index} out of range for {total} shards")
    return configs[index::total]


def pending_runs(configs: list[RunConfig]) -> list[RunConfig]:
    """Drop configs whose result directory already exists.

    Result directories are written atomically by RunLogger.finalize (build
    `seed<k>.partial/`, then rename), so an existing directory always means a
    finished run and a leftover `.partial` never counts as done.
    """
    return [c for c in configs if not c.run_dir.exists()]


def _run_one(cfg: RunConfig) -> str:
    """Run one config in a worker process. Never raises -- one broken run must
    not take the other 259 down with it."""
    label = f"{cfg.env_id:<14} {cfg.strategy:<15} seed{cfg.seed}"
    try:
        run_training(cfg)
        return f"ok    {label}"
    except Exception:
        return f"FAIL  {label}\n{traceback.format_exc()}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--shard", default="0/1", help="i/n, e.g. 0/3")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        spec = yaml.safe_load(f)

    index, total = (int(v) for v in args.shard.split("/"))
    all_configs = expand_matrix(spec)
    shard = select_shard(all_configs, index, total)
    configs = pending_runs(shard)

    done = len(shard) - len(configs)
    print(f"shard {index}/{total}: {len(shard)} runs assigned, "
          f"{done} already done, {len(configs)} pending, {args.workers} workers")
    if args.dry_run:
        for c in configs:
            print(f"  {c.env_id:<14} {c.strategy:<15} seed{c.seed}")
        return
    if not configs:
        print("nothing to do")
        return

    failures = 0
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, line in enumerate(pool.map(_run_one, configs), start=1):
            print(f"[{i}/{len(configs)}] {line}", flush=True)
            failures += line.startswith("FAIL")

    print(f"\nfinished: {len(configs) - failures} ok, {failures} failed")
    if failures:
        print("re-run the same command to retry only the failures "
              "(finished runs are skipped)")


if __name__ == "__main__":
    main()
