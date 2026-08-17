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

from rlx.config import RunConfig
from rlx.envs import difficulty_index

#: A 6-instance subset of the real 13. Not rlx.envs.ENV_IDS -- the fixture only
#: needs enough instances to cover all three families at two difficulties each.
FIXTURE_ENV_IDS = ["Empty-5", "Empty-8", "DoorKey-5", "DoorKey-8",
                   "MultiRoom-N2", "MultiRoom-N4"]
STRATEGIES = ["epsilon_greedy", "boltzmann", "count_based", "noisy"]
SEEDS = [0, 1, 2, 3, 4]

#: Read off RunConfig rather than restated, so the fixture cannot drift away from
#: the real experiment settings.
_DEFAULTS = RunConfig(env_id="Empty-5", strategy="epsilon_greedy", seed=0)
TOTAL_STEPS = _DEFAULTS.total_steps
EVAL_EVERY = _DEFAULTS.eval_every
SNAPSHOT_EVERY = _DEFAULTS.snapshot_every

#: Baked-in ground truth: strategies that explore more early also score higher,
#: and the effect is stronger on harder mazes. Task 4's analysis must recover it.
EXPLORE_RATE = {"epsilon_greedy": 1.0, "boltzmann": 1.15, "count_based": 1.4, "noisy": 1.25}


def _difficulty(env_id: str) -> float:
    return {"Empty": 0.2, "DoorKey": 0.6, "MultiRoom": 1.0}[env_id.split("-")[0]]


def _grid_size(env_id: str) -> int:
    """Real MiniGrid dimensions: grid size for Empty/DoorKey, 25x25 for every
    MultiRoom N (verified 2026-08-17).

    Task 3 intersects `counts` with a (W, H) reachability mask from grid_info, so
    a fixture with the wrong shape could not be developed against at all.
    """
    return 25 if env_id.startswith("MultiRoom") else difficulty_index(env_id)


def _early_auc(rate: float, difficulty: float) -> float:
    """Mean coverage over the first fifth of training -- the early-AUC window."""
    steps = np.arange(0, TOTAL_STEPS, EVAL_EVERY)
    coverage = 1.0 - np.exp(-rate * steps / (60_000 * (1 + 2 * difficulty)))
    return float(coverage[: len(coverage) // 5].mean())


def _stable_seed(env_id: str, strategy: str, seed: int) -> int:
    """Deterministic across processes.

    Do NOT use hash((env_id, strategy, seed)): Python salts string hashing per
    process, so the 'same' synthetic dataset would differ between runs and you
    could not reproduce a plot you were looking at yesterday.
    """
    return zlib.crc32(f"{env_id}|{strategy}|{seed}".encode())


def make_run(out_root: Path, env_id: str, strategy: str, seed: int, effect: bool = True) -> None:
    rng = np.random.default_rng(_stable_seed(env_id, strategy, seed))
    w = h = _grid_size(env_id)
    difficulty = _difficulty(env_id)
    rate = EXPLORE_RATE[strategy] * rng.normal(1.0, 0.12)

    eval_steps = np.arange(0, TOTAL_STEPS, EVAL_EVERY)
    coverage = 1.0 - np.exp(-rate * eval_steps / (60_000 * (1 + 2 * difficulty)))
    early_auc = coverage[: len(coverage) // 5].mean()

    # Coverage relative to the epsilon-greedy baseline AT THIS DIFFICULTY. A ratio,
    # not a difference: early_auc shrinks as mazes get harder, so a fixed reference
    # would penalise hard instances twice and clip every strategy to the same floor.
    advantage = early_auc / _early_auc(1.0, difficulty) - 1.0
    gain = (0.5 + difficulty) if effect else 0.0
    ceiling = np.clip(0.75 - 0.40 * difficulty + gain * advantage + rng.normal(0, 0.10),
                      0.0, 0.95)
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

    # Counts accumulate, exactly as RunLogger's do: every snapshot adds visits to
    # the cells seen so far, so the array is monotone non-decreasing over T. The
    # earlier version redrew each snapshot from scratch, which let 37% of cells
    # DECREASE between snapshots -- impossible in a real run.
    snap_steps = np.arange(SNAPSHOT_EVERY, TOTAL_STEPS + 1, SNAPSHOT_EVERY)
    counts = np.zeros((len(snap_steps), w, h, 4), dtype=np.int32)
    order = rng.permutation(w * h * 4)
    flat = np.zeros(w * h * 4, dtype=np.int32)
    for i, step in enumerate(snap_steps):
        frac = 1.0 - np.exp(-rate * step / (60_000 * (1 + 2 * difficulty)))
        n_visited = int(frac * len(order))
        flat[order[:n_visited]] += rng.integers(1, 10, size=n_visited)
        counts[i] = flat.reshape(w, h, 4)

    run_dir = out_root / env_id / strategy / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    np.savez_compressed(run_dir / "visitation.npz", steps=snap_steps, counts=counts)
    # The real 26-field config, not a 5-field stand-in: analysis that reads any
    # config field must behave the same on fixture and real data.
    (run_dir / "config.json").write_text(json.dumps(
        RunConfig(env_id=env_id, strategy=strategy, seed=seed).to_dict(), indent=2))
    # synthetic_effect records WHICH fixture this is. Without it the --no-effect
    # dataset is byte-identical to the real one on disk, so overwriting the good
    # one would make the analysis report "no effect" and look like a finding.
    (run_dir / "meta.json").write_text(json.dumps(
        {"git_sha": "synthetic", "hostname": "synthetic", "device": "cpu",
         "wall_time_s": 0.0, "completed": True, "synthetic_effect": effect}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results_synthetic"))
    p.add_argument("--no-effect", action="store_true",
                   help="Negative control: return is independent of coverage. The "
                        "analysis must report NO within-instance correlation on this "
                        "dataset. Without it we only ever test that the analysis can "
                        "say yes, never that it can say no.")
    args = p.parse_args()

    n = 0
    for env_id in FIXTURE_ENV_IDS:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                make_run(args.out, env_id, strategy, seed, effect=not args.no_effect)
                n += 1
    print(f"wrote {n} synthetic runs to {args.out} (effect={'off' if args.no_effect else 'on'})")
