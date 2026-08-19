"""Load result directories into memory and flatten them into a tidy table."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rlx.envs import difficulty_index

REQUIRED_FILES = ("metrics.csv", "visitation.npz", "config.json")

#: Report order for the three families, as CLAUDE.md section 7 lists them. Lives
#: here rather than in figures.py because results.md needs the same order and the
#: two must not disagree.
FAMILIES = ("Empty", "DoorKey", "MultiRoom")


@dataclass
class RunResult:
    env_id: str
    strategy: str
    seed: int
    metrics: pd.DataFrame
    steps: np.ndarray
    counts: np.ndarray          # (T, W, H, 4) cumulative visit counts
    config: dict


def _read_metrics(path: Path) -> pd.DataFrame:
    """A run finalized without any log_step leaves a file with no header row.

    Analysis must never crash on a directory that exists -- an existing directory
    is a finished run by the CLAUDE.md section 5 contract. Such a run has no
    evaluations, so final_return on it is NaN, which is the honest answer.
    """
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["step", "eval_return_mean"])


def load_run(run_dir: Path) -> RunResult:
    """Load one result directory. The identity comes from the path."""
    run_dir = Path(run_dir)
    data = np.load(run_dir / "visitation.npz")
    return RunResult(
        env_id=run_dir.parent.parent.name,
        strategy=run_dir.parent.name,
        seed=int(run_dir.name.removeprefix("seed")),
        metrics=_read_metrics(run_dir / "metrics.csv"),
        steps=data["steps"],
        counts=data["counts"],
        config=json.loads((run_dir / "config.json").read_text()),
    )


def load_all(results_root: Path) -> list[RunResult]:
    """Load every complete run under results_root, sorted for determinism.

    Skips `seed<k>.partial`: a crashed run leaves one behind with all of
    REQUIRED_FILES already written, because those are written before the rename.
    Only the rename marks a run as finished, so the name is the only reliable
    signal -- see CLAUDE.md section 5.
    """
    runs = []
    for run_dir in sorted(Path(results_root).glob("*/*/seed*")):
        if not run_dir.name.removeprefix("seed").isdigit():
            continue
        if all((run_dir / f).exists() for f in REQUIRED_FILES):
            runs.append(load_run(run_dir))
    return runs


def final_return(metrics: pd.DataFrame, n_tail: int = 5) -> float:
    """Mean evaluation return over the last n_tail EVALUATION points.

    Averaging the tail rather than taking the final point reduces the effect of
    one noisy evaluation. NaNs are dropped first: metrics.csv holds one row per
    logged step and training may log more often than it evaluates, so most rows
    can have an empty eval_return_mean. Taking the tail of the raw column would
    then silently average fewer points than asked for and quietly undo the noise
    reduction this function exists for.
    """
    return float(metrics["eval_return_mean"].dropna().tail(n_tail).mean())


def ordered_instances(df: pd.DataFrame) -> list[str]:
    """Instance names in report order: family first, then difficulty.

    Sorting the names alphabetically puts DoorKey-10 before DoorKey-5 and
    Empty-16 before Empty-5, so difficulty does not read left to right -- and the
    tables in results.md came out in a different order from the figures beside
    them. Both now call this.
    """
    def key(env_id):
        return (FAMILIES.index(env_id.split("-")[0]), difficulty_index(env_id))

    return sorted(df["env_id"].unique(), key=key)


def to_dataframe(runs: list[RunResult]) -> pd.DataFrame:
    """One row per run: identity, difficulty, and final return.

    `difficulty` is comparable only WITHIN a family: it is the grid size for
    Empty and DoorKey but the room count for MultiRoom, so Empty-16 outranks
    MultiRoom-N6 on a number that means something different in each. Group by
    `family` before using it. See CLAUDE.md section 9.
    """
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
