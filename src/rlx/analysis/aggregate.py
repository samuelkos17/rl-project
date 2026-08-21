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


#: Evaluation points used by success_rate and conditional_return.
#: Wider than final_return's 5 on purpose: those two split one number into a
#: RATE and a conditional mean, and a rate over 5 samples can only take the
#: values 0, 0.2, 0.4, 0.6, 0.8, 1. Twenty points -- the last 100k of 400k
#: steps -- give 0.05 resolution while still describing only the late phase.
N_LATE = 20


def success_rate(metrics: pd.DataFrame, n_tail: int = N_LATE) -> float:
    """Fraction of the last n_tail evaluations that reached the goal.

    Half of the split that replaces final_return as the headline. Measured
    2026-08-21: a greedy evaluation returns exactly 0 on 45.8% of checks made
    AFTER a run has already solved its maze, because the greedy policy jams in a
    cycle and times out. That rate differs sharply by strategy (noisy 24% to
    count-based 67%), so a single averaged return silently mixes "how often does
    the policy work" with "how well does it do when it works". This is the first
    of those two questions.

    It is a property of the learned greedy policy, not a measurement fault: no
    estimator repairs it and neither random tie-breaking nor 5% epsilon-greedy
    evaluation removes it. See docs/decision_log.md, "We tested three ways to fix
    the evaluation jam".
    """
    tail = metrics["eval_return_mean"].dropna().tail(n_tail)
    if tail.empty:
        return float("nan")
    return float((tail > 0).mean())


def conditional_return(metrics: pd.DataFrame, n_tail: int = N_LATE) -> float:
    """Mean return over the last n_tail evaluations THAT REACHED THE GOAL.

    The other half of the split: how well the policy does when it does not jam.
    Returns NaN, never 0, when none of the tail succeeded -- there is no
    conditional mean to report, and 0 would read as "it performed badly" when
    the truth is "it never completed an episode to score". Analysis must expect
    the NaN; it marks a run whose late-phase greedy policy never worked.
    """
    tail = metrics["eval_return_mean"].dropna().tail(n_tail)
    scored = tail[tail > 0]
    if scored.empty:
        return float("nan")
    return float(scored.mean())


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
            # final_return, split into its two parts. See success_rate.
            "success_rate": success_rate(r.metrics),
            "conditional_return": conditional_return(r.metrics),
        }
        for r in runs
    ])
