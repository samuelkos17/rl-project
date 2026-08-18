import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from rlx.config import RunConfig
from rlx.train import run_training


def _cfg(tmp_path, **kw):
    kw.setdefault("env_id", "Empty-5")
    kw.setdefault("strategy", "epsilon_greedy")
    return RunConfig(
        seed=0, total_steps=2_000, learning_starts=100, eval_every=500,
        snapshot_every=500, buffer_size=1_000, results_root=str(tmp_path), **kw,
    )


def test_a_short_run_writes_a_schema_valid_result_directory(tmp_path):
    run_dir = run_training(_cfg(tmp_path))

    for name in ("config.json", "metrics.csv", "visitation.npz", "meta.json"):
        assert (run_dir / name).exists(), name

    df = pd.read_csv(run_dir / "metrics.csv")
    for col in ("step", "eval_return_mean", "distinct_states"):
        assert col in df.columns
    assert len(df) > 0

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["completed"] is True


def test_no_partial_directory_is_left_behind(tmp_path):
    run_training(_cfg(tmp_path))
    assert not list(tmp_path.rglob("*.partial"))


def test_the_same_seed_reproduces_the_same_metrics(tmp_path):
    a = pd.read_csv(run_training(_cfg(tmp_path / "a")) / "metrics.csv")
    b = pd.read_csv(run_training(_cfg(tmp_path / "b")) / "metrics.csv")
    pd.testing.assert_frame_equal(a, b)


#: Max's NoisyNets module (workstream B task 4) may not be merged yet. Rather
#: than a hand-written xfail someone has to remember to delete, detect it: the
#: moment his module lands, this becomes a real test with no edit required.
_NOISY_READY = importlib.util.find_spec("rlx.exploration.noisy") is not None
_NOISY_PARAM = pytest.param(
    "noisy",
    marks=[] if _NOISY_READY else [pytest.mark.xfail(
        reason="rlx.exploration.noisy not merged yet (Max, workstream B task 4)",
        raises=ModuleNotFoundError, strict=True)],
)


@pytest.mark.parametrize(
    "strategy", ["epsilon_greedy", "boltzmann", "count_based", _NOISY_PARAM])
def test_every_strategy_runs_end_to_end(tmp_path, strategy):
    run_dir = run_training(_cfg(tmp_path / strategy, strategy=strategy))
    assert (run_dir / "metrics.csv").exists()


def test_metrics_has_one_row_per_evaluation(tmp_path):
    """Daniel's final_return depends on this: one row per eval, never a row
    with an empty eval_return_mean."""
    cfg = _cfg(tmp_path)
    df = pd.read_csv(run_training(cfg) / "metrics.csv")
    assert len(df) == cfg.total_steps // cfg.eval_every
    assert df["eval_return_mean"].notna().all()


def test_visitation_snapshots_are_cumulative_and_match_the_grid(tmp_path):
    cfg = _cfg(tmp_path)
    data = np.load(run_training(cfg) / "visitation.npz")
    counts, steps = data["counts"], data["steps"]
    assert counts.shape[1:] == (5, 5, 4)                 # Empty-5 is a 5x5 grid
    totals = counts.sum(axis=(1, 2, 3))
    assert (np.diff(totals) >= 0).all(), "visit counts must never decrease"
    assert totals[-1] > 0
    assert len(steps) == len(counts)


def test_intrinsic_bonus_never_reaches_the_reported_return(tmp_path):
    """count_based adds a bonus inside the buffer only. If it leaked into the
    score, evaluation returns could exceed MiniGrid's 1.0 ceiling."""
    df = pd.read_csv(run_training(_cfg(tmp_path, strategy="count_based")) / "metrics.csv")
    assert (df["eval_return_mean"] <= 1.0).all()
    assert (df["eval_return_mean"] >= 0.0).all()


def test_strategy_stats_are_logged(tmp_path):
    eps = pd.read_csv(run_training(_cfg(tmp_path / "e")) / "metrics.csv")
    assert "epsilon" in eps.columns

    cb = pd.read_csv(run_training(_cfg(tmp_path / "c", strategy="count_based")) / "metrics.csv")
    assert "mean_bonus" in cb.columns


def test_config_json_records_the_resolved_config(tmp_path):
    cfg = _cfg(tmp_path)
    saved = json.loads((run_training(cfg) / "config.json").read_text())
    assert saved["total_steps"] == cfg.total_steps
    assert saved["strategy"] == cfg.strategy
    assert saved["seed"] == cfg.seed
