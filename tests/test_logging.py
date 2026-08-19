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


def test_finalize_refuses_to_overwrite_a_finished_run(cfg):
    """CLAUDE.md section 5: a directory that exists is a directory that finished,
    and the sweep runner skips those. finalize used to rmtree the old one, so a
    direct `python -m rlx.train` on an already-completed run destroyed a result
    that may have taken ten minutes to produce -- without a word."""
    RunLogger(cfg, width=5, height=5).finalize({"completed": True})
    assert cfg.run_dir.exists()

    with pytest.raises(FileExistsError, match="already exists"):
        RunLogger(cfg, width=5, height=5).finalize({"completed": True})


def test_a_refused_finalize_leaves_the_finished_run_untouched(cfg):
    first = RunLogger(cfg, width=5, height=5)
    first.record_visit(1, 1, 0)
    first.log_step(0, eval_return_mean=0.75)
    first.finalize({"completed": True})
    before = (cfg.run_dir / "metrics.csv").read_text()

    with pytest.raises(FileExistsError):
        RunLogger(cfg, width=5, height=5).finalize({"completed": True})

    assert (cfg.run_dir / "metrics.csv").read_text() == before
    assert not cfg.run_dir.with_name(cfg.run_dir.name + ".partial").exists()
