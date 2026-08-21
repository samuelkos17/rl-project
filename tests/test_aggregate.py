import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rlx.config import RunConfig
from rlx.analysis.aggregate import final_return, load_all, load_run, to_dataframe
from rlx.envs import ENV_IDS, difficulty_index
from rlx.exploration import STRATEGIES

#: The fixture covers the real matrix: every instance x every strategy x 5 seeds.
#: Derived, not hard-coded, so extending FIXTURE_ENV_IDS cannot leave a stale
#: number behind in three separate tests.
EXPECTED_RUNS = len(ENV_IDS) * len(STRATEGIES) * 5

#: Resolved from this file, not from the working directory, so the suite passes
#: no matter where pytest was launched from.
GENERATOR = Path(__file__).resolve().parents[1] / "scripts" / "make_synthetic_results.py"


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """The default fixture dataset, generated once for the whole module."""
    out = tmp_path_factory.mktemp("synth")
    subprocess.run([sys.executable, str(GENERATOR), "--out", str(out)], check=True)
    return out


@pytest.fixture(scope="module")
def runs(generated):
    return load_all(generated)


def test_all_synthetic_runs_load(runs):
    assert len(runs) == EXPECTED_RUNS


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


def test_final_return_uses_evaluation_points_not_rows():
    """metrics.csv has one row per logged step, and training may log more often
    than it evaluates, leaving eval_return_mean empty on most rows."""
    step = np.arange(0, 20_001, 1_000)
    metrics = pd.DataFrame({
        "step": step,
        "eval_return_mean": [0.4 if s % 5_000 == 0 else np.nan for s in step],
    })
    assert metrics["eval_return_mean"].notna().sum() == 5
    # The raw tail holds four NaNs and one value: averaging it would use a single
    # evaluation point while claiming to use five.
    assert metrics["eval_return_mean"].tail(5).notna().sum() == 1
    assert final_return(metrics, n_tail=5) == pytest.approx(0.4)


def test_incomplete_runs_are_skipped(tmp_path):
    (tmp_path / "Empty-5" / "epsilon_greedy" / "seed0").mkdir(parents=True)
    assert load_all(tmp_path) == []


def test_partial_directories_are_skipped(generated, tmp_path):
    """A crashed run leaves seed<k>.partial holding every REQUIRED_FILE, because
    those are written before the rename. Only the rename means 'finished'."""
    src = generated / "Empty-5" / "epsilon_greedy" / "seed0"
    dst = tmp_path / "Empty-5" / "epsilon_greedy" / "seed0"
    shutil.copytree(src, dst)
    shutil.copytree(dst, dst.with_name("seed1.partial"))

    assert [r.seed for r in load_all(tmp_path)] == [0]


def test_metrics_csv_without_a_header_does_not_crash(tmp_path):
    """RunLogger.finalize on a run that logged nothing writes a headerless file."""
    run_dir = tmp_path / "Empty-5" / "noisy" / "seed0"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.csv").write_text("\n")
    (run_dir / "config.json").write_text("{}")
    np.savez_compressed(run_dir / "visitation.npz",
                        steps=np.zeros(0, dtype=np.int64),
                        counts=np.zeros((0, 5, 5, 4), dtype=np.int32))

    run = load_run(run_dir)
    assert run.metrics.empty
    assert np.isnan(final_return(run.metrics))


def test_grids_match_the_real_minigrid_dimensions(runs):
    """Task 3 intersects counts with a (W, H) reachability mask from grid_info,
    so a fixture at the wrong shape could not be developed against."""
    # Stated independently of grid_info, which is what the fixture itself uses --
    # deriving the expectation from grid_info would make this test tautological.
    # Empty-N and DoorKey-N are N x N; every MultiRoom grid is 25 x 25 regardless
    # of room count (verified 2026-08-17, see the note in src/rlx/envs.py).
    for r in runs:
        size = 25 if r.env_id.startswith("MultiRoom") else difficulty_index(r.env_id)
        assert r.counts.shape[1:] == (size, size, 4), r.env_id


def test_counts_are_cumulative_like_the_real_logger(runs):
    """RunLogger only ever increments, so no counter can fall between snapshots."""
    for r in runs:
        assert (np.diff(r.counts.astype(np.int64), axis=0) >= 0).all(), r.env_id


def test_config_json_is_the_real_run_config(runs):
    expected = set(RunConfig(env_id="Empty-5", strategy="noisy", seed=0).to_dict())
    for r in runs:
        assert set(r.config) == expected, r.env_id
        assert (r.config["env_id"], r.config["strategy"], r.config["seed"]) == \
               (r.env_id, r.strategy, r.seed)


def test_no_effect_control_is_loadable_and_not_degenerate(tmp_path):
    """The negative control must vary, or a 'no correlation' result would be
    meaningless -- a constant column correlates with nothing by construction."""
    subprocess.run([sys.executable, str(GENERATOR), "--out", str(tmp_path), "--no-effect"],
                   check=True)
    df = to_dataframe(load_all(tmp_path))
    assert len(df) == EXPECTED_RUNS
    assert (df.groupby("env_id")["final_return"].std() > 0.01).all()


def test_the_two_fixtures_are_distinguishable_on_disk(generated, tmp_path):
    """Otherwise a stray --no-effect run silently replaces the real fixture and
    the analysis reports 'no effect' as though it were a finding."""
    subprocess.run([sys.executable, str(GENERATOR), "--out", str(tmp_path), "--no-effect"],
                   check=True)
    null_meta = json.loads((tmp_path / "Empty-5" / "noisy" / "seed0" / "meta.json").read_text())
    real_meta = json.loads((generated / "Empty-5" / "noisy" / "seed0" / "meta.json").read_text())
    assert null_meta["synthetic_effect"] is False
    assert real_meta["synthetic_effect"] is True


def _metrics(returns):
    """A metrics frame with one eval row per value, plus unevaluated rows."""
    rows = []
    for i, v in enumerate(returns):
        rows.append({"step": i * 1000, "eval_return_mean": v})
        rows.append({"step": i * 1000 + 500, "eval_return_mean": np.nan})
    return pd.DataFrame(rows)


def test_success_rate_counts_evaluations_that_reached_the_goal():
    from rlx.analysis.aggregate import success_rate
    assert success_rate(_metrics([0.9] * 20), n_tail=20) == 1.0
    assert success_rate(_metrics([0.0] * 20), n_tail=20) == 0.0
    assert success_rate(_metrics([0.0, 0.0, 0.9, 0.9]), n_tail=4) == 0.5
    # only the tail counts: an early success does not rescue a jammed late phase
    assert success_rate(_metrics([0.9] * 10 + [0.0] * 20), n_tail=20) == 0.0


def test_conditional_return_averages_only_the_successes():
    from rlx.analysis.aggregate import conditional_return
    # the zeros must not drag it down -- that is the whole point of the split
    assert np.isclose(conditional_return(_metrics([0.0, 0.0, 0.8, 0.9]), n_tail=4), 0.85)
    assert np.isclose(conditional_return(_metrics([0.8, 0.9]), n_tail=2), 0.85)


def test_conditional_return_is_nan_not_zero_when_nothing_succeeded():
    """0 would read as "it performed badly"; the truth is "it never scored".

    A run whose late-phase greedy policy always times out has no conditional
    mean. Returning 0 would put it in the same bucket as a run that reached the
    goal by the worst possible route.
    """
    from rlx.analysis.aggregate import conditional_return
    assert np.isnan(conditional_return(_metrics([0.0] * 20), n_tail=20))


def test_the_split_separates_a_jamming_run_from_a_bad_one():
    """The two runs below have the SAME final_return and mean very different
    things. Separating them is why these columns exist."""
    from rlx.analysis.aggregate import conditional_return, final_return, success_rate
    jams = _metrics([0.0, 0.95, 0.0, 0.0, 0.95])      # optimal when it works
    steady = _metrics([0.38, 0.38, 0.38, 0.38, 0.38])  # always mediocre
    assert np.isclose(final_return(jams), final_return(steady), atol=0.01)
    assert success_rate(jams, n_tail=5) == 0.4
    assert success_rate(steady, n_tail=5) == 1.0
    assert conditional_return(jams, n_tail=5) > conditional_return(steady, n_tail=5)
