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


def test_count_bonus_is_keyed_on_the_successor_observation(tmp_path, monkeypatch):
    """The bonus paid into the buffer must be for the state ARRIVED at.

    Keying on the state being left pays every action the same bonus, so nothing
    distinguishes the action that leads somewhere new until the value has
    bootstrapped one level further. Measured 2026-08-20 to cost 8.4% of coverage
    area over 9 paired runs; see docs/decision_log.md, "Where the count-based
    bonus is paid".

    This asserts the wiring in train.py, not the arithmetic in CountBased: it
    records which key each intrinsic_bonus call received and checks it is the
    NEXT observation, not the current one.
    """
    from rlx.exploration.count_based import CountBased

    seen = []
    original = CountBased.intrinsic_bonus

    def recording_bonus(self, count_key):
        seen.append(count_key)
        return original(self, count_key)

    monkeypatch.setattr(CountBased, "intrinsic_bonus", recording_bonus)

    observed = []
    original_observe = CountBased.observe

    def recording_observe(self, count_key):
        observed.append(count_key)
        return original_observe(self, count_key)

    monkeypatch.setattr(CountBased, "observe", recording_observe)

    run_training(_cfg(tmp_path, strategy="count_based"))

    assert len(seen) == len(observed) > 0
    # observe() counts the CURRENT observation; the bonus is paid on the NEXT
    # one. So at each step the two must differ in role: the key the bonus was
    # paid on at step t is the key observe() sees at step t+1 -- except where an
    # episode ended in between and the environment was reset.
    matches = sum(1 for t in range(len(seen) - 1) if seen[t] == observed[t + 1])
    assert matches > 0.5 * (len(seen) - 1), (
        "the bonus key does not track the next step's observation, so it is "
        "being keyed on the current observation instead of the successor"
    )

    # It must also not be trivially the same key every step. Most MiniGrid
    # actions are no-ops for the observation -- walking into a wall, or pickup /
    # drop / done with nothing to act on -- so obs == next_obs on the majority of
    # steps and the two keys legitimately agree there. The discriminating fact is
    # that they do not agree ALWAYS: under the old current-observation keying
    # this count would be exactly len(seen).
    same = sum(1 for t in range(len(seen)) if seen[t] == observed[t])
    assert same < len(seen), "bonus key is always the current observation"


def test_eval_episode_length_is_logged_and_distinguishes_timeout_from_failure(tmp_path):
    """eval_episode_len separates "never learned" from "greedy policy loops".

    eval_return_mean == 0.0 is ambiguous on its own: MiniGrid pays 0 exactly for
    a timeout, and a run that never reaches the goal and a run whose greedy
    policy cycles both time out. The length tells them apart -- a looping policy
    burns max_steps every time. Added 2026-08-20 after 6 of 87 real runs reached
    a training return above 0.7 while evaluating to exactly 0.0.
    """
    run_dir = run_training(_cfg(tmp_path))
    df = pd.read_csv(run_dir / "metrics.csv")

    assert "eval_episode_len" in df.columns
    lengths = df["eval_episode_len"].dropna()
    assert len(lengths) > 0
    assert (lengths > 0).all(), "an evaluation episode cannot have zero steps"

    # the column must be usable for the diagnosis it exists for: every zero-return
    # evaluation must have run to the environment's step limit
    from rlx.envs import make_env
    max_steps = make_env("Empty-5", layout_seed=0).unwrapped.max_steps
    assert (lengths <= max_steps).all(), "an episode ran past max_steps"

    zero_return = df[df["eval_return_mean"] == 0.0]["eval_episode_len"].dropna()
    if len(zero_return):
        assert (zero_return == max_steps).all(), (
            "a zero-return evaluation that did not reach max_steps means the "
            "episode terminated without reward, which MiniGrid does not do"
        )
