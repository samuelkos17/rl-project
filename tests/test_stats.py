import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rlx.analysis.aggregate import load_all
from rlx.analysis.stats import (
    aggregate_correlation, build_analysis_table, compare_coverage_predictors,
    iqm_by_strategy, rank_stability, within_instance_correlation,
)

#: Resolved from this file, not the working directory, so the suite passes no
#: matter where pytest was launched from.
GENERATOR = Path(__file__).resolve().parents[1] / "scripts" / "make_synthetic_results.py"


def _frame(rows):
    return pd.DataFrame(rows, columns=["env_id", "difficulty", "strategy", "seed",
                                       "final_return", "early_auc_raw"])


def test_perfect_positive_relationship_is_detected():
    rows = [["Empty-5", 5, f"s{i}", 0, float(i), float(i)] for i in range(10)]
    out = within_instance_correlation(_frame(rows), "early_auc_raw")
    assert np.isclose(out["rho"].iloc[0], 1.0)


def test_perfect_negative_relationship_is_detected():
    rows = [["Empty-5", 5, f"s{i}", 0, float(-i), float(i)] for i in range(10)]
    assert np.isclose(within_instance_correlation(_frame(rows), "early_auc_raw")["rho"].iloc[0], -1.0)


def test_one_row_per_environment_instance():
    rows = []
    for env in ("Empty-5", "DoorKey-5", "MultiRoom-N2"):
        rows += [[env, 5, f"s{i}", 0, float(i), float(i)] for i in range(10)]
    out = within_instance_correlation(_frame(rows), "early_auc_raw")
    assert len(out) == 3
    assert set(out["env_id"]) == {"Empty-5", "DoorKey-5", "MultiRoom-N2"}


def test_pooling_and_within_instance_disagree():
    """THE REGRESSION TEST. Do not delete.

    Within each maze, coverage and return move in OPPOSITE directions. But the
    hard maze has both lower coverage and lower return, so pooling everything
    produces a strong POSITIVE correlation that is pure difficulty artefact.
    """
    rows = []
    for level, env_id in enumerate(["Empty-5", "DoorKey-5", "DoorKey-8", "MultiRoom-N6"]):
        base_return = 0.9 - level * 0.2
        base_auc = 0.8 - level * 0.15
        for i in range(5):
            rows.append([env_id, level, f"s{i}", i,
                         base_return - i * 0.01, base_auc + i * 0.01])
    df = _frame(rows)

    pooled = df["final_return"].corr(df["early_auc_raw"], method="spearman")
    assert pooled > 0.8, "fixture is wrong: pooling should look strongly positive"

    out = within_instance_correlation(df, "early_auc_raw")
    assert (out["rho"] < 0).all(), "within-instance correlation must be negative here"


def test_instances_with_no_variance_are_reported_as_nan():
    """Every run scores 0 on the hardest mazes. That is a finding, not a crash."""
    rows = [["MultiRoom-N6", 6, f"s{i}", 0, 0.0, 0.1 + i * 0.01] for i in range(10)]
    out = within_instance_correlation(_frame(rows), "early_auc_raw")
    assert np.isnan(out["rho"].iloc[0])


def test_aggregate_reports_a_mean_and_a_bootstrap_interval():
    per_instance = pd.DataFrame({"env_id": [f"e{i}" for i in range(8)],
                                 "difficulty": range(8),
                                 "rho": [0.5, 0.6, 0.4, 0.55, 0.65, 0.45, 0.5, 0.6]})
    agg = aggregate_correlation(per_instance)
    assert 0.4 < agg["mean_rho"] < 0.7
    assert agg["ci_low"] < agg["mean_rho"] < agg["ci_high"]


def test_aggregate_ignores_nan_instances():
    per_instance = pd.DataFrame({"env_id": ["a", "b"], "difficulty": [1, 2],
                                 "rho": [0.5, np.nan]})
    assert np.isclose(aggregate_correlation(per_instance)["mean_rho"], 0.5)


def test_iqm_discards_the_extreme_quarters():
    df = pd.DataFrame({
        "env_id": ["Empty-5"] * 8,
        "strategy": ["a"] * 8,
        "final_return": [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 10.0],
    })
    result = iqm_by_strategy(df, "Empty-5")
    assert np.isclose(result["a"]["iqm"], 0.5), "outliers should not move the IQM"
    assert result["a"]["ci_low"] <= result["a"]["iqm"] <= result["a"]["ci_high"]


def test_rank_stability_is_one_when_the_order_never_changes():
    rows = []
    for env, diff in (("DoorKey-5", 5), ("DoorKey-8", 8)):
        for strat, ret in (("a", 0.9), ("b", 0.6), ("c", 0.3), ("d", 0.1)):
            rows.append([env, diff, strat, 0, ret, 0.5])
    out = rank_stability(_frame(rows))
    assert np.isclose(out["tau"].max(), 1.0)


def test_rank_stability_is_negative_when_the_order_reverses():
    rows = []
    for strat, ret in (("a", 0.9), ("b", 0.6), ("c", 0.3), ("d", 0.1)):
        rows.append(["DoorKey-5", 5, strat, 0, ret, 0.5])
    for strat, ret in (("a", 0.1), ("b", 0.3), ("c", 0.6), ("d", 0.9)):
        rows.append(["DoorKey-8", 8, strat, 0, ret, 0.5])
    out = rank_stability(_frame(rows))
    assert out[out["env_id"] == "DoorKey-8"]["tau"].iloc[0] < 0


def test_difficulty_trend_does_not_mix_families():
    """REGRESSION TEST. Do not delete.

    `difficulty` is grid size for Empty/DoorKey but room count for MultiRoom, so
    it is comparable only WITHIN a family. Here rho rises with difficulty inside
    every family, yet correlating against the raw column across families gives a
    NEGATIVE trend, because MultiRoom's small room counts sit at the 'easy' end
    while carrying the highest rho. The trend must be computed on within-family
    ranks.
    """
    rows = []
    for fam, diffs, rhos in (
        ("MultiRoom", [2, 4, 6], [0.70, 0.80, 0.90]),
        ("Empty", [5, 8, 16], [0.10, 0.20, 0.30]),
        ("DoorKey", [5, 8, 10], [0.20, 0.30, 0.40]),
    ):
        for d, r in zip(diffs, rhos):
            rows.append({"env_id": f"{fam}-{d}", "family": fam, "difficulty": d, "rho": r})
    per_instance = pd.DataFrame(rows)

    from scipy import stats as _st
    naive = _st.spearmanr(per_instance["difficulty"], per_instance["rho"]).statistic
    assert naive < 0, "fixture is wrong: the family-mixing bug should look negative"

    agg = aggregate_correlation(per_instance)
    assert agg["trend_with_difficulty"] > 0.9, (
        f"within every family rho rises with difficulty, so the trend must be "
        f"positive; got {agg['trend_with_difficulty']}"
    )


def test_probability_of_improvement_is_one_when_a_always_wins():
    from rlx.analysis.stats import probability_of_improvement

    rows = [["Empty-5", 5, "a", i, 0.9, 0.5] for i in range(5)]
    rows += [["Empty-5", 5, "b", i, 0.1, 0.5] for i in range(5)]
    assert probability_of_improvement(_frame(rows), "a", "b") == 1.0


def test_probability_of_improvement_scores_all_ties_as_one_half():
    """Both strategies score 0.0 on a maze neither solves. That is a draw, not a
    loss -- and it is the common case on our hardest instances."""
    from rlx.analysis.stats import probability_of_improvement

    rows = [["MultiRoom-N6", 6, "a", i, 0.0, 0.5] for i in range(5)]
    rows += [["MultiRoom-N6", 6, "b", i, 0.0, 0.5] for i in range(5)]
    assert probability_of_improvement(_frame(rows), "a", "b") == 0.5


def test_rliable_aggregate_returns_an_iqm_and_interval_per_strategy():
    from rlx.analysis.stats import rliable_aggregate

    rows = []
    for env in ("Empty-5", "DoorKey-5"):
        for strat, base in (("a", 0.8), ("b", 0.4)):
            for seed in range(5):
                rows.append([env, 5, strat, seed, base + seed * 0.01, 0.5])
    out = rliable_aggregate(_frame(rows))

    assert set(out) == {"a", "b"}
    assert out["a"]["iqm"] > out["b"]["iqm"]
    for r in out.values():
        assert r["ci_low"] <= r["iqm"] <= r["ci_high"]


def test_rliable_aggregate_is_reproducible():
    """rliable resamples internally. Without a pinned seed the report would print
    a different confidence interval every time it is regenerated.

    Compared to a tolerance, not bit-exactly: repeated runs differ in the last
    float bit (~1e-16) from summation order, while an unseeded bootstrap differs
    around the second decimal. 1e-9 separates the two by seven orders of
    magnitude, so this still fails loudly if the seed stops being passed.
    """
    import pytest

    from rlx.analysis.stats import rliable_aggregate

    rows = []
    for env in ("Empty-5", "DoorKey-5"):
        for strat, base in (("a", 0.8), ("b", 0.4)):
            for seed in range(5):
                rows.append([env, 5, strat, seed, base + seed * 0.03, 0.5])
    frame = _frame(rows)

    first = rliable_aggregate(frame, seed=0)
    second = rliable_aggregate(frame, seed=0)
    assert set(first) == set(second)
    for strategy, values in first.items():
        assert values == pytest.approx(second[strategy], rel=1e-9)


def test_rliable_aggregate_rejects_an_incomplete_matrix():
    """A strategy missing a run leaves a hole in the (seeds x instances) matrix.
    rliable would silently return NaN; we refuse instead, the same way
    early_auc does. A crashed run must not become a quiet blank in the report."""
    import pytest

    from rlx.analysis.stats import rliable_aggregate

    rows = []
    for env in ("Empty-5", "DoorKey-5"):
        for strat, base in (("a", 0.8), ("b", 0.4)):
            for seed in range(5):
                if strat == "a" and env == "DoorKey-5" and seed == 4:
                    continue          # the crashed run
                rows.append([env, 5, strat, seed, base + seed * 0.01, 0.5])

    with pytest.raises(ValueError, match="missing"):
        rliable_aggregate(_frame(rows))


def test_rank_stability_ranks_by_iqm_not_by_mean():
    """Spec 7.4 ranks strategies by IQM, which one unlucky seed cannot move.

    Strategy 'a' wins on both instances on IQM (0.90 vs 0.80). On the harder
    instance a single seed collapses to 0.0, which drags a's MEAN to 0.72 and
    flips the order -- so a mean-based ranking reports tau = -1.0 ("the winner
    changed") where nothing about the ranking actually changed.
    """
    rows = [["Empty-5", 5, "a", i, 0.90, 0.5] for i in range(5)]
    rows += [["Empty-5", 5, "b", i, 0.80, 0.5] for i in range(5)]
    rows += [["Empty-8", 8, "a", i, 0.0 if i == 0 else 0.90, 0.5] for i in range(5)]
    rows += [["Empty-8", 8, "b", i, 0.80, 0.5] for i in range(5)]

    tau = rank_stability(_frame(rows)).set_index("env_id")["tau"]
    assert tau["Empty-5"] == 1.0
    assert tau["Empty-8"] == 1.0


def test_h1_is_not_confirmed_when_the_effect_shrinks_with_difficulty():
    """Spec section 1 states TWO conditions for H1: a positive correlation whose
    CI excludes zero, AND a correlation that grows on harder mazes. Here the
    first holds and the second is exactly reversed, so H1 is not confirmed."""
    rows = [{"env_id": f"Empty-{d}", "family": "Empty", "difficulty": d, "rho": r}
            for d, r in ((5, 0.90), (8, 0.70), (16, 0.50))]
    agg = aggregate_correlation(pd.DataFrame(rows))

    assert agg["ci_above_zero"] is True
    assert agg["trend_with_difficulty"] < 0
    assert agg["confirms_h1"] is False


def test_h1_is_confirmed_when_both_conditions_hold():
    rows = [{"env_id": f"Empty-{d}", "family": "Empty", "difficulty": d, "rho": r}
            for d, r in ((5, 0.50), (8, 0.70), (16, 0.90))]
    agg = aggregate_correlation(pd.DataFrame(rows))
    assert agg["ci_above_zero"] is True
    assert agg["confirms_h1"] is True


def test_a_wholly_negative_ci_is_not_reported_as_lying_above_zero():
    """REGRESSION TEST. Do not delete.

    The key used to be called `ci_excludes_zero` while testing `ci_low > 0`. On
    this data the CI is about [-0.80, -0.40] -- it excludes zero emphatically --
    and results.md printed "CI excludes zero: False", a false statement about the
    project's headline number.

    Testing "excludes zero" honestly and feeding that to `confirms_h1` would be
    worse: H1 would be confirmed by a strong NEGATIVE correlation. So the key
    says what it tests, and H1 still needs the correlation to be positive.
    """
    rows = [{"env_id": f"Empty-{d}", "family": "Empty", "difficulty": d, "rho": r}
            for d, r in ((5, -0.50), (8, -0.60), (16, -0.70))]
    agg = aggregate_correlation(pd.DataFrame(rows))

    assert agg["ci_high"] < 0, "fixture is wrong: the CI must sit below zero"
    assert agg["ci_above_zero"] is False
    assert agg["confirms_h1"] is False


def test_an_unmeasurable_trend_does_not_confirm_h1():
    """Two instances is below the three a Spearman needs, so the trend is NaN.
    'We could not measure it' must not read as 'confirmed'."""
    rows = [{"env_id": "Empty-5", "family": "Empty", "difficulty": 5, "rho": 0.8},
            {"env_id": "Empty-8", "family": "Empty", "difficulty": 8, "rho": 0.9}]
    agg = aggregate_correlation(pd.DataFrame(rows))
    assert np.isnan(agg["trend_with_difficulty"])
    assert agg["confirms_h1"] is False


def test_aggregate_returns_the_same_keys_when_nothing_is_measurable():
    """report.py reads agg['confirms_h1'] unconditionally. The degenerate branch
    used to omit the key entirely, so a fully tied sweep raised KeyError while
    writing the report instead of reporting 'not confirmed'."""
    all_nan = pd.DataFrame({"env_id": ["a", "b"], "difficulty": [1, 2],
                            "rho": [np.nan, np.nan]})
    normal = pd.DataFrame({"env_id": ["a", "b", "c"], "family": ["E"] * 3,
                           "difficulty": [1, 2, 3], "rho": [0.5, 0.6, 0.7]})
    assert set(aggregate_correlation(all_nan)) == set(aggregate_correlation(normal))
    assert aggregate_correlation(all_nan)["confirms_h1"] is False


def test_each_instance_gets_a_bootstrap_interval_on_its_correlation():
    """Spec 7.3 step 2: bootstrap over that instance's runs for a CI on its own
    rho, not only on the mean across instances."""
    rng = np.random.default_rng(0)
    auc = np.linspace(0.1, 0.9, 20)
    rows = [["DoorKey-8", 8, f"s{i}", i, float(a + rng.normal(0, 0.25)), float(a)]
            for i, a in enumerate(auc)]
    out = within_instance_correlation(_frame(rows), "early_auc_raw").iloc[0]

    assert "rho_ci_low" in out and "rho_ci_high" in out
    assert out["rho_ci_low"] < out["rho"] < out["rho_ci_high"]
    assert -1.0 <= out["rho_ci_low"] and out["rho_ci_high"] <= 1.0


def test_the_per_instance_interval_is_reproducible():
    rng = np.random.default_rng(1)
    rows = [["DoorKey-8", 8, f"s{i}", i, float(a + rng.normal(0, 0.3)), float(a)]
            for i, a in enumerate(np.linspace(0.1, 0.9, 20))]
    frame = _frame(rows)
    first = within_instance_correlation(frame, "early_auc_raw", seed=0)
    second = within_instance_correlation(frame, "early_auc_raw", seed=0)
    assert first["rho_ci_low"].equals(second["rho_ci_low"])
    assert first["rho_ci_high"].equals(second["rho_ci_high"])


def test_an_instance_with_no_variance_has_no_interval_either():
    """rho is NaN there, so an interval would be an invention."""
    rows = [["MultiRoom-N6", 6, f"s{i}", i, 0.0, 0.1 + i * 0.01] for i in range(10)]
    out = within_instance_correlation(_frame(rows), "early_auc_raw").iloc[0]
    assert np.isnan(out["rho"])
    assert np.isnan(out["rho_ci_low"]) and np.isnan(out["rho_ci_high"])


def _two_predictor_frame(task_predicts: bool, raw_predicts: bool) -> pd.DataFrame:
    """4 instances x 20 runs. Each predictor either tracks final_return or is
    independent of it, so H2 can be constructed either way."""
    rng = np.random.default_rng(0)
    rows = []
    for env_id, difficulty in (("DoorKey-5", 5), ("DoorKey-6", 6),
                               ("DoorKey-7", 7), ("DoorKey-8", 8)):
        signal = np.linspace(0.1, 0.9, 20)
        ret = signal + rng.normal(0, 0.05, 20)
        for i in range(20):
            rows.append({
                "env_id": env_id, "family": "DoorKey", "difficulty": difficulty,
                "strategy": f"s{i % 4}", "seed": i // 4,
                "final_return": float(ret[i]),
                "early_auc_task": float(signal[i] if task_predicts else rng.random()),
                "early_auc_raw": float(signal[i] if raw_predicts else rng.random()),
            })
    return pd.DataFrame(rows)


def test_h2_is_confirmed_when_task_relevant_predicts_and_raw_does_not():
    """Spec section 1: H2 needs a LARGER correlation AND non-overlapping CIs."""
    from rlx.analysis.stats import compare_coverage_predictors

    out = compare_coverage_predictors(_two_predictor_frame(task_predicts=True,
                                                           raw_predicts=False))
    assert out["task"]["mean_rho"] > out["raw"]["mean_rho"]
    assert out["cis_overlap"] is False
    assert out["confirms_h2"] is True
    assert out["task_minus_raw"] > 0


def test_h2_is_not_confirmed_when_both_predict_equally_well():
    """The spec calls this the interesting alternative, not a failure: breadth of
    exploration matters, directedness does not."""
    from rlx.analysis.stats import compare_coverage_predictors

    out = compare_coverage_predictors(_two_predictor_frame(task_predicts=True,
                                                           raw_predicts=True))
    assert out["cis_overlap"] is True
    assert out["confirms_h2"] is False


def _profile_frame() -> pd.DataFrame:
    rows = []
    for env in ("Empty-5", "DoorKey-5"):
        for strat, base in (("a", 0.8), ("b", 0.4)):
            for seed in range(5):
                rows.append([env, 5, strat, seed, base + seed * 0.02, 0.5])
    return _frame(rows)


def test_performance_profile_covers_every_strategy_and_threshold():
    """Spec 7.2: the fraction of runs exceeding each return threshold, which
    shows the whole distribution instead of one summary number."""
    from rlx.analysis.stats import performance_profile

    out = performance_profile(_profile_frame())
    assert set(out["profiles"]) == {"a", "b"}
    for strategy in ("a", "b"):
        assert len(out["profiles"][strategy]) == len(out["taus"])
        assert len(out["ci_low"][strategy]) == len(out["taus"])
        assert len(out["ci_high"][strategy]) == len(out["taus"])


def test_performance_profile_never_rises_with_the_threshold():
    """Fewer runs can clear a higher bar, never more."""
    from rlx.analysis.stats import performance_profile

    out = performance_profile(_profile_frame())
    for strategy, curve in out["profiles"].items():
        assert (np.diff(curve) <= 1e-9).all(), strategy
        assert ((curve >= 0) & (curve <= 1)).all(), strategy


def test_a_dominating_strategy_has_the_higher_profile_everywhere():
    from rlx.analysis.stats import performance_profile

    out = performance_profile(_profile_frame())
    assert (out["profiles"]["a"] >= out["profiles"]["b"] - 1e-9).all()


def test_performance_profile_is_reproducible():
    """Same global-RNG problem as rliable_aggregate: without a pinned seed the
    confidence band moves every time the report is regenerated."""
    from rlx.analysis.stats import performance_profile

    frame = _profile_frame()
    first = performance_profile(frame, seed=0)
    second = performance_profile(frame, seed=0)
    for strategy in first["profiles"]:
        assert np.array_equal(first["ci_low"][strategy], second["ci_low"][strategy])
        assert np.array_equal(first["ci_high"][strategy], second["ci_high"][strategy])


def _fake_run(seed: int, steps, total_steps: int = 100_000):
    from rlx.analysis.aggregate import RunResult

    counts = np.zeros((len(steps), 5, 5, 4), dtype=np.int32)
    counts[:, 1, 1, 0] = 1
    return RunResult(
        env_id="Empty-5", strategy="epsilon_greedy", seed=seed,
        metrics=pd.DataFrame({"step": [0], "eval_return_mean": [0.5]}),
        steps=np.asarray(steps), counts=counts,
        config={"total_steps": total_steps},
    )


def test_build_analysis_table_names_every_unusable_run_at_once():
    """One bad snapshot grid used to abort the whole 260-run table at the first
    run it hit, so you learned about the next one only after re-running. Report
    them all, and say which."""
    good = _fake_run(0, np.arange(10_000, 100_001, 10_000))
    bad_a = _fake_run(1, np.array([50_000, 100_000]))
    bad_b = _fake_run(2, np.array([60_000, 100_000]))

    with pytest.raises(ValueError) as excinfo:
        build_analysis_table([good, bad_a, bad_b])

    message = str(excinfo.value)
    assert "Empty-5/epsilon_greedy/seed1" in message
    assert "Empty-5/epsilon_greedy/seed2" in message
    assert "seed0" not in message


def test_build_analysis_table_is_unchanged_when_every_run_is_usable():
    df = build_analysis_table([_fake_run(0, np.arange(10_000, 100_001, 10_000)),
                               _fake_run(1, np.arange(10_000, 100_001, 10_000))])
    assert len(df) == 2
    assert {"early_auc_raw", "early_auc_task"} <= set(df.columns)


# --- the negative control ----------------------------------------------------
# Every other test in this file checks that the analysis can say YES. These two
# check that it can say NO. Without them a refactor that made every correlation
# come out positive would pass the whole suite.


def _analysis_table(out_dir: Path, *args: str) -> pd.DataFrame:
    subprocess.run([sys.executable, str(GENERATOR), "--out", str(out_dir), *args],
                   check=True)
    return build_analysis_table(load_all(out_dir))


@pytest.fixture(scope="module")
def no_effect(tmp_path_factory):
    """`--no-effect`: final return is generated independently of coverage."""
    return _analysis_table(tmp_path_factory.mktemp("noeffect"), "--no-effect")


@pytest.fixture(scope="module")
def with_effect(tmp_path_factory):
    """The default fixture: more early coverage is MADE to score better, and more
    so on harder instances."""
    return _analysis_table(tmp_path_factory.mktemp("effect"))


def test_h1_is_not_confirmed_on_the_negative_control(no_effect):
    """REGRESSION TEST. Do not delete.

    Return is independent of coverage here by construction, so the honest answer
    is "no relationship". Measured 2026-08-19: mean rho -0.004, CI
    [-0.104, +0.096], per-instance rho spanning -0.282 to +0.281.
    """
    for column in ("early_auc_raw", "early_auc_task"):
        agg = aggregate_correlation(within_instance_correlation(no_effect, column))
        assert agg["n_instances"] == 13, column
        assert abs(agg["mean_rho"]) < 0.15, (column, agg["mean_rho"])
        assert agg["ci_low"] < 0 < agg["ci_high"], (column, agg)
        assert agg["ci_above_zero"] is False, column
        assert agg["confirms_h1"] is False, column
    assert compare_coverage_predictors(no_effect)["confirms_h2"] is False


def test_h1_is_confirmed_on_the_dataset_that_has_the_effect(with_effect):
    """The other half of the control: the same code path must say YES where the
    answer is built in. Measured 2026-08-19: mean rho +0.696, CI
    [+0.546, +0.833], trend +0.900.

    Both halves matter. A pipeline that always says yes and one that always says
    no are equally useless, and only running both fixtures tells them apart.
    """
    agg = aggregate_correlation(within_instance_correlation(with_effect, "early_auc_raw"))
    assert agg["mean_rho"] > 0.5
    assert agg["ci_above_zero"] is True
    assert agg["trend_with_difficulty"] > 0.5
    assert agg["confirms_h1"] is True
