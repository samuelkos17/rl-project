import numpy as np
import pandas as pd

from rlx.analysis.stats import (
    aggregate_correlation, iqm_by_strategy, rank_stability, within_instance_correlation,
)


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
