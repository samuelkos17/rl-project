"""Statistical analysis.

CRITICAL -- read CLAUDE.md section 9 before touching this file.

Coverage and return both fall as difficulty rises, so a correlation pooled across
environment instances is a difficulty artefact and means nothing. Every
correlation here is computed WITHIN an instance, where difficulty is constant,
and only then aggregated.
"""

import numpy as np
import pandas as pd
from scipy import stats

from rlx.analysis.aggregate import RunResult, to_dataframe
from rlx.analysis.coverage import early_auc, raw_coverage, task_relevant_coverage
from rlx.envs import grid_info

N_BOOTSTRAP = 10_000


def build_analysis_table(runs: list[RunResult]) -> pd.DataFrame:
    """One row per run: identity, final return, and both early-coverage AUCs."""
    df = to_dataframe(runs)
    raw_auc, task_auc = [], []
    for r in runs:
        # layout_seed == seed: the maze a run saw is determined by its seed.
        info = grid_info(r.env_id, layout_seed=r.seed)
        total = r.config["total_steps"]
        raw_auc.append(early_auc(r.steps, raw_coverage(r.counts, info), total))
        task_auc.append(early_auc(r.steps, task_relevant_coverage(r.counts, info), total))
    df["early_auc_raw"] = raw_auc
    df["early_auc_task"] = task_auc
    return df


def within_instance_correlation(df: pd.DataFrame, coverage_col: str) -> pd.DataFrame:
    """Spearman correlation of coverage vs final return, computed per instance.

    NEVER pool instances. See the module docstring.
    """
    rows = []
    for env_id, group in df.groupby("env_id", sort=True):
        if group["final_return"].nunique() < 2 or group[coverage_col].nunique() < 2:
            rho, p = np.nan, np.nan       # no variance -- a finding, not an error
        else:
            rho, p = stats.spearmanr(group[coverage_col], group["final_return"])
        rows.append({
            "env_id": env_id,
            "difficulty": group["difficulty"].iloc[0],
            "family": group["family"].iloc[0] if "family" in group else env_id.split("-")[0],
            "n_runs": len(group),
            "rho": rho,
            "p_value": p,
        })
    return pd.DataFrame(rows)


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def aggregate_correlation(per_instance: pd.DataFrame, seed: int = 0) -> dict:
    """Combine per-instance correlations, with a bootstrap CI on the mean.

    Also reports how the correlation trends with difficulty: H1 predicts the
    relationship gets STRONGER on harder mazes.

    The trend is measured WITHIN each family and then averaged, never on the raw
    `difficulty` column pooled across families. `difficulty` is grid size for
    Empty/DoorKey but room count for MultiRoom, so across families it compares
    two different quantities: MultiRoom-N2 scores 2 and lands at the 'easy' end
    while actually being one of the hardest mazes we have (19 reachable cells).
    Correlating the pooled column inverted the sign of this trend. Ranking within
    family fixes the sign but still dilutes it, because the rho levels differ
    between families. See `test_difficulty_trend_does_not_mix_families`.

    A family needs at least 3 instances and 2 distinct difficulties to contribute;
    a 2-point Spearman is +-1 by construction and says nothing. If no family
    qualifies the trend is NaN -- that is "we cannot measure this", not "no trend".
    """
    valid = per_instance.dropna(subset=["rho"])
    if len(valid) == 0:
        return {"mean_rho": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "n_instances": 0, "trend_with_difficulty": np.nan,
                "trend_per_family": {}}

    rho = valid["rho"].to_numpy()
    ci_low, ci_high = _bootstrap_mean_ci(rho, np.random.default_rng(seed))

    family = (valid["family"] if "family" in valid
              else valid["env_id"].str.split("-").str[0])
    per_family = {}
    for fam, g in valid.groupby(family, sort=True):
        if len(g) >= 3 and g["difficulty"].nunique() >= 2:
            per_family[fam] = float(stats.spearmanr(g["difficulty"], g["rho"]).statistic)
    trend = float(np.mean(list(per_family.values()))) if per_family else np.nan
    return {
        "mean_rho": float(rho.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_instances": len(valid),
        "trend_with_difficulty": trend,
        "trend_per_family": per_family,
        "confirms_h1": bool(ci_low > 0),
    }


def _iqm(values: np.ndarray) -> float:
    """Interquartile mean: drop the top and bottom 25%, average the middle."""
    if len(values) == 0:
        return np.nan
    lo, hi = np.percentile(values, [25, 75])
    middle = values[(values >= lo) & (values <= hi)]
    return float(middle.mean()) if len(middle) else float(values.mean())


def iqm_by_strategy(df: pd.DataFrame, env_id: str, seed: int = 0) -> dict:
    """IQM final return per strategy on one instance, with bootstrap CIs."""
    rng = np.random.default_rng(seed)
    out = {}
    for strategy, group in df[df["env_id"] == env_id].groupby("strategy", sort=True):
        values = group["final_return"].to_numpy()
        draws = [_iqm(rng.choice(values, len(values), replace=True))
                 for _ in range(1_000)]
        out[strategy] = {
            "iqm": _iqm(values),
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5)),
            "n": len(values),
        }
    return out


def rank_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Kendall's tau between each instance's strategy ranking and the ranking on
    the easiest instance of the same family.

    tau = 1.0 means identical order, -1.0 exactly reversed, 0 unrelated.
    """
    df = df.copy()
    if "family" not in df:
        df["family"] = df["env_id"].str.split("-").str[0]

    rows = []
    for family, fam_group in df.groupby("family", sort=True):
        easiest = fam_group.loc[fam_group["difficulty"].idxmin(), "env_id"]
        baseline = (fam_group[fam_group["env_id"] == easiest]
                    .groupby("strategy")["final_return"].mean().sort_index())
        for env_id, group in fam_group.groupby("env_id", sort=True):
            here = group.groupby("strategy")["final_return"].mean().sort_index()
            shared = baseline.index.intersection(here.index)
            tau = (stats.kendalltau(baseline[shared], here[shared]).statistic
                   if len(shared) >= 2 else np.nan)
            rows.append({
                "family": family,
                "env_id": env_id,
                "difficulty": group["difficulty"].iloc[0],
                "baseline_env": easiest,
                "tau": tau,
            })
    return pd.DataFrame(rows)


def probability_of_improvement(df: pd.DataFrame, a: str, b: str) -> float:
    """P(a random run of strategy `a` beats a random run of strategy `b`).

    Ties count half, so two identical strategies score 0.5 rather than 0.0. That
    matters here: MiniGrid returns are heavily tied at exactly 0.0 on the mazes
    nothing solves, and scoring those as "a loses" would be wrong.
    """
    x = df[df["strategy"] == a]["final_return"].to_numpy()
    y = df[df["strategy"] == b]["final_return"].to_numpy()
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    return float((x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean())


def rliable_aggregate(df: pd.DataFrame, seed: int = 0) -> dict:
    """Aggregate IQM across every instance, with rliable's stratified bootstrap.

    rliable expects {strategy: array of shape (n_seeds, n_instances)}, where one
    instance is one "task". MiniGrid returns are already in [0, 1], so no
    normalisation is needed.
    """
    # Imported inside the function on purpose: rliable pulls in arch and
    # statsmodels, and every other function in this file must keep working on a
    # machine where that import is broken.
    from rliable import library as rly
    from rliable import metrics

    env_ids = sorted(df["env_id"].unique())
    scores = {}
    for strategy, group in df.groupby("strategy", sort=True):
        pivot = (group.pivot_table(index="seed", columns="env_id",
                                   values="final_return")
                      .reindex(columns=env_ids))
        gaps = pivot.isna().to_numpy()
        if gaps.any():
            # rliable would propagate the hole into a NaN aggregate that still
            # looks like a number. Refuse, the way early_auc does.
            where = sorted({f"{pivot.columns[c]}/seed{pivot.index[r]}"
                            for r, c in zip(*np.where(gaps))})
            raise ValueError(
                f"strategy {strategy!r} is missing {len(where)} run(s) "
                f"({', '.join(where)}); every strategy needs the same seeds on "
                f"every instance. Re-run them or drop the instance."
            )
        scores[strategy] = pivot.to_numpy()

    # rliable's `random_state` argument does NOT work: StratifiedBootstrap
    # overrides update_indices() and draws from the GLOBAL numpy RNG
    # (np.random.choice), ignoring the generator it was handed. Seeding the
    # global RNG is therefore the only way to get a reproducible interval.
    # This is the one place in the codebase that touches global randomness
    # (CLAUDE.md section 11); the state is restored so nothing leaks out.
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        point, interval = rly.get_interval_estimates(
            scores, lambda x: np.array([metrics.aggregate_iqm(x)]), reps=2_000)
    finally:
        np.random.set_state(state)
    return {
        strategy: {
            "iqm": float(point[strategy][0]),
            "ci_low": float(interval[strategy][0][0]),
            "ci_high": float(interval[strategy][1][0]),
        }
        for strategy in scores
    }
