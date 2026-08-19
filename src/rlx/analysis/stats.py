"""Statistical analysis.

CRITICAL -- read CLAUDE.md section 9 before touching this file.

Coverage and return both fall as difficulty rises, so a correlation pooled across
environment instances is a difficulty artefact and means nothing. Every
correlation here is computed WITHIN an instance, where difficulty is constant,
and only then aggregated.
"""

from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy import stats

from rlx.analysis.aggregate import RunResult, to_dataframe
from rlx.analysis.coverage import early_auc, raw_coverage, task_relevant_coverage
from rlx.envs import grid_info

N_BOOTSTRAP = 10_000

#: Per-instance rho intervals resample 20 runs, 13 times over, on every call.
#: 2000 is enough for a 95% percentile interval and keeps the analysis snappy.
N_BOOTSTRAP_RHO = 2_000


def build_analysis_table(runs: list[RunResult]) -> pd.DataFrame:
    """One row per run: identity, final return, and both early-coverage AUCs.

    Every unusable run is collected before raising. Stopping at the first one
    meant a 260-run sweep reported its bad snapshot grids one per re-run; now a
    single failure message names all of them, so they can be re-run or excluded
    together.
    """
    df = to_dataframe(runs)
    raw_auc, task_auc, failures = [], [], []
    for r in runs:
        # layout_seed == seed: the maze a run saw is determined by its seed.
        info = grid_info(r.env_id, layout_seed=r.seed)
        total = r.config["total_steps"]
        try:
            raw_auc.append(early_auc(r.steps, raw_coverage(r.counts, info), total))
            task_auc.append(early_auc(r.steps, task_relevant_coverage(r.counts, info), total))
        except ValueError as exc:
            failures.append(f"{r.env_id}/{r.strategy}/seed{r.seed}: {exc}")

    if failures:
        raise ValueError(
            f"{len(failures)} of {len(runs)} runs have no usable early-coverage "
            f"window:\n  " + "\n  ".join(failures))

    df["early_auc_raw"] = raw_auc
    df["early_auc_task"] = task_auc
    return df


def _bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray,
                           rng: np.random.Generator) -> tuple[float, float]:
    """95% CI on one instance's Spearman rho, resampling that instance's runs.

    Spearman is Pearson on ranks, so the whole bootstrap is one vectorised pass:
    rank every resample at once instead of calling scipy N_BOOTSTRAP times.
    """
    idx = rng.integers(0, len(x), size=(N_BOOTSTRAP_RHO, len(x)))
    rx = stats.rankdata(x[idx], axis=1)
    ry = stats.rankdata(y[idx], axis=1)
    rx = rx - rx.mean(axis=1, keepdims=True)
    ry = ry - ry.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = ((rx * ry).sum(axis=1) /
               np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1)))
    # A resample can draw the same run 20 times, leaving a constant column whose
    # correlation is undefined. Those are dropped, not counted as zero.
    rho = rho[np.isfinite(rho)]
    if len(rho) == 0:
        return np.nan, np.nan
    return float(np.percentile(rho, 2.5)), float(np.percentile(rho, 97.5))


def within_instance_correlation(df: pd.DataFrame, coverage_col: str,
                                seed: int = 0) -> pd.DataFrame:
    """Spearman correlation of coverage vs final return, computed per instance.

    NEVER pool instances. See the module docstring.

    Each instance also gets its own bootstrap CI over its runs (spec 7.3 step 2),
    which is what says whether a single per-instance rho is worth reading at all
    -- with 20 runs, a rho of 0.4 can easily carry an interval spanning zero.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for env_id, group in df.groupby("env_id", sort=True):
        if group["final_return"].nunique() < 2 or group[coverage_col].nunique() < 2:
            # no variance -- a finding, not an error, and no interval to report
            rho, p, ci_low, ci_high = np.nan, np.nan, np.nan, np.nan
        else:
            rho, p = stats.spearmanr(group[coverage_col], group["final_return"])
            ci_low, ci_high = _bootstrap_spearman_ci(
                group[coverage_col].to_numpy(), group["final_return"].to_numpy(), rng)
        rows.append({
            "env_id": env_id,
            "difficulty": group["difficulty"].iloc[0],
            "family": group["family"].iloc[0] if "family" in group else env_id.split("-")[0],
            "n_runs": len(group),
            "rho": rho,
            "p_value": p,
            "rho_ci_low": ci_low,
            "rho_ci_high": ci_high,
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

    `confirms_h1` requires BOTH conditions spec section 1 states: the CI on the
    mean excludes zero, AND the correlation grows with difficulty. `ci_excludes_zero`
    is reported separately because it is the half most likely to hold on its own.
    A NaN trend leaves `confirms_h1` False -- "not confirmed", which is not the
    same as "disconfirmed"; read `trend_with_difficulty` to tell the two apart.
    """
    valid = per_instance.dropna(subset=["rho"])
    per_family = {}
    if len(valid):
        rho = valid["rho"].to_numpy()
        mean_rho = float(rho.mean())
        ci_low, ci_high = _bootstrap_mean_ci(rho, np.random.default_rng(seed))
        family = (valid["family"] if "family" in valid
                  else valid["env_id"].str.split("-").str[0])
        for fam, g in valid.groupby(family, sort=True):
            if len(g) >= 3 and g["difficulty"].nunique() >= 2:
                per_family[fam] = float(stats.spearmanr(g["difficulty"], g["rho"]).statistic)
        trend = float(np.mean(list(per_family.values()))) if per_family else np.nan
    else:
        mean_rho = ci_low = ci_high = trend = np.nan

    # One return shape in every branch. The degenerate branch used to omit
    # confirms_h1, so a fully tied sweep raised KeyError inside report.py.
    ci_excludes_zero = bool(ci_low > 0)
    return {
        "mean_rho": mean_rho,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_instances": len(valid),
        "trend_with_difficulty": trend,
        "trend_per_family": per_family,
        "ci_excludes_zero": ci_excludes_zero,
        "confirms_h1": bool(ci_excludes_zero and trend > 0),
    }


def compare_coverage_predictors(df: pd.DataFrame, seed: int = 0) -> dict:
    """H2: does task-relevant coverage predict final return better than raw?

    Spec section 1 sets two conditions, and the second is the one that is easy to
    skip: the task-relevant correlation must be LARGER *and* the two CIs must not
    overlap. Two means 0.70 and 0.64 with intervals [0.54, 0.84] and [0.51, 0.77]
    are not evidence that one predictor beats the other.

    Overlapping CIs with both correlations positive is the spec's "interesting
    alternative", not a failed experiment: breadth of exploration matters,
    directedness does not.
    """
    both = {
        label: aggregate_correlation(
            within_instance_correlation(df, column, seed=seed), seed=seed)
        for label, column in (("raw", "early_auc_raw"), ("task", "early_auc_task"))
    }
    raw, task = both["raw"], both["task"]
    separated = bool(task["ci_low"] > raw["ci_high"] or raw["ci_low"] > task["ci_high"])
    return {
        "raw": raw,
        "task": task,
        "task_minus_raw": task["mean_rho"] - raw["mean_rho"],
        "cis_overlap": not separated,
        "confirms_h2": bool(task["mean_rho"] > raw["mean_rho"] and separated),
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

    Strategies are ranked by IQM, not by mean (spec 7.4). With 5 seeds a single
    collapsed run moves a mean far enough to reorder two strategies that are
    otherwise clearly separated, and H3 would then report a rank change that
    never happened. Measured on the synthetic fixture: mean and IQM disagree on
    7 of the 13 instances.
    """
    df = df.copy()
    if "family" not in df:
        df["family"] = df["env_id"].str.split("-").str[0]

    rows = []
    for family, fam_group in df.groupby("family", sort=True):
        easiest = fam_group.loc[fam_group["difficulty"].idxmin(), "env_id"]
        baseline = (fam_group[fam_group["env_id"] == easiest]
                    .groupby("strategy")["final_return"]
                    .apply(lambda v: _iqm(v.to_numpy())).sort_index())
        for env_id, group in fam_group.groupby("env_id", sort=True):
            here = (group.groupby("strategy")["final_return"]
                         .apply(lambda v: _iqm(v.to_numpy())).sort_index())
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


@contextmanager
def _global_numpy_seed(seed: int):
    """Seed the GLOBAL numpy RNG for the duration of a block, then restore it.

    rliable's `random_state` argument does not work: StratifiedBootstrap
    overrides update_indices() and draws with np.random.choice, ignoring the
    generator it was handed. Seeding the global RNG is the only way to get a
    reproducible interval out of it. This is the one place in the codebase that
    touches global randomness (CLAUDE.md section 11), and the previous state is
    put back so nothing leaks into the caller.
    """
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


def _score_matrices(df: pd.DataFrame) -> dict:
    """{strategy: (n_seeds, n_instances) final returns}, the shape rliable wants.

    Refuses a matrix with a hole in it. rliable would turn a missing run into a
    NaN aggregate that still looks like a number; a crashed run must not become a
    quiet blank in the report. Same rule as early_auc.
    """
    env_ids = sorted(df["env_id"].unique())
    scores = {}
    for strategy, group in df.groupby("strategy", sort=True):
        pivot = (group.pivot_table(index="seed", columns="env_id",
                                   values="final_return")
                      .reindex(columns=env_ids))
        gaps = pivot.isna().to_numpy()
        if gaps.any():
            where = sorted({f"{pivot.columns[c]}/seed{pivot.index[r]}"
                            for r, c in zip(*np.where(gaps))})
            raise ValueError(
                f"strategy {strategy!r} is missing {len(where)} run(s) "
                f"({', '.join(where)}); every strategy needs the same seeds on "
                f"every instance. Re-run them or drop the instance."
            )
        scores[strategy] = pivot.to_numpy()
    return scores


def rliable_aggregate(df: pd.DataFrame, seed: int = 0) -> dict:
    """Aggregate IQM across every instance, with rliable's stratified bootstrap.

    One instance is one "task". MiniGrid returns are already in [0, 1], so no
    normalisation is needed.
    """
    # Imported inside the function on purpose: rliable pulls in arch and
    # statsmodels, and every other function in this file must keep working on a
    # machine where that import is broken.
    from rliable import library as rly
    from rliable import metrics

    scores = _score_matrices(df)
    with _global_numpy_seed(seed):
        point, interval = rly.get_interval_estimates(
            scores, lambda x: np.array([metrics.aggregate_iqm(x)]), reps=2_000)
    return {
        strategy: {
            "iqm": float(point[strategy][0]),
            "ci_low": float(interval[strategy][0][0]),
            "ci_high": float(interval[strategy][1][0]),
        }
        for strategy in scores
    }


def performance_profile(df: pd.DataFrame, taus=None, seed: int = 0) -> dict:
    """Fraction of runs scoring above each threshold, per strategy (spec 7.2).

    An IQM is one number and hides the shape of the distribution: two strategies
    with the same IQM can differ completely in how often they solve a maze at all
    versus solve it slowly. The profile shows the whole distribution, and where
    two profiles cross is usually the interesting part.
    """
    from rliable import library as rly

    scores = _score_matrices(df)
    taus = np.linspace(0.0, 1.0, 21) if taus is None else np.asarray(taus, dtype=float)
    with _global_numpy_seed(seed):
        profiles, intervals = rly.create_performance_profile(scores, taus, reps=2_000)
    return {
        "taus": taus,
        "profiles": {k: np.asarray(v) for k, v in profiles.items()},
        "ci_low": {k: np.asarray(v[0]) for k, v in intervals.items()},
        "ci_high": {k: np.asarray(v[1]) for k, v in intervals.items()},
    }
