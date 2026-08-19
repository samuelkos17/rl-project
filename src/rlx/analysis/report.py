"""Generate report/results.md: every number the report quotes, in one file.

Run:  python -m rlx.analysis.report --results results --out report/results.md

The report copies its numbers from this file instead of recomputing them by
hand, so the prose cannot drift away from the data. Regenerate it after every
change to the results tree, and re-read it before writing.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rlx.analysis.aggregate import FAMILIES, load_all, ordered_instances
from rlx.analysis.stats import (
    _iqm, _score_matrices, aggregate_correlation, build_analysis_table,
    compare_coverage_predictors, iqm_by_strategy, performance_profile,
    probability_of_improvement, rank_stability, rliable_aggregate,
    within_instance_correlation,
)

#: (column in the analysis table, the name the report gives it).
COVERAGE_COLUMNS = (("early_auc_raw", "Raw coverage"),
                    ("early_auc_task", "Task-relevant coverage"))

#: 11 thresholds is enough to read the shape of a profile off a table; fig5
#: draws the full curve at 21.
PROFILE_TAUS = np.linspace(0.0, 1.0, 11)


def _num(value, digits: int = 3, sign: bool = True) -> str:
    """Format a number, printing NaN as `NaN` rather than as `+nan`."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NaN"
    return f"{value:{'+' if sign else ''}.{digits}f}"


def _ci(low, high, sign: bool = True) -> str:
    """`sign=False` for returns, which are always in [0, 1] and read better
    without a leading plus; correlations keep the sign, where it is the point."""
    return f"[{_num(low, sign=sign)}, {_num(high, sign=sign)}]"


def _trend_per_family(agg: dict) -> str:
    per_family = agg["trend_per_family"]
    if not per_family:
        return "none (no family has 3+ instances spanning 2+ difficulties)"
    return ", ".join(f"{fam} {_num(rho)}" for fam, rho in sorted(per_family.items()))


def _in_report_order(per_instance: pd.DataFrame) -> pd.DataFrame:
    """Sort a per-instance table by family, then difficulty, for printing only.

    Not inside `within_instance_correlation`: `aggregate_correlation` bootstraps
    over the rows it is handed, so reordering them there would move the headline
    CI by pure resampling noise for no reason at all.
    """
    order = {env_id: i for i, env_id in enumerate(ordered_instances(per_instance))}
    # kind="stable": the full per-run table repeats each env_id 20 times, and the
    # default sort would shuffle the strategy/seed order load_all established.
    return per_instance.sort_values("env_id", key=lambda s: s.map(order), kind="stable")


def _h1_section(df: pd.DataFrame) -> str:
    lines = [
        "## H1 -- does early coverage predict final return?", "",
        "Spearman rho between early-coverage AUC and final return, computed",
        "WITHIN each instance (difficulty held constant) and only then aggregated.",
        "A correlation pooled across instances would measure \"hard environments",
        "are hard\" twice and call it a result -- see CLAUDE.md section 9.", "",
    ]
    for column, label in COVERAGE_COLUMNS:
        per = within_instance_correlation(df, column)
        agg = aggregate_correlation(per)
        lines += [
            f"### {label}", "",
            f"- Mean within-instance rho: **{_num(agg['mean_rho'])}**",
            f"- 95% bootstrap CI on that mean: {_ci(agg['ci_low'], agg['ci_high'])}",
            f"- 95% CI lies entirely above zero: **{agg['ci_above_zero']}**",
            f"- Instances with usable variance: {agg['n_instances']} of "
            f"{df['env_id'].nunique()}",
            f"- Trend with difficulty: {_num(agg['trend_with_difficulty'])} "
            f"(H1 predicts positive: stronger on harder mazes)",
            f"- Trend per family: {_trend_per_family(agg)}",
            f"- **H1 confirmed (CI entirely above zero AND trend positive): "
            f"{agg['confirms_h1']}**", "",
        ]
        if agg["n_instances"] < 3:
            # A bootstrap CI over one or two numbers is not an interval, and it
            # can still print "entirely above zero". Say so where the number
            # is, not in a footnote nobody reads.
            lines += [
                "> **Do not quote that CI.** It resamples "
                + ("a single per-instance correlation" if agg["n_instances"] == 1
                   else f"{agg['n_instances']} per-instance correlations")
                + ", so it says nothing about the spread. The honest headline "
                "when this happens is that almost every run scored the same and "
                "there was nothing left to correlate.",
                "",
            ]
        lines += [
            "Per instance. A NaN `rho` means every run on that instance scored the",
            "same, so there was nothing to correlate. That is a finding, not a gap:",
            "", _in_report_order(per).round(3).to_markdown(index=False), "",
        ]
    return "\n".join(lines)


def _identical_predictor_instances(df: pd.DataFrame) -> list[str]:
    """Instances where raw and task-relevant coverage are the same number.

    On the Empty family they are identical for every run and every seed, and not
    by accident: start and goal are opposite corners of an open grid, so every
    reachable cell lies on some shortest path and `task_relevant_mask` equals
    `reachable_mask` (ratio 1.00, measured on all three Empty instances). Those
    instances carry no information about H2 whatsoever, and averaging them in
    pulls the two correlations together -- i.e. towards "the CIs overlap", which
    is exactly the verdict H2 fails on.
    """
    identical = (df.assign(same=df["early_auc_raw"] == df["early_auc_task"])
                   .groupby("env_id")["same"].all())
    return [env_id for env_id in ordered_instances(df) if identical[env_id]]


def _h2_section(df: pd.DataFrame) -> str:
    """H2 is a comparison of two intervals, so the file states the verdict."""
    cmp = compare_coverage_predictors(df)
    raw, task = cmp["raw"], cmp["task"]
    tied = _identical_predictor_instances(df)
    # Which instances DO separate the two measures is read off the data too. On
    # the pilot, DoorKey-5 is itself one of the tied ones, so a sentence that
    # named DoorKey as the place to look would have been wrong there.
    informative = [e for e in ordered_instances(df) if e not in tied]
    where = (", ".join(informative) if informative
             else "**none of them** -- this dataset cannot test H2 at all")
    caveat = [
        f"> **{len(tied)} of {df['env_id'].nunique()} instances cannot answer this "
        f"question at all:** " + ", ".join(tied) + ". There the two coverage "
        "measures are the *same number* for every run, because every reachable "
        "cell lies on some shortest path, so the task-relevant mask equals the "
        "reachable mask. They contribute no evidence either way and drag the two "
        "correlations towards each other -- that is, towards \"the CIs overlap\". "
        f"The instances that actually separate the two measures are: {where}. "
        "Read the verdict below with that in mind, and quote those when the "
        "distinction has to be shown doing real work.",
        "",
    ] if tied else []
    return "\n".join([
        "## H2 -- is task-relevant coverage the better predictor?", "",
        *caveat,
        f"- Raw coverage: {_num(raw['mean_rho'])} "
        f"{_ci(raw['ci_low'], raw['ci_high'])}",
        f"- Task-relevant coverage: {_num(task['mean_rho'])} "
        f"{_ci(task['ci_low'], task['ci_high'])}",
        f"- Difference (task - raw): {_num(cmp['task_minus_raw'])}",
        f"- The two CIs overlap: {cmp['cis_overlap']}",
        f"- **H2 confirmed (larger AND non-overlapping CIs): "
        f"{cmp['confirms_h2']}**", "",
        "Overlapping CIs with both correlations positive is not a failed",
        "experiment: it says breadth of exploration matters and directedness",
        "does not.", "",
    ])


def _iqm_section(df: pd.DataFrame) -> str:
    rows = []
    for env_id in ordered_instances(df):
        for strategy, r in iqm_by_strategy(df, env_id).items():
            rows.append({"env_id": env_id, "strategy": strategy,
                         "iqm": round(r["iqm"], 3),
                         "ci": _ci(r["ci_low"], r["ci_high"], sign=False),
                         "n_seeds": r["n"]})
    across = rliable_aggregate(df)
    aggregate_rows = [{"strategy": s, "iqm": round(r["iqm"], 3),
                       "ci": _ci(r["ci_low"], r["ci_high"], sign=False)}
                      for s, r in sorted(across.items())]
    return "\n".join([
        "## IQM final return", "",
        "Interquartile mean: the top and bottom 25% of runs are dropped, so one",
        "lucky or unlucky seed moves it far less than it moves a mean. Every",
        "interval here is a bootstrap CI.", "",
        "### Across all instances (rliable stratified bootstrap)", "",
        pd.DataFrame(aggregate_rows).to_markdown(index=False), "",
        "### Per instance", "",
        pd.DataFrame(rows).to_markdown(index=False), "",
    ])


def _profile_section(df: pd.DataFrame) -> str:
    profile = performance_profile(df, taus=PROFILE_TAUS)
    rows = []
    for i, tau in enumerate(profile["taus"]):
        row = {"tau": round(float(tau), 2)}
        for strategy in sorted(profile["profiles"]):
            row[strategy] = (f"{profile['profiles'][strategy][i]:.2f} "
                             f"[{profile['ci_low'][strategy][i]:.2f}, "
                             f"{profile['ci_high'][strategy][i]:.2f}]")
        rows.append(row)
    return "\n".join([
        "## Performance profiles", "",
        "Fraction of runs scoring STRICTLY above each threshold `tau`, with",
        "bootstrap CIs. An IQM is one number and hides the shape of the",
        "distribution; where two profiles cross is usually the interesting part.",
        "", "The `tau = 0` row is therefore the fraction of runs that scored",
        "anything at all: a run that never once reached the goal scores exactly",
        "0.0 and does not count as above 0.", "",
        pd.DataFrame(rows).to_markdown(index=False), "",
    ])


def _improvement_section(df: pd.DataFrame) -> str:
    """P(row strategy beats column strategy), one table per family.

    Per family, not pooled over everything: a family is the unit the spec
    compares strategies within, and pooling Empty with MultiRoom would average
    over two very different tasks.
    """
    strategies = sorted(df["strategy"].unique())
    blocks = [
        "## Probability of improvement", "",
        "P(a random run of the ROW strategy beats a random run of the COLUMN",
        "strategy). Ties count half, so two identical strategies score 0.50 --",
        "which matters here, because returns are heavily tied at exactly 0.0 on",
        "the mazes nothing solves.", "",
    ]
    for family in (f for f in FAMILIES if (df["family"] == f).any()):
        group = df[df["family"] == family]
        table = pd.DataFrame(
            [[round(probability_of_improvement(group, a, b), 3) for b in strategies]
             for a in strategies],
            index=strategies, columns=strategies)
        blocks += [f"### {family}", "",
                   table.to_markdown(index=True), ""]
    return "\n".join(blocks)


def _winners_section(df: pd.DataFrame) -> str:
    """The best strategy per instance, ranked by IQM, with ties named as ties.

    Two traps this table used to fall into, both of which the pilot run
    reproduces:

    * Ranking by the MEAN. Spec 7.4 ranks strategies by IQM, and so does
      `rank_stability`, so a mean-ranked winners table could contradict the
      table printed directly above it.
    * Picking one name out of an exact tie. On the pilot, all four strategies
      score exactly 0.0 on DoorKey-5, and epsilon-greedy and NoisyNets score
      exactly 0.23875 on Empty-5. Sorting and taking the first row named a
      winner in both cases -- an ordering artefact that would have been written
      into the report as a result.
    """
    rows = []
    for env_id in ordered_instances(df):
        group = df[df["env_id"] == env_id]
        iqm = group.groupby("strategy")["final_return"].apply(
            lambda v: _iqm(v.to_numpy()))
        best = float(iqm.max())
        # Exact equality on purpose: these are ties because the numbers are
        # identical, not because they are close. Two strategies that merely sit
        # near each other are separated by their CIs, not by this column.
        tied = sorted(iqm.index[iqm == best])
        rows.append({
            "env_id": env_id,
            "best_strategy": ("none -- no strategy ever reached the goal"
                              if best <= 0.0 else " = ".join(tied)),
            "iqm": round(best, 3),
            "tied_strategies": len(tied),
        })
    return "\n".join([
        "## Best strategy per instance", "",
        "Ranked by IQM, the same statistic as the rank-stability table, so the",
        "two cannot contradict each other. An instance nothing solved has no",
        "winner and says so; strategies with an identical IQM are all named.",
        "Before calling any of these a win, check whether the CIs in the IQM",
        "table above actually separate it from the runner-up.", "",
        pd.DataFrame(rows).to_markdown(index=False), "",
    ])


def build_report(results_root: Path, out_path: Path) -> None:
    """Write every number the report needs to `out_path` as one markdown file.

    Refuses two inputs rather than producing a file that looks generated:
    an empty results tree, and a run matrix with a hole in it (some strategy
    missing a seed on some instance). Nothing is written in either case.
    """
    runs = load_all(Path(results_root))
    if not runs:
        raise ValueError(f"no runs found under {results_root}")
    df = build_analysis_table(runs)
    # Same pre-flight as make_all_figures: rliable needs a complete
    # (seeds x instances) matrix, and the message that says which run is missing
    # is worth more at the start than after two minutes of bootstrapping.
    _score_matrices(df)

    # Everything is built before anything is written: a failure halfway through
    # must not leave a half-written results file that still looks generated.
    sections = [
        "# Results\n",
        "Generated by `python -m rlx.analysis.report`. Do not edit by hand.\n",
        f"- Runs analysed: **{len(df)}**",
        f"- Environment instances: **{df['env_id'].nunique()}**",
        f"- Strategies: **{df['strategy'].nunique()}**",
        f"- Seeds per configuration: **{df.groupby(['env_id', 'strategy']).size().max()}**\n",
        _h1_section(df),
        _h2_section(df),
        _iqm_section(df),
        _profile_section(df),
        _improvement_section(df),
        "## Rank stability (H3)\n",
        "Kendall's tau between each instance's strategy ranking (by IQM) and the\n"
        "ranking on the easiest instance of the same family. 1.0 is the same\n"
        "order, -1.0 exactly reversed, 0 unrelated.\n",
        _in_report_order(rank_stability(df)).round(3).to_markdown(index=False) + "\n",
        _winners_section(df),
        "## Full per-run table\n",
        _in_report_order(df).round(4).to_markdown(index=False) + "\n",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("report/results.md"))
    args = parser.parse_args()
    build_report(args.results, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
