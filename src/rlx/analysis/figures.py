"""Every figure in the report. Run:

    python -m rlx.analysis.figures --results results --out report/figures

Conventions, fixed here so all seven look like one report:
  * one colour per strategy, identical everywhere; grey is the baseline
  * every shaded band and error bar is a BOOTSTRAP CI, never a standard deviation
  * axis labels carry units or a range
  * no titles inside a figure -- captions belong in the report text; panel labels
    only identify which panel is which
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

from rlx.analysis.aggregate import RunResult, load_all  # noqa: E402
from rlx.analysis.coverage import raw_coverage, task_relevant_coverage  # noqa: E402
from rlx.analysis.stats import (  # noqa: E402
    aggregate_correlation, build_analysis_table, iqm_by_strategy,
    performance_profile, rank_stability, within_instance_correlation,
)
from rlx.envs import grid_info  # noqa: E402

FIGURE_NAMES = (
    "fig1_learning_curves", "fig2_difficulty_curve", "fig3_coverage_curves",
    "fig4_coverage_vs_return", "fig5_iqm", "fig6_rank_stability",
    "fig7_visitation_heatmaps",
)

#: Okabe-Ito, which stays distinguishable for the common colour-vision
#: deficiencies and in greyscale. epsilon-greedy is grey because it is the
#: baseline everything else is compared against.
COLORS = {
    "epsilon_greedy": "#888888",
    "boltzmann": "#0072B2",
    "count_based": "#D55E00",
    "noisy": "#009E73",
}
LABELS = {
    "epsilon_greedy": "$\\epsilon$-greedy",
    "boltzmann": "Boltzmann",
    "count_based": "Count-based",
    "noisy": "NoisyNets",
}
FAMILIES = ("Empty", "DoorKey", "MultiRoom")
DIFFICULTY_LABEL = {"Empty": "Grid size", "DoorKey": "Grid size",
                    "MultiRoom": "Number of rooms"}
N_BAND_RESAMPLES = 1_000


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _bootstrap_band(values: np.ndarray, seed: int = 0):
    """Mean and 95% bootstrap CI across runs, computed per column.

    A CI, not a standard deviation: an SD band looks tighter and would make the
    results seem more certain than the number of seeds justifies.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    draws = values[rng.integers(0, len(values), (N_BAND_RESAMPLES, len(values)))].mean(axis=1)
    return (values.mean(axis=0),
            np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0))


def _eval_curve(run: RunResult):
    """(steps, returns) at the evaluation points only.

    metrics.csv holds one row per logged step and training may log more often
    than it evaluates, so the eval column carries NaN on most rows.
    """
    rows = run.metrics.dropna(subset=["eval_return_mean"])
    return rows["step"].to_numpy(), rows["eval_return_mean"].to_numpy()


def _stack_curves(curves: list[np.ndarray]) -> np.ndarray:
    """Truncate a set of curves to their common length and stack them."""
    n = min(len(c) for c in curves)
    return np.stack([c[:n] for c in curves])


def _legend(ax) -> None:
    ax.legend(frameon=False, fontsize=9)


def _ordered_instances(df: pd.DataFrame) -> list[str]:
    """Instance names ordered by family, then difficulty.

    Sorting the names alphabetically puts DoorKey-10 before DoorKey-5 and
    Empty-16 before Empty-5, so difficulty would not read left to right.
    """
    order = (df[["env_id", "family", "difficulty"]].drop_duplicates()
               .sort_values(["family", "difficulty"]))
    return order["env_id"].tolist()


def _step_axis(ax) -> None:
    """Label the x axis in thousands of steps.

    Runs are 400,000 steps long, and six-digit ticks collide into one unreadable
    run of digits at report width.
    """
    ax.set_xlabel("Environment steps")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))


def fig1_learning_curves(runs, df, out_dir) -> None:
    """Return over training, one panel per family, bootstrap CI across runs."""
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(15, 4), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        for strategy in COLORS:
            group = [r for r in runs
                     if r.strategy == strategy and r.env_id.startswith(family)]
            if not group:
                continue
            curves = [_eval_curve(r)[1] for r in group]
            stacked = _stack_curves(curves)
            steps = _eval_curve(group[0])[0][:stacked.shape[1]]
            mean, low, high = _bootstrap_band(stacked)
            ax.plot(steps, mean, color=COLORS[strategy], label=LABELS[strategy], lw=1.6)
            ax.fill_between(steps, low, high, color=COLORS[strategy], alpha=0.18, lw=0)
        _step_axis(ax)
        ax.set_title(family, fontsize=10)
        ax.margins(x=0)
    axes[0].set_ylabel("Evaluation return (0-1)")
    _legend(axes[-1])
    _save(fig, out_dir, "fig1_learning_curves")


def fig2_difficulty_curve(runs, df, out_dir) -> None:
    """Final return against difficulty. The curve the professor asked for."""
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(15, 4), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        family_rows = df[df["family"] == family]
        if family_rows.empty:
            continue
        ticks = sorted(family_rows["difficulty"].unique())
        # Four strategies share every x position, so their intervals would sit on
        # top of each other. Nudge them apart by a fraction of the axis span.
        dodge = 0.02 * max(max(ticks) - min(ticks), 1)
        for i, strategy in enumerate(COLORS):
            sub = family_rows[family_rows["strategy"] == strategy]
            if sub.empty:
                continue
            levels, means, low, high = [], [], [], []
            for difficulty, group in sub.groupby("difficulty"):
                m, lo, hi = _bootstrap_band(group["final_return"].to_numpy()[:, None])
                levels.append(difficulty + (i - 1.5) * dodge)
                means.append(m[0])
                low.append(m[0] - lo[0])
                high.append(hi[0] - m[0])
            ax.errorbar(levels, means, yerr=[low, high], marker="o", capsize=3,
                        color=COLORS[strategy], label=LABELS[strategy], lw=1.6, ms=5)
        ax.set_xticks(ticks)
        ax.set_xlabel(DIFFICULTY_LABEL[family])
        ax.set_title(family, fontsize=10)
    axes[0].set_ylabel("Final return (0-1)")
    _legend(axes[-1])
    _save(fig, out_dir, "fig2_difficulty_curve")


def fig3_coverage_curves(runs, df, out_dir) -> None:
    """Both coverage measures over training, pooled across instances."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    panels = ((raw_coverage, "Raw coverage"),
              (task_relevant_coverage, "Task-relevant coverage"))
    for ax, (measure, panel) in zip(axes, panels):
        for strategy in COLORS:
            group = [r for r in runs if r.strategy == strategy]
            if not group:
                continue
            curves = [measure(r.counts, grid_info(r.env_id, r.seed)) for r in group]
            stacked = _stack_curves(curves)
            steps = group[0].steps[:stacked.shape[1]]
            mean, low, high = _bootstrap_band(stacked)
            ax.plot(steps, mean, color=COLORS[strategy], label=LABELS[strategy], lw=1.6)
            ax.fill_between(steps, low, high, color=COLORS[strategy], alpha=0.18, lw=0)
        _step_axis(ax)
        ax.set_title(panel, fontsize=10)
        ax.margins(x=0)
    axes[0].set_ylabel("Fraction of states visited (0-1)")
    _legend(axes[-1])
    _save(fig, out_dir, "fig3_coverage_curves")


def fig4_coverage_vs_return(runs, df, out_dir) -> None:
    """THE central result: early coverage against final return, per instance.

    Panels 1-2 are the scatter with ONE REGRESSION LINE PER INSTANCE. Never one
    line through everything: coverage and return both fall with difficulty, so a
    pooled line measures "hard mazes are hard" twice. See CLAUDE.md section 9.

    Panel 3 shows each instance's own correlation with its bootstrap interval,
    which is what separates a solid per-maze result from a coincidence.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    per_instance = {}
    for ax, (column, panel) in zip(axes, (("early_auc_raw", "Raw coverage"),
                                          ("early_auc_task", "Task-relevant coverage"))):
        for strategy in COLORS:
            sub = df[df["strategy"] == strategy]
            ax.scatter(sub[column], sub["final_return"], s=22, alpha=0.75,
                       color=COLORS[strategy], label=LABELS[strategy], linewidths=0)
        for _, group in df.groupby("env_id"):
            if group[column].nunique() < 2:
                continue
            slope, intercept = np.polyfit(group[column], group["final_return"], 1)
            xs = np.linspace(group[column].min(), group[column].max(), 10)
            ax.plot(xs, slope * xs + intercept, color="black", alpha=0.18, lw=1)

        per_instance[column] = within_instance_correlation(df, column)
        agg = aggregate_correlation(per_instance[column])
        ax.set_xlabel("Early-coverage AUC (first 20% of training)")
        ax.set_title(f"{panel}\nwithin-instance $\\rho$ = {agg['mean_rho']:+.2f} "
                     f"[{agg['ci_low']:+.2f}, {agg['ci_high']:+.2f}]", fontsize=10)
    axes[0].set_ylabel("Final return (0-1)")
    _legend(axes[1])

    forest = per_instance["early_auc_raw"].sort_values(["family", "difficulty"])
    forest = forest.reset_index(drop=True)
    positions = np.arange(len(forest))
    # Families are told apart by MARKER, not colour: the four strategy colours
    # already mean something else in the two panels to the left, and reusing them
    # here for families inside the same figure would be genuinely misleading.
    for marker, (family, group) in zip("os^", forest.groupby("family")):
        axes[2].errorbar(
            group["rho"], group.index.to_numpy(),
            xerr=[group["rho"] - group["rho_ci_low"], group["rho_ci_high"] - group["rho"]],
            fmt=marker, ms=4, capsize=2, lw=1.2, color="#333333",
            markerfacecolor="white", label=family)
    axes[2].axvline(0, color="black", lw=0.8, ls=":")
    axes[2].set_yticks(positions)
    axes[2].set_yticklabels(forest["env_id"], fontsize=8)
    axes[2].set_xlim(-1.05, 1.05)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Within-instance Spearman $\\rho$ (raw coverage)")
    axes[2].set_title("Per instance, with bootstrap CI", fontsize=10)
    _legend(axes[2])
    _save(fig, out_dir, "fig4_coverage_vs_return")


def fig5_iqm(runs, df, out_dir) -> None:
    """Which strategy wins, two ways: one robust number, and the whole shape.

    An IQM hides the distribution -- two strategies with the same IQM can differ
    completely in how often they solve a maze at all. The performance profile on
    the right shows that (spec 7.2), and where two profiles cross is usually the
    interesting part.
    """
    env_ids = _ordered_instances(df)
    # Once per instance, not once per (instance, strategy): each call already
    # returns every strategy and runs its own bootstrap.
    by_env = {env_id: iqm_by_strategy(df, env_id) for env_id in env_ids}

    fig, axes = plt.subplots(1, 2, figsize=(max(13, len(env_ids) * 1.05), 4.6),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    width = 0.2
    for i, strategy in enumerate(COLORS):
        xs, ys, errs = [], [], []
        for j, env_id in enumerate(env_ids):
            result = by_env[env_id].get(strategy)
            if result is None:
                continue
            xs.append(j + (i - 1.5) * width)
            ys.append(result["iqm"])
            errs.append([result["iqm"] - result["ci_low"], result["ci_high"] - result["iqm"]])
        axes[0].bar(xs, ys, width, color=COLORS[strategy], label=LABELS[strategy])
        axes[0].errorbar(xs, ys, yerr=np.array(errs).T, fmt="none", ecolor="black",
                         capsize=2, lw=0.8)
    axes[0].set_xticks(range(len(env_ids)))
    axes[0].set_xticklabels(env_ids, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("IQM final return (0-1)")
    axes[0].set_title("Per instance", fontsize=10)

    profile = performance_profile(df)
    for strategy in COLORS:
        if strategy not in profile["profiles"]:
            continue
        axes[1].plot(profile["taus"], profile["profiles"][strategy],
                     color=COLORS[strategy], label=LABELS[strategy], lw=1.6)
        axes[1].fill_between(profile["taus"], profile["ci_low"][strategy],
                             profile["ci_high"][strategy],
                             color=COLORS[strategy], alpha=0.18, lw=0)
    axes[1].set_xlabel("Return threshold $\\tau$ (0-1)")
    axes[1].set_ylabel("Fraction of runs scoring above $\\tau$")
    axes[1].set_title("Performance profile, all instances", fontsize=10)
    axes[1].margins(x=0)
    # The bar panel is full edge to edge, so the shared legend lives here, in the
    # one corner a decreasing profile always leaves empty.
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    _save(fig, out_dir, "fig5_iqm")


def fig6_rank_stability(runs, df, out_dir) -> None:
    """Does the strategy that wins on easy mazes still win on hard ones? (H3)

    One panel per family, like fig1 and fig2, and for the same reason: difficulty
    is grid size for Empty/DoorKey but room count for MultiRoom, so putting all
    three on one x axis would place MultiRoom-N6 beside DoorKey-6 as though the
    two numbers meant the same thing.
    """
    stability = rank_stability(df)
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(13, 3.8), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        group = stability[stability["family"] == family].sort_values("difficulty")
        if not group.empty:
            ax.plot(group["difficulty"], group["tau"], marker="o", color="#333333",
                    lw=1.6, ms=5)
            ax.set_xticks(group["difficulty"].tolist())
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel(DIFFICULTY_LABEL[family])
        ax.set_title(family, fontsize=10)
    axes[0].set_ylabel("Kendall's $\\tau$ vs easiest\ninstance of the family (-1 to 1)")
    _save(fig, out_dir, "fig6_rank_stability")


def fig7_visitation_heatmaps(runs, df, out_dir, env_id: str | None = None,
                             seed: int = 0) -> None:
    """Where each strategy actually went. The poster figure.

    All four panels share one colour scale, so they can be compared at all: with
    per-panel scaling every strategy looks equally thorough, which is the exact
    opposite of the point.
    """
    env_id = env_id or ("DoorKey-8" if any(r.env_id == "DoorKey-8" for r in runs)
                        else runs[0].env_id)
    panels = {}
    for strategy in COLORS:
        match = [r for r in runs if r.env_id == env_id
                 and r.strategy == strategy and r.seed == seed]
        if match:
            # Sum over the 4 directions: the picture is about places, not facings.
            panels[strategy] = np.log1p(match[0].counts[-1].sum(axis=2)).T
    if not panels:
        raise ValueError(f"no runs for {env_id} seed {seed} to draw heatmaps from")

    top = max(float(grid.max()) for grid in panels.values()) or 1.0
    fig, axes = plt.subplots(1, len(COLORS), figsize=(15, 4))
    image = None
    for ax, strategy in zip(axes, COLORS):
        if strategy not in panels:
            ax.axis("off")
            continue
        image = ax.imshow(panels[strategy], cmap="viridis", origin="upper",
                          vmin=0.0, vmax=top)
        ax.set_title(LABELS[strategy], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    # Instance identity belongs in the caption, but the poster shows this figure
    # without one, so it goes on the axis rather than into a title.
    axes[0].set_ylabel(f"{env_id}, seed {seed}", fontsize=9)
    if image is not None:
        bar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
        bar.set_label("log(1 + visits)", fontsize=9)
    _save(fig, out_dir, "fig7_visitation_heatmaps")


def make_all_figures(results_root: Path, out_dir: Path) -> None:
    """Render every report figure from a results tree."""
    runs = load_all(Path(results_root))
    if not runs:
        raise ValueError(f"no runs found under {results_root}")
    df = build_analysis_table(runs)
    for draw in (fig1_learning_curves, fig2_difficulty_curve, fig3_coverage_curves,
                 fig4_coverage_vs_return, fig5_iqm, fig6_rank_stability,
                 fig7_visitation_heatmaps):
        draw(runs, df, Path(out_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("report/figures"))
    args = parser.parse_args()
    make_all_figures(args.results, args.out)
    print(f"wrote {len(FIGURE_NAMES)} figures to {args.out}")


if __name__ == "__main__":
    main()
