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
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

from rlx.analysis.aggregate import (  # noqa: E402
    FAMILIES, RunResult, load_all, ordered_instances,
)
from rlx.analysis.coverage import raw_coverage, task_relevant_coverage  # noqa: E402
from rlx.analysis.stats import (  # noqa: E402
    _score_matrices, build_analysis_table, iqm_by_strategy,
    performance_profile, rank_stability, rliable_aggregate,
    within_instance_correlation,
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
DIFFICULTY_LABEL = {"Empty": "Grid size", "DoorKey": "Grid size",
                    "MultiRoom": "Number of rooms"}
N_BAND_RESAMPLES = 1_000

#: Applied module-wide, so the seven figures look like one document. The report
#: reproduces them at 5.5in inside 10pt text, where matplotlib's defaults put the
#: frame at a heavier weight than the data and the tick labels below reading
#: size. A horizontal grid only: every figure here is read off the y axis.
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


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


def _stack_curves(curves: list[np.ndarray], what: str) -> np.ndarray:
    """Stack equal-length curves. Refuses ragged input rather than truncating.

    fig1 and fig3 average a whole family of runs together, so one run with fewer
    evaluation or snapshot points used to silently shorten every other curve in
    the panel -- the figure still drew, just over less training than its axis
    claimed. Runs of one sweep share `total_steps`, `eval_every` and
    `snapshot_every`, so ragged input means two different configurations got
    mixed into one results tree. Same rule as early_auc and _score_matrices: say
    so, do not quietly stand in for it.
    """
    lengths = sorted({len(c) for c in curves})
    if len(lengths) > 1:
        raise ValueError(
            f"{what}: runs disagree on their number of points ({lengths}). "
            f"Averaging them would truncate every curve to {lengths[0]} while the "
            f"axis still claims the full run. The results tree mixes two "
            f"configurations -- separate them, or re-run the odd ones out."
        )
    return np.stack(curves)


def _legend(ax) -> None:
    ax.legend(frameon=False, fontsize=9)


def _present_families(df: pd.DataFrame) -> tuple:
    """The families that actually have runs, in the fixed report order.

    configs/pilot.yaml runs two of the three, and drawing a panel for the absent
    one left an empty axis autoscaled to nonsense ticks -- with the legend, which
    goes on the last panel, stranded inside it.
    """
    return tuple(family for family in FAMILIES if (df["family"] == family).any())


def _family_axes(families: tuple, height: float = 2.4):
    """One panel per present family, sized so a two-family figure is not stretched.

    Drawn at 2.5in per panel, not 5in. The report places these at 5.5in wide, so
    a 15in figure is scaled by 0.37 and 10pt tick labels arrive as 3.7pt --
    exactly the "don't make us zoom in to read your axis descriptions" the
    template warns about. At 7.5in the scale factor is 0.73 and the same labels
    arrive at 7.3pt.
    """
    fig, axes = plt.subplots(1, len(families), figsize=(2.5 * len(families), height),
                             sharey=True, squeeze=False)
    return fig, axes[0]


def _degenerate_note(group: pd.DataFrame) -> str | None:
    """Label for a family whose rank correlations are all undefined, else None.

    Kendall's tau needs two different rankings to compare. When every strategy
    scores the same on every instance of a family -- which happens as soon as
    nothing solves the maze -- there is no ranking and tau is NaN for the whole
    family. Left alone the panel is simply empty, which reads as a broken plot
    rather than as a result. The project's rule is "no variance, excluded",
    never a silent NaN.
    """
    if group["tau"].isna().all():
        return "no variance:\nevery strategy scored\nthe same"
    return None


def _step_axis(ax) -> None:
    """Label the x axis in thousands of steps.

    Runs are 400,000 steps long, and six-digit ticks collide into one unreadable
    run of digits at report width.
    """
    ax.set_xlabel("Environment steps")
    # 3, not 5: at 2.5in per panel five ticks on a 400,000-step axis print
    # as "160k240k320k" with no gap between them.
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))


def fig1_learning_curves(runs: list[RunResult], df: pd.DataFrame,
                         out_dir: Path) -> None:
    """Return over training, one panel per family, bootstrap CI across runs."""
    families = _present_families(df)
    fig, axes = _family_axes(families)
    for ax, family in zip(axes, families):
        for strategy in COLORS:
            group = [r for r in runs
                     if r.strategy == strategy and r.env_id.startswith(family)]
            if not group:
                continue
            curves = [_eval_curve(r)[1] for r in group]
            stacked = _stack_curves(curves, f"fig1 {family}/{strategy}")
            steps = _eval_curve(group[0])[0]
            mean, low, high = _bootstrap_band(stacked)
            ax.plot(steps, mean, color=COLORS[strategy], label=LABELS[strategy], lw=1.6)
            ax.fill_between(steps, low, high, color=COLORS[strategy], alpha=0.18, lw=0)
        _step_axis(ax)
        ax.set_title(family, fontsize=10)
        ax.margins(x=0)
    axes[0].set_ylabel("Evaluation return (0-1)")
    _legend(axes[-1])
    _save(fig, out_dir, "fig1_learning_curves")


def fig2_difficulty_curve(runs: list[RunResult], df: pd.DataFrame,
                          out_dir: Path) -> None:
    """Final return against difficulty. The curve the professor asked for."""
    families = _present_families(df)
    fig, axes = _family_axes(families)
    for ax, family in zip(axes, families):
        family_rows = df[df["family"] == family]
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
            # The intervals are genuinely this wide -- five seeds, and on DoorKey
            # some of them solve the maze while others never do. Drawn at full
            # weight they are all the eye sees, so they are thinned and lightened
            # to let the means carry the trend the panel is about.
            ax.errorbar(levels, means, yerr=[low, high], marker="o", capsize=2,
                        color=COLORS[strategy], label=LABELS[strategy], lw=1.6, ms=5,
                        elinewidth=0.9, ecolor=tuple(
                            0.55 * c + 0.45 for c in to_rgb(COLORS[strategy])))
        ax.set_xticks(ticks)
        ax.set_xlabel(DIFFICULTY_LABEL[family])
        ax.set_title(family, fontsize=10)
    # "Mean", explicitly: fig5 plots the IQM of the same quantity, and an
    # unqualified "final return" on both invites cross-reading two bar heights
    # that are not the same statistic.
    axes[0].set_ylabel("Mean final return (0-1)")
    _legend(axes[-1])
    _save(fig, out_dir, "fig2_difficulty_curve")


def fig3_coverage_curves(runs: list[RunResult], df: pd.DataFrame,
                         out_dir: Path) -> None:
    """Both coverage measures over training, pooled across instances."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    panels = ((raw_coverage, "Raw coverage"),
              (task_relevant_coverage, "Task-relevant coverage"))
    for ax, (measure, panel) in zip(axes, panels):
        for strategy in COLORS:
            group = [r for r in runs if r.strategy == strategy]
            if not group:
                continue
            curves = [measure(r.counts, grid_info(r.env_id, r.seed)) for r in group]
            stacked = _stack_curves(curves, f"fig3 {panel}/{strategy}")
            steps = group[0].steps
            mean, low, high = _bootstrap_band(stacked)
            ax.plot(steps, mean, color=COLORS[strategy], label=LABELS[strategy], lw=1.6)
            ax.fill_between(steps, low, high, color=COLORS[strategy], alpha=0.18, lw=0)
        _step_axis(ax)
        ax.set_title(panel, fontsize=10)
        ax.margins(x=0)
    axes[0].set_ylabel("Fraction of states visited (0-1)")
    _legend(axes[-1])
    _save(fig, out_dir, "fig3_coverage_curves")


def fig4_coverage_vs_return(runs: list[RunResult], df: pd.DataFrame,
                            out_dir: Path) -> None:
    """THE central result: early coverage against final return, per instance.

    Panels 1-2 are the scatter with ONE REGRESSION LINE PER INSTANCE. Never one
    line through everything: coverage and return both fall with difficulty, so a
    pooled line measures "hard mazes are hard" twice. See CLAUDE.md section 9.

    Panel 3 shows each instance's own correlation with its bootstrap interval,
    which is what separates a solid per-maze result from a coincidence.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8),
                             gridspec_kw={"width_ratios": [1, 1, 1.15],
                                          "wspace": 0.42})
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
            # Thin and pale on purpose: there are 13 of these with very different
            # slopes, and at any heavier weight they read as scribble over the
            # scatter rather than as the fits the correlation is computed from.
            ax.plot(xs, slope * xs + intercept, color="#555555", alpha=0.12, lw=0.8)

        per_instance[column] = within_instance_correlation(df, column)
        # Short on purpose. "(first 20% of training)" is in the caption: at
        # this panel width the long form ran into the neighbouring panel.
        ax.set_xlabel("Early-coverage AUC")
        # Pinned, not autoscaled: the fits extrapolate past the data, and letting
        # them set the limits gave the two panels different y ranges for the same
        # quantity -- one reaching to -0.2, which no return can take.
        ax.set_ylim(-0.03, 1.03)
        # The rho values are stated in the report text and drawn in panel 3.
        # As a second title line they overlapped the neighbouring title.
        ax.set_title(panel, fontsize=10)
    axes[0].set_ylabel("Final return (0-1)")

    order = {env_id: i for i, env_id in enumerate(ordered_instances(df))}
    forest = (per_instance["early_auc_raw"]
              .sort_values("env_id", key=lambda s: s.map(order))
              .reset_index(drop=True))
    positions = np.arange(len(forest))
    # Families are told apart by MARKER, not colour: the four strategy colours
    # already mean something else in the two panels to the left, and reusing them
    # here for families inside the same figure would be genuinely misleading.
    # sort=False: `forest` is already in report order, so the families come out
    # Empty, DoorKey, MultiRoom -- the order of the panels in fig1, fig2 and fig6
    # and of the legend here. Sorting would relabel them alphabetically.
    for marker, (family, group) in zip("os^", forest.groupby("family", sort=False)):
        # .to_numpy() throughout: a family with one instance would otherwise hand
        # matplotlib one-element Series, which it deprecates and will later reject.
        rho = group["rho"].to_numpy()
        axes[2].errorbar(
            rho, group.index.to_numpy(),
            xerr=[rho - group["rho_ci_low"].to_numpy(),
                  group["rho_ci_high"].to_numpy() - rho],
            fmt=marker, ms=4, capsize=2, lw=1.2, color="#333333",
            markerfacecolor="white", label=family)
    axes[2].axvline(0, color="black", lw=0.8, ls=":")
    axes[2].set_yticks(positions)
    axes[2].set_yticklabels(forest["env_id"], fontsize=7)
    axes[2].set_xlim(-1.05, 1.05)
    # Half a row of air top and bottom, or the first and last instance labels sit
    # on the axis line and the descenders are clipped.
    axes[2].set_ylim(len(forest) - 0.4, -0.6)
    axes[2].set_xlabel("Spearman $\\rho$ (raw coverage)")
    axes[2].set_title("Per instance", fontsize=10)
    # One legend, centred below the figure with real air above it. Two earlier
    # attempts failed for different reasons: inside the axes it covered the only
    # populated corner of the scatter, and flush against the bottom edge it read
    # as stray text rather than as a legend.
    #
    # There is deliberately NO family legend. Panel 3 labels every row
    # "Empty-5", "DoorKey-6", "MultiRoom-N2" already, so the marker shape repeats
    # the tick label and a key for it is clutter carrying no information.
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=10,
               columnspacing=2.2, handletextpad=0.5)
    _save(fig, out_dir, "fig4_coverage_vs_return")


def fig5_iqm(runs: list[RunResult], df: pd.DataFrame,
             out_dir: Path) -> None:
    """Which strategy wins, two ways: one robust number, and the whole shape.

    An IQM hides the distribution -- two strategies with the same IQM can differ
    completely in how often they solve a maze at all. The performance profile on
    the right shows that (spec 7.2), and where two profiles cross is usually the
    interesting part.
    """
    env_ids = ordered_instances(df)
    # Once per instance, not once per (instance, strategy): each call already
    # returns every strategy and runs its own bootstrap.
    by_env = {env_id: iqm_by_strategy(df, env_id) for env_id in env_ids}

    fig, axes = plt.subplots(1, 3, figsize=(max(8.6, len(env_ids) * 0.66), 2.8),
                             gridspec_kw={"width_ratios": [0.85, 2.2, 1],
                                          "wspace": 0.34})

    # Panel 1 is the one cross-instance number the report quotes in its abstract,
    # and it was missing from this figure entirely. Horizontal, because four
    # strategy names do not fit under vertical bars at this panel width.
    aggregate = rliable_aggregate(df)
    order = [strategy for strategy in COLORS if strategy in aggregate]
    rows = np.arange(len(order))
    point = np.array([aggregate[s]["iqm"] for s in order])
    axes[0].barh(rows, point, height=0.5,
                 color=[COLORS[s] for s in order])
    axes[0].errorbar(
        point, rows,
        xerr=np.array([point - [aggregate[s]["ci_low"] for s in order],
                       [aggregate[s]["ci_high"] for s in order] - point]),
        fmt="none", ecolor="#333333", capsize=2, lw=1.0)
    axes[0].set_yticks(rows)
    axes[0].set_yticklabels([LABELS[s] for s in order])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("IQM final return (0-1)")
    axes[0].set_title("All instances", fontsize=10)
    axes[0].grid(axis="x", color="#DDDDDD", lw=0.6)
    axes[0].grid(axis="y", visible=False)

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
        axes[1].bar(xs, ys, width, color=COLORS[strategy], label=LABELS[strategy])
        # A darkened strategy colour, not one neutral grey: on the instances where
        # the IQM is 0 the bar has no height, so the interval is the only thing
        # drawn, and a lone grey whisker reads as a plotting fault rather than as
        # "this strategy, zero, with this much uncertainty above it".
        axes[1].errorbar(xs, ys, yerr=np.array(errs).T, fmt="none", capsize=2, lw=0.9,
                         ecolor=tuple(0.55 * c for c in to_rgb(COLORS[strategy])))
    axes[1].set_xticks(range(len(env_ids)))
    axes[1].set_xticklabels(env_ids, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("IQM final return (0-1)")
    axes[1].set_title("Per instance", fontsize=10)

    profile = performance_profile(df)
    for strategy in COLORS:
        if strategy not in profile["profiles"]:
            continue
        axes[2].plot(profile["taus"], profile["profiles"][strategy],
                     color=COLORS[strategy], label=LABELS[strategy], lw=1.6)
        axes[2].fill_between(profile["taus"], profile["ci_low"][strategy],
                             profile["ci_high"][strategy],
                             color=COLORS[strategy], alpha=0.18, lw=0)
    axes[2].set_xlabel("Return threshold $\\tau$ (0-1)")
    axes[2].set_ylabel("Fraction above $\\tau$")
    axes[2].set_title("Performance profile", fontsize=10)
    axes[2].margins(x=0)
    # Above the figure, not inside a panel: all three panels encode the same four
    # strategies, so one shared legend belongs to the figure rather than to the
    # profile. It goes on top because the middle panel's rotated instance names
    # already occupy everything below.
    fig.legend(*axes[2].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.5, 1.04), ncol=4, frameon=False, fontsize=9,
               columnspacing=1.8, handletextpad=0.5)
    _save(fig, out_dir, "fig5_iqm")


def fig6_rank_stability(runs: list[RunResult], df: pd.DataFrame,
                        out_dir: Path) -> None:
    """Does the strategy that wins on easy mazes still win on hard ones? (H3)

    One panel per family, like fig1 and fig2, and for the same reason: difficulty
    is grid size for Empty/DoorKey but room count for MultiRoom, so putting all
    three on one x axis would place MultiRoom-N6 beside DoorKey-6 as though the
    two numbers meant the same thing.
    """
    stability = rank_stability(df)
    families = _present_families(df)
    fig, axes = _family_axes(families, height=2.3)
    for ax, family in zip(axes, families):
        group = stability[stability["family"] == family].sort_values("difficulty")
        # .to_numpy(): a family with a single instance hands matplotlib a
        # one-element Series, which it deprecates and will later reject outright.
        note = _degenerate_note(group)
        if note:
            # Boxed, because the zero line runs straight through the middle of
            # the panel and would otherwise strike the text through.
            ax.text(0.5, 0.5, note, transform=ax.transAxes, ha="center",
                    va="center", fontsize=9, color="#777777",
                    bbox=dict(facecolor="white", edgecolor="none", pad=3))
            span = group["difficulty"]
            ax.set_xlim(span.min() - 1, span.max() + 1)
        else:
            # Instances where nothing solved the maze have no ranking to compare,
            # so tau is NaN and the point is simply absent. Left unmarked that
            # reads as missing data; shaded, it reads as the result it is.
            missing = group[group["tau"].isna()]["difficulty"]
            if not missing.empty:
                ax.axvspan(missing.min() - 0.45, missing.max() + 0.45,
                           color="#F2F2F2", lw=0, zorder=0)
                ax.text(missing.mean(), -0.62,
                        "no ranking:\nnothing solved reliably", ha="center",
                        va="center", fontsize=8, color="#888888")
            ax.plot(group["difficulty"].to_numpy(), group["tau"].to_numpy(),
                    marker="o", color="#333333", lw=1.6, ms=5)
        ax.set_xticks(group["difficulty"].tolist())
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel(DIFFICULTY_LABEL[family])
        ax.set_title(family, fontsize=10)
    axes[0].set_ylabel("Kendall's $\\tau$ vs easiest\ninstance of the family (-1 to 1)")
    _save(fig, out_dir, "fig6_rank_stability")


def fig7_visitation_heatmaps(runs: list[RunResult], df: pd.DataFrame, out_dir: Path,
                             env_id: str | None = None, seed: int = 0) -> None:
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
    fig, axes = plt.subplots(1, len(COLORS), figsize=(7.5, 2.2))
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
    """Render every report figure from a results tree.

    Everything that can reject the data does so BEFORE the first figure is
    written. fig5 needs a complete (seeds x instances) matrix and used to find a
    hole only once it got there, leaving four fresh figures on disk beside three
    stale ones from an earlier render -- so the report would have shown two
    different datasets side by side without saying so.
    """
    runs = load_all(Path(results_root))
    if not runs:
        raise ValueError(f"no runs found under {results_root}")
    df = build_analysis_table(runs)
    # Pre-flight for fig5. Private, but same package, and running it here costs
    # one pivot per strategy against the 2000 bootstrap resamples fig5 would do.
    _score_matrices(df)
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
