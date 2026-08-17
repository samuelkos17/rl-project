# Task 5 — The seven report figures

**Files:**
- Create: `src/rlx/analysis/figures.py`
- Test: `tests/test_figures.py`

**Interfaces:**
- Consumes: everything from tasks 2–4.
- Produces: `python -m rlx.analysis.figures --results <dir> --out <dir>` writing
  7 PDFs and 7 PNGs.

Develop against `results_synthetic/` so every figure is finished and reviewed
before real data lands.

---

## The seven figures

| # | Filename | Shows | Priority |
|---|---|---|---|
| 1 | `fig1_learning_curves` | return vs steps, 4 strategies, CI bands, one panel per family | high |
| 2 | `fig2_difficulty_curve` | final return vs difficulty, 4 lines per family | **headline** |
| 3 | `fig3_coverage_curves` | raw and task-relevant coverage vs steps | high |
| 4 | `fig4_coverage_vs_return` | early-coverage AUC vs final return, per-instance regression | **the central result** |
| 5 | `fig5_iqm` | rliable IQM with CIs per strategy | high |
| 6 | `fig6_rank_stability` | Kendall's tau vs difficulty | medium |
| 7 | `fig7_visitation_heatmaps` | where each strategy actually went, one instance | **poster centrepiece** |

Figures 4 and 7 are what the poster is built around. If you run short of time,
figures 1, 2, 4, 5, 7 are the ones the report cannot do without.

## Plotting conventions

Fix these once, at the top of the module, so every figure looks like it belongs
to the same report:

- One colour per strategy, identical everywhere. `epsilon_greedy` grey (it is the
  baseline), the other three distinct.
- Every uncertainty band is a bootstrap CI, never a standard deviation.
- Axis labels always carry units or a range. "Final return (0–1)", not "return".
- Save both `.pdf` (for the report, vector) and `.png` at 200 dpi (for the
  poster and for looking at quickly).
- No titles inside the figure — captions belong in the report text.

---

- [ ] **Step 1: Write the failing tests**

Figure tests check that files are produced and non-trivial, not that pixels match.
Judging whether a figure is *good* is your job, by looking at it.

```python
import subprocess
import sys

import pytest

from rlx.analysis.figures import FIGURE_NAMES, make_all_figures


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    subprocess.run([sys.executable, "scripts/make_synthetic_results.py", "--out", str(out)],
                   check=True)
    return out


def test_all_seven_figures_are_produced(synthetic, tmp_path):
    make_all_figures(synthetic, tmp_path)
    for name in FIGURE_NAMES:
        assert (tmp_path / f"{name}.pdf").exists(), name
        assert (tmp_path / f"{name}.png").exists(), name


def test_figures_are_not_blank(synthetic, tmp_path):
    """A near-empty PDF means the plot silently drew nothing."""
    make_all_figures(synthetic, tmp_path)
    for name in FIGURE_NAMES:
        assert (tmp_path / f"{name}.pdf").stat().st_size > 5_000, name


def test_there_are_exactly_seven(synthetic):
    assert len(FIGURE_NAMES) == 7
```

- [ ] **Step 2: Run them and watch them fail**

```bash
pytest tests/test_figures.py -v
```

Expected: `ModuleNotFoundError: No module named 'rlx.analysis.figures'`.

Do not skip this because the failure is obvious. Seeing the test fail for the
reason you expect is what proves the test is actually running and actually
checking the thing you think it checks.

- [ ] **Step 3: Write `src/rlx/analysis/figures.py`**

Build it incrementally: write one figure, run it, **look at the output**, then
write the next. Do not write all seven and render them at the end.

```python
"""Every figure in the report. Run:

    python -m rlx.analysis.figures --results results --out report/figures
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rlx.analysis.aggregate import load_all
from rlx.analysis.coverage import raw_coverage, task_relevant_coverage
from rlx.analysis.stats import (
    aggregate_correlation, build_analysis_table, iqm_by_strategy, rank_stability,
    within_instance_correlation,
)
from rlx.envs import grid_info

FIGURE_NAMES = (
    "fig1_learning_curves", "fig2_difficulty_curve", "fig3_coverage_curves",
    "fig4_coverage_vs_return", "fig5_iqm", "fig6_rank_stability",
    "fig7_visitation_heatmaps",
)

COLORS = {
    "epsilon_greedy": "#888888",   # grey: it is the baseline
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


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _bootstrap_band(values: np.ndarray, n: int = 1_000, seed: int = 0):
    """Mean and 95% bootstrap CI across runs, per column."""
    rng = np.random.default_rng(seed)
    draws = np.stack([values[rng.integers(0, len(values), len(values))].mean(axis=0)
                      for _ in range(n)])
    return values.mean(axis=0), np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def fig1_learning_curves(runs, df, out_dir):
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(15, 4), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        for strategy in COLORS:
            curves = [r.metrics["eval_return_mean"].to_numpy() for r in runs
                      if r.strategy == strategy and r.env_id.startswith(family)]
            if not curves:
                continue
            n = min(len(c) for c in curves)
            stacked = np.stack([c[:n] for c in curves])
            steps = runs[0].metrics["step"].to_numpy()[:n]
            mean, lo, hi = _bootstrap_band(stacked)
            ax.plot(steps, mean, color=COLORS[strategy], label=LABELS[strategy])
            ax.fill_between(steps, lo, hi, color=COLORS[strategy], alpha=0.2)
        ax.set_xlabel("Environment steps")
        ax.set_title(family)
    axes[0].set_ylabel("Evaluation return (0-1)")
    axes[-1].legend(frameon=False)
    _save(fig, out_dir, "fig1_learning_curves")


def fig2_difficulty_curve(runs, df, out_dir):
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(15, 4), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        sub = df[df["family"] == family]
        for strategy in COLORS:
            g = (sub[sub["strategy"] == strategy]
                 .groupby("difficulty")["final_return"].agg(["mean", "sem"]))
            if g.empty:
                continue
            ax.errorbar(g.index, g["mean"], yerr=g["sem"], marker="o", capsize=3,
                        color=COLORS[strategy], label=LABELS[strategy])
        ax.set_xlabel("Grid size" if family != "MultiRoom" else "Number of rooms")
        ax.set_title(family)
    axes[0].set_ylabel("Final return (0-1)")
    axes[-1].legend(frameon=False)
    _save(fig, out_dir, "fig2_difficulty_curve")


def fig3_coverage_curves(runs, df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (fn, title) in zip(axes, ((raw_coverage, "Raw coverage"),
                                      (task_relevant_coverage, "Task-relevant coverage"))):
        for strategy in COLORS:
            curves = []
            for r in runs:
                if r.strategy != strategy:
                    continue
                info = grid_info(r.env_id, layout_seed=r.seed)
                curves.append(fn(r.counts, info))
            if not curves:
                continue
            n = min(len(c) for c in curves)
            mean, lo, hi = _bootstrap_band(np.stack([c[:n] for c in curves]))
            steps = runs[0].steps[:n]
            ax.plot(steps, mean, color=COLORS[strategy], label=LABELS[strategy])
            ax.fill_between(steps, lo, hi, color=COLORS[strategy], alpha=0.2)
        ax.set_xlabel("Environment steps")
        ax.set_title(title)
    axes[0].set_ylabel("Fraction of states visited (0-1)")
    axes[-1].legend(frameon=False)
    _save(fig, out_dir, "fig3_coverage_curves")


def fig4_coverage_vs_return(runs, df, out_dir):
    """THE central result: early coverage against final return, per instance."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, col, title in zip(axes, ("early_auc_raw", "early_auc_task"),
                              ("Raw coverage", "Task-relevant coverage")):
        for strategy in COLORS:
            sub = df[df["strategy"] == strategy]
            ax.scatter(sub[col], sub["final_return"], s=22, alpha=0.75,
                       color=COLORS[strategy], label=LABELS[strategy])

        # One regression line per instance -- never one line through everything.
        for env_id, group in df.groupby("env_id"):
            if group[col].nunique() < 2:
                continue
            slope, intercept = np.polyfit(group[col], group["final_return"], 1)
            xs = np.linspace(group[col].min(), group[col].max(), 10)
            ax.plot(xs, slope * xs + intercept, color="black", alpha=0.15, lw=1)

        agg = aggregate_correlation(within_instance_correlation(df, col))
        ax.set_xlabel("Early-coverage AUC (first 20% of training)")
        ax.set_title(f"{title}\nwithin-instance $\\rho$ = {agg['mean_rho']:+.2f} "
                     f"[{agg['ci_low']:+.2f}, {agg['ci_high']:+.2f}]")
    axes[0].set_ylabel("Final return (0-1)")
    axes[-1].legend(frameon=False, fontsize=8)
    _save(fig, out_dir, "fig4_coverage_vs_return")


def fig5_iqm(runs, df, out_dir):
    env_ids = sorted(df["env_id"].unique())
    # Computed once per instance, not once per (instance, strategy): each call
    # already returns every strategy and runs 1000 bootstrap resamples.
    by_env = {env_id: iqm_by_strategy(df, env_id) for env_id in env_ids}

    fig, ax = plt.subplots(figsize=(max(8, len(env_ids) * 1.1), 4.5))
    width = 0.2
    for i, strategy in enumerate(COLORS):
        xs, ys, errs = [], [], []
        for j, env_id in enumerate(env_ids):
            res = by_env[env_id].get(strategy)
            if res is None:
                continue
            xs.append(j + (i - 1.5) * width)
            ys.append(res["iqm"])
            errs.append([res["iqm"] - res["ci_low"], res["ci_high"] - res["iqm"]])
        ax.bar(xs, ys, width, color=COLORS[strategy], label=LABELS[strategy])
        ax.errorbar(xs, ys, yerr=np.array(errs).T, fmt="none", ecolor="black",
                    capsize=2, lw=0.8)
    ax.set_xticks(range(len(env_ids)))
    ax.set_xticklabels(env_ids, rotation=45, ha="right")
    ax.set_ylabel("IQM final return (0-1)")
    ax.legend(frameon=False)
    _save(fig, out_dir, "fig5_iqm")


def fig6_rank_stability(runs, df, out_dir):
    stability = rank_stability(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    for family, group in stability.groupby("family"):
        group = group.sort_values("difficulty")
        ax.plot(group["difficulty"], group["tau"], marker="o", label=family)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Difficulty (grid size / room count)")
    ax.set_ylabel("Kendall's $\\tau$ vs easiest instance")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(frameon=False)
    _save(fig, out_dir, "fig6_rank_stability")


def fig7_visitation_heatmaps(runs, df, out_dir, env_id: str | None = None, seed: int = 0):
    """Where each strategy actually went. The poster figure."""
    env_id = env_id or ("DoorKey-8" if any(r.env_id == "DoorKey-8" for r in runs)
                        else runs[0].env_id)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, strategy in zip(axes, COLORS):
        match = [r for r in runs if r.env_id == env_id
                 and r.strategy == strategy and r.seed == seed]
        if not match:
            ax.axis("off")
            continue
        counts = match[0].counts[-1].sum(axis=2)          # sum over the 4 directions
        ax.imshow(np.log1p(counts).T, cmap="viridis", origin="upper")
        ax.set_title(LABELS[strategy])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{env_id}, seed {seed} -- log visit counts", y=1.02)
    _save(fig, out_dir, "fig7_visitation_heatmaps")


def make_all_figures(results_root: Path, out_dir: Path) -> None:
    runs = load_all(Path(results_root))
    if not runs:
        raise SystemExit(f"no runs found under {results_root}")
    df = build_analysis_table(runs)
    for fn in (fig1_learning_curves, fig2_difficulty_curve, fig3_coverage_curves,
               fig4_coverage_vs_return, fig5_iqm, fig6_rank_stability,
               fig7_visitation_heatmaps):
        fn(runs, df, Path(out_dir))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("report/figures"))
    args = p.parse_args()
    make_all_figures(args.results, args.out)
    print(f"wrote {len(FIGURE_NAMES)} figures to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate them from synthetic data**

```bash
python -m rlx.analysis.figures --results results_synthetic --out scratch/figs
```

(`scratch/`, not `/tmp/` — we are on Windows and `/tmp` does not exist outside
Git Bash. Add `scratch/` to `.gitignore`.)

- [ ] **Step 5: Actually look at every one of them**

Open all seven PNGs. Do not skip this — the tests only prove the files are not
empty, not that they are readable. For each, ask:

- Can you read the axis labels at report size?
- Are the four strategies distinguishable, including in greyscale?
- Do the confidence bands look plausible, or suspiciously narrow?
- Is anything clipped, overlapping, or running off the edge?

Fix what is wrong and regenerate. Repeat until you would be happy putting each
one in front of the professor. This is Rule 2 — loop until the output is good.

- [ ] **Step 6: Run the tests**

```bash
pytest tests/test_figures.py -v
```

Expected: 3 passed.

- [ ] **Step 7: Log the change**

Append to `docs/decision_log.md`, in plain language:

```markdown
## 2026-08-21 — The seven figures, and which two actually matter

**Status:** Active

**What changed:** Wrote the code that produces every figure in the report, so all
seven regenerate from one command whenever the results change.

**The seven:**
1. Learning curves — score over time, all four strategies, one panel per maze family.
2. Score against difficulty — the headline plot the professor asked for.
3. Coverage over time — both our coverage measures.
4. Early coverage against final score — **the central result of the project**.
5. IQM with confidence intervals — which strategy wins, with honest error bars.
6. Rank stability — does the winner on easy mazes still win on hard ones?
7. Visitation heatmaps — a picture of where each strategy actually went.

**If we had to cut to just two,** it would be 4 and 7. Number 4 is the actual
claim we are making. Number 7 is the one that makes someone walking past the
poster stop and look, because you can *see* the difference between strategies
instead of reading it off an axis.

**One rule we set:** every shaded band on every plot is a bootstrap confidence
interval, never a standard deviation. Standard deviations look tighter and would
make our results look more certain than they are.

**What it means for the results:** Nothing changes the numbers. But regenerating
from one command means the report can never end up showing a figure from an older
version of the data than the tables next to it.
```

- [ ] **Step 8: Commit**

```bash
git add src/rlx/analysis/figures.py tests/test_figures.py docs/decision_log.md
git commit -m "feat: all seven report figures"
```
