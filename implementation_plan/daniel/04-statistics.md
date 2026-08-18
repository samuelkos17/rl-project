# Task 4 — Statistics: the central test, rliable, rank stability

This task contains the result the whole project is built to produce. It also
contains the one mistake that would silently invalidate it.

**Files:**
- Create: `src/rlx/analysis/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: `to_dataframe` (task 2), coverage functions (task 3).
- Produces:
  - `build_analysis_table(runs) -> pd.DataFrame` — one row per run, with coverage columns
  - `within_instance_correlation(df, coverage_col) -> pd.DataFrame` — one row per env instance
  - `aggregate_correlation(per_instance) -> dict`
  - `iqm_by_strategy(df, env_id) -> dict`
  - `rank_stability(df) -> pd.DataFrame`
  - `rliable_aggregate(df) -> dict` (step 6)
  - `probability_of_improvement(df, a, b) -> float` (step 6)

---

## READ THIS BEFORE WRITING ANY CODE

Both coverage and final return **fall as mazes get harder**. So if you correlate
them across all 260 runs pooled together, you get a big positive correlation that
means **absolutely nothing** — it measures "hard mazes are hard" twice and calls
the agreement a discovery.

**The correlation must be computed separately within each maze instance**, across
that instance's 20 runs (4 strategies × 5 seeds), where difficulty is constant
and therefore cannot manufacture the effect. Only then do you combine the
per-instance results.

`test_pooling_and_within_instance_disagree` in this task constructs data where
the pooled correlation is strongly positive and the true within-instance
correlation is negative. **If someone ever refactors this into a pooled
correlation, that test fails. Do not delete it.**

## What the numbers are for

| Function | Answers |
|---|---|
| `within_instance_correlation` | Does exploring more early predict scoring higher later? (**H1**) |
| ...run for both coverage columns | Does *task-relevant* coverage predict better than *raw*? (**H2**) |
| `iqm_by_strategy` | Which strategy wins, with honest error bars? |
| `rank_stability` | Does the winner on easy mazes still win on hard ones? (**H3**) |

---

- [x] **Step 1: Write the failing tests**

Create `tests/test_stats.py`:

```python
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
    # 4 instances x 5 seeds, mirroring the real experiment's shape. Both return
    # and coverage fall as difficulty rises (the between-instance trend), while
    # inside every instance they move in OPPOSITE directions.
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
```

- [x] **Step 2: Run and watch them fail**

```bash
pytest tests/test_stats.py -v
```

- [x] **Step 3: Write `src/rlx/analysis/stats.py`**

```python
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
    """
    valid = per_instance.dropna(subset=["rho"])
    if len(valid) == 0:
        return {"mean_rho": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "n_instances": 0, "trend_with_difficulty": np.nan}

    rho = valid["rho"].to_numpy()
    ci_low, ci_high = _bootstrap_mean_ci(rho, np.random.default_rng(seed))
    trend = (stats.spearmanr(valid["difficulty"], rho).statistic
             if len(valid) >= 3 and valid["difficulty"].nunique() >= 2 else np.nan)
    return {
        "mean_rho": float(rho.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_instances": len(valid),
        "trend_with_difficulty": trend,
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
```

- [x] **Step 4: Run the tests**

```bash
pytest tests/test_stats.py -v
```

Expected: 11 passed. **`test_pooling_and_within_instance_disagree` passing is the
single most important assertion in this workstream.**

- [x] **Step 5: Run the whole pipeline on synthetic data**

The synthetic generator baked in a known effect — count-based explores most,
epsilon-greedy least, and more early coverage means better final return. Your
analysis must recover it. This is how you know the statistics work before real
data exists.

```bash
python -c "
from pathlib import Path
from rlx.analysis.aggregate import load_all
from rlx.analysis.stats import aggregate_correlation, build_analysis_table, rank_stability, within_instance_correlation

runs = load_all(Path('results_synthetic'))
df = build_analysis_table(runs)
print(f'{len(df)} runs\n')

for col in ('early_auc_raw', 'early_auc_task'):
    per = within_instance_correlation(df, col)
    agg = aggregate_correlation(per)
    print(f\"{col:<16} mean_rho={agg['mean_rho']:+.3f}  \"
          f\"CI=[{agg['ci_low']:+.3f},{agg['ci_high']:+.3f}]  \"
          f\"trend={agg['trend_with_difficulty']:+.3f}  H1={agg['confirms_h1']}\")

print()
print(rank_stability(df).to_string(index=False))
"
```

**Expected on synthetic data:** a clearly positive `mean_rho` with a CI excluding
zero. If it comes out near zero or negative, your pipeline is broken — the fake
data has the effect built in by construction. Debug against synthetic data now,
not against real results on the 22nd.

`build_analysis_table` needs `grid_info` from Samuel's task 3. If it is not
merged yet, test the correlation functions with the hand-built frames above and
run this end-to-end check once it lands.

- [ ] **Step 6: Add the rliable aggregate comparison** — PARTIAL, 2026-08-18. `probability_of_improvement` is done and tested (needs no library). `rliable_aggregate` is NOT written: `rliable` does not import (arch 7.2.0 vs pandas 3.0.5). Team deferred the dependency decision; see `docs/decision_log.md`, "The statistics trap we nearly walked into, and one we walked into".

The `iqm_by_strategy` above is a hand-rolled IQM with a bootstrap CI — fine for
per-instance bars, and only ~10 lines. But the proposal commits to `rliable`, and
the professor's feedback explicitly approved that choice, so the aggregate
comparison must genuinely use the library rather than our reimplementation of it.

Append to `src/rlx/analysis/stats.py`:

```python
def rliable_aggregate(df: pd.DataFrame) -> dict:
    """Aggregate IQM with stratified bootstrap CIs, via rliable.

    rliable expects {algorithm: array of shape (n_runs, n_tasks)}, where a task
    is one environment instance. MiniGrid returns are already in [0, 1], so no
    normalisation is needed.
    """
    from rliable import library as rly
    from rliable import metrics

    env_ids = sorted(df["env_id"].unique())
    scores = {}
    for strategy, group in df.groupby("strategy", sort=True):
        pivot = (group.pivot_table(index="seed", columns="env_id",
                                   values="final_return")
                      .reindex(columns=env_ids))
        scores[strategy] = pivot.to_numpy()

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


def probability_of_improvement(df: pd.DataFrame, a: str, b: str) -> float:
    """P(a random run of strategy `a` beats a random run of strategy `b`)."""
    x = df[df["strategy"] == a]["final_return"].to_numpy()
    y = df[df["strategy"] == b]["final_return"].to_numpy()
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    return float((x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean())
```

Add a test to `tests/test_stats.py`:

```python
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


def test_probability_of_improvement_is_one_when_a_always_wins():
    from rlx.analysis.stats import probability_of_improvement

    rows = [["Empty-5", 5, "a", i, 0.9, 0.5] for i in range(5)]
    rows += [["Empty-5", 5, "b", i, 0.1, 0.5] for i in range(5)]
    assert probability_of_improvement(_frame(rows), "a", "b") == 1.0
```

```bash
pytest tests/test_stats.py -v
```

**If `rliable`'s API differs from what is written above, do not guess a second
time.** Read the installed package (`python -c "from rliable import library; help(library.get_interval_estimates)"`),
fix the call to match, and write a `docs/decision_log.md` entry recording the
real signature. This is Rule 1.

- [x] **Step 7: Log the change**

This is the most important entry in the whole log. Append to
`docs/decision_log.md`:

```markdown
## 2026-08-19 — The statistics trap we nearly walked into

**Status:** Active

**What changed:** We compute our main correlation — does exploring more early
predict scoring better later — separately inside each individual maze, and only
then combine the results. Not across all 260 runs mixed together.

**Why this matters so much:** both coverage and final score go *down* as mazes
get harder. So if you throw all 260 runs into one correlation, you get a big
impressive positive number — and it means nothing at all. It is just saying "hard
mazes are hard", measured two different ways, and noticing that the two agree.

We would very likely have reported that number as our headline result. It is the
kind of mistake that looks like a finding.

**How we avoid it:** inside a single maze, every run faces the same difficulty,
so difficulty cannot create a fake correlation. We compute the correlation there,
where it is honest, and then average across mazes.

**The safeguard:** there is a test called
`test_pooling_and_within_instance_disagree` built from made-up data where the
pooled answer is strongly positive and the true answer is strongly negative. If
anyone ever refactors this back into a single pooled correlation, that test
fails. **Do not delete it, even if it looks strange.** It is the only thing
standing between us and reporting a result that is not real.

**What it means for the results:** Our headline correlation will be a smaller,
less exciting number than the pooled one would have been. That smaller number is
the true one.
```

- [ ] **Step 8: Commit**

```bash
git add src/rlx/analysis/stats.py tests/test_stats.py docs/decision_log.md
git commit -m "feat: within-instance correlation, rliable IQM, and rank stability"
```
