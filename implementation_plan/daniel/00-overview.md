# Workstream C — Logging, metrics & analysis (Daniel)

> **For Claude sessions:** Read `CLAUDE.md` (especially §8 and §9), then
> `docs/specs/2026-08-17-exploration-comparison-design.md` §6 and §7, then this
> file, then the numbered task file you are on. Work one task at a time.

**Goal:** Record where the agent went, turn that into coverage numbers, test
whether early coverage predicts final performance, and produce every figure in
the report.

**Architecture:** Training writes raw visit counts and nothing else. Every metric
is derived afterwards from those saved arrays. That separation is deliberate: you
can redefine coverage on the last day without anyone re-running a single
experiment.

**Tech Stack:** Python 3.11, numpy, pandas, scipy, rliable, matplotlib, pytest.

**Spec:** `docs/specs/2026-08-17-exploration-comparison-design.md`

---

## Before your first task: set up your environment

The conda environment is not in git. Cloning the repo gives you the code but no
Python to run it with. Do this once, on your machine:

```bash
conda create -n rl python=3.11 -y
```

```bash
conda activate rl && pip install -r requirements.txt && pip install -e .
```

Verify:

```bash
pytest -q
```

Expect **8 passed** (Samuel's scaffold). If you get
`ModuleNotFoundError: No module named 'rlx'`, you skipped `pip install -e .` —
nothing is broken on Samuel's side. Run `conda activate rl` in every new terminal.

---

## Tasks in order

| # | File | Deliverable | Depends on |
|---|---|---|---|
| 1 | `01-visitation-logging.md` | `logging.py` — **Samuel's training loop imports this** | Samuel task 1 |
| 2 | `02-aggregation.md` | `analysis/aggregate.py` + a synthetic results generator | task 1, Samuel task 1 (`difficulty_index`) |
| 3 | `03-coverage-metrics.md` | `analysis/coverage.py` — raw, task-relevant, early AUC | Samuel task 3 |
| 4 | `04-statistics.md` | `analysis/stats.py` — rliable, the central test, rank stability | task 3 |
| 5 | `05-figures.md` | `analysis/figures.py` — all 7 report figures | task 4 |
| 6 | `06-report-scaffold.md` | report outline with numbers filled in automatically | task 5 |

**Task 1 is on the critical path** — Samuel's training loop imports `RunLogger`.
Do it first and merge it early. After that, task 2 generates fake result
directories and you are independent of everyone for the rest of the week.

## Your two contracts

**Consumed from Samuel** (his task 1 — available immediately):

```python
ENV_IDS: tuple[str, ...]
difficulty_index(env_id: str) -> int
```

**Consumed from Samuel** (his task 3 — needed from your task 3 onward):

```python
grid_info(env_id: str, layout_seed: int) -> GridInfo
    # fields: width, height, walls (bool (W,H)), start, goal, key|None, door|None
reachable_mask(info: GridInfo) -> np.ndarray        # bool (W, H)
bfs_distances(info: GridInfo, source) -> np.ndarray # int (W, H), -1 = unreachable
```

**Produced for Samuel** (your task 1) — he calls exactly these six:

```python
RunLogger(cfg: RunConfig, width: int, height: int)
.record_visit(x: int, y: int, direction: int) -> None
.log_step(step: int, **scalars: float) -> None
.snapshot(step: int) -> None
.distinct_states() -> int
.finalize(meta: dict) -> None
```

Changing that signature breaks his training loop. Tell him first.

## The two things you must not get wrong

**1. Privileged information (`CLAUDE.md` §8).** You log and analyse the true
`(x, y, direction)`. The agent never sees it. That is fine — it is measurement,
not learning — but the report must say so explicitly, and no analysis output may
ever be fed back into training.

**2. The statistical trap (`CLAUDE.md` §9).** Both coverage and return fall as
mazes get harder. Correlating them across all 260 runs pooled gives a big
positive number that means nothing — it measures "hard mazes are hard" twice.

**The correlation must be computed within each maze instance separately**, where
difficulty is constant, and only then aggregated. Task 4 has a regression test
that fails if anyone ever pools them. Do not delete that test.

## Definition of done

```bash
python -m rlx.analysis.figures --results results --out report/figures
```

produces all 7 figures, and `report/results.md` contains the real numbers.

## What you must not do

- Do not compute coverage metrics during training. Only raw counts get logged.
- Do not pool runs from different maze instances into one correlation.
- Do not drop seeds that look like outliers. IQM handles them; deleting them is
  fabrication.
