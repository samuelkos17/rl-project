# Implementation plan — how the three of us work in parallel

> **For Claude sessions:** Read `CLAUDE.md` first, then
> `docs/specs/2026-08-17-exploration-comparison-design.md`, then the overview
> file in your person's directory. Use `superpowers:executing-plans` or
> `superpowers:subagent-driven-development` to work through task files.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a controlled comparison of four DQN exploration strategies across
13 MiniGrid instances, and test whether early state coverage predicts final
performance.

**Deadline:** 2026-08-23.

---

## Who does what

| Directory | Person | Workstream | Owns |
|---|---|---|---|
| `implementation_plan/samuel/` | Samuel Kostiuk | **A — Core & Infrastructure** | env factory, network, buffer, Double DQN, training loop, sweep runner |
| `implementation_plan/max/` | Max Bullach | **B — Exploration strategies** | the four `Explorer` implementations |
| `implementation_plan/daniel/` | Daniel Gleim | **C — Logging, metrics & analysis** | visitation logging, coverage metrics, statistics, figures, report scaffold |

Work only in your own files. If you need a change in someone else's file, say so
in the PR description — do not edit it silently.

---

## First-time setup — every one of us, on every machine

**Do this before your first task.** The conda environment is not in git; cloning
the repo gives you the code but no Python to run it with.

```bash
conda create -n rl python=3.11 -y
```

```bash
conda activate rl && pip install -r requirements.txt && pip install -e .
```

Check it worked:

```bash
pytest -q
```

You should see **8 passed**. If instead you see
`ModuleNotFoundError: No module named 'rlx'`, you skipped `pip install -e .` —
that is what makes the `rlx` package importable from anywhere. Nothing is broken
on anyone else's side.

Remember `conda activate rl` in every new terminal.

---

## The one blocking dependency

**Samuel's Task 1 (`samuel/01-scaffold-and-contracts.md`) must be merged to
`main` before Max and Daniel start.** It creates the package skeleton and the
three frozen contracts everyone codes against. It is a mechanical copy-paste task
and should take under 30 minutes.

After that, **nothing blocks anything**:

- Max develops against fake Q-value arrays. He never needs a working DQN.
- Daniel develops against synthetic result directories. He never needs a real run.

Both have a task that generates the fake data they need. If Samuel slips a day,
neither of them stops.

---

## Frozen contracts

Created in Samuel's Task 1. **Changing any of these requires telling the other
two people**, because it silently breaks work in progress:

1. `src/rlx/exploration/base.py` — the `Explorer` interface.
2. `src/rlx/config.py` — the `RunConfig` field names.
3. The results directory format (`CLAUDE.md` §5).

---

## Schedule

| Date | Samuel (A) | Max (B) | Daniel (C) |
|---|---|---|---|
| **17.08** | Tasks 1–2: scaffold, contracts, API verification, benchmark | — | — |
| **18.08** | Tasks 3–4: env factory, network, buffer | Tasks 1–2: epsilon-greedy, Boltzmann | Tasks 1–2: visitation logging, aggregation |
| **19.08** | Tasks 5–6: agent, training loop, sweep runner | Tasks 3–4: count-based, NoisyNets | Tasks 3–4: coverage metrics, statistics |
| **20.08** | **Integration day** — full smoke test, pilot sweep, then launch the full sweep across all 3 PCs | | |
| **21.08** | monitor sweep, re-run failures | strategy write-ups for the report | Tasks 5–6: figures, report scaffold |
| **22.08** | analysis complete, all figures final, write the report | | |
| **23.08** | buffer, poster, submit | | |

---

## Daily routine (all three of us)

Every morning:

```bash
git checkout main && git pull && git checkout <your-branch> && git rebase main
```

Every evening: open a PR to `main`, get it merged. Do not sit on work
overnight — integration is continuous, not a cliff on the 20th.

Branch names: `core/<topic>` (Samuel), `exploration/<topic>` (Max),
`analysis/<topic>` (Daniel).

---

## Rules that apply to every task

From `CLAUDE.md` §2. Repeated here because they are easy to skip:

1. **If you are uncertain, say so.** Never invent an API signature.
2. **One step at a time.** Review your output against the step. If it is bad,
   redo the step. Loop until it is good.
3. **Test your code.** Run the tests. Read the output. Report what actually
   happened, including failures.
4. **`.md` files are read by LLMs.** Be exact. No `TBD`.
5. **Log every real change in `docs/decision_log.md`**, including discarded ones,
   in plain language for the team.
6. **Less code is better.** No abstraction for one implementation.
7. **Claude never runs `git add` or `git commit`.** The "Step N: Commit" steps in
   these task files, and every `git add` / `git commit` command inside them, are
   **for us to run**. When Claude reaches one, it stops and reports what changed,
   what it tested, and a suggested commit message. We review and commit. See
   `CLAUDE.md` Rule 7.

---

## Global constraints

Copied verbatim from the spec. Every task's requirements implicitly include
these:

- Python **3.11** (not 3.13). Conda env named `rl`.
- Dependencies: `torch`, `gymnasium`, `minigrid`, `numpy`, `pandas`, `scipy`,
  `matplotlib`, `rliable`, `pyyaml`, `pytest`. Nothing else without a decision-log
  entry.
- **Double DQN**, fixed for every strategy and every environment.
- Hyperparameters are **identical across all strategies**. Never tune per
  strategy.
- Step budget is **identical across all environment instances**.
- Evaluation is **greedy, extrinsic reward only**, intrinsic bonus off, NoisyNets
  noise off.
- The agent sees **only** the 7x7x3 observation. `(x, y, dir)` is used **only for
  analysis, never by any agent**.
- Layout is **pinned per run**: every `reset()` uses the run's seed, so one run
  sees exactly one maze.
- All paths via `pathlib`. The team is on Windows; hard-coded `/` breaks.
- Type-hint public functions only. No `print()` in library code.
