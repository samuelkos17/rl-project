# CLAUDE.md

Instructions for every Claude session working in this repository. Read this file
completely before taking any action.

---

## 1. What this project is

A master's-level Reinforcement Learning project exam.

**Research question:** Does more exploration actually help? We take **one** DQN
implementation and swap **only** the exploration strategy, then measure both
*which* strategy wins and *why* it wins.

**Central hypothesis:** Strategies that cover more of the environment early in
training end up performing better later, and this effect grows stronger as the
environment gets harder to explore.

**Team:** Samuel Kostiuk, Max Bullach, Daniel Gleim.
**Hard deadline:** report due **2026-08-23**.

Source documents live in `context/`:
- `context/rl_proposal.pdf` — the original project proposal.
- `context/proposal_response.md` — the professor's feedback. This feedback is
  binding: the person who wrote it grades the report. Every point in it is
  addressed by the design.

The full design lives in
`docs/specs/2026-08-17-exploration-comparison-design.md`. Read it before
implementing anything.

---

## 2. Rules for every session

These are non-negotiable and come directly from the team.

### Rule 1 — Uncertainty must be stated, never papered over
If you are not sure whether an API exists, whether a number is right, or whether
an approach works, **say so explicitly**. Do not invent a plausible-sounding
answer. Do not guess a function signature and present it as fact. Saying "I don't
know, let me verify this by running X" is always the correct move.

Flag uncertainty inline in the response *and*, if it affects the code, add a
`# UNVERIFIED:` comment at the relevant line.

### Rule 2 — Work step by step, and review your own output
Do one step at a time. After finishing a step, stop and review what you produced
against what the step asked for. If the output is wrong, incomplete, or sloppy,
**redo that step**. Repeat the loop until the output is genuinely good. Only then
move to the next step.

Do not batch five steps together and hope. Do not move on from a step you know is
shaky.

### Rule 3 — Test your code
Every module gets tests. A change is not finished until its tests pass and you
have **seen** them pass. Run the tests, read the output, and report the actual
result — including failures. Never claim something works because it looks like it
should.

### Rule 4 — Markdown files are written for LLMs
Everything written into a `.md` file in this repo is read by future Claude
sessions as their primary context. Therefore:
- Be explicit and unambiguous. Prefer exact names, paths, and numbers over prose.
- State facts, not vibes. `buffer_size = 100_000` beats "a reasonably large buffer".
- Use structure (headings, tables, lists) so a session can find one fact fast.
- Never leave `TBD`, `TODO`, or placeholders in a spec or plan file.
- If something is uncertain, mark it explicitly as `UNVERIFIED` rather than
  omitting it or stating it confidently.

### Rule 5 — Document every real change in `docs/`
Every change you actually make gets written down in `docs/`, **including changes
that are later discarded** — in that case record *why* they were discarded, because
that reasoning is the valuable part.

`docs/` is written for **the three humans**, who are not fluent in RL
terminology. So in `docs/`:
- Explain jargon the first time it appears in a document. "Replay buffer (a
  memory of past experiences the agent learns from repeatedly, instead of
  learning once and forgetting)" — that level.
- Keep sentences plain. Prefer concrete over formal.
- Say what changed, why, and what it means for the results.

This is the opposite audience from Rule 4. `docs/` is for humans;
`CLAUDE.md`, `docs/specs/`, and `implementation_plan/` are for LLMs. Both must be
maintained.

See `docs/README.md` for how the docs directory is organised.

### Rule 6 — Less code is better than more code
The smallest thing that correctly does the job wins. Do not add abstraction
layers, configuration options, plugin systems, or "flexibility for later". Do not
write a base class for one implementation. Do not add a feature nobody asked for.

If you are about to write a file longer than ~200 lines, stop and ask whether it
is doing too much.

### Rule 7 — Never run `git add` or `git commit`. Report instead.
**We commit our own work. You never do.**

Do not run `git add`, `git commit`, `git push`, `git merge`, `git rebase`,
`git checkout -b`, or anything else that writes to git history or the staging
area. Read-only git commands (`git status`, `git diff`, `git log`) are fine and
often useful.

When a piece of work is finished, **stop and say so**, and give us:
- what changed, as a list of file paths
- the test command you ran and its actual output
- a suggested commit message we can copy

Then wait. Do not stage anything "to be helpful" — an unexpected staged change is
worse than no help at all, because we might commit something we never reviewed.

**This overrides the plan files.** `implementation_plan/**` contains steps like
"Step N: Commit" with `git add` / `git commit` commands in them. Those commands
are **for us to run, not for you**. When you reach such a step, treat it as
"report that this task is done" and stop there.

---

## 3. Tech stack

| Component | Choice | Notes |
|---|---|---|
| Python | **3.11** | Not 3.13. `rliable` and parts of the MiniGrid stack lag the newest Python. |
| Env manager | conda | `conda create -n rl python=3.11` |
| Deep learning | `torch` | Small CNN; device decided by benchmark (see §7). |
| RL environments | `gymnasium` + `minigrid` | Partial 7x7x3 observations. |
| Numerics | `numpy`, `pandas`, `scipy` | |
| Statistics | `rliable` | Aggregate IQM, stratified bootstrap CIs, performance profiles. |
| Plotting | `matplotlib` | No seaborn. |
| Config | `pyyaml` + dataclasses | |
| Tests | `pytest` | |
| Markdown tables | `tabulate` | Required by `pandas.DataFrame.to_markdown()`. |

Nothing else gets added without a note in `docs/decision_log.md`.

**No experiment-tracking service.** No Weights & Biases, no TensorBoard. Results
are plain files on disk (see §5). This is deliberate: three machines produce
results independently and merge by copying directories.

### Setup

```bash
conda create -n rl python=3.11 -y
conda activate rl
pip install -r requirements.txt
pip install -e .
```

**Every team member does this on their own machine** — the environment is not in
git, so cloning the repo gives you the code but nothing to run it with. Verify
with `pytest -q`. A `ModuleNotFoundError: No module named 'rlx'` means
`pip install -e .` was skipped; it is not a broken commit.

`pip install torch` on Windows yields a **CPU-only** build (`torch==X.Y.Z+cpu`,
`torch.cuda.is_available() == False`). That is fine for workstreams B and C. See
`docs/decision_log.md`, entry "pip installed a CPU-only torch", before running
the task-2 benchmark.

---

## 4. Repository layout and ownership

```
src/rlx/
  config.py            dataclass configs + YAML loading                [A]
  envs.py              env factory, difficulty registry,
                       reachable-state BFS, distance-to-goal field     [A]
  networks.py          QNetwork + NoisyLinear                          [A shell, B fills NoisyLinear]
  buffer.py            replay buffer                                   [A]
  agent.py             Double DQN agent                                [A]
  train.py             training loop, eval protocol, result writer     [A]
  sweep.py             parallel launcher with sharding                 [A]
  logging.py           visitation logger, metrics writer               [C]
  exploration/
    base.py            Explorer interface (FROZEN CONTRACT)            [A wrote, B owns]
    epsilon_greedy.py                                                  [B]
    boltzmann.py                                                       [B]
    count_based.py                                                     [B]
    noisy.py                                                           [B]
  analysis/
    coverage.py        raw + task-relevant coverage, early AUC         [C]
    aggregate.py       results tree -> tidy DataFrame                  [C]
    stats.py           rliable, Spearman, Kendall, bootstrap           [C]
    figures.py         all report figures                              [C]

tests/                 pytest suite, mirrors src layout
configs/               YAML experiment configs
context/               proposal + professor feedback (read-only)
docs/                  human-facing notes, decision log, glossary
docs/specs/            the design spec (LLM-facing)
implementation_plan/   per-person step files (LLM-facing)
report/                report scaffolding and generated figures
results/               experiment output (see §5)
```

**Workstream owners:**
- **A — Core & Infrastructure — Samuel.** Env factory, DQN, training loop, sweep runner.
- **B — Exploration strategies — Max.** The four strategy modules and their tests.
- **C — Logging, metrics & analysis — Daniel.** Visitation logging, coverage metrics, statistics, figures.

Work in your own area. If you need a change in someone else's file, say so in the
PR description rather than silently editing it.

---

## 5. Results format (FROZEN CONTRACT)

Every run writes exactly this, and nothing else:

```
results/<env_id>/<strategy>/seed<k>/
    config.json      the exact resolved config that produced this run
    metrics.csv      one row per logged step
    visitation.npz   arrays: steps (T,), counts (T, W, H, 4)
    meta.json        git sha, hostname, device, wall_time_s, completed (bool)
```

`metrics.csv` columns (exact names):

```
step, eval_return_mean, eval_return_std, train_return_mean,
episodes, distinct_states, loss, <strategy stats>
```

`<strategy stats>` is whatever that run's `Explorer.stats()` returns, so the
columns differ by strategy and that is expected:

| strategy | extra columns |
|---|---|
| `epsilon_greedy` | `epsilon` |
| `boltzmann` | `temperature` |
| `count_based` | `epsilon`, `mean_bonus`, `distinct_keys` |
| `noisy` | none |

Analysis must therefore never assume a strategy column exists — read with
`pandas` and expect `NaN` where a strategy did not emit it.

`distinct_states` is a running count of distinct `(x,y,dir)` seen so far. It
exists for **progress monitoring only** — it is not a coverage metric (it has no
denominator) and analysis must not use it. All coverage metrics are derived from
`visitation.npz` after training.

Rules:
- A run directory is written **atomically at completion**: write to
  `seed<k>.partial/`, then rename. A directory that exists is a directory that
  finished.
- The sweep runner **skips** any run whose directory already exists. Sweeps are
  resumable and safe to re-run.
- `env_id` format: `Empty-5`, `DoorKey-8`, `MultiRoom-N4`.
- `strategy` is one of: `epsilon_greedy`, `boltzmann`, `count_based`, `noisy`.

Changing this format breaks all three workstreams at once. Do not change it
without telling the other two people.

---

## 6. Frozen contracts

These three things were agreed on 2026-08-17 so that three people can work in
parallel. **Changing any of them requires notifying the other two team members**,
because they will silently break work in progress:

1. `src/rlx/exploration/base.py` — the `Explorer` interface.
2. `src/rlx/config.py` — the config schema field names.
3. The results directory format in §5.

The `Explorer` interface:

```python
class Explorer(ABC):
    uses_noisy_net: bool = False

    @abstractmethod
    def act(self, q_values: np.ndarray, count_key: Hashable, step: int) -> int:
        """Choose an action given Q-values for the current observation."""

    def intrinsic_bonus(self, count_key: Hashable) -> float:
        """Extra reward added to the transition stored in the replay buffer.
        Never enters evaluation return. Default: no bonus."""
        return 0.0

    def observe(self, count_key: Hashable) -> None:
        """Called once per environment step, after acting. Default: no-op."""

    def stats(self) -> dict[str, float]:
        """Scalars to log this step (epsilon, temperature, mean bonus)."""
        return {}
```

`count_key` is the agent's **own 7x7x3 observation as raw bytes**. It is NOT the
privileged `(x, y, direction)` state. See §8.

---

## 7. Experiment design (summary)

The full reasoning is in `docs/specs/2026-08-17-exploration-comparison-design.md`.
The load-bearing facts:

**Algorithm:** Double DQN, fixed across every strategy and environment. Chosen
because vanilla DQN overestimates action values, and inflated Q-values act as an
accidental exploration bonus — which would contaminate exactly the comparison
this project makes. Not dueling.

**Fixed hyperparameters** (identical for all strategies, all environments):

```
buffer_size        100_000
batch_size         32
optimizer          Adam
learning_rate      1e-4
gamma              0.99
target_update      1000 steps (hard copy)
learning_starts    1000 steps
train_freq         every 4 environment steps
grad_clip          10.0
total_steps        400_000        # UNVERIFIED until the day-1 benchmark
eval_every         5_000 steps
eval_episodes      1              # evaluation is deterministic -- see below
snapshot_every     10_000 steps   # visitation array; 8 points in the early-AUC window
seeds              0, 1, 2, 3, 4
```

**Layouts are pinned per run.** MiniGrid regenerates its maze on every `reset()`,
which would leave state coverage without a fixed denominator. Every `reset()`
therefore passes the run's seed, so one run sees exactly one maze; the 5 seeds
give 5 layouts per instance.

Because the layout is pinned and evaluation is greedy, MiniGrid is fully
deterministic at evaluation time, so `eval_episodes = 1` — 10 would produce 10
identical numbers. Per-run learning curves are step functions; smooth curves come
from aggregating across seeds. Spec §4.5 has the full reasoning.

**Observation:** standard partial 7x7x3 egocentric view via `ImgObsWrapper`,
through a 3-layer CNN. Partial observability is kept deliberately — the professor's
feedback assumes it.

**Action space:** full 7 actions, unrestricted, identical for every strategy.

**Experiment matrix** — 13 instances x 4 strategies x 5 seeds = **260 runs**:

| Family | Instances | Difficulty axis |
|---|---|---|
| Empty | size 5, 8, 16 | grid size |
| DoorKey | size 5, 6, 7, 8, 10 | grid size |
| MultiRoom | N = 2, 3, 4, 5, 6 | room count |

Environments are constructed by **direct class instantiation**, not registered
gym IDs, because arbitrary sizes are needed for the continuous difficulty axis.

**Step budget is the same for every instance.** Do not give harder environments
more steps: difficulty and budget would vary together and the difficulty curve
would mean nothing. Hard instances failing is a *finding*, not a bug.

**Evaluation is always greedy, on extrinsic reward only.** Every 5k steps: run
the pinned layout with greedy action selection, intrinsic bonus off, NoisyNets
noise off (mean weights). Intrinsic bonuses exist only inside the replay buffer
and never reach a reported number.

---

## 8. The privileged-information rule

This is the point the professor pressed hardest on, and it is easy to get wrong.

- The **agent** sees only the partial 7x7x3 observation. Nothing else. Ever.
- The **analysis** uses the true `(x, y, direction)` state, which is privileged
  information the agent never receives. This is fine — it is measurement, not
  learning — but it must be stated explicitly in the report.
- Therefore the count-based bonus counts **the agent's own observations**, not `(x, y, dir)`.
  Using true state for the bonus would hand one strategy privileged information
  the other three do not get, breaking the controlled comparison.

One sentence to keep straight: **`(x, y, dir)` is used only for analysis, never by
any agent.**

---

## 9. The statistical trap (do not fall into this)

Both coverage and return fall as difficulty rises. Correlating them naively
across all runs produces a large positive correlation that means nothing — it is
"hard environments are hard", measured twice.

**The hypothesis test is run within each environment instance** (across 4
strategies x 5 seeds, where difficulty is held constant), then aggregated across
instances with bootstrap confidence intervals.

Any analysis code that pools runs from different environment instances into a
single correlation is wrong. Reject it.

---

## 10. Git workflow

**All of this is done by us, the three humans. Claude never runs a git command
that changes anything — see Rule 7.**

- Branches: `core/<topic>` (Samuel), `exploration/<topic>` (Max),
  `analysis/<topic>` (Daniel).
- Open a PR to `main`. PRs merge each evening.
- Rebase on `main` daily, every day.
- Commit messages: imperative mood, one line, plus a body if the change is not
  obvious.
- Never commit `results/` output during development. The final result set is
  committed once, at the end, in a single dedicated commit.

---

## 11. Working conventions

- Type-hint public functions. Do not type-hint local variables.
- Docstrings on public functions only, one line unless genuinely complex.
- No `print()` in library code. `train.py` and `sweep.py` may print progress.
- Seeds: every run seeds `random`, `numpy`, `torch`, and the environment from a
  single integer. Evaluation builds a **separate environment instance on the same
  pinned layout** (`layout_seed = cfg.seed`) — it must be the same maze, since
  that is the task being scored; isolation comes from the separate instance, not
  from a different layout.
- All randomness goes through an explicitly passed generator. No global RNG use
  inside library functions.
- Paths via `pathlib`, never string concatenation. The team is on Windows;
  hard-coded `/` separators will break.

---

## 12. Where to look

| Question | File |
|---|---|
| What are we building and why? | `docs/specs/2026-08-17-exploration-comparison-design.md` |
| What am I supposed to do next? | `implementation_plan/<name>/` |
| What does this RL term mean? | `docs/glossary.md` |
| Why was it built this way? | `docs/decision_log.md` |
| What did we try that failed? | `docs/decision_log.md` (discarded entries) |
| What did the professor ask for? | `context/proposal_response.md` |
