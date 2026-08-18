# STATUS — who is on what, right now

**Update this whenever you finish a task, before you open the PR.** It is the one
place the other two look to answer "can I start yet?".


Last updated: **2026-08-18**, by Daniel.


---

## Where everyone is

| | Workstream | Done | Currently on | Next |
|---|---|---|---|---|
| **Samuel** | A — Core & Infrastructure | ✅ 1 scaffold, ✅ 2 verify+benchmark, ✅ 3 env factory, ✅ 4 network + buffer, ✅ 5 agent + training loop, ✅ 6 sweep runner | — | **all core tasks done** |
| **Max** | B — Exploration strategies | ✅ 1 epsilon-greedy, ✅ 2 boltzmann, ✅ 3 count-based | — | 4 noisy-nets (blocked on Samuel 4) |
| **Daniel** | C — Logging, metrics & analysis | ✅ 1 visitation logging, ✅ 2 aggregation, ✅ 3 coverage metrics | — | 4 statistics |


## What is available on `main` right now

Everything below is merged and safe to import.

| Symbol | From | Who needs it |
|---|---|---|
| `RunConfig` | `rlx.config` | Max, Daniel |
| `Explorer` (ABC) | `rlx.exploration.base` | Max |
| `make_explorer`, `STRATEGIES` | `rlx.exploration` | Max |
| `ENV_IDS`, `difficulty_index` | `rlx.envs` | Daniel |
| `RunLogger` | `rlx.logging` | Samuel |
| `make_env`, `grid_info`, `GridInfo` | `rlx.envs` | Daniel (coverage), Samuel |
| `reachable_mask`, `bfs_distances` | `rlx.envs` | Daniel (coverage) |
| `QNetwork`, `NoisyLinear` (placeholder), `obs_to_tensor`, `obs_batch_to_tensor` | `rlx.networks` | **Max (NoisyNets)**, Samuel |
| `ReplayBuffer` | `rlx.buffer` | Samuel |
| `DoubleDQNAgent` | `rlx.agent` | Samuel |
| `run_training(cfg)`, `evaluate`, `python -m rlx.train` | `rlx.train` | Samuel (sweep), anyone wanting a real run |
| `expand_matrix`, `select_shard`, `pending_runs`, `python -m rlx.sweep` | `rlx.sweep` | all three of us on the 20th |
| `configs/pilot.yaml` | 16-run smoke sweep | integration day |
| `EpsilonGreedy` | `rlx.exploration.epsilon_greedy` | Samuel (training loop), Max |
| `Boltzmann` | `rlx.exploration.boltzmann` | Samuel (training loop), Max |
| `CountBased` | `rlx.exploration.count_based` | Samuel (training loop), Max |
| `cfg`, `rng`, `q_values`, `key` fixtures | `tests/test_exploration/conftest.py` | Max |
| `RunResult`, `load_run`, `load_all`, `to_dataframe`, `final_return` | `rlx.analysis.aggregate` | Daniel |
| `scripts/make_synthetic_results.py` (`--out`, `--no-effect`) | fake results in the real format | Daniel, anyone testing analysis |

Settled by Daniel's task 2:
- Analysis can be developed with **no real experiments**: the synthetic fixture
  writes the frozen §5 format (`steps` int64, `counts` int32 `(T, W, H, 4)`).
- The fixture ships **two** datasets: the default has the hypothesis baked in
  (within-instance Spearman **+0.49 to +0.94**, rising with difficulty, all
  p < 0.03), `--no-effect` has none (**−0.22 to +0.19**, all p > 0.34). Task 4
  must pass **both**. Re-measured 2026-08-18 after the reachability fix below;
  the ranges before it were 0.58–0.93 and −0.22 to +0.15.
- **Task 4 requirement:** an instance where every run scores the same has an
  undefined within-instance correlation. Report those as "no variance, excluded",
  never as a silent `NaN`. This is a plausible real outcome for `DoorKey-10` and
  `MultiRoom-N6`, not just a fixture artefact.
- **`difficulty` is comparable only within a family** — grid size for
  Empty/DoorKey, room count for MultiRoom. Never sort or correlate across
  families on it; group by `family` first.
- **2026-08-18 fix:** the fixture used to scatter visits across the *whole*
  grid, walls included, which let `raw_coverage()` read above 1.0 (observed 54.4
  on MultiRoom-N2, where only 1.8% of the 25x25 grid is reachable). The
  `distinct_states` column had the same bug. Both now only count reachable
  `(x, y, dir)` states, via the same
  `grid_info` / `reachable_mask` real runs use. If you have a `results_synthetic*`
  directory from before this date, **regenerate it**:
  `python scripts/make_synthetic_results.py --out results_synthetic` (and
  `--no-effect` for the control) — it is gitignored, so nothing was silently
  fixed under you. Details in `docs/decision_log.md`, "The fake data had agents
  walking through walls".

Settled by Daniel's task 3 (`analysis/coverage.py`, tests pass, **`envs.py` not
part of it**):
- `coverage.py` was written and tested against a **local, uncommitted copy** of
  Samuel's task 3 (`GridInfo`, `bfs_distances`, `reachable_mask`, `grid_info`),
  pasted verbatim from `samuel/03-env-factory.md`. `tests/test_coverage.py`
  touches only the frozen contract, so it must pass unchanged against Samuel's
  real implementation. When Samuel 3 merges: `git checkout main -- src/rlx/envs.py`,
  then re-run `pytest tests/test_coverage.py`.
- **A door only counts as a route waypoint when the maze also has a key.** The
  planned `_landmark_chain` appended any non-`None` door, which is right for
  DoorKey and wrong for MultiRoom — MultiRoom's rooms are joined by doors,
  `grid_info` keeps only the *last* of the N−1 it finds, and routing through
  that arbitrary door dragged an unrelated room into the task-relevant mask.
  `test_a_multiroom_connecting_door_is_not_a_waypoint` fails if this is reverted.
- **On the Empty family raw and task-relevant coverage are identical (ratio 1.00),
  by construction, for every seed** — start and goal are opposite corners of an
  open grid, so every interior cell is on some shortest path. Not fixable in
  code. Task 4 and 5 must not present the two as independent predictors on Empty.
  The distinction only does real work on DoorKey-7/8/10 (ratios 0.81 / 0.65 / 0.47).
- Full denominator table (reachable / on-route / +neighbours per instance) is in
  `docs/decision_log.md`, entry "Two ways of measuring coverage".

**For Samuel — from Daniel's task 3.** `CLAUDE.md` §7 says "the 5 seeds give 5
layouts per instance". That is **false for the Empty family**: `grid_info("Empty-N", s)`
returns start `(1,1)` and goal `(N-2,N-2)` for every `s` in 0..4 — verified on
Empty-5/8/16. The seeds still vary agent and network randomness, so the runs are
not duplicates, but the *maze* does not change. DoorKey and MultiRoom do vary.
Worth a correction in `CLAUDE.md` §7 and the spec, since it is your section.

**For Samuel — one question from Daniel's task 2 review.** `final_return` is
defined in spec §7.1 as the mean of the **last 5 evaluation points**, but
`metrics.csv` holds one row per *logged* step. If `train.py` logs `loss` more
often than it evaluates, most rows carry an empty `eval_return_mean`.
`rlx.analysis.aggregate.final_return` now drops the empty entries before taking
the tail, so it is correct either way — but if you intend to log at a different
cadence than you evaluate, say so, because it changes how `metrics.csv` looks.

**For Samuel — `configs/pilot.yaml` cannot measure the project's main predictor.**

With `total_steps: 20000` and the default `snapshot_every: 10000`, `train.py`
writes snapshots at steps **10,000 and 20,000** (there is none at step 0). The
early-coverage window is `0.2 * 20000 = 4000` steps wide, so **zero snapshots
land inside it**.

`early_auc` used to return `coverage[0]` in that case — the coverage at step
10,000, which is 2.5x outside the window, handed back as a normal-looking float.
As of 2026-08-18 it **raises ValueError** instead, so the pilot now fails loudly
rather than printing a fake number. Details in `docs/decision_log.md`, entry
"The pilot could not measure early coverage".

One of these has to happen before the 20th:
- **`snapshot_every: 1000` in `configs/pilot.yaml`** — 4 points in the window
  (1000/2000/3000/4000). 20 snapshots per run instead of 2, ~1 KB each, so the
  cost is nil. **Recommended:** it makes the pilot actually exercise `early_auc`,
  which is what a smoke test is for.
- Or skip `early_auc` when smoke-testing analysis on `results_pilot/`. Then the
  first time that code path runs for real is on the 260-run sweep.

**Real runs are unaffected:** 400,000 steps at `snapshot_every: 10000` puts **8**
snapshots in the 80,000-step window — which is exactly what the `snapshot_every`
comment in `config.py` was sized for.

**For Samuel — a test of yours failed once and did not reproduce.**
`tests/test_agent.py::test_double_dqn_target_differs_from_vanilla_max` failed on
2026-08-18 at **line 82**, `assert not np.allclose(double, vanilla)` — the
"the two networks must disagree" assertion, **not** the
`double <= vanilla` invariant on the line below it, which held.

Seen **once in ~6 full-suite runs**; 5 later full runs and the test in isolation
were green, and a sweep over 40 different ambient `torch` RNG states did not
reproduce it. **UNVERIFIED: the trigger is not understood.**

Structural reason it *can* vary at all: the test builds `DoubleDQNAgent(...)`
**before** it calls `torch.manual_seed(1)`, so the online network's weights come
from whatever global torch RNG state earlier tests happened to leave behind. The
target net is then `online + randn(seed 1) * 0.5`. If online's argmax coincides
with target's argmax for all 64 sampled observations, `double == vanilla` and the
assertion fires. Moving the `torch.manual_seed(1)` above the constructor would
make the test independent of execution history.

Not caused by Daniel's 2026-08-18 fixture fix: that change touches only
`scripts/make_synthetic_results.py`, which no test imports (`tests/test_aggregate.py`
runs it as a **subprocess**), and it edits neither `src/` nor `tests/`.
It is your file, so nobody has changed it. **If it fails again, it is real.**

**For Samuel and Daniel — one question from Max's task 3.** `count_beta = 0.05`
is provisional and the task file told us to measure it before trusting it. Over
one 300-step episode with 100 distinct views, the intrinsic bonus totals **11.42**
against a maze reward of ~0.9 — **12.7x** — falling below the maze reward only
once every view has been seen ~1,000 times. `count_beta = 0.0039` would put the
first episode level with the maze reward. **Nothing was changed.** It may well be
correct as is, since these mazes pay nothing at all until the goal is first
reached, so early on the bonus is the only learning signal there is. Decide
together before anyone edits `config.py`; full numbers in `docs/decision_log.md`
under "2026-08-18 — Count-based exploration implemented".

**Answer for Daniel — logging cadence.** `train.py` calls `logger.log_step(...)`
**only inside the evaluation block**, so `metrics.csv` gets exactly one row per
evaluation point and `eval_return_mean` is never empty. I am committing to that
design, so `final_return` can rely on it. Your `dropna` before taking the tail is
harmless and worth keeping as a guard — do not remove it.

**Answer for Max — the bonus scale, measured on the real mazes.** You were right.
Measured with a random policy: Empty-5 1.6x, **DoorKey-8 14.1x**, MultiRoom-N4
1.6x, MultiRoom-N6 2.1x the value of winning.

The cause is **episode length**, which MiniGrid varies from 80 to 640 steps
across our mazes. The bonus is paid per step, so long-episode mazes accumulate
far more of it. No single `count_beta` balances all 13.

**MEASURED 2026-08-18 at 100k steps with a bonus-free control** (3 seeds each,
full table in the decision log):

| strategy | Empty-5 | DoorKey-5 |
|---|---|---|
| epsilon-greedy (no bonus) | **0.637** | 0.000 |
| Boltzmann (no bonus) | **0.064** | 0.000 |
| count_based beta=0.05 (current) | 0.126 | 0.000 |
| count_based beta=0.01 | **0.573** | 0.000 |
| count_based beta=0.005 / 0.001 | 0.127 | 0.000 |

**DoorKey-5 is unsolved by everything at 100k**, so count_based's zeros there were
never a bonus problem. On Empty-5, beta=0.01 (0.573) lands next to the bonus-free
baseline (0.637) while our current 0.05 (0.126) does not.

Treat this as a sanity check, **not** as the reason: 3 seeds with a 0.955/0.000/
0.764 spread cannot decide a hyperparameter. We decide on the pre-registered scale
argument, which points the same way.

**TWO decisions needed before the sweep, not one:**

1. **`count_beta`: 0.05 -> 0.01.**
2. **Boltzmann's temperature schedule.** Boltzmann scored 0.064 against
   epsilon-greedy's 0.637 on the *easiest* maze, with no bonus involved. Max
   predicted why in his own log entry: temperature only means something relative
   to the Q-values, MiniGrid's action gaps are ~0.01 early on, and our tau is
   still 0.47 at step 40,000 — so it is still choosing almost at random a quarter
   of the way through training. Same illness as the bonus: a round number picked
   without checking it against MiniGrid's actual reward scale.

If we launch as-is, "Boltzmann came last" is a statement about our schedule, not
about Boltzmann.

**Proposed: `count_beta = 0.01`** — worst case falls from 14x to ~3x while the
bonus stays meaningful on MultiRoom. **Nobody has changed `config.py`.** This
needs all three of us to agree, and it must be settled **before the sweep on the
20th** — changing it after seeing which strategy wins would be choosing our own
result. Full numbers in `docs/decision_log.md`, "Measuring Max's count-bonus
question against the real mazes".

**For Daniel — two things about the mazes that affect your analysis.**
`Empty` gives the *same* maze for all 5 seeds (it has no random parts), and
scoring is deterministic, so Empty may show **zero variance in final score** —
your "no variance, excluded" path will fire there systematically, not by chance.
And MultiRoom maze *size* varies up to 3x between seeds (N2: 11 to 33 reachable
squares), so its within-maze correlations will be noisier than DoorKey's.
Reachable counts per maze are in the decision log.

Settled by Samuel's task 2, no longer provisional:
- `device: cpu` — measured, the GPU is 13% slower. **Do not install a CUDA torch.**
- `total_steps: 400000`, `snapshot_every: 10000`, **`--workers 12`** (was 8;
  see the 2026-08-18 sweep entry — torch is now pinned to one thread per run,
  which made the pilot 3.4x faster and changed the best worker count)
- All 13 MiniGrid instances verified on minigrid 3.1.0
- **MultiRoom grids are 25x25 for every N** — do not hard-code 16

## Who is blocked on what

| Waiting task | Needs | From | Status |
|---|---|---|---|
| Max 1, 2, 3 | `RunConfig`, `Explorer` | Samuel 1 | ✅ **unblocked** |
| Daniel 1, 2 | `RunConfig`, `difficulty_index` | Samuel 1 | ✅ **unblocked** |
| Daniel 3 (coverage) | `grid_info`, `reachable_mask`, `bfs_distances` | Samuel 3 | ✅ **unblocked** |
| Max 4 (NoisyNets) | `QNetwork`, `NoisyLinear` placeholder | Samuel 4 | ✅ **unblocked** |
| Samuel 5 (training loop) | `RunLogger` | Daniel 1 | ✅ **done** |


**Nothing is blocked any more.** Every task in the plan can now proceed.

**SWEEP COMMAND for the 20th** — one per machine, run them at the same time:
```
python -m rlx.sweep --config configs/main.yaml --shard 0/3 --workers 12
```
Samuel `0/3`, Max `1/3`, Daniel `2/3`. 87/87/86 runs. Expect **3-4 hours**.
Safe to re-run: finished runs are skipped, so a crash resumes where it stopped.
**Do not launch until `count_beta` is decided.**

**The pipeline is verified end to end.** A real 20k-step run on Empty-5 reached
a score of 0.955, and **Daniel's `load_all` read the result folder our training
loop wrote with no adjustment** — folder layout, `metrics.csv` columns and
`visitation.npz` arrays all line up. The biggest integration risk is behind us.

**For Max — one test is waiting on you.** `tests/test_train.py` runs all four
strategies end to end; the `noisy` case is marked expected-to-fail until
`rlx.exploration.noisy` exists. It **detects your module automatically**, so the
moment you merge it becomes a real test. Nothing to delete, but do check it goes
green rather than staying an `x` in the pytest output.

**For Daniel — a note on `final_return`.** Averaging the last 5 evaluation
points is right for real runs (80 points), but on short test runs it mixes in
the pre-learning zeros: our 20k smoke run scored 0.955 at the end and
`final_return` reported 0.239. Correct by definition, just surprising. Worth a
sentence in the report so nobody reads a low number as a failed run.

**For Max — your NoisyNets task.** `NoisyLinear` in `src/rlx/networks.py` is a
placeholder that behaves like a plain `nn.Linear`. Replace its body only: keep
the class name, the `(in_features, out_features, sigma0)` signature, and the
`reset_noise()` / `noise_enabled` members, because `QNetwork.reset_noise` and
`set_noise_enabled` find your layers with `isinstance(m, NoisyLinear)`. Your task
file has you change it from subclassing `nn.Linear` to subclassing `nn.Module` —
that is fine and the isinstance checks still work.

---

## How to update this file

Three things, takes a minute:

1. Move your task from **Currently on** to **Done** in the first table.
2. If you produced something others import, add it to **What is available**.
3. If you unblocked someone, flip their row in **Who is blocked** to ✅ and tell
   them — do not rely on them noticing this file.

Also tick the `- [x]` checkboxes in your own task file as you go. Someone reading
`samuel/03-env-factory.md` should be able to see how far through it you are
without asking.

**This file is a courtesy, not a source of truth.** `main` is the source of
truth. If this file and the code disagree, the code is right and this file is
stale — fix it.
