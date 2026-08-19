# STATUS — who is on what, right now

**Update this whenever you finish a task, before you open the PR.** It is the one
place the other two look to answer "can I start yet?".


Last updated: **2026-08-19** (second update, after the tasks 1-4 review), by Daniel.


---

## Where everyone is

| | Workstream | Done | Currently on | Next |
|---|---|---|---|---|
| **Samuel** | A — Core & Infrastructure | ✅ 1 scaffold, ✅ 2 verify+benchmark, ✅ 3 env factory, ✅ 4 network + buffer, ✅ 5 agent + training loop, ✅ 6 sweep runner | — | **all core tasks done** |
| **Max** | B — Exploration strategies | ✅ 1 epsilon-greedy, ✅ 2 boltzmann, ✅ 3 count-based, ✅ 4 noisy-nets, ✅ knob calibration (`tau_start` 0.1 after pilot) | — | 5 write-ups |
| **Daniel** | C — Logging, metrics & analysis | ✅ 1 visitation logging, ✅ 2 aggregation, ✅ 3 coverage metrics, ✅ 4 statistics, ✅ 5 figures | — | 6 report scaffold |


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
| `NoisyExplorer` | `rlx.exploration.noisy` | Samuel (training loop), Max |
| `NoisyLinear` (real, factorised Gaussian) | `rlx.networks` | Samuel — placeholder is gone, `test_every_strategy_runs_end_to_end[noisy]` now runs for real |
| `cfg`, `rng`, `q_values`, `key` fixtures | `tests/test_exploration/conftest.py` | Max |
| `FIGURE_NAMES`, `COLORS`, `LABELS`, `make_all_figures`, `python -m rlx.analysis.figures` | `rlx.analysis.figures` | Daniel (task 6), report |
| `RunResult`, `load_run`, `load_all`, `to_dataframe`, `final_return` | `rlx.analysis.aggregate` | Daniel |
| `raw_coverage`, `task_relevant_coverage`, `task_relevant_mask`, `early_auc` | `rlx.analysis.coverage` | Daniel |
| `build_analysis_table`, `within_instance_correlation`, `aggregate_correlation`, `compare_coverage_predictors`, `iqm_by_strategy`, `rank_stability`, `probability_of_improvement`, `rliable_aggregate`, `performance_profile` | `rlx.analysis.stats` | Daniel (figures), report |
| `scripts/make_synthetic_results.py` (`--out`, `--no-effect`) | fake results in the real format | Daniel, anyone testing analysis |

Settled by Daniel's task 2:
- Analysis can be developed with **no real experiments**: the synthetic fixture
  writes the frozen §5 format (`steps` int64, `counts` int32 `(T, W, H, 4)`).
- The fixture ships **two** datasets: the default has the hypothesis baked in,
  `--no-effect` has none. Task 4 must pass **both**. Re-measured **2026-08-19**
  after the extension to all 13 instances (`early_auc_raw`):

  | | effect dataset | `--no-effect` control |
  |---|---|---|
  | within-instance Spearman | **+0.13 to +0.95** | **−0.28 to +0.28** |
  | p-value range | 0.0000 to 0.5845 | 0.2309 to 0.7837 |
  | mean rho, 95% CI | **+0.696** [+0.544, +0.836] | **−0.003** [−0.102, +0.095] |
  | `trend_with_difficulty` | **+0.900** | **+0.167** |
  | mean per-instance CI width | **0.450** | **0.935** |
  | `ci_excludes_zero` | `True` | `False` |
  | `confirms_h1` | `True` | `False` |
  | `confirms_h2` | `False` (by construction) | `False` |

  Re-measured after the 2026-08-19 review; identical to before it, because the
  `early_auc` fix is a monotone rescaling and Spearman is rank-based.

  Earlier ranges, for reference: 6-instance fixture 2026-08-18 was +0.49..+0.94
  and −0.22..+0.19; before the reachability fix, 0.58..0.93 and −0.22..+0.15.
  **Individual easy instances are no longer all significant** (Empty-5 is now
  rho +0.13, p 0.58) — that is deliberate, see the difficulty-trend note below.
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

**For BOTH of you — RESOLVED 2026-08-19, but you must re-install.**

`rliable` did not import (pandas 3.0.5 vs arch 7.2.0). It works now.

```bash
pip install -r requirements.txt
```

**Do this before your next pull, or `rlx.analysis.stats.rliable_aggregate` will
raise on your machine.** Verify with:

```bash
python -c "from rliable import library; print('ok')"
```

**The fix is NOT the one this file recommended yesterday.** `arch>=8.0` does not
work: arch 8.0.0 renamed `IIDBootstrap`'s `random_state` parameter to `seed`, and
`rliable` 1.2.0 (the newest release, no update pending) still passes
`random_state=`, so every bootstrap raises `TypeError`. Yesterday's note had only
verified the *import*, never a *call*.

What actually works, and what `requirements.txt` now pins — **both together, they
do not work apart**:

| pin | why |
|---|---|
| `pandas>=2.0,<3` | arch 7.2.0 calls pandas' internal `deprecate_kwarg` the old way; pandas 3 changed it, so arch will not import |
| `arch>=7.2,<8` | arch 8 renamed the parameter rliable passes |

Windows wheel confirmed present: `arch-8.0.0-cp311-cp311-win_amd64.whl` exists,
and 7.2.0 likewise ships `cp311-win_amd64`. Nothing here needs a compiler.

`rliable_aggregate(df, seed=0)` is written, tested and available.
`iqm_by_strategy` (hand-rolled) is unchanged and still used for the per-instance
bars; `rliable_aggregate` is the single cross-instance number the proposal
promised.

**One thing to know if you ever call rliable directly:** its `random_state=`
argument is ignored. `StratifiedBootstrap.update_indices` calls `np.random.choice`,
i.e. the global numpy RNG. `rliable_aggregate` seeds the global RNG and restores
the previous state in a `finally` block — the one place in the codebase that
touches global randomness (`CLAUDE.md` §11). Do not "clean this up" into an
explicitly passed generator; it does not work.

**Difficulty trend — FIXED 2026-08-19. The fixture now covers all 13 instances.**

`FIXTURE_ENV_IDS` was a 6-instance subset (two per family), which is below the
three a Spearman needs, so `aggregate_correlation`'s `trend_with_difficulty` came
back `NaN` and hypothesis H2 had no end-to-end test. It is now `list(ENV_IDS)` —
**260 synthetic runs, 8.6 MB, ~3 s to generate** (was 120 runs, 3.5 MB, 1.3 s).

**Regenerate yours** — anything older than 2026-08-19 has the old shape and will
fail `tests/test_aggregate.py`:

```bash
python scripts/make_synthetic_results.py --out results_synthetic
```

**Extending the instance list was not sufficient on its own**, and this is worth
knowing before anyone touches the fixture again. The baked-in effect strength was
tied to *absolute* difficulty, whose span inside one family is only 0.30. Every
MultiRoom instance therefore sat at rho 0.90–0.97 — against a hard ceiling of
1.0, with no headroom for a rising effect to show. First measurement:
**trend +0.13 on the effect dataset against +0.17 on the `--no-effect` control**,
i.e. signal below noise, a useless test.

The effect is now tied to an instance's position **within its family**
(`_family_position`), which is exactly what `trend_per_family` measures. After
the change: **+0.900 against +0.167**, a factor of 5.

Consequence to expect: the easy end of each family now correlates weakly on
purpose (Empty-5 is rho +0.13, p 0.58). The aggregate still confirms H1
(`confirms_h1=True`, CI excludes zero), but **the fixture no longer has every
instance individually significant** — do not write a test that assumes it does.

**Three tests in `tests/test_aggregate.py` had the old size baked in** and were
updated: the two `len(...) == 120` assertions now derive `EXPECTED_RUNS` from
`len(ENV_IDS) * len(STRATEGIES) * 5`, and
`test_grids_match_the_real_minigrid_dimensions` no longer carries a hand-written
6-entry dict.

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

**BOTH DECISIONS ARE SETTLED — agreed by all three on 2026-08-19 and applied to
`config.py`. `tau_start` 1.0 -> 0.01, `tau_end` 0.05 -> 0.001, `count_beta` 0.05
-> 0.01.** Boltzmann now goes from 28% to 93% on picking its own favourite action
(it was 14.3% to 15.1%, against 14.3% for a coin flip); the novelty bonus over an
early episode drops from 11.42 to 2.28 against a maze reward of ~0.9.
`noisy_sigma0` stays 0.5, `tau_decay_frac` stays 0.4, epsilon-greedy untouched.

**SUPERSEDED for `tau_start` ONLY — see the pilot-run section below.** `tau_start`
is now **0.1**, not 0.01. `tau_end` 0.001, `count_beta` 0.01 and every other value
above stand unchanged. The 28% -> 93% figures quoted above were computed against
the trained Q-scale at both ends of the schedule, which is the error the pilot
caught: at step 0 the scale is 7x wider, so 0.01 actually gave 86% — committed
before learning anything. Full before/after in `docs/decision_log.md`, "The pilot
run caught that we had over-corrected Boltzmann". A regression test now fails if
the schedule goes back to being effectively random OR to being greedy at step 0.

### 2026-08-19 (evening) — pilot run: `tau_start` 0.01 -> 0.1, and two items for you

The 16-run pilot finished, 16 ok / 0 failed, all four contract files present in
every run directory. `early_auc` returned a real number on all 16 — the
`snapshot_every: 1000` fix works. Three things came out of it.

**1. Mine, fixed. Boltzmann visited 4 of ~36 states on Empty-5** — one square,
four directions, for 20k steps. `tau_start = 0.01` was calibrated against the
*trained* Q-gap (0.0030), but a freshly initialised network's gap is **0.0206**,
measured over 6 instances x 15 seeds (spread across instances < 0.002 — it is a
property of the network init, not the maze). So at step 0 it gave p(best) = 0.86:
committed to a random preference before learning anything.

`tau_start` 0.01 -> **0.1**, giving p(best) 0.28 at step 0 and 0.95 at the end.
Re-ran the 4 Boltzmann pilot runs to confirm rather than trusting the arithmetic:
states visited went 4 -> 32 (Empty-5 seed1), 4 -> 24 (both DoorKey-5 seeds).
Eval return was **0.000 in both arms**, so this was chosen on coverage, not on the
outcome variable. Suite: 220 passed. Regression test now checks both ends of the
schedule and fails on both historical values.

**2. For Samuel — greedy eval scores 0 while training scores 0.9.** On Empty-5 all
four strategies reach `train_return_mean` 0.88-0.95 but `eval_return_mean` 0.000.
Reaching the goal always scores > 0.1, so exactly 0 means the eval episode timed
out without ever reaching it. The behaviour policy solves the maze; the greedy
policy does not. Most likely a deterministic cycle that a single random action
would break.

I read `evaluate()` (`src/rlx/train.py:61`) and could not find anything wrong with
it — noise off, no bonus, pinned layout, all correct. My guess is 20k steps is
just too short (5,000 gradient updates vs 100,000 in the real run), and one run
did reach 0.955 and hold. **Not claiming this is a bug — flagging it because if it
persists at 400k steps every strategy scores zero and we have no result.** Worth a
single 400k-step run on Empty-5 before committing 3-4 hours to the full sweep.

**3. For Daniel — `task_relevant_coverage` == `raw_coverage` on all 16 pilot
runs.** Not a bug: on Empty-5 and DoorKey-5 the task-relevant mask covers 100% of
free cells, so there is nothing to separate. It separates properly on bigger
instances (DoorKey-8 0.65, DoorKey-10 0.47, MultiRoom-N2 0.02, MultiRoom-N6 0.09).

But one case is structural rather than a size artefact: for the **whole Empty
family** the two measures are identical at every size, Empty-16 included (ratio
1.00, 196/196 cells). `_on_shortest_path` admits any cell with
`d_start + d_goal == total`, and in a room with no walls every interior cell
satisfies that. Correct by definition, but it means task-relevant coverage carries
no information for 3 of our 13 instances. Probably worth one sentence in the
report rather than a code change — your call, it is your module.

The original discussion follows, kept for the record:

**~~TWO decisions needed before the sweep, not one:~~** (both now done)

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

**MEASURED 2026-08-19, and it is worse than "still random at 40k" — Boltzmann is
random for the WHOLE run.** Real Double DQN, 6 instances x 2 seeds x 160,000
steps, epsilon-greedy as the behaviour policy (using Boltzmann would make the
measurement depend on the parameter being chosen). Recorded the best-vs-second
Q gap at every step:

| instance | solved? | median gap |
|---|---|---|
| Empty-5 | 0.95 | 0.0059 |
| Empty-16 | 0.76 | 0.0034 |
| DoorKey-5 | 0.97 | 0.0029 |
| DoorKey-8 | 0.01 | 0.0001 |
| MultiRoom-N4 | 0.00 | 0.0001 |
| MultiRoom-N6 | 0.00 | 0.0001 |

Gaps track whether a reward was ever found, and **did not grow** over 160k steps.
They are ~0.003, not the ~0.01 estimated. Against tau running 1.0 -> 0.05, that
gives p(pick own favourite action) of **0.143 at start, 0.144 at 40k, 0.151 at
tau_end** — uniform is 0.143, epsilon-greedy ends at 0.957. `tau_end` is the
broken end, not `tau_start`, and it is ~500x too large.

**PROPOSED — `tau_start` 1.0 -> 0.01, `tau_end` 0.05 -> 0.001.** Shape and
`tau_decay_frac = 0.4` unchanged. Endpoints derived from the measured 0.0034 gap
by stating a target and inverting the softmax, not by trying values. Gives
p(favourite) 0.28 -> 0.93 on instances with a reward signal, ~0.18 on the ones
without (correct: nothing to exploit there). **Nobody has edited `config.py`.**

Full derivation, limits and the rejected adaptive-tau alternative are in
`docs/decision_log.md`, "Boltzmann's temperature was measured against the real
mazes". Reproduce with `scripts/measure_q_gaps.py`.

**`noisy_sigma0` was checked the same way on 2026-08-19 and is FINE — it stays at
0.5, so this remains TWO decisions, not three.** Measured how often the greedy
action changes when the noise is redrawn (86% = uniform random, 0% = no
exploration): Empty-5 32% -> 43% -> 9%, DoorKey-5 32% -> 70% -> 16%,
MultiRoom-N4 32% -> 80% -> 76% over 50k steps. Explores early, commits once it
learns, keeps exploring where it never learns. It self-corrects because sigma is
a *learned* parameter — it fell ~26% on its own — which is exactly why tau and
count_beta cannot self-correct and it can. Details in `docs/decision_log.md`,
"NoisyNets implemented, and its knob turns out to be fine"; reproduce with
`scripts/measure_sigma.py`.

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

---

## Tasks 1-4 reviewed against every requirement, 2026-08-19 — nine fixes

Full read-back of `context/proposal_response.md`, the spec, and `CLAUDE.md`
against the code. Nine findings, all fixed on `analysis/statistics`. Suite went
from 183 to **202 passed, 1 xfailed**. Details in `docs/decision_log.md`,
"A full review of tasks 1-4 found nine things, and we fixed all nine".

**For Samuel — one behaviour change in a file your `train.py` calls.**

`RunLogger.finalize` now raises `FileExistsError` when the run directory already
exists. It used to `rmtree` it. The sweep runner skips existing directories, so
nothing in a normal sweep changes — but **a direct `python -m rlx.train` on an
already-finished run now fails instead of overwriting it**, which is the
`CLAUDE.md` §5 contract ("a directory that exists is a directory that finished").
If any of your tooling relies on re-running over a finished directory, it needs
to delete the directory first, deliberately.

**API changes in `rlx.analysis.stats`** — relevant to figures (task 5) and the
report scaffold (task 6):

| change | effect |
|---|---|
| `within_instance_correlation(df, col, seed=0)` | new `seed` argument; two new columns `rho_ci_low`, `rho_ci_high` |
| `aggregate_correlation` | new key `ci_excludes_zero`; **`confirms_h1` now requires the difficulty trend to be positive too**, per spec §1 |
| `aggregate_correlation` degenerate branch | now returns the same keys as the normal branch — it used to omit `confirms_h1` and crash `report.py` |
| `rank_stability` | ranks by **IQM**, not mean (spec §7.4). **Tau values differ from before on 7 of 13 instances.** |
| `early_auc` | normalises by the window, not the snapshot span. **Values are ~12% lower than before.** Rank-invariant, so no correlation changes. |
| `build_analysis_table` | raises once naming **every** unusable run, instead of aborting at the first |
| new: `compare_coverage_predictors(df, seed=0)` | the H2 test — larger correlation **and** non-overlapping CIs |
| new: `performance_profile(df, taus=None, seed=0)` | spec §7.2, returns `taus`, `profiles`, `ci_low`, `ci_high` per strategy |

**For task 5 (figures), three consequences:**

- **fig5 was under-specified.** Spec §7.5 asks for "rliable IQM with CIs, **plus
  performance profiles**"; the plan in `05-figures.md` only draws IQM bars. Use
  `performance_profile` for the second panel.
- **fig4 can now draw per-instance CIs.** `within_instance_correlation` returns
  `rho_ci_low` / `rho_ci_high`, which is what distinguishes a solid per-maze
  result from a coincidence.
- **fig6 will look different** from anything rendered before 2026-08-19, because
  the ranking is now IQM-based. That is the fix, not a regression.

**Known limitation, deliberately not papered over.** The synthetic fixture cannot
confirm H2: its effect is baked into raw coverage only, with no extra signal for
task-relevant coverage, so `compare_coverage_predictors` correctly returns
`confirms_h2=False` on it. The comparison has unit tests covering both outcomes
(`test_h2_is_confirmed_...`, `test_h2_is_not_confirmed_...`), but the end-to-end
"confirmed" path will first be exercised on real results. Fixing this would mean
making raw and task-relevant coverage diverge in the generator — and on the Empty
family they are identical by construction (ratio 1.00), so it cannot be done
uniformly. Left as is.

---

## Task 5 done, 2026-08-19 — seven figures, and what deviates from the plan

```bash
python -m rlx.analysis.figures --results results_synthetic --out scratch/figs
```

Suite is **208 passed, 1 xfailed**. `src/rlx/analysis/figures.py` is 397 lines —
over the ~200-line guidance in `CLAUDE.md` Rule 6, kept as one file because it is
seven deliverables plus six small helpers, and `CLAUDE.md` §4 specifies exactly
one `figures.py`.

**Deviations from `05-figures.md`, all deliberate:**

| what | why |
|---|---|
| `make_all_figures` raises `ValueError`, not `SystemExit` | `SystemExit` in a library function kills a caller's process; the house style is `ValueError` (`early_auc`, `_score_matrices`) |
| fig2 uses a bootstrap CI, not `sem` | the plan's own convention: every uncertainty is a bootstrap CI, never a standard deviation |
| fig4 has a **third panel** | per-instance rho with its bootstrap CI — the spec §7.3 step 2 numbers, and the panel that says which per-maze results are solid |
| fig5 has a **second panel** | performance profiles, required by spec §7.2 and §7.5 and missing from the plan |
| fig6 is three panels, not one axes | `difficulty` is grid size for Empty/DoorKey and room count for MultiRoom; one shared x axis would put MultiRoom-N6 beside DoorKey-6 |
| fig7 has no `suptitle` | "no titles inside the figure"; the instance identity is on the y-axis label instead, because the poster shows the figure without a caption |
| fig7 panels share one colour scale | with per-panel scaling every strategy looks equally thorough regardless of actual coverage — that would invert the figure's meaning |
| x axes are labelled in thousands (`80k`) | six-digit ticks collided into one unreadable run of digits |
| fig5 instances sorted by family then difficulty | alphabetical put DoorKey-10 before DoorKey-5 and Empty-16 before Empty-5 |

**For task 6.** `report.py` needs nothing from `figures.py`; its smoke test is
appended to `tests/test_figures.py`, which now has a module-scoped `synthetic`
fixture and a `rendered` fixture. Reuse `synthetic`; do **not** re-render the
figures for a report test.

**Caveat on fig7, to check against real data.** On the synthetic fixture all four
heatmaps look nearly identical, because the generator fills almost every reachable
cell by the last snapshot regardless of strategy. That is the fixture, not the
figure. If the real runs behave the same way, the honest fix is to draw an
**earlier** snapshot, not to rescale the picture.

---

## Two follow-ups after the task 5 review, 2026-08-19

Both on `analysis/figures`. Suite **220 passed**.

- **`make_all_figures` now validates before drawing.** It called `_score_matrices`
  only inside fig5, so an incomplete run matrix left fig1–fig4 on disk beside
  stale fig5–fig7 from an earlier render. Reproduced by deleting
  `DoorKey-8/noisy/seed3`; covered by
  `test_nothing_is_written_when_the_run_matrix_has_a_hole`.
- **`06-report-scaffold.md` amended before task 6 starts.** Its draft printed
  `confirms_h1` under the label "CI excludes zero", which stopped being true when
  `confirms_h1` began requiring the difficulty trend as well. The label now uses
  `ci_excludes_zero` and the hypothesis verdict is a separate line. The banner at
  the top of that file also points task 6 at `compare_coverage_predictors` for
  H2, at `performance_profile`, and at the new per-instance CI columns.

**Still open, not Daniel's:** the Boltzmann `tau` decision (`config.py` + spec
§5.2) and the unverified 14x claim on the count-based bonus.

---

## The pilot ran, 2026-08-19 — pipeline works end to end, three figure fixes

First real data through the whole chain. `configs/pilot.yaml` with
`--workers 4`: 16 ok, 0 failed, ~2 min. Suite now **223 passed**.

**Confirmed working:** `snapshot_every: 1000` gives 20 snapshots per run and
`early_auc` computes for all 16. **The tau calibration works** — Boltzmann was
0.064 against epsilon-greedy's 0.637 before, and is now level with it.

**Three figure defects, all invisible on the synthetic fixture** because it always
has all 13 instances and all 3 families:

| defect | cause | fix |
|---|---|---|
| fig1/fig2 drew an empty third panel and lost the legend inside it | panels came from `FAMILIES`, not from the data | `_present_families(df)` + `_family_axes` |
| `FutureWarning` from matplotlib | one instance per family hands it one-element Series (fig4 forest, fig6) | `.to_numpy()` |
| fig6's DoorKey panel silently blank | all four strategies scored 0.0, so tau is undefined | `_degenerate_note` labels it "no variance" |

The third one matters for the real sweep: if nothing solves DoorKey-10 or
MultiRoom-N6, those panels would have been blank in the report with no
explanation.

**Read pilot `final_return` with care — it is not a final return.** It averages
the last five evaluation points, and the pilot only has four in total, so it
averages the whole curve including the pre-learning zeros. One Empty-5 run ends at
**0.955** and is reported as **0.477**. Real runs have 80 eval points, so the
definition is fine there; we did not special-case it.

**Two observations from the pilot, NOT conclusions** — 20k steps is 5% of the real
budget, 2 seeds, 2 easy mazes:
- `count_based` scores 0.000 on Empty-5 where the other three reach the same value.
- `early_auc_raw` is highest for **epsilon_greedy** (0.774 on Empty-5) and lowest
  for `count_based` (0.283) — the opposite of the project hypothesis. At
  `epsilon ~ 1.0` the baseline is near-random and covers ground fast, so this is
  what one would expect this early. Worth re-checking on the real sweep.
