# STATUS — who is on what, right now

**Update this whenever you finish a task, before you open the PR.** It is the one
place the other two look to answer "can I start yet?".

Last updated: **2026-08-18**, by Samuel.

---

## Where everyone is

| | Workstream | Done | Currently on | Next |
|---|---|---|---|---|
| **Samuel** | A — Core & Infrastructure | ✅ 1 scaffold, ✅ 2 verify+benchmark, ✅ 3 env factory, ✅ 4 network + buffer | — | 5 agent + training loop |
| **Max** | B — Exploration strategies | ✅ 1 epsilon-greedy, ✅ 2 boltzmann, ✅ 3 count-based | — | 4 noisy-nets (blocked on Samuel 4) |
| **Daniel** | C — Logging, metrics & analysis | ✅ 1 visitation logging, ✅ 2 aggregation | — | 3 coverage metrics |

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
  (within-instance Spearman 0.58–0.93, rising with difficulty), `--no-effect`
  has none (−0.22 to +0.15, no significance). Task 4 must pass **both**.
- **Task 4 requirement:** an instance where every run scores the same has an
  undefined within-instance correlation. Report those as "no variance, excluded",
  never as a silent `NaN`. This is a plausible real outcome for `DoorKey-10` and
  `MultiRoom-N6`, not just a fixture artefact.
- **`difficulty` is comparable only within a family** — grid size for
  Empty/DoorKey, room count for MultiRoom. Never sort or correlate across
  families on it; group by `family` first.

**For Samuel — one question from Daniel's task 2 review.** `final_return` is
defined in spec §7.1 as the mean of the **last 5 evaluation points**, but
`metrics.csv` holds one row per *logged* step. If `train.py` logs `loss` more
often than it evaluates, most rows carry an empty `eval_return_mean`.
`rlx.analysis.aggregate.final_return` now drops the empty entries before taking
the tail, so it is correct either way — but if you intend to log at a different
cadence than you evaluate, say so, because it changes how `metrics.csv` looks.

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
- `total_steps: 400000`, `snapshot_every: 10000`, `--workers 8`
- All 13 MiniGrid instances verified on minigrid 3.1.0
- **MultiRoom grids are 25x25 for every N** — do not hard-code 16

## Who is blocked on what

| Waiting task | Needs | From | Status |
|---|---|---|---|
| Max 1, 2, 3 | `RunConfig`, `Explorer` | Samuel 1 | ✅ **unblocked** |
| Daniel 1, 2 | `RunConfig`, `difficulty_index` | Samuel 1 | ✅ **unblocked** |
| Daniel 3 (coverage) | `grid_info`, `reachable_mask`, `bfs_distances` | Samuel 3 | ✅ **unblocked** |
| Max 4 (NoisyNets) | `QNetwork`, `NoisyLinear` placeholder | Samuel 4 | ✅ **unblocked** |
| Samuel 5 (training loop) | `RunLogger` | Daniel 1 | ✅ **unblocked** |

**Nothing is blocked any more.** Every task in the plan can now proceed.

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
