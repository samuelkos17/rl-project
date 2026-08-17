# STATUS — who is on what, right now

**Update this whenever you finish a task, before you open the PR.** It is the one
place the other two look to answer "can I start yet?".

Last updated: **2026-08-18**, by Max.

---

## Where everyone is

| | Workstream | Done | Currently on | Next |
|---|---|---|---|---|
| **Samuel** | A — Core & Infrastructure | ✅ 1 scaffold, ✅ 2 verify+benchmark | — | 3 env factory |
| **Max** | B — Exploration strategies | ✅ 1 epsilon-greedy | — | 2 boltzmann |
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
| `EpsilonGreedy` | `rlx.exploration.epsilon_greedy` | Samuel (training loop), Max |
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
| Daniel 3 (coverage) | `grid_info`, `reachable_mask`, `bfs_distances` | Samuel 3 | ⏳ blocked |
| Max 4 (NoisyNets) | `QNetwork`, `NoisyLinear` placeholder | Samuel 4 | ⏳ blocked |
| Samuel 5 (training loop) | `RunLogger` | Daniel 1 | ✅ **unblocked** |

Neither blocked task stops anyone today — Max has three tasks before he needs
Samuel 4, and Daniel has two before he needs Samuel 3. If a block does bite,
stub it locally rather than waiting; both task files say how.

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
