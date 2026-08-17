# STATUS — who is on what, right now

**Update this whenever you finish a task, before you open the PR.** It is the one
place the other two look to answer "can I start yet?".

Last updated: **2026-08-17**, by Samuel.

---

## Where everyone is

| | Workstream | Done | Currently on | Next |
|---|---|---|---|---|
| **Samuel** | A — Core & Infrastructure | ✅ 1 scaffold, ✅ 2 verify+benchmark | — | 3 env factory |
| **Max** | B — Exploration strategies | — | not started | 1 epsilon-greedy |
| **Daniel** | C — Logging, metrics & analysis | — | not started | 1 visitation logging |

## What is available on `main` right now

Everything below is merged and safe to import.

| Symbol | From | Who needs it |
|---|---|---|
| `RunConfig` | `rlx.config` | Max, Daniel |
| `Explorer` (ABC) | `rlx.exploration.base` | Max |
| `make_explorer`, `STRATEGIES` | `rlx.exploration` | Max |
| `ENV_IDS`, `difficulty_index` | `rlx.envs` | Daniel |

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
| Samuel 5 (training loop) | `RunLogger` | Daniel 1 | ⏳ blocked |

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
