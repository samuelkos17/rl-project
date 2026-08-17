# Workstream A — Core & Infrastructure (Samuel)

> **For Claude sessions:** Read `CLAUDE.md`, then
> `docs/specs/2026-08-17-exploration-comparison-design.md`, then this file, then
> the numbered task file you are on. Work one task at a time, in order.

**Goal:** Everything that runs an experiment — build the environment, build the
network, train a Double DQN, write a result directory, and launch 260 of them
across three machines.

**Architecture:** One shallow package, `src/rlx`. The training loop owns the
interaction between agent, environment, explorer, and logger; every other module
is a passive component it calls. The explorer is injected, so swapping strategies
changes one line.

**Tech Stack:** Python 3.11, torch, gymnasium, minigrid, numpy, pyyaml, pytest.

**Spec:** `docs/specs/2026-08-17-exploration-comparison-design.md`

---

## Tasks in order

| # | File | Deliverable | Blocks others? |
|---|---|---|---|
| 1 | `01-scaffold-and-contracts.md` | package skeleton + 3 frozen contracts | **YES — merge before Max and Daniel start** |
| 2 | `02-verify-api-and-benchmark.md` | verified MiniGrid API, CPU-vs-GPU numbers, final step budget | yes (settles `total_steps`) |
| 3 | `03-env-factory.md` | `envs.py` — 13 instances, grid info, BFS | Daniel's coverage needs `bfs_distances` |
| 4 | `04-network-and-buffer.md` | `networks.py`, `buffer.py` | Max's NoisyNets needs `QNetwork` |
| 5 | `05-agent-and-training-loop.md` | `agent.py`, `train.py` — a full run writes a result dir | no |
| 6 | `06-sweep-runner.md` | `sweep.py` — sharded parallel launcher | no |

## What you must not do

- Do not tune hyperparameters per strategy. They are fixed for everyone.
- Do not give harder environments more steps.
- Do not let `(x, y, dir)` reach the agent. It goes to the logger only.
- Do not change a frozen contract without telling Max and Daniel.

## Definition of done for this workstream

```bash
python -m rlx.train --env-id DoorKey-5 --strategy epsilon_greedy --seed 0 --total-steps 20000
```

completes, writes a schema-valid `results/DoorKey-5/epsilon_greedy/seed0/`, and

```bash
python -m rlx.sweep --config configs/main.yaml --shard 0/3 --workers 8
```

runs a third of the matrix in parallel, skipping anything already done.
