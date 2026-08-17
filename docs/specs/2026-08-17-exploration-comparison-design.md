# Design: Comparing Exploration Strategies in DQN

- **Date:** 2026-08-17
- **Status:** Approved
- **Authors:** Samuel Kostiuk, Max Bullach, Daniel Gleim (with Claude)
- **Deadline:** 2026-08-23

This document is the authoritative design. `CLAUDE.md` is the short operational
summary; where the two disagree, this document wins and `CLAUDE.md` should be
corrected.

---

## 1. Research question and hypothesis

**Question.** Does the choice of exploration strategy in DQN actually matter, and
if so, why?

**Approach.** Hold one DQN implementation completely fixed and swap only the
exploration module. Compare four strategies across a range of environments with
increasing exploration difficulty.

**Central hypothesis (H1).** Strategies that achieve higher *state coverage*
early in training achieve higher *final return*, and this relationship
strengthens as environments become harder to explore.

**Secondary hypothesis (H2).** *Task-relevant* coverage (states on or near the
optimal path, or adjacent to the key, door, or goal) predicts final return better
than *raw* coverage.

**Tertiary question (H3).** Is the ranking of strategies stable across difficulty?
Does the strategy that wins on `Empty` still win on `MultiRoom-N6`?

### What would confirm each hypothesis

Stated in advance so the analysis is not a fishing expedition.

- **H1 confirmed** if the within-instance rank correlation between early-coverage
  AUC and final return is positive with a bootstrap CI excluding zero, and the
  correlation is larger on harder instances than on easier ones.
- **H1 disconfirmed** if the CI includes zero, or if the correlation is flat or
  decreasing in difficulty.
- **H2 confirmed** if task-relevant coverage yields a larger correlation than raw
  coverage, with non-overlapping CIs.
- **H2 disconfirmed / interesting alternative** if both predict equally well.
  Conclusion then becomes "breadth of exploration matters, directedness does not"
  — a genuine result, just a different one.
- **H3** is descriptive, not a hypothesis test. Report Kendall's tau; low tau at
  high difficulty is itself the interesting finding.

---

## 2. How the professor's feedback maps onto this design

Every point in `context/proposal_response.md` is addressed. This table exists so
the report can point at it.

| Feedback point | Where it is handled |
|---|---|
| Define state coverage carefully; note it is privileged | §6, §7. Agent sees only 7x7x3; `(x,y,dir)` used only for analysis, stated explicitly. |
| Turn the hypothesis into a quantitative test (early-coverage AUC vs final return) | §7.3. Within-instance Spearman with bootstrap CIs. |
| Count-based bonus can use cheap tabular counts | §5.3. Counts the agent's own observations rather than true state — deviation, justified in §5.3. |
| NoisyNets is a clean drop-in | §5.4. |
| Boltzmann needs a stated temperature schedule | §5.2. Exponential decay 1.0 to 0.05 over the first 40% of training. |
| Say how intrinsic bonuses interact with evaluation return | §4.4. Evaluation is greedy, extrinsic-only, bonus never enters it. |
| Make difficulty continuous (5-6 instances) | §3. DoorKey sizes {5,6,7,8,10}, MultiRoom N {2,3,4,5,6}. |
| Decompose coverage into raw vs task-relevant | §6.2, §6.3. |
| Check rank stability across difficulty | §7.4. Kendall's tau. |
| Commit to >=5 seeds and specific variants/sizes | §3, §4.2. Seeds 0-4; exact sizes listed. |
| Say what result would confirm the hypothesis | §1. |
| Specify the DQN variant and keep it fixed | §4.1. Double DQN, fixed everywhere. |

---

## 3. Environments

MiniGrid via `gymnasium`. Environments are built by **direct class
instantiation**, not registered gym IDs, because the continuous difficulty axis
needs sizes that are not all pre-registered.

| Family | Instances | `env_id` | Difficulty axis |
|---|---|---|---|
| Empty | size 5, 8, 16 | `Empty-5`, `Empty-8`, `Empty-16` | grid size |
| DoorKey | size 5, 6, 7, 8, 10 | `DoorKey-5` ... `DoorKey-10` | grid size |
| MultiRoom | N = 2, 3, 4, 5, 6 | `MultiRoom-N2` ... `MultiRoom-N6` | room count |

13 instances. With 4 strategies and 5 seeds: **260 runs**.

`Empty` is the sanity check — if a strategy fails here, it is broken, not
outperformed. `DoorKey` requires a correct action *sequence* (find key, pick it
up, open door, reach goal). `MultiRoom` is the sparse-reward stress test.

Alongside each instance the env factory exposes a **difficulty index** for
plotting: the family parameter (grid size or room count), plus the count of
reachable `(x, y, dir)` states, which doubles as the coverage denominator (§6.1).

### UNVERIFIED assumptions about the MiniGrid API

Nothing is installed yet, so the following are from memory and **must be verified
by the day-1 API-verification script before any code depends on them**:

- `minigrid.envs.EmptyEnv(size=n)`
- `minigrid.envs.DoorKeyEnv(size=n)`
- `minigrid.envs.MultiRoomEnv(minNumRooms=n, maxNumRooms=n, maxRoomSize=s)`
- `minigrid.wrappers.ImgObsWrapper` yields the 7x7x3 `image` field
- `env.unwrapped.agent_pos` -> `(x, y)`, `env.unwrapped.agent_dir` -> `int` in 0..3
- `env.unwrapped.grid` is walkable-queryable for the reachability BFS
- `MultiRoomEnv` still ships in the current `minigrid` release

If any of these are wrong, fix the factory and record the correction in
`docs/decision_log.md`. Do not work around a failing import by guessing a second
API.

---

## 4. The agent

### 4.1 Algorithm: Double DQN, fixed

Double DQN everywhere, for every strategy and every environment.

The justification is specific to this project, not generic. Vanilla DQN
systematically **overestimates** action values, because the max operator in the
target picks up noise as well as signal. Overestimated Q-values behave like an
accidental optimism bonus, which is *itself a form of exploration*. That would
directly contaminate the thing being measured. Double DQN decouples action
selection from action evaluation in the target and largely removes this bias.

Not dueling: it changes the architecture without addressing any confound here,
and Rule 6 says less code wins.

Target:

```
y = r + gamma * (1 - done) * Q_target(s', argmax_a Q_online(s', a))
```

### 4.2 Fixed hyperparameters

Identical across every strategy and environment. Any deviation invalidates the
controlled comparison.

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
total_steps        400_000        # provisional, confirmed after day-1 benchmark
eval_every         5_000 steps
eval_episodes      1              # see 4.5 -- evaluation is deterministic
snapshot_every     10_000 steps   # see 6.4 -- sets early-AUC resolution
seeds              0, 1, 2, 3, 4
```

`train_freq = 4` means one gradient update per four environment steps. This is
standard DQN and cuts compute roughly fourfold relative to updating every step.

**The step budget is identical for every instance.** Scaling the budget with
difficulty would confound difficulty with compute, and the difficulty curve —
the headline result — would become uninterpretable. Hard instances scoring zero
is a finding.

### 4.3 Observation and network

Standard partial **7x7x3** egocentric view via `ImgObsWrapper`. The three
channels are integer indices (object type, colour, state), normalised before
entering the network.

Network: 3 convolutional layers (16, 32, 64 channels, 2x2 kernels, ReLU),
flatten, one 64-unit fully-connected layer, then a linear head to 7 action
values. Roughly 10^5 parameters.

Action space: all 7 MiniGrid actions, unrestricted. Restricting to the 5 useful
ones would speed learning slightly, but adds a wrapper and a source of bugs, and
the same 7 actions apply to every strategy, so the comparison stays fair.

### 4.4 Evaluation protocol

Every `eval_every` steps:

- Build a **separate** environment instance on the **same pinned layout**
  (`layout_seed = cfg.seed`). It must be the same maze — that maze is the task
  being scored. Isolation from training comes from the separate instance, not
  from a different layout.
- Run `eval_episodes` episodes with **greedy** action selection (`argmax Q`).
- Intrinsic bonuses are **off**. NoisyNets noise is **off** (mean weights used).
- Record mean and standard deviation of **extrinsic** return only.

Intrinsic bonuses exist solely inside the replay buffer. They shape learning;
they are never part of any reported score. This is the professor's question about
bonus/evaluation interaction, answered structurally rather than in prose.

### 4.5 Pinned layouts, and why `eval_episodes = 1`

MiniGrid regenerates its layout on every `reset()`: DoorKey re-places the wall,
key, and door; MultiRoom builds a fresh room chain. Left alone, this makes state
coverage undefined — with a different maze each episode there is no fixed
denominator to take a fraction of.

**Every run therefore pins one layout for its whole life**, by passing the run's
seed to every `reset()`. The 5 seeds give 5 different layouts per instance, so no
result is an artefact of one lucky maze. A run's seed thus controls layout,
network initialisation, and exploration randomness together — which is the normal
meaning of "seed", but worth stating.

Two consequences follow, and both are deliberate:

- **Evaluation is deterministic.** A pinned layout plus greedy action selection
  plus a deterministic simulator means all evaluation episodes of a given
  checkpoint are byte-identical. Running 10 would produce 10 copies of one
  number, so `eval_episodes = 1`. Evaluation gets 10x cheaper at zero
  information cost.
- **Per-run learning curves are step functions.** A run sits near 0 and jumps
  once it cracks the maze. Smooth aggregate curves come from averaging across
  seeds, not from within a run. Spread in `final_return` among solved runs comes
  from path efficiency, since MiniGrid's return already encodes solution speed.

The scope cost, which §11 records and the report's limitations section must
state: this measures exploration of a *single* maze, not generalisation across
mazes. That is the right target for this project — the hypothesis is about
covering an environment — but it is a real limitation, not a free choice.

---

## 5. Exploration strategies

All four implement the frozen `Explorer` interface in
`src/rlx/exploration/base.py` (reproduced in `CLAUDE.md` §6).

Design note: the interface has four methods and one class attribute. It covers
action selection, reward shaping, per-step bookkeeping, and logging — which is
exactly the union of what the four strategies need, and nothing more.

### 5.1 Epsilon-greedy (baseline)

With probability epsilon act uniformly at random, otherwise greedily.

Schedule: linear decay from `1.0` to `0.05` over the **first 20%** of
`total_steps`, constant at `0.05` thereafter.

### 5.2 Boltzmann (softmax) exploration

Sample the action from `softmax(Q / tau)`. High temperature approaches uniform;
low temperature approaches greedy. Unlike epsilon-greedy, exploratory actions are
weighted by how good the agent thinks they are.

Schedule (the professor asked for this explicitly): **exponential** decay of
`tau` from `1.0` to `0.05` over the first **40%** of `total_steps`, constant
thereafter.

```
tau(t) = max(tau_min, tau_0 * (tau_min / tau_0) ** (t / (0.4 * total_steps)))
```

Numerical note: subtract `max(Q)` before exponentiating, or small `tau` overflows.

### 5.3 Count-based bonus

Maintain a tabular visit count `N(k)` over count keys. Add an intrinsic bonus to
the reward stored in the replay buffer:

```
bonus = beta / sqrt(N(k))        beta = 0.05
```

Action selection is epsilon-greedy with a **fixed small** `epsilon = 0.05`, so
exploration pressure comes from the bonus rather than from randomness, while the
agent still cannot get stuck in a fully deterministic loop.

**Deviation from the feedback, stated explicitly.** The feedback suggests "cheap
tabular state counts on MiniGrid", which would naturally mean counting
`(x, y, dir)`. This design counts the **agent's own 7x7x3 observation** instead,
keyed on its raw bytes.

Reason: `(x, y, dir)` is privileged information the agent never receives. Using
it for the bonus would give one of the four strategies information the other
three do not get, breaking the controlled comparison that the whole project rests
on. Hashing the observation is what pseudo-count methods do in practice, and it
keeps a clean line: **`(x, y, dir)` is used only for analysis, never by any
agent.** That line is precisely what the feedback asked to be made explicit.

Consequence to acknowledge in the report: distinct observations are a coarser
partition than distinct states — two different positions can produce an identical
7x7x3 view (this is called *perceptual aliasing*). This weakens the bonus
somewhat. That is the honest price of not cheating, and it should be named.

`beta = 0.05` is provisional; if the bonus visibly dominates the extrinsic reward
(MiniGrid returns are in [0, 1]) it must be reduced, and the change recorded in
`docs/decision_log.md`.

### 5.4 NoisyNets

Replace the two fully-connected layers of the head with `NoisyLinear` layers
using factorised Gaussian noise, `sigma_0 = 0.5`. Action selection is purely
greedy — exploration comes from the network's own weight noise, and the amount of
noise is learned rather than scheduled.

- Noise is resampled once per environment step, before action selection.
- Noise is **disabled** during evaluation (mean weights only).
- **Documented deviation:** the paper also resamples noise for the online and
  target networks on every gradient update. We do not, because that would require
  a strategy-specific branch inside the training loop and the controlled
  comparison depends on that loop being identical for all four strategies.
  Exploration is unaffected — the noise driving action selection is still
  resampled every step. This belongs in the report's limitations section.
- `uses_noisy_net = True` tells the agent builder to construct the noisy head.
  This is the only place a strategy touches the network, and it is the reason
  that flag exists on the interface.

---

## 6. Coverage measurement

### 6.1 What is logged during training

Each run maintains a cumulative integer array of shape `(W, H, 4)` counting
visits to every `(x, y, direction)`, taken from `env.unwrapped.agent_pos` and
`agent_dir`. The array is snapshotted every `snapshot_every` steps into
`visitation.npz` as `steps (T,)` and `counts (T, W, H, 4)`.

Sizes are trivial: `16 * 16 * 4 = 1024` integers per snapshot, 40 snapshots per
run — about 1 KB each before compression.

**No coverage metric is computed during training.** All metrics are derived from
these snapshots later. This is a deliberate separation: metric definitions can be
changed on the last day without re-running a single experiment.

**Denominator.** A breadth-first search over the static grid layout gives the set
of reachable cells; multiplied by 4 directions this is the reachable-state count.
Computed once per environment instance and cached.

### 6.2 Raw coverage

```
raw_coverage(t) = |distinct (x,y,dir) visited by step t| / |reachable (x,y,dir)|
```

A number in [0, 1]. "How much of the environment has this agent ever seen?"

### 6.3 Task-relevant coverage

The same numerator, restricted to states that matter for solving the task.

Definition, computed from the grid layout:

1. Compute a BFS distance field to each landmark. For `Empty` the landmark chain
   is start -> goal; for `DoorKey` it is start -> key -> door -> goal; for
   `MultiRoom` it is start -> goal.
2. A cell is **task-relevant** if it lies on some shortest path through that
   landmark chain, or is within 1 cell of one, or is adjacent to a key, door, or
   goal.
3. Task-relevant coverage is distinct visited task-relevant states divided by
   total task-relevant states.

The `DoorKey` chain is the important detail: a shortest path from start straight
to goal is not the task, because the door is shut.

### 6.4 Early-coverage AUC

The predictor variable in the central test. For a coverage curve `c(t)` sampled
at snapshot points over the first `k = 0.2 * total_steps` steps:

```
early_auc = trapezoidal_integral(c, 0, k) / k
```

Normalising by `k` puts it in [0, 1] and makes it comparable across instances.
Computed for both raw and task-relevant coverage.

**`snapshot_every` sets the resolution of this integral**, and this is the only
reason it is 10_000 rather than something coarser. With `total_steps = 400_000`
the early window is 80_000 steps, so 10_000 gives 8 snapshots inside it and
20_000 would give only 4. A 4-point trapezoid on the project's main predictor is
needlessly imprecise when the extra resolution costs about 20 KB per run. **If
`total_steps` changes after the day-1 benchmark, re-check that at least ~8
snapshots still fall inside `frac * total_steps`** and adjust `snapshot_every`
accordingly.

---

## 7. Analysis

### 7.1 Response variable

`final_return` = mean greedy evaluation return over the **last 5 evaluation
points** of a run. Averaging the tail rather than taking the final point reduces
evaluation noise.

MiniGrid returns are already in [0, 1] (success gives `1 - 0.9 * steps/max_steps`,
failure gives 0), so no normalisation is required for `rliable`.

### 7.2 Aggregate strategy comparison

Using `rliable`, per environment family and per instance:

- **IQM** (interquartile mean) of final return with stratified bootstrap CIs.
  IQM discards the top and bottom 25% of runs, so it is far less swayed by one
  lucky or unlucky seed than a plain mean.
- **Performance profiles** — the fraction of runs exceeding each return
  threshold, which shows the whole distribution rather than one summary number.
- **Probability of improvement** for each pair of strategies.

### 7.3 The central test (H1 and H2)

**This is the load-bearing analysis and it has a trap in it.**

Both coverage and return decline as difficulty rises. Correlating them across all
260 runs pooled together yields a large, meaningless positive correlation — it
measures "hard environments are hard" twice and calls the agreement a result.

Correct procedure:

1. For each of the 13 environment instances **separately**, compute the Spearman
   rank correlation between `early_auc` and `final_return` across that instance's
   20 runs (4 strategies x 5 seeds). Difficulty is constant within an instance,
   so it cannot manufacture the effect.
2. Bootstrap over seeds to get a CI on each per-instance correlation.
3. Aggregate the 13 per-instance correlations, and additionally **plot
   correlation against difficulty index** — H1 predicts the correlation grows
   with difficulty.
4. Repeat the whole procedure for raw and for task-relevant coverage, and compare
   (H2).

Any analysis function that pools runs from different instances into one
correlation is wrong and must be rejected in review.

### 7.4 Rank stability (H3)

For each instance, rank the four strategies by IQM final return. Compute
Kendall's tau between each instance's ranking and the ranking on the easiest
instance of its family. Plot tau against difficulty.

Kendall's tau measures how similar two orderings are: 1.0 is identical order,
-1.0 is exactly reversed, 0 is unrelated.

### 7.5 Figures

Produced by `src/rlx/analysis/figures.py` into `report/figures/`:

1. Learning curves — return vs steps, 4 strategies, CI bands, per family.
2. Final return vs difficulty — 4 lines per family. *The difficulty curve.*
3. Coverage curves — raw and task-relevant vs steps.
4. **Early-coverage AUC vs final return** — scatter with per-instance regression.
   *The central result.*
5. `rliable` IQM with CIs, plus performance profiles.
6. Rank stability — Kendall's tau vs difficulty, or a rank bump chart.
7. **Visitation heatmaps** — per strategy on one instance, summed over direction.
   *The best poster figure.*

Figures 4 and 7 are the poster centrepieces.

---

## 8. Compute and execution

### 8.1 Device choice is an open question, resolved by measurement

The networks here are tiny (~10^5 parameters, batch 32). For networks this small,
GPU per-update cost can be dominated by kernel-launch overhead and host-device
transfers, and CPU can win. Additionally, each environment step is pure Python
inside MiniGrid and cannot move to the GPU at all.

**This is a prior, not a measurement.** Day 1 begins with a benchmark: the same
short run on CPU and on CUDA, reporting environment steps per second. The result
sets `device` in the configs, and the measured steps/second sets the final
`total_steps`.

Independently of the outcome: with 260 independent runs, throughput is maximised
by many parallel worker processes across the three machines, not by one job at a
time on one GPU.

### 8.2 Sweep execution

```bash
python -m rlx.sweep --config configs/main.yaml --shard 0/3 --workers 8
```

- `--shard i/n` — machine `i` of `n` takes every `n`-th run. Three machines, three
  shards, no coordination needed.
- `--workers` — parallel worker processes on this machine.
- Runs whose result directory already exists are **skipped**, so sweeps are
  resumable and safe to re-launch after a crash.
- Result directories are written atomically (`seed<k>.partial/` then rename), so
  a directory that exists is a run that finished.

Results merge across machines by copying directories. There is no shared state
and no possibility of conflict.

---

## 9. Testing strategy

`pytest`, mirroring the source layout. The tests that actually matter:

**Core (A)**
- All 13 environment instances construct, reset, and step.
- Reachability BFS returns the hand-computed answer on a small fixture grid.
- A 2000-step training run completes and writes a schema-valid result directory.
- Same seed twice produces identical `metrics.csv`.

**Exploration (B)**
- Epsilon schedule hits its documented endpoints at step 0, at 20%, and at the end.
- Boltzmann approaches `argmax` as `tau -> 0` and approaches uniform as `tau` grows.
- Boltzmann does not overflow for small `tau` (the max-subtraction is present).
- Count-based: repeated keys increase `N`, bonus decreases monotonically, bonus is
  never `inf` on first visit.
- NoisyLinear: two forward passes with resampled noise differ; with noise
  disabled they are identical.
- Every strategy returns an action in `range(7)` for arbitrary Q-value inputs.

**Analysis (C)**
- Coverage on a synthetic visitation array equals the hand-computed value.
- Task-relevant set on a fixture grid matches the hand-computed set.
- Early AUC of a known curve matches the analytic integral.
- `aggregate` loads a synthetic results tree into the expected DataFrame shape.
- **A regression test that the correlation is computed within instances, not
  pooled** — construct data where pooled and within-instance correlations have
  opposite signs, and assert the within-instance sign.

**B and C both develop against synthetic data** — fake Q-values and fake
visitation arrays — so neither is blocked if A slips.

---

## 10. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| MiniGrid API differs from §3 assumptions | Medium | Day-1 verification script before anything depends on it. |
| DQN scores 0 on all hard instances, no ranking to report | Medium | Expected and acceptable. Coverage still differentiates strategies, and "all methods fail past N4" is a legitimate finding. Empty and DoorKey-5/6 guarantee a working signal. |
| 400k steps too slow for the deadline | Medium | Budget is set *after* the day-1 benchmark, not before. Reducible to 200k without redesign. |
| Integration on day 3 fails | Medium | Runnable skeleton exists on day 0; daily rebase and evening PR merges make integration continuous. |
| One person falls behind | Medium | Synthetic-data development means no workstream hard-blocks another. Core is the only critical path. |
| Count bonus scale wrong (`beta`) | Medium | Checked in the pilot sweep; `mean_bonus` is logged in `metrics.csv` precisely so it can be compared against extrinsic return. |
| Results too large for git | Low | Only kilobytes per run. 260 runs is a few MB. |

---

## 11. Explicitly out of scope

- Any algorithm other than Double DQN.
- Any exploration strategy beyond the four named.
- Hyperparameter tuning per strategy. Hyperparameters are fixed; tuning them per
  strategy would destroy the controlled comparison.
- Environments outside MiniGrid.
- Distributed or cloud compute. Three local machines.
- A general-purpose RL framework. This is one experiment, not a library.
- **Generalisation across layouts.** Layouts are pinned per run (§4.5), so this
  work measures exploration of a single maze. Whether a strategy produces
  policies that transfer to unseen layouts is a different question and is not
  asked here.
