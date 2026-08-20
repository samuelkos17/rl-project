# 5. Coverage measurement

This section defines the explanatory variable of the study. Section 6 uses it to
ask whether exploring more of an environment early predicts scoring better in it
later; here we say precisely what "more of an environment" means, how it is
recorded, and what the agent is and is not allowed to know about it.

## 5.1 The agent's view and the analyst's view

The two are deliberately different, and keeping them apart is what makes the
comparison valid.

**What the agent receives** is the standard MiniGrid egocentric observation: a
7x7x3 array describing the cells in front of it, wrapped by `ImgObsWrapper` and
passed to a three-layer CNN. It contains no coordinates, no compass, and no map.
Partial observability is retained on purpose rather than replaced by a fully
observable wrapper, because the exploration problem we are studying is largely
created by it.

**What the analysis uses** is the true underlying state, the triple
`(x, y, direction)`, where `direction` is one of the four orientations the agent
can face. During training, every environment step increments a counter for the
agent's true `(x, y, direction)` in a `(W, H, 4)` integer array, and the array is
snapshotted every 10,000 steps into `visitation.npz`.

**This is privileged information that no agent ever receives.** It is used for
measurement only: it is read after training has finished, it never enters an
observation, a reward, a replay buffer or a gradient, and no quantity derived
from it is fed back into any run. Coverage over `(x, y, direction)` is the
meaningful notion of "how much of this environment has been seen" — an agent that
has stood in every cell facing every direction has genuinely been everywhere,
whereas a count over 7x7x3 observations would not tell us that (Section 5.5).
Using the true state for that purpose is legitimate, but it must be stated
plainly, and this paragraph is that statement.

No coverage metric is computed during a run. Training writes raw counts and
nothing else; every metric below is derived afterwards from the saved arrays.
This separates the definition of coverage from the cost of collecting it, so a
definition can be revised without re-running any experiment.

## 5.2 Raw coverage

Raw coverage at snapshot `t` is the fraction of attainable states the agent has
visited at least once:

```
raw_coverage(t) = |distinct (x, y, dir) visited by step t| / |loggable (x, y, dir)|
```

The denominator is obtained by a breadth-first search over the static layout from
the agent's start cell, which yields the set of reachable cells; each contributes
four states. Doors are treated as passable, since a locked door can be opened
with the key and every cell behind it is genuinely reachable during a run.

One cell is deducted: **the goal**. The training loop records the agent's
position at the start of each step and the environment terminates the episode the
instant the agent moves onto the goal, so no run can ever record a visit there,
in any of the four orientations. This was verified on all 16 pilot runs — the
goal counter is zero in every one of them, including the two runs that solved
Empty-5 with a return of 0.955. Leaving those four states in the denominator
would have capped coverage at `1 - 1/reachable`, that is 0.857 on DoorKey-5 and
0.889 on Empty-5, so that a perfectly thorough agent could not reach 1.0. They
are excluded from numerator and denominator alike.

The result lies in [0, 1] and answers: *how much of this environment has this
agent ever seen?*

## 5.3 Task-relevant coverage

Raw coverage treats an irrelevant corner and the corridor to the key as equally
worth visiting. Task-relevant coverage restricts the same numerator and
denominator to the states that matter for solving the task.

A cell is task-relevant if it lies on some shortest path through the landmark
chain of its environment family, or within one cell of such a path, or is a
landmark itself:

| Family | Landmark chain |
|---|---|
| Empty | start → goal |
| DoorKey | start → **key** → **door** → goal |
| MultiRoom | start → goal |

The DoorKey chain is the substantive case. A shortest path from start straight to
goal is not the task, because the door is shut; the agent must detour to the key
first, and a coverage measure that ignored this would score the detour as wasted
exploration. MultiRoom also reports a door, but its doors are plain openings that
any shortest path already crosses, so they are not treated as waypoints.

Cells on a shortest path from A to B are identified without enumerating paths: a
cell `c` lies on some shortest A→B path exactly when
`dist(A, c) + dist(c, B) == dist(A, B)`, which costs two breadth-first searches
per segment.

The two denominators, per instance, over the five layouts the seeds produce:

| Instance | Loggable states | Task-relevant states | Ratio (seeds 0-4) |
|---|---|---|---|
| Empty-5 | 32 | 32 | 1.00 |
| Empty-8 | 140 | 140 | 1.00 |
| Empty-16 | 780 | 780 | 1.00 |
| DoorKey-5 | 24 | 24 | 1.00 |
| DoorKey-6 | 48 | 44–48 | 0.92–1.00 |
| DoorKey-7 | 80 | 64–80 | 0.80–1.00 |
| DoorKey-8 | 120 | 76–120 | 0.63–1.00 |
| DoorKey-10 | 224 | 104–216 | 0.46–0.96 |
| MultiRoom-N2 | 40–128 | 32–96 | 0.62–0.90 |
| MultiRoom-N3 | 100–156 | 64–100 | 0.59–0.83 |
| MultiRoom-N4 | 136–192 | 112–160 | 0.65–1.00 |
| MultiRoom-N5 | 192–244 | 160–192 | 0.79–0.92 |
| MultiRoom-N6 | 220–296 | 176–264 | 0.80–0.95 |

Two features of this table matter for reading Section 6. First, **on the whole
Empty family the ratio is exactly 1.00 for every seed**: start and goal sit in
opposite corners of an open grid, so every reachable cell lies on some shortest
path and the task-relevant mask is the reachable mask. On those three instances
the two measures are not merely similar, they are the same number, and they
cannot distinguish between the two hypotheses of Section 6.5. Second, the ratio
varies with the layout as well as the instance, most widely on DoorKey-10 (0.46
to 0.96), so it is a property of the individual maze rather than of the family.
Only the wider-ranging DoorKey and MultiRoom layouts put real distance between
the two measures.

## 5.4 Early-coverage AUC

The hypothesis concerns coverage *early* in training, so the predictor is a
single number summarising the first part of the coverage curve rather than its
final value. For a coverage curve `c(t)` sampled at the snapshot points, the
early-coverage AUC over the first `k = 0.2 * total_steps` steps is

```
early_auc = trapezoidal_integral(c, 0, k) / k
```

Dividing by the window width places the result in [0, 1] and makes it comparable
across instances and step budgets. At the settings used here — 400,000 steps per
run, snapshots every 10,000 — the window is 80,000 steps wide and contains eight
snapshots, the eighth falling exactly on its edge. The snapshot interval was
chosen for this reason: at 20,000 the window would hold four points, which is a
needlessly coarse trapezoid for the study's principal predictor.

Curves are integrated from step 0, where coverage is taken to be zero. No run has
a snapshot at step 0, and normalising instead by the span of the observed
snapshots — 10,000 to 80,000 rather than 0 to 80,000 — inflates the result. By how
much depends on the shape of the curve, because integrating from the origin also
adds a first trapezoid that partly offsets the smaller divisor: 12.5% on a
straight line, 10.9% on a saturating curve of the kind these runs produce, 6.7%
on one already flat. Treating the origin as zero understates the result by the
agent's single starting state, whose effect on the integral is about 0.2%.

The AUC is computed for both coverage definitions, giving `early_auc_raw` and
`early_auc_task` for every run.

## 5.5 Why the count-based bonus counts observations, not states

One of the four strategies, the count-based bonus, needs a notion of "have I been
here before". It would be natural to reuse the `(x, y, direction)` state that
Section 5.1 describes, and it would be wrong.

`(x, y, direction)` is privileged information. Handing it to one of the four
strategies would give that strategy a source of knowledge the other three do not
have, and any advantage it then showed would be an artefact of the extra
information rather than a property of count-based exploration. The comparison is
controlled only if all four strategies see exactly the same thing.

The bonus therefore counts the agent's own 7x7x3 observation, taken as raw bytes:

```
count_key = observation.tobytes()
```

This is the honest version of the method under partial observability, and it is
weaker than a tabular state count would be, because distinct positions can
produce identical views. That weakness is a real property of the method in this
setting rather than a defect of our implementation, and Section 8.1 quantifies
how severe it is on each environment.

Finally, the bonus never reaches a reported number. It is added to the reward
stored in the replay buffer and nowhere else. Evaluation is greedy, uses
extrinsic reward only, disables NoisyNets noise, and runs on the same pinned
layout the run trained on; `final_return` is the mean evaluation return over the
last five evaluation points of a run.

---

*Reproducing the numbers in this section.* The denominator table comes from
`rlx.envs.reachable_mask` and `rlx.analysis.coverage.task_relevant_mask`
evaluated on `grid_info(env_id, layout_seed=seed)` for seeds 0–4, each then
passed through `rlx.analysis.coverage._loggable`, which removes the goal's four
states. That last step is not optional: without it the first column reads 36,
144, 784 and 28 rather than the 32, 140, 780 and 24 printed above. The pilot
goal-counter check is recorded in `docs/decision_log.md` under "The goal square
was in the denominator and could never be reached". The 0.2% figure is derived in
`rlx.analysis.coverage` itself, as `0.5 * c(0) * snapshot_every / window`; on
Empty-5 that is 0.00174 in absolute terms, or 0.25% of an early-AUC of 0.70. The
three normalisation figures are under "The normalisation error is not one number"
in the same log, and reproduce exactly — 12.50%, 10.90% and 6.67% — from the
three curves that entry names. Everything else in this section is fixed in
`src/rlx/config.py`.
