# Decision log

Every real change we make, in order, newest at the bottom. Including the ones we
threw away — for those, the "why discarded" part is the whole point.

Format is described in `docs/README.md`. Terms are explained in
`docs/glossary.md`.

---

## 2026-08-17 — Project design settled

**Status:** Active

**What changed:** We turned the proposal plus the professor's feedback into a
concrete design, written up in
`docs/specs/2026-08-17-exploration-comparison-design.md`. We also created
`CLAUDE.md` so every Claude session starts with the same context and the same
rules.

**Why:** We have 6 days and three people working at once. Without a written
design and fixed interfaces, three parallel Claude sessions will build three
incompatible halves of the same thing.

**What it means for the results:** Nothing directly — this is setup.

---

## 2026-08-17 — We do all three of the professor's extensions

**Status:** Active

**What changed:** The proposal had three fixed mazes. We are instead using 13
mazes across three families, with size or room count as a smooth difficulty axis.
We also split coverage into "raw" and "task-relevant", and added a rank-stability
check.

**Why:** The professor suggested all three, and the person who wrote the feedback
is the person who grades the report. They are also cheap: all three reuse the
same visitation logs we were already collecting, so the extra cost is analysis
code rather than new infrastructure or extra experiments.

**What it means for the results:** This is what turns the report from "here is a
ranking" into "here is a ranking *and* an explanation, with a curve". It is the
main thing that makes the project interesting.

---

## 2026-08-17 — Double DQN, not vanilla DQN

**Status:** Active

**What changed:** We committed to Double DQN as the single algorithm, fixed
across all strategies and mazes. The professor asked us to pick one and stick to
it.

**Why:** Plain DQN systematically overestimates how good actions are. Being
overly optimistic about untried actions makes the agent try them — which is
*itself* a form of exploration. Since our entire project is about comparing
exploration strategies, an accidental extra exploration effect baked into the
algorithm would contaminate every comparison we make. Double DQN mostly removes
it. It also costs about two lines of code more than vanilla.

We did **not** add dueling DQN, because it does not fix any confound we have and
just adds moving parts.

**What it means for the results:** It makes the comparison between the four
strategies cleaner. Worth one sentence in the report.

---

## 2026-08-17 — Count-based bonus counts observations, not true positions

**Status:** Active — and this is a deliberate deviation from the feedback

**What changed:** The professor suggested the count-based strategy could use
"cheap tabular state counts", which most naturally means counting how often the
agent has been at each `(x, y, direction)`. We are instead counting how often the
agent has *seen* each 7x7x3 view.

**Why:** The agent never gets told where it is. `(x, y, direction)` is
information only we can see (this is what "privileged information" means). If we
let the count-based strategy use it, that one strategy would be running on better
information than the other three, and the comparison would no longer be fair —
which is the entire point of the setup.

Counting what the agent actually sees keeps a clean line that we can state in one
sentence in the report: **`(x, y, dir)` is used only for analysis, never by any
agent.** That is exactly the distinction the professor asked us to be explicit
about, so we think this deviation serves the feedback rather than ignoring it.

**What it means for the results:** One honest downside we should name in the
report. Two different spots in a maze can look identical through a 7x7 window
(this is called *perceptual aliasing*), so counting views is a blurrier measure
than counting positions, and the bonus is somewhat weaker as a result. That is
the price of not cheating, and we should say so rather than hide it.

---

## 2026-08-17 — Same step budget for every maze

**Status:** Active

**What changed:** Every one of the 13 mazes gets the same training budget
(provisionally 400,000 steps), instead of giving harder mazes more time.

**Why:** It is tempting to give the hard mazes more steps so they have a chance.
But then difficulty and training time change together, and the headline plot —
performance against difficulty — becomes impossible to read. You would not know
whether performance dropped because the maze got harder or because the budget got
relatively smaller.

**What it means for the results:** The hardest mazes (MultiRoom with 5 or 6
rooms) will probably score zero for every strategy. **This is a result, not a
bug.** "Every method we tested collapses beyond 4 rooms" is a perfectly good
finding, and the coverage numbers still tell us how the strategies differ even
when none of them succeed.

---

## 2026-08-17 — Local files for results, no Weights & Biases

**Status:** Active

**What changed:** Each run writes its own self-contained folder with a config
file, a metrics CSV, a visitation array, and some metadata. No experiment
tracking service.

**Why:** Three machines producing results independently. With self-contained
folders, merging results is literally copying directories — there is no shared
state, so there is nothing to conflict. Setting up W&B accounts for three people
costs time we do not have, and the visitation arrays are awkward to store there
anyway. Our analysis needs plain arrays on disk regardless.

**What it means for the results:** Nothing. Just plumbing.

---

## 2026-08-17 — Coverage metrics are computed after training, not during

**Status:** Active

**What changed:** During a run we only save the raw visitation counts. Every
actual coverage number is computed later from those saved snapshots.

**Why:** It means we can change the *definition* of coverage on the last day
without re-running a single experiment. Given a 6-day deadline, being able to fix
a metric without redoing 260 runs is worth a lot.

**What it means for the results:** It de-risks the analysis, which is the part of
the project most likely to need a late change.

---

## 2026-08-17 — Each run gets one fixed maze, and evaluation runs 1 episode

**Status:** Active

**What changed:** MiniGrid normally builds a brand new maze every single episode —
DoorKey moves the wall, key and door around, MultiRoom generates a whole new
chain of rooms. We switched that off. Each run now sees the same maze from start
to finish, chosen by that run's seed, so our 5 seeds give us 5 different mazes
per setting.

**Why:** We only noticed this while writing the implementation plan, and it would
have broken the project quietly. Our central measurement is "what fraction of the
maze did the agent visit". If the maze is different every episode, there is no
fixed thing to take a fraction of, and the number is meaningless. We would have
produced coverage numbers all week without realising they meant nothing.

**What it means for the results:** Two knock-on effects, both fine.

First, evaluation becomes completely predictable. Same maze, plus always picking
the action the agent thinks is best, plus a simulator with no randomness in it,
means every evaluation episode comes out exactly the same. So we run 1 instead of
10. That is not less information — it is the same information, ten times cheaper.

Second, a single run's learning curve becomes a step: it sits at 0 for a long
time and then jumps once the agent finally solves the maze. Smooth curves come
from averaging our 5 seeds together, not from within one run. Among runs that do
solve it, the scores still vary, because MiniGrid pays more for solving faster.

**The honest limitation, which belongs in the report:** this means we are
measuring how well a strategy explores *one* maze, not whether what it learns
carries over to mazes it has not seen. That is the right question for us — our
hypothesis is about covering an environment — but it is a real limitation and we
should name it rather than let someone else spot it.

---

## 2026-08-17 — CPU vs GPU is an open question, to be measured

**Status:** RESOLVED 2026-08-17 — the benchmark ran; see "Benchmark result: the processor beats the graphics card" below. CPU won.

**What changed:** Nothing yet. Recording that we do not know the answer.

**Why:** The proposal assumed runs go on the GPU. Our network is very small
(around 100,000 parameters, batches of 32), and for networks this small the GPU
can actually be *slower*, because the overhead of shipping work to the graphics
card outweighs the tiny amount of arithmetic. On top of that, stepping the
MiniGrid maze itself is plain Python and cannot run on the GPU at all.

**We have not measured this.** It is a reasonable expectation, not a fact, and we
should not build a plan on it. Day 1 starts with a short benchmark: the same run
on CPU and on GPU, measuring steps per second. That result sets the `device`
setting and also tells us what step budget we can afford.

One thing is true either way: we have 260 completely independent runs, so the
fastest approach is many runs side by side across our three PCs, not one run at a
time on one GPU.

**What it means for the results:** Nothing scientific. It decides how long we
wait.

---

## 2026-08-17 — Package skeleton and the three frozen interfaces

**Status:** Active

**What changed:** Created the `rlx` package, the run configuration, and the
exploration-strategy interface. These three things are now "frozen", meaning we
agreed not to change them without telling each other. Max and Daniel can start.

**Why:** All three of us are writing code at the same time. If we each invented
our own idea of what a config looks like, nothing would fit together on
integration day. Agreeing on the shapes first means we can work independently.

**What it means for the results:** Nothing. Plumbing.

**Environment we ended up with:** Python 3.11.15 in a conda env called `rl`.
Notable versions, because they are newer than the plan assumed: minigrid 3.1.0,
torch 2.13.0, numpy 2.4.6, pandas 3.0.5, rliable 1.2.0. The plan was written
against minigrid 2.x, so task 2's API check matters more than we thought — see
the next entry.

---

## 2026-08-17 — pip installed a CPU-only torch, which task 2 must know about

**Status:** RESOLVED 2026-08-17 — see "Benchmark result: the processor beats the graphics card"

**What changed:** Nothing yet. Recording a trap we walked into and spotted.

**What happened:** `pip install torch` on Windows quietly gives you a build with
no graphics-card support at all (it calls itself `2.13.0+cpu`). Torch happily
reports "no GPU available" — not "you installed the wrong version".

**Why this matters:** task 2 is supposed to measure whether our runs are faster
on the processor or on the graphics card. With this build, that measurement
cannot happen: it would report "no GPU" and we would write down "the GPU is
unusable", which is simply false. Samuel's machine has an RTX 3060 Ti.

**What to do about it:** before running the benchmark in task 2, decide one of:
(a) reinstall torch with graphics-card support using
`pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124`
and then benchmark both properly, or
(b) consciously decide we are running on the processor only, and write that in
the report as a choice rather than pretending we measured it.

Either is defensible. Silently benchmarking a CPU-only build and calling it a
CPU-vs-GPU comparison is not. There is a note about this at the top of
`requirements.txt` so nobody hits it by surprise.

**What it means for the results:** Only how long we wait for the sweep. It does
not change any number in the report.

---

## 2026-08-17 — Benchmark result: the processor beats the graphics card, budget stays 400k

**Status:** Active — resolves the open entry "pip installed a CPU-only torch"

**What changed:** We installed the proper graphics-card version of torch
(`2.13.0+cu130`, matching the CUDA 13.3 driver) and measured both properly
instead of guessing. Then we chose the processor anyway.

**The numbers**, same 3,000-step loop on both:

| device | speed | time for one 400k-step run |
|---|---|---|
| processor (CPU) | 630 steps/s | 10.6 min |
| graphics card (GPU) | 550 steps/s | 12.1 min |

The graphics card is about 13% **slower**.

**Why, and this is worth understanding:** we first assumed the graphics card
would not help because stepping the maze is plain Python that a graphics card
cannot touch. We measured that too, and **we were wrong about the reason** —
stepping the maze is only 20% of the time; the neural network is the other 80%.

The graphics card still loses, for a different reason. Our network is tiny and we
ask it for one single decision at a time. Sending one tiny job to a graphics card
costs more in overhead than just doing the arithmetic on the processor. Graphics
cards win on big batches, and DQN's "look at one situation, pick one action"
pattern is the opposite of that.

So: right conclusion, wrong reason, and we only know that because we measured.

**Parallel speed, which is what actually decides our schedule:**

| workers | each | total | efficiency |
|---|---|---|---|
| 1 | 760 steps/s | 760 | 100% |
| 4 | 660 | 2,641 | 87% |
| 8 | 503 | 4,022 | 66% |
| 10 | 364 | 3,635 | 48% |
| 12 | 377 | 4,527 | 50% |

Total speed stops improving after about 8 workers (10 is actually worse than 8).
**Use `--workers 8`.**

**What this means for the schedule:** the full 260 runs is 104 million steps.
One machine at 8 workers: about 7.2 hours. Our three machines: **about 2.4
hours.** That fits comfortably, so we keep the full 400,000-step budget rather
than cutting it.

**Settings now fixed in `configs/main.yaml`:** `total_steps: 400000`,
`device: cpu`. Not provisional any more — measured.

---

## 2026-08-17 — All 13 mazes verified against the real MiniGrid, plus two surprises

**Status:** Active

**What changed:** Ran `scripts/verify_api.py`, which builds all 13 mazes and
checks every assumption the plan made about how MiniGrid works. **All 13 passed**
on minigrid 3.1.0 — newer than the 2.x the plan was written against.

Confirmed working: the 7x7x3 view, 7 actions, reading the agent's true position
and facing, and — most importantly — that asking for the same seed twice rebuilds
exactly the same maze. That last one is what our whole coverage measurement rests
on.

Also confirmed: `Empty` mazes are the same regardless of seed (expected, they
have no random parts), while `DoorKey` and `MultiRoom` do change with the seed.
That is what gives us 5 different mazes across our 5 seeds.

**Surprise 1: MultiRoom mazes are 25x25 no matter how many rooms.** The plan
assumed they grew with room count and that grids were at most 16x16. They are not.
This makes our visit-count arrays bigger than written down (2,500 numbers instead
of 1,024), which is still small, but we corrected the spec and Daniel's task so
nobody hard-codes 16.

**Surprise 2, and this one could have quietly ruined the project:** MultiRoom
gives the agent very little time — only 20 steps per room, so 120 steps for the
6-room maze, against 640 for DoorKey-8. We checked whether the goal is even
*reachable* in that time, because if it is not, those mazes would score zero for a
reason that has nothing to do with exploration, and our whole difficulty curve
would be measuring the wrong thing.

It is fine. Worst case we found needs roughly 37-53 steps against a 120 limit.
So when the hard mazes score zero — and they probably will — that will genuinely
be because exploring is hard, which is exactly what we are trying to measure.

## 2026-08-17 — Visit logging, and why results are written atomically

**Status:** Active

**What changed:** During a run we now record where the agent actually is —
its x, y position and which of the 4 directions it faces — and count how often it
has been in each of those situations. We save a snapshot of those counts every
10,000 steps.

**Why snapshots rather than one final total:** our main question is whether
exploring a lot *early* predicts doing well *later*. That needs coverage measured
over time, not just at the end.

**Why we compute no coverage percentages during the run:** we only save raw
counts. Every actual metric is worked out afterwards from the saved files. That
means if we decide on the last day that our definition of coverage was wrong, we
fix the analysis instead of re-running 260 experiments.

**Why the result folder is written in a funny way:** a run first writes to a
folder ending in `.partial`, and only renames it to the real name once it has
finished successfully. The sweep runner skips any run whose folder already
exists, so if a crashed run left half a folder behind, that experiment would be
skipped forever and we would quietly lose a data point without noticing. This way,
a folder existing always means a run that genuinely finished.

**What it means for the results:** Nothing scientific. It stops us losing data.

## 2026-08-17 — Daniel's machine runs Linux with pyenv instead of conda

**Status:** Active

**What changed:** `CLAUDE.md` says the environment is built with conda. Daniel's
machine is Linux and has no conda, so his environment is a plain Python virtual
environment built on a pyenv-installed Python 3.11.15 instead. Same Python
version, same packages from `requirements.txt`, same test results (16 passed).

**Why:** Nothing in the project actually depends on conda — it only needs
"Python 3.11 plus the listed packages". Installing conda purely to match the
instruction would have cost time and changed nothing.

**The one trap we hit, and it matters for anyone else on Linux:** on Windows,
`pip install torch` gives you the processor-only version, which is what we want.
**On Linux the default is the opposite** — pip pulls the graphics-card version
plus about 2.5 GB of NVIDIA libraries. That is the build our own benchmark showed
to be *slower* for this project, and `requirements.txt` warns against it, but the
warning was written assuming Windows. The fix is to install torch explicitly from
the processor-only package index first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.
Verified afterwards: `torch.__version__` reports `2.13.0+cpu` and
`torch.cuda.is_available()` is `False`.

**What it means for the results:** Nothing, as long as everyone ends up on a
processor-only torch — which is exactly what the fixed `device: cpu` setting
assumes. If someone silently ran the graphics-card build, their runs would be
slower but produce the same numbers.

## 2026-08-17 — We generate fake results to build the analysis against

**Status:** Active

**What changed:** Wrote a script that produces fake experiment folders in exactly
the same format as real ones. All of the analysis code is developed and tested
against these before a single real experiment exists.

**Why:** Three of us are working at once and the real results do not arrive until
the 21st. Without fake data, the analysis work could not start until then, and we
would be debugging plots and statistics on the last two days with no slack.

**The useful part:** the fake data has a known answer built into it — strategies
that explore more early are *made* to score better, and more so on harder mazes.
So when the statistics in task 4 run, we already know what number they should
produce. If they produce something else, the analysis is broken, and we find that
out on the 17th instead of the 22nd.

**What it means for the results:** Nothing goes into the report from this. It is
scaffolding, and the fake folders are gitignored.

## 2026-08-17 — The first version of the fake data had no answer in it

**Status:** Active (supersedes the formula in `implementation_plan/daniel/02-aggregation.md` step 1)

**What changed:** The first version of the generator decided by coin flip whether
a run solves the maze. The chance of solving was worked out as
`1.3 x early_exploration - 0.7 x difficulty`, and then squashed into the range
0.02 to 0.97 so it stays a valid probability.

For the two harder families that sum came out **negative for all four
strategies** — DoorKey between -0.02 and -0.11, MultiRoom between -0.39 and
-0.46. Everything negative gets squashed to the same lower limit of 0.02. So all
four strategies ended up with an identical 2% chance, and with 5 seeds each,
none of them ever solved. Four of the six fake environments contained nothing but
zeros.

We replaced the coin flip with a score that depends smoothly on how much the run
explored early, measured **relative to the plain epsilon-greedy baseline at that
same difficulty**. That last part is the fix: the amount of early exploration
naturally shrinks as mazes get harder, so comparing against one fixed number
punished the hard mazes twice over and pushed them off the bottom of the scale.

**Why it mattered:** the whole point of fake data is that we know the answer in
advance. Our hypothesis is tested *within* each environment separately (see
`CLAUDE.md` section 9). An environment where every run scores exactly 0.000 has
nothing to correlate — the statistics would have returned "not a number" for four
of six environments, and the one claim we most need to check, that the effect
gets stronger on harder mazes, could not have been checked at all.

**Measured before and after** (correlation between exploration and final score,
within each environment; 1.0 is a perfect match, 0.0 is no relationship):

| Environment | before | after |
|---|---|---|
| Empty-5 | not defined (all zeros) | 0.58 |
| Empty-8 | not defined (all zeros) | 0.63 |
| DoorKey-5 | not defined (all zeros) | 0.82 |
| DoorKey-8 | not defined (all zeros) | 0.90 |
| MultiRoom-N2 | not defined (all zeros) | 0.93 |
| MultiRoom-N4 | not defined (all zeros) | 0.90 |

The numbers now also rise from left to right across the difficulty families,
which is the pattern our hypothesis predicts.

**We also added a deliberately empty version.** `--no-effect` produces a second
fake dataset in which the score has *no* relationship to exploration at all. Run
on that one, the analysis must come back with nothing (measured: correlations
between -0.22 and +0.15, none of them statistically meaningful).

This second dataset is the honest half of the exercise. Without it we would only
ever be checking that our analysis can say "yes, there is an effect" — never that
it can say "no, there isn't". An analysis that says yes to everything is the more
dangerous failure, because it is the one that would put a false claim in the
report.

**Is building the answer into the data cheating?** No, and it is worth being
precise about where the line is. Tuning the *analysis* until the *real* data
gives the answer we like would be cheating. Making the *fake* data contain a
pattern, to check whether our measuring instrument detects a pattern it is
supposed to detect, is a calibration test. The fake data contains no evidence
about reinforcement learning whatsoever — we wrote every number in it ourselves,
so it can neither support nor refute the hypothesis.

**One real warning came out of this, and it is for task 4.** The broken version
accidentally simulated something that may well happen for real: if the hardest
instances — `DoorKey-10`, `MultiRoom-N6` — are solved by nobody within 400,000
steps, then their real results will genuinely be all zeros, and the within-
environment correlation for those instances really will be undefined. Section 7
of `CLAUDE.md` already says a hard instance failing is a finding rather than a
bug. Task 4 must therefore **report such instances explicitly as "no variance,
excluded"** rather than silently emitting "not a number" and letting it flow into
an average.

**What it means for the results:** Nothing directly — this is all scaffolding and
none of it is reported. Indirectly it decides whether we can trust the statistics
when the real numbers arrive on the 21st.

## 2026-08-17 — Full review of tasks 1 and 2, and the ten things it found

**Status:** Active

**What changed:** Before merging the aggregation work we went back over both
finished pieces — the visit logger from task 1 and the loaders plus fake-data
generator from task 2 — and checked them line by line against every rule we had
written down, then deliberately attacked the awkward cases instead of only
running the happy path. Ten problems came out. Seven are fixed, three are
knowingly left alone.

Everything that was supposed to be true still is: the logger offers exactly the
six functions Samuel's training loop calls, the result folders have the agreed
four files with the agreed shapes, and no coverage number is ever computed
during a run.

### The two real bugs

**A crashed run could take the whole analysis down with it.** When a run dies
halfway it leaves behind a folder ending in `.partial`. The loader searched for
folders starting with `seed` and matched those too, then tried to read the seed
number out of `seed3.partial` and crashed. Worse than the crash: a half-finished
run already contains all the files the loader checks for, because they are all
written *before* the folder is renamed. So even without the crash it would have
counted an unfinished experiment as a finished one. The loader now only accepts
a folder whose name is `seed` followed by digits and nothing else.

**The final score was quietly being averaged over the wrong thing.** Our agreed
definition is "the average of the last five evaluations of a run". What the code
actually did was "the average of the last five *lines* of the results file".
Those are only the same thing if the training loop writes a line exactly when it
evaluates. It probably will not — the file also holds the training loss, which
is produced far more often. Then most lines have an empty evaluation column,
and the last five lines might contain only one real evaluation. The code did not
crash and did not warn; it simply averaged one number instead of five and
returned a noisier answer than we asked for. It now skips the empty entries
first.

**Samuel needs to know about the second one.** How often `train.py` writes a
line decides whether this ever mattered in practice. It is fixed either way, but
he should be aware the two frequencies are not the same thing.

### Four ways the fake data was not actually fake data

The point of the generator is that the rest of the analysis can be built against
it without waiting for real experiments. It only earns that if it is
indistinguishable from real output. In four ways it was not.

**The mazes were all the same size, and the wrong size.** Every fake run used a
12x12 grid. The real ones are 5x5, 8x8 and — for every MultiRoom — 25x25. Task 3
has to lay the visit counts over a map of which squares are actually reachable,
and a 12x12 array does not line up with a 5x5 map. The fixture would have been
unusable for the very next task. Sizes are now taken from the real environment
names.

**The visit counters could go down.** Real counters only ever increase — the
logger adds one and copies. The generator drew fresh random numbers for every
snapshot, so 37% of squares showed *fewer* visits later than earlier. Coverage
happened to survive this, because coverage only asks "was this square ever
visited", and that set did grow correctly. But anything reading the actual
counts — a heat map, a visit-frequency measure — would have been wrong, and an
obvious sanity check on real data would have failed on the fake data. Counters
now accumulate.

**The saved settings file had 5 of its 26 entries.** Any later analysis that
reads a setting would work on real runs and fail on fake ones. It now writes the
genuine settings object.

**You could not tell the two fake datasets apart.** We produce a second dataset
with `--no-effect` in which exploration and score are unrelated, to check the
analysis does not invent a result out of nothing. Both datasets wrote
byte-identical settings and metadata files, and both default to the same output
folder. One absent-minded command would have replaced the good dataset with the
empty one, and the analysis would then have reported "no effect" — which looks
exactly like a finding rather than an accident. The metadata now records which
dataset it is.

### One more, on the boundary between the two pieces

**A run that finished without logging anything could not be read back.** The
logger writes an empty file in that case and the loader threw an error on it. We
fixed the reading side rather than the writing side: an empty file is an honest
description of "nothing was logged", and the rule that matters is that the
analysis must never fall over on a folder that exists. Such a run now loads with
an empty table and a final score of "not a number", which is the truthful
answer.

### Three we deliberately did not fix

Recorded because deciding not to act is also a decision.

**Negative coordinates wrap around silently.** Asking the logger to record
position -1 quietly increments the last square instead of complaining. MiniGrid
never produces a negative position, so this can only be reached by a bug
elsewhere, and guarding it would add a check to the one function that runs on
every single step of all 260 experiments. Positions that are too *large* already
fail loudly, which is the case that could plausibly happen.

**Overwriting an existing result folder has a one-instant gap.** If a finished
folder is already there, the logger deletes it and then renames the new one into
place. A crash exactly between those two operations loses the old data without
putting anything back. The sweep runner skips folders that already exist, so
this should never run at all; making it fully safe would mean a three-step
rename dance for a situation we have designed out.

**An empty list of runs produces a table with no columns**, and asking that
table for a column then fails with a confusing message. It only happens if you
point the analysis at a folder with no results in it. We would rather see the
confusing message than add a guard to every function.

### A trap that is nobody's bug

The `difficulty` column is **not comparable between families**. For Empty and
DoorKey it is the grid size; for MultiRoom it is the number of rooms. So Empty-16
carries a 16 and MultiRoom-N6 carries a 6, and sorting everything by that column
would claim the six-room maze is easier than an empty 16x16 room. It is not. Any
plot of "effect against difficulty" has to be drawn one family at a time. This
is now written into the function that builds the table, next to the column.

### One simplification that remains

The fake mazes have no walls — every square is treated as visitable, while a
real 5x5 Empty has a solid border and only 3x3 usable squares. This is harmless,
because task 3 measures coverage against the set of genuinely reachable squares
and anything outside it simply drops out of the calculation. It is written down
here so nobody rediscovers it as a bug.

**What it means for the results:** No number in the report comes from any of
this. Two of the fixes, though, are the difference between an analysis that
crashes or quietly misreports and one that does not, and four are the difference
between a practice dataset that supports the next three tasks and one that does
not.

**Measured after the changes:** 30 tests pass. On the dataset with the effect
built in, the within-maze link between exploration and score is 0.53 to 0.93 and
rises across the three families, as the hypothesis predicts. On the empty
dataset it is -0.17 to +0.14 with nothing statistically meaningful. Both
datasets regenerate byte-for-byte identically.

## 2026-08-17 — Two ways of measuring coverage

**Status:** Active

**What changed:** We now measure how much of a maze the agent visited, in two
different ways.

"Raw coverage" is the simple one: of all the places the agent could possibly
reach, what fraction did it actually stand in (counting the 4 directions it can
face as different situations)?

"Task-relevant coverage" only counts places that matter for solving the maze —
on or next to the shortest route, or right beside the key, the door, or the goal.
The idea is that wandering into an irrelevant corner is exploring, but not
*usefully* exploring.

**The DoorKey detail worth remembering:** for those mazes the route is
start → key → door → goal, not start → goal. The straight line to the goal goes
through a locked door, so it is not the actual task. We got this wrong in an
earlier sketch and it would have made task-relevant coverage meaningless on the
whole DoorKey family.

**A neat trick we used:** to find every cell lying on a shortest route from A to
B without listing all the routes, check whether (distance from A to that cell) +
(distance from that cell to B) equals the total distance from A to B. If it does,
the cell is on some shortest route. Two distance calculations instead of an
explosion of paths.

### Correction found while building this: the MultiRoom doors

The planned version of this code treated "the maze has a door" as "the route
must detour through that door". That is right for DoorKey, which has exactly one
door and locks it. It is wrong for MultiRoom, where the rooms are simply joined
by doorways: those doors are not obstacles, the shortest route already passes
through them, and the layout reader only remembers the *last* doorway it happens
to find. So the code would have bent the route towards one arbitrary doorway and
pulled a completely unrelated room into the "task-relevant" set.

The fix is one line of judgement: a door only counts as a waypoint if the maze
also has a key. Only DoorKey has one. There is a test that fails if anyone
reverts this.

### A limitation we should state in the report rather than hide

On the **Empty family the two coverage numbers are identical, by construction**,
for every seed. In an open room the start is one corner and the goal is the
opposite corner, and in an open grid *every* square between two opposite corners
lies on some shortest route. So "the useful part of the maze" is the whole maze,
and task-relevant coverage equals raw coverage exactly. This is not a bug and no
change to the code fixes it — it is what the definition means on an empty room.

Measured, one layout per instance, before any widening of the route:

| maze | reachable squares | on the route | + neighbours | final ratio |
|---|---|---|---|---|
| Empty-5 | 9 | 9 | 9 | 1.00 |
| Empty-8 | 36 | 36 | 36 | 1.00 |
| Empty-16 | 196 | 196 | 196 | 1.00 |
| DoorKey-5 | 7 | 7 | 7 | 1.00 |
| DoorKey-6 | 13 | 9 | 13 | 1.00 |
| DoorKey-7 | 21 | 10 | 17 | 0.81 |
| DoorKey-8 | 31 | 12 | 20 | 0.65 |
| DoorKey-10 | 57 | 17 | 27 | 0.47 |
| MultiRoom-N2 | 19 | 8 | 14 | 0.74 |
| MultiRoom-N3 | 29 | 11 | 23 | 0.79 |
| MultiRoom-N4 | 46 | 21 | 37 | 0.80 |
| MultiRoom-N5 | 53 | 26 | 44 | 0.83 |
| MultiRoom-N6 | 58 | 29 | 50 | 0.86 |

Two things to take from that table. First, the distinction between the two
coverage measures only does real work on the larger DoorKey mazes, where it
narrows the target to under half the maze. That is also where we expect the
hypothesis to show itself, so this is the useful case. Second, counting
neighbours of the route roughly doubles the target on MultiRoom (0.38 → 0.79 on
N3), so that "within 1 cell" rule is carrying a lot of weight — worth a sentence
in the report, since a stricter rule would give visibly different numbers.

**A separate thing this uncovered:** the Empty mazes looked, from this local
stub, like the *same layout* for all 5 seeds — start and goal sit in fixed
corners and there is nothing else to randomise, so the claim in `CLAUDE.md` §7
that "the 5 seeds give 5 layouts per instance" would not hold for this family.
Samuel confirmed this independently against his real `envs.py` and worked out
the consequence for the statistics — see "Environment factory built, and what
the mazes actually look like" (2026-08-18), Finding 1, below.

**What it means for the results:** These two numbers are what we correlate
against final performance. Which of the two predicts better is one of the three
questions the report answers — but on the Empty family the question cannot be
asked at all, because the two numbers are the same number.

**Measured after the changes:** 45 tests pass (15 new), against the local stub
of Samuel's task 3 that this entry's own note above describes. After the
2026-08-18 rebase onto his real `envs.py` (merged in "Environment factory
built", below), the same 15 coverage tests still pass, now as part of the full
165 passed, 1 xfailed suite.


## 2026-08-18 — Epsilon-greedy baseline implemented

**Status:** Active

**What changed:** Added our baseline exploration strategy. With probability
"epsilon" the agent throws away what it has learned and picks a random action;
otherwise it does what it thinks is best. Epsilon starts at 1.0 (everything
random, because at the start the agent knows nothing) and drops in a straight
line to 0.05 over the first fifth of training, then stays at 0.05 for the rest.

**Why:** It is the standard baseline that every other strategy gets compared
against, and it is deliberately the dumbest of the four.

**What it means for the results:** The 0.05 floor means the agent always keeps a
little randomness rather than becoming completely predictable. If a smarter
strategy cannot beat this one, that is a genuinely interesting finding, not a
bug.

**Two small things worth knowing:**

The random branch draws from **all seven actions**, including the one the agent
already thinks is best. So the true chance of acting randomly-and-differently is
slightly below epsilon. Both definitions are defensible; this is the standard
one, and a test pins it down so nobody "fixes" it later.

The decay length is worked out once when the strategy is built, but the start and
end values are read fresh on every step. That is not a principled distinction —
it is just how the code fell out — and one test leans on it by lowering the end
value after construction. Worth knowing before anyone edits this file.

**Measured after the change:** 38 tests pass (30 before, 8 new). The whole suite
runs in 5.79 seconds.

**Also noticed, not changed:** installing on Linux pulled the graphics-card
version of torch again — the same trap Daniel logged on 2026-08-17, now hit by a
second person, because `requirements.txt` does not say which version to fetch.
It costs 3.4 GB of disk and changes no result, since `device: cpu` is fixed. A
one-line pin in `requirements.txt` would stop the third person hitting it.

---

## 2026-08-18 — Boltzmann exploration implemented

**Status:** Active

**What changed:** Added the second strategy. Instead of "best action, or a
completely random one", it picks each action with a probability based on how good
the agent thinks it is. A second-best action gets chosen fairly often; an action
the agent rates as useless almost never does.

The "temperature" controls how picky it is: high temperature means nearly random,
low temperature means nearly always the best action. We start at 1.0 and shrink it
to 0.05 over the first 40% of training. Our professor specifically asked us to
write this schedule down, so it is in the spec too.

**One implementation detail worth recording:** the maths involves raising e to the
power of (Q divided by temperature). At low temperatures that number gets
astronomically large and the computer gives up and returns "infinity", which
poisons everything downstream. The standard fix is subtracting the largest Q-value
first, which changes nothing mathematically but keeps the numbers small. We have a
test for it, and we checked that the test genuinely catches the problem: without
the fix the probabilities come out as `[nan nan 0. 0. nan nan nan]`.

### A trap that is nobody's bug: temperature only means something relative to the Q-values

Epsilon-greedy does not care how large the Q-values are. If you multiply every
Q-value by ten, it behaves identically. **Boltzmann is not like that.** What
matters to it is the size of the *differences* between Q-values compared to the
temperature.

That matters here because MiniGrid's scores are small. We checked the installed
source: a successful episode scores `1 - 0.9 * (steps taken / step limit)`, so
between 0.1 and 1.0, and a failed one scores 0. So the gap between the best and
worst action can never exceed about 0.9, and early in training — when the network
is barely trained and almost nothing has been rewarded yet — it is far smaller
than that.

The practical effect, measured on the real 400,000-step budget:

| step | temperature |
|---|---|
| 0 | 1.00 |
| 20,000 | 0.69 |
| 40,000 | 0.47 |
| 80,000 | 0.22 |
| 160,000 and after | 0.05 |

At temperature 1.0, even with the largest gap the environment can produce, the
best action only gets picked about 26% of the time against 14% for a coin-flip
across all seven. In other words, early Boltzmann is close to picking at random.
It only starts genuinely favouring good actions once the temperature drops below
about 0.2, which happens at step 85,959 — **just after the 80,000-step window
closes that we measure early exploration over.**

**What this means for the results:** if Boltzmann turns out to have high early
coverage, we cannot immediately claim that its cleverness caused it — during most
of the window we measure, it is behaving a lot like uniform random. The report has
to say this. It is exactly the sort of thing our professor would ask about.

**We changed nothing.** The schedule is the one in the spec, and tuning a strategy
to make it look good is forbidden. This is written down as something to interpret,
not something to fix.

**Honestly uncertain:** how far apart the Q-values actually drift during training
is not something we can know before running the experiments. The numbers above use
the widest gap the environment allows, which is the most generous case for
Boltzmann. If the real gaps are smaller, it stays close to random for longer, not
less. Worth re-checking once we have real runs.

**Measured after the change:** 47 tests pass (38 before, 9 new).

---

## 2026-08-18 — Count-based exploration implemented

**Status:** Active. One open question for the team, at the bottom.

**What changed:** Added the third strategy. It keeps a tally of how often the
agent has seen each situation, and pays a small bonus for being somewhere
unfamiliar — a lot on the first visit, almost nothing on the hundredth. The agent
mostly acts greedily, with a constant 5% chance of a random action so it cannot
get stuck repeating itself forever.

**What it means for the results:** This is the only one of the four strategies
that changes the *reward*, not just the *action choice*. The bonus only ever
affects what the agent learns from — every score we report is the maze's real
reward, with the bonus switched off.

**It counts what the agent sees, not where it is.** The tally is keyed on the raw
7x7 view, never on the true position. Two different corners of a maze that look
identical through that window share a tally entry, which makes the bonus blurrier
than it would otherwise be. We accept that cost on purpose: giving this one
strategy the true position would mean racing a strategy that knows where it is
against three that do not, and the comparison would be worthless.

### Open question: the bonus is larger than the maze reward early on

The task file asked us to measure the total bonus collected over one episode and
compare it against the ~0.9 the maze pays for being solved, and **not** to change
anything without agreement. Here is the measurement. One 300-step episode with
100 distinct views, at the current setting of 0.05:

| how often each view has been seen before | bonus collected | discounted | vs maze reward |
|---|---|---|---|
| never (start of training) | 11.42 | 3.63 | 12.7x |
| 10 times | 4.34 | 1.38 | 4.8x |
| 100 times | 1.49 | 0.47 | 1.7x |
| 1,000 times | 0.47 | 0.15 | 0.5x |
| 10,000 times | 0.15 | 0.05 | 0.2x |

So at the very start the novelty bonus is worth about thirteen times solving the
maze, and it falls below the maze reward once every view has been seen roughly a
thousand times. Setting the bonus to **0.0039** instead of 0.05 would make the
first episode come out level with the maze reward.

**We have not changed it, and we should discuss before anyone does.** Two honest
readings, and we do not yet know which is right:

- *This is fine, possibly necessary.* These mazes pay nothing at all until the
  agent stumbles onto the goal for the first time. Until that happens the novelty
  bonus is the only signal it has to learn from. A bonus that starts large and
  fades as places become familiar is the intended behaviour of this method, not a
  fault in it.
- *This is too strong.* An agent paid thirteen times more for sightseeing than
  for finishing may keep sightseeing well after it knows where the goal is.

**One more thing we noticed but could not measure yet (UNVERIFIED):** how fast the
bonus fades depends on how many distinct views a maze contains. A small empty room
has few, so its tallies grow quickly and the bonus dies away early. A six-room
maze has many more, so its tallies stay low and the bonus stays strong for longer.
That means this strategy's intrinsic drive is automatically stronger in exactly
the harder environments — which is either precisely what we want or a confound we
have to declare, depending on how the results come out. We cannot check it without
running the environments, so it is written down rather than resolved.

`mean_bonus` goes into `metrics.csv` on every run specifically so this can be
checked against real return on integration day.

### A smaller trap, for whoever reads the logged numbers

`distinct_keys` counts every key the strategy has ever been *asked about*, not
only the ones it was told to record. Asking for the bonus of a never-visited
situation quietly adds it to the tally at zero. In real runs the training loop
asks about and records the same situation on the same step, so the two coincide
and the number is honest. It is only misleading if someone calls the bonus
function on its own, as our tests do.

**Measured after the change:** 56 tests pass (47 before, 9 new).

---

## 2026-08-18 — Environment factory built, and what the mazes actually look like

**Status:** Active — implements the 2026-08-17 entry "Each run gets one fixed maze"

**What changed:** Wrote the code that builds all 13 mazes, pins each run to one
layout, and works out which squares are reachable. 48 tests pass.

**The reachable-square counts**, which are the denominators every coverage
percentage in the report gets divided by:

| maze | grid | reachable squares | states (x4 facings) |
|---|---|---|---|
| Empty-5 / 8 / 16 | 5x5 / 8x8 / 16x16 | 9 / 36 / 196 | 36 / 144 / 784 |
| DoorKey-5 ... 10 | 5x5 ... 10x10 | 7 / 13 / 21 / 31 / 57 | 28 / 52 / 84 / 124 / 228 |
| MultiRoom-N2 ... N6 | 25x25 | 19 / 29 / 46 / 53 / 58 | 76 / 116 / 184 / 212 / 232 |

Empty-5 = 9 and Empty-8 = 36 match what we worked out by hand, which is the check
that the wall detection is right.

**Note for MultiRoom:** only 3-9% of the 25x25 grid is reachable, because the
rooms sit in a corner of a big canvas. Coverage is measured against reachable
squares, not the whole grid — otherwise every MultiRoom number would look
absurdly small for no real reason.

**Finding 1: `Empty` gives the same maze for all 5 seeds.** It has no random
parts — the agent always starts in the same corner and the goal is always in the
opposite one. DoorKey and MultiRoom do give 5 genuinely different mazes.

So for Empty, our 5 seeds differ only in how the network is initialised and how
exploration rolls its dice, not in the maze. That is fine — Empty is our sanity
check, not a real test — but it means **Empty's five runs are less independent
than the other families' five runs**, and the report should not present them as
if they were the same kind of replicate.

There is a knock-on effect Daniel should expect: because the maze is identical
*and* scoring is deterministic, every seed that solves Empty optimally gets the
**same** score. So Empty may well come out with little or no variation in final
score, which makes a within-maze correlation undefined there. That is the
"no variance, excluded" case Daniel already built handling for — we are telling
him it is likely to happen systematically on Empty, not just by bad luck.

**Finding 2: MultiRoom mazes vary a lot in size between seeds.** MultiRoom-N2
ranges from 11 reachable squares (seed 2) to 33 (seed 1) — three times bigger.
The room count is fixed but the room *sizes* are random.

Each run's coverage is divided by its own maze's size, so the percentages are
still correct. But it does mean the five runs of one MultiRoom setting are not
five attempts at the same difficulty — they are five attempts at noticeably
different mazes. Expect the MultiRoom correlations to be noisier than the DoorKey
ones, and say so in the report rather than treating it as a surprise.

DoorKey does not have this problem: the wall and key move around, but the number
of reachable squares is identical across seeds.

**What it means for the results:** Nothing is broken. Two things to write in the
report's limitations: Empty's seeds are weaker replicates, and MultiRoom's seeds
vary in maze size.

---

## 2026-08-18 — Measuring Max's count-bonus question against the real mazes

**Status:** OPEN — needs all three of us to agree, **before the sweep launches on
the 20th**

**What changed:** Nothing yet. Max flagged that the novelty bonus looks far too
big (`docs/decision_log.md`, "Count-based exploration implemented"). His estimate
used a made-up episode; we measured it on the actual mazes with a random policy.

**He was right, and the reason matters:**

| maze | episode length | novelty reward per episode | vs the 0.9 for winning |
|---|---|---|---|
| Empty-5 | 91 steps | 1.42 | 1.6x |
| DoorKey-8 | 640 steps | 12.71 | **14.1x** |
| MultiRoom-N4 | 80 steps | 1.41 | 1.6x |
| MultiRoom-N6 | 120 steps | 1.92 | 2.1x |

**The cause is episode length, not the bonus itself.** MiniGrid gives DoorKey-8
640 steps per attempt but MultiRoom-N4 only 80. The novelty bonus is paid per
step, so long-episode mazes accumulate roughly eight times more of it. One single
setting of `count_beta` therefore lands very differently across our 13 mazes.

**Why this is a problem for the report, not just for tuning:** on DoorKey the
agent would be paid roughly 14x more for sightseeing than for winning, so it may
simply never learn the task there. We would then report "count-based does badly
on DoorKey" when the honest statement is "our bonus size was wrong for DoorKey's
episode length". That is a conclusion about our settings masquerading as a
conclusion about the method.

**The awkward part: no single value fixes it.** Sized to make DoorKey balanced
(`beta` about 0.0035), the bonus becomes almost invisible on MultiRoom — the
hardest mazes, where we most want it working. Sized for MultiRoom, DoorKey stays
swamped. Episode lengths differ by 7x and the bonus cannot know that.

**Recommendation: `count_beta = 0.01`** (down from 0.05). The worst case drops
from 14x to about 3x, and the bonus stays meaningful on MultiRoom. It is a
compromise, chosen on this scale argument alone.

**The rule that matters more than the value:** we pick this **now, before any
real runs**, and we do not touch it again. Changing it after seeing which
strategy wins would mean choosing our result, which is exactly the fishing
expedition the spec was written to prevent. If 0.01 turns out badly, that is a
finding we report, not a number we quietly revise.

Whatever we choose goes in the report's limitations with these measurements
attached, because a reader can reasonably ask why 0.01 and not something else.

---

## 2026-08-18 — The neural network and the memory of past experiences

**Status:** Active

**What changed:** Built the two pieces the agent needs before it can learn: the
network that estimates how good each action is, and the replay buffer that
remembers past experiences to learn from. 16 new tests, 120 passing overall.

**The network:** it reads the agent's 7x7 view through three small
pattern-detecting layers, then two ordinary layers that output one number per
action. **76,599 numbers get adjusted during learning** — genuinely small, which
is exactly why the graphics card lost the benchmark on the 17th.

**The replay buffer** holds the last 100,000 experiences and uses **31 MB**. We
store the views as whole numbers rather than decimals, which costs a quarter of
the memory and loses nothing, because MiniGrid views are whole numbers to begin
with.

**One number worth checking, and we did:** the network divides every input by 10
to keep values in a sensible range. That only works if the raw values never go
much above 10. We measured all 13 mazes with a random agent: the largest value
ever seen was **8**, giving 0.8 after dividing. So inputs stay under 1 as
intended, in every maze.

**Something deliberately left unfinished:** `NoisyLinear` is currently a normal
network layer that does nothing special. It is a **placeholder** so the network
can be built and tested now. Max replaces its insides in his task 4 — the name
and the way it is called are fixed, so his change will drop straight in without
touching anything else. **Max is now unblocked.**

**Two safety tests worth knowing about**, because both catch mistakes that would
be invisible rather than loud:

- One checks that a remembered experience keeps its parts together — the right
  action with the right reward with the right next view. If those ever got
  shuffled, the agent would train on nonsense and nothing would crash.
- One checks the buffer never hands back empty slots while it is still filling.
  Early in training it holds only a few thousand of its 100,000 slots, and
  sampling blank ones would quietly teach the agent that doing nothing is fine.

**What it means for the results:** Nothing yet. This is machinery. The training
loop that uses it is next.

---

## 2026-08-18 — The agent and the training loop, and it actually learns

**Status:** Active

**What changed:** Built the Double DQN agent and the training loop that ties
everything together — maze, network, memory, exploration strategy, and Daniel's
logging. 139 tests pass, 1 expected-to-fail (see below).

**It works.** A 20,000-step run on Empty-5 went from a score of 0 to **0.955**,
which is close to the best possible. Full pipeline, end to end, for the first
time.

**We checked the two halves fit together.** Daniel's results loader read the
folder written by our training loop without a single adjustment: it found the
run, parsed the maze name, strategy and seed from the folder path, and produced
the tidy table his analysis expects. That was the biggest integration risk in the
whole project and it is now behind us.

**One number that looks wrong but is not.** The loader reported a final score of
0.239 for a run whose last measurement was 0.955. That is because "final score"
averages the **last five** measurements, and this short test run only had four —
three zeros and one 0.955. In a real 400,000-step run there are 80 measurements,
so the last five all come from after the agent has learned. Nothing to fix, but
worth knowing before someone panics at a low number in a short test.

**One test is deliberately marked as expected-to-fail:** the training loop cannot
run the NoisyNets strategy until Max merges his task 4. Rather than write a
reminder to delete the marker later — which someone would forget — the test
**checks whether his module exists** and quietly becomes a real test the moment
it lands. No cleanup step, nothing to remember.

**Speed, measured on the real loop rather than the benchmark:** about 500-560
steps per second, against 630 in the artificial benchmark from the 17th. The
difference is the visit logging and the exploration strategy, which the benchmark
did not include. Re-projecting the full 260-run sweep: roughly **3 hours across
our three machines** instead of 2.4. Still comfortable, no change needed.

**Three rules are enforced in code and marked `CRITICAL` in the file**, because
breaking any one of them would invalidate results without breaking anything
visibly:

1. The maze is pinned to the run's seed, so one run sees one maze.
2. The novelty bonus is added to the replay memory and nowhere else — scoring
   uses the maze's real reward only, with the bonus and any network noise
   switched off. A test checks scores never exceed MiniGrid's ceiling of 1.0,
   which is what would happen if the bonus leaked in.
3. What gets counted for the novelty bonus is the agent's own view, never its
   true position.

**One small change from the plan:** the Double DQN target calculation was pulled
out into its own function so it can be tested directly. There is now a test that
fails if anyone ever rewrites it as plain DQN — that single line is the whole
reason we chose Double DQN, and it would otherwise be very easy to "simplify"
back by accident.

---

## 2026-08-18 — Sweep runner, and a 3.4x speed bug caught by running it for real

**Status:** Active

**What changed:** Built the launcher that runs all 260 experiments in parallel
across our three machines. Then ran a 16-run pilot for real, which is where the
interesting part happened.

**The sweep works.** Each machine takes every third experiment from a fixed list,
so the three of us never need to coordinate. Anything already finished is
skipped, so a crashed sweep is resumed by simply re-running the same command.

**The failure handling is proven, not assumed.** Four of the pilot's 16 runs used
NoisyNets, which Max has not merged yet, so they failed. The other 12 finished
normally and the sweep reported `12 ok, 4 failed`. One broken experiment cannot
take down the other 259.

**The bug: we were using a quarter of the machine.** The pilot took 6 minutes 6
seconds. Our day-1 benchmark said it should take about 90 seconds.

The cause: PyTorch defaults to using 6 processor threads per run. That is
sensible for one big job, but we run 8 experiments side by side, so we had 48
threads fighting over 12 processor cores. They spent their time interrupting each
other.

We now tell each run to use exactly one thread. **The same pilot dropped from 6
minutes 6 seconds to 1 minute 48 — a 3.4x speed-up.** And measured on its own
with nothing else running, one thread is *still* faster than six (608 steps per
second against 495), because our network is so small that coordinating threads
costs more than it saves.

**Why our earlier benchmark missed it:** the day-1 benchmark script set one
thread explicitly, so it measured the fast case and the real training loop did
not. A benchmark that does not match what you actually run is worth very little.
Lesson recorded rather than just fixed.

**Use 12 workers, not 8.** With one thread each, more workers now helps. Measured
on the pilot: 4 workers 137 s, 8 workers 108 s, **12 workers 81 s**. Twelve
workers on twelve cores is the natural fit.

**Revised timing for the full sweep:** about **3 to 4 hours across our three
machines** (13 hours if one machine did it alone). Speed is much the same on
every maze — between 518 and 629 steps per second — because the agent always
sees the same small 7x7 view no matter how big the grid is.

**One caveat on the 12-worker number:** the pilot only has 16 experiments, so
with 12 workers almost everything runs at once and it never reaches a steady
state. The real sweep has 87 per machine. Watch the first few minutes of the real
run and drop to 8 workers if the rate looks worse than the pilot suggested.

---

## 2026-08-18 — The count-bonus test at 20k steps was inconclusive

**Status:** OPEN — the count_beta decision is still not settled

**What changed:** Nothing. Recording a test that did not answer the question, so
nobody repeats it.

**What we tried:** the pilot showed count-based scoring 0 on Empty-5 while
Boltzmann scored 0.955, which looked like confirmation that the novelty bonus is
too large. We then ran Empty-5 with four different bonus sizes, three seeds each.

**The result did not support the story:**

| count_beta | mean score over 3 seeds |
|---|---|
| 0.05 | 0.000 |
| 0.01 | 0.000 |
| 0.005 | 0.318 (one seed of three learned) |
| 0.001 | 0.000 |

No pattern. One run out of twelve learned, and it was not at the value we
proposed.

**Why the test was bad:** 20,000 steps is 5% of a real run. On that budget even
plain epsilon-greedy only managed 0.478, and that strategy has no bonus at all.
Everything was noise, so the pilot's `count_based = 0` is **not** evidence that
the bonus is too big — it is evidence that 20,000 steps decides nothing.

**We are not treating this as support for any value of `count_beta`.** Picking
0.005 because it produced the one non-zero number would be choosing a
hyperparameter from a single lucky seed, which is exactly the sort of thing the
report should never contain.

**What we are doing instead:** re-running the comparison at 100,000 steps, which
is a quarter of the real budget. That result goes in the next entry. The decision
still has to be made before the sweep launches.

---

## 2026-08-18 — The 100k-step test, with a proper control

**Status:** OPEN — two decisions needed before the sweep launches

**What changed:** Re-ran the bonus-size comparison at 100,000 steps (a quarter of
the real budget) instead of 20,000, and — the part we were missing before — also
ran the bonus-free strategies on the same budget as a **control**. Without that
control, count-based scoring zero somewhere tells us nothing, because we would
not know whether anything else scores above zero there either.

**All numbers are 3 seeds, score averaged over the last 5 measurements:**

| strategy | Empty-5 | DoorKey-5 |
|---|---|---|
| epsilon-greedy (no bonus) | **0.637** | 0.000 |
| Boltzmann (no bonus) | 0.064 | 0.000 |
| count-based, beta = 0.05 (current) | 0.126 | 0.000 |
| count-based, beta = 0.01 | **0.573** | 0.000 |
| count-based, beta = 0.005 | 0.127 | 0.000 |
| count-based, beta = 0.001 | 0.127 | 0.000 |

**First and most useful result: DoorKey-5 is unsolved by everything.** Every
strategy, every bonus size, every seed, all zero. So count-based's zeros there
were never a bonus problem — nothing learns DoorKey-5 in 100,000 steps. That is
what the control bought us, and it is why the earlier test was uninterpretable.

**Second: beta = 0.05 does look too big, and 0.01 looks right.** On the one maze
where anything learns, 0.01 scores 0.573 against 0.126 for our current 0.05 —
and 0.573 is in line with the bonus-free baseline of 0.637. That is exactly what
"the bonus should help exploration without drowning the task" should look like.

**How much weight to put on that: some, not a lot.** Three seeds, and the spread
within one setting is enormous (0.955, 0.000, 0.764 at beta = 0.01). A two-out-of-
three versus one-out-of-three difference is not a real measurement.

**So we are deciding on the scale argument, not on this table.** The reasoning
from 2026-08-18 — that a bonus worth 14x the value of winning cannot be right —
was made *before* any of these runs, and it points at the same answer. This table
is a sanity check that fails to contradict it, and that is all we should claim.
Picking 0.01 because it won a noisy three-seed comparison would be choosing our
own result.

**Recommendation: `count_beta = 0.01`.** Still needs all three of us.

### Third, and this one is new: Boltzmann has the same illness

Boltzmann scored **0.064 on Empty-5 against epsilon-greedy's 0.637** — ten times
worse, on the easiest maze we have, with no bonus involved at all.

Max predicted this in his own entry above ("temperature only means something
relative to the Q-values") and our numbers confirm it. MiniGrid scores run from 0
to 1, so the gap between a good and a bad action is small — often around 0.01
early on. Our temperature starts at 1.0 and is still 0.47 at step 40,000. Dividing
a 0.01 difference by a temperature of 0.47 leaves the choices almost
indistinguishable, so Boltzmann is still picking nearly at random a quarter of
the way through training.

**This is the same mistake as the bonus size**: both `tau` and `count_beta` were
chosen as round numbers without checking them against how big MiniGrid's rewards
actually are.

**We have not changed the temperature schedule.** It needs the same decision, and
the same discipline: settle it on the scale argument now, before the sweep, and
do not touch it afterwards. If we run the sweep as-is, "Boltzmann came last" would
be a statement about our schedule, not about Boltzmann.

---

## 2026-08-18 — The fake data had agents walking through walls

**Status:** Fixed

**What changed:** `scripts/make_synthetic_results.py` now only places synthetic
visits on cells the agent could actually reach, using the same `grid_info` /
`reachable_mask` functions `analysis/coverage.py` uses to grade real runs.

**The bug:** the script picked which `(x, y, direction)` combinations to mark as
"visited" by shuffling *every* cell in the grid, walls included. A real agent can
never stand inside a wall or a corner MultiRoom's rooms never reach, so this
handed `raw_coverage()` a numerator that could be much bigger than its
denominator, which only counts reachable cells. The size of the error is exactly
the ratio of "all cells" to "reachable cells", measured per instance and seed:

| family | inflation of raw coverage |
|---|---|
| Empty-5 / Empty-8 | 2.8x / 1.8x |
| DoorKey-5 / DoorKey-8 | 3.6x / 2.1x |
| MultiRoom-N4 | 12.8x – 17.9x |
| MultiRoom-N2 | 18.9x – **56.8x** |

MultiRoom is worst because its rooms sit in a corner of a fixed 25x25 canvas, so
as little as 1.8% of the grid is reachable. The highest raw coverage actually
observed before the fix was **54.4** — a number that cannot exceed 1.0 by
definition.

This could not happen on real data — an agent cannot walk through a wall — so it
was invisible until someone actually ran `raw_coverage()` against the fixture.
Task 4 (`analysis/stats.py`) is exactly that someone: its step 5 runs the whole
correlation pipeline on this fixture, and it would have been correlating
nonsense coverage numbers against return without any error being raised.

**The fix:** build the list of fillable states once per `(env_id, seed)` from
`reachable_mask(grid_info(env_id, seed))` — the same `layout_seed = seed`
convention a real run uses — and shuffle only that list. Unreachable cells now
stay at zero forever, exactly like in a real run.

**A second thing the fix broke, and we then fixed too.** `metrics.csv` has a
`distinct_states` column — how many distinct positions the agent has stood in so
far. The fixture computed it against the whole grid, which stopped matching once
the counts only filled reachable cells: one MultiRoom run claimed 2,225 distinct
states in `metrics.csv` while its own `visitation.npz` held 67. Both now use the
reachable count, so a fixture run says the same thing about itself in both files
(they still differ by up to 2%, because the two are sampled on different step
grids — evaluations every 5,000 steps, snapshots every 10,000).

**Measured after the fix**, both fixtures, all 120 runs each:

| check | result |
|---|---|
| max raw coverage | 0.993 (was up to 54.4) |
| max task-relevant coverage | 0.993 |
| runs outside `[0, 1]` | 0 |
| visits in unreachable cells | 0 |
| non-monotone count arrays | 0 |
| `distinct_states` above the reachable maximum | 0 |
| identical `counts` and `metrics.csv` across two separate generator runs | 120 / 120 |

**The baked-in ground truth survives the fix**, which is the property Task 4
depends on — within-instance Spearman between early coverage and final return:

| dataset | range | shape |
|---|---|---|
| default | **+0.49 … +0.94** | rises with difficulty, all p < 0.03 |
| `--no-effect` | **−0.22 … +0.19** | no instance significant (all p > 0.34) |

Full test suite: 165 passed, 1 xfailed, unchanged — no test had pinned down the
old, wrong numbers.

**What it means for the results:** Nothing in `coverage.py` or its tests was
wrong; the bug was entirely in the fixture generator. `results_synthetic/` and
`results_synthetic_noeffect/` must be regenerated with the fixed script before
Task 4's pipeline test runs against them — they are gitignored, so this is a
local `python scripts/make_synthetic_results.py` step for whoever runs Task 4,
not a data file to commit.

---

## 2026-08-18 — The pilot could not measure early coverage, and said nothing

**Status:** Fixed in `analysis/coverage.py`; one open item for Samuel in
`configs/pilot.yaml`

**What changed:** `early_auc()` now stops with an error when it does not have
enough data to compute an answer, instead of quietly returning a substitute.

**What "early coverage" is:** our central claim is that agents which spread out
early do better later. To measure "early", we take the coverage curve over the
first fifth of training and average it. That single number is the main predictor
in the whole project.

**The bug:** during a run we do not record coverage continuously — we save a
snapshot of where the agent has been every 10,000 steps. The real runs are
400,000 steps long, so the "first fifth" is 80,000 steps and **8 snapshots** fall
inside it. Plenty to average.

But the pilot run — the cheap 20,000-step rehearsal we do before committing to
the real thing — is 20 times shorter. Its "first fifth" is only 4,000 steps, and
the first snapshot is not taken until step 10,000. So **not a single snapshot
falls inside the window**.

Instead of saying so, the function fell back to returning the coverage at the
first snapshot it had — step 10,000. That value is two and a half times further
into the run than the window it claims to describe, and it comes back as an
ordinary-looking number. Nothing in the output distinguishes it from a real
measurement.

**Why that is worse than crashing:** the pilot exists to rehearse the entire
pipeline before we spend 3-4 hours on 260 real runs. Of all the numbers it
produces, this is the one that matters most. A silent substitute means the pilot
would report a plausible value for a quantity it never actually computed — so if
the calculation itself were broken, the rehearsal would pass anyway.

This is the same principle we already agreed on elsewhere: an environment where
every run scores identically has no meaningful correlation, and we record that as
"no variance, excluded" rather than letting an empty value slip through. A number
we could not compute must never be replaced by one we could.

**The fix:** `early_auc()` raises an error naming the actual numbers and the
setting to change:

```
early-AUC window is 4000 steps (20% of 20000) but only 0 of 2 snapshots
fall inside it (first snapshot at step 10000). Lower snapshot_every for
this run: the window needs at least 2 points.
```

**Still open, for Samuel:** `configs/pilot.yaml` should set
`snapshot_every: 1000`, which puts 4 snapshots in the window and lets the pilot
genuinely test this metric. 20 snapshots per run instead of 2, about 1 KB each —
the cost is nothing. Noted for him in `implementation_plan/STATUS.md`.

**Measured after the change:** 167 tests pass, 1 expected-to-fail (was 165 + 1;
two new tests cover the pilot's exact shape and the one-snapshot case). All 120
synthetic runs still compute normally, `early_auc` between 0.161 and 0.522 —
real 400,000-step runs are untouched by this, as intended.

**What it means for the results:** No published number changes. What changes is
that a badly configured run can no longer produce a quiet, wrong one.

---

## 2026-08-18 — The statistics trap we nearly walked into, and one we walked into

**Status:** Active. One part of the task is blocked — see the last section.

**What changed:** Wrote `src/rlx/analysis/stats.py`, the code that turns 260 runs
into the answers the report is built around. 13 tests.

### The trap the plan warned us about

Both things we measure get worse as mazes get harder. Agents cover less of a hard
maze, and they score worse on it. So if you throw all 260 runs into one pile and
ask "does more coverage go with a better score?", you get a big confident yes —
and it means nothing at all. It is the sentence "hard mazes are hard", measured
twice and mistaken for a discovery.

The fix is to ask the question **separately inside each maze**, where every run
faced the same difficulty, and only then combine the answers. There is a test,
`test_pooling_and_within_instance_disagree`, built on data where the piled-up
answer is strongly positive and the correct answer is negative. If anyone ever
"simplifies" this back into one big correlation, that test goes red.

### The trap we did walk into

The plan's own code contained the same mistake in a second place, and we only
caught it by running the finished pipeline and looking at the number.

Alongside the main result we report a trend: *does the effect get stronger as
mazes get harder?* That is one of the three things the professor asked us to
show. The code compared each maze's result against its "difficulty" number — and
that number means **different things in different families**. For Empty and
DoorKey it is the width of the grid (5, 8, 16). For MultiRoom it is the number of
rooms (2 to 6).

So MultiRoom-N2 carries the number 2 and sorts to the "easiest" end, while it is
in truth one of the hardest mazes we have: 19 walkable squares on a 25x25 canvas.
Sorting all 13 mazes by that mixed-up number and drawing a trend through them
produced **-0.53 on our test data — the wrong sign**. The effect looked like it
was fading on harder mazes when the data said no such thing.

We had already written this rule down twice, in `STATUS.md` and in the comments
of the aggregation code: *difficulty is comparable only within a family, never
across.* We wrote the rule and then broke it ourselves one file later.

**The fix:** measure the trend inside each family separately, then average. A
family needs at least 3 mazes to count — with only 2 the answer is +1 or -1 by
arithmetic and carries no information. `test_difficulty_trend_does_not_mix_families`
builds data where the effect rises inside every family while the mixed-up version
comes out negative, and fails if anyone reverts this.

**What it means for the results:** this would not have crashed and would not have
looked wrong. It would have put a confident negative number in the report,
against one of the three headline questions.

### What the fake data says

Running the finished pipeline over the 120 synthetic runs recovers the effect
that was deliberately built into them:

| coverage measure | mean correlation | 95% interval | supports H1? |
|---|---|---|---|
| raw | **+0.773** | +0.628 to +0.903 | yes |
| task-relevant | **+0.713** | +0.597 to +0.816 | yes |

The trend-with-difficulty number comes out as "cannot measure" on this data, and
that is correct: the fake data has only 2 mazes per family, below the 3 we
require, **and it never varied the effect strength within a family in the first
place** — the generator sets it per family, so Empty-5 and Empty-8 are given the
same strength by construction. The trend code is covered by its unit test, but
the fixture cannot check it end to end. Worth fixing in the fixture before the
real numbers arrive, so that this path is exercised on something.

### Blocked: the rliable comparison

The proposal commits to the `rliable` library for the aggregate strategy
comparison, and the professor's feedback approved that choice. **It does not
currently import**, on Daniel's machine and probably on all three:

```
TypeError: deprecate_kwarg() missing 1 required positional argument
```

The chain: our `requirements.txt` allows any `pandas>=2.0`, so pip installed
**pandas 3.0.5**. pandas 3 changed the signature of an internal helper. The
`arch` library (version 7.2.0) still calls it the old way and dies on import.
`rliable` imports `arch` at the top of its file, so `rliable` dies with it. The
part of `arch` involved is exactly the stratified bootstrap our confidence
intervals are supposed to use.

Nothing detected this earlier because no test imported `rliable` until now. Left
alone, it would have surfaced on the 22nd, while writing the report.

**Verified fix, not yet applied:** `arch` 8.0.0 exists and ships its own copy of
that helper instead of borrowing pandas' internal one (checked inside the
downloaded package, not guessed), and it declares `pandas>=1.4.0` with no upper
limit. Upgrading `arch` is a one-line change and leaves pandas alone. The
alternative — holding pandas below version 3 — also works but steps a core
library backwards to avoid fixing a small one.

**Decision deferred by the team on 2026-08-18.** `rliable_aggregate` is therefore
**not written**, rather than written and untested: we do not ship code we have
not seen run. Everything else in the task is complete, including
`probability_of_improvement`, which needs no external library. `iqm_by_strategy`
already produces interquartile means with bootstrap intervals per maze, so the
per-maze plots are unaffected; only the single cross-maze aggregate is waiting.

**Superseded on 2026-08-19** — see "rliable runs now, and the fix we had
recommended was the wrong one" at the end of this file. The `arch` 8.0.0 upgrade
recommended above does NOT work; it was only ever checked as an import.

## 2026-08-19 — rliable runs now, and the fix we had recommended was the wrong one

**Status:** Active. Supersedes the "Blocked: the rliable comparison" section of
the 2026-08-18 entry above.

**Background in one sentence:** `rliable` is the statistics library the proposal
promises and the professor approved; it produces the "which strategy wins, with
honest error bars" number by re-drawing our results thousands of times and seeing
how much the answer wobbles.

**What we thought yesterday:** that upgrading one small library (`arch`, from
7.2.0 to 8.0.0) would fix it. That recommendation was written down in
`STATUS.md` and in the log entry above, and it was **wrong**.

**What actually happened:** the upgrade fixed the crash we could see and revealed
a second one behind it. `arch` 8.0.0 had renamed a setting from `random_state` to
`seed`, and `rliable` still asks for it by the old name — so every single
calculation failed instead of only the import. We had verified that `rliable`
could be *loaded*, never that it could be *used*. That is the whole lesson: an
import check is not a test.

**What we did instead:** held both libraries one version back — `pandas` below 3
and `arch` below 8 — which is the combination we actually watched produce
numbers. Both pins are in `requirements.txt` with the reason written next to
them, because removing either one brings the breakage straight back.
`rliable` 1.2.0 is the newest release there is, so waiting for an update was not
an option.

**Everyone must re-run this**, or the analysis will crash on your machine:

```bash
pip install -r requirements.txt
```

**A second surprise, and why our numbers would otherwise have moved on their
own:** `rliable` offers a setting that is supposed to make its results
repeatable. Reading its source shows the setting is ignored — the library asks
the computer for fresh random numbers from a different place than the one that
setting controls. Left alone, every regeneration of the report would print a
slightly different confidence interval, and nobody would know which was in the
submitted PDF. We now set that other source explicitly and put it back the way we
found it afterwards. Verified: 20 runs in a row, deliberately shaken up in
between, produce **bit-for-bit identical** output.

**What it means for the results:** no number changes. `pandas` 2 versus 3 does
not affect any calculation we do — we only use it to hold tables. What changes is
that the cross-maze comparison the proposal promised now exists and is
reproducible, instead of being missing.

## 2026-08-19 — The fake data now covers all 13 mazes, and one dial had to move

**Status:** Active

**Reminder of what the fake data is for:** we build the analysis before any real
experiment has run, using invented results in the exact format real runs will
have. The invented data has a known right answer, so if the analysis fails to
find it, the analysis is broken.

**What changed:** the fake dataset covered 6 of the 13 mazes, two per family. It
now covers all 13.

**Why:** one of our three hypotheses is that the coverage effect gets *stronger*
as mazes get harder. Measuring "does it get stronger" needs at least three mazes
of increasing difficulty in a family. With only two, the measurement could not be
computed at all and came back blank — so that hypothesis had no end-to-end test.

**The dial that had to move, and this is the interesting part:** extending to 13
mazes was not enough on its own. On first measurement the trend came out at
**+0.13 for the real dataset and +0.17 for the deliberately-effect-free control
dataset** — in other words, our "signal" was smaller than our known-nothing case,
which would have made the test worthless.

The cause was a ceiling. The invented effect was tied to how hard a maze is
overall, and on the MultiRoom family every instance already sat at the very top
of the scale (0.90 to 0.97 out of a maximum of 1.0). There was no room left to
climb, so a rising effect could not show up as rising. We retied the invented
effect to a maze's position *within its own family* — which is exactly what the
analysis measures — so the easiest maze in each family now starts low and the
hardest ends high.

After the change: **+0.90 for the real dataset against +0.17 for the control.**
Five times the separation, so the test can now actually tell the two apart.

**What it means for the results:** nothing about the real experiment changes —
this is invented data, and no reported number comes from it. What changes is that
one more thing can now fail loudly if we break it. Two consequences to know
about:

- The fake dataset is now 260 runs and 8.6 MB instead of 120 runs and 3.5 MB. It
  takes about 3 seconds to generate and is not stored in git.
- **If you have a `results_synthetic` folder from before 2026-08-19, delete and
  regenerate it**, otherwise you are looking at yesterday's shape:

```bash
python scripts/make_synthetic_results.py --out results_synthetic
```

## 2026-08-19 — A full review of tasks 1-4 found nine things, and we fixed all nine

**Status:** Active

**What this was:** before building the figures, we read every requirement — the
professor's feedback, the design spec, our own rules — back against the code that
is supposed to satisfy them, and re-measured anything we were not sure about.
Nine things came out of it. None of them made the code crash, which is exactly
why a review was needed to find them.

**Two of them would have put wrong numbers in the report.**

**1. Rank stability was ranking strategies by their average, not by the robust
average we said we would use.** With five runs per strategy, one collapsed run
drags an average far enough to swap two strategies that are otherwise clearly
apart — and our third question is precisely "does the winner change as mazes get
harder". So a single unlucky run could have been reported as a rank change that
never happened. Measured on the fake data: the two methods disagreed on **7 of
the 13 mazes**. Now fixed to use the robust version (IQM — drop the best and
worst quarter, average the middle), as the spec said all along.

**2. Our main predictor was 12% too high.** "Early coverage" is meant to be the
average coverage over the first fifth of training. The code was averaging over
the *snapshots it happened to have* instead of over the *window it claims to
summarise*. Since training takes its first snapshot at step 10,000 and the window
runs from 0 to 80,000, the first eighth of the window was silently missing.

Every test passed because every test used a made-up snapshot list that started at
step 0 — which is the one case a real run never produces. That is the lesson
worth keeping: **a test that only exercises a shape your production code never
creates is not testing your production code.**

**The good news, and we checked it rather than assuming:** the central result is
untouched. Our main test ranks runs against each other, and this error scaled
every run by the same factor, so the ranking — and therefore the correlation, the
difficulty trend, everything we actually conclude — came out **bit-for-bit
identical**. Only the printed number was wrong.

**Four things the spec asked for were simply missing.**

**3. Performance profiles.** The spec and the professor both wanted them: instead
of one summary number per strategy, a curve showing what fraction of runs beat
each possible score. Two strategies can have the same average and completely
different shapes — one solves half the mazes brilliantly and fails the rest, the
other is mediocre everywhere. The average hides that; the profile shows it.

**4. A confidence interval for each individual maze.** We had one for the overall
average but not for the per-maze numbers, so there was no way to tell a solid
per-maze result from a coincidence. With 20 runs per maze that matters: on the
fake data the intervals average **0.45 wide** where the effect is real and
**0.94 wide** where it is not — the width alone separates signal from noise.

**5. A real test of our second hypothesis.** We had promised to check whether
"coverage of the states that matter" predicts better than "coverage of
everything", and specifically that one only wins if the two confidence intervals
do not overlap. Nothing computed that; the report would have printed two numbers
side by side and left a human to eyeball it. Now there is a function that answers
it, including the case the spec calls the interesting alternative: if they
predict equally well, that is a genuine finding — exploring broadly matters,
exploring cleverly does not.

**6. Our "hypothesis confirmed" flag only checked half the hypothesis.** We wrote
down in advance that the main hypothesis needs *two* things: a positive
relationship whose confidence interval excludes zero, **and** a relationship that
gets stronger on harder mazes. The flag only checked the first. It could have
printed "confirmed" on data where the effect shrinks with difficulty — the
opposite of what we predicted. Both halves are now required, and they are also
reported separately so the report can say which one failed.

**Three were traps waiting to spring.**

**7.** If every maze had come out tied, the summary function returned a result
missing the field the report generator reads — so the report would have crashed
while being written, on the 22nd, instead of reporting "not confirmed".

**8.** One unusable run used to abort the whole 260-run analysis at the first one
it met, so you found out about the next one only after re-running. It now
collects them all and names every one in a single message.

**9. Finishing a run could destroy a finished run.** Our own rule says a result
folder that exists is a run that completed, and the sweep runner skips those. But
the writer deleted any folder in its way before renaming. Anyone re-running one
run by hand — the obvious thing to do when checking something — would have
silently destroyed a ten-minute result. It now refuses and says so.

**What it means for the results:** no conclusion changes. The correlation, the
difficulty trend and the hypothesis outcomes on our fake data are identical
before and after. What changes is that the numbers printed in the report now mean
what the report says they mean, four promised analyses exist, and three ways of
losing work or crashing at the worst moment are closed.

**One limitation we are recording rather than papering over:** the fake dataset
cannot confirm the second hypothesis, because its effect is built into total
coverage only, with nothing extra for task-relevant coverage. On it the
comparison correctly returns "not confirmed". The comparison logic has its own
tests covering both outcomes, but the end-to-end path will first meet a real
"confirmed" case on the actual results. We chose to write this down rather than
redesign the fake data to flatter us.


---

## 2026-08-19 — Boltzmann's temperature was measured against the real mazes, and it is 500x too high

**Status:** Active. **A proposal, not a change — `config.py` is Samuel's file and
nobody has edited it.** Must be settled before the sweep on the 20th.

**Background.** The pilot had Boltzmann scoring 0.064 on Empty-5 where plain
epsilon-greedy scored 0.637, with no novelty bonus involved anywhere. Our own
earlier log entry predicted why, so we went and measured it properly.

### What we measured

We ran a real Double DQN on six mazes — two from each family, easy and hard —
for 160,000 steps each, two seeds apiece, twelve runs in total. At every single
step we recorded how far apart the network's action scores were: the gap between
its favourite action and its runner-up.

The agent was driven by **epsilon-greedy**, deliberately. Driving it with
Boltzmann would have made the measurement depend on the very number we were
trying to choose.

### The result

| maze | did it ever solve it? | typical gap between best and second-best |
|---|---|---|
| Empty-5 | yes (0.95) | 0.0059 |
| Empty-16 | yes (0.76) | 0.0034 |
| DoorKey-5 | yes (0.97) | 0.0029 |
| DoorKey-8 | no (0.01) | 0.0001 |
| MultiRoom-N4 | no (0.00) | 0.0001 |
| MultiRoom-N6 | no (0.00) | 0.0001 |

Two things stand out. The gaps are **tiny** — thousandths, not tenths. And they
line up exactly with whether the agent ever found the goal: mazes it solves
develop gaps around 0.003 to 0.006, mazes it never solves stay near 0.0001. That
makes sense and is a good sign the measurement is real rather than a bug: if
nothing is ever rewarded, the network has no reason to rate any action above any
other, so all its scores collapse together.

The gaps also did **not** grow over the 160,000 steps we watched. They were flat
or shrinking.

### Why this breaks Boltzmann completely

Boltzmann's temperature is divided into those scores, so it only means something
next to them. Ours runs from 1.0 down to 0.05. The scores are around 0.003.

Working out what the agent actually does, where picking at random would give any
one of the seven actions a 14.3% chance:

| point in training | temperature | chance it picks its favourite action |
|---|---|---|
| start | 1.0 | 14.3% |
| 40,000 steps | 0.47 | 14.4% |
| 160,000 steps onward | 0.05 | 15.1% |

**Boltzmann is picking at random for the entire run.** Not "explores too long" —
it never stops exploring, on any of the thirteen mazes, at any point in 400,000
steps. Epsilon-greedy by comparison ends up picking its favourite 95.7% of the
time. That fully explains the 0.064: we were not testing Boltzmann against
epsilon-greedy, we were testing random against epsilon-greedy.

### What we propose instead

Keep the shape of the schedule and the 40% decay window. Move the two endpoints
so they are derived from the measured gap of 0.0034 rather than picked as round
numbers:

| | now | proposed |
|---|---|---|
| `tau_start` | 1.0 | **0.01** |
| `tau_end` | 0.05 | **0.001** |

The endpoints come from stating what we want and solving backwards: start
slightly above random, end clearly decisive. That gives:

| point in training | chance it picks its favourite |
|---|---|
| start | 28% |
| 40,000 steps | 39% |
| 160,000 steps onward | 93% |

which is a strategy that explores and then commits, and is comparable to
epsilon-greedy's 14% to 96% while still committing more gradually — which is what
makes Boltzmann a different strategy rather than a copy.

**On the mazes it never solves, the proposal still behaves nearly at random**
(around 18%). We think that is correct rather than a flaw: with no reward ever
found, there is nothing to be decisive *about*, and confidently repeating an
arbitrary action would be worse.

### Why this is calibration and not cheating

Our own rules forbid tuning a strategy to make it win. The line we are drawing:
the rule is **"every knob measured in reward-units is set against the actual
reward scale of the environment"**, it is written down here before the sweep runs,
and it is applied to all four strategies. Epsilon-greedy needs no such treatment
because its knob is a probability and has no units — that is luck of definition,
not virtue. NoisyNets' `sigma0` does have units and must be checked the same way
before the 20th.

What we are explicitly **not** doing is trying several values and keeping the one
that scores best. We ran no Boltzmann at all in this measurement.

### Honest limits of this measurement

- The gaps come from a network trained by **epsilon-greedy**. Boltzmann's own
  run will produce a somewhat different network. There is no way around this
  without circularity, but it means the endpoints are calibrated to a reference
  scale rather than to Boltzmann's own.
- **Two seeds, six of thirteen mazes, 160,000 of 400,000 steps.** The gaps were
  flat over that window so we expect them to stay flat, but we have not seen the
  last 240,000 steps.
- The gaps differ **59-fold** between the easiest and hardest maze, so no single
  temperature is right everywhere — the same structural problem the novelty bonus
  has. We are choosing one number for all thirteen because the schedule has to be
  stated in the report, and because on the low-signal mazes being near-random is
  the sensible fallback.

### One alternative we considered and rejected

Rescale the temperature automatically from the spread of scores the network is
currently producing, so it adapts per maze and per moment. It would handle the
59-fold spread properly. We rejected it because it changes what "Boltzmann
exploration" means, adds code and a tuning knob of its own, and the professor
asked for a schedule we can state plainly in the report. Written down because if
the results look odd on the hard mazes, this is the first thing to reconsider.

**Reproduce with:** `python scripts/measure_q_gaps.py --envs Empty-5 DoorKey-5
MultiRoom-N4 --seeds 0 1 --steps 160000 --window 10000 --out gaps.csv`
(about 3 minutes per run at ~960 steps/s).

---

## 2026-08-19 — NoisyNets implemented, and its knob turns out to be fine

**Status:** Active

**What changed:** Added the fourth and last strategy. The other three add
randomness to *which action* the agent picks. This one adds randomness to the
network's *own weights* instead, and then chooses actions with no randomness at
all.

**The interesting bit:** the amount of noise is a learned parameter. The network
works out for itself which parts of it still benefit from randomness, instead of
us picking a decay schedule by hand like we did for the other strategies. That is
the main argument for it in the original paper.

**What it means for the results:** We switch the noise off when measuring
performance, so reported scores use the network's average weights. If we forgot
that, every score would have random jitter in it. There is a test that fails if
someone breaks it.

**One implementation note:** we use the "factorised" noise variant, which draws
one random number per input and per output and multiplies them, instead of one
per weight. For our biggest layer that is about 1,000 random numbers per step
instead of 65,000 — same idea, far cheaper.

### We checked `noisy_sigma0` the same way we checked the other two knobs, and it does not need changing

Having found that Boltzmann's temperature was 500x mis-scaled and the novelty
bonus 14x, we were expecting a third problem. There isn't one, and the reason is
worth writing down.

We measured how often the agent's chosen action actually *changes* when the noise
is redrawn. That is the direct measure of how much this strategy explores.
Picking completely at random would change it about 86% of the time; never
exploring would be 0%.

| maze | does it solve it? | start | 10,000 | 25,000 | 50,000 |
|---|---|---|---|---|---|
| Empty-5 | yes | 32% | 39% | 43% | **9%** |
| DoorKey-5 | eventually | 32% | 70% | 45% | **16%** |
| MultiRoom-N4 | never | 32% | 80% | 57% | **76%** |

This is the behaviour we wanted and did not get from Boltzmann. On mazes the
agent solves, it explores heavily early and then settles down and commits. On the
maze it never solves, it keeps exploring — which is the right response to having
learned nothing.

**Why this one self-corrects and the other two did not.** Boltzmann's temperature
and the novelty bonus are numbers *we* fix in advance, so if we pick them at the
wrong scale they stay wrong for the whole run. NoisyNets' noise level is a
learned parameter — the training process adjusts it. We watched it fall by about
26% on Empty-5 over 50,000 steps on its own. The knob only sets a starting point,
and the network moves it from there.

So the size of the noise stays comparable to the size of the differences between
action scores throughout, instead of dwarfing them by 500x the way the
temperature did.

**Decision: `noisy_sigma0` stays at 0.5. No change to `config.py`.**

**Honest limits:** 50,000 steps of the eventual 400,000, one seed, three of
thirteen mazes. We are reading the shape of the curve, not certifying a value.
And unlike the temperature question, there is a real safety net here: if 0.5 were
somewhat wrong, learning would pull it back. That is exactly why we are
comfortable leaving it alone on thinner evidence than we demanded for Boltzmann.

**Reproduce with:** `python scripts/measure_sigma.py --env Empty-5 --steps 50000`

### A deliberate simplification, for the report's limitations section

The original paper redraws the noise for both the online and the target network
on every learning update. Our training loop only redraws it before the agent
acts, so one learning update reuses the noise from that step, and the target
network keeps the noise it started with.

This is on purpose, not an oversight: it keeps the training loop byte-identical
across all four strategies, which is the controlled comparison the whole project
rests on. Exploration still works, because the noise that actually drives the
agent's choices is redrawn every single step. **This belongs in the report.**
Deviating from a cited paper is fine; deviating from it silently is not.

### Nothing needed from Samuel

His end-to-end test was written to detect our module rather than to be edited:
it checks whether `rlx.exploration.noisy` can be imported and drops its
"expected to fail" marker automatically. It became a real passing test the moment
our file landed. Worth copying that trick.

## 2026-08-19 — The seven figures, and the four things looking at them caught

**Status:** Active

**What changed:** wrote the code that produces every figure in the report, so all
seven regenerate from one command whenever the results change:

```bash
python -m rlx.analysis.figures --results results --out report/figures
```

**The seven:**
1. Learning curves — score over time, all four strategies, one panel per maze family.
2. Score against difficulty — the headline plot the professor asked for.
3. Coverage over time — both our coverage measures.
4. Early coverage against final score — **the central result of the project**.
5. Which strategy wins — robust average per maze, plus the full distribution.
6. Rank stability — does the winner on easy mazes still win on hard ones?
7. Visitation heatmaps — a picture of where each strategy actually went.

**If we had to cut to two,** it would be 4 and 7. Number 4 is the actual claim we
are making. Number 7 is the one that makes someone walking past the poster stop,
because you can *see* the difference instead of reading it off an axis.

**Four problems only became visible by opening the pictures**, which is why the
plan insists on looking at every one rather than trusting that the tests passed:

- **The step axis was unreadable.** Runs are 400,000 steps long and the tick
  labels ran into each other as one solid line of digits — "50000100000150000".
  Now labelled in thousands: 80k, 160k, 240k.
- **Figure 5 listed the mazes alphabetically**, which put DoorKey-10 before
  DoorKey-5 and Empty-16 before Empty-5. Difficulty has to read left to right or
  the figure argues against itself. Now sorted by family, then difficulty.
- **A legend sat on top of the bars** it was meant to explain.
- **Figure 4 used the same four colours for two different things.** Its first two
  panels colour points by strategy; the third panel had coloured the mazes by
  family — in the same figure, so blue meant "Boltzmann" on the left and
  "DoorKey" on the right. Families are now told apart by marker shape instead.

**One thing that is a correctness fix, not a cosmetic one.** The four heatmaps in
figure 7 now share a single colour scale. With one scale per panel, every
strategy would look equally thorough no matter how much of the maze it actually
covered — which is the exact opposite of what the figure is for.

**Two additions beyond the original plan, both because the design document asks
for them:**

- Figure 5 gained a second panel, the **performance profile**: for every possible
  score, what fraction of runs beat it. A single average can hide the shape
  completely — two strategies can average the same while one solves half the
  mazes brilliantly and fails the rest.
- Figure 4 gained a third panel showing **each maze's own correlation with its
  confidence interval**. This is the panel that says which per-maze results are
  solid and which are coincidence, and on our fake data it does exactly that:
  Empty-5's interval straddles zero while DoorKey-8's sits tight and high.

**One rule we kept:** every shaded band and every error bar in all seven figures
is a bootstrap confidence interval, never a standard deviation. Standard
deviations look tighter and would make our results seem more certain than five
seeds justify. The plan's own draft used a standard error for figure 2; we
changed it to match the rule.

**What it means for the results:** nothing changes any number. But regenerating
from one command means the report can never show a figure built from older data
than the tables beside it.

**One honest caveat about figure 7.** On our fake data the four heatmaps look
nearly identical, because the generator fills almost every reachable cell by the
end of training regardless of strategy. That is a property of the fake data, not
of the figure. Whether this figure actually separates the strategies can only be
judged on the real runs — and if it does not, the honest fix is to draw an
earlier snapshot rather than a later one, not to adjust the picture.

---

## 2026-08-19 — The three of us agreed the knob values, and they are now fixed

**Status:** Active. **These numbers are settled before any experiment has run.**

**What changed:** Three numbers in `src/rlx/config.py`:

| knob | was | now | why |
|---|---|---|---|
| `tau_start` | 1.0 | **0.01** | 300x the action-score differences it is meant to soften |
| `tau_end` | 0.05 | **0.001** | 15x those differences, so Boltzmann never became decisive |
| `count_beta` | 0.05 | **0.01** | novelty bonus was worth up to 14x solving the maze |

Left alone deliberately: the epsilon-greedy settings (its knob is a probability,
so the size of the scores cannot affect it), `tau_decay_frac` (the *shape* of the
schedule was never the problem, only where it started and ended), and
`noisy_sigma0` (measured on 2026-08-19 and found to be fine, because it corrects
itself as the network learns).

**What it does, measured with the real code before and after:**

| | before | after |
|---|---|---|
| Boltzmann picks its favourite action, at the start | 14.3% | 28.4% |
| ...a quarter of the way through | 14.4% | 39.1% |
| ...at the end | 15.1% | **93.4%** |
| Novelty bonus collected in one early episode | 11.42 | **2.28** |

For reference, picking completely at random gives 14.3%, and epsilon-greedy ends
at 95.7%. The "before" column is the whole problem in one line: Boltzmann was
random at the start, random in the middle, and random at the end.

**Why this is allowed, since we have a rule against tuning strategies to win.**
The rule we applied is *"any knob measured in reward-units gets set against the
actual reward scale of the environment"*. It was written down, applied to all four
strategies at once, and settled **before the sweep**, so no number here was chosen
by looking at who won. Epsilon-greedy needed nothing because its knob has no
units. We did not try several values and keep the best — we solved for the
endpoints from the measured numbers.

**These values are now frozen.** Changing any of them after the sweep would mean
picking our result after seeing it. If they turn out to be wrong, that is a
finding for the report, not an edit.

### A test had to change, and it is worth explaining why

`test_better_actions_are_sampled_more_often_than_worse_ones` started failing. It
was not a real breakage. The made-up action scores our tests use span 0.85, while
a real network produces differences around 0.003 — roughly 250 times smaller. At
the new temperature the maths saturates on those made-up numbers and every draw
picks the same action, so the property became untestable on them rather than
untrue. We pinned the temperature inside that test, exactly as the two tests
beside it already do, so it no longer depends on the config.

**We also added the test that would have caught this bug in the first place.**
Nothing in our suite noticed that Boltzmann was behaving randomly for an entire
run. There is now a test using the *measured* score differences that fails if the
schedule does not start meaningfully above random and finish meaningfully
decisive. Checked both ways: it fails on the old values (finishing at 18.2%) and
passes on the new ones (97.3%).

---

## 2026-08-19 — The pilot run caught that we had over-corrected Boltzmann, and one number moves again

**Status:** Active

**Short version:** We ran the 16-run pilot. It did its job: it found a real bug.
The temperature fix from earlier today fixed one end of the schedule and broke
the other. One number, `tau_start`, moves from 0.01 to 0.1. Nothing else changes.

### What the pilot showed

All 16 runs finished and wrote every file they were supposed to. The pipeline
works end to end. But Boltzmann was visiting **4 of about 36 possible positions**
on Empty-5 — a 5x5 empty room. Four positions means one square, facing four
directions. It stood still and turned on the spot for 20,000 steps.

Our earlier entry today said Boltzmann's temperature was 500x too high and it
never stopped behaving randomly. That was true. The fix went too far the other
way: it now behaved *decisively* from the very first step, before it had learned
anything at all, so it locked onto a preference that came out of the random
numbers the network was created with and never had a reason to reconsider.

### Why we got it wrong, in one sentence

We measured how far apart the agent's action scores are *after* it has trained,
and used that one number to set both ends of the schedule — but the scores are
about **7 times further apart at the start of training than at the end**, so the
start of the schedule was calibrated against the wrong ruler.

Measured, on a brand-new untrained network:

| when | typical gap between the best and second-best action |
|---|---|
| brand-new network, before any training | 0.0206 |
| after training | 0.0030 |

The 0.0206 figure is the average over 6 different mazes with 15 different random
starts each, and it barely moves between mazes (all six sit between 0.0200 and
0.0222). That is because it is a property of how the network is built, not of the
maze — which is convenient, because it means one number is right everywhere.

### What that does to the agent's behaviour

"Chance of picking its current favourite action" — 14.3% would be a coin flip
between the 7 actions, 100% would be no exploration at all:

| temperature setting | at the first step | at the last step |
|---|---|---|
| original (1.0 -> 0.05) | 14.5% — random | 18.4% — still basically random |
| this morning's fix (0.01 -> 0.001) | **86.1% — already decided** | 94.6% |
| **new (0.1 -> 0.001)** | **28.0% — favours the better actions, but keeps looking** | 94.6% |

28% is the number we want at the start. It is meaningfully above a coin flip,
which is Boltzmann's entire selling point over epsilon-greedy — when it explores,
it leans toward actions it rates highly rather than picking uniformly at random.
But it is nowhere near committed.

### We checked this by re-running, not by trusting the arithmetic

We re-ran the four Boltzmann pilot runs with the new value:

| run | positions visited, old | positions visited, new |
|---|---|---|
| Empty-5 seed 0 | 32 | 32 |
| Empty-5 seed 1 | **4** | **32** |
| DoorKey-5 seed 0 | **4** | **24** |
| DoorKey-5 seed 1 | **4** | **24** |

The standing-still behaviour is gone, and Boltzmann now explores about as much as
the other three strategies do.

### Are we allowed to change a number we said was frozen?

This is the uncomfortable part, so it gets stated plainly rather than buried.

Earlier today we wrote "these values are now frozen — changing any of them after
the sweep would mean picking our result after seeing it." We are changing one.
Three things make us think this is honest rather than convenient:

1. **The sweep has not been run.** The pilot is a 20,000-step pipeline check, 5%
   of the real budget. It exists precisely to catch this kind of thing.
2. **We fixed a strategy that was broken, not one that was losing.** Standing on
   one square for an entire run is not a weak result, it is a non-functioning
   strategy. If we had left it in, the report's finding would have been an
   artefact of our own bug.
3. **The score was 0.000 in both versions.** This is the important one. Every
   Boltzmann pilot run scored zero before the change and zero after it. We chose
   the new value by looking at *how much of the maze got visited*, and the number
   we are actually trying to measure did not move at all. So it is not possible
   that we picked this value because it made Boltzmann look better — there was
   no "better" visible.

What we are giving up is that this is the second correction today, and each one
costs a little of the "we decided in advance" argument. **We are stating both
corrections in the report rather than presenting the final numbers as if we had
picked them first time.** The honest version — "we set these against a measured
scale, got the reasoning wrong once, caught it with a pilot run, and here is what
changed" — is a better story than a suspiciously clean one, and the professor's
feedback asked us to be explicit about how schedules were chosen.

**Now genuinely frozen.** The real sweep runs against these values.

### The test that should have caught it, and now does

We already had a test checking Boltzmann eventually becomes decisive. It only
checked the *end* of the schedule, which is exactly why the new bug walked
straight past it. It has been replaced with one that checks both ends, and — the
important part — checks each end against the ruler that actually applies at that
point: the untrained score gaps for the start, the trained ones for the end.

It uses real action scores taken from an actual freshly-built network rather than
numbers we made up, since made-up numbers on the wrong scale are what caused this
in the first place.

Verified it rejects both bugs rather than just passing on the new value:

| values | result |
|---|---|
| original 1.0 -> 0.05 | **fails** — "never commits to its favourite action: 0.184" |
| this morning's 0.01 -> 0.001 | **fails** — "already committed at step 0: 0.861" |
| new 0.1 -> 0.001 | passes |

Full suite: 220 passed.

### Two things the pilot found that are NOT ours, passed to the others

**1. For Samuel — the greedy score is 0 while the training score is 0.9.**

On Empty-5, all four strategies reach a training score around 0.88-0.95, meaning
the agent reaches the goal regularly while it is exploring. The evaluation score,
which switches exploration off and just takes the best-rated action every time,
is 0.000 for nearly all of them.

In these mazes, reaching the goal *always* scores above 0.1. So a score of
exactly 0 means the agent never reached the goal in that episode at all — it ran
out of time. The agent solves the maze when it has a bit of randomness in it, and
gets stuck when it does not. The likely reason is that with no randomness the
agent can walk into a repeating cycle (turn, turn, turn, turn) with nothing to
break it, whereas a single random action escapes.

We do not think this is a broken evaluation — we read `evaluate()` and it does
the right things (exploration off, no bonus, same maze). It is more likely that
20,000 steps is simply not enough: that is only 5,000 learning updates, against
100,000 in the real run. One run did reach 0.955 and hold it. **Flagging it so
Samuel can decide whether to watch for it in the real sweep**, because if it is
still happening at 400,000 steps, every strategy scores zero and we have no
result at all.

**2. For Daniel — the two coverage measures are identical on the pilot mazes.**

"Raw coverage" (how much of the maze was visited) and "task-relevant coverage"
(how much of the part that matters was visited) came out to exactly the same
number on all 16 pilot runs. This is correct behaviour, not a bug: on the two
smallest mazes the "part that matters" is the whole maze, so there is nothing to
tell apart. On bigger mazes they separate properly — on MultiRoom-N2 the relevant
part is 2% of the maze.

One consequence worth putting in the report: for the **Empty** family the two
measures are identical *at every size*, including Empty-16. In a room with no
walls, every square lies on some shortest route from the start to the goal, so
"the part that matters" is the entire room by definition. The task-relevant
measure only adds information for DoorKey and MultiRoom.

`early_auc`, the project's main predictor, computed a real number on all 16 runs
with no failures — which is the specific thing the pilot's snapshot fix was
meant to enable.
## 2026-08-19 — A half-written figure set was possible, and a label that would have lied

**Status:** Active

**Two small fixes found by reviewing the finished work rather than by a failing
test. Neither changes a number.**

**1. Regenerating the figures could leave the report showing two datasets at
once.** The figure code checks whether every run is present — but it checked in
the middle, while drawing figure 5. Figures 1 to 4 were already written to disk
by then. So a sweep with one crashed run produced four fresh figures sitting
beside three left over from an earlier render, with nothing saying so, and the
report would have shown numbers from one dataset next to pictures from another.

That is exactly the failure this log claimed was impossible three entries ago
("regenerating from one command means the report can never end up showing a
figure from an older version of the data"). It now checks before drawing
anything, so a refused render leaves the folder untouched. There is a test that
deletes a run and fails if any file appears.

**2. The report generator was about to print something false.** It had a line
reading "CI excludes zero: <value>", filled with a field that, since the review
earlier today, means something else: our first hypothesis needs *two* things, and
that field now reports whether both hold. On results where the interval excludes
zero but the effect shrinks with difficulty, the report would have printed
"CI excludes zero: False" — a plain false statement about the interval.

Fixed in the plan before the code was written: the two conditions are now printed
as two separate lines, one for the interval and one for the hypothesis as a
whole.

**What it means for the results:** nothing changes. One is a failure mode that
had not happened yet, the other a sentence that had not been written yet. Both
were cheaper to fix today than to notice on the 22nd.

## 2026-08-19 — The pilot found three things in the figures that the fake data never could

**Status:** Active

**What happened:** we ran the real 16-run pilot for the first time and pointed the
figure code at it. All seven figures appeared, but with two warnings and one
silently wrong picture. None of these could have shown up on our fake dataset,
because the fake dataset always contains all thirteen mazes and the pilot
contains two.

**1. A panel for a family that is not there.** Figures 1 and 2 always drew three
columns, one per maze family. The pilot runs two of the three, so the third came
out blank with meaningless numbers on its axis — and the legend, which we put on
the last column, landed inside that empty one and disappeared. So the pilot's two
most-read figures had no key telling you which colour was which strategy. The
figures now have one column per family that actually has runs.

**2. A deprecation that will become a crash.** With a single maze per family,
matplotlib was handed one-element pandas tables where it expects plain numbers.
It still works and warns; a future version will refuse. Fixed by handing it plain
numbers.

**3. A panel that was empty for a real reason and did not say so.** Rank
stability compares the ordering of the four strategies on a hard maze against the
ordering on an easy one. On DoorKey-5 in the pilot, all four strategies scored
exactly zero — nothing solved it in 20,000 steps — so there is no ordering to
compare and the number is undefined, correctly. The panel simply came out empty,
which looks like a broken plot rather than a result. It now says "no variance:
every strategy scored the same".

**This one matters beyond the pilot.** Our own notes predict the same outcome on
the hardest real mazes: if nothing ever solves DoorKey-10 or MultiRoom-N6, those
panels would have been blank in the final report with no explanation.

**One thing we are NOT changing, but everyone should know when reading pilot
numbers.** Our headline score per run is "the average of the last five
measurements", which smooths out one lucky evaluation. The pilot is short enough
to have only four measurements in total, so that average covers the entire run —
including the zeros from before the agent had learned anything. In the pilot one
agent ends at 0.955 and gets reported as 0.477. On the real runs there are eighty
measurements and the last five are all from the finished agent, so the number
means what it says. We left the definition alone rather than special-casing it;
the pilot is a plumbing test, not a source of results.

**What it means for the results:** no number changes. Three figure defects fixed
before they could reach the report, all found by ten minutes of real data that
two days of fake data could not surface.

## 2026-08-19 — On the smallest maze our main measurement cannot tell anything apart

**Status:** Active

**What we found:** re-running the pilot after the temperature correction, two
strategies came out with *exactly* the same early-coverage number on Empty-5.
That is not a coincidence and not a bug.

Empty-5 has nine walkable squares. With four directions to face, that is 36
distinct situations the agent can be in. Both strategies had seen 32 of them
after **one thousand steps** and never saw another one after that — the missing
four are almost certainly at the goal square, which ends the episode the moment
the agent steps on it, so it is only ever seen from one direction.

```
boltzmann       coverage at 1k / 2k / 3k / 4k steps = 0.889  0.889  0.889  0.889
epsilon-greedy  coverage at 1k / 2k / 3k / 4k steps = 0.889  0.889  0.889  0.889
```

**Why it matters.** Our central measurement is "how much of the maze did the
agent see early on", and we compare that against how well it eventually scores.
On Empty-5 every strategy sees everything almost immediately, so that measurement
is the same number for all of them and can predict nothing. There is no effect to
find there — not because exploration does not matter, but because the maze is too
small for the question to have an answer.

**This is the maze, not the short test run.** Empty-5 will be exhausted inside the
first one percent of a full-length run too.

**It agrees with what we already saw on the fake data**, where Empty-5 was the
weakest of the thirteen and the only instance whose confidence interval included
zero.

**What it means for the results:** the report should say this in the limitations
section rather than let Empty-5 turn up later as a weak result that needs
explaining away. The honest sentence is: on the smallest instances the coverage
measure runs out of room before the early window closes, so it has no
discriminating power there, and the hypothesis can only be tested where the maze
is big enough for strategies to differ. Nothing about the other twelve instances
changes.

---

## 2026-08-19 — The report now has an outline, and every number it needs is generated

**Status:** Active

**What changed:** two new things, plus a place for Max to write.

- `report/outline.md` — the section-by-section structure of the report, with the
  owner of each section and, for every section, which figure and which numbers it
  uses.
- `src/rlx/analysis/report.py` — one command that reads the results directory and
  writes `report/results.md`, a single file containing every number the report
  quotes: both hypothesis verdicts, the per-instance correlations, IQM scores,
  performance profiles, probability of improvement, rank stability, the winner
  per maze, and the full per-run table.
- `report/sections/` — one file per hand-written section, so the three of us can
  write at the same time without editing the same file. Max's
  `03-strategies.md` goes here.

```bash
python -m rlx.analysis.report --results results --out report/results.md
```

**Why:** the last two days before the deadline are for writing sentences, not for
opening CSV files and hunting for a number. The rule that follows from this is
short: **never retype a number into the report — copy it out of
`report/results.md`.** A number typed from memory is a number that can drift away
from the data without anyone noticing.

**Three decisions inside it worth knowing about:**

1. **The file states the verdicts, it does not leave them to the eye.** Our first
   hypothesis needs *two* things to be true at once — the confidence interval has
   to exclude zero *and* the effect has to get stronger on harder mazes — and the
   second hypothesis needs the task-relevant correlation to be both larger *and*
   separated from the raw one by non-overlapping intervals. Printing the
   ingredients and letting a reader eyeball the conclusion is exactly how a
   report ends up claiming something the data does not say. Both verdicts are
   computed and printed as `True` or `False`.

2. **A confidence interval built from fewer than three mazes is marked
   unquotable.** If nearly every run scores the same, only one or two mazes have
   anything left to correlate, and the interval is then built by resampling one
   or two numbers. It still prints, and it can still say "excludes zero" — which
   would look like a confirmed result and would not be one. When that happens the
   file now prints a warning right next to the number instead of a footnote.

3. **Probability of improvement is reported per family, never pooled.** Mixing
   Empty with MultiRoom would average over two very different tasks and hide
   exactly the difficulty effect we are trying to measure.

**What it means for the results:** nothing changes numerically. This is a
presentation layer over the statistics we already had — it recomputes nothing and
decides nothing. Two things it *adds* to what we can see: performance profiles
and probability of improvement were required by the design (spec §7.2) and until
now existed only as functions nobody called outside a figure.

**How we know it works:** run against the 260 fake runs it produces a 500-line
file with no empty section and no missing number, and against a deliberately
broken copy — one maze where every run was forced to score zero — it reports the
missing correlation as `NaN` and carries on instead of crashing. It also runs
against the real pilot results, which is where the two bugs below turned up.
`pytest -q`: **233 passed**.

**Two bugs, both caught by re-checking before the PR, both on real data:**

1. **The winners table named a winner where there was none.** It ranked
   strategies by their *mean*, and picked the first row after sorting. On the
   pilot that printed "boltzmann" as the best strategy on DoorKey-5 — a maze
   where all four strategies scored exactly 0.0 and nothing ever reached the
   goal. On Empty-5 it printed epsilon-greedy, where epsilon-greedy and
   NoisyNets had scored *exactly* the same. Both are ordering artefacts, and
   both would have been written into the report as results.

   Fixed: the table ranks by IQM (the same statistic as the rank-stability
   table, so the two cannot contradict each other), names every tied strategy
   instead of one of them, and says "no strategy ever reached the goal" when
   the best score is zero.

2. **A missing run would have been discovered two minutes in.** The generator
   now refuses an incomplete run matrix before it computes anything, the same
   way the figures do.

**What has not happened yet:** the real numbers. Steps 7 to 9 of the task (run
both commands against the real results, read them, and write down what they
showed) are for 2026-08-22, after the sweep. The section of this log that
records what the experiment actually found is still unwritten, on purpose.
