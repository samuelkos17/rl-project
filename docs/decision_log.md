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
