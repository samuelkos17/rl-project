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

**Status:** Open — resolved by the day-1 benchmark

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
