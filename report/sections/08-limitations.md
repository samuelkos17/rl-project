# 8. Limitations

The design of this study fixes one algorithm and varies one component, which buys
a clean comparison at the cost of a narrow one. This section states what the
results therefore cannot support. Several of these limits are measured rather
than merely acknowledged, because a reader can reasonably ask how large they are.

## 8.1 The count-based bonus cannot always tell two places apart

Section 5.5 explains why the count-based bonus counts the agent's 7x7x3
observation rather than its true position: using the position would hand one of
the four strategies privileged information that the other three do not receive.
The price is perceptual aliasing — two different places that look identical share
a single count, so the second of them is not treated as novel.

The effect is not small, and it is very uneven across our environments. Measuring
every attainable `(x, y, direction)` state on each instance and counting how many
distinct observations they produce, in the initial configuration and on the
layout of seed 0:

| Instance | States | Distinct views | Collapse | Largest group |
|---|---|---|---|---|
| Empty-5 | 32 | 26 | 19% | 2 |
| Empty-8 | 140 | 86 | 39% | 3 |
| Empty-16 | 780 | 109 | **86%** | **252** |
| DoorKey-5 | 24 | 21 | 12% | 4 |
| DoorKey-6 | 48 | 45 | 6% | 4 |
| DoorKey-7 | 80 | 73 | 9% | 4 |
| DoorKey-8 | 120 | 106 | 12% | 4 |
| DoorKey-10 | 224 | 173 | 23% | 4 |
| MultiRoom-N2 | 72 | 64 | 11% | 4 |
| MultiRoom-N3 | 112 | 85 | 24% | 8 |
| MultiRoom-N4 | 180 | 149 | 17% | 12 |
| MultiRoom-N5 | 208 | 177 | 15% | 16 |
| MultiRoom-N6 | 228 | 191 | 16% | 20 |

Empty-16 is the extreme case: 780 states collapse into 109 distinct views, and a
single view — an unobstructed patch of empty floor — is shared by 252 of them. On
that instance the bonus is close to blind, since almost everywhere in the middle
of the room looks the same as everywhere else. The structured environments fare
much better, typically losing 10–25%, because walls and objects break the view up.

Two caveats on the table itself. It is measured in the initial configuration, so
in DoorKey it does not capture how the view changes once the key has been removed
from the floor and the door opened; during a run some of these groups split and
others merge. And it is one layout per instance, that of seed 0.

The consequence for interpretation is specific: **where count-based exploration
underperforms, we cannot cleanly separate "the method is weak here" from "the
method cannot see this environment properly".** On Empty-16 in particular, a poor
result for count-based is at least as much a statement about aliasing as about
counting.

## 8.2 One bonus scale had to serve thirteen environments

The size of the count-based bonus, `count_beta`, is a single constant applied
everywhere, and no single value is right everywhere. The bonus is paid per step,
while MiniGrid grants very different episode lengths — 640 steps for DoorKey-8
against 80 for MultiRoom-N4 — so long-episode environments accumulate far more of
it. Measured with a random policy before any real run, the novelty reward
collected per episode compared against the 0.9 available for solving the maze was
1.6x on Empty-5 and MultiRoom-N4, 2.1x on MultiRoom-N6, and 14.1x on DoorKey-8.

At the original setting the agent would have been paid roughly fourteen times
more for sightseeing than for winning on DoorKey. We reduced `count_beta` from
0.05 to 0.01, which brings the worst case down to about 3x while keeping the
bonus meaningful on MultiRoom, where we most want it working. Sizing it for
DoorKey instead — about 0.0035 — would have made it nearly invisible on
MultiRoom. It is a compromise chosen on this scale argument alone.

Two things follow. First, count-based results on the long-episode DoorKey
instances remain the least favourable to the method, and part of that is our
constant rather than the method. Second, and more importantly, **this value was
fixed before any experiment ran and was not revised afterwards.** Adjusting it
after seeing which strategy won would have meant choosing our own result.

## 8.3 Pinned layouts: single-maze exploration, not generalisation

MiniGrid regenerates its maze on every reset. We pin the layout to the run's seed
so that one run sees exactly one maze, because state coverage needs a fixed
denominator to be a fraction of anything. The consequence is that we study how
thoroughly an agent explores *one* environment, not whether exploration helps it
generalise across environments — a related but distinct question this study does
not address.

The five seeds give five layouts per instance on DoorKey and MultiRoom. **On the
Empty family they do not**: `Empty-N` produces the same start, goal and walls for
every seed, so the five runs differ in network initialisation and action sampling
but face an identical maze. Seed variation there measures the agent's variance,
not the environment's.

Because the layout is pinned and evaluation is greedy, MiniGrid is deterministic
at evaluation time, so each evaluation is a single episode; ten would return ten
identical numbers. Individual learning curves are therefore step functions, and
the smooth curves in Section 6 come from aggregating across seeds.

## 8.4 One algorithm, one setting of every other knob

Every result here is a statement about Double DQN with the hyperparameters in
Section 4, not about exploration strategies in general. Three restrictions are
worth naming:

- **One algorithm.** Double DQN was fixed because vanilla DQN's overestimation
  acts as an accidental exploration bonus that would contaminate exactly this
  comparison. Whether the ranking survives a different value-based method, or a
  policy-gradient one, is untested.
- **No per-strategy tuning.** Learning rate, buffer size, batch size and the rest
  are identical for all four strategies. This is what makes the comparison
  controlled, but it also means each strategy is evaluated at settings that were
  not chosen to suit it. A strategy that would win after tuning may lose here.
- **Each strategy has its own schedule constants**, calibrated against measured
  quantities rather than tuned for score: the Boltzmann temperatures against the
  spread of the network's action values at initialisation and after training, the
  count bonus as in Section 8.2, and the NoisyNets noise scale left unchanged
  after measuring how often the injected noise actually changes the chosen
  action. Every one of these was fixed before the experiments and none was chosen
  by looking at which strategy scored better.

## 8.5 A fixed step budget

Every instance receives the same 400,000 environment steps, deliberately: giving
harder environments more steps would let difficulty and budget vary together and
would make the difficulty curve uninterpretable. The cost is that an instance
where every strategy scores zero is ambiguous between "no strategy can solve this"
and "no strategy can solve this *within 400,000 steps*". We report such instances
as failures at this budget and not as evidence of impossibility.

## 8.6 Where the coverage measure itself runs out

The explanatory variable has two blind spots of its own, both structural.

**On the smallest instances it saturates before the measurement window closes.**
Empty-5 has nine walkable cells and 32 loggable states; in the pilot, every
strategy had seen all 32 within the first thousand steps and never gained another.
Early coverage is then the same number for all four strategies and can predict
nothing. This is a property of the maze, not a defect of the measure — the
instance is too small for the question to have an answer — but it means the
hypothesis of Section 6.4 is genuinely testable only where the environment is
large enough for strategies to differ.

**On the whole Empty family the raw/task-relevant comparison is undefined.** As
Section 5.3 shows, the two masks are identical there for every seed, so three of
the thirteen instances contribute no evidence to the comparison in Section 6.5
and pull the two correlations towards each other. The comparison rests on the
DoorKey and MultiRoom instances, where the ratio falls as low as 0.46.

**Finally, the goal cell is never observed.** Because the agent's position is
recorded before it acts and reaching the goal ends the episode immediately, the
one cell that defines success is the one cell coverage cannot see. It is excluded
from the measure (Section 5.2) rather than counted as permanently unvisited, but
it remains true that our notion of "visited the whole maze" means "visited all of
it except the exit".

---

*Reproducing the numbers in this section.* The aliasing table is measured by
setting `agent_pos` and `agent_dir` for every loggable state and hashing
`gen_obs()["image"]`; it and the mask ratios are recorded in
`docs/decision_log.md` under "How much of each maze looks the same to the agent".
The bonus-scale measurements are in the same file under "Measuring Max's
count-bonus question against the real mazes", and the Empty-5 saturation
observation under "On the smallest maze our main measurement cannot tell anything
apart".
