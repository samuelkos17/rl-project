# Glossary — every term in this project, in plain language

For us three. If you hit a term in the code, the spec, or a paper and it is not
here, add it. Terms are grouped by where you will run into them.

---

## The basics

**Agent** — the thing that acts. In our case, a neural network that looks at what
it can see and picks one of 7 actions.

**Environment** — the world the agent acts in. For us, a MiniGrid maze.

**Episode** — one complete attempt: the agent starts, acts until it reaches the
goal or runs out of time, then everything resets.

**Step** — one action. An episode is many steps. Training is many episodes.
When we say "400k steps" we mean 400,000 individual actions across however many
episodes that takes.

**Return** — the total reward collected in one episode. This is the score we
actually care about.

**Reward in MiniGrid** — you get `1 - 0.9 * (steps_taken / max_steps)` if you
reach the goal, and `0` otherwise. So returns are always between 0 and 1: 0 means
failure, and closer to 1 means you solved it faster. This is a **sparse** reward
— you get nothing at all until you succeed, which is exactly what makes
exploration hard.

**Policy** — the agent's rule for choosing actions. "What do I do in this
situation?"

**Greedy** — always take the action the agent currently believes is best, with no
randomness. Our evaluation is always greedy.

---

## Q-learning and DQN

**Q-value** — the agent's estimate of "how good is taking action *a* in situation
*s*, counting all future reward". Written `Q(s, a)`. The agent picks actions by
looking at Q-values.

**DQN (Deep Q-Network)** — using a neural network to estimate Q-values, instead
of a giant lookup table. The lookup table is impossible when there are too many
situations to list.

**Replay buffer** — a memory of past experiences (situation, action, reward, next
situation). Instead of learning from each experience once and forgetting it, the
agent stores it and re-learns from random batches of old experiences. This makes
learning much more stable. Ours holds 100,000 experiences.

**Batch** — a group of experiences (32 of them for us) sampled from the replay
buffer and learned from all at once.

**Target network** — a frozen copy of the Q-network that gets updated only every
1000 steps. Without it, the network chases its own moving estimates and training
blows up. Think of it as a slowly-updating reference point.

**Gamma (discount factor)** — how much the agent cares about future reward versus
immediate reward. Ours is 0.99, meaning it cares about the long run almost as
much as right now.

**Double DQN** — the variant we use. Plain DQN systematically **overestimates**
how good actions are, because it always takes the maximum of noisy estimates and
noise sometimes points up. Double DQN fixes this by using one network to *pick*
the best action and a different network to *score* it.

We care about this for a specific reason: overestimated Q-values make the agent
optimistic about actions it has not tried much, which is **itself a kind of
exploration**. Since our whole project is about comparing exploration strategies,
we do not want a hidden extra exploration effect leaking in from the algorithm.

**Overestimation bias** — the problem Double DQN fixes, described just above.

**Gradient clipping** — capping how big a single learning update can be, so one
weird experience cannot wreck the network. Ours is capped at 10.

---

## Exploration — the actual topic

**Exploration vs. exploitation** — the core tension. *Exploitation* is doing what
you already believe is best. *Exploration* is trying something else to find out
whether there is something better. Too much exploitation and you never find the
goal; too much exploration and you never get good at reaching it.

**Epsilon-greedy** — our baseline. With probability epsilon, act completely at
random; otherwise act greedily. We start at epsilon = 1.0 (fully random) and
decay to 0.05 over the first 20% of training. Simple and surprisingly hard to
beat.

**Boltzmann exploration (softmax exploration)** — instead of "best action or a
totally random one", pick actions with probability proportional to how good they
look. An action the agent thinks is second-best gets chosen fairly often; an
action it thinks is terrible almost never does. Smarter than epsilon-greedy in
principle, because its random choices are not *uniformly* random.

**Temperature (tau)** — the knob controlling Boltzmann. High temperature makes
all actions nearly equally likely (lots of exploration). Low temperature makes it
nearly greedy. We decay tau from 1.0 to 0.05 over the first 40% of training.

**Intrinsic reward / bonus** — extra reward the agent gives *itself* for doing
something interesting (in our case, visiting somewhere unfamiliar), on top of the
real reward from the environment. It is a way of paying the agent to explore.

**Extrinsic reward** — the real reward from the environment. **Every score we
report is extrinsic only.** Intrinsic bonuses affect what the agent learns, but
they never count toward the numbers in our report.

**Count-based exploration** — one of our four strategies. Keep a tally of how
often each situation has been seen, and give a bonus of `beta / sqrt(N)` for
being in a situation seen `N` times. Rare situations pay well, familiar ones pay
almost nothing.

**NoisyNets** — our fourth strategy. Instead of adding randomness to the *action*
choice, add randomness to the network's *weights*. The amount of noise is
**learned** during training rather than following a schedule we picked, so the
network can decide for itself where it still needs to explore. At evaluation time
we switch the noise off.

**Perceptual aliasing** — when two genuinely different situations look identical
to the agent. This happens to us: our agent only sees a small 7x7 window, so two
different corridors can look exactly the same. It matters for the count-based
strategy, which counts what the agent *sees* rather than where it actually is.

---

## Our specific setup

**MiniGrid** — the family of small grid-world mazes we use. The agent is on a
grid, faces one of 4 directions, and turns, moves forward, picks things up, and
opens doors.

**Partial observability** — our agent does **not** see the whole maze. It sees a
7x7 grid of cells in front of it, with 3 numbers per cell (what object is there,
what colour, what state). That is the "7x7x3 observation". It does not know where
it is on the map.

**True state** — where the agent actually is: `(x, y, direction)`. We can read
this from the simulator, but **the agent never sees it**.

**Privileged information** — information we as researchers can see but the agent
cannot. `(x, y, direction)` is privileged. We use it **only for measuring
coverage**, never for training. The professor specifically asked us to be clear
about this, so it is worth repeating: *`(x, y, dir)` is used only for analysis,
never by any agent.*

**Empty / DoorKey / MultiRoom** — our three maze families, in increasing order of
how hard they are to explore. `Empty` is an open room (sanity check). `DoorKey`
requires finding a key, picking it up, unlocking a door, then reaching the goal —
a specific *sequence*. `MultiRoom` is a chain of rooms connected by doors, where
the goal is far away and you get no reward until you find it.

**Difficulty axis** — instead of three fixed mazes, we use a *range* of sizes
(DoorKey at sizes 5, 6, 7, 8, 10) and room counts (MultiRoom with 2 to 6 rooms).
This lets us plot performance as a smooth curve against difficulty, rather than
comparing three unrelated points.

---

## Coverage — our explanation variable

**State coverage** — how much of the maze the agent has actually visited. This is
the number we use to explain *why* one strategy beats another, not just *that* it
does.

**Raw coverage** — distinct `(x, y, direction)` states visited, divided by the
number of states that are reachable at all. A number between 0 and 1. "How much
of the maze has this agent ever seen?"

**Task-relevant coverage** — the same thing, but only counting places that
actually matter for solving the task: on or near the shortest route, or next to
the key, the door, or the goal. The idea is that wandering into an irrelevant
corner is exploration, but not *useful* exploration.

For DoorKey the route is start → key → door → goal, **not** start → goal, because
the door is locked.

**Reachable states** — all the places the agent could possibly get to (everything
that is not a wall), times 4 directions. This is the denominator for coverage. We
compute it with a breadth-first search.

**Breadth-first search (BFS)** — a standard algorithm for exploring a maze layout
outward from a starting point, level by level. We use it for two things: finding
which cells are reachable, and computing how far every cell is from the goal.

**Visitation array** — how we record coverage. A grid of counters, one per
`(x, y, direction)`, that goes up by one each time the agent is there. We save a
snapshot every 10,000 steps. It is tiny (about 1000 numbers per snapshot).

**Early-coverage AUC** — "AUC" is *area under the curve*. We plot coverage over
the first 20% of training and measure the area under that plot. A strategy that
explores broadly early has a big area; one that explores slowly has a small one.
**This is the number we correlate with final performance** — it is the heart of
the whole project.

---

## Statistics

**Seed** — a number that fixes all the randomness in a run, so the same seed
gives the same result twice. We run 5 seeds (0 to 4) per configuration, because a
single run can be lucky or unlucky and tell us nothing.

**Confidence interval (CI)** — a range that expresses how sure we are. "IQM 0.62,
CI [0.55, 0.68]" means the true value is probably somewhere in that range. If two
strategies' intervals overlap heavily, we cannot claim one is better.

**Bootstrap** — a way of computing confidence intervals by repeatedly resampling
the data we already have. No formula or distributional assumption needed, which
is why it is standard in RL.

**IQM (interquartile mean)** — throw away the best 25% and worst 25% of runs, then
average the middle 50%. Much less swayed by one freak run than a plain average,
and much more informative than a median. `rliable` recommends it.

**rliable** — the library and methodology from the paper we cite, for comparing RL
results properly with confidence intervals instead of single runs. It gives us
IQM, bootstrap CIs, performance profiles, and probability of improvement.

**Performance profile** — a plot showing, for every score threshold, what
fraction of runs beat it. Shows the entire distribution rather than compressing
it into one number.

**Probability of improvement** — "if I pick a random run of strategy A and a
random run of strategy B, how often does A win?" Easy to state in a report.

**Spearman rank correlation** — measures whether two things move together in
*rank order*, without assuming the relationship is a straight line. This is our
main test: do the strategies that explore more early also score higher at the
end? Ranges from -1 (perfect opposite) through 0 (unrelated) to +1 (perfect
agreement).

**Kendall's tau** — measures how similar two *orderings* are. We use it for rank
stability: is the ranking of the four strategies on a hard maze the same as on an
easy one? 1.0 means identical order, -1.0 means exactly reversed.

**Confound** — something that messes up a comparison by varying alongside the
thing you are actually measuring. We have one big one to avoid, described next.

**Our specific confound — read this one twice.** Both coverage and final score go
*down* as mazes get harder. So if we just correlate coverage against score across
all 260 runs mixed together, we get a big positive correlation that means
*nothing at all* — it only says "hard mazes are hard", measured two different
ways.

The fix: run the correlation **separately within each individual maze**, where
difficulty is the same for all runs, and only then combine the results. Any
analysis that mixes different mazes into one correlation is wrong.
