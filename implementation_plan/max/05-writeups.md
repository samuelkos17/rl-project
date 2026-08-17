# Task 5 — Strategy write-ups for the report

You built the four strategies, so you write the section of the report that
explains them. Daniel's report scaffold has a placeholder waiting for this.

Do this on **21.08**, while the sweep is running and there is nothing to code.

**Files:**
- Create: `report/sections/03-strategies.md`

**Interfaces:**
- Consumes: your four implementations, and `docs/glossary.md` for the plain-language
  definitions.
- Produces: a report section Daniel's scaffold includes.

---

- [ ] **Step 1: Write one subsection per strategy**

Create `report/sections/03-strategies.md`. For each of the four, cover exactly
these four things and nothing else:

1. **What it does** — two or three sentences, no equations.
2. **The equation** — the actual rule, with our parameter values in it.
3. **Our schedule/parameters** — the concrete numbers from `config.py`, not
   symbols. The professor asked us to state schedules explicitly, so this is
   graded.
4. **What we expect** — one sentence on how you expect it to behave, written
   *before* looking at results.

Structure:

```markdown
## 3. Exploration strategies

All four strategies implement one interface, so the training loop swaps them by
changing a single string. Every other component of the agent -- network,
optimiser, replay buffer, target network, and all hyperparameters -- is
identical across strategies.

### 3.1 Epsilon-greedy (baseline)

...

### 3.2 Boltzmann exploration

...

### 3.3 Count-based exploration bonus

...

### 3.4 NoisyNets

...

### 3.5 How intrinsic bonuses interact with evaluation

...
```

- [ ] **Step 2: Fill in the exact numbers from the config**

Do not write "epsilon decays over the early part of training". Write the numbers.
Read them out of the code rather than from memory:

```bash
python -c "
from rlx.config import RunConfig
c = RunConfig(env_id='Empty-5', strategy='epsilon_greedy', seed=0)
for f in ('epsilon_start','epsilon_end','epsilon_decay_frac','tau_start','tau_end','tau_decay_frac','count_beta','count_epsilon','noisy_sigma0','total_steps'):
    print(f'{f:<20} {getattr(c, f)}')
"
```

If any number in your section disagrees with that output, the section is wrong.

- [ ] **Step 3: Write section 3.5 carefully**

This one answers a direct question from the professor's feedback ("say how each
intrinsic bonus interacts with the evaluation return"), so it is worth getting
right. It must state:

- Only count-based produces an intrinsic bonus; the other three are zero.
- The bonus is added **only** to the reward stored in the replay buffer.
- Evaluation is greedy, on extrinsic reward only, with the bonus off and
  NoisyNets noise off.
- Therefore no reported number anywhere in the report contains intrinsic reward.
- The measured mean bonus during training was `<X>` versus a task reward of about
  0.9 — read the real value out of the results:

```bash
python -c "
import pandas as pd, glob
fs = glob.glob('results/*/count_based/*/metrics.csv')
vals = [pd.read_csv(f).mean_bonus.mean() for f in fs]
print(f'mean intrinsic bonus across {len(fs)} count_based runs: {sum(vals)/len(vals):.4f}')
"
```

- [ ] **Step 4: Add the privileged-information paragraph**

In §3.3, state the deviation plainly — do not hide it, it is a strength:

> Our count-based bonus counts the agent's own 7x7x3 observations rather than true
> `(x, y, direction)` states. Counting true states would give this strategy
> information the agent never receives and the other three strategies do not
> have, which would invalidate the controlled comparison. The cost is perceptual
> aliasing: distinct positions producing identical local views are counted as one.
> Throughout this work, `(x, y, direction)` is used only for analysis, never by
> any agent.

- [ ] **Step 5: Review it against the glossary**

Every term you use that a non-expert would not know should either be in
`docs/glossary.md` or explained inline. Read your section once as if you had not
written it.

- [ ] **Step 6: Log and commit**

Append to `docs/decision_log.md`:

```markdown
## 2026-08-21 — Strategy section written for the report

**Status:** Active

**What changed:** Wrote the report section explaining all four exploration
strategies, with the exact numbers read out of the config rather than from
memory.

**Two things in it that answer the professor directly:** the temperature schedule
for Boltzmann is written out explicitly (he asked for that), and there is a
subsection saying exactly how the count-based bonus interacts with the scores we
report — namely that it does not, because it only ever touches the replay buffer
and evaluation runs with it switched off.

**Measured while writing it:** the average intrinsic bonus across all
count-based runs was <X>, against a task reward of about 0.9. <One sentence on
whether that is comfortably small, which is what we wanted.>

**What it means for the results:** Nothing changes. But if any number in that
section ever disagrees with `src/rlx/config.py`, the section is the one that is
wrong.
```

```bash
git add report/sections/03-strategies.md docs/decision_log.md
git commit -m "docs: report section on the four exploration strategies"
```
