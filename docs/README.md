# docs/ — notes for us three

This directory is written **for humans**, specifically for the three of us. We are
not fluent in reinforcement learning terminology, so everything here is written in
plain language and any jargon gets explained the first time it shows up.

This is the opposite of `CLAUDE.md` and `docs/specs/`, which are written to be
read by Claude and are dense and precise on purpose. Both are maintained. Do not
merge them.

## What lives where

| File | What it is | Who reads it |
|---|---|---|
| `docs/README.md` | this file | us |
| `docs/glossary.md` | every RL term used in the project, explained plainly | us |
| `docs/decision_log.md` | every real change we made and why, **including the ones we threw away** | us |
| `docs/specs/` | the full technical design | Claude, and us when we need detail |

## The rule about the decision log

Every change that actually gets made goes into `docs/decision_log.md`. That
includes changes we later discard — and for those, the important part is
**writing down why we discarded it**. Six months from now, "we tried X and it
didn't work because Y" is worth more than the code was.

An entry looks like this:

```markdown
## 2026-08-18 — Short title of the change

**Status:** Active | Discarded | Superseded by <entry>

**What changed:** One or two sentences. Plain language.

**Why:** The reasoning. What problem this solved.

**What it means for the results:** Does this affect any number in the report?
If not, say "nothing" explicitly.

**Why discarded (only if discarded):** What went wrong, and what we did instead.
```

Do not batch a day's work into one vague entry. One entry per real decision.

## Writing style here

- Explain the term the first time you use it. "Replay buffer (a memory of past
  experiences the agent learns from repeatedly, instead of learning from each
  experience once and forgetting it)."
- If a sentence needs a second reading, rewrite it.
- Concrete numbers over adjectives. "Runs take 7 minutes" beats "runs are fast".
- If you are not sure about something, write that you are not sure. An honest
  "we think this is why, but we did not verify it" is far more useful than a
  confident guess we later build on.
