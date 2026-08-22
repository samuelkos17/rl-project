# report/sections/

One file per report section, so the three of us can write in parallel without
touching the same file. Samuel assembles them into `report/tex/report.tex` in
filename order.

**The finished sections are `.tex`, not `.md`.** They are LaTeX fragments that
get pasted straight into Overleaf, so they contain `\section{}`, `\textbf{}`,
`\emph{}` and `\citep{}` — never Markdown. Markdown `**bold**`, `*italic*` and
`#` headings print literally in LaTeX; that mistake reached Overleaf once
already on 2026-08-22.

**The report is four pages of text.** Figures and tables do not count against
that. Read `report/outline.md` before writing — it has the page arithmetic,
measured rather than estimated, and the per-section word budgets. The budgets
are hard.

| File | Template section | Owner | Budget | State |
|---|---|---|---|---|
| `00-abstract.tex` | Abstract | Samuel | 110 | **done (117 w)** |
| `01-introduction.tex` | 1. Introduction | Samuel | 280 | **done (287 w)** |
| `02-related-work.tex` | 2. Related Work | Samuel | 170 | **done (160 w)** |
| `03a-approach-setup.tex` | 3.1 Agent and protocol | Samuel | 160 | **done (155 w)** |
| `03b-approach-strategies.tex` | 3.2 The four strategies | **Max** | 200 | not written |
| `03c-approach-coverage.tex` | 3.3 Coverage measurement | **Daniel** | 120 | not written |
| `04-experiments.tex` | 4. Experiments | **Daniel** | 330 | not written |
| `05-discussion.tex` | 5. Discussion | all three | 230 | not written |

Section 3 is split three ways because three people write it. The
`\section{Approach}` line lives in `03a` only; `03b` and `03c` begin at their own
`\subsection`.

## Check before you paste

```bash
python scripts/check_report_tex.py
```

It reports the word count against each budget and fails on the four mistakes
that have actually bitten us:

1. **Control characters from an eaten backslash.** A shell heredoc turned
   `\times` into a literal tab twice on 2026-08-22, and it reached Overleaf as an
   italic "imes". Write these files with an editor, not with `sed` or a heredoc.
2. **Markdown syntax** — `**bold**`, `*italic*`, `#`.
3. **A bare `%`**, which comments out the rest of the line, so the text silently
   disappears rather than erroring.
4. **Non-ASCII characters.** Use `---` for an em dash and math mode for a minus.
   T1 cannot set U+2212 at all, and hand-stripping the Unicode em dashes has
   twice broken a sentence in place — once turning `module --- epsilon-greedy`
   into `module e(psilon-greedy`.

One thing the checker cannot catch: **consecutive lines are one paragraph in
LaTeX.** A list of short items written on separate lines runs together into a
blob unless you end each with `\\` or use a list environment.

## For Daniel, before writing 3.3 and 4

**Table 2 in `03a` already gives the coverage denominator per instance** (cells,
loggable states, all 13 rows). Reference `Table~\ref{tab:instances}` rather than
restating those numbers; spend the 120 words on the raw-vs-task-relevant
distinction and the privileged-information statement.

**The task-relevant fraction was deliberately cut from Table 2 and is yours to
place.** It was in the setup table briefly on 2026-08-22 and removed for two
reasons: it uses a term the report does not define until 3.3, and it is not a
setup fact in the way cell counts are. Regenerate it with:

```bash
python scripts/instance_table.py --task-relevant
```

Two things to carry over when you use it:

- **A fraction of 1.00 means H2 cannot be tested on that instance** --- raw and
  task-relevant coverage are literally the same number for every run there. It
  is 1.00 on Empty-5, Empty-8, Empty-16 and DoorKey-5, which is exactly the "4 of
  13 instances" caveat already in `report/results.md`. The two were computed by
  different code paths and agree, so this is a real cross-check.
- **Never quote it as a bare mean.** It swings hard between layouts of the same
  instance: DoorKey-10 runs 0.46 / 0.96 / 0.66 / 0.66 / 0.66 across seeds 0--4, a
  spread of 0.50, and DoorKey-8 spreads 0.37. The mean alone (0.68) hides that.

## The two long drafts

`05-coverage-measurement.md` (1,747 w) and `08-limitations.md` (1,538 w) were
written against the old nine-section outline, before we had the template. They
are correct and carry measurements recorded nowhere else, so **they stay** — as
working notes, not report text. Their content is distilled into ~120 words inside
Approach and ~80 words inside Discussion. That is Daniel's call to make.

## Rules

Numbers are never typed from memory. They come from `report/results.md`:

```bash
python -m rlx.analysis.report --results results --out report/results.md
```
