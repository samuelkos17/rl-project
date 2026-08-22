# Report outline — rebuilt against the LaTeX template, 2026-08-22

The template is `report/tex/report.tex` (a NeurIPS'24 variant via `adrl.sty`).
**Read this file before writing a word.** The previous version of this outline
described a nine-section report and was written before we had the template. It
did not match, and following it would have produced roughly four times too much
text.

Deadline: **2026-08-23**.

---

## 1. The hard constraint: four pages

The template is single-column, `textwidth=5.5in`, `textheight=9in`, 10pt type on
11pt leading. References and the checklist come after `\newpage` and do not
count.

**The four-page limit applies to TEXT only. Figures and tables do not count
against it,** so the finished document may run past four pages once they are
placed. Confirmed 2026-08-22. An earlier version of this file assumed figures ate
the budget and told everyone to drop one — that was wrong. Do not cut figures to
save space.

**Measured, not estimated.** Samuel compiled the abstract, introduction and
related work — 560 words — in Overleaf: **1.5 pages**. Backing out the fixed
costs (title block ~2.2in, abstract block ~1.4in, three headings ~0.9in), the 442
words of body prose fill ~9.0in:

> **~49 words per vertical inch, i.e. ~440 words per full page of prose.**

The earlier estimate in this file was ~78 words/inch, optimistic by about 60%:
the words-per-line figure was too high and paragraph spacing was not counted.

| | vertical inches |
|---|---|
| 4 pages x 9in of text | **36.0** |
| less title block | -2.2 |
| less abstract block | -1.4 |
| less ~6 section and subsection headings | -2.4 |
| **= left for prose** | **30.0** |

30.0in x 49 words/in is about **1,470 words of prose**, plus the abstract. The
per-section budgets in section 3 total 1,600 including the abstract, which lands
within a few percent, so **those budgets are correct and stand.**

Checked against the real page count: 1.5 pages used, 2.5 remain (22.5in). Less
~2.0in of remaining headings leaves ~1,004 words for Approach + Experiments +
Discussion, which are budgeted 1,040. No cuts required.

---

## 2. The template has five sections, not nine

Our old outline had nine. They map like this. **This remapping is the main
change and it affects all three of us.**

| Template section | Absorbs our old sections | Owner |
|---|---|---|
| Abstract | — | Samuel |
| 1. Introduction | old §1 (intro + hypotheses) and §2 (background) | Samuel |
| 2. Related Work | **new — we had no content for this** | Samuel |
| 3. Approach | old §3 (strategies), §4 (setup), §5 (coverage measurement) | Max + Samuel + Daniel |
| 4. Experiments | old §6 (results) | Daniel |
| 5. Discussion | old §7 (discussion), §8 (limitations), §9 (conclusion) | all three |

**"Related Work" is a genuine gap.** Neither the old outline nor any draft has
it, the template asks for it explicitly, and the professor's feedback says "it is
likely that there are at least 2-3 related works you want to look at". The
proposal already cites three: Bellemare et al. 2016 (count-based), Fortunato et
al. 2018 (NoisyNets), Agarwal et al. 2021 (rliable). `references.bib` currently
contains one unrelated placeholder entry and needs all of them added, cited from
**dblp.org, not arXiv** — the template says this explicitly and it is the kind of
thing that gets noticed.

---

## 3. Word budget per section

| Section | Words | Owner | Must contain |
|---|---|---|---|
| Abstract | 110 | Samuel | question, what we did, the three verdicts, the one surprising finding |
| 1. Introduction | 280 | Samuel | the question, H1/H2/H3, **what would confirm each** |
| 2. Related Work | 170 | Samuel | Bellemare, Fortunato, Agarwal + how our idea relates |
| 3. Approach | 480 | split across three files | see the three rows below |
| &nbsp;&nbsp;3.1 — `03a` | 160 | Samuel | **done, 151 w** — Double DQN fixed, hyperparameters, pinned layouts, greedy extrinsic-only evaluation |
| &nbsp;&nbsp;3.2 — `03b` | 200 | **Max** | one paragraph per strategy, each with its exact schedule from `config.py`; the professor asked for schedules by name, and for how each bonus interacts with evaluation return |
| &nbsp;&nbsp;3.3 — `03c` | 120 | **Daniel** | raw vs task-relevant coverage, and the privileged-information statement |
| 4. Experiments | 330 | Daniel | setup numbers, the four figures, what they show |
| 5. Discussion | 230 | all three | what held, what did not, the evaluation jam, limitations, future work |
| **Total** | **1,600** | | ~50 words over budget; absorbed by dropping one figure if needed |

**These are hard budgets, not targets to exceed.** If a section needs more, it
takes the words from another section and that has to be agreed, not assumed.

---

## 4. What happens to the long drafts

`report/sections/*.md` currently holds three drafts totalling 4,871 words:

| File | Words | Status |
|---|---|---|
| `01-introduction.md` | 1,586 → rewritten to 280 | Samuel, done 2026-08-22 |
| `05-coverage-measurement.md` | 1,747 | **Daniel — needs cutting to ~120 words inside Approach** |
| `08-limitations.md` | 1,538 | **Daniel — needs cutting to ~80 words inside Discussion** |

**Do not delete the long versions.** They are correct, they carry measurements
that exist nowhere else, and they are the source the short version is distilled
from. They stay in `report/sections/` as internal documentation. The four-page
report is written into `report/tex/report.tex` and cites them by nothing — the
reader never sees them.

**Writing order.** Keep writing into `report/sections/*.md`, one file per
section — that is what lets three people write at once without fighting over one
`.tex` file on the last day, and it is why the directory exists. But **assemble
into `report.tex` today, not tomorrow**: the page count is the binding
constraint and it is invisible until the document compiles. Assemble early,
compile, measure, and cut against the real page count rather than against the
word estimate in §1 above, which is an estimate.

Samuel assembles. Everyone else writes markdown to their word budget.

---

## 5. Figures: pick 4 of 7

Our figures are wide and short, which is good for page economy — at full
`5.5in` width they are only 1.5–2.3in tall.

| Figure | At 5.5in wide | Recommendation |
|---|---|---|
| `fig4_coverage_vs_return` | 1.9in | **Include** — this is H1, the central claim |
| `fig5_iqm` | 2.2in | **Include** — which strategy wins, with CIs |
| `fig2_difficulty_curve` | 1.7in | **Include** — performance vs difficulty |
| `fig6_rank_stability` | 1.6in | **Include** — H3, the one hypothesis supported |
| `fig1_learning_curves` | 1.7in | Cut — 13 instances at this width will be unreadable |
| `fig3_coverage_curves` | 2.3in | Cut |
| `fig7_visitation_heatmaps` | 1.5in | Cut — nice to look at, carries no verdict |

**All four stay.** Figures do not count against the page limit (see section 1),
so there is no reason to drop one. Each also answers a point the professor
raised by name: the coverage-versus-performance curve, rliable-style intervals,
the continuous difficulty axis, and rank stability.

**Unverified: nobody has checked these are legible at 5.5in.** They were drawn
without a target width. Before committing to four, open the PDFs at print size
and read the axis labels. The template says explicitly: "Don't make us zoom in to
read your axis descriptions!" If a figure fails, either redraw it narrower or
cut it — do not shrink it further.

Use the **PDF** versions, not the PNGs. Both are in `report/figures/`.

---

## 6. Rules that do not change

**Never retype a number by hand.** Every number comes from
`report/results.md`, regenerated by:

```
python -m rlx.analysis.report  --results results --out report/results.md
python -m rlx.analysis.figures --results results --out report/figures
```

**H1 and H2 are reported under two definitions of performance** (`final_return`
and `success_rate`). Both give the same verdict, so in a four-page report this is
one sentence, not a subsection — but it must be there, because it is what makes
the null result credible. See `docs/decision_log.md`, 2026-08-22.

**The correlation is computed within each environment instance, never pooled.**
See CLAUDE.md §9.

---

## 7. The checklist at the end of the template

It does not count toward the page limit and is not mandatory, but it is cheap
marks and it is a reproducibility checklist we can mostly answer yes to. One
answer needs care:

> "Did you run at least 10 repetitions of your method?"

We ran **5 seeds** per configuration, 260 runs total. The honest answer is
`\answerNo{}` with a one-line note that the budget went into 13 environment
instances rather than more seeds per instance. Do not answer yes.
