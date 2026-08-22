# report/sections/

One markdown file per report section, so the three of us can write in parallel
without touching the same file. Samuel assembles them into
`report/tex/report.tex`, in filename order.

Section 3 is split into `03a`/`03b`/`03c` because three people write it. The
`# 3. Approach` heading lives in `03a` only; `03b` and `03c` start at their
own subsection heading.

**The report is four pages.** Read `report/outline.md` before writing — it has
the page arithmetic, the section mapping and the per-section word budgets. The
budgets are hard.

| File | Template section | Owner | Budget | State |
|---|---|---|---|---|
| `00-abstract.md` | Abstract | Samuel | 110 | **drafted 2026-08-22 (118 w)** |
| `01-introduction.md` | 1. Introduction | Samuel | 280 | **drafted 2026-08-22 (290 w)** |
| `02-related-work.md` | 2. Related Work | Samuel | 170 | **drafted 2026-08-22 (152 w)** |
| `03a-approach-setup.md` | 3.1 Agent and protocol | Samuel | 160 | **drafted 2026-08-22 (151 w)** |
| `03b-approach-strategies.md` | 3.2 The four strategies | **Max** | 200 | not written |
| `03c-approach-coverage.md` | 3.3 Coverage measurement | **Daniel** | 120 | not written |
| `04-experiments.md` | 4. Experiments | Daniel | 330 | not written |
| `05-discussion.md` | 5. Discussion | all three | 230 | not written |

## The two long drafts

`05-coverage-measurement.md` (1,747 w) and `08-limitations.md` (1,538 w) were
written against the old nine-section outline, before we had the template. They
are correct and they carry measurements recorded nowhere else, so **they stay**
— but as working notes, not as report text. Their content has to be distilled
into ~120 words inside Approach and ~80 words inside Discussion respectively.
That is Daniel's call to make, not something to do for him.

`01-introduction.md` previously held a 1,586-word draft aimed at the old
outline. It was rewritten to budget on 2026-08-22; the long version is not kept,
because everything in it that was not a number already appears in
`docs/decision_log.md`.

## Rules

Numbers are never typed from memory. They come from `report/results.md`:

```
python -m rlx.analysis.report --results results --out report/results.md
```

## LaTeX-safety

These files are assembled into `report.tex`, so they are kept compilable:
`\%` not `%` (a bare percent comments out the rest of the line and the text
silently vanishes), the LaTeX `\times` macro rather than the Unicode
multiplication sign, and ASCII hyphens inside `$...$` rather than the Unicode
minus U+2212, which the T1 font encoding cannot set and which stops the build.
Em dashes and `§` are fine.

**Check for eaten backslashes before pasting into Overleaf.** On 2026-08-22 a
shell heredoc turned `\times` into a literal tab character in two files, which
renders in LaTeX as an italic "imes". Scan for control characters:

```
python -c "from pathlib import Path; [print(f.name,i,repr(l)) for f in Path('report/sections').glob('*.md') for i,l in enumerate(f.read_text(encoding='utf-8').splitlines(),1) if any(ord(c)<32 for c in l)]"
```

`references.bib` holds seven entries pulled from dblp on 2026-08-22 (via the
`dblp.uni-trier.de` mirror; `dblp.org` itself refused connections). Conference
versions throughout, never arXiv preprints.
