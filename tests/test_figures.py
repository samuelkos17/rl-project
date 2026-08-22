import shutil
import warnings
import subprocess
import sys
from pathlib import Path

import pytest

from rlx.analysis.figures import (
    COLORS, FIGURE_NAMES, _degenerate_note, make_all_figures,
)
from rlx.exploration import STRATEGIES

#: Resolved from this file, not the working directory, so the suite passes no
#: matter where pytest was launched from.
GENERATOR = Path(__file__).resolve().parents[1] / "scripts" / "make_synthetic_results.py"


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    subprocess.run([sys.executable, str(GENERATOR), "--out", str(out)], check=True)
    return out


@pytest.fixture(scope="module")
def rendered(synthetic, tmp_path_factory):
    """Render once for the whole module. Seven figures over 260 runs is not
    something to repeat per test."""
    out = tmp_path_factory.mktemp("figs")
    make_all_figures(synthetic, out)
    return out


def test_there_are_exactly_seven():
    assert len(FIGURE_NAMES) == 7
    assert len(set(FIGURE_NAMES)) == 7


def test_all_seven_figures_are_produced_in_both_formats(rendered):
    """PDF for the report (vector), PNG for the poster and for looking at."""
    for name in FIGURE_NAMES:
        assert (rendered / f"{name}.pdf").exists(), name
        assert (rendered / f"{name}.png").exists(), name


def test_figures_are_not_blank(rendered):
    """A near-empty PDF means the plot silently drew nothing."""
    for name in FIGURE_NAMES:
        assert (rendered / f"{name}.pdf").stat().st_size > 5_000, name


def test_every_strategy_has_its_own_colour_and_the_baseline_is_grey(rendered):
    """One colour per strategy, identical in every figure, so the reader learns
    the mapping once. Grey is reserved for epsilon-greedy: it is the baseline."""
    assert set(COLORS) == set(STRATEGIES)
    assert len(set(COLORS.values())) == len(STRATEGIES)
    r, g, b = (int(COLORS["epsilon_greedy"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    assert r == g == b, "the baseline colour must be neutral grey"


def test_the_command_line_entry_point_writes_the_figures(synthetic, tmp_path):
    """The report is regenerated with one command, so that command must work."""
    subprocess.run([sys.executable, "-m", "rlx.analysis.figures",
                    "--results", str(synthetic), "--out", str(tmp_path)], check=True)
    assert (tmp_path / f"{FIGURE_NAMES[0]}.png").exists()


def test_an_empty_results_directory_is_refused(tmp_path):
    """Silently writing seven empty axes would look like a rendering bug hours
    later; say what is wrong at the point where it is knowable."""
    with pytest.raises(ValueError, match="no runs"):
        make_all_figures(tmp_path, tmp_path / "out")


def test_nothing_is_written_when_the_run_matrix_has_a_hole(synthetic, tmp_path):
    """fig5 needs a complete (seeds x instances) matrix, and it used to discover
    a missing run only when it got there -- after fig1 to fig4 were already on
    disk. report/figures/ would then hold four fresh figures beside three stale
    ones from an earlier render, and the report would show two different
    datasets side by side. Fail before writing anything instead."""
    incomplete = tmp_path / "runs"
    shutil.copytree(synthetic, incomplete)
    shutil.rmtree(incomplete / "DoorKey-8" / "noisy" / "seed3")
    out = tmp_path / "figs"

    with pytest.raises(ValueError, match="missing"):
        make_all_figures(incomplete, out)

    assert not list(out.glob("*.pdf")) and not list(out.glob("*.png")), \
        "a refused render must leave no half-written figure set behind"


@pytest.fixture(scope="module")
def two_instances(synthetic, tmp_path_factory):
    """The pilot's shape: one instance in each of two families, no MultiRoom.

    configs/pilot.yaml really produces this, so it is a supported input, not a
    hypothetical.
    """
    out = tmp_path_factory.mktemp("pilotshape")
    for name in ("Empty-5", "DoorKey-5"):
        shutil.copytree(synthetic / name, out / name)
    return out


def test_a_missing_family_leaves_no_empty_panel_and_keeps_the_legend(two_instances, tmp_path):
    """The pilot runs two of the three families, so fig1 and fig2 drew a third
    panel with no data in it -- autoscaled to nonsense ticks -- and the legend,
    which sat on the last panel, landed inside that empty one and vanished.

    A single instance per family also handed matplotlib one-element pandas
    Series, which it deprecates and will later reject outright.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_all_figures(two_instances, tmp_path)

    messages = [str(w.message) for w in caught]
    assert not [m for m in messages if "No artists with labels" in m], messages
    assert not [m for m in messages if "single element Series" in m], messages
    for name in FIGURE_NAMES:
        assert (tmp_path / f"{name}.png").exists(), name


def test_a_family_with_no_ranking_is_labelled_not_left_blank():
    """Kendall's tau is undefined when every strategy scores the same, which the
    pilot hits on DoorKey-5 and the real sweep may hit on DoorKey-10 and
    MultiRoom-N6. An empty panel reads as a broken plot; the project's rule is
    "no variance, excluded", never a silent NaN."""
    import numpy as np
    import pandas as pd

    degenerate = pd.DataFrame({"difficulty": [5], "tau": [np.nan]})
    partial = pd.DataFrame({"difficulty": [5, 8], "tau": [np.nan, 1.0]})
    ranked = pd.DataFrame({"difficulty": [5, 8], "tau": [1.0, 0.5]})

    assert _degenerate_note(degenerate) is not None
    assert "variance" in _degenerate_note(degenerate)
    assert _degenerate_note(partial) is None
    assert _degenerate_note(ranked) is None


# --- report/results.md -------------------------------------------------------
# Task 6's generator lives next to the figures because it reads the same results
# tree through the same fixture, and building that tree twice costs more than
# the two modules are worth separating.

REPORT_HEADINGS = ("# Results", "H1 -- does early coverage", "H2 -- is task-relevant",
                   "IQM final return", "Performance profiles",
                   "Probability of improvement", "Rank stability",
                   "Best strategy per instance", "Full per-run table")


def test_report_generation_produces_every_section(synthetic, tmp_path):
    from rlx.analysis.report import build_report

    out = tmp_path / "results.md"
    build_report(synthetic, out)
    text = out.read_text(encoding="utf-8")
    for heading in REPORT_HEADINGS:
        assert heading in text, heading
    assert len(text) > 2_000


def test_the_report_states_both_hypothesis_verdicts(synthetic, tmp_path):
    """The verdicts must be computed, not left to the reader's eye. H1 needs the
    CI to exclude zero AND the trend to be positive; H2 needs a larger
    correlation AND non-overlapping CIs. Printing the ingredients without the
    verdict is how a report ends up claiming the wrong one."""
    from rlx.analysis.report import build_report

    out = tmp_path / "results.md"
    build_report(synthetic, out)
    text = out.read_text(encoding="utf-8")
    assert "H1 confirmed (CI entirely above zero AND trend positive)" in text
    assert "H2 confirmed (larger AND non-overlapping CIs)" in text
    assert "95% CI lies entirely above zero: **" in text, \
        "the half-criterion is reported separately"


@pytest.fixture(scope="module")
def one_tied_instance(synthetic, tmp_path_factory):
    """Two instances of one family, one of which every run scored 0.0 on.

    This is the shape the real sweep is expected to produce on the hard end of
    each family, so the report has to survive it rather than crash on the NaN.
    """
    import pandas as pd

    out = tmp_path_factory.mktemp("tied")
    for name in ("Empty-5", "Empty-8"):
        shutil.copytree(synthetic / name, out / name)
    for metrics in (out / "Empty-5").glob("*/seed*/metrics.csv"):
        df = pd.read_csv(metrics)
        df.loc[df["eval_return_mean"].notna(), "eval_return_mean"] = 0.0
        df.to_csv(metrics, index=False)
    return out


def test_an_instance_with_no_variance_is_reported_not_crashed_on(one_tied_instance, tmp_path):
    """An instance where every run scored the same has no correlation to report.
    That is a finding -- "nothing we tried solved this maze" -- and the file has
    to say it. It must also not print a NaN as `+nan`."""
    from rlx.analysis.report import build_report

    out = tmp_path / "results.md"
    build_report(one_tied_instance, out)
    text = out.read_text(encoding="utf-8")
    for heading in REPORT_HEADINGS:
        assert heading in text, heading
    assert "Instances with usable variance: 1 of 2" in text
    assert "+nan" not in text, "a missing number must read as NaN, not as +nan"


def test_a_ci_over_too_few_instances_is_marked_unquotable(one_tied_instance, tmp_path):
    """With one usable instance the bootstrap resamples a single number, so the
    interval is degenerate and can still print "excludes zero: True". The
    warning sits next to the number because that is where it gets read."""
    from rlx.analysis.report import build_report

    out = tmp_path / "results.md"
    build_report(one_tied_instance, out)
    assert "Do not quote that CI" in out.read_text(encoding="utf-8")


def test_the_report_command_line_entry_point_writes_the_file(synthetic, tmp_path):
    """`python -m rlx.analysis.report` is what gets run on 22.08, so it is what
    the test runs."""
    out = tmp_path / "nested" / "results.md"
    subprocess.run([sys.executable, "-m", "rlx.analysis.report",
                    "--results", str(synthetic), "--out", str(out)], check=True)
    assert out.exists()


def test_an_empty_results_directory_is_refused_by_the_report(tmp_path):
    from rlx.analysis.report import build_report

    with pytest.raises(ValueError, match="no runs"):
        build_report(tmp_path, tmp_path / "results.md")
    assert not (tmp_path / "results.md").exists()


def test_no_winner_is_named_on_an_instance_nothing_solved(one_tied_instance, tmp_path):
    """Every strategy scoring exactly 0.0 is a four-way tie, not a win. Sorting
    and taking the first row named one of them anyway -- on the pilot that put
    "boltzmann" in the winners column of DoorKey-5, which nothing solved."""
    from rlx.analysis.report import build_report

    out = tmp_path / "results.md"
    build_report(one_tied_instance, out)
    text = out.read_text(encoding="utf-8")
    # The winners section only -- "| Empty-5" also starts rows of the per-run
    # table further down, which is not what this test is about.
    section = text.split("## Best strategy per instance")[1].split("\n## ")[0]
    line = [l for l in section.splitlines() if l.startswith("| Empty-5 ")][0]
    assert "no strategy solved it reliably" in line, line
    # Nothing scored here at all, so the count column must say so. The label
    # alone cannot: it also appears where one trimmed-away seed did solve the
    # maze (2026-08-22, real DoorKey-7 had a run finishing at 0.976).
    assert "0 of 20" in line, line


def test_an_exact_tie_names_every_tied_strategy():
    """Not hypothetical: on the pilot, epsilon-greedy and NoisyNets both scored
    exactly 0.23875 on Empty-5 from completely different per-seed returns."""
    import pandas as pd
    from rlx.analysis.report import _winners_section

    df = pd.DataFrame({
        "env_id": ["Empty-5"] * 4,
        "strategy": ["epsilon_greedy", "epsilon_greedy", "noisy", "noisy"],
        "final_return": [0.23875, 0.23875, 0.4775, 0.0],
    })
    assert "epsilon_greedy = noisy" in _winners_section(df)


def test_the_winner_is_ranked_by_iqm_not_by_mean():
    """Spec 7.4 ranks by IQM, and so does the rank-stability table. Ranking the
    winners table by mean instead lets the two tables name different winners on
    the same instance: here one collapsed seed drags count-based's mean below
    Boltzmann's while its IQM stays above."""
    import pandas as pd
    from rlx.analysis.report import _winners_section

    df = pd.DataFrame({
        "env_id": ["DoorKey-8"] * 10,
        "strategy": ["count_based"] * 5 + ["boltzmann"] * 5,
        "final_return": [1.0, 1.0, 1.0, 1.0, 0.0] + [0.9] * 5,
    })
    assert df.groupby("strategy")["final_return"].mean().idxmax() == "boltzmann"
    assert "count_based" in _winners_section(df)
    assert "boltzmann" not in _winners_section(df)


def test_the_report_refuses_an_incomplete_run_matrix_before_writing(synthetic, tmp_path):
    """Same rule as the figures: a crashed run must not become a quiet blank in
    the results file, and the message that names it is worth more before two
    minutes of bootstrapping than after."""
    from rlx.analysis.report import build_report

    incomplete = tmp_path / "runs"
    shutil.copytree(synthetic, incomplete)
    shutil.rmtree(incomplete / "DoorKey-8" / "noisy" / "seed3")
    out = tmp_path / "results.md"

    with pytest.raises(ValueError, match="missing"):
        build_report(incomplete, out)

    assert not out.exists(), "a refused report must leave no half-written file"


def test_tables_run_easiest_to_hardest_within_each_family(synthetic, tmp_path):
    """REGRESSION TEST. Do not delete.

    Sorting instance names alphabetically puts DoorKey-10 before DoorKey-5 and
    Empty-16 before Empty-5, so every table in results.md ran in a different
    order from the figures printed beside it, with difficulty jumping around
    inside a family.
    """
    from rlx.analysis.report import build_report

    out = tmp_path / "results.md"
    build_report(synthetic, out)
    section = (out.read_text(encoding="utf-8")
               .split("### Per instance")[1].split("\n## ")[0])
    seen = [line.split("|")[1].strip() for line in section.splitlines()
            if line.startswith("| ") and "-" in line.split("|")[1]]
    ordered = list(dict.fromkeys(seen))

    assert ordered[:3] == ["Empty-5", "Empty-8", "Empty-16"], ordered
    assert ordered[3:8] == ["DoorKey-5", "DoorKey-6", "DoorKey-7", "DoorKey-8",
                            "DoorKey-10"], ordered
    assert ordered[8] == "MultiRoom-N2", ordered


def test_h2_names_the_instances_that_cannot_answer_it(synthetic, tmp_path):
    """On the Empty family raw and task-relevant coverage are the same number for
    every run -- every reachable cell is on some shortest path, so the two masks
    are identical (ratio 1.00 on all three instances). Those instances carry no
    evidence about H2 and pull the two correlations together, i.e. towards the
    "CIs overlap" verdict H2 fails on. The file has to say so where the verdict
    is, not leave it to whoever remembers STATUS.md.
    """
    from rlx.analysis.report import build_report

    out = tmp_path / "results.md"
    build_report(synthetic, out)
    section = (out.read_text(encoding="utf-8")
               .split("## H2 -- is task-relevant")[1].split("\n## ")[0])

    assert "cannot answer this question at all" in section
    for env_id in ("Empty-5", "Empty-8", "Empty-16"):
        assert env_id in section, env_id
    # And it names where the distinction DOES work, read off the same data --
    # on the pilot's two instances DoorKey-5 is itself tied, so a sentence
    # hard-coding "DoorKey" as the place to look would have been wrong there.
    assert "The instances that actually separate the two measures are" in section
    assert "DoorKey-8" in section.split("actually separate the two measures are")[1]


def test_identical_predictor_instances_are_detected_from_the_data(): 
    """Derived from the numbers, not from a hard-coded list of Empty instances:
    if the mask definition ever changes, the warning follows it."""
    import pandas as pd
    from rlx.analysis.report import _identical_predictor_instances

    df = pd.DataFrame({
        "env_id": ["Empty-5"] * 2 + ["DoorKey-8"] * 2,
        "early_auc_raw": [0.4, 0.6, 0.4, 0.6],
        "early_auc_task": [0.4, 0.6, 0.5, 0.9],
    })
    assert _identical_predictor_instances(df) == ["Empty-5"]


def test_curves_of_different_lengths_are_refused_not_truncated():
    """fig1 and fig3 average a whole family together. One short run used to
    silently shorten every curve in the panel while the axis still claimed the
    full run -- the figure drew, and drew the wrong thing."""
    import numpy as np
    from rlx.analysis.figures import _stack_curves

    assert _stack_curves([np.zeros(4), np.ones(4)], "fig1 Empty/noisy").shape == (2, 4)
    with pytest.raises(ValueError, match="disagree on their number of points"):
        _stack_curves([np.zeros(4), np.ones(3)], "fig1 Empty/noisy")


def test_the_report_states_h1_under_both_response_definitions(synthetic, tmp_path):
    """REGRESSION TEST. Do not delete.

    Decision of 2026-08-22: final_return blends how OFTEN the greedy policy
    reaches the goal with how WELL it does when it gets there, so the report
    states H1 under both definitions rather than picking the one that agrees
    with us. Dropping either half turns a transparent report into a choice we
    would then have to defend.
    """
    from rlx.analysis.report import build_report

    out = tmp_path / "results.md"
    build_report(synthetic, out)
    text = out.read_text(encoding="utf-8")

    assert "### Scored by `final_return`" in text
    assert "### Scored by `success_rate`" in text
    assert "Verdicts at a glance" in text
    # Four verdicts: two response definitions x two coverage measures.
    assert text.count("H1 confirmed (CI entirely above zero AND trend positive)") == 4


def test_a_lone_solved_seed_is_not_reported_as_nobody_reaching_the_goal():
    """REGRESSION TEST. Do not delete.

    IQM trims the top and bottom quarter, so one solved seed in five is trimmed
    away and every strategy's IQM is 0.0. The label said "no strategy ever
    reached the goal", which is simply false: on the real DoorKey-7 one run
    finished with 0.976. Found 2026-08-22 while checking a decision-log claim.
    """
    import pandas as pd
    from rlx.analysis.report import _winners_section

    rows = []
    for strategy in ("boltzmann", "count_based", "epsilon_greedy", "noisy"):
        for seed in range(5):
            scored = strategy == "boltzmann" and seed == 0
            rows.append({"env_id": "DoorKey-7", "family": "DoorKey", "difficulty": 7,
                         "strategy": strategy, "seed": seed,
                         "final_return": 0.976 if scored else 0.0})
    text = _winners_section(pd.DataFrame(rows))

    assert "no strategy ever reached the goal" not in text
    assert "1" in text, "the table must say how many runs did reach the goal"
