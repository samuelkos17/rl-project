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
    assert "H1 confirmed (CI excludes zero AND trend positive)" in text
    assert "H2 confirmed (larger AND non-overlapping CIs)" in text
    assert "CI excludes zero: **" in text, "the half-criterion is reported separately"


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
