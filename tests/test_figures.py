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
