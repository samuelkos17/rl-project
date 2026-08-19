import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rlx.analysis.figures import COLORS, FIGURE_NAMES, make_all_figures
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
