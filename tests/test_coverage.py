import numpy as np
import pytest

from rlx.analysis.coverage import (
    early_auc, raw_coverage, task_relevant_coverage, task_relevant_mask,
)
from rlx.envs import GridInfo


@pytest.fixture
def corridor():
    """A 5-cell horizontal corridor at y=1, key at x=2, door at x=3."""
    walls = np.ones((7, 3), dtype=bool)
    walls[1:6, 1] = False
    return GridInfo(width=7, height=3, walls=walls,
                    start=(1, 1), goal=(5, 1), key=(2, 1), door=(3, 1))


def test_raw_coverage_of_an_empty_visit_array_is_zero(corridor):
    counts = np.zeros((1, 7, 3, 4), dtype=np.int32)
    assert raw_coverage(counts, corridor)[0] == 0.0


def test_raw_coverage_of_a_fully_visited_maze_is_one(corridor):
    counts = np.zeros((1, 7, 3, 4), dtype=np.int32)
    counts[0, 1:6, 1, :] = 5
    assert np.isclose(raw_coverage(counts, corridor)[0], 1.0)


def test_raw_coverage_is_the_hand_computed_fraction(corridor):
    """5 reachable cells, minus the goal cell no run can log, is 4 x 4 = 16
    states. Visit 5 of them."""
    counts = np.zeros((1, 7, 3, 4), dtype=np.int32)
    counts[0, 1, 1, :] = 1          # 4 states
    counts[0, 2, 1, 0] = 1          # 1 more
    assert np.isclose(raw_coverage(counts, corridor)[0], 5 / 16)


def test_repeat_visits_do_not_increase_coverage(corridor):
    counts = np.zeros((1, 7, 3, 4), dtype=np.int32)
    counts[0, 1, 1, 0] = 1
    a = raw_coverage(counts, corridor)[0]
    counts[0, 1, 1, 0] = 9999
    assert raw_coverage(counts, corridor)[0] == a


def test_coverage_is_computed_per_snapshot(corridor):
    counts = np.zeros((3, 7, 3, 4), dtype=np.int32)
    counts[0, 1, 1, 0] = 1
    counts[1, 1:3, 1, :] = 1
    counts[2, 1:6, 1, :] = 1
    cov = raw_coverage(counts, corridor)
    assert cov.shape == (3,)
    assert cov[0] < cov[1] < cov[2]


def test_task_relevant_mask_excludes_walls(corridor):
    mask = task_relevant_mask(corridor)
    assert not (mask & corridor.walls).any()


def test_task_relevant_mask_includes_the_landmarks(corridor):
    mask = task_relevant_mask(corridor)
    for cell in (corridor.start, corridor.goal, corridor.key, corridor.door):
        assert mask[cell], cell


def test_task_relevant_is_a_subset_of_reachable(corridor):
    from rlx.envs import reachable_mask
    assert (task_relevant_mask(corridor) <= reachable_mask(corridor)).all()


def test_a_detour_is_not_task_relevant():
    """A dead-end branch off the corridor should be excluded."""
    walls = np.ones((7, 5), dtype=bool)
    walls[1:6, 1] = False
    walls[3, 2:4] = False            # dead-end branch hanging off x=3
    info = GridInfo(width=7, height=5, walls=walls,
                    start=(1, 1), goal=(5, 1), key=None, door=None)
    mask = task_relevant_mask(info)
    assert mask[1, 1] and mask[5, 1]
    assert not mask[3, 3], "the far end of a dead end is not on the route"


def test_task_relevant_coverage_uses_the_smaller_denominator(corridor):
    """Same visits score higher on the task-relevant metric, because its
    denominator is smaller."""
    counts = np.zeros((1, 7, 3, 4), dtype=np.int32)
    counts[0, 1:4, 1, :] = 1
    assert task_relevant_coverage(counts, corridor)[0] >= raw_coverage(counts, corridor)[0]


def test_the_goal_cell_is_not_in_the_denominator(corridor):
    """REGRESSION TEST. Do not delete.

    `train.py` records the agent's position at the top of each step and resets in
    the same iteration the episode ends, so the goal cell is never logged -- in
    any direction. Counting its 4 states in the denominator capped raw coverage
    at 1 - 1/reachable: 0.857 on DoorKey-5, 0.889 on Empty-5. Verified on the
    pilot: goal_visits == 0 in all 16 runs, including two that solved Empty-5.

    Here every cell a run CAN log is visited, so coverage must be exactly 1.0.
    """
    counts = np.zeros((1, 7, 3, 4), dtype=np.int32)
    counts[0, 1:5, 1, :] = 3         # everything except the goal at (5, 1)
    assert raw_coverage(counts, corridor)[0] == 1.0
    assert task_relevant_coverage(counts, corridor)[0] == 1.0


def test_a_visit_recorded_at_the_goal_cannot_push_coverage_above_one(corridor):
    """The goal leaves the numerator as well as the denominator, so a count
    there -- which a fixture or a future logger change could produce -- cannot
    make a fraction exceed 1.0."""
    counts = np.zeros((1, 7, 3, 4), dtype=np.int32)
    counts[0, 1:6, 1, :] = 3         # the goal included this time
    assert raw_coverage(counts, corridor)[0] == 1.0


def test_the_goal_stays_task_relevant_even_though_it_is_not_measured(corridor):
    """Only the measurement is blind to the goal. The mask still describes the
    task, and the report's denominator table is read off it."""
    assert task_relevant_mask(corridor)[corridor.goal]


def test_early_auc_of_a_flat_curve_equals_its_level():
    steps = np.arange(0, 100_000, 10_000)
    assert np.isclose(early_auc(steps, np.full(len(steps), 0.5), 100_000, frac=0.2), 0.5)


def test_early_auc_of_a_ramp_is_the_analytic_integral():
    """Coverage rising 0 -> 1 linearly. Over the first 20%, mean value is 0.1."""
    steps = np.linspace(0, 100_000, 101)
    assert np.isclose(early_auc(steps, steps / 100_000, 100_000, frac=0.2), 0.1, atol=0.01)


def test_early_auc_ignores_everything_after_the_window():
    steps = np.linspace(0, 100_000, 101)
    cov = np.where(steps <= 20_000, 0.3, 1.0)
    assert np.isclose(early_auc(steps, cov, 100_000, frac=0.2), 0.3, atol=0.02)


def test_early_auc_refuses_a_window_it_cannot_integrate():
    """The pilot config's shape: 20k steps, snapshots every 10k. The early window
    is 4k wide, so NO snapshot lands in it. Returning a number here would be a
    quiet lie -- the only honest value available is coverage at step 10,000,
    which is 2.5x outside the window it claims to summarise."""
    steps = np.array([10_000, 20_000])
    cov = np.array([0.30, 0.55])
    with pytest.raises(ValueError, match="snapshot_every"):
        early_auc(steps, cov, total_steps=20_000, frac=0.2)


def test_early_auc_refuses_a_single_point_in_the_window():
    """One point is not an area. Trapezoid over a zero-width interval would also
    divide by zero, so this must raise rather than return inf or NaN."""
    steps = np.array([10_000, 50_000, 90_000])
    with pytest.raises(ValueError, match="snapshot_every"):
        early_auc(steps, np.array([0.1, 0.5, 0.9]), total_steps=100_000, frac=0.2)


def test_early_auc_refuses_a_grid_that_stops_short_of_the_window_edge():
    """The integral runs to the last snapshot inside the window but is divided by
    the whole window. A grid of 30_000 against an 80_000 window stops at 60_000,
    so the result reads 25% low -- and, before this guard, silently.

    Two points fall inside, so the older "needs 2 points" check passes it.
    """
    steps = np.arange(30_000, 400_001, 30_000)
    coverage = np.full(len(steps), 0.5)
    with pytest.raises(ValueError, match="25% low"):
        early_auc(steps, coverage, total_steps=400_000, frac=0.2)


def test_the_real_snapshot_grid_covers_its_window_exactly():
    """400_000 steps at snapshot_every=10_000: the window is 80_000 and the
    eighth snapshot lands exactly on it, so the guard never binds on the settings
    the sweep actually runs.

    Checked on a linear ramp, where the trapezoid rule is exact and the answer is
    known: coverage rising 0 -> 1 over 400_000 steps has mean 0.1 over the first
    fifth. A flat curve would NOT give its own level here -- prepending the
    origin costs half of the first interval -- which is the documented and
    deliberate 0.2% price of integrating from 0.
    """
    steps = np.arange(10_000, 400_001, 10_000)
    assert np.isclose(early_auc(steps, steps / 400_000, 400_000), 0.1)


def test_faster_exploration_gives_a_larger_auc():
    steps = np.linspace(0, 100_000, 101)
    fast = 1 - np.exp(-steps / 10_000)
    slow = 1 - np.exp(-steps / 50_000)
    assert early_auc(steps, fast, 100_000) > early_auc(steps, slow, 100_000)


def test_a_multiroom_connecting_door_is_not_a_waypoint():
    """MultiRoom sets `door` but no `key`, and grid_info keeps only the last of
    the N-1 connecting doors. That arbitrary door must not become a waypoint --
    the chain for MultiRoom is start -> goal (task file, landmark chain table).

    Layout: a straight corridor from start to goal, plus a stub branch holding a
    door far off the route. Routing through it would drag the branch into the
    mask.
    """
    walls = np.ones((9, 5), dtype=bool)
    walls[1:8, 1] = False            # corridor y=1, start (1,1) -> goal (7,1)
    walls[4, 2:4] = False            # branch hanging down off x=4
    info = GridInfo(width=9, height=5, walls=walls,
                    start=(1, 1), goal=(7, 1), key=None, door=(4, 3))
    mask = task_relevant_mask(info)
    assert not mask[4, 3], "a MultiRoom connecting door is not a landmark"


def test_early_auc_integrates_from_zero_when_there_is_no_snapshot_at_step_zero():
    """The shape of every real run: train.py snapshots at `step > 0`, so the
    first point is at snapshot_every, never at 0.

    Coverage ramps linearly 0 -> 1 over 100k steps, so the true mean over the
    first 20k is 0.1. Normalising by the observed snapshot span (10k..20k)
    instead of by the 20k window returns 0.15 -- 50% too high here, and ~12%
    too high at the real 400k/10k settings.
    """
    steps = np.arange(10_000, 100_001, 10_000)
    coverage = steps / 100_000
    assert np.isclose(early_auc(steps, coverage, 100_000, frac=0.2), 0.1)
