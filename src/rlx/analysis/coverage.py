"""Coverage metrics, derived from saved visitation snapshots.

These use the true (x, y, direction), which is privileged information the agent
never receives. That is fine -- it is measurement, not learning -- but the report
must state it. See CLAUDE.md section 8.
"""

import numpy as np

from rlx.envs import GridInfo, bfs_distances, reachable_mask

N_DIRECTIONS = 4

#: The integral must cover at least this fraction of the early window before the
#: result is reported. `early_auc` divides by the FULL window, so a snapshot grid
#: whose last in-window point falls well short of the edge is integrated over
#: less than it is normalised by, and silently reads low. At the real settings
#: (400_000 steps, snapshot_every=10_000) the last point lands exactly on the
#: 80_000 edge, so this never binds; it exists for the day someone changes
#: snapshot_every to a value that does not divide the window. See spec 6.4.
MIN_WINDOW_COVERED = 0.9


def _loggable(mask: np.ndarray, info: GridInfo) -> np.ndarray:
    """`mask` minus the goal cell, which no run can ever record a visit to.

    `train.py` records the agent's position at the TOP of each step and resets
    the environment in the same iteration the episode terminates, so the position
    the agent holds when it steps onto the goal is never logged -- in any of the
    4 directions. Verified 2026-08-19 on the pilot: `goal_visits == 0` in all 16
    runs, including the two that solved Empty-5 with a return of 0.955.

    Leaving those 4 states in the denominator caps coverage below 1.0 by
    `1 / reachable`: 14.3% on DoorKey-5, 11.1% on Empty-5, 0.5% on Empty-16. The
    cap is a constant per run, so no within-instance correlation moved -- but
    every coverage LEVEL printed in the report was too low, and "1.0" did not
    mean "saw everything". They are dropped from the numerator too, so a stray
    count there could never push a fraction above 1.0.

    `task_relevant_mask` keeps the goal: it is task-relevant by definition. Only
    the measurement drops it, because only the measurement is blind to it.
    """
    out = mask.copy()
    out[info.goal] = False
    return out


def raw_coverage(counts: np.ndarray, info: GridInfo) -> np.ndarray:
    """Fraction of loggable reachable (x, y, dir) states visited, per snapshot.

    counts: (T, W, H, 4) cumulative visit counts.
    """
    mask = _loggable(reachable_mask(info), info)
    denominator = mask.sum() * N_DIRECTIONS
    visited = ((counts > 0) & mask[None, :, :, None]).sum(axis=(1, 2, 3))
    return visited / denominator


def _on_shortest_path(info: GridInfo, a: tuple[int, int], b: tuple[int, int]) -> np.ndarray:
    """Cells lying on some shortest path from a to b.

    A cell c qualifies exactly when dist(a, c) + dist(c, b) == dist(a, b), which
    avoids enumerating paths.
    """
    d_a = bfs_distances(info, a)
    d_b = bfs_distances(info, b)
    total = d_a[b]
    if total < 0:
        return np.zeros((info.width, info.height), dtype=bool)
    reachable_both = (d_a >= 0) & (d_b >= 0)
    return reachable_both & (d_a + d_b == total)


def _dilate(mask: np.ndarray) -> np.ndarray:
    """Grow a mask by one cell in the 4 cardinal directions."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _locked_door(info: GridInfo) -> tuple[int, int] | None:
    """The door only counts as a waypoint when the maze also has a key.

    DoorKey locks its single door, so the route genuinely detours through it.
    MultiRoom also reports a door -- its rooms are joined by them -- but those
    are plain openings that a shortest path already crosses, and grid_info keeps
    only the last of the N-1 it finds. Routing through that arbitrary door would
    drag an unrelated room into the mask.
    """
    return info.door if info.key is not None else None


def _landmark_chain(info: GridInfo) -> list[tuple[int, int]]:
    """Waypoints the agent must actually pass through, in order.

    For DoorKey this is start -> key -> door -> goal, NOT start -> goal: the
    direct route is blocked by a locked door.
    """
    chain = [info.start]
    if info.key is not None:
        chain.append(info.key)
    door = _locked_door(info)
    if door is not None:
        chain.append(door)
    chain.append(info.goal)
    return chain


def task_relevant_mask(info: GridInfo) -> np.ndarray:
    """Cells on or beside the route through the landmark chain."""
    chain = _landmark_chain(info)
    on_route = np.zeros((info.width, info.height), dtype=bool)
    for a, b in zip(chain, chain[1:]):
        on_route |= _on_shortest_path(info, a, b)

    for landmark in (info.key, _locked_door(info), info.goal):
        if landmark is not None:
            on_route[landmark] = True

    return _dilate(on_route) & reachable_mask(info)


def task_relevant_coverage(counts: np.ndarray, info: GridInfo) -> np.ndarray:
    """Fraction of loggable task-relevant states visited, one value per snapshot."""
    mask = _loggable(task_relevant_mask(info), info)
    denominator = mask.sum() * N_DIRECTIONS
    if denominator == 0:
        return np.zeros(counts.shape[0])
    visited = ((counts > 0) & mask[None, :, :, None]).sum(axis=(1, 2, 3))
    return visited / denominator


def early_auc(steps: np.ndarray, coverage: np.ndarray,
              total_steps: int, frac: float = 0.2) -> float:
    """Normalised area under the coverage curve over the first `frac` of training.

    Dividing by the window width `frac * total_steps` puts the result in [0, 1],
    so it is comparable across environments, step budgets and snapshot cadences.

    `train.py` snapshots at `step > 0`, so no real run has a point at step 0 and
    the window's first slice would otherwise be missing. Coverage at step 0 is
    taken as 0.0 and prepended. That is not exactly true -- the agent already
    occupies its start state, one loggable state out of `4 * (reachable - 1)`,
    which is 3.1% on Empty-5 and 4.2% on DoorKey-5. Its effect on the RESULT is
    far smaller: it shifts only the first trapezoid, by
    `0.5 * c(0) * snapshot_every / window` = `c(0) / 16` at the real settings,
    i.e. 0.2% of a typical early-AUC. The alternative -- normalising by the
    observed snapshot span -- reads high at those same settings by an amount that
    depends on the curve, because integrating from the origin also adds a first
    trapezoid: 12.5% on a straight line, 10.9% on a saturating curve, 6.7% on a
    flat one.

    Raises ValueError if fewer than two snapshots fall inside the window, or if
    the last one inside it falls short of `MIN_WINDOW_COVERED` of the edge. An
    earlier version returned `coverage[0]` instead, which on a short run is the
    coverage at the FIRST snapshot -- a point outside the window entirely, handed
    back as a normal-looking float. This is the project's main predictor, so a
    quiet substitute here is worse than a crash: it would be read as a real
    measurement. Same rule as "no variance, excluded" in the statistics layer --
    never silently stand in for a number we could not compute.
    """
    window = frac * total_steps
    inside = steps <= window
    if inside.sum() < 2:
        raise ValueError(
            f"early-AUC window is {window:.0f} steps ({frac:.0%} of {total_steps}) "
            f"but only {int(inside.sum())} of {len(steps)} snapshots fall inside it "
            f"(first snapshot at step {int(steps[0]) if len(steps) else 'n/a'}). "
            f"Lower snapshot_every for this run: the window needs at least 2 points."
        )
    # The integral runs to the last snapshot inside the window but is divided by
    # the whole window, so a grid that stops short of the edge reads low by
    # exactly the fraction it stops short by -- and reads low silently.
    last = float(steps[inside][-1])
    if last < MIN_WINDOW_COVERED * window:
        raise ValueError(
            f"early-AUC window is {window:.0f} steps but the last snapshot "
            f"inside it is at step {last:.0f}, covering only {last / window:.0%} "
            f"of it. The integral would be divided by the full window and read "
            f"about {1 - last / window:.0%} low. Lower snapshot_every so the "
            f"grid reaches the window edge."
        )
    # The guard above counts REAL snapshots, so prepending the origin cannot
    # rescue a window that has too little measured data in it.
    xs, ys = steps[inside], coverage[inside]
    if xs[0] > 0:
        xs = np.concatenate(([0], xs))
        ys = np.concatenate(([0.0], ys))
    return float(np.trapezoid(ys, xs) / window)
