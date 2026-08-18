"""Coverage metrics, derived from saved visitation snapshots.

These use the true (x, y, direction), which is privileged information the agent
never receives. That is fine -- it is measurement, not learning -- but the report
must state it. See CLAUDE.md section 8.
"""

import numpy as np

from rlx.envs import GridInfo, bfs_distances, reachable_mask

N_DIRECTIONS = 4


def raw_coverage(counts: np.ndarray, info: GridInfo) -> np.ndarray:
    """Fraction of reachable (x, y, dir) states visited, one value per snapshot.

    counts: (T, W, H, 4) cumulative visit counts.
    """
    denominator = reachable_mask(info).sum() * N_DIRECTIONS
    visited = (counts > 0).sum(axis=(1, 2, 3))
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
    """Fraction of task-relevant states visited, one value per snapshot."""
    mask = task_relevant_mask(info)
    denominator = mask.sum() * N_DIRECTIONS
    if denominator == 0:
        return np.zeros(counts.shape[0])
    visited = ((counts > 0) & mask[None, :, :, None]).sum(axis=(1, 2, 3))
    return visited / denominator


def early_auc(steps: np.ndarray, coverage: np.ndarray,
              total_steps: int, frac: float = 0.2) -> float:
    """Normalised area under the coverage curve over the first `frac` of training.

    Dividing by the window width puts the result in [0, 1], so it is comparable
    across environments and step budgets.

    Raises ValueError if fewer than two snapshots fall inside the window. An
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
    return float(np.trapezoid(coverage[inside], steps[inside]) /
                 (steps[inside][-1] - steps[inside][0]))
