# Task 3 — Coverage metrics

The measurement the whole "why" story rests on.

**Files:**
- Create: `src/rlx/analysis/coverage.py`
- Test: `tests/test_coverage.py`

**Interfaces:**
- Consumes: `grid_info`, `reachable_mask`, `bfs_distances` (Samuel task 3);
  `RunResult` (your task 2).
- Produces:
  - `raw_coverage(counts, info) -> np.ndarray`  # one value per snapshot
  - `task_relevant_mask(info) -> np.ndarray`    # bool (W, H)
  - `task_relevant_coverage(counts, info) -> np.ndarray`
  - `early_auc(steps, coverage, total_steps, frac=0.2) -> float`

---

## The three metrics

**Raw coverage** — how much of the maze the agent has ever seen:

```
distinct (x, y, dir) visited  /  reachable (x, y, dir)
```

The denominator is `reachable_mask(info).sum() * 4`.

**Task-relevant coverage** — the same, restricted to places that matter. A cell
is task-relevant if it is:

- on a shortest path through the landmark chain, or within 1 cell of one, **or**
- adjacent to the key, the door, or the goal.

The landmark chain differs by family, and getting `DoorKey` right is the point of
the exercise:

| Family | Chain |
|---|---|
| Empty | start → goal |
| DoorKey | start → **key** → **door** → goal |
| MultiRoom | start → goal |

For DoorKey, "shortest path from start to goal" is not the task — the door is
locked. The agent must detour to the key first.

**How to find cells on a shortest path from A to B** without enumerating paths: a
cell `c` lies on some shortest A→B path exactly when

```
dist_from_A[c] + dist_from_B[c] == dist_from_A[B]
```

Two BFS calls per segment, no path enumeration.

**Early-coverage AUC** — the predictor in the central test. Take the coverage
curve over the first 20% of training and integrate it (trapezoid rule), then
divide by the width so the result lands in [0, 1] and is comparable across
mazes. A strategy that spreads out fast has a large area; a slow one has a small
area.

---

- [x] **Step 1: Write the failing tests**

Create `tests/test_coverage.py`. The hand-built fixtures matter more than the
end-to-end checks — they are what catch an off-by-one in the denominator.

```python
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
    """5 reachable cells x 4 directions = 20 states. Visit 5 of them."""
    counts = np.zeros((1, 7, 3, 4), dtype=np.int32)
    counts[0, 1, 1, :] = 1          # 4 states
    counts[0, 2, 1, 0] = 1          # 1 more
    assert np.isclose(raw_coverage(counts, corridor)[0], 5 / 20)


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


def test_faster_exploration_gives_a_larger_auc():
    steps = np.linspace(0, 100_000, 101)
    fast = 1 - np.exp(-steps / 10_000)
    slow = 1 - np.exp(-steps / 50_000)
    assert early_auc(steps, fast, 100_000) > early_auc(steps, slow, 100_000)
```

- [x] **Step 2: Run and watch them fail**

```bash
pytest tests/test_coverage.py -v
```

- [x] **Step 3: Write `src/rlx/analysis/coverage.py`**

```python
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


def _landmark_chain(info: GridInfo) -> list[tuple[int, int]]:
    """Waypoints the agent must actually pass through, in order.

    For DoorKey this is start -> key -> door -> goal, NOT start -> goal: the
    direct route is blocked by a locked door.
    """
    chain = [info.start]
    if info.key is not None:
        chain.append(info.key)
    if info.door is not None:
        chain.append(info.door)
    chain.append(info.goal)
    return chain


def task_relevant_mask(info: GridInfo) -> np.ndarray:
    """Cells on or beside the route through the landmark chain."""
    chain = _landmark_chain(info)
    on_route = np.zeros((info.width, info.height), dtype=bool)
    for a, b in zip(chain, chain[1:]):
        on_route |= _on_shortest_path(info, a, b)

    for landmark in (info.key, info.door, info.goal):
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
    """
    window = frac * total_steps
    inside = steps <= window
    if inside.sum() < 2:
        return float(coverage[0]) if len(coverage) else 0.0
    return float(np.trapezoid(coverage[inside], steps[inside]) /
                 (steps[inside][-1] - steps[inside][0]))
```

Note: `np.trapezoid` is the numpy 2.0 name. On older numpy it is `np.trapz`. If
you get an `AttributeError`, check your numpy version and use the name that
exists — do not write your own trapezoid rule.

- [x] **Step 4: Run the tests**

```bash
pytest tests/test_coverage.py -v
```

Expected: 14 passed.

Likely failure and its cause: `test_a_detour_is_not_task_relevant` passing when
it should fail usually means `_dilate` is being applied more than once, which
swells the mask until it covers everything. One dilation only.

- [x] **Step 5: Sanity-check the denominators on real mazes**

```bash
python -c "
from rlx.analysis.coverage import task_relevant_mask
from rlx.envs import ENV_IDS, grid_info, reachable_mask
for e in ENV_IDS:
    i = grid_info(e, 0)
    r, t = reachable_mask(i).sum(), task_relevant_mask(i).sum()
    print(f'{e:<14} reachable={r:4d}  task_relevant={t:4d}  ratio={t/r:.2f}')
"
```

**Read these numbers.** Task-relevant should always be smaller than reachable,
and the ratio should be well under 1.0 on the bigger mazes — that is the whole
point of the distinction. If the ratio is ~1.0 everywhere, the mask is too
generous and the raw/task-relevant comparison will show nothing.

- [x] **Step 6: Log the change**

Append to `docs/decision_log.md`, in plain language:

```markdown
## 2026-08-19 — Two ways of measuring coverage

**Status:** Active

**What changed:** We now measure how much of a maze the agent visited, in two
different ways.

"Raw coverage" is the simple one: of all the places the agent could possibly
reach, what fraction did it actually stand in (counting the 4 directions it can
face as different situations)?

"Task-relevant coverage" only counts places that matter for solving the maze —
on or next to the shortest route, or right beside the key, the door, or the goal.
The idea is that wandering into an irrelevant corner is exploring, but not
*usefully* exploring.

**The DoorKey detail worth remembering:** for those mazes the route is
start → key → door → goal, not start → goal. The straight line to the goal goes
through a locked door, so it is not the actual task. We got this wrong in an
earlier sketch and it would have made task-relevant coverage meaningless on the
whole DoorKey family.

**A neat trick we used:** to find every cell lying on a shortest route from A to
B without listing all the routes, check whether (distance from A to that cell) +
(distance from that cell to B) equals the total distance from A to B. If it does,
the cell is on some shortest route. Two distance calculations instead of an
explosion of paths.

**What it means for the results:** These two numbers are what we correlate
against final performance. Which of the two predicts better is one of the three
questions the report answers.
```

- [ ] **Step 7: Commit**

```bash
git add src/rlx/analysis/coverage.py tests/test_coverage.py docs/decision_log.md
git commit -m "feat: raw and task-relevant coverage metrics"
```
