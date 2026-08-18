# Task 3 — Environment factory, grid info, and BFS

**Files:**
- Modify: `src/rlx/envs.py` — task 1 already shipped `ENV_IDS` and
  `difficulty_index` there as part of the frozen contract. **Keep both exactly as
  they are** (Daniel imports `difficulty_index`) and add the rest around them.
- Test: `tests/test_envs.py`

**Interfaces:**
- Consumes: verified API names from Task 2.
- Produces (Daniel's `coverage.py` depends on all four):
  - `make_env(env_id: str, layout_seed: int) -> gym.Env`
  - `grid_info(env_id: str, layout_seed: int) -> GridInfo`
  - `reachable_mask(info: GridInfo) -> np.ndarray`  # bool, shape (W, H)
  - `bfs_distances(info: GridInfo, source: tuple[int, int]) -> np.ndarray`  # int (W, H), -1 = unreachable
  - `ENV_IDS: tuple[str, ...]`, `difficulty_index(env_id: str) -> int`

`GridInfo` fields: `width, height, walls, start, goal, key, door` — `walls` is a
bool array shape `(W, H)` where True means impassable; `key` and `door` are
`None` for families without them.

---

- [x] **Step 1: Write the failing tests**

Create `tests/test_envs.py`:

```python
import numpy as np
import pytest

from rlx.envs import (
    ENV_IDS, bfs_distances, difficulty_index, grid_info, make_env, reachable_mask,
)


def test_there_are_thirteen_instances():
    assert len(ENV_IDS) == 13


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_every_instance_builds_and_steps(env_id):
    env = make_env(env_id, layout_seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (7, 7, 3)
    obs, r, term, trunc, _ = env.step(2)
    assert obs.shape == (7, 7, 3)
    env.close()


def test_difficulty_index_increases_within_a_family():
    assert difficulty_index("DoorKey-5") < difficulty_index("DoorKey-10")
    assert difficulty_index("MultiRoom-N2") < difficulty_index("MultiRoom-N6")


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_grid_info_is_self_consistent(env_id):
    info = grid_info(env_id, layout_seed=0)
    assert info.walls.shape == (info.width, info.height)
    assert not info.walls[info.start]
    assert not info.walls[info.goal]
    if "DoorKey" in env_id:
        assert info.key is not None
        assert info.door is not None


def test_layout_is_reproducible_for_a_given_seed():
    a = grid_info("DoorKey-8", layout_seed=3)
    b = grid_info("DoorKey-8", layout_seed=3)
    assert np.array_equal(a.walls, b.walls)
    assert a.key == b.key


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_goal_is_reachable_from_start(env_id):
    info = grid_info(env_id, layout_seed=0)
    d = bfs_distances(info, info.start)
    assert d[info.goal] > 0, f"{env_id}: goal unreachable from start"


def test_bfs_on_a_hand_built_corridor():
    from rlx.envs import GridInfo

    walls = np.ones((5, 3), dtype=bool)
    walls[1:4, 1] = False              # a 3-cell horizontal corridor at y=1
    info = GridInfo(width=5, height=3, walls=walls,
                    start=(1, 1), goal=(3, 1), key=None, door=None)

    d = bfs_distances(info, (1, 1))
    assert d[1, 1] == 0
    assert d[2, 1] == 1
    assert d[3, 1] == 2
    assert d[0, 1] == -1               # wall, unreachable

    assert reachable_mask(info).sum() == 3


def test_reachable_mask_excludes_walls():
    info = grid_info("Empty-5", layout_seed=0)
    assert not (reachable_mask(info) & info.walls).any()
```

- [x] **Step 2: Run and watch them fail**

```bash
pytest tests/test_envs.py -v
```

Expected: `ModuleNotFoundError: No module named 'rlx.envs'`.

- [x] **Step 3: Write `src/rlx/envs.py`**

```python
"""Environment construction, grid layout extraction, and grid BFS.

The layout is PINNED per run: every reset uses the run's layout_seed, so one run
sees exactly one maze for its whole life. Without this, MiniGrid regenerates the
maze every episode and state coverage has no fixed denominator.
"""

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from minigrid.envs import DoorKeyEnv, EmptyEnv, MultiRoomEnv
from minigrid.wrappers import ImgObsWrapper

ENV_IDS = (
    "Empty-5", "Empty-8", "Empty-16",
    "DoorKey-5", "DoorKey-6", "DoorKey-7", "DoorKey-8", "DoorKey-10",
    "MultiRoom-N2", "MultiRoom-N3", "MultiRoom-N4", "MultiRoom-N5", "MultiRoom-N6",
)

_MULTIROOM_MAX_ROOM_SIZE = 6


@dataclass
class GridInfo:
    """Static layout of one pinned maze. Analysis-only -- the agent never sees this."""
    width: int
    height: int
    walls: np.ndarray                    # bool (W, H); True = impassable
    start: tuple[int, int]
    goal: tuple[int, int]
    key: tuple[int, int] | None
    door: tuple[int, int] | None


class PinnedLayout(gym.Wrapper):
    """Force every reset to regenerate the same maze."""

    def __init__(self, env: gym.Env, layout_seed: int):
        super().__init__(env)
        self._layout_seed = layout_seed

    def reset(self, **kwargs):
        kwargs["seed"] = self._layout_seed
        return self.env.reset(**kwargs)


def _base_env(env_id: str) -> gym.Env:
    family, param = env_id.split("-")
    if family == "Empty":
        return EmptyEnv(size=int(param))
    if family == "DoorKey":
        return DoorKeyEnv(size=int(param))
    if family == "MultiRoom":
        n = int(param.lstrip("N"))
        return MultiRoomEnv(minNumRooms=n, maxNumRooms=n,
                            maxRoomSize=_MULTIROOM_MAX_ROOM_SIZE)
    raise ValueError(f"unknown env_id {env_id!r}")


def make_env(env_id: str, layout_seed: int) -> gym.Env:
    """Build one environment with its layout pinned to layout_seed."""
    return PinnedLayout(ImgObsWrapper(_base_env(env_id)), layout_seed)


def difficulty_index(env_id: str) -> int:
    """Grid size for Empty/DoorKey, room count for MultiRoom."""
    param = env_id.split("-")[1]
    return int(param.lstrip("N"))


def grid_info(env_id: str, layout_seed: int) -> GridInfo:
    """Extract the static layout of the pinned maze. Analysis only."""
    env = make_env(env_id, layout_seed)
    env.reset()
    u = env.unwrapped

    walls = np.zeros((u.width, u.height), dtype=bool)
    key = door = None
    goal = None
    for x in range(u.width):
        for y in range(u.height):
            cell = u.grid.get(x, y)
            if cell is None:
                continue
            if cell.type == "wall":
                walls[x, y] = True
            elif cell.type == "key":
                key = (x, y)
            elif cell.type == "door":
                door = (x, y)
            elif cell.type == "goal":
                goal = (x, y)

    start = (int(u.agent_pos[0]), int(u.agent_pos[1]))
    if goal is None:
        raise RuntimeError(f"{env_id}: no goal cell found in the grid")
    env.close()
    return GridInfo(u.width, u.height, walls, start, goal, key, door)


def reachable_mask(info: GridInfo) -> np.ndarray:
    """Cells reachable from the start, ignoring the locked door.

    The door is treated as passable because the agent can open it with the key,
    so every cell behind it is genuinely reachable during a run.
    """
    return bfs_distances(info, info.start) >= 0


def bfs_distances(info: GridInfo, source: tuple[int, int]) -> np.ndarray:
    """Breadth-first step distance from source to every cell. -1 = unreachable."""
    dist = np.full((info.width, info.height), -1, dtype=int)
    if info.walls[source]:
        return dist

    dist[source] = 0
    queue = deque([source])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (0 <= nx < info.width and 0 <= ny < info.height):
                continue
            if info.walls[nx, ny] or dist[nx, ny] >= 0:
                continue
            dist[nx, ny] = dist[x, y] + 1
            queue.append((nx, ny))
    return dist
```

- [x] **Step 4: Run the tests**

```bash
pytest tests/test_envs.py -v
```

Expected: all pass.

Two failures are plausible and each has a specific cause:

- `test_goal_is_reachable_from_start` fails on `DoorKey-*` — the door is being
  counted as a wall. Check that `cell.type == "door"` is handled *before* any
  wall branch, and that doors are not written into `walls`.
- `test_grid_info_is_self_consistent` fails with `walls[start] == True` — the
  agent position is being read before `reset()` populates it. Confirm `reset()`
  is called first.

Fix, re-run, and only move on when the output actually says all passed.

- [x] **Step 5: Sanity-check the coverage denominators by hand**

```bash
python -c "
from rlx.envs import ENV_IDS, grid_info, reachable_mask
for e in ENV_IDS:
    i = grid_info(e, 0)
    print(f'{e:<14} grid={i.width}x{i.height} reachable_cells={reachable_mask(i).sum():4d} states={reachable_mask(i).sum()*4:5d}')
"
```

Read the numbers and check they are sensible: `Empty-5` should have 9 reachable
cells (a 3x3 interior), `Empty-8` should have 36. If a number is wildly off, the
wall extraction is wrong — fix it before Daniel builds coverage on top.

- [x] **Step 6: Log the change**

The layout-pinning *decision* is already recorded in `docs/decision_log.md`
(entry "Each run gets one fixed maze, and evaluation runs 1 episode",
2026-08-17). Do not duplicate it. Append a short entry confirming the
*implementation*:

```markdown
## 2026-08-18 — Layout pinning implemented and verified

**Status:** Active — implements the 2026-08-17 entry "Each run gets one fixed maze"

**What changed:** Built the environment factory. Every reset now passes the run's
seed, so the maze really does stay the same for a whole run. Verified by a test
that builds the same maze twice with the same seed and checks the walls match.

**What it means for the results:** The measured reachable-cell counts per maze
are: <paste the numbers printed in step 5>. These are the denominators every
coverage percentage in the report is divided by, so they are worth recording.
```

- [ ] **Step 7: Commit**

```bash
git add src/rlx/envs.py tests/test_envs.py docs/decision_log.md
git commit -m "feat: environment factory with pinned layouts and grid BFS"
```

Tell Daniel this is merged — his coverage task consumes `bfs_distances`,
`grid_info`, and `reachable_mask`.
