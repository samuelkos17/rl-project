"""Environment construction, grid layout extraction, and grid BFS.

The layout is PINNED per run: every reset uses the run's layout_seed, so one run
sees exactly one maze for its whole life. Without this, MiniGrid regenerates the
maze every episode and state coverage has no fixed denominator.

ENV_IDS and difficulty_index are a FROZEN CONTRACT (Daniel's analysis imports
difficulty_index). Do not change their behaviour.
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

#: Verified 2026-08-17: MultiRoom grids are 25x25 for every N, and max_steps is
#: 20*N. Room size is ours to choose; 6 keeps the optimal path comfortably inside
#: the step limit (worst case measured: ~53 steps against a 120 limit on N6).
_MULTIROOM_MAX_ROOM_SIZE = 6


def difficulty_index(env_id: str) -> int:
    """Grid size for Empty/DoorKey, room count for MultiRoom."""
    return int(env_id.split("-")[1].lstrip("N"))


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
    raise ValueError(f"unknown env_id {env_id!r}, expected one of {ENV_IDS}")


def make_env(env_id: str, layout_seed: int) -> gym.Env:
    """Build one environment with its layout pinned to layout_seed."""
    return PinnedLayout(ImgObsWrapper(_base_env(env_id)), layout_seed)


def grid_info(env_id: str, layout_seed: int) -> GridInfo:
    """Extract the static layout of the pinned maze. Analysis only."""
    env = make_env(env_id, layout_seed)
    env.reset()
    u = env.unwrapped

    walls = np.zeros((u.width, u.height), dtype=bool)
    key = door = goal = None
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
    env.close()
    if goal is None:
        raise RuntimeError(f"{env_id}: no goal cell found in the grid")
    return GridInfo(u.width, u.height, walls, start, goal, key, door)


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


def reachable_mask(info: GridInfo) -> np.ndarray:
    """Cells reachable from the start, ignoring the locked door.

    The door is treated as passable because the agent can open it with the key,
    so every cell behind it is genuinely reachable during a run. Doors are never
    written into `walls`, so this falls out of the BFS for free.
    """
    return bfs_distances(info, info.start) >= 0
