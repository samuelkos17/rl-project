"""Verify every MiniGrid API assumption in the spec before anything depends on it.

Run:  python scripts/verify_api.py
"""

import numpy as np
from minigrid.envs import DoorKeyEnv, EmptyEnv, MultiRoomEnv
from minigrid.wrappers import ImgObsWrapper

CASES = [
    ("Empty-5", lambda: EmptyEnv(size=5)),
    ("Empty-8", lambda: EmptyEnv(size=8)),
    ("Empty-16", lambda: EmptyEnv(size=16)),
    ("DoorKey-5", lambda: DoorKeyEnv(size=5)),
    ("DoorKey-6", lambda: DoorKeyEnv(size=6)),
    ("DoorKey-7", lambda: DoorKeyEnv(size=7)),
    ("DoorKey-8", lambda: DoorKeyEnv(size=8)),
    ("DoorKey-10", lambda: DoorKeyEnv(size=10)),
    ("MultiRoom-N2", lambda: MultiRoomEnv(minNumRooms=2, maxNumRooms=2, maxRoomSize=6)),
    ("MultiRoom-N3", lambda: MultiRoomEnv(minNumRooms=3, maxNumRooms=3, maxRoomSize=6)),
    ("MultiRoom-N4", lambda: MultiRoomEnv(minNumRooms=4, maxNumRooms=4, maxRoomSize=6)),
    ("MultiRoom-N5", lambda: MultiRoomEnv(minNumRooms=5, maxNumRooms=5, maxRoomSize=6)),
    ("MultiRoom-N6", lambda: MultiRoomEnv(minNumRooms=6, maxNumRooms=6, maxRoomSize=6)),
]


def _layout_fingerprint(env) -> str:
    u = env.unwrapped
    cells = []
    for y in range(u.height):
        for x in range(u.width):
            c = u.grid.get(x, y)
            cells.append("." if c is None else c.type[0])
    return "".join(cells)


def check(name, factory):
    env = ImgObsWrapper(factory())
    obs, info = env.reset(seed=0)

    assert obs.shape == (7, 7, 3), f"{name}: obs shape {obs.shape}, expected (7,7,3)"
    assert env.action_space.n == 7, f"{name}: {env.action_space.n} actions, expected 7"

    u = env.unwrapped
    assert hasattr(u, "agent_pos"), f"{name}: no agent_pos"
    assert hasattr(u, "agent_dir"), f"{name}: no agent_dir"
    assert hasattr(u, "grid"), f"{name}: no grid"
    assert u.agent_dir in (0, 1, 2, 3), f"{name}: agent_dir={u.agent_dir}"
    assert len(u.agent_pos) == 2, f"{name}: agent_pos={u.agent_pos}"

    obs2, r, term, trunc, info = env.step(2)
    assert obs2.shape == (7, 7, 3)
    assert isinstance(float(r), float)

    # Layout pinning: resetting with the same seed must reproduce the same maze.
    env.reset(seed=7)
    layout_a = _layout_fingerprint(env)
    env.reset(seed=7)
    layout_b = _layout_fingerprint(env)
    assert layout_a == layout_b, f"{name}: reset(seed=7) is not reproducible"

    env.reset(seed=8)
    layout_c = _layout_fingerprint(env)
    varies = layout_a != layout_c

    print(f"OK  {name:<14} grid={u.width}x{u.height}  "
          f"max_steps={u.max_steps:<5} layout_varies_by_seed={varies}")
    env.close()


if __name__ == "__main__":
    for name, factory in CASES:
        check(name, factory)
    print("\nAll API assumptions verified.")
