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
    obs, _ = env.reset()
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


def test_different_seeds_give_different_layouts():
    """5 seeds must give 5 mazes, or the seeds are not independent replicates."""
    a = grid_info("DoorKey-8", layout_seed=0)
    b = grid_info("DoorKey-8", layout_seed=1)
    assert not np.array_equal(a.walls, b.walls) or a.key != b.key


def test_pinned_layout_survives_repeated_resets():
    """Every reset inside one run must rebuild the SAME maze."""
    env = make_env("MultiRoom-N4", layout_seed=2)

    def fingerprint():
        u = env.unwrapped
        return "".join("." if (c := u.grid.get(x, y)) is None else c.type[0]
                       for y in range(u.height) for x in range(u.width))

    env.reset()
    first = fingerprint()
    for _ in range(3):
        env.reset()
        assert fingerprint() == first
    env.close()


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


def test_bfs_from_a_wall_returns_all_unreachable():
    from rlx.envs import GridInfo

    walls = np.ones((3, 3), dtype=bool)
    walls[1, 1] = False
    info = GridInfo(width=3, height=3, walls=walls,
                    start=(1, 1), goal=(1, 1), key=None, door=None)
    assert (bfs_distances(info, (0, 0)) == -1).all()


def test_reachable_mask_excludes_walls():
    info = grid_info("Empty-5", layout_seed=0)
    assert not (reachable_mask(info) & info.walls).any()


def test_doorkey_area_behind_the_door_is_reachable():
    """The door is passable because the agent can unlock it. If it were treated
    as a wall, the goal would look unreachable and coverage denominators would be
    wrong for the whole DoorKey family."""
    info = grid_info("DoorKey-8", layout_seed=0)
    d = bfs_distances(info, info.start)
    assert d[info.goal] > 0
    assert d[info.key] >= 0
