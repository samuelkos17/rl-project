from rlx.envs import ENV_IDS, difficulty_index


def test_there_are_thirteen_instances():
    assert len(ENV_IDS) == 13


def test_difficulty_increases_within_a_family():
    assert difficulty_index("DoorKey-5") < difficulty_index("DoorKey-10")
    assert difficulty_index("MultiRoom-N2") < difficulty_index("MultiRoom-N6")


def test_every_env_id_has_a_difficulty():
    assert all(difficulty_index(e) > 0 for e in ENV_IDS)
