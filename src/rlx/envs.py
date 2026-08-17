"""Environment naming and difficulty. FROZEN CONTRACT -- Daniel's analysis
imports difficulty_index. Task 3 adds the factory and grid BFS to this module."""

ENV_IDS = (
    "Empty-5", "Empty-8", "Empty-16",
    "DoorKey-5", "DoorKey-6", "DoorKey-7", "DoorKey-8", "DoorKey-10",
    "MultiRoom-N2", "MultiRoom-N3", "MultiRoom-N4", "MultiRoom-N5", "MultiRoom-N6",
)


def difficulty_index(env_id: str) -> int:
    """Grid size for Empty/DoorKey, room count for MultiRoom."""
    return int(env_id.split("-")[1].lstrip("N"))
