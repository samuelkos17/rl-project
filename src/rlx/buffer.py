"""Fixed-size circular replay buffer."""

import numpy as np

OBS_SHAPE = (7, 7, 3)


class ReplayBuffer:
    """Stores the last `capacity` transitions and samples uniformly from them.

    Observations are held as uint8 on purpose: 100,000 entries at (7,7,3) is
    about 15 MB per array, and float32 would be four times that for no benefit.
    """

    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self._obs = np.zeros((capacity, *OBS_SHAPE), dtype=np.uint8)
        self._next_obs = np.zeros((capacity, *OBS_SHAPE), dtype=np.uint8)
        self._action = np.zeros(capacity, dtype=np.int64)
        self._reward = np.zeros(capacity, dtype=np.float32)
        self._done = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        i = self._pos
        # Assigning into a preallocated row copies, so a later mutation of the
        # caller's array cannot rewrite what we stored.
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._action[i] = action
        self._reward[i] = reward
        self._done[i] = float(done)
        self._pos = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Returns (obs, action, reward, next_obs, done) as numpy arrays.

        Samples only from the first `_size` slots, so unwritten rows are never
        drawn while the buffer is still filling up.
        """
        idx = self.rng.integers(0, self._size, size=batch_size)
        return (self._obs[idx], self._action[idx], self._reward[idx],
                self._next_obs[idx], self._done[idx])
