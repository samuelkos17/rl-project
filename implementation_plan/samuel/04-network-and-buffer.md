# Task 4 — Q-network and replay buffer

**Files:**
- Create: `src/rlx/networks.py`, `src/rlx/buffer.py`
- Test: `tests/test_networks.py`, `tests/test_buffer.py`

**Interfaces:**
- Consumes: `RunConfig` from Task 1.
- Produces:
  - `QNetwork(n_actions: int, noisy: bool = False, sigma0: float = 0.5)` with
    `forward(x)`, `reset_noise()`, `set_noise_enabled(flag: bool)`
  - `ReplayBuffer(capacity: int, rng: np.random.Generator)` with `add(...)`,
    `sample(batch_size)`, `__len__`

**Max depends on `QNetwork`** — his NoisyNets task fills in `NoisyLinear`. This
task writes a *placeholder* `NoisyLinear` that is a plain `nn.Linear` so the
network is testable now; Max replaces the body, not the class name or signature.

---

- [ ] **Step 1: Write the failing network test**

Create `tests/test_networks.py`:

```python
import torch

from rlx.networks import QNetwork


def test_forward_maps_a_batch_of_observations_to_action_values():
    net = QNetwork(n_actions=7)
    x = torch.rand(5, 3, 7, 7)
    assert net(x).shape == (5, 7)


def test_single_observation_works():
    net = QNetwork(n_actions=7)
    assert net(torch.rand(1, 3, 7, 7)).shape == (1, 7)


def test_noisy_flag_builds_without_error_and_keeps_the_same_shape():
    net = QNetwork(n_actions=7, noisy=True, sigma0=0.5)
    assert net(torch.rand(2, 3, 7, 7)).shape == (2, 7)


def test_reset_noise_and_set_noise_enabled_exist_on_both_variants():
    for noisy in (False, True):
        net = QNetwork(n_actions=7, noisy=noisy)
        net.reset_noise()
        net.set_noise_enabled(False)
        net.set_noise_enabled(True)
```

- [ ] **Step 2: Run and watch it fail**

```bash
pytest tests/test_networks.py -v
```

Expected: `ModuleNotFoundError: No module named 'rlx.networks'`.

- [ ] **Step 3: Write `src/rlx/networks.py`**

```python
"""The Q-network. One architecture, used by every strategy.

NoisyLinear is a placeholder here (behaves exactly like nn.Linear). Max fills in
the real factorised-Gaussian implementation in workstream B task 4. The class
name, constructor signature, and the reset_noise / set_noise_enabled methods are
the contract -- do not change them.
"""

import torch
import torch.nn as nn

#: Observation channel values are small integers (object, colour, state indices).
#: Dividing by this keeps network inputs roughly in [0, 1].
OBS_SCALE = 10.0


class NoisyLinear(nn.Linear):
    """PLACEHOLDER -- currently a plain linear layer. Owned by Max (B task 4)."""

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__(in_features, out_features)
        self.sigma0 = sigma0
        self.noise_enabled = True

    def reset_noise(self) -> None:
        """Resample the noise. No-op until Max implements it."""


class QNetwork(nn.Module):
    """3-layer CNN over the 7x7x3 partial observation, then a 2-layer head."""

    def __init__(self, n_actions: int, noisy: bool = False, sigma0: float = 0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 2), nn.ReLU(),
            nn.Flatten(),
        )
        # 7x7 -> 6x6 -> 5x5 -> 4x4, so 64 * 4 * 4 features reach the head.
        n_features = 64 * 4 * 4
        if noisy:
            self.head = nn.Sequential(
                NoisyLinear(n_features, 64, sigma0), nn.ReLU(),
                NoisyLinear(64, n_actions, sigma0),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(n_features, 64), nn.ReLU(),
                nn.Linear(64, n_actions),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x / OBS_SCALE))

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def set_noise_enabled(self, flag: bool) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.noise_enabled = flag


def obs_to_tensor(obs, device: str) -> torch.Tensor:
    """(7,7,3) uint8 observation -> (1,3,7,7) float tensor on device."""
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    return x.permute(2, 0, 1).unsqueeze(0)


def obs_batch_to_tensor(obs_batch, device: str) -> torch.Tensor:
    """(B,7,7,3) uint8 batch -> (B,3,7,7) float tensor on device."""
    x = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    return x.permute(0, 3, 1, 2)
```

- [ ] **Step 4: Run the network tests**

```bash
pytest tests/test_networks.py -v
```

Expected: 4 passed. If the flatten size is wrong you will see a shape mismatch
naming the real number — use that number, do not guess a different one.

- [ ] **Step 5: Write the failing buffer test**

Create `tests/test_buffer.py`:

```python
import numpy as np

from rlx.buffer import ReplayBuffer


def _obs(v):
    return np.full((7, 7, 3), v, dtype=np.uint8)


def test_length_grows_then_caps_at_capacity():
    buf = ReplayBuffer(capacity=3, rng=np.random.default_rng(0))
    assert len(buf) == 0
    for i in range(5):
        buf.add(_obs(i), i % 7, float(i), _obs(i + 1), False)
    assert len(buf) == 3


def test_sample_returns_correctly_shaped_arrays():
    buf = ReplayBuffer(capacity=100, rng=np.random.default_rng(0))
    for i in range(50):
        buf.add(_obs(i), i % 7, 1.0, _obs(i + 1), i % 10 == 0)
    obs, act, rew, next_obs, done = buf.sample(8)
    assert obs.shape == (8, 7, 7, 3)
    assert next_obs.shape == (8, 7, 7, 3)
    assert act.shape == (8,)
    assert rew.shape == (8,)
    assert done.shape == (8,)


def test_oldest_entries_are_overwritten_first():
    buf = ReplayBuffer(capacity=2, rng=np.random.default_rng(0))
    buf.add(_obs(1), 0, 1.0, _obs(1), False)
    buf.add(_obs(2), 0, 2.0, _obs(2), False)
    buf.add(_obs(3), 0, 3.0, _obs(3), False)
    _, _, rew, _, _ = buf.sample(50)
    assert 1.0 not in set(rew.tolist())


def test_same_rng_seed_gives_the_same_sample():
    def draw():
        buf = ReplayBuffer(capacity=100, rng=np.random.default_rng(7))
        for i in range(50):
            buf.add(_obs(i), i % 7, float(i), _obs(i + 1), False)
        return buf.sample(8)[2]

    assert np.array_equal(draw(), draw())
```

- [ ] **Step 6: Run and watch it fail**

```bash
pytest tests/test_buffer.py -v
```

Expected: `ModuleNotFoundError: No module named 'rlx.buffer'`.

- [ ] **Step 7: Write `src/rlx/buffer.py`**

Observations are stored as `uint8` on purpose: 100,000 entries at `(7,7,3)` is
~15 MB per array, and `float32` would be four times that for no benefit.

```python
"""Fixed-size circular replay buffer."""

import numpy as np

OBS_SHAPE = (7, 7, 3)


class ReplayBuffer:
    """Stores the last `capacity` transitions and samples uniformly from them."""

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
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._action[i] = action
        self._reward[i] = reward
        self._done[i] = float(done)
        self._pos = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Returns (obs, action, reward, next_obs, done) as numpy arrays."""
        idx = self.rng.integers(0, self._size, size=batch_size)
        return (self._obs[idx], self._action[idx], self._reward[idx],
                self._next_obs[idx], self._done[idx])
```

- [ ] **Step 8: Run the buffer tests**

```bash
pytest tests/test_buffer.py -v
```

Expected: 4 passed.

- [ ] **Step 9: Run the whole suite**

```bash
pytest -v
```

Expected: everything green. Read the output — do not assume.

- [ ] **Step 10: Log and commit**

Append a short `docs/decision_log.md` entry noting that observations are stored
as `uint8` to keep the replay buffer around 15 MB rather than 60 MB, and that
`NoisyLinear` currently behaves like a normal layer until Max fills it in.

```bash
git add src/rlx/networks.py src/rlx/buffer.py tests/ docs/decision_log.md
git commit -m "feat: Q-network and replay buffer"
```

Tell Max that `QNetwork` and the `NoisyLinear` placeholder are on `main`.
