"""Epsilon-greedy: the baseline. Act randomly with probability epsilon,
otherwise act greedily. Epsilon decays linearly then holds at a floor."""

from typing import Hashable

import numpy as np

from rlx.config import RunConfig
from rlx.exploration.base import Explorer


class EpsilonGreedy(Explorer):
    def __init__(self, cfg: RunConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self._decay_steps = max(1, int(cfg.epsilon_decay_frac * cfg.total_steps))
        self._epsilon = cfg.epsilon_start

    def epsilon(self, step: int) -> float:
        """Linear decay from epsilon_start to epsilon_end, then constant."""
        frac = min(1.0, step / self._decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, q_values: np.ndarray, count_key: Hashable, step: int) -> int:
        self._epsilon = self.epsilon(step)
        if self.rng.random() < self._epsilon:
            return int(self.rng.integers(len(q_values)))
        return int(np.argmax(q_values))

    def stats(self) -> dict[str, float]:
        return {"epsilon": self._epsilon}
