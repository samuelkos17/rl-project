"""Count-based exploration: pay the agent a shrinking bonus for visiting
situations it has seen rarely.

The counted key is the agent's OWN observation as raw bytes, never its true
(x, y, direction). See CLAUDE.md section 8.
"""

from collections import defaultdict
from typing import Hashable

import numpy as np

from rlx.config import RunConfig
from rlx.exploration.base import Explorer


class CountBased(Explorer):
    def __init__(self, cfg: RunConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        # Keys are raw observation bytes handed in by the training loop.
        self.counts: dict[Hashable, int] = defaultdict(int)
        self._recent_bonuses: list[float] = []

    def act(self, q_values: np.ndarray, count_key: Hashable, step: int) -> int:
        if self.rng.random() < self.cfg.count_epsilon:
            return int(self.rng.integers(len(q_values)))
        return int(np.argmax(q_values))

    def observe(self, count_key: Hashable) -> None:
        self.counts[count_key] += 1

    def intrinsic_bonus(self, count_key: Hashable) -> float:
        # .get() rather than self.counts[key]: counts is a defaultdict, so a bare
        # lookup INSERTS a zero entry, and distinct_keys would then report every
        # key ever asked about rather than every key actually observed. Harmless
        # in the real loop, which observes before it pays, but it made the metric
        # depend on call order for no reason.
        # max(count, 1) keeps an unseen key finite instead of dividing by zero.
        bonus = self.cfg.count_beta / np.sqrt(max(self.counts.get(count_key, 0), 1))
        self._recent_bonuses.append(bonus)
        if len(self._recent_bonuses) > 1000:
            del self._recent_bonuses[:-1000]
        return float(bonus)

    def stats(self) -> dict[str, float]:
        return {
            "epsilon": self.cfg.count_epsilon,
            "mean_bonus": float(np.mean(self._recent_bonuses)) if self._recent_bonuses else 0.0,
            "distinct_keys": float(len(self.counts)),
        }
