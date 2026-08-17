"""The exploration strategy interface. FROZEN CONTRACT -- Max owns the
implementations, but the interface is agreed by all three. See CLAUDE.md
section 6.

`count_key` is the agent's OWN 7x7x3 observation, as raw bytes. It is NOT the
privileged (x, y, direction) state. See CLAUDE.md section 8.
"""

from abc import ABC, abstractmethod
from typing import Hashable

import numpy as np


class Explorer(ABC):
    #: True if this strategy needs the Q-network head built from NoisyLinear.
    uses_noisy_net: bool = False

    @abstractmethod
    def act(self, q_values: np.ndarray, count_key: Hashable, step: int) -> int:
        """Choose an action given Q-values for the current observation."""

    def intrinsic_bonus(self, count_key: Hashable) -> float:
        """Extra reward added to the transition stored in the replay buffer.

        Never enters the evaluation return. Default: no bonus.
        """
        return 0.0

    def observe(self, count_key: Hashable) -> None:
        """Called once per environment step, after acting. Default: no-op."""

    def stats(self) -> dict[str, float]:
        """Scalars to log this step (epsilon, temperature, mean bonus)."""
        return {}
