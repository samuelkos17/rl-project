"""NoisyNets: exploration comes from learned weight noise inside the network,
so action selection is purely greedy."""

from typing import Hashable

import numpy as np

from rlx.config import RunConfig
from rlx.exploration.base import Explorer


class NoisyExplorer(Explorer):
    uses_noisy_net = True

    def __init__(self, cfg: RunConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def act(self, q_values: np.ndarray, count_key: Hashable, step: int) -> int:
        # Greedy on purpose. The noise already happened inside the network.
        return int(np.argmax(q_values))
