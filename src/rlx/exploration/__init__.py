"""Exploration strategies. Implementations are owned by Max (workstream B)."""

import numpy as np

from rlx.config import RunConfig
from rlx.exploration.base import Explorer

STRATEGIES = ("epsilon_greedy", "boltzmann", "count_based", "noisy")


def make_explorer(name: str, cfg: RunConfig, rng: np.random.Generator) -> Explorer:
    """Build the named exploration strategy."""
    if name == "epsilon_greedy":
        from rlx.exploration.epsilon_greedy import EpsilonGreedy
        return EpsilonGreedy(cfg, rng)
    if name == "boltzmann":
        from rlx.exploration.boltzmann import Boltzmann
        return Boltzmann(cfg, rng)
    if name == "count_based":
        from rlx.exploration.count_based import CountBased
        return CountBased(cfg, rng)
    if name == "noisy":
        from rlx.exploration.noisy import NoisyExplorer
        return NoisyExplorer(cfg, rng)
    raise ValueError(f"unknown strategy {name!r}, expected one of {STRATEGIES}")
