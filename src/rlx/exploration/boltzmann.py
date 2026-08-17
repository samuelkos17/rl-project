"""Boltzmann (softmax) exploration: sample actions in proportion to
exp(Q / tau), with tau decaying exponentially."""

from typing import Hashable

import numpy as np

from rlx.config import RunConfig
from rlx.exploration.base import Explorer


class Boltzmann(Explorer):
    def __init__(self, cfg: RunConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self._decay_steps = max(1, int(cfg.tau_decay_frac * cfg.total_steps))
        self._tau = cfg.tau_start

    def temperature(self, step: int) -> float:
        """Exponential decay from tau_start to tau_end, then constant."""
        frac = min(1.0, step / self._decay_steps)
        tau = self.cfg.tau_start * (self.cfg.tau_end / self.cfg.tau_start) ** frac
        return max(self.cfg.tau_end, tau)

    def probabilities(self, q_values: np.ndarray, tau: float) -> np.ndarray:
        """Softmax over Q/tau. Subtracting the max prevents overflow."""
        scaled = q_values / max(tau, 1e-8)
        shifted = np.exp(scaled - scaled.max())
        return shifted / shifted.sum()

    def act(self, q_values: np.ndarray, count_key: Hashable, step: int) -> int:
        self._tau = self.temperature(step)
        p = self.probabilities(q_values, self._tau)
        return int(self.rng.choice(len(q_values), p=p))

    def stats(self) -> dict[str, float]:
        return {"temperature": self._tau}
