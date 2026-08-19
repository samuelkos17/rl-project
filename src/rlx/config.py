"""Run configuration. FROZEN CONTRACT -- field names are depended on by all
three workstreams. See CLAUDE.md section 6."""

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    env_id: str
    strategy: str
    seed: int

    # --- fixed across every strategy and every environment ---
    total_steps: int = 400_000
    buffer_size: int = 100_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    gamma: float = 0.99
    target_update: int = 1_000
    learning_starts: int = 1_000
    train_freq: int = 4
    grad_clip: float = 10.0

    # --- evaluation ---
    # 1, not 10: the layout is pinned per run and evaluation is greedy, so
    # MiniGrid is fully deterministic and 10 episodes would be identical.
    eval_every: int = 5_000
    eval_episodes: int = 1

    # --- logging ---
    # 10k, not 20k: the early-coverage window is the first 20% of training, so
    # at 20k only 4 snapshots would land inside it -- a 4-point trapezoid for the
    # project's main predictor. 10k gives 8 points. Storage is ~1 KB per
    # snapshot, so the resolution is effectively free.
    snapshot_every: int = 10_000

    # --- strategy hyperparameters (only the relevant ones are read) ---
    # tau_* and count_beta are calibrated to MiniGrid's reward scale, not
    # picked as round numbers: measured Q-gaps are ~0.0034, so the old
    # tau_end=0.05 left Boltzmann uniform-random for the whole run. Agreed by
    # all three of us on 2026-08-19, BEFORE any result was seen -- changing
    # them now would mean choosing our own result. See docs/decision_log.md.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_frac: float = 0.2
    tau_start: float = 0.01
    tau_end: float = 0.001
    tau_decay_frac: float = 0.4
    count_beta: float = 0.01
    count_epsilon: float = 0.05
    noisy_sigma0: float = 0.5

    # --- execution ---
    device: str = "cpu"
    results_root: str = "results"

    @property
    def run_dir(self) -> Path:
        return Path(self.results_root) / self.env_id / self.strategy / f"seed{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)


def load_base_config(path: Path) -> dict:
    """Read the shared defaults block from a sweep YAML file."""
    with open(path) as f:
        return yaml.safe_load(f).get("defaults", {})
