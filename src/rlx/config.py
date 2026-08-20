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
    # tau_* and count_beta are calibrated to MiniGrid's reward scale, not picked
    # as round numbers. The Q-value scale is NOT constant over a run, so the two
    # tau endpoints are calibrated against different measurements:
    # Both p(best) figures below use one stated convention: the best action
    # against six OTHER actions tied at the bottom of the gap. That is the
    # pessimistic reading -- the real 7-value softmax is more committed than
    # this (0.28 at step 0, 0.95 at the end, on the Q_INIT / Q_TRAINED vectors
    # in tests/test_exploration/test_boltzmann.py). Naming the convention
    # matters because the two readings differ by 18 points at the end.
    #   tau_start vs the RANDOM-INIT spread, 0.0206 (mean over 6 instances x 15
    #     seeds, sd across instances < 0.002 -- it is a property of the network
    #     init, not the maze). Gives p(best action) = 0.17 at step 0: above the
    #     uniform 0.143, which is Boltzmann's whole point, but not committed.
    #   tau_end vs the TRAINED best-vs-second gap, 0.0030 (6 instances x 2 seeds
    #     x 160k steps). Gives p(best) = 0.77 on an instance the agent solved and
    #     ~uniform on one it never solved -- commit where there is signal, keep
    #     exploring where there is none. (This said 0.98 until 2026-08-20; that
    #     figure reproduced under neither convention. The DECISION is unchanged
    #     -- 0.77 against a uniform 0.143 still commits where there is signal.)
    # Calibrating tau_start against the trained gap instead was the 2026-08-19
    # error: at tau_start=0.01 the pilot showed Boltzmann visiting 4 of ~36
    # states, i.e. spinning in place for 20k steps. Fixed 2026-08-19 on a
    # coverage criterion; eval return was 0.000 in both arms, so no value here
    # was chosen by looking at which strategy scores better. See
    # docs/decision_log.md.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_frac: float = 0.2
    tau_start: float = 0.1
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
