"""Visitation logging and the result-directory writer.

We log the agent's TRUE (x, y, direction). That is privileged information the
agent never receives: it is measurement, not learning. See CLAUDE.md section 8.

Only raw counts are written. Every coverage metric is derived later in
rlx.analysis.coverage, so metric definitions can change without re-running
experiments.
"""

import json
import shutil

import numpy as np
import pandas as pd

from rlx.config import RunConfig

N_DIRECTIONS = 4


class RunLogger:
    """Accumulates visit counts and metric rows, then writes one result directory."""

    def __init__(self, cfg: RunConfig, width: int, height: int):
        self.cfg = cfg
        self.counts = np.zeros((width, height, N_DIRECTIONS), dtype=np.int32)
        self._snapshot_steps: list[int] = []
        self._snapshots: list[np.ndarray] = []
        self._rows: list[dict] = []
        self._partial = cfg.run_dir.with_name(cfg.run_dir.name + ".partial")

    def record_visit(self, x: int, y: int, direction: int) -> None:
        self.counts[x, y, direction] += 1

    def distinct_states(self) -> int:
        """Number of distinct (x, y, direction) triples seen at least once."""
        return int((self.counts > 0).sum())

    def log_step(self, step: int, **scalars: float) -> None:
        self._rows.append({"step": step, **scalars})

    def snapshot(self, step: int) -> None:
        """Store a copy of the cumulative count array at this step."""
        self._snapshot_steps.append(step)
        self._snapshots.append(self.counts.copy())

    def finalize(self, meta: dict) -> None:
        """Write the result directory atomically: build .partial, then rename."""
        if self._partial.exists():
            shutil.rmtree(self._partial)
        self._partial.mkdir(parents=True)

        (self._partial / "config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))
        (self._partial / "meta.json").write_text(json.dumps(meta, indent=2))
        pd.DataFrame(self._rows).to_csv(self._partial / "metrics.csv", index=False)
        np.savez_compressed(
            self._partial / "visitation.npz",
            steps=np.array(self._snapshot_steps, dtype=np.int64),
            counts=(np.stack(self._snapshots) if self._snapshots
                    else np.zeros((0, *self.counts.shape), dtype=np.int32)),
        )

        if self.cfg.run_dir.exists():
            shutil.rmtree(self.cfg.run_dir)
        self._partial.rename(self.cfg.run_dir)
