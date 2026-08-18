# Task 5 — Double DQN agent and the training loop

This is the centre of the project. Everything else feeds it or reads its output.

**Files:**
- Create: `src/rlx/agent.py`, `src/rlx/train.py`
- Test: `tests/test_agent.py`, `tests/test_train.py`

**Interfaces:**
- Consumes:
  - `RunConfig` (Task 1), `make_env` / `grid_info` (Task 3),
    `QNetwork` / `obs_to_tensor` / `obs_batch_to_tensor` (Task 4)
  - `make_explorer(name, cfg, rng)` (Task 1) — Max's implementations
  - **`RunLogger` from Daniel's `src/rlx/logging.py`**, exactly this interface:
    ```python
    RunLogger(cfg: RunConfig, width: int, height: int)
    .record_visit(x: int, y: int, direction: int) -> None
    .log_step(step: int, **scalars: float) -> None
    .snapshot(step: int) -> None
    .distinct_states() -> int
    .finalize(meta: dict) -> None
    ```
- Produces: `DoubleDQNAgent`, `run_training(cfg) -> Path`, and a CLI entry point.

**If Daniel's `logging.py` is not merged yet**, write a 20-line stub locally with
those six methods doing nothing, and delete it when his lands. Do not block.

---

- [x] **Step 1: Write the failing agent test**

Create `tests/test_agent.py`:

```python
import numpy as np
import torch

from rlx.agent import DoubleDQNAgent
from rlx.config import RunConfig


def _cfg(**kw):
    return RunConfig(env_id="Empty-5", strategy="epsilon_greedy", seed=0, **kw)


def _batch(n=32):
    rng = np.random.default_rng(0)
    return (rng.integers(0, 10, (n, 7, 7, 3)).astype(np.uint8),
            rng.integers(0, 7, n),
            rng.random(n).astype(np.float32),
            rng.integers(0, 10, (n, 7, 7, 3)).astype(np.uint8),
            np.zeros(n, dtype=np.float32))


def test_q_values_have_one_entry_per_action():
    agent = DoubleDQNAgent(n_actions=7, cfg=_cfg(), noisy=False)
    q = agent.q_values(np.zeros((7, 7, 3), dtype=np.uint8))
    assert q.shape == (7,)
    assert isinstance(q, np.ndarray)


def test_update_returns_a_finite_loss_and_changes_the_weights():
    agent = DoubleDQNAgent(n_actions=7, cfg=_cfg(), noisy=False)
    before = agent.online.head[0].weight.detach().clone()
    loss = agent.update(_batch())
    assert np.isfinite(loss)
    assert not torch.equal(before, agent.online.head[0].weight.detach())


def test_target_network_only_changes_when_synced():
    agent = DoubleDQNAgent(n_actions=7, cfg=_cfg(), noisy=False)
    before = agent.target.head[0].weight.detach().clone()
    agent.update(_batch())
    assert torch.equal(before, agent.target.head[0].weight.detach())
    agent.sync_target()
    assert torch.equal(agent.online.head[0].weight.detach(),
                       agent.target.head[0].weight.detach())


def test_target_and_online_start_identical():
    agent = DoubleDQNAgent(n_actions=7, cfg=_cfg(), noisy=False)
    assert torch.equal(agent.online.head[0].weight.detach(),
                       agent.target.head[0].weight.detach())
```

- [x] **Step 2: Run and watch it fail**

```bash
pytest tests/test_agent.py -v
```

Expected: `ModuleNotFoundError: No module named 'rlx.agent'`.

- [x] **Step 3: Write `src/rlx/agent.py`**

The one line that makes this Double DQN rather than vanilla DQN is marked. Read
the comment there — it is the thing the report has to justify.

```python
"""Double DQN. One algorithm, fixed for every strategy and every environment."""

import numpy as np
import torch
import torch.nn as nn

from rlx.config import RunConfig
from rlx.networks import QNetwork, obs_batch_to_tensor, obs_to_tensor


class DoubleDQNAgent:
    """Online + target Q-networks with the Double DQN target."""

    def __init__(self, n_actions: int, cfg: RunConfig, noisy: bool):
        self.cfg = cfg
        self.device = cfg.device
        self.online = QNetwork(n_actions, noisy, cfg.noisy_sigma0).to(cfg.device)
        self.target = QNetwork(n_actions, noisy, cfg.noisy_sigma0).to(cfg.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.learning_rate)

    def q_values(self, obs) -> np.ndarray:
        """Q-values for one observation, as a plain numpy array."""
        with torch.no_grad():
            q = self.online(obs_to_tensor(obs, self.device))
        return q.squeeze(0).cpu().numpy()

    def update(self, batch) -> float:
        obs, action, reward, next_obs, done = batch
        obs_t = obs_batch_to_tensor(obs, self.device)
        next_obs_t = obs_batch_to_tensor(next_obs, self.device)
        action_t = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        q = self.online(obs_t).gather(1, action_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # THIS is Double DQN: the ONLINE net picks the best next action, the
            # TARGET net scores it. Vanilla DQN uses target.max(), which lets the
            # max operator pick up noise and systematically overestimates Q. That
            # overestimation acts like an accidental exploration bonus -- exactly
            # the thing this project is trying to measure, so we remove it.
            best_next = self.online(next_obs_t).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs_t).gather(1, best_next).squeeze(1)
            target_q = reward_t + self.cfg.gamma * (1.0 - done_t) * next_q

        loss = nn.functional.smooth_l1_loss(q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
```

- [x] **Step 4: Run the agent tests**

```bash
pytest tests/test_agent.py -v
```

Expected: 4 passed.

- [x] **Step 5: Write the failing training-loop test**

Create `tests/test_train.py`. These are slow-ish (a few seconds); that is fine.

```python
import json

import pandas as pd

from rlx.config import RunConfig
from rlx.train import run_training


def _cfg(tmp_path, **kw):
    return RunConfig(
        env_id="Empty-5", strategy="epsilon_greedy", seed=0,
        total_steps=2_000, learning_starts=100, eval_every=500,
        snapshot_every=500, buffer_size=1_000,
        results_root=str(tmp_path), **kw,
    )


def test_a_short_run_writes_a_schema_valid_result_directory(tmp_path):
    run_dir = run_training(_cfg(tmp_path))

    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "visitation.npz").exists()
    assert (run_dir / "meta.json").exists()

    df = pd.read_csv(run_dir / "metrics.csv")
    for col in ("step", "eval_return_mean", "distinct_states"):
        assert col in df.columns
    assert len(df) > 0

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["completed"] is True


def test_no_partial_directory_is_left_behind(tmp_path):
    run_training(_cfg(tmp_path))
    assert not list(tmp_path.rglob("*.partial"))


def test_the_same_seed_reproduces_the_same_metrics(tmp_path):
    a = pd.read_csv(run_training(_cfg(tmp_path / "a")) / "metrics.csv")
    b = pd.read_csv(run_training(_cfg(tmp_path / "b")) / "metrics.csv")
    pd.testing.assert_frame_equal(a, b)


def test_every_strategy_runs_end_to_end(tmp_path):
    for strategy in ("epsilon_greedy", "boltzmann", "count_based", "noisy"):
        cfg = _cfg(tmp_path / strategy)
        cfg.strategy = strategy
        run_dir = run_training(cfg)
        assert (run_dir / "metrics.csv").exists()
```

- [x] **Step 6: Run and watch it fail**

```bash
pytest tests/test_train.py -v
```

Expected: `ModuleNotFoundError: No module named 'rlx.train'`.

- [x] **Step 7: Write `src/rlx/train.py`**

Three things in here are easy to get wrong and each one would quietly invalidate
the experiment. They are marked `CRITICAL` in the code.

```python
"""The training loop. One run = one (env_id, strategy, seed) triple."""

import argparse
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from rlx.agent import DoubleDQNAgent
from rlx.buffer import ReplayBuffer
from rlx.config import RunConfig
from rlx.envs import make_env
from rlx.exploration import make_explorer
from rlx.logging import RunLogger


def _seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _count_key(obs) -> bytes:
    """The agent's OWN observation, as raw bytes, used as a dict key.

    CRITICAL: this is deliberately NOT (x, y, direction). The agent never sees
    its true position, so letting one strategy count true states would give it
    privileged information the other three do not get. See CLAUDE.md section 8.

    We return the bytes themselves rather than hash(obs.tobytes()). Both work:
    hashing would also be correct, because nothing here depends on the key's
    value, only on identity. Raw bytes are used simply because they remove hash
    collisions entirely rather than making them merely unlikely, and dicts hash
    the bytes internally anyway. The cost is a few MB of extra memory.
    """
    return obs.tobytes()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def evaluate(agent: DoubleDQNAgent, cfg: RunConfig) -> tuple[float, float]:
    """Greedy evaluation on extrinsic reward only.

    CRITICAL: no intrinsic bonus, no exploration noise. The bonus exists only
    inside the replay buffer and must never reach a reported number.
    """
    agent.online.set_noise_enabled(False)
    env = make_env(cfg.env_id, layout_seed=cfg.seed)
    returns = []
    for _ in range(cfg.eval_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            action = int(np.argmax(agent.q_values(obs)))
            obs, reward, term, trunc, _ = env.step(action)
            total += float(reward)
            done = term or trunc
        returns.append(total)
    env.close()
    agent.online.set_noise_enabled(True)
    return float(np.mean(returns)), float(np.std(returns))


def run_training(cfg: RunConfig) -> Path:
    """Train one configuration and write its result directory. Returns the path."""
    rng = _seed_everything(cfg.seed)

    # CRITICAL: layout_seed = cfg.seed pins one maze for the whole run.
    env = make_env(cfg.env_id, layout_seed=cfg.seed)
    obs, _ = env.reset()
    n_actions = env.action_space.n
    u = env.unwrapped

    explorer = make_explorer(cfg.strategy, cfg, rng)
    agent = DoubleDQNAgent(n_actions, cfg, noisy=explorer.uses_noisy_net)
    buffer = ReplayBuffer(cfg.buffer_size, rng)
    logger = RunLogger(cfg, width=u.width, height=u.height)

    episode_return, episode_returns, loss = 0.0, [], float("nan")
    start_time = time.perf_counter()

    for step in range(cfg.total_steps):
        logger.record_visit(int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir))

        key = _count_key(obs)
        if explorer.uses_noisy_net:
            agent.online.reset_noise()
        action = explorer.act(agent.q_values(obs), key, step)

        next_obs, reward, term, trunc, _ = env.step(action)
        explorer.observe(key)

        # The intrinsic bonus goes into the buffer and nowhere else.
        stored_reward = float(reward) + explorer.intrinsic_bonus(key)
        buffer.add(obs, action, stored_reward, next_obs, term)

        episode_return += float(reward)
        obs = next_obs
        if term or trunc:
            episode_returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()

        if step >= cfg.learning_starts and step % cfg.train_freq == 0:
            loss = agent.update(buffer.sample(cfg.batch_size))
        if step > 0 and step % cfg.target_update == 0:
            agent.sync_target()
        if step > 0 and step % cfg.snapshot_every == 0:
            logger.snapshot(step)

        if step % cfg.eval_every == 0:
            mean, std = evaluate(agent, cfg)
            logger.log_step(
                step,
                eval_return_mean=mean,
                eval_return_std=std,
                train_return_mean=float(np.mean(episode_returns[-20:]))
                                  if episode_returns else 0.0,
                episodes=len(episode_returns),
                distinct_states=logger.distinct_states(),
                loss=loss,
                **explorer.stats(),
            )

    logger.snapshot(cfg.total_steps)
    env.close()
    logger.finalize({
        "git_sha": _git_sha(),
        "hostname": platform.node(),
        "device": cfg.device,
        "wall_time_s": round(time.perf_counter() - start_time, 1),
        "completed": True,
    })
    return cfg.run_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", required=True)
    p.add_argument("--strategy", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--total-steps", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--results-root", default="results")
    args = p.parse_args()

    cfg = RunConfig(env_id=args.env_id, strategy=args.strategy, seed=args.seed,
                    device=args.device, results_root=args.results_root)
    if args.total_steps:
        cfg.total_steps = args.total_steps

    run_dir = run_training(cfg)
    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
```

- [x] **Step 8: Run the training tests**

```bash
pytest tests/test_train.py -v
```

Expected: 4 passed.

`test_every_strategy_runs_end_to_end` fails until Max's four modules exist. If
they are not merged yet, mark that one test `@pytest.mark.xfail(reason="needs
workstream B")` and remove the marker on integration day. **Do not delete the
test.**

- [x] **Step 9: Run one real short run by hand**

```bash
python -m rlx.train --env-id Empty-5 --strategy epsilon_greedy --seed 0 --total-steps 20000
```

Then look at what it produced:

```bash
python -c "
import pandas as pd
df = pd.read_csv('results/Empty-5/epsilon_greedy/seed0/metrics.csv')
print(df.to_string())
"
```

**Read the numbers.** On `Empty-5`, epsilon-greedy should reach an evaluation
return near 0.9 within 20k steps. If it is stuck at 0.0, something is broken —
debug it now, on the easiest environment, not on integration day.

Two likely causes: the observation is not being permuted to `(3,7,7)` before the
conv layers, or the target network is never being synced.

- [x] **Step 10: Run the whole suite**

```bash
pytest -v
```

- [ ] **Step 11: Log and commit**

Append a `docs/decision_log.md` entry recording that the run works end to end,
what evaluation return `Empty-5` reached, and how long 20k steps took — the last
number tells all three of us whether the step budget from Task 2 still holds.

```bash
git add src/rlx/agent.py src/rlx/train.py tests/ docs/decision_log.md
git commit -m "feat: Double DQN agent and training loop"
```
