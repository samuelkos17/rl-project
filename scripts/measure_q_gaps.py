"""Measure the Q-value gap scale a Double DQN produces on MiniGrid.

Throwaway measurement for choosing Boltzmann's temperature schedule -- not
library code. The behaviour policy is epsilon-greedy on purpose: driving the
agent with Boltzmann would make the measured gaps depend on the tau we are
trying to choose.

cfg.total_steps stays at the real 400_000 so the epsilon schedule matches a real
run; --steps controls how much of it we actually execute.
"""

import argparse
import csv
import sys
import time

import numpy as np

from rlx.agent import DoubleDQNAgent
from rlx.buffer import ReplayBuffer
from rlx.config import RunConfig
from rlx.envs import make_env
from rlx.exploration import make_explorer
from rlx.train import _count_key, _pin_to_one_thread, _seed_everything


def measure(env_id, seed, run_steps, window):
    cfg = RunConfig(env_id=env_id, strategy="epsilon_greedy", seed=seed)  # total_steps=400_000
    rng = _seed_everything(cfg.seed)
    env = make_env(cfg.env_id, layout_seed=cfg.seed)
    obs, _ = env.reset()
    explorer = make_explorer(cfg.strategy, cfg, rng)
    agent = DoubleDQNAgent(env.action_space.n, cfg, noisy=False)
    buffer = ReplayBuffer(cfg.buffer_size, rng)

    rows, gaps, spreads, returns, ep_return = [], [], [], [], 0.0
    for step in range(run_steps):
        q = agent.q_values(obs)
        s = np.sort(q)
        gaps.append(float(s[-1] - s[-2]))      # best vs second-best: what Boltzmann trades on
        spreads.append(float(s[-1] - s[0]))    # best vs worst: the full range
        key = _count_key(obs)
        action = explorer.act(q, key, step)
        next_obs, reward, term, trunc, _ = env.step(action)
        explorer.observe(key)
        buffer.add(obs, action, float(reward), next_obs, term)
        ep_return += float(reward)
        obs = next_obs
        if term or trunc:
            returns.append(ep_return)
            ep_return = 0.0
            obs, _ = env.reset()
        if step >= cfg.learning_starts and step % cfg.train_freq == 0:
            agent.update(buffer.sample(cfg.batch_size))
        if step > 0 and step % cfg.target_update == 0:
            agent.sync_target()
        if (step + 1) % window == 0:
            rows.append({
                "env_id": env_id, "seed": seed, "step": step + 1,
                "gap_median": np.median(gaps), "gap_mean": np.mean(gaps),
                "gap_p90": np.percentile(gaps, 90), "spread_median": np.median(spreads),
                "train_return_last20": np.mean(returns[-20:]) if returns else 0.0,
            })
            gaps, spreads = [], []
    env.close()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--window", type=int, default=5_000)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    _pin_to_one_thread()
    all_rows = []
    for env_id in a.envs:
        for seed in a.seeds:
            t0 = time.perf_counter()
            rows = measure(env_id, seed, a.steps, a.window)
            all_rows += rows
            dt = time.perf_counter() - t0
            print(f"{env_id} seed{seed}: {a.steps} steps in {dt:.0f}s "
                  f"({a.steps/dt:.0f} steps/s)", flush=True)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {a.out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
