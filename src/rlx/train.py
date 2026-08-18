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
    # Same pinned layout as training -- that maze IS the task being scored.
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

        # CRITICAL: the intrinsic bonus goes into the buffer and nowhere else.
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
