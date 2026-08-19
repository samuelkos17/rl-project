"""How much does NoisyNets' weight noise actually perturb the Q-values, relative
to the Q-gaps the network produces? Same question we asked of Boltzmann's tau.

Reports P(argmax flips when noise is resampled) -- the direct measure of how much
NoisyNets explores. 1/7 = 0.143 means uniform-random; 0.0 means no exploration.
"""
import argparse
import numpy as np
import torch

from rlx.agent import DoubleDQNAgent
from rlx.buffer import ReplayBuffer
from rlx.config import RunConfig
from rlx.envs import make_env
from rlx.exploration import make_explorer
from rlx.networks import obs_to_tensor
from rlx.train import _count_key, _pin_to_one_thread, _seed_everything


def flip_rate(net, obs_list, device, draws=30):
    """Fraction of (obs, resample) pairs where the greedy action changes."""
    flips, gaps, noise_sd = [], [], []
    for obs in obs_list:
        x = obs_to_tensor(obs, device)
        net.set_noise_enabled(False)
        with torch.no_grad():
            clean = net(x).cpu().numpy().ravel()
        s = np.sort(clean)
        gaps.append(s[-1] - s[-2])
        base = int(clean.argmax())
        net.set_noise_enabled(True)
        qs = []
        for _ in range(draws):
            net.reset_noise()
            with torch.no_grad():
                q = net(x).cpu().numpy().ravel()
            qs.append(q)
            flips.append(int(q.argmax()) != base)
        noise_sd.append(np.std(np.array(qs), axis=0).mean())
    net.set_noise_enabled(True)
    return np.mean(flips), np.median(gaps), np.median(noise_sd)


def sigma_stats(net):
    vals = [m.weight_sigma.detach().abs().mean().item()
            for m in net.modules() if hasattr(m, "weight_sigma")]
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Empty-5")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--sigma0", type=float, nargs="+", default=[0.5])
    a = ap.parse_args()
    _pin_to_one_thread()

    for sigma0 in a.sigma0:
        cfg = RunConfig(env_id=a.env, strategy="noisy", seed=0, noisy_sigma0=sigma0)
        rng = _seed_everything(cfg.seed)
        env = make_env(cfg.env_id, layout_seed=cfg.seed)
        obs, _ = env.reset()
        explorer = make_explorer("noisy", cfg, rng)
        agent = DoubleDQNAgent(env.action_space.n, cfg, noisy=True)
        buffer = ReplayBuffer(cfg.buffer_size, rng)

        probe = [obs]
        for step in range(a.steps):
            agent.online.reset_noise()
            action = explorer.act(agent.q_values(obs), _count_key(obs), step)
            next_obs, reward, term, trunc, _ = env.step(action)
            buffer.add(obs, action, float(reward), next_obs, term)
            obs = next_obs
            if term or trunc:
                obs, _ = env.reset()
            if len(probe) < 40 and step % 97 == 0:
                probe.append(obs)
            if step >= cfg.learning_starts and step % cfg.train_freq == 0:
                agent.update(buffer.sample(cfg.batch_size))
            if step > 0 and step % cfg.target_update == 0:
                agent.sync_target()
            if step in (0, 10_000, 25_000, a.steps - 1):
                fr, gap, nsd = flip_rate(agent.online, probe, cfg.device)
                print(f"sigma0={sigma0:<5} {a.env:<13} step {step:>6}  "
                      f"flip={fr:.3f}  gap={gap:.5f}  noise_sd={nsd:.5f}  "
                      f"mean|sigma|={sigma_stats(agent.online):.5f}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
