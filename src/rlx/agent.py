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
        # The target scores with mean weights only, for every strategy.
        #
        # weight_epsilon / bias_epsilon are BUFFERS, so load_state_dict copies the
        # online net's current noise sample into the target on every sync_target().
        # That sample then sits frozen across the next 1000 steps (250 gradient
        # updates), which is a fixed bias on the TD target rather than zero-mean
        # noise -- measured at |dQ| 0.041 against a signal of |Q| 0.057. Only the
        # noisy arm would get it, so the four arms would no longer share one
        # learning algorithm, which CLAUDE.md section 7 requires. Exploration noise
        # belongs in action selection (the online net), not in the target.
        #
        # noise_enabled is a plain attribute, not a buffer, so load_state_dict
        # leaves it alone and this survives every later sync_target().
        self.target.set_noise_enabled(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.learning_rate)

    def q_values(self, obs) -> np.ndarray:
        """Q-values for one observation, as a plain numpy array."""
        with torch.no_grad():
            q = self.online(obs_to_tensor(obs, self.device))
        return q.squeeze(0).cpu().numpy()

    def _compute_target(self, next_obs, reward, done) -> np.ndarray:
        """The Double DQN target. Extracted so it can be tested directly.

        THIS is what makes it Double DQN: the ONLINE net picks the best next
        action, the TARGET net scores it. Vanilla DQN uses target.max(), which
        lets the max operator pick up noise and systematically overestimates Q.
        That overestimation acts like an accidental exploration bonus -- exactly
        the thing this project measures -- so we remove it.
        """
        next_obs_t = obs_batch_to_tensor(next_obs, self.device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            best_next = self.online(next_obs_t).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs_t).gather(1, best_next).squeeze(1)
            target_q = reward_t + self.cfg.gamma * (1.0 - done_t) * next_q
        return target_q.cpu().numpy()

    def update(self, batch) -> float:
        obs, action, reward, next_obs, done = batch
        obs_t = obs_batch_to_tensor(obs, self.device)
        action_t = torch.as_tensor(action, dtype=torch.int64, device=self.device)

        q = self.online(obs_t).gather(1, action_t.unsqueeze(1)).squeeze(1)
        target_q = torch.as_tensor(self._compute_target(next_obs, reward, done),
                                   dtype=torch.float32, device=self.device)

        loss = nn.functional.smooth_l1_loss(q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
