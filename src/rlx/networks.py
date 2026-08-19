"""The Q-network. One architecture, used by every strategy.

NoisyLinear implements factorised Gaussian weight noise (workstream B task 4).
The class name, the constructor signature, and the reset_noise / noise_enabled
members are the contract -- do not change them.
"""

import math

import torch
import torch.nn as nn

#: Observation channel values are small integers (object, colour, state indices).
#: Dividing by this keeps network inputs roughly in [0, 1].
OBS_SCALE = 10.0


class NoisyLinear(nn.Module):
    """Linear layer with learned factorised Gaussian weight noise.

    weight = weight_mu + weight_sigma * epsilon, with epsilon resampled by
    reset_noise(). weight_sigma is learned, so the network decides for itself
    how much randomness each weight still needs.
    """

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma0 = sigma0
        self.noise_enabled = True

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Buffers, not parameters: noise is resampled, never learned.
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, sigma0 * bound)
        nn.init.constant_(self.bias_sigma, sigma0 * bound)

        self.reset_noise()

    @staticmethod
    def _scaled_noise(size: int, device) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """Resample the factorised noise: one vector per input, one per output."""
        device = self.weight_mu.device
        eps_in = self._scaled_noise(self.in_features, device)
        eps_out = self._scaled_noise(self.out_features, device)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.noise_enabled:
            return nn.functional.linear(x, self.weight_mu, self.bias_mu)
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return nn.functional.linear(x, weight, bias)


class QNetwork(nn.Module):
    """3-layer CNN over the 7x7x3 partial observation, then a 2-layer head."""

    def __init__(self, n_actions: int, noisy: bool = False, sigma0: float = 0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 2), nn.ReLU(),
            nn.Flatten(),
        )
        # 7x7 -> 6x6 -> 5x5 -> 4x4, so 64 * 4 * 4 features reach the head.
        n_features = 64 * 4 * 4
        if noisy:
            self.head = nn.Sequential(
                NoisyLinear(n_features, 64, sigma0), nn.ReLU(),
                NoisyLinear(64, n_actions, sigma0),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(n_features, 64), nn.ReLU(),
                nn.Linear(64, n_actions),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x / OBS_SCALE))

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def set_noise_enabled(self, flag: bool) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.noise_enabled = flag


def obs_to_tensor(obs, device: str) -> torch.Tensor:
    """(7,7,3) uint8 observation -> (1,3,7,7) float tensor on device."""
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    return x.permute(2, 0, 1).unsqueeze(0)


def obs_batch_to_tensor(obs_batch, device: str) -> torch.Tensor:
    """(B,7,7,3) uint8 batch -> (B,3,7,7) float tensor on device."""
    x = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    return x.permute(0, 3, 1, 2)
