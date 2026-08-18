"""The Q-network. One architecture, used by every strategy.

NoisyLinear is a PLACEHOLDER here (it behaves exactly like nn.Linear). Max fills
in the real factorised-Gaussian implementation in workstream B task 4. The class
name, the constructor signature, and the reset_noise / noise_enabled members are
the contract -- do not change them.
"""

import torch
import torch.nn as nn

#: Observation channel values are small integers (object, colour, state indices).
#: Dividing by this keeps network inputs roughly in [0, 1].
OBS_SCALE = 10.0


class NoisyLinear(nn.Linear):
    """PLACEHOLDER -- currently a plain linear layer. Owned by Max (B task 4)."""

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__(in_features, out_features)
        self.sigma0 = sigma0
        self.noise_enabled = True

    def reset_noise(self) -> None:
        """Resample the noise. No-op until Max implements it."""


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
