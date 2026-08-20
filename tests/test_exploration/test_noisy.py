import math

import numpy as np
import torch

from rlx.exploration.noisy import NoisyExplorer
from rlx.networks import NoisyLinear, QNetwork


def test_noisy_layer_has_learnable_mu_and_sigma():
    layer = NoisyLinear(8, 4, sigma0=0.5)
    names = {n for n, p in layer.named_parameters() if p.requires_grad}
    assert {"weight_mu", "weight_sigma", "bias_mu", "bias_sigma"} <= names


def test_output_changes_when_noise_is_resampled():
    torch.manual_seed(0)
    layer = NoisyLinear(8, 4, sigma0=0.5)
    x = torch.ones(1, 8)
    layer.reset_noise()
    a = layer(x).detach().clone()
    layer.reset_noise()
    b = layer(x).detach().clone()
    assert not torch.allclose(a, b)


def test_output_is_deterministic_when_noise_is_disabled():
    torch.manual_seed(0)
    layer = NoisyLinear(8, 4, sigma0=0.5)
    layer.noise_enabled = False
    x = torch.ones(1, 8)
    layer.reset_noise()
    a = layer(x).detach().clone()
    layer.reset_noise()
    assert torch.allclose(a, layer(x).detach())


def test_disabled_noise_uses_the_mu_weights_exactly():
    layer = NoisyLinear(8, 4, sigma0=0.5)
    layer.noise_enabled = False
    x = torch.ones(1, 8)
    expected = torch.nn.functional.linear(x, layer.weight_mu, layer.bias_mu)
    assert torch.allclose(layer(x), expected)


def test_sigma_receives_gradients():
    """If sigma never learns, this is not NoisyNets."""
    layer = NoisyLinear(8, 4, sigma0=0.5)
    layer.reset_noise()
    layer(torch.ones(1, 8)).sum().backward()
    assert layer.weight_sigma.grad is not None
    assert layer.weight_sigma.grad.abs().sum() > 0


def test_qnetwork_with_noisy_head_produces_varying_outputs():
    torch.manual_seed(0)
    net = QNetwork(n_actions=7, noisy=True, sigma0=0.5)
    x = torch.rand(1, 3, 7, 7)
    net.reset_noise()
    a = net(x).detach().clone()
    net.reset_noise()
    assert not torch.allclose(a, net(x).detach())


def test_qnetwork_with_noise_disabled_is_deterministic():
    net = QNetwork(n_actions=7, noisy=True, sigma0=0.5)
    net.set_noise_enabled(False)
    x = torch.rand(1, 3, 7, 7)
    net.reset_noise()
    a = net(x).detach().clone()
    net.reset_noise()
    assert torch.allclose(a, net(x).detach())


def test_explorer_declares_it_needs_a_noisy_network(cfg, rng):
    assert NoisyExplorer(cfg, rng).uses_noisy_net is True


def test_explorer_acts_purely_greedily(cfg, rng, q_values, key):
    e = NoisyExplorer(cfg, rng)
    assert all(e.act(q_values, key, s) == 3 for s in (0, 5000, 10_000))


def test_explorer_adds_no_intrinsic_bonus(cfg, rng, key):
    assert NoisyExplorer(cfg, rng).intrinsic_bonus(key) == 0.0


def test_noise_is_factorised_not_independent():
    """Factorised Gaussian noise: eps_out outer eps_in, so the matrix is rank 1.

    Independent per-weight noise would draw in_features*out_features Gaussians
    instead of in_features+out_features, giving a full-rank matrix. Both look
    identical from the outside -- same shape, same rough magnitude, noise still
    changes on resample -- so nothing else in this suite separates them. Verified
    2026-08-20 by monkeypatching reset_noise to draw independent noise: the whole
    suite still passed.
    """
    torch.manual_seed(0)
    layer = NoisyLinear(16, 8, sigma0=0.5)
    layer.reset_noise()
    assert np.linalg.matrix_rank(layer.weight_epsilon.numpy()) == 1

    # The bias noise IS the output factor, so weight_epsilon must reconstruct
    # exactly as outer(bias_epsilon, eps_in) -- recovering eps_in from any one
    # column. This is what pins bias_epsilon to the same draw as the weights
    # rather than being an independent vector that merely has the right shape.
    eps_in = layer.weight_epsilon[0, :] / layer.bias_epsilon[0]
    assert torch.allclose(layer.weight_epsilon,
                          torch.outer(layer.bias_epsilon, eps_in), atol=1e-6)


def test_sigma_is_scaled_by_fan_in():
    """sigma_init = sigma0 / sqrt(in_features), per Fortunato et al.

    Without the fan-in division sigma would be 32x too large on the 1024-input
    head layer, which would invalidate the scripts/measure_sigma.py calibration
    that chose noisy_sigma0 = 0.5. Verified 2026-08-20 that dropping the scaling
    passes every other test in the suite.
    """
    for in_features, sigma0 in [(8, 0.5), (64, 0.5), (1024, 0.5), (16, 0.25)]:
        layer = NoisyLinear(in_features, 4, sigma0=sigma0)
        expected = sigma0 / math.sqrt(in_features)
        assert np.isclose(layer.weight_sigma[0, 0].item(), expected), in_features
        assert np.isclose(layer.bias_sigma[0].item(), expected), in_features
