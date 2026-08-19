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
