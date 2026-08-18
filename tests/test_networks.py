import torch

from rlx.networks import QNetwork, obs_batch_to_tensor, obs_to_tensor


def test_forward_maps_a_batch_of_observations_to_action_values():
    net = QNetwork(n_actions=7)
    x = torch.rand(5, 3, 7, 7)
    assert net(x).shape == (5, 7)


def test_single_observation_works():
    net = QNetwork(n_actions=7)
    assert net(torch.rand(1, 3, 7, 7)).shape == (1, 7)


def test_noisy_flag_builds_without_error_and_keeps_the_same_shape():
    net = QNetwork(n_actions=7, noisy=True, sigma0=0.5)
    assert net(torch.rand(2, 3, 7, 7)).shape == (2, 7)


def test_reset_noise_and_set_noise_enabled_exist_on_both_variants():
    for noisy in (False, True):
        net = QNetwork(n_actions=7, noisy=noisy)
        net.reset_noise()
        net.set_noise_enabled(False)
        net.set_noise_enabled(True)


def test_obs_to_tensor_moves_channels_first_and_adds_a_batch_dim():
    import numpy as np
    obs = np.zeros((7, 7, 3), dtype=np.uint8)
    obs[0, 0, 2] = 10                      # channel 2, corner
    t = obs_to_tensor(obs, "cpu")
    assert t.shape == (1, 3, 7, 7)
    assert t[0, 2, 0, 0] == 10.0           # the value landed in the right channel


def test_obs_batch_to_tensor_moves_channels_first():
    import numpy as np
    batch = np.zeros((4, 7, 7, 3), dtype=np.uint8)
    batch[1, 0, 0, 2] = 10
    t = obs_batch_to_tensor(batch, "cpu")
    assert t.shape == (4, 3, 7, 7)
    assert t[1, 2, 0, 0] == 10.0


def test_gradients_reach_every_parameter():
    """A layer with no gradient is a layer that never learns."""
    net = QNetwork(n_actions=7)
    net(torch.rand(4, 3, 7, 7)).sum().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"


def test_two_networks_start_with_different_weights():
    """Online and target must be independently initialised before syncing."""
    a, b = QNetwork(n_actions=7), QNetwork(n_actions=7)
    assert not torch.equal(a.head[0].weight, b.head[0].weight)
