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


def test_terminal_transitions_bootstrap_nothing():
    """done=1 must drop the future term, or the agent learns value past the goal."""
    agent = DoubleDQNAgent(n_actions=7, cfg=_cfg(gamma=0.99), noisy=False)
    obs = np.zeros((1, 7, 7, 3), dtype=np.uint8)
    reward = np.array([1.0], dtype=np.float32)

    done_target = agent._compute_target(obs, reward, np.array([1.0], dtype=np.float32))
    live_target = agent._compute_target(obs, reward, np.array([0.0], dtype=np.float32))
    assert np.isclose(float(done_target[0]), 1.0)
    assert not np.isclose(float(live_target[0]), 1.0)


def test_double_dqn_target_differs_from_vanilla_max():
    """If these ever agree for a deliberately disagreeing pair of networks, the
    target is using target.max() and this is vanilla DQN, not Double DQN."""
    # Seed BEFORE the networks are built. torch's default RNG is seeded from OS
    # entropy, so with the seed below the constructor the weights differed every
    # process, and roughly 1 run in 20 drew a pair whose argmaxes agreed on all
    # 64 observations -- making the Double-DQN target equal the vanilla max and
    # failing the assert below on correct code. Seeded here, the two disagree on
    # all 64. (Daniel, 2026-08-19, from the tasks 1-6 review.)
    torch.manual_seed(1)
    agent = DoubleDQNAgent(n_actions=7, cfg=_cfg(), noisy=False)
    with torch.no_grad():
        for p in agent.target.parameters():
            p.add_(torch.randn_like(p) * 0.5)   # push target away from online

    obs = np.random.default_rng(0).integers(0, 10, (64, 7, 7, 3)).astype(np.uint8)
    reward = np.zeros(64, dtype=np.float32)
    done = np.zeros(64, dtype=np.float32)

    double = agent._compute_target(obs, reward, done)
    from rlx.networks import obs_batch_to_tensor
    with torch.no_grad():
        vanilla = (agent.cfg.gamma *
                   agent.target(obs_batch_to_tensor(obs, "cpu")).max(dim=1).values)
    assert not np.allclose(double, vanilla.numpy())
    assert (double <= vanilla.numpy() + 1e-6).all(), "double target must not exceed the max"


def test_gradients_are_clipped():
    """grad_clip caps the update; without it one freak batch can wreck training."""
    agent = DoubleDQNAgent(n_actions=7, cfg=_cfg(grad_clip=1e-6), noisy=False)
    before = agent.online.head[0].weight.detach().clone()
    agent.update(_batch())
    delta = (agent.online.head[0].weight.detach() - before).abs().max()
    assert delta < 1e-3, f"weights moved {delta} despite a tiny grad_clip"


def test_noisy_flag_builds_a_noisy_network():
    agent = DoubleDQNAgent(n_actions=7, cfg=_cfg(), noisy=True)
    from rlx.networks import NoisyLinear
    assert any(isinstance(m, NoisyLinear) for m in agent.online.modules())


def test_target_network_never_uses_noise():
    """The TD target must be deterministic for every strategy, noisy included.

    weight_epsilon/bias_epsilon are BUFFERS, so sync_target()'s load_state_dict
    copies the online net's current noise sample into the target. Before the
    2026-08-20 fix that sample then sat frozen for 1000 steps, acting as a fixed
    bias on the bootstrap target for the noisy arm only -- a difference in the
    LEARNING algorithm, which CLAUDE.md section 7 forbids across strategies.
    """
    from rlx.networks import NoisyLinear, obs_batch_to_tensor
    torch.manual_seed(0)
    agent = DoubleDQNAgent(n_actions=7, cfg=_cfg(), noisy=True)

    def flags(net):
        return [m.noise_enabled for m in net.modules() if isinstance(m, NoisyLinear)]

    assert flags(agent.target) == [False, False], "target must score with mean weights"
    assert flags(agent.online) == [True, True], "online must still explore"

    # and it must survive a sync, which is where the buffers get overwritten
    agent.online.reset_noise()
    agent.sync_target()
    assert flags(agent.target) == [False, False], "sync_target re-enabled target noise"

    # the target's output is identical across a resample + sync
    obs = np.random.default_rng(0).integers(0, 10, (4, 7, 7, 3)).astype(np.uint8)
    t = obs_batch_to_tensor(obs, "cpu")
    with torch.no_grad():
        before = agent.target(t).clone()
        agent.online.reset_noise()
        agent.sync_target()
        after = agent.target(t).clone()
    assert torch.equal(before, after), "target Q-values moved when only noise changed"
