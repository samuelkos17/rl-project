from pathlib import Path

from rlx.config import RunConfig


def test_run_dir_is_nested_by_env_strategy_seed():
    cfg = RunConfig(env_id="DoorKey-5", strategy="epsilon_greedy", seed=3)
    assert cfg.run_dir == Path("results") / "DoorKey-5" / "epsilon_greedy" / "seed3"


def test_defaults_match_the_spec():
    cfg = RunConfig(env_id="Empty-5", strategy="epsilon_greedy", seed=0)
    assert cfg.buffer_size == 100_000
    assert cfg.batch_size == 32
    assert cfg.gamma == 0.99
    assert cfg.train_freq == 4
    assert cfg.eval_episodes == 1
