import numpy as np
import pytest

from rlx.config import RunConfig
from rlx.exploration import make_explorer
from rlx.exploration.base import Explorer


class _Dummy(Explorer):
    def act(self, q_values, count_key, step):
        return int(np.argmax(q_values))


def test_explorer_cannot_be_instantiated_without_act():
    with pytest.raises(TypeError):
        Explorer()


def test_default_bonus_is_zero_and_observe_is_a_noop():
    e = _Dummy()
    assert e.intrinsic_bonus(("k",)) == 0.0
    assert e.observe(("k",)) is None
    assert e.stats() == {}
    assert e.uses_noisy_net is False


def test_act_returns_an_int_action():
    e = _Dummy()
    assert e.act(np.array([0.1, 0.9, 0.3]), ("k",), 0) == 1


@pytest.mark.parametrize("strategy,expected", [
    ("epsilon_greedy", {"epsilon"}),
    ("boltzmann", {"temperature"}),
    ("count_based", {"epsilon", "mean_bonus", "distinct_keys"}),
    ("noisy", set()),
])
def test_stats_keys_match_the_frozen_results_contract(strategy, expected):
    """CLAUDE.md section 5 fixes the extra metrics.csv column per strategy.

    Asserted as an EXACT set, not with `in`. Every other stats() test here uses
    membership, which catches a renamed key but not an ADDED one -- and an added
    key silently adds a column to metrics.csv, which is a frozen contract the
    analysis workstream reads. Changing this test means renegotiating that
    contract with the other two workstreams first.
    """
    cfg = RunConfig(env_id="Empty-5", strategy=strategy, seed=0, total_steps=10_000)
    explorer = make_explorer(strategy, cfg, np.random.default_rng(0))
    q = np.array([0.1, 0.2, 0.15, 0.9, 0.05, 0.3, 0.25])
    key = b"obs"
    explorer.act(q, key, 0)
    explorer.observe(key)
    explorer.intrinsic_bonus(key)

    assert set(explorer.stats()) == expected
    assert all(isinstance(v, float) for v in explorer.stats().values())
