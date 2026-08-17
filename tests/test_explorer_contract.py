import numpy as np
import pytest

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
