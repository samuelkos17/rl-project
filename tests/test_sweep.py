import pytest

from rlx.sweep import expand_matrix, pending_runs, select_shard

SPEC = {
    "defaults": {"total_steps": 1000, "device": "cpu"},
    "env_ids": ["Empty-5", "DoorKey-5"],
    "strategies": ["epsilon_greedy", "noisy"],
    "seeds": [0, 1],
}


def test_matrix_is_the_full_cross_product():
    assert len(expand_matrix(SPEC)) == 2 * 2 * 2


def test_defaults_are_applied_to_every_config():
    assert all(c.total_steps == 1000 for c in expand_matrix(SPEC))


def test_matrix_order_is_deterministic():
    a = [(c.env_id, c.strategy, c.seed) for c in expand_matrix(SPEC)]
    b = [(c.env_id, c.strategy, c.seed) for c in expand_matrix(SPEC)]
    assert a == b


def test_shards_partition_the_matrix_exactly_once():
    all_configs = expand_matrix(SPEC)
    covered = []
    for i in range(3):
        covered.extend(select_shard(all_configs, i, 3))
    keys = sorted((c.env_id, c.strategy, c.seed) for c in covered)
    expected = sorted((c.env_id, c.strategy, c.seed) for c in all_configs)
    assert keys == expected


def test_shards_are_roughly_balanced():
    all_configs = expand_matrix(SPEC)
    sizes = [len(select_shard(all_configs, i, 3)) for i in range(3)]
    assert max(sizes) - min(sizes) <= 1


def test_shard_index_out_of_range_is_rejected():
    configs = expand_matrix(SPEC)
    for bad in (-1, 3, 5):
        with pytest.raises(ValueError):
            select_shard(configs, bad, 3)


def test_completed_runs_are_skipped(tmp_path):
    spec = {**SPEC, "defaults": {**SPEC["defaults"], "results_root": str(tmp_path)}}
    configs = expand_matrix(spec)
    done = configs[0]
    done.run_dir.mkdir(parents=True)

    remaining = pending_runs(configs)
    assert len(remaining) == len(configs) - 1
    assert done.run_dir not in [c.run_dir for c in remaining]


def test_a_partial_directory_does_not_count_as_done(tmp_path):
    """A crashed run leaves seed<k>.partial/. If that counted as finished, the
    sweep would skip it forever and we would silently lose a data point."""
    spec = {**SPEC, "defaults": {**SPEC["defaults"], "results_root": str(tmp_path)}}
    configs = expand_matrix(spec)
    partial = configs[0].run_dir.with_name(configs[0].run_dir.name + ".partial")
    partial.mkdir(parents=True)

    assert len(pending_runs(configs)) == len(configs)


def test_the_real_matrix_is_260_runs():
    import yaml
    with open("configs/main.yaml") as f:
        spec = yaml.safe_load(f)
    configs = expand_matrix(spec)
    assert len(configs) == 260
    assert len({(c.env_id, c.strategy, c.seed) for c in configs}) == 260


def test_three_shards_of_the_real_matrix_cover_it_exactly():
    import yaml
    with open("configs/main.yaml") as f:
        spec = yaml.safe_load(f)
    configs = expand_matrix(spec)
    seen = []
    for i in range(3):
        seen.extend((c.env_id, c.strategy, c.seed) for c in select_shard(configs, i, 3))
    assert sorted(seen) == sorted((c.env_id, c.strategy, c.seed) for c in configs)


def test_configs_are_picklable():
    """Windows uses spawn for multiprocessing: every config crosses a process
    boundary, so an unpicklable field would only fail once the sweep launches."""
    import pickle
    for c in expand_matrix(SPEC):
        assert pickle.loads(pickle.dumps(c)) == c
