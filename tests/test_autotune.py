"""Autotune pinning without a device: which configuration each autotuner keeps, and the shipped tables."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from prismyra.kernels.autotune import PINNED_DIR, load_table, matches, pin_all


def config(warps, stages, **kwargs):
    return SimpleNamespace(num_warps=warps, num_stages=stages, kwargs=kwargs)


def tuner(name, *configs):
    fn = SimpleNamespace(__name__=name)
    return SimpleNamespace(fn=fn, base_fn=fn, configs=list(configs), cache={("earlier", "key"): configs[-1]})


def test_a_named_kernel_keeps_the_tables_configuration_and_forgets_earlier_choices():
    t = tuner("inverse_kernel", config(2, 2), config(4, 5), config(8, 3))
    pinned, fallback = pin_all([t], {"inverse_kernel": {"num_warps": 4, "num_stages": 5, "kwargs": {}}})
    assert [(c.num_warps, c.num_stages) for c in t.configs] == [(4, 5)]
    assert t.cache == {}
    assert "inverse_kernel" in pinned and not fallback


def test_a_kernel_the_table_does_not_name_keeps_its_first_candidate_and_is_reported():
    t = tuner("other_kernel", config(8, 4, BK=64), config(4, 2, BK=32))
    pinned, fallback = pin_all([t], {})
    assert [(c.num_warps, c.kwargs) for c in t.configs] == [(8, {"BK": 64})]
    assert "other_kernel" in fallback and not pinned


def test_a_table_entry_that_is_not_a_candidate_is_not_forced_on_the_kernel():
    t = tuner("inverse_kernel", config(2, 2), config(4, 5))
    _, fallback = pin_all([t], {"inverse_kernel": {"num_warps": 16, "num_stages": 1, "kwargs": {}}})
    assert [(c.num_warps, c.num_stages) for c in t.configs] == [(2, 2)]
    assert "not a candidate" in fallback["inverse_kernel"]


def test_kwargs_are_part_of_the_match():
    assert matches(config(4, 3, BV=64), {"num_warps": 4, "num_stages": 3, "kwargs": {"BV": 64}})
    assert not matches(config(4, 3, BV=32), {"num_warps": 4, "num_stages": 3, "kwargs": {"BV": 64}})


def test_a_generation_without_a_table_has_none(tmp_path):
    assert load_table("sm_1", tmp_path) == ({}, None)
    assert load_table(None) == ({}, None)


@pytest.mark.parametrize("path", sorted(PINNED_DIR.glob("sm_*.json")), ids=lambda p: p.stem)
def test_every_shipped_table_is_well_formed(path):
    data = json.loads(path.read_text())
    assert data["measured_on"]
    assert data["kernels"]
    for name, entry in data["kernels"].items():
        assert set(entry) == {"num_warps", "num_stages", "kwargs"}, name
        assert isinstance(entry["num_warps"], int) and isinstance(entry["num_stages"], int), name
