"""`kernels.onepass_moe_tuning` without a device: the shipped tables are well-formed, and loading one by name
behaves the way `scope()` and `engine.py`'s `require_kernels` check both depend on. No regression test named this
module before the RTX PRO 4500 file was added (`fix/sm120-dense-nvfp4-pins`) -- `--require-kernels` failing on that
card for lack of a file was only ever caught by hand, on real hardware, not by anything CI runs on every change.

`speed-tiles` (`RUN-speed-tiles.md`) changed the file format from one flat tile to `{"<m bucket>": tile, ...}`:
a per-M table, measured by `tools/speedtiles_build.py` with `BLOCK_SIZE_K`/`SPLIT_K` held fixed (checked
`torch.equal` against the old flat tile on every candidate, at every bucket, on real hardware -- see that
script's own module docstring) so the bucket chosen never moves the answer, only the time. `tile_for()`'s own
nearest-bucket lookup, not `pinned_config()`'s raw-table read, is what `scope()` now calls; these tests cover
both.
"""

from __future__ import annotations

import json

import pytest

from prismyra.kernels import onepass_moe_tuning as m

#: Every one-pass tile this project currently ships -- expanding this set is the whole point of a new
#: `pinned/onepass_moe_configs/<device>.json` file, so a new one should show up here too.
SHIPPED_DEVICES = {"NVIDIA_L40S", "NVIDIA_RTX_PRO_4500_Blackwell_Server_Edition"}

_TILE_KEYS = {
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "SPLIT_K",
    "num_warps",
    "num_stages",
}


def test_every_shipped_device_has_a_file():
    on_disk = {p.stem for p in m.PINNED_DIR.glob("*.json")}
    assert on_disk == SHIPPED_DEVICES, on_disk


@pytest.mark.parametrize("device", sorted(SHIPPED_DEVICES))
def test_every_shipped_table_is_a_bucketed_table_of_well_formed_tiles(device):
    data = json.loads((m.PINNED_DIR / f"{device}.json").read_text())
    assert not m.is_flat_tile(data), (
        "every shipped table is bucketed as of speed-tiles; a flat tile is still a "
        "valid *format* (tile_for() reads one) but this project no longer ships one"
    )
    assert data, "an empty table is indistinguishable from no table at all"
    for bucket_key, tile in data.items():
        assert bucket_key.isdigit(), f"bucket key {bucket_key!r} must be a plain row count"
        assert set(tile) == _TILE_KEYS, (bucket_key, tile)
        assert all(isinstance(v, int) for v in tile.values()), (bucket_key, tile)
        # The one invariant this whole mechanism exists to protect: a bucketed table must never vary the
        # K-reduction order, only how the output tile is chosen and scheduled.
        assert tile["BLOCK_SIZE_K"] == 128, (bucket_key, tile, "BLOCK_SIZE_K must stay fixed across every bucket")
        assert tile["SPLIT_K"] == 1, (bucket_key, tile, "SPLIT_K must stay 1 across every bucket")


def test_every_shipped_table_covers_the_same_buckets():
    """Not required to *agree* on the chosen tile per bucket (unlike the pre-`speed-tiles` single tile, a
    per-M table is genuinely allowed to differ by card -- a real timing search, not vLLM's own row-count-only
    heuristic, decided each bucket), but every card this project ships one for should have measured the same
    M ladder, so a future card missing a bucket the others have is a measurement gap, not a design choice."""
    files = {p.stem: set(json.loads(p.read_text())) for p in m.PINNED_DIR.glob("*.json")}
    assert len(files) >= 2
    keysets = list(files.values())
    assert all(k == keysets[0] for k in keysets[1:]), files


def test_load_rejects_a_name_with_path_characters(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PINNED_DIR", tmp_path)
    assert m._load("../etc/passwd") is None


def test_load_returns_none_for_a_card_with_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PINNED_DIR", tmp_path)
    assert m._load("NVIDIA_SOME_FUTURE_CARD") is None


def test_pinned_config_is_read_once_and_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(m, "_PINNED", False)
    monkeypatch.setattr(m, "device_name", lambda: "NVIDIA_L40S")

    real_load = m._load

    def counting_load(name):
        calls.append(name)
        return real_load(name)

    monkeypatch.setattr(m, "_load", counting_load)
    first = m.pinned_config()
    second = m.pinned_config()
    assert first == second == real_load("NVIDIA_L40S")
    assert calls == ["NVIDIA_L40S"], "pinned_config must read the file at most once per process"


#: A minimal, realistic one-bucket table for the tests below -- `tile_for()`'s nearest-bucket lookup over a
#: table with a single key always returns this one tile, regardless of the `m` scope()/tile_for() is asked
#: about, so these tests can exercise the override/restore machinery without needing a second bucket.
_ONE_BUCKET_TABLE = {
    "64": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "SPLIT_K": 1,
        "num_warps": 4,
        "num_stages": 3,
    }
}


def test_tile_for_picks_the_nearest_bucket():
    table = {"64": {"BLOCK_SIZE_M": 64}, "4096": {"BLOCK_SIZE_M": 128}, "16384": {"BLOCK_SIZE_M": 256}}
    assert m._nearest_bucket(table, 1) == "64"
    assert m._nearest_bucket(table, 100) == "64"
    assert m._nearest_bucket(table, 2048) == "64"  # |2048-64|=1984 < |2048-4096|=2048
    assert m._nearest_bucket(table, 5000) == "4096"
    assert m._nearest_bucket(table, 20000) == "16384"
    assert m._nearest_bucket({}, 100) is None


def test_tile_for_reads_a_flat_legacy_table_for_every_m(monkeypatch):
    flat = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "SPLIT_K": 1,
        "num_warps": 4,
        "num_stages": 3,
    }
    monkeypatch.setattr(m, "_PINNED", flat)
    assert m.tile_for(1) == flat
    assert m.tile_for(16384) == flat


def test_scope_is_a_no_op_when_nothing_is_pinned(monkeypatch):
    monkeypatch.setattr(m, "_PINNED", None)  # "checked, nothing for this card"
    with m.scope(3446):
        pass  # must not raise even though vllm.model_executor.layers.fused_moe is not importable here


def test_scope_restores_the_override_even_if_the_block_raises(monkeypatch):
    import sys
    import types

    # `import vllm.model_executor.layers.fused_moe` resolves every dotted segment through `sys.modules`, not
    # only the leaf: on a machine with no real `vllm` installed (this project's own CPU test job, which
    # installs `[dev]` but not `[fast]`), faking only the leaf still tries to import the real, absent `vllm`
    # package to resolve its parent and fails with `ModuleNotFoundError` before `scope()`'s own code runs at
    # all -- every ancestor package needs its own fake entry too.
    fake_moe = types.SimpleNamespace(_config=None)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.model_executor", types.ModuleType("vllm.model_executor"))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers"))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe", fake_moe)
    monkeypatch.setattr(m, "_PINNED", _ONE_BUCKET_TABLE)

    with pytest.raises(ZeroDivisionError), m.scope(3446):
        assert fake_moe._config == _ONE_BUCKET_TABLE["64"]
        raise ZeroDivisionError

    assert fake_moe._config is None


def test_the_escape_hatch_disables_scope_without_touching_what_is_pinned(monkeypatch):
    monkeypatch.setattr(m, "_PINNED", _ONE_BUCKET_TABLE)
    monkeypatch.setenv("PRISMYRA_ONEPASS_MOE_PIN", "0")
    with m.scope(3446):
        pass  # the escape hatch takes effect without a device or vllm installed, by not importing fused_moe at all
    assert m.pinned_config() == _ONE_BUCKET_TABLE, "disabling scope() must not have cleared the cached table"
