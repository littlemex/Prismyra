"""`kernels.onepass_moe_tuning` without a device: the shipped tables are well-formed, and loading one by name
behaves the way `scope()` and `engine.py`'s `require_kernels` check both depend on. No regression test named this
module before the RTX PRO 4500 file was added (`fix/sm120-dense-nvfp4-pins`) -- `--require-kernels` failing on that
card for lack of a file was only ever caught by hand, on real hardware, not by anything CI runs on every change.
"""

from __future__ import annotations

import json

import pytest

from prismyra.kernels import onepass_moe_tuning as m

#: Every one-pass tile this project currently ships -- expanding this set is the whole point of a new
#: `pinned/onepass_moe_configs/<device>.json` file, so a new one should show up here too.
SHIPPED_DEVICES = {"NVIDIA_L40S", "NVIDIA_RTX_PRO_4500_Blackwell_Server_Edition"}


def test_every_shipped_device_has_a_file():
    on_disk = {p.stem for p in m.PINNED_DIR.glob("*.json")}
    assert on_disk == SHIPPED_DEVICES, on_disk


@pytest.mark.parametrize("device", sorted(SHIPPED_DEVICES))
def test_every_shipped_table_is_one_well_formed_tile(device):
    data = json.loads((m.PINNED_DIR / f"{device}.json").read_text())
    expected_keys = {
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "SPLIT_K",
        "num_warps",
        "num_stages",
    }
    assert set(data) == expected_keys
    assert all(isinstance(v, int) for v in data.values()), data


def test_the_two_shipped_tiles_agree():
    """Both cards this project ships `fast` kernels for are plain CUDA (no ROCm branch, no shared-memory limit
    this project's own `get_default_config` reading -- `kernels/onepass_moe_tuning.py`'s own module docstring --
    found reason to expect a difference by card for this one tile): measured independently on each card
    (`tools/pin_onepass_moe.py`), not assumed to transfer, they came back identical. A future card's own
    measurement might not agree; this records that today's two do, so a change to either file is a deliberate
    edit, not a silent drift caught nowhere."""
    files = {p.stem: json.loads(p.read_text()) for p in m.PINNED_DIR.glob("*.json")}
    assert len(files) >= 2
    values = list(files.values())
    assert all(v == values[0] for v in values[1:]), files


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


def test_scope_is_a_no_op_when_nothing_is_pinned(monkeypatch):
    monkeypatch.setattr(m, "_PINNED", None)  # "checked, nothing for this card"
    with m.scope():
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
    monkeypatch.setattr(m, "_PINNED", {"BLOCK_SIZE_M": 64})

    with pytest.raises(ZeroDivisionError), m.scope():
        assert fake_moe._config == {"BLOCK_SIZE_M": 64}
        raise ZeroDivisionError

    assert fake_moe._config is None


def test_the_escape_hatch_disables_scope_without_touching_what_is_pinned(monkeypatch):
    monkeypatch.setattr(m, "_PINNED", {"BLOCK_SIZE_M": 64})
    monkeypatch.setenv("PRISMYRA_ONEPASS_MOE_PIN", "0")
    with m.scope():
        pass  # the escape hatch takes effect without a device or vllm installed, by not importing fused_moe at all
    assert m.pinned_config() == {"BLOCK_SIZE_M": 64}, "disabling scope() must not have cleared the cached table"
