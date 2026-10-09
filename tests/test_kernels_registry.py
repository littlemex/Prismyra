"""The adapter contract: recognise a model or fail closed. No device needed."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from torch import nn

from prismyra import kernels
from prismyra.kernels import AdapterError, Applied, Swap


@dataclass
class FakeConfig:
    """Only the fields the adapters look at: the architecture names and the counts `supports` checks."""

    architectures: list[str]
    num_hidden_layers: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0


def test_an_unknown_architecture_runs_slowly_by_default():
    applied = kernels.apply(nn.Linear(2, 2), FakeConfig(architectures=["SomethingElse"]))
    assert applied.adapter == "none"
    assert applied.skipped


def test_an_unknown_architecture_can_be_made_fatal():
    with pytest.raises(AdapterError, match="no adapter"):
        kernels.apply(nn.Linear(2, 2), FakeConfig(architectures=["SomethingElse"]), required=True)


def test_a_wrong_count_fails_closed():
    """A swap that matches nothing looks exactly like a swap that worked, so the count is the guard."""

    class Miscounting:
        name = "miscounting"

        def supports(self, config):
            return True

        def replace(self, model, config):
            return Applied(adapter=self.name, swaps=[Swap("thing", replaced=3, expected=40)])

    kernels.register(Miscounting())
    try:
        with pytest.raises(AdapterError, match="replaced 3 of an expected 40"):
            kernels.apply(nn.Linear(2, 2), FakeConfig(architectures=["Whatever"]))
    finally:
        kernels._ADAPTERS.pop()


def test_the_qwen_adapter_claims_only_the_configuration_it_was_measured_on():
    """The family name is not enough. Another checkpoint in the same family has different per-layer shapes, and an
    adapter that claimed it would mutate the model and only then refuse it, leaving a half-replaced model behind."""
    from prismyra.kernels.qwen3_moe import MEASURED_CONFIG

    measured = FakeConfig(architectures=["Qwen3_5MoeForConditionalGeneration"], num_hidden_layers=40, **MEASURED_CONFIG)
    adapter = kernels.find(measured)
    assert adapter is not None and adapter.name == "qwen3-moe"

    assert kernels.find(FakeConfig(architectures=["LlamaForCausalLM"], num_hidden_layers=40, **MEASURED_CONFIG)) is None

    wrong_shape = dict(MEASURED_CONFIG, num_attention_heads=32)
    assert (
        kernels.find(
            FakeConfig(architectures=["Qwen3_5MoeForConditionalGeneration"], num_hidden_layers=40, **wrong_shape)
        )
        is None
    )


def test_a_truncated_or_extended_checkpoint_still_matches():
    """Depth is how many layers there are, not what one layer looks like. A checkpoint cut down to fewer layers, or
    grown to more, still gets the fused kernels as long as every per-layer shape matches what was measured -- this
    is what makes a truncated checkpoint serve at full speed with no override."""
    from prismyra.kernels.qwen3_moe import MEASURED_CONFIG

    for depth in (32, 36, 40, 48):
        truncated = FakeConfig(
            architectures=["Qwen3_5MoeForConditionalGeneration"], num_hidden_layers=depth, **MEASURED_CONFIG
        )
        adapter = kernels.find(truncated)
        assert adapter is not None and adapter.name == "qwen3-moe"


def test_applied_reports_what_it_did_without_carrying_timings():
    """Measured figures belong in docs; an object that carries them goes stale silently."""
    applied = Applied(adapter="x", swaps=[Swap("a", 2, 2)], skipped=["b"])
    assert applied.ok
    assert "a=2" in applied.summary() and "skipped=b" in applied.summary()
    assert not hasattr(applied, "ms")


def test_required_means_required_even_when_every_module_was_found():
    """A skipped kernel leaves `ok` true: nothing was missing, a kernel just will not run. Required must still fail."""
    from prismyra.kernels import apply, register

    class Partial:
        name = "partial"

        def supports(self, config):
            return True

        def replace(self, model, config):
            return Applied(adapter=self.name, swaps=[Swap("norm", 4, 4)], skipped=["triton is not available"])

    adapter = Partial()
    register(adapter)
    try:
        assert apply(nn.Identity(), FakeConfig(architectures=["Partial"]), required=False).skipped
        with pytest.raises(AdapterError, match="could not apply every kernel"):
            apply(nn.Identity(), FakeConfig(architectures=["Partial"]), required=True)
    finally:
        from prismyra.kernels import _ADAPTERS

        _ADAPTERS.remove(adapter)


def test_applied_reports_degradation_as_data_not_only_as_a_string():
    applied = Applied(adapter="a", swaps=[Swap("norm", 4, 4)], skipped=["no triton"], notes=["context pass only"])
    as_dict = applied.as_dict()
    assert as_dict["complete"] is False
    assert as_dict["applied"] == {"norm": 4}
    assert as_dict["skipped"] == ["no triton"]
    assert as_dict["notes"] == ["context pass only"]


def test_a_resource_exhausted_self_check_fails_construction_instead_of_silently_choosing_a_fallback():
    """A kernel-replacement self-check (`_compare`, and by the same helper `_swap_gated_norm`,
    `_delta_disagreement`, `_probe_head_duplication`) that runs out of a hardware resource -- GPU memory, or a
    Triton kernel's static shared-memory requirement for the shapes at hand -- while comparing a replacement
    against the original is not evidence the two disagree; it is evidence this process could not run the probe at
    all right now. Treating it the same as a numeric disagreement would let how much free GPU memory happens to
    exist at construction time choose which kernel path answers every request for the rest of the process's life,
    moving the answer the same way an unseeded random probe did before this file's own seeding fix -- a
    construction-time resource shortage standing in for a construction-time coin flip. Found on real hardware: a
    module-scoped engine elsewhere in the same pytest process still holding another checkpoint's weights made
    `_compare`'s 64-row probe raise `triton.errors.OutOfResources` on a shared-memory limit for one checkpoint's
    `dense_matmul` shapes, which the pre-fix code silently treated as "decline" and moved on. No device needed:
    this is checked with a module whose `forward` raises the exception directly, not an actual OOM."""
    import torch

    from prismyra.kernels import AdapterError
    from prismyra.kernels.qwen3_moe import _compare

    class RaisesOOM(nn.Module):
        in_features = 8

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            raise torch.OutOfMemoryError("simulated: out of memory")

    class MustNotRun(nn.Module):
        def forward(self, x):
            raise AssertionError("the replacement must never be called once the original already raised")

    with pytest.raises(AdapterError, match="ran out of GPU memory"):
        _compare(RaisesOOM(), MustNotRun(), "dense_matmul")


def test_a_triton_resource_limit_in_a_self_check_also_fails_construction():
    """The exact exception seen on real hardware (`triton.errors.OutOfResources`, a `TritonError`) must take the
    same path as `torch.OutOfMemoryError`: raise, not decline. `pytest.importorskip` rather than `pytest.mark.gpu`
    because the exception class itself needs no device, only the triton package."""
    triton_runtime_errors = pytest.importorskip("triton.runtime.errors")

    from prismyra.kernels import AdapterError
    from prismyra.kernels.qwen3_moe import _compare

    class RaisesTritonResourceError(nn.Module):
        in_features = 8

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(__import__("torch").zeros(1))

        def forward(self, x):
            raise triton_runtime_errors.OutOfResources(106496, 101376, "shared memory")

    class MustNotRun(nn.Module):
        def forward(self, x):
            raise AssertionError("the replacement must never be called once the original already raised")

    with pytest.raises(AdapterError, match="Triton compiler/runtime resource limit"):
        _compare(RaisesTritonResourceError(), MustNotRun(), "dense_matmul")


def test_a_plain_structural_mismatch_still_declines_quietly():
    """The opposite case must still work exactly as before this fix: a replacement that is simply the wrong shape
    or structure for this model's weights is a reason to decline the swap, not to fail construction -- only a
    resource-exhaustion exception should raise."""
    import torch

    from prismyra.kernels.qwen3_moe import _compare

    class RaisesShapeError(nn.Module):
        in_features = 8

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            raise RuntimeError("simulated: shape mismatch, nothing to do with a resource limit")

    assert _compare(RaisesShapeError(), RaisesShapeError(), "dense_matmul") is None
