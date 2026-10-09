"""`_compare` and `_swap_gated_norm` decide whether to install a kernel replacement by measuring how far it
disagrees with the implementation it replaces on one random probe, against a tolerance (5e-2 for "dense_matmul"
and "norm", `2 * BF16_ULP` for "gated_norm"). Both probes are seeded (`torch.Generator(device="cpu").manual_seed(...)`,
keyed by the module's own width), not drawn from the process-global RNG state, so the same checkpoint measures the
same disagreement -- and makes the same install/decline decision -- every time, in every process.

Measured directly on `nvfp4-36l`'s own weights before this fix: four unseeded draws of "dense_matmul"'s probe (then
4 rows, now 64) gave 2.912e-02, 3.893e-02, 3.968e-02, and 5.018e-02 against the 5e-2 tolerance -- the fourth is over
it. A real `nn.Module` comparison needs a device and weights this file does not have; what it can check without
either is the one property the fix adds: calling the same comparison twice, in the same process, with nothing in
between to reseed anything by hand, gives the identical probe and therefore the identical measurement -- which an
unseeded `torch.randn()` (drawing from the process-global generator, which `torch.manual_seed()` only resets once,
not before every call) would not have, since the global generator's state moves on between the two calls.
"""

from __future__ import annotations

import torch
from torch import nn

from prismyra.kernels.qwen3_moe import _compare, _swap_and_verify, _warn_if_close_to_tolerance
from prismyra.kernels import Applied


def test_compare_measures_the_same_disagreement_every_time():
    """Two calls, same two modules, nothing reseeded by hand in between: the measured disagreement is identical."""
    torch.manual_seed(0)
    width = 2048
    original = nn.Linear(width, width, bias=False, dtype=torch.bfloat16)
    # A replacement that is not quite the same as the original -- the comparison has something real to measure,
    # not just floating-point noise around zero either way.
    replacement = nn.Linear(width, width, bias=False, dtype=torch.bfloat16)
    replacement.weight.data = original.weight.data + 1e-3

    first = _compare(original, replacement)
    # Advance the process-global RNG a lot, the way unrelated code elsewhere in a real construction would, before
    # this measurement is repeated -- an unseeded probe would answer differently afterward; a seeded one will not.
    torch.randn(10_000, 10_000)
    second = _compare(original, replacement)

    assert first == second


def test_compare_seeds_by_width_not_by_process_global_state():
    """The probe itself (not just the measurement two calls on the identical modules happen to agree on) is the
    same across what two independent process launches of the same checkpoint would see: a width-2048 module probed
    after the global generator has been seeded differently gets the identical probe either way."""
    width = 2048

    def probe_for(seed_before: int) -> torch.Tensor:
        torch.manual_seed(seed_before)  # stands in for "whatever a different process happened to do first"
        captured = {}
        original = nn.functional.linear

        def spy(x, *a, **k):
            captured["x"] = x
            return original(x, *a, **k)

        linear = nn.Linear(width, width, bias=False, dtype=torch.bfloat16)
        nn.functional.linear = spy
        try:
            _compare(linear, linear)
        finally:
            nn.functional.linear = original
        return captured["x"]

    assert torch.equal(probe_for(seed_before=1), probe_for(seed_before=99999))


def test_identical_modules_measure_zero_regardless_of_the_seeded_probe():
    width = 128
    original = nn.Linear(width, width, bias=True, dtype=torch.bfloat16)
    assert _compare(original, original) == 0.0


def test_a_measurement_that_uses_most_of_its_tolerance_is_noted_not_silently_installed():
    """`_warn_if_close_to_tolerance`: a passing swap that measured more than half its tolerance's budget leaves a
    note (`engine.applied.notes`), not just a swap entry -- `MARGIN_WARN_FRACTION`'s own module docstring has the
    measured incident (`nvfp4-36l`'s "dense_matmul" at 2.820e-02 against a 5.0e-02 tolerance, 56% of the budget)."""
    applied = Applied(adapter="test")
    _warn_if_close_to_tolerance(applied, "dense_matmul", 0.028, 0.05)
    assert any("dense_matmul" in n and "56%" in n for n in applied.notes), applied.notes


def test_a_comfortable_measurement_is_not_noted():
    applied = Applied(adapter="test")
    _warn_if_close_to_tolerance(applied, "dense_matmul", 0.006, 0.05)
    assert applied.notes == []


def test_swap_and_verify_declines_and_records_every_tried_candidate_when_none_agree():
    """Unit-level companion to the GPU self-check this module runs at `kernels.apply()` time: a candidate that
    disagrees by more than its tolerance is declined and the reason -- which candidates were tried and by how much
    each missed -- is recorded, not silently dropped."""
    applied = Applied(adapter="test")
    width = 64

    class WrongScale(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.weight = inner.weight

        def forward(self, x):
            return nn.functional.linear(x, self.weight) * 10.0  # off by a factor, not a rounding step

    root = nn.Module()
    root.target = nn.Linear(width, width, bias=False, dtype=torch.bfloat16)  # `_find_children` looks at children

    _swap_and_verify(applied, root, "fake_swap", "Linear", [("wrong_scale", WrongScale)], tolerance=5e-2)
    assert applied.swaps == []
    (message,) = applied.skipped
    assert "fake_swap left alone" in message and "wrong_scale" in message
