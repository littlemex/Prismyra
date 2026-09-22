"""What admission promises, checked without a device.

These are arithmetic and bookkeeping tests. Whether a pass actually fits is a question only an allocator can answer,
and `Prismyra` catches that separately; what is checked here is that the engine budgets the answering transient at all,
since budgeting only the held cache is what let a context be admitted and then run out of memory on its first question.
"""

from __future__ import annotations

import pytest
import torch

from prismyra.cache import join_bytes_per_token
from prismyra.engine import ANSWERING_MARGIN, WIDTHS, Prismyra, row_constant


class Config:
    """The two shapes the arithmetic part of the budget reads, as the supported model declares them."""

    num_key_value_heads = 2
    head_dim = 256
    hidden_size = 2048
    num_attention_heads = 16


class Fake:
    """A bare engine with only the fields these methods read. Constructing a real one loads 35 GiB of weights."""

    group = 32
    dtype = torch.bfloat16
    config = Config()
    _observed_row_constant: int | None = None
    _observed_at_rows = 0
    #: The borrowed kernel reads the join in the layout it is stored in; the framework's fallback is handed a strided
    #: view and is budgeted at twice the slope because nothing has measured whether it copies.
    _borrowed_kernel = True

    answering_bytes = Prismyra.answering_bytes
    budget_is_evidenced = Prismyra.budget_is_evidenced
    _observe_peak = Prismyra._observe_peak


PER_TOKEN = join_bytes_per_token(Config(), torch.bfloat16)


def test_nothing_is_budgeted_before_a_pass_has_been_observed():
    """Zero rather than a guess. The constant part is one model's activations and kernel workspace, and the refusal
    message says outright that the first pass is unbudgeted."""
    assert Fake().answering_bytes(10_000) == 0


def test_the_slope_is_arithmetic_and_halved_by_token_major_storage():
    """2,048 bytes per context token per row from these shapes: keys and values, and two layers' joins overlapping.

    It was 4,096 while the join was head-major and the kernel's `transpose(1, 2).reshape(...)` had to copy the whole
    thing a second time to reach the layout it reads. The measured slope was 3,838 against that 4,096, so the
    arithmetic over-stated by 7%; with the copy gone the prediction halves and `prismyra-bench widths` is what says
    whether the measurement followed it.
    """
    assert PER_TOKEN == 2_048


def test_the_budget_grows_with_the_rows_a_request_will_use():
    engine = Fake()
    engine._observed_row_constant = 1_000
    per_row = 1_000 + PER_TOKEN * (1_000 + WIDTHS[-1])
    assert engine.answering_bytes(1_000, questions=1) == int(per_row * ANSWERING_MARGIN)
    assert engine.answering_bytes(1_000, questions=4) == int(4 * per_row * ANSWERING_MARGIN)
    # Capped by the group: a request with more questions than the group is answered in several passes, and what has to
    # fit is one pass.
    assert engine.answering_bytes(1_000, questions=1_000) == int(32 * per_row * ANSWERING_MARGIN)


def test_a_longer_context_is_budgeted_higher_by_the_arithmetic_slope():
    engine = Fake()
    engine._observed_row_constant = 1_000
    short = engine.answering_bytes(3_000, questions=1)
    longer = engine.answering_bytes(30_000, questions=1)
    assert longer - short == pytest.approx(27_000 * PER_TOKEN * ANSWERING_MARGIN, rel=1e-6)


def test_nothing_is_observed_without_a_device():
    engine = Fake()
    engine._observe_peak(None, 5_000, 4)
    assert engine._observed_row_constant is None


def test_the_observed_constant_is_the_peak_with_the_scaling_part_taken_out():
    assert row_constant(None, 1_000 + 4_096 * 10, 10, 4_096) == 1_000


def test_the_constant_ratchets_upwards_and_is_comparable_across_context_lengths():
    """The whole reason for subtracting the slope. Two passes at very different lengths yield the same constant, so an
    observation at either is usable for the other -- which is what stops the estimate drifting towards refusal.

    Keeping the largest observed *prediction* instead would ratchet the other way: a prediction scaled up from a short
    context is larger per token than one from a long context, so the shortest context ever seen would win and be kept,
    and a full-width pass at 24,327 tokens would be budgeted at 17.7 GiB when it needs 4.97.
    """
    short = row_constant(None, 1_000 + 4_096 * 3_552, 3_552, 4_096)
    both = row_constant(short, 1_000 + 4_096 * 24_839, 24_839, 4_096)
    assert short == both == 1_000
    assert row_constant(both, 500, 10, 4_096) == 1_000


def test_a_peak_below_the_arithmetic_slope_floors_at_zero():
    """A measured peak smaller than the slope alone means the layers never overlapped the way the slope assumes. A
    negative constant would budget less than the arithmetic, which is the one part that is not in doubt."""
    assert row_constant(None, 100, 10, 4_096) == 0


def test_a_pass_wider_than_any_measured_is_not_claimed_to_be_budgeted():
    """The estimate exists for every width; the claim that it is evidence does not.

    Per-row cost used not to be flat in width -- 0.111 GiB per row at width 1 against 0.155 at widths 8 and 32 -- so an
    observation at width 1 under-stated a width-32 pass by 29%. Storing the join in the layout the kernel reads removed
    the second copy that caused it, and the two now agree to three decimals. This stays true anyway: one measurement on
    one model is not a reason to promise the shape holds, and refusing on a figure nothing supports is the failure this
    guards against.
    """
    engine = Fake()
    engine._observed_row_constant = 1_000
    engine._observed_at_rows = 8
    assert engine.budget_is_evidenced(questions=4)
    assert engine.budget_is_evidenced(questions=8)
    assert not engine.budget_is_evidenced(questions=9)
    assert not engine.budget_is_evidenced()
    # The estimate is still produced, because one that already exceeds free memory is a refusal either way.
    assert engine.answering_bytes(1_000, questions=32) > 0


def test_nothing_is_evidenced_before_the_first_pass():
    assert not Fake().budget_is_evidenced(questions=1)


def test_the_fallback_path_is_budgeted_at_twice_the_slope():
    """Not because it is known to copy, but because it is not known not to.

    The join is stored in the layout the borrowed kernel reads. The framework's own attention takes the same storage as
    a strided view, and whether it works on those strides or makes itself a contiguous copy is inside the framework. A
    path nobody has measured is budgeted at the larger figure.
    """
    assert join_bytes_per_token(Config(), torch.bfloat16, doubled=True) == 2 * PER_TOKEN

    fast, slow = Fake(), Fake()
    fast._observed_row_constant = slow._observed_row_constant = 0
    slow._borrowed_kernel = False
    assert slow.answering_bytes(10_000, questions=8) == 2 * fast.answering_bytes(10_000, questions=8)
