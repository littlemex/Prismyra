"""The invariants that need a device. Marked `gpu`, so CI skips them and the machine that measures runs them.

These are the properties the whole design rests on, and each one was a real bug at some point:

* a row's answer must not depend on the other rows it travelled with, or batching would change answers;
* a second group of questions must start from the context, not from the first group's advanced recurrence -- the failure
  mode is an answer that is plausible and wrong;
* the convolution must not read across a sequence boundary;
* skipping the head duplication must be bit-identical, since it is a flag set through an attribute whose name no longer
  describes its value.

Run with:

    pytest -m gpu                                    # needs the weights
    PRISMYRA_TEST_MODEL=<repo> pytest -m gpu         # against another checkpoint
"""

from __future__ import annotations

import os

import pytest
import torch

from prismyra import Boolean, Choice, Prismyra, PrismyraError

pytestmark = pytest.mark.gpu

MODEL = os.environ.get("PRISMYRA_TEST_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")

CONTEXT = (
    "Returns are accepted within thirty days of delivery. Unopened items are refunded in full. Opened items are "
    "exchanged rather than refunded, unless a manufacturing fault is confirmed. Return shipping is paid by the seller "
    "when the item is faulty and by the buyer otherwise."
)


@pytest.fixture(scope="module")
def engine():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return Prismyra(MODEL)


def questions(n: int) -> list[Boolean]:
    return [Boolean(id=f"q{i}", prompt=f"Is clause {i} about shipping?") for i in range(n)]


def test_an_answer_does_not_depend_on_its_companions(engine):
    """Fork isolation. Without it every measured number would be meaningless, because batching would change answers."""
    alone = engine.ask(CONTEXT, [Boolean(id="faulty", prompt="Does the seller pay when the item is faulty?")])
    crowded = engine.ask(
        CONTEXT,
        [Boolean(id="faulty", prompt="Does the seller pay when the item is faulty?"), *questions(31)],
    )
    assert alone["faulty"].option == crowded["faulty"].option
    for option, p in alone["faulty"].probabilities.items():
        assert abs(p - crowded["faulty"].probabilities[option]) < 1e-3


def test_the_order_questions_arrive_in_does_not_change_them(engine):
    asked = [
        Boolean(id="unopened", prompt="Are unopened items refunded in full?"),
        Choice(id="who_pays", prompt="Who pays return shipping on a faulty item?", choices=["seller", "buyer"]),
        Boolean(id="thirty", prompt="Is there a thirty day limit?"),
    ]
    forwards = engine.ask(CONTEXT, asked)
    backwards = engine.ask(CONTEXT, list(reversed(asked)))
    for q in asked:
        assert forwards[q.id].option == backwards[q.id].option


def test_a_second_group_starts_from_the_context_and_not_from_the_first(engine):
    """More questions than one group, so the second group forks from the snapshot rather than from advanced state.

    The bug this catches does not raise: the recurrent layers write their advanced state back regardless, so a second
    group reading it answers plausibly and wrongly.
    """
    one = Boolean(id="probe", prompt="Are unopened items refunded in full?")
    first_group = engine.ask(CONTEXT, [one, *questions(31)])
    two_groups = engine.ask(CONTEXT, [*questions(32), one])
    assert first_group["probe"].option == two_groups["probe"].option
    for option, p in first_group["probe"].probabilities.items():
        assert abs(p - two_groups["probe"].probabilities[option]) < 1e-3


def test_padding_a_batch_does_not_reach_an_answer(engine):
    """Three questions in a batch pinned to thirty-two rows must answer as three questions in three rows.

    Two engines rather than one call with a different group: the group is fixed at construction because the cache is
    preallocated for it, so this is the only way to compare the two widths.
    """
    asked = questions(3)
    padded = engine.ask(CONTEXT, asked)
    exact = Prismyra(MODEL, group=3).ask(CONTEXT, asked)
    for q in asked:
        assert padded[q.id].option == exact[q.id].option


def test_an_open_context_answers_as_a_fresh_one(engine):
    """The saving is only legitimate if reading once and reusing gives what re-reading gives."""
    asked = [Boolean(id="thirty", prompt="Is there a thirty day limit?")]
    reused = engine.open_context(CONTEXT)
    assert reused.ask(asked)["thirty"].option == engine.ask(CONTEXT, asked)["thirty"].option
    assert reused.ask(asked)["thirty"].option == reused.ask(asked)["thirty"].option


def test_a_question_wider_than_the_widest_branch_is_refused_before_the_device(engine):
    """It used to be accommodated, which wrote past the end of the cache -- asynchronously, so the failure moved."""
    huge = Boolean(id="huge", prompt="word " * 4_000 + "?")
    with pytest.raises(PrismyraError, match="widest branch"):
        engine.ask(CONTEXT, [huge])


def test_the_counted_kernels_match_what_the_config_implies(engine):
    """A swap that matches nothing looks exactly like a swap that worked, so what can be counted is asserted.

    Only the counts that follow from the config. The normalisation and the dense projections are found by structure, so
    they carry a verification instead of a number, which this checks is present.
    """
    from prismyra.kernels.qwen3_moe import expected_counts

    if engine.applied.adapter != "qwen3-moe":
        pytest.skip(f"no qwen adapter for {MODEL}")
    assert engine.applied.ok, engine.applied.as_dict()

    decoder = getattr(engine.config, "text_config", engine.config)
    want = expected_counts(decoder)
    got = {s.name: s.replaced for s in engine.applied.swaps if s.expected is not None}
    assert got == {k: v for k, v in want.items() if k in got}

    verified = {s.name for s in engine.applied.swaps if s.verified}
    assert verified == {"norm", "dense_matmul"}, engine.applied.as_dict()


# --------------------------------------------------------------------------- the convolution on its own
def reference_conv(x: torch.Tensor, weight: torch.Tensor, starts: torch.Tensor | None) -> torch.Tensor:
    """The same thing written plainly, in float32, to compare against."""
    tokens = x.shape[0]
    width = weight.shape[1]
    out = torch.zeros_like(x, dtype=torch.float32)
    xf, wf = x.float(), weight.float()
    for t in range(tokens):
        begin = int(starts[t]) if starts is not None else 0
        for j in range(width):
            src = t - (width - 1 - j)
            if src >= begin:
                out[t] += xf[src] * wf[:, j]
    return torch.nn.functional.silu(out)


@pytest.mark.parametrize("packed", [False, True])
def test_the_convolution_matches_a_plain_one_and_respects_boundaries(packed):
    triton = pytest.importorskip("triton")
    assert triton is not None
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from prismyra.kernels.conv import causal_depthwise_conv1d, starts_from_boundaries

    torch.manual_seed(0)
    lengths = [37, 11, 64] if packed else [112]
    total = sum(lengths)
    x = (torch.randn(total, 256, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    weight = (torch.randn(256, 4, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()

    starts = None
    if packed:
        boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
        starts = starts_from_boundaries(boundaries, total)

    got = causal_depthwise_conv1d(x, weight, seq_starts=starts, activation="silu")
    want = reference_conv(x, weight, starts)
    assert (got.float() - want).abs().max().item() < 2e-2
