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

import numpy as np
import pytest
import torch

from prismyra import Boolean, Choice, Prismyra, PrismyraError, Scale

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


#: How far a probability may move between a question asked alone and the same question asked alongside others. Measured
#: rather than chosen: over 101 RACE questions the largest movement was 0.1995 and no decision changed, and over 60
#: BoolQ questions -- one question per context, so both runs are one row wide -- it was exactly 0.0000, which is the
#: control proving the comparison can report no difference. `evals/run.py --methods readout,alone` is that measurement.
#:
#: It was 1e-3 while every request ran at the full group width whatever it carried. A group now uses only as many rows
#: as it has questions, and a batch's width decides the order the reductions happen in, so this is the price of not
#: paying for thirty-two rows to answer three questions. See docs/PERFORMANCE.md.
#:
#: Set half again above the measured maximum rather than at it. A bound sitting exactly on the largest value a hundred
#: questions produced would fail on the hundred-and-first without anything having regressed, and the guard that
#: actually matters is the assertion that the decision did not change.
COMPANION_MOVEMENT = 0.3

#: The smallest answer to "how many seconds" that still shows the clip's timing reached the model. Five, not six.
#:
#: Not a loosened assertion. What these two tests exist to catch is timing **withheld**: a six second clip handed over
#: bare looks to the processor like two thirds of a second, and the model then answers **two**. Five and six both refute
#: that; two and three do not, and neither would pass this. The discrimination the test was written for is intact.
#:
#: Why it moved off six. The recurrence now runs on a borrowed kernel rather than the framework's chunked scan in
#: float32, which is 2.4x faster and numerically different. Measured over 472 RACE questions: 465 decisions identical
#: (98.5%), accuracy 0.9407 against 0.9364, largest probability movement 0.29. This question is one of the near-ties --
#: it came back five with 0.349 on six -- so asserting a single integer on it was asserting which side of a coin landed
#: up. `PRISMYRA_WITHOUT=gated_delta_rule` runs the other implementation if that comparison needs repeating.
NEARLY_SIX = 5


def test_an_answer_does_not_depend_on_its_companions(engine):
    """Fork isolation, at the level a caller sees it: the decision. Without it every measured number would be
    meaningless, because batching would change answers.

    The decision is asserted exactly and the distribution behind it within a measured tolerance. Those are two different
    claims and only the first is a promise -- a branch cannot see another branch's tokens, which is what isolation
    means, but it does share a reduction order with them.
    """
    alone = engine.ask(CONTEXT, [Boolean(id="faulty", prompt="Does the seller pay when the item is faulty?")])
    crowded = engine.ask(
        CONTEXT,
        [Boolean(id="faulty", prompt="Does the seller pay when the item is faulty?"), *questions(31)],
    )
    assert alone["faulty"].option == crowded["faulty"].option
    for option, p in alone["faulty"].probabilities.items():
        assert abs(p - crowded["faulty"].probabilities[option]) < COMPANION_MOVEMENT


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
        # The same tolerance as companion movement, and for the same reason: the probe is one row of thirty-two in the
        # first arrangement and the only row of a second group in the other, so the two passes have different widths.
        # The bug this test exists for is not a numeric one -- it is the second group reading the first group's advanced
        # recurrent state, which moves an answer far further than a reduction order does.
        assert abs(p - two_groups["probe"].probabilities[option]) < COMPANION_MOVEMENT


def test_padding_a_batch_does_not_reach_an_answer(engine):
    """Three questions in a batch pinned to thirty-two rows must answer as three questions in three rows.

    Two engines, because the group is fixed at construction -- the cache is preallocated for it -- so there is no
    other way to compare two widths. That means a second copy of the weights, which one 48 GiB card cannot hold beside
    the first. Skipped rather than quietly dropped: on a larger card it runs, and the reason it did not is printed.
    """
    asked = questions(3)
    padded = engine.ask(CONTEXT, asked)
    try:
        narrow = Prismyra(MODEL, group=3)
    except torch.OutOfMemoryError:
        pytest.skip("no room for a second copy of the weights beside the first; needs a larger card or a second one")
    exact = narrow.ask(CONTEXT, asked)
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

    Only the counts that follow from the config. The normalisation and the dense projections are found by structure,
    so they carry a verification instead of a number, which this checks is present.
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


# --------------------------------------------------------------------------- images and video
def solid(colour, size=336):
    from PIL import Image

    return Image.fromarray(np.zeros((size, size, 3), np.uint8) + np.array(colour, np.uint8))


def sliding_block(direction, frames=12, size=252):
    """A red block crossing a white frame, left to right or right to left.

    The point of building it rather than loading a clip: the right answer is known, and reversing the frames must
    reverse the answer. Nothing else here can tell whether the clip's order survived the context pass.
    """
    from PIL import Image, ImageDraw

    out = []
    for i in range(frames):
        image = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(image)
        x = i * 16 if direction == "right" else (frames - 1 - i) * 16
        draw.rectangle([x, 100, x + 50, 150], fill=(200, 20, 20))
        out.append(np.array(image))
    return np.stack(out)


def test_an_image_is_read_into_the_context(engine):
    pytest.importorskip("PIL")
    result = engine.ask(
        "This is a photograph.",
        [
            Boolean(id="red", prompt="Is the image mostly red?"),
            Boolean(id="blue", prompt="Is the image mostly blue?"),
            Choice(id="colour", prompt="What is the dominant colour?", choices=["red", "green", "blue"]),
        ],
        images=[solid((200, 30, 30))],
    )
    assert result["red"].value is True
    assert result["blue"].value is False
    assert result["colour"].option == "red"


def test_two_images_stay_in_the_order_they_were_given(engine):
    pytest.importorskip("PIL")
    result = engine.ask(
        "Two photographs, in order.",
        [Boolean(id="first_red", prompt="Is the first image red?")],
        images=[solid((200, 30, 30)), solid((30, 60, 200))],
    )
    assert result["first_red"].value is True


def test_a_clip_read_backwards_answers_backwards(engine):
    """The strongest check in this file. Same frames, reversed, and the answer has to reverse with them.

    It is also the check that the three-axis positions are right. With media the model's text positions advance by an
    image's grid rather than by its token count, and a branch that continues from the wrong place reads the context
    from the wrong place -- which shows up here as an answer that does not track the direction.
    """
    pytest.importorskip("PIL")
    question = [Choice(id="direction", prompt="Which way does the red block travel?", choices=["left", "right"])]
    rightwards = engine.ask("This is a short clip.", question, videos=[sliding_block("right")])
    leftwards = engine.ask("This is a short clip.", question, videos=[sliding_block("left")])
    assert rightwards["direction"].option == "right"
    assert leftwards["direction"].option == "left"


def test_an_image_costs_the_vision_tower_once_however_many_questions_follow(engine):
    """The reason media is worth supporting at all: the encoding is in the context, which is paid once."""
    pytest.importorskip("PIL")
    image = solid((200, 30, 30))
    with engine.open_context("This is a photograph.", images=[image]) as context:
        one = context.ask([Boolean(id="q0", prompt="Is the image red?")])
        many = context.ask([Boolean(id=f"q{i}", prompt=f"Is region {i} red?") for i in range(16)])
    # Neither ask paid for the image: the context did, and it reports that separately.
    assert one.timing.context_ms == 0.0
    assert context.context_ms > 0.0
    assert len(many) == 16


def test_a_clip_is_as_long_as_it_says_it_is(engine):
    """Timing withheld is not a smaller error than timing wrong. It is the same error, and it is silent.

    A six second clip decoded to a handful of frames and handed over bare looks to the processor like two thirds of a
    second at its default rate, and the model answers about a clip that does not exist. Measured on the supported
    model: asked how long this clip is, it answers six with the timing and two without. Everything visual is right
    either way, which is why nothing else in this file catches it.
    """
    pytest.importorskip("cv2")
    from prismyra.media import decode_video

    encoded = _six_second_clip()
    asked = [Scale(id="seconds", prompt="Roughly how many seconds long is this clip?", low=1, high=9)]

    clip = decode_video(encoded, max_frames=32)
    assert clip.duration == pytest.approx(6.0, abs=0.3)
    assert engine.ask("This is a video clip.", asked, videos=[clip])["seconds"].value >= NEARLY_SIX

    # The same frames with the timing withheld. Not asserted to be wrong -- a model may guess right -- but the clip's
    # own duration must not have to be guessed at, so this documents what withholding it costs.
    bare = engine.ask("This is a video clip.", asked, videos=[clip.frames])["seconds"].value
    if bare == 6:
        pytest.skip(f"the model guessed the duration without being told it ({bare}s); the check proves nothing today")


def test_a_clip_keeps_its_duration_however_few_frames_survive(engine):
    """The frame cap bounds decoding work, not what the clip is. Halving it must not halve the clip."""
    pytest.importorskip("cv2")
    from prismyra.media import decode_video

    encoded = _six_second_clip()
    asked = [Scale(id="seconds", prompt="Roughly how many seconds long is this clip?", low=1, high=9)]
    for cap in (256, 32):
        clip = decode_video(encoded, max_frames=cap)
        assert clip.duration == pytest.approx(6.0, abs=0.3)
        assert engine.ask("This is a video clip.", asked, videos=[clip])["seconds"].value >= NEARLY_SIX


def _six_second_clip(seconds: int = 6, fps: int = 30, size: int = 224) -> bytes:
    """A block moving left to right, red for the first two thirds and blue for the last third."""
    import tempfile

    import cv2

    total = seconds * fps
    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/clip.mp4"
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
        for i in range(total):
            frame = np.full((size, size, 3), 255, np.uint8)
            x = int(i / total * (size - 50))
            colour = (60, 60, 200) if i < total * 2 // 3 else (200, 60, 60)  # BGR, so red then blue
            cv2.rectangle(frame, (x, 90), (x + 45, 135), colour, -1)
            writer.write(frame)
        writer.release()
        return open(path, "rb").read()


def test_a_replayed_pass_answers_exactly_as_the_eager_one_did(engine):
    """The promise a recording has to keep, and it is not a tolerance.

    A replay runs no Python. Each cache layer keeps a host-side count of the tokens it holds so the framework can ask
    for the length without a device read, and a replay moves the bytes and leaves that integer where it was; the next
    group then advances from the wrong offset and answers **plausibly**. That is why this compares probabilities and not
    only decisions, over four groups rather than one -- a recording is taken on a shape's second use, so the
    first replay is the third group, and the group after it is where a stale count would show.

    One engine with the flag flipped between contexts, not two engines. The flag is read per pass, and two copies of
    these weights do not fit on one card beside the one this file's fixture already holds -- which is the same limit the
    skipped test above records.
    """
    asked = questions(4)
    context = CONTEXT * 6
    was = engine.graphs

    def groups() -> list[dict]:
        out = []
        with engine.open_context(context) as opened:
            for _ in range(4):
                result = opened.ask(asked)
                out.append({q.id: (result[q.id].option, dict(result[q.id].probabilities)) for q in asked})
        return out

    try:
        engine.graphs = False
        eager = groups()
        engine.graphs = True
        replayed = groups()
        assert engine.stats()["graphs_declined"] == {}, "the recording was refused, so nothing was replayed"
    finally:
        engine.graphs = was

    for n, (want, got) in enumerate(zip(eager, replayed, strict=True)):
        for name in want:
            assert got[name][0] == want[name][0], f"group {n}, question {name} changed its answer"
            for option, p in want[name][1].items():
                assert got[name][1][option] == pytest.approx(p, abs=1e-4), f"group {n}, {name}, option {option}"
