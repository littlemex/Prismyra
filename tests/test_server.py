"""The wire translation, without a device: a bad question is refused before the model is touched, and the response
carries what a stored answer needs."""

from __future__ import annotations

import pytest

from prismyra import Boolean, Choice
from prismyra.schema import Answer, QuestionError, Result, Timing
from prismyra.server import as_json, build_questions, with_queue_time


def test_each_kind_survives_the_wire():
    questions = build_questions(
        [
            {"id": "a", "prompt": "Returnable?", "kind": "boolean"},
            {"id": "b", "prompt": "Who pays?", "kind": "choice", "choices": ["seller", "buyer"]},
            {"id": "c", "prompt": "How urgent?", "kind": "scale", "low": 1, "high": 3},
        ]
    )
    assert [q.kind for q in questions] == ["boolean", "choice", "scale"]
    assert questions[2].options == ["1", "2", "3"]


def test_a_missing_id_is_filled_from_the_position():
    """So a caller who sends a bare list still gets answers it can index."""
    assert [q.id for q in build_questions([{"prompt": "One?"}, {"prompt": "Two?"}])] == ["q0", "q1"]


def test_an_unknown_kind_is_refused_by_name():
    with pytest.raises(QuestionError, match="unknown kind"):
        build_questions([{"id": "a", "prompt": "?", "kind": "freeform"}])


def test_a_bad_question_is_refused_before_any_device_time():
    """A choice with one option cannot be scored, and finding that out inside the worker would delay the queue."""
    with pytest.raises(QuestionError):
        build_questions([{"id": "a", "prompt": "?", "kind": "choice", "choices": ["only"]}])


def test_the_response_separates_waiting_from_working_and_carries_the_scoring_version():
    result = Result(
        answers={
            "a": Answer(
                id="a", kind="boolean", value=True, option="yes", probabilities={"yes": 0.811234567, "no": 0.188765433}
            )
        },
        timing=Timing(context_ms=138.25, readout_ms=91.64),
        model="m",
        context_tokens=4924,
    )
    body = as_json(with_queue_time(result, queue_ms=12.34))

    assert body["timing"] == {"queue_ms": 12.3, "context_ms": 138.2, "readout_ms": 91.6, "total_ms": 242.2}
    assert body["scoring_version"] == result.scoring_version
    assert body["context_tokens"] == 4924
    assert body["answers"]["a"]["value"] is True
    assert body["answers"]["a"]["probabilities"] == {"yes": 0.811235, "no": 0.188765}
    assert "confidence" not in body["answers"]["a"]


def test_a_filled_id_that_collides_with_a_given_one_is_refused():
    """Reachable without the caller repeating anything: the missing id is filled from the position."""
    with pytest.raises(QuestionError, match="duplicate question ids"):
        build_questions([{"id": "q1", "prompt": "One?"}, {"prompt": "Two?"}])


def test_the_endpoint_accepts_a_request_body_over_real_http():
    """Regression, and the reason the server module has no `from __future__ import annotations`.

    That import turns annotations into strings, which the web framework resolves against the module's globals -- and the
    request models live inside `create_app`. It then decided the body parameter was a query parameter and rejected every
    request with "field required" for a field the caller did send. Nothing below the transport could see it: the schema,
    the read-out and the queue were all fine. Only a real request finds it, so this makes one, with a stub engine so no
    device is needed.
    """
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import prismyra
    from prismyra.server import create_app

    class StubEngine:
        model_name = "stub"
        group = 32

        class _Tok:
            def __call__(self, text, **_):
                return {"input_ids": list(range(len(text.split())))}

        tokenizer = _Tok()

        def cache_bytes(self, tokens):
            return tokens * 1024

        def stats(self):
            return {"model": "stub"}

        def ask(self, context, questions, images=None, videos=None):
            self.saw_media = (images, videos)
            answers = {
                q.id: Answer(
                    id=q.id,
                    kind=q.kind,
                    value=q.value_of(q.options[0]),
                    option=q.options[0],
                    probabilities=dict.fromkeys(q.options, 1.0 / len(q.options)),
                )
                for q in questions
            }
            return Result(answers=answers, timing=Timing(context_ms=1.0, readout_ms=2.0), model="stub")

    real = prismyra.Prismyra
    prismyra.Prismyra = lambda *a, **k: StubEngine()
    try:
        client = TestClient(create_app("stub/model"))
        response = client.post(
            "/ask",
            json={
                "context": "Returns are accepted within thirty days.",
                "questions": [{"id": "thirty", "prompt": "Is there a limit?", "kind": "boolean"}],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["answers"]["thirty"]["kind"] == "boolean"
        assert "queue_ms" in body["timing"]
        assert client.get("/health").json()["ok"] is True
    finally:
        prismyra.Prismyra = real


def test_an_image_arrives_as_bytes_and_reaches_the_engine():
    """Base64 rather than a path or a URL: a path names a file on the server, and a URL sends the server fetching.

    Also covers that a request with media and no text is accepted -- a picture is a context.
    """
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("PIL")
    import base64
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    import prismyra
    from prismyra.server import create_app

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (200, 30, 30)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()

    seen = {}

    class StubEngine:
        model_name = "stub"
        group = 32
        tokenizer = staticmethod(lambda text, **_: {"input_ids": []})

        def cache_bytes(self, tokens):
            return 0

        def stats(self):
            return {}

        def ask(self, context, questions, images=None, videos=None):
            seen["images"] = images
            answer = Answer(id="q0", kind="boolean", value=True, option="yes", probabilities={"yes": 1.0, "no": 0.0})
            return Result(answers={"q0": answer}, timing=Timing(), model="stub")

    real = prismyra.Prismyra
    prismyra.Prismyra = lambda *a, **k: StubEngine()
    try:
        client = TestClient(create_app("stub/model"))
        response = client.post(
            "/ask",
            json={"context": "", "images": [encoded], "questions": [{"id": "q0", "prompt": "Is it red?"}]},
        )
        assert response.status_code == 200, response.text
        assert len(seen["images"]) == 1
        assert seen["images"][0].size == (32, 32)
        assert seen["images"][0].mode == "RGB"

        empty = client.post("/ask", json={"context": "  ", "questions": [{"id": "q0", "prompt": "Is it?"}]})
        assert empty.status_code == 422

        bad = client.post(
            "/ask",
            json={"context": "x", "images": ["not base64!"], "questions": [{"id": "q0", "prompt": "Is it?"}]},
        )
        assert bad.status_code == 422
    finally:
        prismyra.Prismyra = real


def test_a_clip_reports_the_rate_its_frames_actually_run_at():
    """The timing is what the processor needs, and getting it wrong is silent.

    Hand a processor sixteen frames with nothing else and it assumes 24 per second, decides the clip is two thirds of a
    second long, and keeps a handful. Every question about when something happened is then answered about a clip that
    does not exist. So the decoder reports the rate of the array it returns, which is the source rate over the stride it
    used -- not the source rate, and not a default.
    """
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    import numpy as np

    from prismyra.media import decode_video

    encoded = _tiny_video(frames=60, fps=30)

    whole = decode_video(encoded, max_frames=256)
    assert whole.frames.shape[0] == 60
    assert whole.fps == pytest.approx(30, abs=0.5)
    assert whole.duration == pytest.approx(2.0, abs=0.2)

    # Capped: a stride of three, so the array runs at a third of the source rate and says so.
    strided = decode_video(encoded, max_frames=20)
    assert strided.frames.shape[0] == 20
    assert strided.fps == pytest.approx(10, abs=0.5)
    assert strided.duration == pytest.approx(whole.duration, abs=0.2)
    assert strided.source_frames == 60

    meta = strided.metadata
    assert meta.total_num_frames == 20
    assert meta.fps == pytest.approx(10, abs=0.5)
    assert np.isclose(meta.duration, whole.duration, atol=0.2)


def _tiny_video(frames: int, fps: int) -> bytes:
    """A clip written with the same decoder that reads it, so the test needs no fixture file."""
    import tempfile

    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/clip.mp4"
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
        for i in range(frames):
            frame = np.zeros((64, 64, 3), np.uint8)
            frame[:, min(i, 63)] = 255
            writer.write(frame)
        writer.release()
        return open(path, "rb").read()


# --------------------------------------------------------------------------- the evaluation harness's parser
def test_a_generated_answer_is_read_without_crediting_or_robbing_the_model():
    """The parser in `evals/generate.py`, which decides what the comparison baseline scored.

    Every case below was wrong at some point in writing it, and every one of those errors went against generation: a
    single-letter option matched as a prefix reads "because the passage says so" as B, "a few people" as A and
    "definitely B" as D. A baseline that loses points to its own parser is not a baseline, and the accuracy it makes the
    read-out look better than is not a result.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
    from generate import parse

    letters = Choice(id="c", prompt="?", choices=["A", "B", "C", "D"])
    yes_no = Boolean(id="b", prompt="?")

    assert parse("B", letters)[0] == "B"
    assert parse("<think>\n\n</think>\n\nB", letters)[0] == "B"
    assert parse("B.", letters)[0] == "B"
    assert parse("The answer is B", letters)[0] == "B"
    assert parse("definitely B", letters)[0] == "B"

    # Prose that happens to start with, or contain, a letter that is also an option.
    assert parse("because the passage says so", letters)[0] is None
    assert parse("a few people were hurt", letters)[0] is None

    # Two options named is not an answer to a closed question.
    assert parse("It is either A or B", letters) == (None, "ambiguous")

    # An unclosed reasoning block means the budget ran out, which is unanswered rather than wrong.
    assert parse("<think>", letters) == (None, "empty")

    # The option's own text, when the question was rendered as letters.
    assert parse("Over 2,000 people", letters, aliases={"A": ["Over 2,000 people."]})[0] == "A"

    assert parse("yes, because the policy says so", yes_no)[0] is True
    assert parse("no.", yes_no)[0] is False
    assert parse("I am not sure", yes_no)[0] is None
