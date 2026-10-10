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

    That import turns annotations into strings, which the web framework resolves against the module's globals -- and
    the request models live inside `create_app`. It then decided the body parameter was a query parameter and rejected
    every request with "field required" for a field the caller did send. Nothing below the transport could see it: the
    schema, the read-out and the queue were all fine. Only a real request finds it, so this makes one, with a stub
    engine so no device is needed.
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


def test_the_served_version_is_the_package_version():
    """`create_app` passes `__version__` to FastAPI once, at construction, so a caller reading the OpenAPI document
    -- or anything built against it -- sees the same number `import prismyra` does, not a copy that can drift from
    it."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import prismyra
    from prismyra.server import create_app

    class StubEngine:
        model_name = "stub"
        group = 32
        tokenizer = None

        def cache_bytes(self, tokens):
            return tokens * 1024

        def stats(self):
            return {"model": "stub"}

    real = prismyra.Prismyra
    prismyra.Prismyra = lambda *a, **k: StubEngine()
    try:
        app = create_app("stub/model")
        assert app.version == prismyra.__version__ == "0.4.4"
        client = TestClient(app)
        assert client.get("/openapi.json").json()["info"]["version"] == prismyra.__version__
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

    Hand a processor sixteen frames with nothing else and it assumes 24 per second, decides the clip is two thirds of
    a second long, and keeps a handful. Every question about when something happened is then answered about a clip
    that does not exist. So the decoder reports the rate of the array it returns, which is the source rate over the
    stride it used -- not the source rate, and not a default.
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
    "definitely B" as D. A baseline that loses points to its own parser is not a baseline, and the accuracy it makes
    the read-out look better than is not a result.
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


# --------------------------------------------------------------------------- --batcher: routing, without a device
#
# `Batcher` (prismyra/schedule.py) is its own, device-free-testable object -- see tests/test_schedule.py's
# `FakeEngine`/`FakeShelf`. What is new here, and not covered there, is `/ask`'s own choice of which path a request
# takes: `StubBatchableEngine` is that same shape of stand-in, extended with `ask()` so a request this endpoint
# cannot send through `Batcher` (media, or more questions than one pass holds) still has somewhere to go, and the
# two paths leave different traces (`direct_asks` against `shelf_asks`) so a test can tell which one a request
# actually travelled through without reading any probability.


class _StubShelf:
    """Enough of `prismyra.engine.Shelf` for `Batcher._answer`'s non-fused path (`StubBatchableEngine.interleaved_fork`
    stays `False`, same as `tests/test_schedule.py`'s `FakeEngine`, so the fused paths are never reached here)."""

    def __init__(self, engine):
        self.engine = engine
        self.documents: dict[int, object] = {}
        self._next = 0

    def put_many(self, contexts):
        handles = []
        for context in contexts:
            self.documents[self._next] = type("Shelved", (), {"tokens": len(context.split()), "snapshot_bytes": 0})()
            handles.append(self._next)
            self._next += 1
        return handles

    def ask(self, asked: dict, lane: int = 0) -> dict:
        out = {}
        for handle, questions in asked.items():
            self.engine.shelf_asks.append((handle, len(questions)))
            out[handle] = _stub_result(questions)
        return out

    def would_fit(self, token_counts: list[int]) -> bool:
        """Enough of `Shelf.would_fit` for `Batcher._make_room` to call -- this stub has no page pool to fragment."""
        return True

    def drop(self, handle):
        self.documents.pop(handle, None)

    def close(self):
        self.documents.clear()


def _stub_result(questions):
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
    return Result(answers=answers, timing=Timing(context_ms=1.0, readout_ms=1.0), model="stub")


class StubBatchableEngine:
    """`paged = True` and the shelf methods are what let `Batcher` build against this at all; `ask()` is what the
    plain queue (`Worker`) calls for everything `Batcher` cannot take. `interleaved_fork = False`, same reason as
    `tests/test_schedule.py`'s `FakeEngine`: this file's own job is which path `/ask` picks, not the fused path's
    own device-only correctness (covered on real hardware by `tests/test_gpu.py::test_a_shelf_matches_ask_bit_for_bit`
    and `tools/audit_sm120.py`).
    """

    model_name = "stub"
    paged = True
    wide_group = False
    interleaved_fork = False

    def __init__(self, group: int = 8):
        self.group = group
        self.longest_context = 4096
        self.fastest_read_ms = 1.0
        self.torch_device = type("Device", (), {"type": "cpu"})()
        self.direct_asks: list[tuple] = []
        self.shelf_asks: list[tuple] = []
        self.tokenizer = staticmethod(lambda text, **_: {"input_ids": text.split()})

    def cache_bytes(self, tokens):
        return tokens * 1024

    def stats(self):
        return {"model": "stub"}

    def validate(self, questions):
        return None

    def encode_context(self, text: str):
        return type("Encoded", (), {"text": text, "tokens": len(text.split())})()

    def open_shelf(self, room=None, lane: int = 0, group=None) -> _StubShelf:
        return _StubShelf(self)

    def ask(self, context, questions, images=None, videos=None):
        self.direct_asks.append((context, len(questions), images, videos))
        return _stub_result(questions)


def _client_with(engine, **app_kwargs):
    """A `TestClient` wired to `engine` the same way `test_the_endpoint_accepts_a_request_body_over_real_http`
    wires a `StubEngine`: `prismyra.Prismyra` is replaced for the one call that builds the app."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import prismyra
    from prismyra.server import create_app

    real = prismyra.Prismyra
    prismyra.Prismyra = lambda *a, **k: engine
    try:
        app = create_app("stub/model", **app_kwargs)
    finally:
        prismyra.Prismyra = real
    return TestClient(app)


def _ask(client, questions, **body):
    return client.post("/ask", json={"context": "a short document", "questions": questions, **body})


def test_batcher_off_by_default_behaves_exactly_as_before():
    """The default (`batcher=False`) must be the plain queue, unchanged: no `Batcher` is built, `/stats` carries
    no `batcher` key, and every request answers through `ask()`."""
    engine = StubBatchableEngine()
    client = _client_with(engine)
    assert client.app.state.batcher is None

    response = _ask(client, [{"id": "q0", "prompt": "Ships today?"}])
    assert response.status_code == 200, response.text
    assert engine.direct_asks and not engine.shelf_asks
    assert "batcher" not in client.get("/stats").json()


def test_batcher_on_routes_a_plain_text_request_through_the_scheduler():
    """A text-only request that fits in one pass (2 questions against a group of 8) takes the batched path, and
    `/stats` now reports it."""
    engine = StubBatchableEngine(group=8)
    client = _client_with(engine, batcher=True)
    assert client.app.state.batcher is not None

    response = _ask(client, [{"id": "q0", "prompt": "Ships today?"}, {"id": "q1", "prompt": "Returnable?"}])
    assert response.status_code == 200, response.text
    assert engine.shelf_asks and not engine.direct_asks
    assert response.json()["answers"]["q0"]["kind"] == "boolean"
    assert "passes" in client.get("/stats").json()["batcher"]


def test_batcher_on_still_sends_media_through_the_plain_queue():
    """`Batcher` is text-only (`schedule.py`'s own docstring, same restriction as `open_batch`); a request with an
    image must still answer, through `ask()`, exactly as it would with `batcher` off."""
    import base64
    import io

    pytest.importorskip("PIL")
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()

    engine = StubBatchableEngine(group=8)
    client = _client_with(engine, batcher=True)
    response = _ask(client, [{"id": "q0", "prompt": "Is it blue?"}], images=[encoded])
    assert response.status_code == 200, response.text
    assert engine.direct_asks and not engine.shelf_asks


def test_batcher_on_still_sends_an_over_wide_request_through_the_plain_queue():
    """A request asking more questions than one pass holds (`_batcher_capacity`) is not something `Batcher.submit`
    will take -- `engine.ask()` answers it in several branch passes on its own, so it still goes to the plain
    queue rather than being refused."""
    engine = StubBatchableEngine(group=2)
    client = _client_with(engine, batcher=True)
    questions = [{"id": f"q{i}", "prompt": f"Clause {i}?"} for i in range(3)]
    response = _ask(client, questions)
    assert response.status_code == 200, response.text
    assert engine.direct_asks and not engine.shelf_asks


def test_namespaced_ids_never_collide_across_two_requests_sharing_a_document():
    """`/ask` promises a request is stateless and carries its own context; a caller reasonably reads that as "my
    ids only have to be unique within my own request body". `Batcher` does not keep that promise on its own:
    `schedule.py`'s own `_answer` docstring says two callers asking about the *same* document in one pass are
    "merged instead -- one document, both callers' questions, one set of rows", and the merge is a plain
    `list.extend`, not a disjoint union -- two callers who both call their question `"ok"` collide. Found on real
    hardware (`measure_packed.py`'s open-loop HTTP run against a 64-document pool at a sustained arrival rate):
    several requests got a 422 for "duplicate question ids" naming an id their own request never sent, because a
    *different* concurrent caller's identically-named question about the same document landed in the same pass.

    This is the pure, device-free half of the fix: two different requests about the same document, using the
    exact same caller-chosen id, must still namespace to two different ids (so a real `Batcher`'s merge cannot
    collide them), and restoring must hand each caller back only its own.
    """
    from prismyra import Boolean
    from prismyra.schema import Answer, Result, Timing
    from prismyra.server import _namespaced, _restore_ids

    request_a = [Boolean(id="ok", prompt="Ships today?")]
    request_b = [Boolean(id="ok", prompt="Returnable within 30 days?")]  # same id, different question, same doc

    namespaced_a, map_a = _namespaced(request_a)
    namespaced_b, map_b = _namespaced(request_b)

    assert namespaced_a[0].id != namespaced_b[0].id, "two requests' namespaced ids must never collide"
    assert namespaced_a[0].prompt == "Ships today?"  # renaming touches only the id, nothing the model reads

    def stub_result(renamed_id: str, value: bool) -> Result:
        answer = Answer(id=renamed_id, kind="boolean", value=value, option="yes" if value else "no", probabilities={})
        return Result(answers={renamed_id: answer}, timing=Timing(), model="stub")

    restored_a = _restore_ids(stub_result(namespaced_a[0].id, True), map_a)
    restored_b = _restore_ids(stub_result(namespaced_b[0].id, False), map_b)

    assert set(restored_a.answers) == {"ok"}
    assert set(restored_b.answers) == {"ok"}
    assert restored_a["ok"].value is True
    assert restored_b["ok"].value is False  # not swapped with request_a's answer


def test_restore_ids_drops_a_companions_answers_from_the_shared_merged_result():
    """The deeper half of the same finding. `_answer`'s own merge does not give each job a *slice* of the
    document's answers -- it gives every job sharing a handle the *same* `Result` object, whatever every
    companion in that pass also asked about that document (`return [answers[self._resident[job.payload.digest]]
    for job in formed.jobs]`: one shared value, read once per job). Found on real hardware exactly this way: once
    namespacing alone stopped the id collision, a *different* failure appeared at a higher arrival rate --
    `KeyError` on a companion's own namespaced id, raised by the first draft of `_restore_ids`, which assumed
    every key in `result.answers` was this request's own and tried to restore all of them.

    `id_map` is this request's own and only this request's own, so filtering by it (`if k in id_map`) is what
    turns the companion's leaked answer into something silently dropped rather than a `KeyError` -- or worse,
    something returned to a caller that never asked for it.
    """
    from prismyra.schema import Answer, Result, Timing
    from prismyra.server import _restore_ids

    # One document, two callers who both asked about it in the same pass: this request's own answer, plus a
    # companion's -- under a namespaced id this request's own `id_map` was never given.
    mine = Answer(id="mine:q0", kind="boolean", value=True, option="yes", probabilities={})
    companions = Answer(id="companion:q0", kind="boolean", value=False, option="no", probabilities={})
    merged = Result(answers={"mine:q0": mine, "companion:q0": companions}, timing=Timing(), model="stub")

    restored = _restore_ids(merged, {"mine:q0": "q0"})

    assert set(restored.answers) == {"q0"}, "a companion's answer must not leak into this request's response"
    assert restored["q0"].value is True


# --------------------------------------------------------------------------- /v1/decide, over real HTTP
#
# `prismyra.decide`'s own tests (`tests/test_decide.py`) cover the parsing, labelling and grouping logic without a
# device. What is new here is the wiring: the route exists, takes a single JEV-shaped object or a JSON array, and
# answers through the same `worker` `/ask` already uses -- not by calling `engine.ask` from the HTTP handler's own
# thread, which `server.py`'s own docstring says must not happen.


class _DecideStubEngine:
    model_name = "stub"
    group = 32

    class _Tok:
        def __call__(self, text, **_):
            return {"input_ids": [hash(text.strip()) % 10_000_000]}

    tokenizer = _Tok()

    def __init__(self):
        self.asks: list[tuple] = []

    def cache_bytes(self, tokens):
        return tokens * 1024

    def stats(self):
        return {"model": "stub"}

    def ask(self, context, questions, images=None, videos=None):
        self.asks.append((context, [q.id for q in questions]))
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
        return Result(answers=answers, timing=Timing(context_ms=1.0, readout_ms=1.0), model="stub")


def _decide_client(engine):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import prismyra
    from prismyra.server import create_app

    real = prismyra.Prismyra
    prismyra.Prismyra = lambda *a, **k: engine
    try:
        app = create_app("stub/model")
    finally:
        prismyra.Prismyra = real
    return TestClient(app)


def test_decide_endpoint_answers_a_single_jev_shaped_object():
    engine = _DecideStubEngine()
    client = _decide_client(engine)
    response = client.post(
        "/v1/decide",
        json={"kind": "noul", "state": "a short document", "question": "Is it urgent?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "noul"
    assert body["options"] == ["false", "true"]
    assert body["protocol"] == "prismyra-decide-v1"
    assert "results" not in body  # a single object in gets a single object back, JEV's own shape


def test_decide_endpoint_merges_a_batch_sharing_one_state_into_one_ask_call():
    engine = _DecideStubEngine()
    client = _decide_client(engine)
    response = client.post(
        "/v1/decide",
        json=[
            {"kind": "noul", "state": "the same record", "question": "Is it urgent?", "id": "a"},
            {"kind": "score", "state": "the same record", "question": "How clear is it?", "id": "b"},
        ],
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["num_model_requests"] == 1
    assert len(engine.asks) == 1 and len(engine.asks[0][1]) == 2
    by_id = {r["id"]: r for r in body["results"]}
    assert by_id["a"]["merged_with"] == ["b"]


def test_decide_endpoint_refuses_a_bad_kind_before_touching_the_device():
    engine = _DecideStubEngine()
    client = _decide_client(engine)
    response = client.post("/v1/decide", json={"kind": "freeform", "state": "x", "question": "?"})
    assert response.status_code == 422
    assert engine.asks == []
