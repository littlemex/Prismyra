"""The wire translation, without a device: a bad question is refused before the model is touched, and the response
carries what a stored answer needs."""

from __future__ import annotations

import pytest

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

        def ask(self, context, questions):
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
