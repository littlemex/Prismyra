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
