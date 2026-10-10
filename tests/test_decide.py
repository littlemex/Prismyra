"""`prismyra.decide`: the JEV-compatible `/v1/decide` read-out, without a device.

A fake tokenizer stands in for the real one everywhere here. `choice_labels` only needs a tokenizer that answers
`__call__(text, add_special_tokens=False) -> {"input_ids": [...]}`, the same contract `prismyra.readout.plan` already
depends on, so a fake that gives every distinct piece of text its own single token id is enough to test the label
search itself -- whether it finds a usable pool, and what it does when the pool is smaller than requested -- without
needing a real vocabulary.
"""

from __future__ import annotations

import pytest

from prismyra.decide import (
    _LABEL_CANDIDATES,
    DecideError,
    choice_labels,
    decide_many,
    parse_item,
)
from prismyra.schema import Answer, Result, Timing


class _UnlimitedTokenizer:
    """Every distinct string is one token, distinct from every other -- a vocabulary with no collisions and no
    multi-token words, which is as permissive as `choice_labels` will ever see."""

    def __call__(self, text, **_):
        bare = text.strip()
        return {"input_ids": [hash(bare) % 10_000_000]}


class _LimitedTokenizer:
    """Only the first `limit` label candidates are usable as single tokens; everything else, including every
    option text this file's own tests use, comes back as two tokens -- which `prismyra.readout.plan` refuses.
    Exercises `choice_labels`'s binary-search fallback without a real, size-constrained vocabulary.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self._usable = _LABEL_CANDIDATES[:limit]

    def __call__(self, text, **_):
        bare = text.strip()
        if bare in self._usable:
            return {"input_ids": [self._usable.index(bare)]}
        return {"input_ids": [900001, 900002]}


# --------------------------------------------------------------------------- choice_labels


def test_choice_labels_finds_the_full_pool_when_the_tokenizer_allows_it():
    labels = choice_labels(_UnlimitedTokenizer(), limit=255)
    assert len(labels) == 255
    assert labels[:3] == ["A", "B", "C"]


def test_choice_labels_falls_back_to_a_smaller_pool_the_tokenizer_actually_supports():
    labels = choice_labels(_LimitedTokenizer(limit=10), limit=255)
    assert len(labels) == 10
    assert labels == _LABEL_CANDIDATES[:10]


def test_parse_item_labels_a_choice_question_past_the_26_letter_alphabet():
    """`choice_labels`' own candidate order continues past `Z` with two-letter labels (`AA`, `AB`, ...). A
    `choice` question with more than 26 options must actually reach and use them, not just have them exist in
    the candidate pool `choice_labels` returns."""
    labels = _labels(255)
    options = [f"opt{i}" for i in range(30)]
    item = parse_item({"kind": "choice", "state": "x", "question": "?", "options": options}, 0, labels)
    assert item.symbols["opt26"] == "AA"
    assert item.symbols["opt29"] == "AD"
    assert set(item.question.choices) == set(labels[:30])


# --------------------------------------------------------------------------- parse_item


def _labels(n=255):
    return choice_labels(_UnlimitedTokenizer(), limit=n)


def test_parse_item_builds_a_noul_question():
    item = parse_item({"kind": "noul", "state": "a document", "question": "Is it true?"}, 0, _labels())
    assert item.kind == "noul"
    assert item.question.kind == "boolean"
    assert item.reported_options == ("false", "true")
    assert item.external_id == "0"


def test_parse_item_builds_a_score_question():
    item = parse_item({"kind": "score", "state": "a document", "question": "How clear?"}, 1, _labels())
    assert item.question.kind == "scale"
    assert item.question.options == ["0", "1", "2", "3", "4", "5"]
    assert item.reported_options == ("0", "1", "2", "3", "4", "5")


def test_parse_item_builds_a_choice_question_with_symbol_labels_in_the_callers_own_order():
    item = parse_item(
        {"kind": "choice", "state": "a document", "question": "Who pays?", "options": ["seller", "buyer"]},
        2,
        _labels(),
    )
    assert item.question.kind == "choice"
    assert item.reported_options == ("seller", "buyer")  # caller's own order, not sorted
    assert item.symbols == {"seller": "A", "buyer": "B"}
    assert set(item.question.choices) == {"A", "B"}
    assert "A) seller" in item.question.prompt and "B) buyer" in item.question.prompt


def test_parse_item_fills_the_id_from_the_position_like_build_questions_does():
    item = parse_item({"kind": "noul", "state": "x", "question": "?"}, 7, _labels())
    assert item.external_id == "7"
    with_id = parse_item({"kind": "noul", "state": "x", "question": "?", "id": "my-id"}, 7, _labels())
    assert with_id.external_id == "my-id"


def test_parse_item_refuses_an_unknown_kind():
    with pytest.raises(DecideError, match="kind must be one of"):
        parse_item({"kind": "freeform", "state": "x", "question": "?"}, 0, _labels())


def test_parse_item_refuses_an_empty_question():
    with pytest.raises(DecideError, match="non-empty question"):
        parse_item({"kind": "noul", "state": "x", "question": "  "}, 0, _labels())


def test_parse_item_refuses_an_empty_state_with_an_explanation_of_the_architectural_difference():
    """JEV's own protocol allows an empty `state`; Prismyra's `ask()` requires a non-empty context. This is a
    real boundary, not a bug, and the message says so rather than failing opaquely."""
    with pytest.raises(DecideError, match="Prismyra's engine does not"):
        parse_item({"kind": "noul", "state": "", "question": "Is it true?"}, 0, _labels())
    with pytest.raises(DecideError, match="Prismyra's engine does not"):
        parse_item({"kind": "noul", "state": "   ", "question": "Is it true?"}, 0, _labels())


def test_parse_item_refuses_a_list_shaped_state_by_name():
    with pytest.raises(DecideError, match="image or video parts"):
        parse_item({"kind": "noul", "state": ["text", {"image": "data:..."}], "question": "?"}, 0, _labels())


def test_parse_item_accepts_a_dict_state_rendered_as_json_like_jevs_own_parts_does():
    item = parse_item({"kind": "noul", "state": {"a": 1}, "question": "?"}, 0, _labels())
    assert item.context == '{"a": 1}'


def test_parse_item_refuses_choice_above_jevs_own_256_ceiling():
    options = [f"opt{i}" for i in range(257)]
    with pytest.raises(DecideError, match=r"2-256 options \(JEV's own ceiling\)"):
        parse_item({"kind": "choice", "state": "x", "question": "?", "options": options}, 0, _labels())


def test_parse_item_refuses_choice_above_what_this_tokenizer_can_label_even_under_jevs_ceiling():
    labels = _labels(5)
    options = [f"opt{i}" for i in range(6)]
    with pytest.raises(DecideError, match=r"2-5 options on this build"):
        parse_item({"kind": "choice", "state": "x", "question": "?", "options": options}, 0, labels)


def test_parse_item_refuses_fewer_than_two_options():
    with pytest.raises(DecideError, match="at least 2 options"):
        parse_item({"kind": "choice", "state": "x", "question": "?", "options": ["only"]}, 0, _labels())


def test_parse_item_refuses_a_repeated_option():
    with pytest.raises(DecideError, match="repeats an option"):
        parse_item(
            {"kind": "choice", "state": "x", "question": "?", "options": ["a", "a"]},
            0,
            _labels(),
        )


# --------------------------------------------------------------------------- decide_many


class _FakeEngine:
    model_name = "stub-model"
    tokenizer = _UnlimitedTokenizer()


def _uniform_ask(context: str, questions: list, calls: list) -> Result:
    """Stands in for `Worker.submit(...).result`: answers every question with a uniform distribution over its
    own options, the same stub shape `tests/test_server.py` already uses for the plain `/ask` route."""
    calls.append((context, [q.id for q in questions]))
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
    return Result(answers=answers, timing=Timing(), model="stub")


def test_decide_many_answers_a_single_noul_item():
    calls = []
    engine = _FakeEngine()
    responses, num_requests = decide_many(
        engine,
        [{"kind": "noul", "state": "a document", "question": "Is it true?"}],
        lambda ctx, qs: _uniform_ask(ctx, qs, calls),
    )
    assert num_requests == 1
    assert len(responses) == 1
    r = responses[0]
    assert r["kind"] == "noul"
    assert r["options"] == ["false", "true"]
    assert r["probabilities"] == [0.5, 0.5]
    assert r["value"] is False  # uniform stub's own value_of(q.options[0]) == value_of("no") == False
    assert r["protocol"] == "prismyra-decide-v1"
    assert r["model"] == "stub-model"
    assert r["merged_with"] == []


def test_decide_many_merges_items_sharing_one_state_into_one_ask_call():
    """The point of this module: two decisions about the same record cost one `ask()`, not two."""
    calls = []
    engine = _FakeEngine()
    raw = [
        {"kind": "noul", "state": "the same document", "question": "Is it urgent?", "id": "urgent"},
        {"kind": "score", "state": "the same document", "question": "How clear is it?", "id": "clarity"},
    ]
    responses, num_requests = decide_many(engine, raw, lambda ctx, qs: _uniform_ask(ctx, qs, calls))
    assert num_requests == 1
    assert len(calls) == 1
    assert len(calls[0][1]) == 2  # both questions went into the one ask() call
    by_id = {r["id"]: r for r in responses}
    assert by_id["urgent"]["merged_with"] == ["clarity"]
    assert by_id["clarity"]["merged_with"] == ["urgent"]


def test_decide_many_does_not_merge_items_with_different_states():
    calls = []
    engine = _FakeEngine()
    raw = [
        {"kind": "noul", "state": "document one", "question": "Is it urgent?", "id": "a"},
        {"kind": "noul", "state": "document two", "question": "Is it urgent?", "id": "b"},
    ]
    responses, num_requests = decide_many(engine, raw, lambda ctx, qs: _uniform_ask(ctx, qs, calls))
    assert num_requests == 2
    assert len(calls) == 2
    assert all(r["merged_with"] == [] for r in responses)


def test_decide_many_choice_response_reports_options_in_the_callers_order_and_a_symbol_map():
    calls = []
    engine = _FakeEngine()
    raw = [
        {"kind": "choice", "state": "a document", "question": "Who pays?", "options": ["seller", "buyer"], "id": "x"}
    ]
    responses, _ = decide_many(engine, raw, lambda ctx, qs: _uniform_ask(ctx, qs, calls))
    r = responses[0]
    assert r["options"] == ["seller", "buyer"]
    assert r["adaptation"] == "symbol-labelled"
    assert r["symbols"] == {"seller": "A", "buyer": "B"}
    # Uniform stub probabilities: the first reported option wins ties, deterministically.
    assert r["choice_index"] == 0
    assert r["choice"] == "seller"
    assert r["value"] == "seller"


def test_decide_many_is_all_or_nothing_one_bad_item_refuses_the_whole_request():
    calls = []
    engine = _FakeEngine()
    raw = [
        {"kind": "noul", "state": "a document", "question": "Is it urgent?", "id": "ok"},
        {"kind": "freeform", "state": "a document", "question": "?", "id": "bad"},
    ]
    with pytest.raises(DecideError):
        decide_many(engine, raw, lambda ctx, qs: _uniform_ask(ctx, qs, calls))
    assert calls == []  # nothing was submitted to the device
