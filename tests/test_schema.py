"""The contract: what a question may declare, and what an answer means. No device needed."""

from __future__ import annotations

import pytest

from prismyra import Boolean, Choice, QuestionError, Request, Scale
from prismyra.schema import SCORING_VERSION, Answer, Result, Timing, render_question


def test_a_question_needs_at_least_two_options():
    with pytest.raises(QuestionError, match="at least two"):
        Choice(id="x", prompt="p", choices=["only"])


def test_repeated_options_are_refused():
    with pytest.raises(QuestionError, match="repeats"):
        Choice(id="x", prompt="p", choices=["a", "b", "a"])


def test_an_empty_prompt_is_refused_at_construction():
    with pytest.raises(QuestionError, match="empty prompt"):
        Boolean(id="x", prompt="   ")


def test_a_scale_must_ascend():
    with pytest.raises(QuestionError, match="not above"):
        Scale(id="x", prompt="p", low=5, high=5)


def test_values_are_typed_by_kind():
    assert Boolean(id="b", prompt="p").value_of("yes") is True
    assert Boolean(id="b", prompt="p").value_of("no") is False
    assert Scale(id="s", prompt="p", low=1, high=3).value_of("2") == 2
    assert Choice(id="c", prompt="p", choices=["x", "y"]).value_of("x") == "x"


def test_options_are_rendered_in_a_canonical_order():
    """Load-bearing: the caller's order moved answers, and omitting the list put three-way questions at chance."""
    one = Choice(id="c", prompt="Who pays?", choices=["seller", "buyer", "nobody"])
    other = Choice(id="c", prompt="Who pays?", choices=["nobody", "seller", "buyer"])
    assert render_question(one) == render_question(other)
    assert "buyer, nobody, seller" in render_question(one)


def test_a_request_refuses_duplicate_question_ids():
    with pytest.raises(QuestionError, match="duplicate"):
        Request(context="c", questions=[Boolean(id="same", prompt="a"), Boolean(id="same", prompt="b")])


def test_a_request_needs_a_context_and_a_question():
    with pytest.raises(QuestionError, match="context"):
        Request(context="  ", questions=[Boolean(id="x", prompt="p")])
    with pytest.raises(QuestionError, match="at least one question"):
        Request(context="c", questions=[])


def test_the_scoring_version_is_an_integer_callers_can_store():
    assert isinstance(SCORING_VERSION, int)


def test_one_string_is_not_a_list_of_choices():
    """`choices="yes"` would quietly become three options, because a string is a sequence of characters."""
    with pytest.raises(QuestionError, match="one string as its choices"):
        Choice(id="c", prompt="Yes?", choices="yes")


def test_an_empty_option_is_refused():
    with pytest.raises(QuestionError, match="empty or non-text option"):
        Choice(id="c", prompt="Which?", choices=["yes", "  "])


def test_a_scale_is_refused_before_its_range_is_built():
    """Checked on the span, not the list: a scale of a billion would otherwise materialise before being refused."""
    with pytest.raises(QuestionError, match="spans"):
        Scale(id="s", prompt="How much?", low=0, high=1_000_000)


def test_a_result_behaves_as_a_mapping():
    """Iteration yields ids, as a mapping's does. It used to yield answers, which broke `dict(result)` and every piece
    of code that treats a mapping as one."""
    answer = Answer(id="a", kind="boolean", value=True, option="yes", probabilities={"yes": 0.8, "no": 0.2})
    result = Result(answers={"a": answer}, timing=Timing(), model="m")

    assert list(result) == ["a"]
    assert list(result.values()) == [answer]
    assert dict(result) == {"a": answer}
    assert "a" in result and "b" not in result
    assert result["a"] is answer
    assert len(result) == 1
