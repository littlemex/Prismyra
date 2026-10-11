"""`prismyra.learn.tagging`: a request is tagged by name or by exact shape, or it is not tagged at all."""

from __future__ import annotations

import json

import pytest

from prismyra.learn.spec import load_spec
from prismyra.learn.tagging import normalize, tag_for

PHISHING = {
    "task": "phishing-v1",
    "question": "Is this email a phishing attempt?",
    "options": ["no", "yes"],
    "kind": "noul",
    "normalize": "strip+lower",
    "retain_days": 30,
    "keep_hidden": False,
    "eval_set": "evals/phishing-v1-gold.jsonl",
}
PRIORITY = {
    "task": "ticket-priority-v1",
    "question": "How urgent is this ticket?",
    "options": ["low", "normal", "urgent"],
    "kind": "choice",
    "normalize": "none",
    "retain_days": 7,
    "keep_hidden": True,
    "eval_set": "evals/priority-v1-gold.jsonl",
}


@pytest.fixture
def spec(tmp_path):
    path = tmp_path / "learn.json"
    path.write_text(json.dumps([PHISHING, PRIORITY]))
    return load_spec(path)


def test_normalize_rules():
    assert normalize("  Hi There  ", "none") == "  Hi There  "
    assert normalize("  Hi There  ", "strip") == "Hi There"
    assert normalize("  Hi There  ", "lower") == "  hi there  "
    assert normalize("  Hi There  ", "strip+lower") == "hi there"


def test_an_exact_shape_match_is_tagged(spec):
    tag = tag_for(
        spec,
        task=None,
        question="Is this email a phishing attempt?",
        options=["yes", "no"],  # order does not matter, same as heads.py's own matching rule
        kind="noul",
    )
    assert tag is not None
    assert tag.task == "phishing-v1"


def test_normalization_is_applied_before_comparing(spec):
    tag = tag_for(spec, task=None, question="  IS THIS EMAIL A PHISHING ATTEMPT?  ", options=["no", "yes"], kind="noul")
    assert tag is not None and tag.task == "phishing-v1"


def test_a_different_option_set_is_untagged(spec):
    assert tag_for(spec, task=None, question=PHISHING["question"], options=["no", "yes", "maybe"], kind="noul") is None


def test_a_different_kind_is_untagged(spec):
    assert tag_for(spec, task=None, question=PHISHING["question"], options=["no", "yes"], kind="choice") is None


def test_a_different_question_is_untagged(spec):
    assert tag_for(spec, task=None, question="Something else entirely?", options=["no", "yes"], kind="noul") is None


def test_a_document_never_changes_the_tag(spec):
    """Section 1: context does not participate in matching at all -- `tag_for` has no parameter for it."""
    a = tag_for(spec, task=None, question=PHISHING["question"], options=["no", "yes"], kind="noul")
    b = tag_for(spec, task=None, question=PHISHING["question"], options=["no", "yes"], kind="noul")
    assert a is b is not None


def test_naming_a_registered_task_tags_it(spec):
    tag = tag_for(spec, task="phishing-v1", question="anything", options=["no", "yes"], kind="noul")
    assert tag is not None and tag.task == "phishing-v1"


def test_naming_an_unregistered_task_is_untagged(spec):
    assert tag_for(spec, task="not-a-real-task", question="x", options=["no", "yes"], kind="noul") is None


def test_naming_a_task_with_the_wrong_shape_is_untagged(spec):
    """A caller cannot borrow a tag's name for a different judgment -- section 1's own "違う判断が混ざらない"."""
    assert tag_for(spec, task="phishing-v1", question="x", options=["no", "yes", "maybe"], kind="noul") is None
    assert tag_for(spec, task="phishing-v1", question="x", options=["no", "yes"], kind="choice") is None


def test_an_empty_spec_tags_nothing():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "learn.json"
        p.write_text("[]")
        empty = load_spec(p)
    assert tag_for(empty, task=None, question="x", options=["a", "b"], kind="boolean") is None
    assert tag_for(empty, task="whatever", question="x", options=["a", "b"], kind="boolean") is None
