"""`prismyra.learn.spec`'s `"admission"` field (PREREG-R2 idea 1: "admission for what is served"). Companion
to `tests/test_learn_spec_router.py`, which covers `"router"` -- kept separate for the same reason that file
gives for being its own file rather than folded into `tests/test_learn_spec.py`.
"""

from __future__ import annotations

import json

import pytest

from prismyra.learn.spec import SpecError, load_spec

BASE_ENTRY = {
    "task": "phishing-v1",
    "question": "Is this email a phishing attempt?",
    "options": ["no", "yes"],
    "kind": "noul",
    "normalize": "strip+lower",
    "retain_days": 30,
    "keep_hidden": False,
    "eval_set": "evals/phishing-v1-gold.jsonl",
}


def write(tmp_path, entries):
    path = tmp_path / "learn.json"
    path.write_text(json.dumps(entries))
    return path


def test_a_tag_with_no_admission_field_has_admission_none(tmp_path):
    spec = load_spec(write(tmp_path, [BASE_ENTRY]))
    assert spec.tags[0].admission is None


def test_a_confidence_admission_entry_loads(tmp_path):
    entry = {**BASE_ENTRY, "admission": {"mode": "confidence", "tau": 0.8}}
    spec = load_spec(write(tmp_path, [entry]))
    admission = spec.tags[0].admission
    assert admission is not None
    assert admission.mode == "confidence"
    assert admission.tau == 0.8


@pytest.mark.parametrize("tau", [0.0, 1.0])
def test_tau_accepts_its_own_boundaries(tmp_path, tau):
    entry = {**BASE_ENTRY, "admission": {"mode": "confidence", "tau": tau}}
    spec = load_spec(write(tmp_path, [entry]))
    assert spec.tags[0].admission.tau == tau


def test_an_unknown_admission_mode_is_refused(tmp_path):
    entry = {**BASE_ENTRY, "admission": {"mode": "greedy", "tau": 0.8}}
    with pytest.raises(SpecError, match="'mode' must be one of"):
        load_spec(write(tmp_path, [entry]))


def test_admission_missing_tau_is_refused(tmp_path):
    entry = {**BASE_ENTRY, "admission": {"mode": "confidence"}}
    with pytest.raises(SpecError, match="missing field"):
        load_spec(write(tmp_path, [entry]))


def test_admission_missing_mode_is_refused(tmp_path):
    entry = {**BASE_ENTRY, "admission": {"tau": 0.8}}
    with pytest.raises(SpecError, match="missing field"):
        load_spec(write(tmp_path, [entry]))


def test_admission_unknown_field_is_refused(tmp_path):
    entry = {**BASE_ENTRY, "admission": {"mode": "confidence", "tau": 0.8, "typo_field": 1}}
    with pytest.raises(SpecError, match="unknown field"):
        load_spec(write(tmp_path, [entry]))


@pytest.mark.parametrize("tau", [-0.1, 1.1])
def test_tau_out_of_range_is_refused(tmp_path, tau):
    entry = {**BASE_ENTRY, "admission": {"mode": "confidence", "tau": tau}}
    with pytest.raises(SpecError, match="'tau'"):
        load_spec(write(tmp_path, [entry]))


def test_admission_not_an_object_is_refused(tmp_path):
    entry = {**BASE_ENTRY, "admission": "confidence"}
    with pytest.raises(SpecError, match="must be a JSON object"):
        load_spec(write(tmp_path, [entry]))


def test_admission_is_independent_of_router(tmp_path):
    # Idea 1's own point: a tag can declare its admission policy, and separately, a router that serves it --
    # the two fields do not have to agree on `tau` (an operator may admit at one tau and route at a stricter
    # one later, see prismyra.learn.students.StudentHandle for the check that actually enforces that).
    entry = {
        **BASE_ENTRY,
        "admission": {"mode": "confidence", "tau": 0.8},
        "router": {"mode": "confidence", "student_path": "s.json", "tau": 0.9},
    }
    spec = load_spec(write(tmp_path, [entry]))
    assert spec.tags[0].admission.tau == 0.8
    assert spec.tags[0].router.tau == 0.9
