"""`prismyra.learn.spec`'s stage 4 fields: `"router"` and `"retain_context"`. Companion to
`tests/test_learn_spec.py`, which covers everything stage 1/2 already checked -- kept separate so a reader
looking for "what does a router block have to look like" does not have to wade through stage 1/2's own cases.
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


def test_a_tag_with_no_router_field_has_router_none(tmp_path):
    spec = load_spec(write(tmp_path, [BASE_ENTRY]))
    assert spec.tags[0].router is None
    assert spec.tags[0].retain_context == "full"


def test_a_minimal_ab_router_loads_with_its_own_defaults(tmp_path):
    entry = {**BASE_ENTRY, "router": {"mode": "ab", "student_path": "students/phishing-v1.json", "ratio": 2500}}
    spec = load_spec(write(tmp_path, [entry]))
    router = spec.tags[0].router
    assert router is not None
    assert router.mode == "ab"
    assert router.ratio == 2500
    assert router.ramp_step == 500
    assert router.rollback_min_match_rate == 0.85
    assert router.bandit_arms == ()


def test_confidence_mode_requires_tau(tmp_path):
    entry = {**BASE_ENTRY, "router": {"mode": "confidence", "student_path": "s.json"}}
    with pytest.raises(SpecError, match="requires 'tau'"):
        load_spec(write(tmp_path, [entry]))


def test_confidence_mode_with_tau_loads(tmp_path):
    entry = {**BASE_ENTRY, "router": {"mode": "confidence", "student_path": "s.json", "tau": 0.9}}
    spec = load_spec(write(tmp_path, [entry]))
    assert spec.tags[0].router.tau == 0.9


def test_bandit_mode_requires_bandit_arms(tmp_path):
    entry = {**BASE_ENTRY, "router": {"mode": "bandit", "student_path": "s.json"}}
    with pytest.raises(SpecError, match="requires a non-empty 'bandit_arms'"):
        load_spec(write(tmp_path, [entry]))


def test_bandit_mode_with_arms_loads(tmp_path):
    entry = {
        **BASE_ENTRY,
        "router": {"mode": "bandit", "student_path": "s.json", "bandit_arms": [0, 2500, 5000]},
    }
    spec = load_spec(write(tmp_path, [entry]))
    assert spec.tags[0].router.bandit_arms == (0, 2500, 5000)


def test_an_unknown_mode_is_refused(tmp_path):
    entry = {**BASE_ENTRY, "router": {"mode": "greedy", "student_path": "s.json"}}
    with pytest.raises(SpecError, match="'mode' must be one of"):
        load_spec(write(tmp_path, [entry]))


def test_router_missing_student_path_is_refused(tmp_path):
    entry = {**BASE_ENTRY, "router": {"mode": "ab"}}
    with pytest.raises(SpecError, match="missing field"):
        load_spec(write(tmp_path, [entry]))


def test_router_unknown_field_is_refused(tmp_path):
    entry = {**BASE_ENTRY, "router": {"mode": "ab", "student_path": "s.json", "typo_field": 1}}
    with pytest.raises(SpecError, match="unknown field"):
        load_spec(write(tmp_path, [entry]))


def test_ratio_out_of_range_is_refused(tmp_path):
    entry = {**BASE_ENTRY, "router": {"mode": "ab", "student_path": "s.json", "ratio": 10_001}}
    with pytest.raises(SpecError, match="'ratio'"):
        load_spec(write(tmp_path, [entry]))


def test_ramp_min_ratio_above_max_is_refused(tmp_path):
    entry = {
        **BASE_ENTRY,
        "router": {"mode": "ramp", "student_path": "s.json", "ramp_min_ratio": 5000, "ramp_max_ratio": 1000},
    }
    with pytest.raises(SpecError, match="exceeds"):
        load_spec(write(tmp_path, [entry]))


def test_audit_min_fraction_above_max_is_refused(tmp_path):
    entry = {
        **BASE_ENTRY,
        "router": {"mode": "ab", "student_path": "s.json", "audit_min_fraction": 3000, "audit_max_fraction": 100},
    }
    with pytest.raises(SpecError, match="exceeds"):
        load_spec(write(tmp_path, [entry]))


def test_tau_out_of_range_is_refused(tmp_path):
    entry = {**BASE_ENTRY, "router": {"mode": "confidence", "student_path": "s.json", "tau": 1.5}}
    with pytest.raises(SpecError, match="'tau'"):
        load_spec(write(tmp_path, [entry]))


@pytest.mark.parametrize("value", ["full", "features_only", "none"])
def test_retain_context_accepts_every_rule(tmp_path, value):
    entry = {**BASE_ENTRY, "retain_context": value}
    spec = load_spec(write(tmp_path, [entry]))
    assert spec.tags[0].retain_context == value


def test_retain_context_rejects_an_unknown_value(tmp_path):
    entry = {**BASE_ENTRY, "retain_context": "redacted"}
    with pytest.raises(SpecError, match="'retain_context'"):
        load_spec(write(tmp_path, [entry]))
