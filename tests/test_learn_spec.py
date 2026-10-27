"""`prismyra.learn.spec`: the registered-tag file format, checked in full before anything else trusts it."""

from __future__ import annotations

import json

import pytest

from prismyra.learn.spec import SpecError, load_spec

ENTRY = {
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


def test_a_valid_entry_loads(tmp_path):
    spec = load_spec(write(tmp_path, [ENTRY]))
    assert len(spec.tags) == 1
    t = spec.tags[0]
    assert t.task == "phishing-v1"
    assert t.options == ("no", "yes")
    assert spec.by_task("phishing-v1") is t
    assert spec.by_task("nope") is None


def test_the_version_is_a_content_hash_that_changes_with_the_file(tmp_path):
    a = load_spec(write(tmp_path, [ENTRY]))
    b = load_spec(write(tmp_path, [{**ENTRY, "retain_days": 31}]))
    assert a.version != b.version
    assert len(a.version) == 16


def test_the_same_bytes_give_the_same_version(tmp_path):
    a = load_spec(write(tmp_path, [ENTRY]))
    b = load_spec(write(tmp_path, [ENTRY]))
    assert a.version == b.version


def test_not_a_list_is_refused(tmp_path):
    with pytest.raises(SpecError, match="JSON list"):
        load_spec(write(tmp_path, ENTRY))


def test_not_json_is_refused(tmp_path):
    path = tmp_path / "learn.json"
    path.write_text("{not json")
    with pytest.raises(SpecError, match="not valid JSON"):
        load_spec(path)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(SpecError, match="could not read"):
        load_spec(tmp_path / "nope.json")


@pytest.mark.parametrize("missing_field", list(ENTRY.keys()))
def test_a_missing_field_is_refused(tmp_path, missing_field):
    broken = {k: v for k, v in ENTRY.items() if k != missing_field}
    with pytest.raises(SpecError, match="missing field"):
        load_spec(write(tmp_path, [broken]))


def test_an_unknown_field_is_refused(tmp_path):
    with pytest.raises(SpecError, match="unknown field"):
        load_spec(write(tmp_path, [{**ENTRY, "surprise": 1}]))


def test_an_empty_task_is_refused(tmp_path):
    with pytest.raises(SpecError, match="'task'"):
        load_spec(write(tmp_path, [{**ENTRY, "task": ""}]))


def test_one_option_is_refused(tmp_path):
    with pytest.raises(SpecError, match="'options'"):
        load_spec(write(tmp_path, [{**ENTRY, "options": ["only"]}]))


def test_a_repeated_option_is_refused(tmp_path):
    with pytest.raises(SpecError, match="repeats"):
        load_spec(write(tmp_path, [{**ENTRY, "options": ["yes", "yes"]}]))


def test_an_unknown_normalize_rule_is_refused(tmp_path):
    with pytest.raises(SpecError, match="'normalize'"):
        load_spec(write(tmp_path, [{**ENTRY, "normalize": "trim"}]))


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "30"])
def test_a_bad_retain_days_is_refused(tmp_path, bad):
    with pytest.raises(SpecError, match="'retain_days'"):
        load_spec(write(tmp_path, [{**ENTRY, "retain_days": bad}]))


def test_a_non_boolean_keep_hidden_is_refused(tmp_path):
    with pytest.raises(SpecError, match="'keep_hidden'"):
        load_spec(write(tmp_path, [{**ENTRY, "keep_hidden": "false"}]))


def test_an_empty_eval_set_is_refused(tmp_path):
    with pytest.raises(SpecError, match="'eval_set'"):
        load_spec(write(tmp_path, [{**ENTRY, "eval_set": ""}]))


def test_two_entries_with_the_same_task_are_refused(tmp_path):
    with pytest.raises(SpecError, match="already used"):
        load_spec(write(tmp_path, [ENTRY, {**ENTRY, "question": "Different wording?"}]))


def test_two_entries_with_the_same_normalized_shape_are_refused(tmp_path):
    other = {**ENTRY, "task": "phishing-v2", "question": ENTRY["question"].upper()}
    with pytest.raises(SpecError, match="same \\(question, option set, kind\\)"):
        load_spec(write(tmp_path, [ENTRY, other]))


def test_an_empty_spec_loads_with_no_tags(tmp_path):
    spec = load_spec(write(tmp_path, []))
    assert spec.tags == ()


# --------------------------------------------------------------------------- the optional "features" field


def test_an_entry_without_features_defaults_to_none(tmp_path):
    spec = load_spec(write(tmp_path, [ENTRY]))
    assert spec.tags[0].features is None


def test_an_entry_with_a_features_name_keeps_it(tmp_path):
    spec = load_spec(write(tmp_path, [{**ENTRY, "features": "snake_board_v1"}]))
    assert spec.tags[0].features == "snake_board_v1"


def test_an_empty_features_name_is_refused(tmp_path):
    with pytest.raises(SpecError, match="'features'"):
        load_spec(write(tmp_path, [{**ENTRY, "features": ""}]))


def test_a_non_string_features_name_is_refused(tmp_path):
    with pytest.raises(SpecError, match="'features'"):
        load_spec(write(tmp_path, [{**ENTRY, "features": 1}]))


# --------------------------------------------------------------------------- the optional admission-condition
# overrides, "min_match_rate" and "min_accuracy_ratio" (2026-10-11: student admission loosened to 0.90/0.90)


def test_an_entry_without_admission_overrides_gets_the_module_defaults(tmp_path):
    import prismyra.learn.spec as spec_module

    spec = load_spec(write(tmp_path, [ENTRY]))
    t = spec.tags[0]
    assert t.min_match_rate == spec_module.DEFAULT_MIN_MATCH_RATE
    assert t.min_accuracy_ratio == spec_module.DEFAULT_MIN_ACCURACY_RATIO
    # The defaults themselves are the single place these numbers are decided -- pin their value so a change
    # to spec.py's constants is a deliberate edit here too, not a silent drift this test stops noticing.
    assert spec_module.DEFAULT_MIN_MATCH_RATE == 0.90
    assert spec_module.DEFAULT_MIN_ACCURACY_RATIO == 0.90


def test_an_entry_can_override_min_match_rate_independently(tmp_path):
    spec = load_spec(write(tmp_path, [{**ENTRY, "min_match_rate": 0.97}]))
    t = spec.tags[0]
    assert t.min_match_rate == 0.97
    # Overriding one does not change the other's default.
    import prismyra.learn.spec as spec_module

    assert t.min_accuracy_ratio == spec_module.DEFAULT_MIN_ACCURACY_RATIO


def test_an_entry_can_override_min_accuracy_ratio_independently(tmp_path):
    spec = load_spec(write(tmp_path, [{**ENTRY, "min_accuracy_ratio": 1.0}]))
    t = spec.tags[0]
    assert t.min_accuracy_ratio == 1.0
    # Overriding one does not change the other's default.
    import prismyra.learn.spec as spec_module

    assert t.min_match_rate == spec_module.DEFAULT_MIN_MATCH_RATE


def test_an_entry_can_override_both_admission_fields_at_once(tmp_path):
    spec = load_spec(write(tmp_path, [{**ENTRY, "min_match_rate": 0.95, "min_accuracy_ratio": 0.80}]))
    t = spec.tags[0]
    assert t.min_match_rate == 0.95
    assert t.min_accuracy_ratio == 0.80


@pytest.mark.parametrize("bad", [0, -0.1, 1.1, True, "0.9"])
def test_a_bad_min_match_rate_is_refused(tmp_path, bad):
    with pytest.raises(SpecError, match="'min_match_rate'"):
        load_spec(write(tmp_path, [{**ENTRY, "min_match_rate": bad}]))


@pytest.mark.parametrize("bad", [0, -0.1, 1.1, True, "0.9"])
def test_a_bad_min_accuracy_ratio_is_refused(tmp_path, bad):
    with pytest.raises(SpecError, match="'min_accuracy_ratio'"):
        load_spec(write(tmp_path, [{**ENTRY, "min_accuracy_ratio": bad}]))


def test_min_match_rate_of_exactly_one_is_allowed(tmp_path):
    spec = load_spec(write(tmp_path, [{**ENTRY, "min_match_rate": 1.0}]))
    assert spec.tags[0].min_match_rate == 1.0


# --------------------------------------------------------------------------- the optional "student" field


def test_an_entry_without_student_defaults_to_lightgbm(tmp_path):
    spec = load_spec(write(tmp_path, [ENTRY]))
    assert spec.tags[0].student == "lightgbm"


def test_an_entry_can_opt_into_xgboost(tmp_path):
    spec = load_spec(write(tmp_path, [{**ENTRY, "student": "xgboost"}]))
    assert spec.tags[0].student == "xgboost"


def test_an_unknown_student_backend_is_refused(tmp_path):
    with pytest.raises(SpecError, match="'student'"):
        load_spec(write(tmp_path, [{**ENTRY, "student": "catboost"}]))


def test_an_empty_student_backend_is_refused(tmp_path):
    with pytest.raises(SpecError, match="'student'"):
        load_spec(write(tmp_path, [{**ENTRY, "student": ""}]))
