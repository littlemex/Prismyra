"""`prismyra.learn.students`: loading a real, admitted `prismyra.learn.fit.StudentArtifact` and predicting
from it -- the one place in stage 4 that needs the `learn` extra (LightGBM). Everything else stage 4 adds
(`prismyra.learn.router`, `.controller`, and `prismyra.learn.hook.LearnHook`'s own wiring) is exercised with a
fake student elsewhere (`tests/test_learn_hook_router.py`, `tests/test_learn_server_router.py`) precisely so
those files do not need this import. CI's own "test" job does not install this extra (`pyproject.toml`'s
`dev` extra omits it on purpose -- see that file's own comment); `pytest.importorskip` below is what makes
this file skip there rather than error, the same way `tests/test_learn_fit.py` already does.
"""

from __future__ import annotations

import json

import pytest

from prismyra.learn.fit import run
from prismyra.learn.spec import load_spec

lightgbm = pytest.importorskip("lightgbm")

from prismyra.learn.students import StudentError, StudentRegistry, student_version  # noqa: E402

TASK = "sentiment-v1"
ENTRY = {
    "task": TASK,
    "question": "Is this tweet positive?",
    "options": ["negative", "positive"],
    "kind": "noul",
    "normalize": "strip+lower",
    "retain_days": 30,
    "keep_hidden": False,
    "eval_set": "eval.jsonl",
    "min_match_rate": 0.5,
    "min_accuracy_ratio": 0.5,  # generous: this fixture is about loading an artifact, not distillation quality
}

POSITIVE_WORDS = ["love", "great", "amazing", "wonderful", "fantastic", "awesome", "happy", "best"]
NEGATIVE_WORDS = ["hate", "terrible", "awful", "worst", "bad", "sad", "horrible", "angry"]


def synthetic_rows(n_per_class=40):
    rows = []
    idx = 0
    for i in range(n_per_class):
        rows.append((f"this is {POSITIVE_WORDS[i % len(POSITIVE_WORDS)]} news item {i}", 0.95, idx))
        idx += 1
        rows.append((f"this is {NEGATIVE_WORDS[i % len(NEGATIVE_WORDS)]} news item {i}", 0.05, idx))
        idx += 1
    return rows


def _write_experience(tmp_path, spec, rows):
    tag_dir = tmp_path / "experience" / TASK
    tag_dir.mkdir(parents=True)
    with (tag_dir / "2026-01-01.jsonl").open("w") as fh:
        for context, p_positive, idx in rows:
            fh.write(
                json.dumps(
                    {
                        "tag": TASK,
                        "spec_version": spec.version,
                        "ts": 1_700_000_000.0 + idx,
                        "context": context,
                        "question": ENTRY["question"],
                        "options": ["negative", "positive"],
                        "kind": "noul",
                        "probabilities": {"negative": 1 - p_positive, "positive": p_positive},
                        "answered_by": "prismyra",
                        "versions": {"package": "0", "backbone": "stub", "spec": spec.version},
                    }
                )
                + "\n"
            )


def _write_eval_set(path, rows):
    with path.open("w") as fh:
        for context, p_positive, _ in rows:
            gold = "positive" if p_positive > 0.5 else "negative"
            fh.write(
                json.dumps(
                    {
                        "context": context,
                        "gold": gold,
                        "options": ["negative", "positive"],
                        "prismyra_probabilities": {"negative": 1 - p_positive, "positive": p_positive},
                    }
                )
                + "\n"
            )


def fit_and_save(tmp_path, out_name="student.json", *, entry_overrides=None, backend_name=None):
    spec_path = tmp_path / "learn.json"
    spec_path.write_text(json.dumps([{**ENTRY, **(entry_overrides or {})}]))
    spec = load_spec(spec_path)
    _write_experience(tmp_path, spec, synthetic_rows(60))
    _write_eval_set(tmp_path / "eval.jsonl", synthetic_rows(15))

    artifact = run(
        spec_path=spec_path,
        task=TASK,
        log_dir=None,
        package_version="0.0.0-test",
        backbone="stub/model",
        dims=128,
        held_out_fraction=0.25,
        num_threads=1,
        seed=0,
        num_boost_round=30,
        min_data_in_leaf=2,
        backend_name=backend_name,
    )
    assert artifact.admitted, artifact.reasons
    out_path = tmp_path / out_name
    out_path.write_text(json.dumps(artifact.to_json()))
    return spec_path, out_path, artifact


def test_student_registry_loads_an_admitted_artifact_and_predicts(tmp_path):
    spec_path, out_path, artifact = fit_and_save(tmp_path)
    registry = StudentRegistry()
    handle = registry.load(TASK, spec_path.parent, out_path.name)
    assert handle.version == student_version(artifact)
    assert registry.get(TASK) is handle

    # A context pulled straight out of the training rows (not a novel phrase): with only 60 rows and a
    # hashed, order-independent feature space, a *novel* sentence can land on noise even when the artifact
    # passed admission in aggregate; a row the booster actually saw is what isolates "does loading and
    # predicting reproduce fit.py's own numbers" from "how well did this toy fixture generalise".
    positive_context = synthetic_rows(1)[0][0]
    probs = handle.predict(positive_context)
    assert set(probs) == {"negative", "positive"}
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs["positive"] > probs["negative"]


def test_predicting_the_same_input_twice_gives_the_same_answer(tmp_path):
    spec_path, out_path, _ = fit_and_save(tmp_path)
    handle = StudentRegistry().load(TASK, spec_path.parent, out_path.name)
    a = handle.predict("this is great news item 1")
    b = handle.predict("this is great news item 1")
    assert a == b


def test_a_not_admitted_artifact_is_refused_at_load_time(tmp_path):
    spec_path, out_path, artifact = fit_and_save(tmp_path)
    broken = {**artifact.to_json(), "admitted": False, "reasons": ["eval_set: student accuracy too low"]}
    out_path.write_text(json.dumps(broken))
    with pytest.raises(StudentError, match="not admitted"):
        StudentRegistry().load(TASK, spec_path.parent, out_path.name)


def test_an_artifact_for_a_different_tag_is_refused(tmp_path):
    spec_path, out_path, artifact = fit_and_save(tmp_path)
    mismatched = {**artifact.to_json(), "tag": "a-different-tag"}
    out_path.write_text(json.dumps(mismatched))
    with pytest.raises(StudentError, match="is an artifact for tag"):
        StudentRegistry().load(TASK, spec_path.parent, out_path.name)


def test_a_missing_student_path_is_refused(tmp_path):
    spec_path, _out_path, _artifact = fit_and_save(tmp_path)
    with pytest.raises(StudentError, match="does not exist"):
        StudentRegistry().load(TASK, spec_path.parent, "nope.json")


def test_student_version_changes_when_model_text_or_spec_version_changes(tmp_path):
    _spec_path, _out_path, artifact = fit_and_save(tmp_path)
    v1 = student_version(artifact)
    from dataclasses import replace

    v2 = student_version(replace(artifact, spec_version="a-different-spec-version"))
    assert v1 != v2
    # Same artifact, computed twice -- the join key `prismyra.learn.router.decide`'s `served_by` depends on
    # being stable across a process restart, not just within one.
    assert v1 == student_version(artifact)


# --------------------------------------------------------------------------- the "xgboost" student backend


xgboost = pytest.importorskip("xgboost")


def test_student_registry_loads_an_xgboost_backed_artifact_and_predicts(tmp_path):
    spec_path, out_path, artifact = fit_and_save(tmp_path, backend_name="xgboost")
    assert artifact.backend == "xgboost"
    registry = StudentRegistry()
    handle = registry.load(TASK, spec_path.parent, out_path.name)
    assert handle.version == student_version(artifact)

    positive_context = synthetic_rows(1)[0][0]
    probs = handle.predict(positive_context)
    assert set(probs) == {"negative", "positive"}
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs["positive"] > probs["negative"]


def test_xgboost_backed_prediction_is_the_same_across_two_independently_loaded_handles(tmp_path):
    # Not the same StudentHandle instance predicting twice (that would only show the booster itself is a
    # pure function) -- two separate StudentHandle constructions from the same on-disk artifact, the shape
    # a server restart actually takes: load the artifact fresh, build a fresh xgb.Booster from its model_text,
    # predict. See test_xgboost_backend_survives_a_real_process_restart (test_learn_backends.py) for the same
    # claim through an actual subprocess, not just a second in-process object.
    spec_path, out_path, _artifact = fit_and_save(tmp_path, backend_name="xgboost")
    handle_a = StudentRegistry().load(TASK, spec_path.parent, out_path.name)
    handle_b = StudentRegistry().load(TASK, spec_path.parent, out_path.name)
    context = synthetic_rows(1)[0][0]
    assert handle_a.predict(context) == handle_b.predict(context)


def test_student_version_differs_between_backends_on_the_same_tag(tmp_path):
    lgb_dir = tmp_path / "lgb"
    xgb_dir = tmp_path / "xgb"
    lgb_dir.mkdir()
    xgb_dir.mkdir()
    _spec_path, _out_path, lgb_artifact = fit_and_save(lgb_dir, out_name="lgb.json")
    _spec_path2, _out_path2, xgb_artifact = fit_and_save(xgb_dir, out_name="xgb.json", backend_name="xgboost")
    assert lgb_artifact.backend == "lightgbm"
    assert xgb_artifact.backend == "xgboost"
    assert student_version(lgb_artifact) != student_version(xgb_artifact)


def test_a_tags_own_spec_student_field_is_honoured_without_a_registry_level_override(tmp_path):
    spec_path, out_path, artifact = fit_and_save(tmp_path, entry_overrides={"student": "xgboost"})
    assert artifact.backend == "xgboost"
    handle = StudentRegistry().load(TASK, spec_path.parent, out_path.name)
    context = synthetic_rows(1)[0][0]
    probs = handle.predict(context)
    assert set(probs) == {"negative", "positive"}
