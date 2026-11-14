"""Admission mode ``"external"``: someone outside Prismyra decides that a student may be served, and Prismyra checks
only that the decision's record matches what it is about to serve.

Written against `prismyra.learn.spec` (the mode and its one field) and `prismyra.learn.students.AdmissionRecord`
(the record's own format): a student Prismyra's own bar rejected is served under a matching record; each bound
field is refused on its own when it does not match (the tag, the artifact's version, the router block); a record
that is malformed or missing is a refusal at load, never a silent fall-back to Prismyra's own bar.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from prismyra.learn import students as studmod
from prismyra.learn.fit import StudentArtifact
from prismyra.learn.spec import AdmissionConfig, SpecError, _check_admission
from prismyra.learn.students import AdmissionRecord, StudentError, StudentRegistry, artifact_version

TASK = "sent-v1"
ROUTER = {"mode": "confidence", "student_path": "student.json", "tau": 0.9}


def _artifact(**kw) -> StudentArtifact:
    base = {
        "tag": TASK,
        "spec_version": "spec-v1",
        "package_version": "0",
        "backbone": "stub",
        "options": ("neg", "pos"),
        "kind": "choice",
        "dims": 4,
        "idf": (1.0,) * 4,
        "model_text": "m1",
        "created_at": 0.0,
        "admitted": False,
        "reasons": ("eval_set: below the bar",),
        "metrics": {},
    }
    return StudentArtifact(**{**base, **kw})


def _record(artifact, **kw) -> dict:
    base = {
        "format": 1,
        "tag": TASK,
        "version": artifact_version(artifact),
        "router": dict(ROUTER),
        "admitted_by": "an outside certifier",
        "evidence": {"uri": "policy.json", "sha256": "a" * 64},
    }
    return {**base, **kw}


class _Model:
    def predict_proba(self, rows):
        return [[0.2, 0.8] for _ in rows]


class _Backend:
    def load(self, text):
        return _Model()


@pytest.fixture
def served(tmp_path, monkeypatch):
    """Write an artifact and its record, and load them as the server would; no real booster is built."""
    monkeypatch.setattr(studmod, "get_backend", lambda name: _Backend())

    def load(artifact, record, router=ROUTER, admission=None):
        (tmp_path / "student.json").write_text(json.dumps(artifact.to_json()))
        (tmp_path / "admission.json").write_text(json.dumps(record))
        rc = studmod.parse_router(router, "spec")
        return StudentRegistry().load(
            TASK,
            tmp_path,
            "student.json",
            router=rc,
            admission=admission or AdmissionConfig(mode="external", record="admission.json"),
        )

    return load


def test_a_student_prismyras_own_bar_rejected_is_served_under_a_matching_record(served):
    a = _artifact()
    handle = served(a, _record(a))
    assert handle.record.admitted_by == "an outside certifier"
    assert handle.predict("x") == {"neg": 0.2, "pos": 0.8}


def test_a_refit_under_the_old_record_is_refused(served):
    old = _artifact()
    with pytest.raises(StudentError, match="A refit needs a new judgement"):
        served(replace(old, model_text="m2"), _record(old))


@pytest.mark.parametrize(
    "change",
    [
        {"features": "mod:other"},
        {"options": ("pos", "neg")},
        {"idf": (2.0,) * 4},
        {"dims": 5},
        {"backend": "xgboost"},
        {"spec_version": "spec-v2"},
    ],
)
def test_every_field_that_decides_the_answers_is_in_the_version(served, change):
    old = _artifact()
    with pytest.raises(StudentError, match="A refit needs a new judgement"):
        served(replace(old, **change), _record(old))


def test_what_prismyra_says_about_an_artifact_is_not_in_its_version():
    a = _artifact()
    assert artifact_version(a) == artifact_version(
        replace(a, admitted=True, reasons=(), metrics={"x": 1}, created_at=9.0)
    )


def test_a_record_for_another_tag_is_refused(served):
    a = _artifact()
    with pytest.raises(StudentError, match="for tag 'other'"):
        served(a, _record(a, tag="other"))


@pytest.mark.parametrize(
    "router",
    [{**ROUTER, "tau": 0.8}, {**ROUTER, "tau": 0.95}, {"mode": "ab", "student_path": "student.json", "ratio": 10_000}],
)
def test_a_live_router_other_than_the_admitted_one_is_refused(served, router):
    a = _artifact()
    with pytest.raises(StudentError, match="not the one the admission record declares"):
        served(a, _record(a), router=router)


@pytest.mark.parametrize(
    "change,why",
    [
        ({"format": 2}, "'format'"),
        ({"format": True}, "'format'"),
        ({"format": 1.0}, "'format'"),
        ({"version": ""}, "'version' must be"),
        ({"versions": ["x"]}, "keys must be exactly"),
        ({"evidence": {"uri": "p", "sha256": "A" * 64}}, "'evidence'"),
        ({"evidence": {"uri": "p", "sha256": "a" * 64 + "\n"}}, "'evidence'"),
        ({"evidence": {"uri": "p"}}, "'evidence'"),
        ({"admitted_by": ""}, "'admitted_by' must be"),
        ({"router": {"mode": "nope"}}, "its router block"),
        ({"extra": 1}, "keys must be exactly"),
    ],
)
def test_a_malformed_record_is_refused(served, change, why):
    a = _artifact()
    with pytest.raises(StudentError, match=why):
        served(a, {k: v for k, v in _record(a, **change).items() if not (k == "version" and "versions" in change)})


def test_a_missing_record_is_refused_not_ignored(served, tmp_path):
    a = _artifact()
    with pytest.raises(StudentError, match=r"nowhere\.json: No such file"):
        served(a, _record(a), admission=AdmissionConfig(mode="external", record="nowhere.json"))


def test_without_external_admission_prismyras_own_bar_still_decides(served):
    a = _artifact()
    with pytest.raises(StudentError, match="not admitted"):
        served(a, _record(a), admission=AdmissionConfig(mode="confidence", tau=0.9))


def test_the_spec_field_takes_a_record_and_nothing_else(tmp_path):
    assert _check_admission(tmp_path, "e", {"mode": "external", "record": "r.json"}) == AdmissionConfig(
        mode="external", record="r.json"
    )
    for bad in (
        {"mode": "external"},
        {"mode": "external", "record": "r", "tau": 0.9},
        {"mode": "external", "record": ""},
    ):
        with pytest.raises(SpecError):
            _check_admission(tmp_path, "e", bad)
    for bad in ({"mode": "external"}, {"mode": "external", "record": "r", "tau": 0.9}, {"mode": "confidence"}):
        with pytest.raises(ValueError):
            AdmissionConfig(**bad)


def test_a_record_an_outside_writer_produced_loads(tmp_path):
    """The contract test with no import: a record exactly as tierbook's `export-policy --target prismyra-learn`
    writes it (copied from its output), around a real artifact version."""
    a = _artifact()
    raw = json.loads((Path(__file__).parent / "fixtures" / "admission-record-external-writer.json").read_text())
    raw["version"] = artifact_version(a)
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    rec = AdmissionRecord.load(p)
    rec.check(a, studmod.parse_router(raw["router"], "spec"))


def test_a_spec_with_external_admission_is_served_by_the_hook(tmp_path, monkeypatch):
    from prismyra.learn.hook import LearnHook

    monkeypatch.setattr(studmod, "get_backend", lambda name: _Backend())
    a = _artifact(tag=TASK)
    entry = {
        "task": TASK,
        "question": "q",
        "options": ["neg", "pos"],
        "kind": "choice",
        "normalize": "none",
        "retain_days": 30,
        "keep_hidden": False,
        "eval_set": "e.jsonl",
        "router": dict(ROUTER),
        "admission": {"mode": "external", "record": "admission.json"},
    }
    (tmp_path / "learn.json").write_text(json.dumps([entry]))
    (tmp_path / "student.json").write_text(json.dumps(a.to_json()))
    rec = _record(a)
    (tmp_path / "admission.json").write_text(json.dumps(rec))
    hook = LearnHook(tmp_path / "learn.json", package_version="9", backbone="stub", log_dir=tmp_path / "log")
    assert hook.reload_student(TASK).record.version == rec["version"]
    (tmp_path / "student.json").write_text(json.dumps(replace(a, model_text="refit").to_json()))
    with pytest.raises(StudentError, match="A refit needs a new judgement"):
        hook.reload_student(TASK)
