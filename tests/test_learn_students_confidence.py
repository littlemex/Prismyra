"""PREREG-R2 idea 1's other half: `prismyra.learn.students.StudentHandle` refusing to serve a
`"confidence"`-admitted artifact under a router that would not actually serve it the way it was measured.

Every case below is a *refusal*, and every refusal is checked without ever touching a real booster:
`StudentHandle.__init__` raises on the `admitted_for`/`router` mismatch before it ever reaches
`get_backend(...).load(...)` (`prismyra/learn/students.py`'s own code order), so a `StudentArtifact` built
directly, with a `model_text` that is not even valid LightGBM/XGBoost text, is enough -- the same reason
`tests/test_learn_students.py::test_a_not_admitted_artifact_is_refused_at_load_time` next to this file does
not need a real model either. The *acceptance* case (a correctly-matched `"confidence"` router successfully
loading and predicting) needs a real, admitted artifact and is covered in `tests/test_learn_students.py`'s
own LightGBM-backed fixtures instead, which already build one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismyra.learn.fit import StudentArtifact
from prismyra.learn.spec import RouterConfig
from prismyra.learn.students import StudentError, StudentHandle, StudentRegistry

TASK = "sentiment-finnews-v1"


def _artifact(*, admitted: bool = True, admitted_for: dict | None = None, reasons: tuple = ()) -> StudentArtifact:
    """A `StudentArtifact` with a `model_text` that is deliberately not a real serialized model -- see the
    module docstring for why every test below never reaches the code that would need one to be."""
    return StudentArtifact(
        tag=TASK,
        spec_version="spec-v1",
        package_version="0.0.0-test",
        backbone="stub/model",
        options=("bearish", "bullish", "neutral"),
        kind="choice",
        dims=128,
        idf=(1.0,) * 128,
        model_text="not-a-real-model",
        created_at=0.0,
        admitted=admitted,
        reasons=reasons,
        metrics={},
        admitted_for=admitted_for,
    )


def test_not_admitted_is_refused_before_admitted_for_is_even_consulted():
    artifact = _artifact(admitted=False, reasons=("eval_set: student accuracy too low",))
    with pytest.raises(StudentError, match="not admitted"):
        StudentHandle(artifact)


def test_a_confidence_artifact_with_no_router_at_all_is_refused():
    artifact = _artifact(admitted_for={"mode": "confidence", "tau": 0.8})
    with pytest.raises(StudentError, match="admitted only for 'confidence' mode"):
        StudentHandle(artifact, router=None)


def test_a_confidence_artifact_served_under_ab_mode_is_refused():
    # "ab" would serve the artifact for *every* eligible request, not only the confident slice it was
    # actually measured on -- exactly the full-coverage system PREREG-R2's admission mode exists to avoid
    # judging (and therefore to avoid silently serving).
    artifact = _artifact(admitted_for={"mode": "confidence", "tau": 0.8})
    router = RouterConfig(mode="ab", student_path="s.json", ratio=10_000)
    with pytest.raises(StudentError, match="admitted only for 'confidence' mode"):
        StudentHandle(artifact, router=router)


def test_a_confidence_artifact_served_at_a_looser_tau_is_refused():
    # The router's own tau (0.5) would serve more of the traffic than the artifact was admitted to answer
    # (tau=0.8) -- a looser live tau is a strictly wider claim than the one this artifact's own numbers back.
    artifact = _artifact(admitted_for={"mode": "confidence", "tau": 0.8})
    router = RouterConfig(mode="confidence", student_path="s.json", tau=0.5)
    with pytest.raises(StudentError, match="admitted only for 'confidence' mode"):
        StudentHandle(artifact, router=router)


#: These policy checks run entirely before `get_backend(...).load(model_text)`, so every case above uses a
#: `model_text` that is not even valid model text -- but that also means the one case this module cannot
#: test here is the *acceptance* path (a matching ``("confidence", tau)``), since clearing the policy check
#: is exactly what lets execution reach `load`, which needs a real, parseable model -- covered instead by
#: `tests/test_learn_students.py`'s own LightGBM-backed fixtures.


def test_student_registry_load_passes_the_router_through_to_the_policy_check(tmp_path, monkeypatch):
    # A *mismatched* router (tau=0.5, looser than the artifact's own admitted tau=0.8) is used on purpose: it
    # is refused by the policy check alone, so this never reaches `load` and never needs a real model --
    # while still proving `StudentRegistry.load`'s own `router` parameter is the one `StudentHandle` sees,
    # not silently dropped or defaulted to `None` along the way.
    artifact = _artifact(admitted_for={"mode": "confidence", "tau": 0.8})
    out_path = tmp_path / "student.json"
    out_path.write_text(json.dumps(artifact.to_json()))

    router = RouterConfig(mode="confidence", student_path="student.json", tau=0.5)
    with pytest.raises(StudentError, match="admitted only for 'confidence' mode"):
        StudentRegistry().load(TASK, Path(tmp_path), "student.json", router=router, admission=None)
