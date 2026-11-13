"""`prismyra.identity` against a committed copy of perigraph's spec, so a drift between the two is a failure here.

The copy is vendored rather than imported or fetched for the reason tierbook gives for its own copy: importing
perigraph would put a dependency in front of the serving path, and a test that fetches skips when offline and so stops
running. The digests below are what the copies hashed to when they were taken; a deliberate re-vendor changes the file
and the constant in one diff, so a reviewer sees the protocol move.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prismyra import identity as I

SPEC = Path(__file__).parent / "spec"
VOCAB_DIGEST = "d2e12f5d935a623e1e89673e40744aca6d8a84a2c6a821b43605469d47e83074"
FIXTURE_DIGEST = "cf92077e43a57d39847a9e5287664a3391c39b92add90a505f2dcc5eed6e642c"


def _load(name: str, digest: str) -> dict:
    raw = (SPEC / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, f"{name} changed; re-vendor deliberately and update its digest"
    return json.loads(raw)


VOCAB = _load("perigraph-vocabularies.json", VOCAB_DIGEST)
FIXTURES = _load("perigraph-model_identity.json", FIXTURE_DIGEST)


def test_vocabularies_match_the_spec_this_writer_follows():
    assert VOCAB["spec_version"] == I.SPEC_VERSION
    assert sorted(VOCAB["model_parts"]["values"]) == sorted(I.MODEL_PARTS)
    assert sorted(VOCAB["answer_modes"]["values"]) == sorted(I.ANSWER_MODES)
    assert VOCAB["absence_reasons"]["blames"] == I.ABSENCE_BLAMES
    assert VOCAB["digest"]["identity_length"] == I.IDENTITY_LENGTH
    assert "loaded_by_us" in VOCAB["sourcing"]["values"]


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda c: c["name"][:40])
def test_reproduces_every_model_identity_fixture(case):
    entries = [(p, hashlib.sha256(c.encode("utf-8")).hexdigest()) for p, c in case["files"].items()]
    cp = I.checkpoint_digest(entries)
    pr = I.precision_digest(case["precision"])
    rt = I.lines_digest(case["runtime"], "==")
    am = hashlib.sha256(case["answer_mode"].encode("utf-8")).hexdigest()
    assert (cp, pr, rt) == (case["checkpoint_digest"], case["precision_digest"], case["runtime_digest"])
    pairs = [("checkpoint", cp), ("precision", pr), ("runtime", rt), ("answer_mode", am)]
    assert I.identity(pairs) == case["model_identity"]
    assert I.identity([("checkpoint", cp)]) == case["artifact_identity"]


def test_attributes_use_only_keys_the_wire_form_defines():
    stamp = I.ModelStamp({"checkpoint": I.Part(digest="a" * 64), "precision": I.Part(digest="b" * 64),
                          "runtime": I.Part(digest="c" * 64),
                          "answer_mode": I.Part(digest=hashlib.sha256(b"option_scores").hexdigest())})
    attrs = stamp.attributes()
    record = set(VOCAB["attributes"]["record"])
    per_part = VOCAB["attributes"]["per_model_part"]
    for key in attrs:
        if key in record:
            continue
        parts = key.split(".")
        pattern = ".".join([*parts[:2], "<name>", *parts[3:]])
        assert pattern in per_part, key
        assert parts[2] in I.MODEL_PARTS
    assert attrs["perigraph.model_identity"] == stamp.model_identity
    assert stamp.versions()["artifact_identity"].startswith("perigraph/2/artifact/")


def test_an_absent_part_means_no_model_identity_but_keeps_the_artifact_identity():
    stamp = I.ModelStamp({"checkpoint": I.Part(digest="a" * 64),
                          "precision": I.Part(absent="extraction_failed", detail="unresolved"),
                          "runtime": I.Part(digest="c" * 64), "answer_mode": I.Part(digest="d" * 64)})
    assert stamp.model_identity is None
    assert stamp.artifact_identity is not None
    attrs = stamp.attributes()
    assert "perigraph.model_identity" not in attrs
    assert attrs["perigraph.model_absent.precision.reason"] == "extraction_failed"


def test_a_detail_is_required_exactly_when_the_absence_is_ours():
    with pytest.raises(I.IdentityError):
        I.ModelStamp({k: I.Part(absent="not_reachable") for k in I.MODEL_PARTS})
    with pytest.raises(I.IdentityError):
        I.ModelStamp({k: I.Part(absent="not_provided", detail="why") for k in I.MODEL_PARTS})


class _Tiny(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4).to(dtype)


def _engine(dtype):
    return SimpleNamespace(backbone=_Tiny(dtype), torch_device=torch.device("cpu"))


def _checkpoint(tmp_path: Path, weights: bytes) -> Path:
    root = tmp_path / "ckpt"
    root.mkdir(exist_ok=True)
    (root / "model.safetensors").write_bytes(weights)
    (root / "config.json").write_text('{"model_type": "tiny"}')
    (root / "tokenizer.json").write_text('{"v": 1}')
    (root / "optimizer.pt").write_bytes(b"not read by the loader")
    return root


def test_two_precisions_of_one_checkpoint_are_two_models_over_one_artifact(tmp_path):
    root = _checkpoint(tmp_path, b"weights")
    a = I.backbone_stamp(_engine(torch.bfloat16), model=str(root))
    b = I.backbone_stamp(_engine(torch.float32), model=str(root))
    assert a.artifact_identity == b.artifact_identity
    assert a.model_identity != b.model_identity
    assert a.parts["answer_mode"].label == "option_scores"


def test_a_file_the_loader_does_not_read_does_not_change_the_identity(tmp_path):
    root = _checkpoint(tmp_path, b"weights")
    before = I.backbone_stamp(_engine(torch.bfloat16), model=str(root)).artifact_identity
    (root / "optimizer.pt").write_bytes(b"changed")
    assert I.backbone_stamp(_engine(torch.bfloat16), model=str(root)).artifact_identity == before
    (root / "model.safetensors").write_bytes(b"retrained")
    assert I.backbone_stamp(_engine(torch.bfloat16), model=str(root)).artifact_identity != before


def test_heads_are_part_of_what_answered(tmp_path):
    root = _checkpoint(tmp_path, b"weights")
    heads = tmp_path / "heads.json"
    heads.write_text('{"h": 1}')
    plain = I.backbone_stamp(_engine(torch.bfloat16), model=str(root))
    with_heads = I.backbone_stamp(_engine(torch.bfloat16), model=str(root), heads=str(heads))
    assert plain.artifact_identity != with_heads.artifact_identity


def test_a_checkpoint_that_cannot_be_read_is_absent_with_why_not_guessed(tmp_path):
    stamp = I.backbone_stamp(_engine(torch.bfloat16), model=str(tmp_path / "missing"))
    assert stamp.parts["checkpoint"].absent == "extraction_failed"
    assert stamp.parts["checkpoint"].detail
    assert stamp.model_identity is None and stamp.artifact_identity is None


def _student(tmp_path: Path, text: str):
    art = SimpleNamespace(backend="lightgbm", model_text=text)
    path = tmp_path / f"student-{hashlib.sha256(text.encode()).hexdigest()[:6]}.json"
    path.write_text(json.dumps({"model_text": text}))
    return path, art


def test_a_retrained_student_is_a_different_artifact(tmp_path):
    p1, a1 = _student(tmp_path, "tree\nleaf_value=0.1 0.2\n")
    p2, a2 = _student(tmp_path, "tree\nleaf_value=0.1 0.3\n")
    s1, s2 = I.student_stamp(p1, a1), I.student_stamp(p2, a2)
    assert s1.artifact_identity != s2.artifact_identity
    assert s1.parts["answer_mode"].label == "classifier_output"
    assert I.student_stamp(p1, a1).model_identity == s1.model_identity


def test_a_student_whose_leaves_this_collector_cannot_count_has_no_model_identity(tmp_path):
    path, _ = _student(tmp_path, "booster")
    stamp = I.student_stamp(path, SimpleNamespace(backend="xgboost", model_text="booster"))
    assert stamp.model_identity is None
    assert stamp.artifact_identity is not None
    assert stamp.parts["precision"].absent == "extraction_failed"


def test_unreachable_is_every_part_absent_with_the_same_reason():
    stamp = I.unreachable("identify=False")
    assert stamp.model_identity is None and stamp.artifact_identity is None
    assert all(p.absent == "not_reachable" and p.detail for p in stamp.parts.values())


def _exp(versions, answered_by="prismyra", probs=None):
    return SimpleNamespace(ts=1.0, context="doc", question="q?", answered_by=answered_by,
                           comparison_probabilities=None, route_mode="confidence", versions=versions,
                           request_id="r1", route_params_version="p", probabilities=probs or {"a": 0.9, "b": 0.1},
                           options=("a", "b"))


BASE = {"backbone": "m", "package": "1", "spec": "s", "tag": "t",
        "model_identity": "perigraph/2/model/" + "1" * 24, "artifact_identity": "perigraph/2/artifact/" + "2" * 24}


def test_export_writes_format_4_rows_naming_the_backbone_artifact():
    from prismyra.learn.observations import export_experience

    rows, _ = export_experience([_exp(dict(BASE))], task_type="tt", arm="router", student_arm="s",
                                fallback_arm="teacher", per_component=True)
    (r,) = rows
    assert r["format"] == 4 and r["arm"] == "teacher"
    assert r["arm_artifact"] == BASE["artifact_identity"]
    assert r["arm_version"].endswith("|model:" + BASE["model_identity"])


def test_export_without_identities_still_writes_format_3():
    from prismyra.learn.observations import export_experience

    old = {k: v for k, v in BASE.items() if "identity" not in k}
    rows, _ = export_experience([_exp(old)], task_type="tt", arm="router", student_arm="s", fallback_arm="teacher",
                                per_component=True)
    assert rows[0]["format"] == 3 and "arm_artifact" not in rows[0]


def test_export_refuses_a_row_logged_beside_another_student_file(tmp_path):
    from prismyra.learn.observations import ObservationError, export_experience

    path, art = _student(tmp_path, "tree\nleaf_value=0.1 0.2\n")
    declared = I.student_stamp(path, art)
    logged = {**BASE, "student_artifact_identity": "perigraph/2/artifact/" + "9" * 24}
    with pytest.raises(ObservationError, match="declared"):
        export_experience([_exp(logged)], task_type="tt", arm="router", student_arm="s", fallback_arm="teacher",
                          per_component=True, student_stamp=declared)
