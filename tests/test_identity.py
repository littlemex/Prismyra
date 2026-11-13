"""`prismyra.identity` against a committed copy of perigraph's spec, so a drift between the two is a failure here.

The copy is vendored rather than imported or fetched for the reason tierbook gives for its own copy: importing
perigraph would put a dependency in front of the serving path, and a test that fetches skips when offline and so stops
running. The copies are perigraph main ab50373's ``spec/vocabularies.json`` and ``spec/fixtures/model_identity.json``;
the digests pin them, so a re-vendor changes the file and the constant in one diff and a reviewer sees the protocol
move.
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
    assert stamp.versions()["perigraph_artifact"] == "perigraph/2/artifact/" + stamp.artifact_identity
    assert stamp.versions("student_")["student_perigraph_model"] == "perigraph/2/model/" + stamp.model_identity


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
    return SimpleNamespace(backbone=_Tiny(dtype))


def _checkpoint(tmp_path: Path, weights: bytes) -> Path:
    root = tmp_path / "ckpt"
    root.mkdir(exist_ok=True)
    (root / "model.safetensors").write_bytes(weights)
    (root / "config.json").write_text('{"model_type": "tiny"}')
    (root / "tokenizer.json").write_text('{"v": 1}')
    (root / "optimizer.pt").write_bytes(b"not read by the loader")
    return root


def _stamp(root, dtype=torch.bfloat16, **kw):
    return I.backbone_stamp(_engine(dtype), before=I.preload(str(root)), **kw)


def test_two_precisions_of_one_checkpoint_are_two_models_over_one_artifact(tmp_path):
    root = _checkpoint(tmp_path, b"weights")
    a, b = _stamp(root, torch.bfloat16), _stamp(root, torch.float32)
    assert a.artifact_identity == b.artifact_identity
    assert a.model_identity != b.model_identity
    assert a.parts["answer_mode"].label == "option_scores"
    assert "quantisation=none" in a.parts["precision"].label and "device=cpu" in a.parts["precision"].label


def test_quantisation_is_read_off_the_loaded_tensors_not_asked_for(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMYRA_EXPERTS", "nvfp4")
    root = _checkpoint(tmp_path, b"weights")
    assert "quantisation=none" in _stamp(root).parts["precision"].label
    eng = _engine(torch.bfloat16)
    eng.backbone.register_buffer("scale", torch.zeros(2, dtype=torch.float8_e4m3fn))
    assert I.quantisation_of([eng.backbone]) == "fp8"


def test_parameters_on_two_devices_are_not_rendered_as_one(tmp_path):
    eng = _engine(torch.bfloat16)
    eng.backbone.register_buffer("on_meta", torch.zeros(1, device="meta"))
    stamp = I.backbone_stamp(eng, before=I.preload(str(_checkpoint(tmp_path, b"w"))))
    assert stamp.parts["precision"].absent == "extraction_failed"
    assert stamp.artifact_identity is not None and stamp.model_identity is None


def test_a_file_the_loader_does_not_read_does_not_change_the_identity(tmp_path):
    root = _checkpoint(tmp_path, b"weights")
    before = _stamp(root).artifact_identity
    (root / "optimizer.pt").write_bytes(b"changed")
    assert _stamp(root).artifact_identity == before
    (root / "model.safetensors").write_bytes(b"retrained")
    assert _stamp(root).artifact_identity != before


def test_a_single_weights_file_wins_over_a_stale_index_as_the_loader_does(tmp_path):
    root = _checkpoint(tmp_path, b"weights")
    (root / "model.safetensors.index.json").write_text('{"weight_map": {"a": "gone-00001.safetensors"}}')
    assert "model.safetensors" in I.backbone_files(root)
    assert "gone-00001.safetensors" not in I.backbone_files(root)


def test_sharded_weights_are_the_index_and_every_shard_it_names(tmp_path):
    root = tmp_path / "sharded"
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "m-00001.safetensors", "b": "m-00002.safetensors", "c": "m-00001.safetensors"}}')
    for n in ("m-00001.safetensors", "m-00002.safetensors", "m-00003-unused.safetensors"):
        (root / n).write_bytes(n.encode())
    assert I.backbone_files(root) == ["model.safetensors.index.json", "m-00001.safetensors", "m-00002.safetensors",
                                      "config.json"]


def test_remote_code_and_adapters_are_part_of_the_checkpoint(tmp_path):
    root = _checkpoint(tmp_path, b"weights")
    (root / "config.json").write_text('{"auto_map": {"AutoModel": "modeling_x.X"}}')
    (root / "modeling_x.py").write_text("code")
    (root / "adapter_model.safetensors").write_bytes(b"lora")
    names = I.backbone_files(root)
    assert "modeling_x.py" in names and "adapter_model.safetensors" in names


def test_a_layout_this_collector_cannot_name_is_refused_not_identified(tmp_path):
    root = tmp_path / "bin"
    root.mkdir()
    (root / "pytorch_model.bin").write_bytes(b"pickle")
    (root / "config.json").write_text("{}")
    stamp = _stamp(root)
    assert stamp.parts["checkpoint"].absent == "extraction_failed"
    tik = _checkpoint(tmp_path, b"w")
    (tik / "o200k.tiktoken").write_text("ranks")
    assert _stamp(tik).parts["checkpoint"].absent == "extraction_failed"


def test_a_file_changed_between_load_and_hashing_is_a_failure_not_a_digest(tmp_path):
    root = _checkpoint(tmp_path, b"weights")
    before = I.preload(str(root))
    (root / "model.safetensors").write_bytes(b"overwritten by a trainer after load")
    stamp = I.backbone_stamp(_engine(torch.bfloat16), before=before)
    assert stamp.parts["checkpoint"].absent == "extraction_failed"
    assert "changed" in stamp.parts["checkpoint"].detail


def test_heads_are_part_of_what_answered_under_their_own_prefix(tmp_path):
    root = _checkpoint(tmp_path, b"weights")
    heads = tmp_path / "heads.json"
    heads.write_text('{"h": 1}')
    plain = _stamp(root)
    with_heads = _stamp(root, heads=str(heads))
    assert plain.artifact_identity != with_heads.artifact_identity
    assert with_heads.parts["checkpoint"].label == "4 files"


def test_a_checkpoint_that_cannot_be_read_is_absent_with_why_not_guessed(tmp_path):
    stamp = _stamp(tmp_path / "missing")
    assert stamp.parts["checkpoint"].absent == "extraction_failed"
    assert stamp.parts["checkpoint"].detail
    assert stamp.model_identity is None and stamp.artifact_identity is None
    assert not stamp.parts["runtime"].absent and not stamp.parts["precision"].absent


def test_a_student_is_its_bytes_whatever_the_file_is_called():
    art = SimpleNamespace(backend="lightgbm")
    a = I.student_stamp(b'{"model_text": "tree\\nleaf_value=0.1 0.2"}', art)
    b = I.student_stamp(b'{"model_text": "tree\\nleaf_value=0.1 0.3"}', art)
    assert a.artifact_identity != b.artifact_identity
    assert I.student_stamp(b'{"model_text": "tree\\nleaf_value=0.1 0.2"}', art).artifact_identity == a.artifact_identity
    assert a.parts["answer_mode"].label == "classifier_output"
    expected = I.identity([("checkpoint", I.checkpoint_digest(
        [("student.json", hashlib.sha256(b'{"model_text": "tree\\nleaf_value=0.1 0.2"}').hexdigest())]))])
    assert a.artifact_identity == expected


def test_a_student_has_an_artifact_identity_and_no_model_identity():
    stamp = I.student_stamp(b"{}", SimpleNamespace(backend="xgboost"))
    assert stamp.model_identity is None and stamp.artifact_identity is not None
    assert stamp.parts["precision"].absent == "extraction_failed"
    assert "lightgbm" not in stamp.parts["runtime"].label


def test_a_stand_in_is_every_part_absent_with_the_same_reason():
    stamp = I.stand_in("built in memory")
    assert stamp.model_identity is None and stamp.artifact_identity is None
    assert all(p.absent == "not_reachable" and p.detail for p in stamp.parts.values())


def test_the_registry_stamps_the_bytes_it_parsed(tmp_path, monkeypatch):
    from prismyra.learn import students as S

    raw = b'{"tag": "t"}'
    p = tmp_path / "any-name.json"
    p.write_bytes(raw)
    seen = {}

    class _Handle:
        def __init__(self, artifact, router=None, record=None, raw=None):
            seen["raw"] = raw

    monkeypatch.setattr(S, "StudentHandle", _Handle)
    monkeypatch.setattr(S.StudentArtifact, "from_json", staticmethod(lambda d: SimpleNamespace(**d)))
    S.StudentRegistry().load("t", tmp_path, str(p), admission=None)
    assert seen["raw"] == raw


def _outcome(served_by, student_ran, stamp):
    return SimpleNamespace(served_by=served_by, student_probabilities={"a": 1.0} if student_ran else None,
                           student_stamp=stamp)


def test_a_row_names_the_student_that_ran_for_it_and_no_other():
    from prismyra.learn.hook import LearnHook

    hook = LearnHook.__new__(LearnHook)
    hook.package_version, hook.backbone = "1", "m"
    hook.spec = SimpleNamespace(version="s")
    hook.backbone_perigraph = {"perigraph_model": "perigraph/2/model/x"}
    tag = SimpleNamespace(task="t")
    stamp = I.student_stamp(b"{}", SimpleNamespace(backend="lightgbm"))
    import prismyra.learn.hook as H

    H_tag_version = H.tag_version
    try:
        H.tag_version = lambda t: "tv"
        ran = hook._row_versions(tag, _outcome("prismyra", True, stamp))
        skipped = hook._row_versions(tag, _outcome("prismyra", False, stamp))
        unrouted = hook._row_versions(tag)
    finally:
        H.tag_version = H_tag_version
    assert ran["student_perigraph_artifact"] == "perigraph/2/artifact/" + stamp.artifact_identity
    assert ran["perigraph_model"] == "perigraph/2/model/x"
    assert "student_perigraph_artifact" not in skipped and "student_perigraph_artifact" not in unrouted


def _exp(versions, answered_by="prismyra", probs=None):
    return SimpleNamespace(ts=1.0, context="doc", question="q?", answered_by=answered_by,
                           comparison_probabilities=None, route_mode="confidence", versions=versions,
                           request_id="r1", route_params_version="p", probabilities=probs or {"a": 0.9, "b": 0.1},
                           options=("a", "b"))


BASE = {"backbone": "m", "package": "1", "spec": "s", "tag": "t",
        "perigraph_model": "perigraph/2/model/" + "1" * 24, "perigraph_artifact": "perigraph/2/artifact/" + "2" * 24}


def test_export_writes_format_4_rows_naming_the_backbone_artifact():
    from prismyra.learn.observations import export_experience

    rows, _ = export_experience([_exp(dict(BASE))], task_type="tt", arm="router", student_arm="s",
                                fallback_arm="teacher", per_component=True)
    (r,) = rows
    assert r["format"] == 4 and r["arm"] == "teacher"
    assert r["arm_artifact"] == BASE["perigraph_artifact"]
    assert r["arm_version"].endswith("|" + BASE["perigraph_model"])


def test_export_without_identities_still_writes_format_3():
    from prismyra.learn.observations import export_experience

    old = {k: v for k, v in BASE.items() if "perigraph" not in k}
    rows, _ = export_experience([_exp(old)], task_type="tt", arm="router", student_arm="s", fallback_arm="teacher",
                                per_component=True)
    assert rows[0]["format"] == 3 and "arm_artifact" not in rows[0]


def test_export_skips_and_counts_a_student_component_logged_beside_another_artifact(monkeypatch):
    from prismyra.learn import observations as O

    student = SimpleNamespace()
    monkeypatch.setattr("prismyra.learn.students.artifact_version", lambda a: "student-v")
    monkeypatch.setattr("prismyra.learn.students.student_version", lambda a: "sv")
    declared = I.student_stamp(b"{}", SimpleNamespace(backend="lightgbm"))
    logged = {**BASE, "student_perigraph_artifact": "perigraph/2/artifact/" + "9" * 24}
    exp = _exp(logged)
    exp.comparison_probabilities = {"a": 0.6, "b": 0.4}
    rows, counts = O.export_experience([exp], task_type="tt", arm="router", student_arm="s", fallback_arm="teacher",
                                       per_component=True, student=student, student_stamp=declared)
    assert [r["arm"] for r in rows] == ["teacher"]
    assert counts["student component skipped: logged beside another student artifact"] == 1


def test_a_student_row_keeps_the_version_an_offline_score_writes(monkeypatch):
    from prismyra.learn import observations as O

    monkeypatch.setattr("prismyra.learn.students.artifact_version", lambda a: "student-v")
    monkeypatch.setattr("prismyra.learn.students.student_version", lambda a: "sv")
    declared = I.student_stamp(b"{}", SimpleNamespace(backend="lightgbm"))
    logged = {**BASE, **declared.versions("student_")}
    exp = _exp(logged)
    exp.comparison_probabilities = {"a": 0.6, "b": 0.4}
    rows, _ = O.export_experience([exp], task_type="tt", arm="router", student_arm="s", fallback_arm="teacher",
                                  per_component=True, student=SimpleNamespace(), student_stamp=declared)
    s_row = next(r for r in rows if r["arm"] == "s")
    assert s_row["arm_version"] == "student-v"
    assert s_row["arm_artifact"] == declared.versions()["perigraph_artifact"]
    scored = O.score_items(SimpleNamespace(), [{"item": "i", "context": "c", "question": "q", "label": "a"}],
                           task_type="tt", arm="s", label_source="gold", predict=lambda c, q: {"a": 0.9, "b": 0.1},
                           stamp=declared)
    assert scored[0]["arm_artifact"] == s_row["arm_artifact"] and scored[0]["format"] == 4
