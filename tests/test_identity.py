"""`prismyra.identity` against a committed copy of perigraph's spec, so a drift between the two is a failure here.

The copy is vendored rather than imported or fetched for the reason tierbook gives for its own copy: importing
perigraph would put a dependency in front of the serving path, and a test that fetches skips when offline and so stops
running. The copies are perigraph main f24674a's ``spec/vocabularies.json`` and ``spec/fixtures/model_identity.json``;
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
VOCAB_DIGEST = "1ca5b18c7a2f66f3fd8d0140709b4e570004590ff2f11b4e3932f74b9e9e62b3"
FIXTURE_DIGEST = "cdb1264048274508866a586a4c45462cda6da3a01352f4ffee6905d1b62394f7"


def _load(name: str, digest: str) -> dict:
    raw = (SPEC / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, f"{name} changed; re-vendor deliberately and update its digest"
    return json.loads(raw)


VOCAB = _load("perigraph-vocabularies.json", VOCAB_DIGEST)
FIXTURES = _load("perigraph-model_identity.json", FIXTURE_DIGEST)


def test_vocabularies_match_the_spec_this_writer_follows():
    assert VOCAB["spec_version"] == I.SPEC_VERSION
    assert sorted(VOCAB["model_parts"]["values"]) == sorted(I.MODEL_PARTS)
    assert sorted(VOCAB["parts"]["values"]) == sorted(I.HARNESS_PARTS)
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


@pytest.mark.parametrize("fields", FIXTURES["refused_precision"]["cases"], ids=lambda f: repr(f)[:40])
def test_refuses_every_unresolved_precision_fixture(fields):
    with pytest.raises(I.IdentityError, match="unresolved"):
        I.precision_digest(fields)


@pytest.mark.parametrize("value", ["\x1cauto", "\x1f", "auto\x1e"])
def test_only_unicode_white_space_is_trimmed_not_pythons_wider_set(value):
    """U+001C-U+001F are not White_Space (perigraph section 1.2), so these values are neither empty nor ``auto``."""
    assert I.precision_digest({"dtype.bfloat16": "10", "quantisation": value, "device": "cuda"})


def _named(attrs: dict, prefix_held: str, prefix_absent: str, part: str) -> int:
    return (f"{prefix_held}{part}.sourcing" in attrs) + (f"{prefix_absent}{part}.reason" in attrs)


@pytest.mark.parametrize(
    "stamp",
    [
        I.ModelStamp(
            {
                "checkpoint": I.Part(digest="a" * 64),
                "precision": I.Part(digest="b" * 64),
                "runtime": I.Part(digest="c" * 64),
                "answer_mode": I.Part(digest="d" * 64),
            }
        ),
        I.stand_in("built in memory"),
    ],
    ids=["held", "stand_in"],
)
def test_every_part_is_named_once_as_held_or_absent(stamp):
    """perigraph section 1.2: each of the eleven parts and each of the four model parts, held or absent, never both and
    never silent. An absence that blames the collector carries a detail; one that does not, carries none."""
    attrs = stamp.attributes()
    for part in VOCAB["parts"]["values"]:
        assert _named(attrs, "perigraph.part.", "perigraph.absent.", part) == 1, part
    for part in VOCAB["model_parts"]["values"]:
        assert _named(attrs, "perigraph.model.", "perigraph.model_absent.", part) == 1, part
    blames = VOCAB["absence_reasons"]["blames"]
    for key, reason in attrs.items():
        if key.endswith(".reason"):
            has_detail = key[: -len("reason")] + "detail" in attrs
            assert has_detail == (blames[reason] == "collector"), key
    assert attrs["perigraph.collector_reaches"] == ""
    assert set(attrs["perigraph.model_reaches"].split(",")) == set(VOCAB["model_parts"]["values"])
    assert attrs["perigraph.absent.tool_extension.reason"] == "not_observable"
    assert attrs["perigraph.absent.request_template.reason"] == "not_reachable"
    assert attrs["perigraph.absent.request_template.detail"] == attrs["perigraph.absent.instruction.detail"]


def test_identity_strings_keep_version_2_across_the_record_version():
    """perigraph section 11 recommends ``perigraph/2/model/<id>``; the identities are derived as version 2 derived them,
    so a version 3 record of the same load keeps the same string and the same downstream ``arm_version``."""
    stamp = I.ModelStamp({k: I.Part(digest=c * 64) for k, c in zip(I.MODEL_PARTS, "abcd", strict=True)})
    assert stamp.attributes()["perigraph.spec_version"] == 3
    assert stamp.versions()["perigraph_model"] == f"perigraph/2/model/{stamp.model_identity}"


def test_attributes_use_only_keys_the_wire_form_defines():
    stamp = I.ModelStamp(
        {
            "checkpoint": I.Part(digest="a" * 64),
            "precision": I.Part(digest="b" * 64),
            "runtime": I.Part(digest="c" * 64),
            "answer_mode": I.Part(digest=hashlib.sha256(b"option_scores").hexdigest()),
        }
    )
    attrs = stamp.attributes()
    record = set(VOCAB["attributes"]["record"])
    per_model = VOCAB["attributes"]["per_model_part"]
    per_part = VOCAB["attributes"]["per_part"]
    for key in attrs:
        if key in record:
            continue
        parts = key.split(".")
        pattern = ".".join([*parts[:2], "<name>", *parts[3:]])
        if pattern in per_model:
            assert parts[2] in I.MODEL_PARTS, key
        else:
            assert pattern in per_part, key
            assert parts[2] in I.HARNESS_PARTS, key
    assert attrs["perigraph.model_identity"] == stamp.model_identity
    assert stamp.versions()["perigraph_artifact"] == "perigraph/2/artifact/" + stamp.artifact_identity
    assert stamp.versions("student_")["student_perigraph_model"] == "perigraph/2/model/" + stamp.model_identity


def test_an_absent_part_means_no_model_identity_but_keeps_the_artifact_identity():
    stamp = I.ModelStamp(
        {
            "checkpoint": I.Part(digest="a" * 64),
            "precision": I.Part(absent="extraction_failed", detail="unresolved"),
            "runtime": I.Part(digest="c" * 64),
            "answer_mode": I.Part(digest="d" * 64),
        }
    )
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


# --- the fp8 scale description, label-only -----------------------------------------------------------------------


def _fp8_checkpoint(root: Path, tensors: dict) -> Path:
    """A minimal checkpoint whose ``model.safetensors`` carries real float8 tensors, so `fp8_scale_description`
    reads real shapes rather than the garbage bytes `_checkpoint` above writes (which is fine for the tests that
    never ask for an fp8 label at all, i.e. every one above this point)."""
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(root / "model.safetensors"))
    (root / "config.json").write_text('{"model_type": "tiny"}')
    return root


def _fp8_engine() -> SimpleNamespace:
    """An engine whose backbone carries an in-memory float8 buffer, so `quantisation_of`'s dtype scan finds
    ``"fp8"`` regardless of the buffer's name -- the trigger for reading the checkpoint's own files at all. Not a
    model of `Fp8Linear`'s own naming: the real kernel replacement leaves the float8 tensor itself named ``weight``
    and adds a separate float32 ``scale`` buffer, neither of which this fixture needs to reproduce, because
    `quantisation_of` dispatches on dtype alone, never on an attribute's name."""
    eng = _engine(torch.bfloat16)
    eng.backbone.register_buffer("scale", torch.zeros(2, dtype=torch.float8_e4m3fn))
    return eng


def test_the_fp8_scale_description_names_the_block_size_from_the_checkpoints_own_shapes(tmp_path):
    root = _fp8_checkpoint(
        tmp_path / "ckpt",
        {
            "model.layer.weight": torch.zeros(256, 256, dtype=torch.float8_e4m3fn),
            "model.layer.weight_scale_inv": torch.zeros(2, 2, dtype=torch.float32),
        },
    )
    stamp = I.backbone_stamp(_fp8_engine(), before=I.preload(str(root)))
    assert "checkpoint_fp8_scale=block128x128:1" in stamp.parts["precision"].label


def test_two_block_sizes_are_told_apart_in_the_label_but_not_in_model_identity(tmp_path):
    """`checkpoint` already hashes these two directories' different bytes, so `model_identity` already differs
    without the label; this is exactly why the description stays out of `precision`'s digest (module docstring).
    The digest side of the same `Part` is asserted equal below: the label differs, nothing that is hashed does."""
    a = I.backbone_stamp(
        _fp8_engine(),
        before=I.preload(
            str(
                _fp8_checkpoint(
                    tmp_path / "a",
                    {
                        "model.layer.weight": torch.zeros(256, 256, dtype=torch.float8_e4m3fn),
                        "model.layer.weight_scale_inv": torch.zeros(2, 2, dtype=torch.float32),
                    },
                )
            )
        ),
    )
    b = I.backbone_stamp(
        _fp8_engine(),
        before=I.preload(
            str(
                _fp8_checkpoint(
                    tmp_path / "b",
                    {
                        "model.layer.weight": torch.zeros(256, 256, dtype=torch.float8_e4m3fn),
                        "model.layer.weight_scale_inv": torch.zeros(4, 4, dtype=torch.float32),
                    },
                )
            )
        ),
    )
    assert "block128x128:1" in a.parts["precision"].label
    assert "block64x64:1" in b.parts["precision"].label
    assert a.model_identity != b.model_identity  # from `checkpoint`, not from the label
    assert a.parts["precision"].digest == b.parts["precision"].digest  # the label-only field changed, nothing hashed


def test_a_row_scale_and_a_tensor_scale_are_told_apart(tmp_path):
    row = I.backbone_stamp(
        _fp8_engine(),
        before=I.preload(
            str(
                _fp8_checkpoint(
                    tmp_path / "row",
                    {
                        "model.layer.weight": torch.zeros(256, 256, dtype=torch.float8_e4m3fn),
                        "model.layer.weight_scale_inv": torch.zeros(256, dtype=torch.float32),
                    },
                )
            )
        ),
    )
    whole = I.backbone_stamp(
        _fp8_engine(),
        before=I.preload(
            str(
                _fp8_checkpoint(
                    tmp_path / "whole",
                    {
                        "model.layer.weight": torch.zeros(256, 256, dtype=torch.float8_e4m3fn),
                        "model.layer.weight_scale_inv": torch.zeros((), dtype=torch.float32),
                    },
                )
            )
        ),
    )
    assert "checkpoint_fp8_scale=per_row:1" in row.parts["precision"].label
    assert "checkpoint_fp8_scale=per_tensor:1" in whole.parts["precision"].label


def test_a_mixture_of_experts_name_with_no_weight_substring_is_not_matched_to_itself(tmp_path):
    """Regression for the self-match bug: a tensor named with no ``"weight"`` substring at all (unlike the real 35B
    checkpoint's own expert keys, which all end in ``.weight`` -- see `_fp8_checkpoint`'s other callers) must not
    have `scale_key_for`'s second convention fall back to the key unchanged and accept the weight as its own scale.
    No scale sibling is given here, so the only correct outcome is ``unresolved``, never a block layout read off the
    weight's own shape."""
    root = _fp8_checkpoint(
        tmp_path / "ckpt",
        {
            "model.experts.0.gate_up_proj": torch.zeros(512, 2048, dtype=torch.float8_e4m3fn),
        },
    )
    stamp = I.backbone_stamp(_fp8_engine(), before=I.preload(str(root)))
    assert "checkpoint_fp8_scale=unresolved:1" in stamp.parts["precision"].label


def test_block_scales_spanning_two_shards_are_still_resolved(tmp_path):
    """The real 35B checkpoint is 38 shards; a weight and its scale can land in different ``.safetensors`` files, so
    the reader has to see every shard's keys before it can match any of them."""
    from safetensors.torch import save_file

    root = tmp_path / "ckpt"
    root.mkdir()
    save_file(
        {"model.layer.weight": torch.zeros(256, 256, dtype=torch.float8_e4m3fn)},
        str(root / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {"model.layer.weight_scale_inv": torch.zeros(2, 2, dtype=torch.float32)},
        str(root / "model-00002-of-00002.safetensors"),
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layer.weight": "model-00001-of-00002.safetensors",
                    "model.layer.weight_scale_inv": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (root / "config.json").write_text('{"model_type": "tiny"}')
    stamp = I.backbone_stamp(_fp8_engine(), before=I.preload(str(root)))
    assert "checkpoint_fp8_scale=block128x128:1" in stamp.parts["precision"].label


def test_an_fp8_weight_with_no_scale_tensor_is_counted_unresolved_without_failing_precision(tmp_path):
    """A scale this reader cannot find never takes `precision` down over a cosmetic label: the digest and the
    dtype-count fields stay present, and the gap is named as a count rather than hidden or guessed at."""
    root = _fp8_checkpoint(tmp_path / "ckpt", {"model.layer.weight": torch.zeros(4, 4, dtype=torch.float8_e4m3fn)})
    stamp = I.backbone_stamp(_fp8_engine(), before=I.preload(str(root)))
    assert stamp.parts["precision"].absent == ""
    assert "checkpoint_fp8_scale=unresolved:1" in stamp.parts["precision"].label
    assert stamp.model_identity is not None


def test_fp8_in_memory_but_not_in_the_checkpoint_files_is_named_rather_than_silently_none(tmp_path):
    root = _fp8_checkpoint(tmp_path / "ckpt", {"model.layer.weight": torch.zeros(4, 4, dtype=torch.bfloat16)})
    stamp = I.backbone_stamp(_fp8_engine(), before=I.preload(str(root)))
    assert "checkpoint_fp8_scale=no_fp8_in_checkpoint_files" in stamp.parts["precision"].label


def test_no_safetensors_shard_at_all_is_named_differently_from_no_fp8_found(tmp_path):
    """Distinct absences: ``no_fp8_in_checkpoint_files`` means shards were opened and none held a float8 tensor;
    this is the case where nothing was opened at all, so saying "no fp8" would overclaim a check that never ran."""
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "model.bin").write_bytes(b"not a safetensors file")
    (root / "config.json").write_text('{"model_type": "tiny"}')
    assert I.fp8_scale_description(root, ("model.bin",)) == "no_safetensors_files_read"


def test_a_scale_shape_this_reader_cannot_classify_is_counted_not_blanked(tmp_path):
    """One tensor `scale_layout` cannot classify (a three-dimensional scale, which no real checkpoint produces,
    confirmed on the pod -- every expert is its own two-dimensional tensor) must not blank the whole description:
    the other, resolvable fp8 tensor in the same checkpoint still gets its own counted entry."""
    root = _fp8_checkpoint(
        tmp_path / "ckpt",
        {
            "model.layer_a.weight": torch.zeros(256, 256, dtype=torch.float8_e4m3fn),
            "model.layer_a.weight_scale_inv": torch.zeros(2, 2, dtype=torch.float32),
            "model.layer_b.weight": torch.zeros(4, 4, 4, dtype=torch.float8_e4m3fn),
            "model.layer_b.weight_scale_inv": torch.zeros(2, 2, 2, dtype=torch.float32),
        },
    )
    stamp = I.backbone_stamp(_fp8_engine(), before=I.preload(str(root)))
    label = stamp.parts["precision"].label
    assert "block128x128:1" in label
    assert "unclassified:1" in label
    assert stamp.model_identity is not None


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
        '{"weight_map": {"a": "m-00001.safetensors", "b": "m-00002.safetensors", "c": "m-00001.safetensors"}}'
    )
    for n in ("m-00001.safetensors", "m-00002.safetensors", "m-00003-unused.safetensors"):
        (root / n).write_bytes(n.encode())
    assert I.backbone_files(root) == [
        "model.safetensors.index.json",
        "m-00001.safetensors",
        "m-00002.safetensors",
        "config.json",
    ]


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


def test_a_file_the_loader_fetched_during_load_is_part_of_the_checkpoint(tmp_path):
    """A loader fills a cache with files it is about to read (seen on the pod: a processor config fetched by the
    first load), so the list is the directory after the load, not before it."""
    root = _checkpoint(tmp_path, b"weights")
    before = I.preload(str(root))
    (root / "preprocessor_config.json").write_text("{}")
    late = I.backbone_stamp(_engine(torch.bfloat16), before=before)
    assert late.parts["checkpoint"].label == "4 files"
    assert late.artifact_identity == _stamp(root).artifact_identity


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
    expected = I.identity(
        [
            (
                "checkpoint",
                I.checkpoint_digest(
                    [("student.json", hashlib.sha256(b'{"model_text": "tree\\nleaf_value=0.1 0.2"}').hexdigest())]
                ),
            )
        ]
    )
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
    return SimpleNamespace(
        served_by=served_by, student_probabilities={"a": 1.0} if student_ran else None, student_stamp=stamp
    )


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
    return SimpleNamespace(
        ts=1.0,
        context="doc",
        question="q?",
        answered_by=answered_by,
        comparison_probabilities=None,
        route_mode="confidence",
        versions=versions,
        request_id="r1",
        route_params_version="p",
        probabilities=probs or {"a": 0.9, "b": 0.1},
        options=("a", "b"),
    )


BASE = {
    "backbone": "m",
    "package": "1",
    "spec": "s",
    "tag": "t",
    "perigraph_model": "perigraph/2/model/" + "1" * 24,
    "perigraph_artifact": "perigraph/2/artifact/" + "2" * 24,
}


def test_export_writes_format_4_rows_naming_the_backbone_artifact():
    from prismyra.learn.observations import export_experience

    rows, _ = export_experience(
        [_exp(dict(BASE))], task_type="tt", arm="router", student_arm="s", fallback_arm="teacher", per_component=True
    )
    (r,) = rows
    assert r["format"] == 4 and r["arm"] == "teacher"
    assert r["arm_artifact"] == BASE["perigraph_artifact"]
    assert r["arm_version"].endswith("|" + BASE["perigraph_model"])


def test_export_without_identities_still_writes_format_3():
    from prismyra.learn.observations import export_experience

    old = {k: v for k, v in BASE.items() if "perigraph" not in k}
    rows, _ = export_experience(
        [_exp(old)], task_type="tt", arm="router", student_arm="s", fallback_arm="teacher", per_component=True
    )
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
    rows, counts = O.export_experience(
        [exp],
        task_type="tt",
        arm="router",
        student_arm="s",
        fallback_arm="teacher",
        per_component=True,
        student=student,
        student_stamp=declared,
    )
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
    rows, _ = O.export_experience(
        [exp],
        task_type="tt",
        arm="router",
        student_arm="s",
        fallback_arm="teacher",
        per_component=True,
        student=SimpleNamespace(),
        student_stamp=declared,
    )
    s_row = next(r for r in rows if r["arm"] == "s")
    assert s_row["arm_version"] == "student-v"
    assert s_row["arm_artifact"] == declared.versions()["perigraph_artifact"]
    scored = O.score_items(
        SimpleNamespace(),
        [{"item": "i", "context": "c", "question": "q", "label": "a"}],
        task_type="tt",
        arm="s",
        label_source="gold",
        predict=lambda c, q: {"a": 0.9, "b": 0.1},
        stamp=declared,
    )
    assert scored[0]["arm_artifact"] == s_row["arm_artifact"] and scored[0]["format"] == 4
