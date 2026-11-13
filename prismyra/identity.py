"""What answered: the model half of a perigraph version 2 record, written by the process that loaded the model.

perigraph (github.com/littlemex/perigraph, SPEC.md section 9) describes a model by four parts, each digested by the
process that loaded it (sourcing ``loaded_by_us``), and two identities over them:

| part | what this module digests |
|---|---|
| ``checkpoint`` | the files `backbone_files` names, plus heads and temperature: ``<path>\\t<sha256>\\n`` lines |
| ``precision`` | ``dtype.<name>=<count>``, ``quantisation``, ``device``, read off the loaded parameters and buffers |
| ``runtime`` | ``<distribution>==<version>`` for the packages declared below, resolved in this process |
| ``answer_mode`` | ``option_scores`` for the backbone, ``classifier_output`` for a student |

The model identity is over all four and is comparable only within one collector and collector version, because
``precision`` and ``runtime`` are this collector's rendering. The artifact identity is over ``checkpoint`` alone and is
what a trainer can compute when it saves a model.

Each part exists because one distillation record changed the answers while every name stayed the same: a retrain from
the same script and seed agreed with its predecessor on 93.4 % of labels (checkpoint), dynamic int8 at load left the
stored bytes unchanged and agreed on 33-40 % (precision), a rebuilt image lost packages and 97 of 100 answers agreed
(runtime), and a student trained on option scores generated "1111" when asked to generate (answer mode).

This module imports nothing from perigraph: it writes the record from the spec, and the test suite checks it against a
committed copy of perigraph's vocabularies and model-identity fixtures. A part that cannot be read is recorded absent
with the reason and a detail, never guessed, so a record without a model identity says why.

A student (`student_stamp`) has an artifact identity and no model identity: a tree model exposes no parameter dtype,
so its ``precision`` is recorded as failed rather than declared.

What it cannot see: files a loader reads that this module does not name, packages that matter and are not in the
declared list, and anything a feature builder computes with (a student's embedding model is not in the student's
checkpoint). A wrong file list gives a confident wrong identity; a closed vocabulary constrains spelling, not truth.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

#: The perigraph version these records follow.
SPEC_VERSION = 2
#: Who wrote the record. With the package version, it is what a model identity is local to.
COLLECTOR = "prismyra"
IDENTITY_LENGTH = 24
MODEL_PARTS = ("answer_mode", "checkpoint", "precision", "runtime")
ANSWER_MODES = ("classifier_output", "generated_text", "option_scores")
#: Reasons a part can be absent, and whose failure each is (perigraph section 1.2). Only the collector's own
#: failures carry a detail.
ABSENCE_BLAMES = {"not_provided": "sender", "redacted": "sender", "not_reachable": "collector",
                  "extraction_failed": "collector", "not_observable": "nobody"}

#: The distributions whose versions decide the backbone's answers. Static per collector version, as perigraph's
#: manifest must be: a list that adapted to what it found could never be wrong, and so could never be checked.
BACKBONE_RUNTIME = ("prismyra", "torch", "transformers", "kernels", "triton", "safetensors", "tokenizers")


class IdentityError(Exception):
    """A part could not be rendered as the spec defines it. The caller records the part absent; it does not guess."""


def _hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_digest(entries) -> str:
    """Over ``(relative POSIX path, sha256)`` pairs, sorted by the path's utf-8 bytes (perigraph section 9)."""
    rows, seen = [], set()
    for path, sha in entries:
        if not path or "\t" in path or "\n" in path or path.startswith("/") or "\\" in path \
                or any(seg in ("", ".", "..") for seg in path.split("/")):
            raise IdentityError(f"checkpoint path {path!r} is not a normalised relative POSIX path")
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise IdentityError(f"checkpoint file {path!r} has digest {sha!r}, which is not lowercase hex sha256")
        if path in seen:
            raise IdentityError(f"checkpoint path {path!r} is listed twice")
        seen.add(path)
        rows.append((path.encode("utf-8"), f"{path}\t{sha}\n"))
    if not rows:
        raise IdentityError("a checkpoint with no files has no content to identify")
    return _hex("".join(r for _, r in sorted(rows)))


def lines_digest(pairs: dict, sep: str) -> str:
    """``precision`` (sep ``=``) and ``runtime`` (sep ``==``): one ``<key><sep><value>`` line each, sorted."""
    if not pairs:
        raise IdentityError("an empty mapping has no content to identify")
    lines = []
    for key, value in pairs.items():
        k, v = str(key), str(value)
        if not k or "\n" in k or "\n" in v or sep in k:
            raise IdentityError(f"{k!r}={v!r} cannot be rendered as one line")
        lines.append((k.encode("utf-8"), f"{k}{sep}{v}\n"))
    return _hex("".join(r for _, r in sorted(lines)))


def precision_digest(fields: dict) -> str:
    for k, v in fields.items():
        if str(v).strip().lower() in ("", "auto"):
            raise IdentityError(f"precision field {k}={v!r} is unresolved")
    return lines_digest(fields, "=")


def identity(pairs) -> str:
    """perigraph section 2.2: ``<part>=<digest>;`` for each part sorted by name, utf-8, sha256, first 24 hex."""
    h = hashlib.sha256()
    for kind, d in sorted(pairs):
        h.update(f"{kind}={d};".encode())
    return h.hexdigest()[:IDENTITY_LENGTH]


def runtime_versions(distributions) -> dict[str, str]:
    """Each declared distribution's version as this process resolves it; ``none`` for one that is not installed, so
    installing it later changes the identity instead of leaving it alone."""
    from importlib import metadata

    out = {}
    for name in distributions:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "none"
    return out


@dataclass(frozen=True)
class Part:
    digest: str = ""
    label: str = ""
    absent: str = ""
    detail: str = ""


@dataclass(frozen=True)
class ModelStamp:
    """The model half of a record: one `Part` per model part, each held with a digest or absent with its reason."""

    parts: dict[str, Part] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.parts) != set(MODEL_PARTS):
            raise IdentityError(f"a stamp names every model part, not {sorted(self.parts)}")
        for kind, p in self.parts.items():
            if bool(p.digest) == bool(p.absent):
                raise IdentityError(f"model part {kind!r} is either held with a digest or absent with a reason")
            if p.absent:
                if p.absent not in ABSENCE_BLAMES:
                    raise IdentityError(f"{p.absent!r} is not an absence reason")
                if (ABSENCE_BLAMES[p.absent] == "collector") != bool(p.detail):
                    raise IdentityError(f"model part {kind!r}: a detail is required exactly when the absence is ours")

    @property
    def model_identity(self) -> str | None:
        if any(p.absent for p in self.parts.values()):
            return None
        return identity((k, p.digest) for k, p in self.parts.items())

    @property
    def artifact_identity(self) -> str | None:
        cp = self.parts["checkpoint"]
        return None if cp.absent else identity([("checkpoint", cp.digest)])

    def attributes(self) -> dict:
        """The wire form's model half (perigraph section 6): flat, prefixed, identities omitted when absent."""
        from prismyra import __version__

        out: dict = {"perigraph.spec_version": SPEC_VERSION, "perigraph.collector": COLLECTOR,
                     "perigraph.collector_version": __version__, "perigraph.status": "complete"}
        for kind in MODEL_PARTS:
            p = self.parts[kind]
            if p.absent:
                out[f"perigraph.model_absent.{kind}.reason"] = p.absent
                if p.detail:
                    out[f"perigraph.model_absent.{kind}.detail"] = p.detail
            else:
                out[f"perigraph.model.{kind}.sourcing"] = "loaded_by_us"
                out[f"perigraph.model.{kind}.digest"] = p.digest
                if p.label:
                    out[f"perigraph.model.{kind}.label"] = p.label
        if self.model_identity:
            out["perigraph.model_identity"] = self.model_identity
        if self.artifact_identity:
            out["perigraph.artifact_identity"] = self.artifact_identity
        return out

    def versions(self, prefix: str = "") -> dict[str, str]:
        """The two identities as opaque version strings, the form perigraph section 11 recommends to an emitter, under
        ``<prefix>perigraph_model`` and ``<prefix>perigraph_artifact``."""
        out = {}
        if self.model_identity:
            out[f"{prefix}perigraph_model"] = f"perigraph/{SPEC_VERSION}/model/{self.model_identity}"
        if self.artifact_identity:
            out[f"{prefix}perigraph_artifact"] = f"perigraph/{SPEC_VERSION}/artifact/{self.artifact_identity}"
        return out


def _try(fn, *args) -> Part:
    """Run one part's collector. Any failure is this collector's, recorded with what went wrong, and never stops the
    model from loading: a record that says a part failed is worth more than no server."""
    try:
        digest, label = fn(*args)
        return Part(digest=digest, label=label)
    except Exception as exc:  # noqa: BLE001 - the collector boundary; the failure is recorded, not hidden
        return Part(absent="extraction_failed", detail=f"{type(exc).__name__}: {exc}")


def _answer_mode(label: str) -> Part:
    if label not in ANSWER_MODES:
        raise IdentityError(f"{label!r} is not one of perigraph's answer modes {ANSWER_MODES}")
    return Part(digest=_hex(label), label=label)


def _runtime(distributions) -> Part:
    rv = runtime_versions(distributions)
    return Part(digest=lines_digest(rv, "=="), label=",".join(f"{k}=={v}" for k, v in sorted(rv.items())))


# --- the backbone -----------------------------------------------------------------------------------------------------

#: Tokenizer and template files a Hugging Face loader may read. Present and listed: digested. A tokenizer file this
#: list does not know is refused below rather than silently left out.
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                    "vocab.json", "vocab.txt", "merges.txt", "tokenizer.model", "spiece.model",
                    "sentencepiece.bpe.model", "chat_template.jinja", "chat_template.json")
_CONFIG_FILES = ("config.json", "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json")
_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def checkpoint_root(model: str) -> Path:
    """The directory the loader reads: ``model`` itself when it is one, else the cached snapshot that ``config.json``
    resolves to without a revision, as the engine's own ``from_pretrained(model)`` does. Never downloads."""
    if os.path.isdir(model):
        return Path(model)
    from transformers.utils import cached_file

    path = cached_file(model, "config.json", local_files_only=True)
    if path is None:
        raise IdentityError(f"{model!r} has no cached config.json")
    return Path(path).parent


def backbone_files(root: Path) -> list[str]:
    """The files a ``from_pretrained`` of ``root`` reads, as far as this collector can name them.

    Weights follow the loader's own preference: a single ``model.safetensors`` over a sharded index. Adapters, remote
    code a configuration's ``auto_map`` names, and known tokenizer files are included. A layout this collector cannot
    name -- pickled ``.bin`` weights only, or a tokenizer file it does not know -- is refused, so the part is recorded
    as failed rather than as a confident identity over the wrong files."""
    names: list[str] = []
    if (root / "model.safetensors").is_file():
        names.append("model.safetensors")
    elif (root / "model.safetensors.index.json").is_file():
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        names += ["model.safetensors.index.json", *sorted(set(index["weight_map"].values()))]
    else:
        raise IdentityError(f"{root} has no safetensors weights this collector can name")
    names += [n for n in (*_CONFIG_FILES, *_TOKENIZER_FILES, *_ADAPTER_FILES) if (root / n).is_file()]
    config = json.loads((root / "config.json").read_text(encoding="utf-8")) if (root / "config.json").is_file() else {}
    if config.get("auto_map"):
        names += sorted(p.name for p in root.glob("*.py"))
    unknown = sorted(p.name for p in root.iterdir() if p.is_file() and p.name.endswith(".tiktoken"))
    if unknown:
        raise IdentityError(f"{root} has tokenizer files this collector does not read: {unknown}")
    return names


def _stats(root: Path, names) -> dict[str, tuple[int, int, int]]:
    out = {}
    for n in names:
        st = (root / n).resolve().stat()
        out[n] = (st.st_size, st.st_mtime_ns, st.st_ino)
    return out


@dataclass(frozen=True)
class Preload:
    """What the checkpoint looked like just before the loader opened it: the root, the file names and each file's
    size, modification time and inode. Hashing happens after the load, so a file that changed in between -- a
    trainer overwriting the directory, a cache pointer moved -- would give a digest of bytes that were not loaded;
    comparing against this turns that into a recorded failure. The file list itself is taken again after the load,
    because a loader can fetch files it reads (a processor config absent from the cache is downloaded into it)."""

    root: Path | None
    names: tuple[str, ...] = ()
    stats: dict = field(default_factory=dict)
    failure: str = ""


def preload(model: str) -> Preload:
    try:
        root = checkpoint_root(model)
        names = tuple(backbone_files(root))
        return Preload(root, names, _stats(root, names))
    except Exception as exc:  # noqa: BLE001 - recorded as the checkpoint part's failure
        return Preload(None, failure=f"{type(exc).__name__}: {exc}")


def _entries(root: Path, names, workers: int = 8) -> list[tuple[str, str]]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        shas = list(pool.map(lambda n: sha256_file((root / n).resolve()), names))
    return list(zip(names, shas, strict=True))


def _backbone_checkpoint(before: Preload, extra: list[tuple[str, str]]):
    if before.root is None:
        raise IdentityError(before.failure)
    names = backbone_files(before.root)
    gone = sorted(set(before.names) - set(names))
    if gone:
        raise IdentityError(f"files the loader saw are gone by hashing: {gone}")
    at_hash = _stats(before.root, names)
    entries = _entries(before.root, names)
    after = _stats(before.root, names)
    changed = sorted(n for n in names
                     if after[n] != at_hash[n] or (n in before.stats and before.stats[n] != at_hash[n]))
    if changed:
        raise IdentityError(f"files changed between load and hashing: {changed}")
    entries += extra
    return checkpoint_digest(entries), f"{len(entries)} files"


def _extra_files(heads, temperature) -> list[tuple[str, str]]:
    """Heads and a calibrated temperature are read from outside the checkpoint and change answers, so they are part of
    what answered. They are entered under ``+heads/`` and ``+temperature/``, which no file under an artifact root can
    be named, so a trainer that publishes a base and its heads together can compute the same artifact identity, and a
    reader can tell the base's files from the additions. A path is digested as a file; an in-memory spec as its
    canonical JSON."""
    out = []
    for kind, spec in (("heads", heads), ("temperature", temperature)):
        if spec is None:
            continue
        if isinstance(spec, (str, os.PathLike)) and Path(spec).is_file():
            out.append((f"+{kind}/{Path(spec).name}", sha256_file(spec)))
        else:
            out.append((f"+{kind}/inline.json", _hex(json.dumps(spec, sort_keys=True, separators=(",", ":")))))
    return out


def tensor_precision(modules, quantisation: str):
    """``dtype.<name>`` element counts over every parameter and buffer, and the device read off the tensors."""
    counts: dict[str, int] = {}
    devices: set[str] = set()
    for m in modules:
        for t in list(m.parameters()) + list(m.buffers()):
            name = str(t.dtype).replace("torch.", "")
            counts[f"dtype.{name}"] = counts.get(f"dtype.{name}", 0) + t.numel()
            devices.add(t.device.type)
    if not counts:
        raise IdentityError("no parameters were loaded")
    if len(devices) != 1:
        raise IdentityError(f"parameters sit on {sorted(devices)}; one device per model is what this part can say")
    fields = {**{k: str(v) for k, v in counts.items()}, "quantisation": quantisation, "device": devices.pop()}
    return precision_digest(fields), ",".join(f"{k}={v}" for k, v in sorted(fields.items()))


def quantisation_of(modules) -> str:
    """The scheme the loaded modules are in, read off their types and dtypes, never off configuration: an
    environment variable that asked for a conversion says nothing about whether it happened."""
    kinds = set()
    for m in modules:
        for sub in m.modules():
            name = type(sub).__name__
            if "Fp4" in name or "FP4" in name:
                kinds.add("nvfp4")
        for t in list(m.parameters()) + list(m.buffers()):
            if "float8" in str(t.dtype):
                kinds.add("fp8")
            elif "int8" in str(t.dtype) or "qint" in str(t.dtype):
                kinds.add("int8")
    return "+".join(sorted(kinds)) or "none"


def backbone_stamp(engine, *, before: Preload, heads=None, temperature=None) -> ModelStamp:
    """The engine's model half, read off what this process loaded. ``before`` is `preload`, taken before loading."""
    try:
        extra = _extra_files(heads, temperature)
        checkpoint = _try(_backbone_checkpoint, before, extra)
    except Exception as exc:  # noqa: BLE001 - the collector boundary
        checkpoint = Part(absent="extraction_failed", detail=f"{type(exc).__name__}: {exc}")
    modules = [engine.backbone]
    precision = _try(lambda: tensor_precision(modules, quantisation_of(modules)))
    return ModelStamp({"checkpoint": checkpoint, "precision": precision, "runtime": _runtime(BACKBONE_RUNTIME),
                       "answer_mode": _answer_mode("option_scores")})


# --- a student served by prismyra.learn ----------------------------------------------------------------------------

#: The fixed entry name a single-file student artifact is digested under, so the same bytes under another file name
#: are the same artifact, and a trainer that saved them can compute the identity without knowing where they are served.
STUDENT_ENTRY = "student.json"
#: What each tree library's student runs on: the library, the arrays it is handed, and the package that featurises.
STUDENT_RUNTIME = {"lightgbm": ("prismyra", "lightgbm", "numpy"), "xgboost": ("prismyra", "xgboost", "numpy")}


def student_stamp(raw: bytes, artifact) -> ModelStamp:
    """A student artifact's model half, from the bytes the loading process read and parsed (``raw``) -- one read, so
    the digest is of what was loaded rather than of whatever the path holds a moment later.

    ``precision`` is recorded as failed: a tree model exposes no parameter dtype to read, and writing the library's
    arithmetic as a constant would be a declaration marked as a reading. So a student has an artifact identity and
    no model identity. What its features compute with (a feature module's code, an embedding model) is not here."""
    runtime = STUDENT_RUNTIME.get(artifact.backend)
    return ModelStamp({
        "checkpoint": Part(digest=checkpoint_digest([(STUDENT_ENTRY, hashlib.sha256(raw).hexdigest())]),
                           label=f"{artifact.backend} student"),
        "precision": Part(absent="extraction_failed", detail="a tree model exposes no parameter dtype to read"),
        "runtime": _runtime(runtime) if runtime else Part(absent="extraction_failed",
                                                          detail=f"no runtime list for backend {artifact.backend!r}"),
        "answer_mode": _answer_mode("classifier_output"),
    })


def stand_in(why: str) -> ModelStamp:
    """Every part absent because this collector could not reach it, with why: for a model this process did not load
    from a file (a handle built in memory, a stand-in engine), a record that says so rather than none."""
    return ModelStamp({k: Part(absent="not_reachable", detail=why) for k in MODEL_PARTS})
