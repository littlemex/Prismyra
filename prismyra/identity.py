"""What answered: the model half of a perigraph version 2 record, written by the process that loaded the model.

perigraph (github.com/littlemex/perigraph, SPEC.md section 9) describes a model by four parts, each digested by the
process that loaded it (sourcing ``loaded_by_us``), and two identities over them:

| part | what this module digests |
|---|---|
| ``checkpoint`` | every file the loader read: ``<relative path>\\t<sha256>\\n`` lines, sorted by path bytes |
| ``precision`` | ``dtype.<name>=<count>``, ``quantisation``, ``device``, read off the loaded parameters |
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
#: The same for a student served by `prismyra.learn`: its tree library and what featurisation runs on.
STUDENT_RUNTIME = ("prismyra", "lightgbm", "xgboost", "numpy", "scikit-learn", "sentence-transformers")

#: Files outside the weights that a Hugging Face loader reads and that can change an answer: the configuration, the
#: tokenizer and its chat template, and a processor. A file listed here and absent from the checkpoint is skipped;
#: a file present and not listed is not read and not digested (an optimiser state, a stale shard).
_SIDE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
               "vocab.json", "merges.txt", "tokenizer.model", "chat_template.jinja", "chat_template.json",
               "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json")


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
        """The two identities as opaque version strings, the form perigraph section 11 recommends to an emitter."""
        out = {}
        if self.model_identity:
            out[f"{prefix}model_identity"] = f"perigraph/{SPEC_VERSION}/model/{self.model_identity}"
        if self.artifact_identity:
            out[f"{prefix}artifact_identity"] = f"perigraph/{SPEC_VERSION}/artifact/{self.artifact_identity}"
        return out


def _try(fn, *args) -> Part:
    try:
        digest, label = fn(*args)
        return Part(digest=digest, label=label)
    except (IdentityError, OSError, ValueError, KeyError) as exc:
        return Part(absent="extraction_failed", detail=f"{type(exc).__name__}: {exc}")


# --- the backbone -----------------------------------------------------------------------------------------------------

def checkpoint_root(model: str) -> Path:
    """The directory the loader read: ``model`` itself when it is one, else the cached Hugging Face snapshot that
    ``config.json`` resolved to. Never downloads; a model that is not on disk was not loaded from disk."""
    if os.path.isdir(model):
        return Path(model)
    from transformers.utils import cached_file

    path = cached_file(model, "config.json", local_files_only=True)
    if path is None:
        raise IdentityError(f"{model!r} has no cached config.json")
    return Path(path).parent


def backbone_files(root: Path) -> list[str]:
    """The weight files the index names (or the single ``model.safetensors``), plus the side files that exist."""
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weights = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
        names = ["model.safetensors.index.json", *weights]
    elif (root / "model.safetensors").is_file():
        names = ["model.safetensors"]
    else:
        raise IdentityError(f"{root} has neither model.safetensors.index.json nor model.safetensors")
    return names + [n for n in _SIDE_FILES if (root / n).is_file()]


def _entries(root: Path, names, workers: int = 8) -> list[tuple[str, str]]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        shas = list(pool.map(lambda n: sha256_file((root / n).resolve()), names))
    return list(zip(names, shas, strict=True))


def _backbone_checkpoint(model: str, extra: list[tuple[str, str]]):
    root = checkpoint_root(model)
    entries = _entries(root, backbone_files(root)) + extra
    return checkpoint_digest(entries), f"{model} ({len(entries)} files)"


def _extra_files(heads, temperature) -> list[tuple[str, str]]:
    """Heads and a calibrated temperature change answers and are read from outside the checkpoint, so they are part
    of what answered. A path is digested as a file; an in-memory spec as its canonical JSON."""
    out = []
    for kind, spec in (("heads", heads), ("temperature", temperature)):
        if spec is None:
            continue
        if isinstance(spec, (str, os.PathLike)):
            out.append((f"+{kind}/{Path(spec).name}", sha256_file(spec)))
        else:
            out.append((f"+{kind}/inline.json", _hex(json.dumps(spec, sort_keys=True, separators=(",", ":")))))
    return out


def tensor_precision(modules, device: str, quantisation: str):
    counts: dict[str, int] = {}
    for m in modules:
        for t in list(m.parameters()) + list(m.buffers()):
            name = str(t.dtype).replace("torch.", "")
            counts[f"dtype.{name}"] = counts.get(f"dtype.{name}", 0) + t.numel()
    if not counts:
        raise IdentityError("no parameters were loaded")
    fields = {**{k: str(v) for k, v in counts.items()}, "quantisation": quantisation, "device": device}
    return precision_digest(fields), ",".join(f"{k}={v}" for k, v in sorted(fields.items()))


def backbone_stamp(engine, *, model: str, heads=None, temperature=None) -> ModelStamp:
    """The engine's model half, read off what this process loaded."""
    try:
        extra = _extra_files(heads, temperature)
    except OSError as exc:
        extra, failure = [], f"OSError: {exc}"
    else:
        failure = ""
    checkpoint = (Part(absent="extraction_failed", detail=failure) if failure
                  else _try(_backbone_checkpoint, model, extra))
    quant = "nvfp4_experts" if os.environ.get("PRISMYRA_EXPERTS") == "nvfp4" else "none"
    if quant == "none" and any("float8" in str(p.dtype) for p in engine.backbone.parameters()):
        quant = "fp8_block"
    precision = _try(tensor_precision, [engine.backbone], engine.torch_device.type, quant)
    rv = runtime_versions(BACKBONE_RUNTIME)
    runtime = Part(digest=lines_digest(rv, "=="), label=",".join(f"{k}=={v}" for k, v in sorted(rv.items())))
    mode = Part(digest=_hex("option_scores"), label="option_scores")
    return ModelStamp({"checkpoint": checkpoint, "precision": precision, "runtime": runtime, "answer_mode": mode})


# --- a student served by prismyra.learn ----------------------------------------------------------------------------

def _student_checkpoint(path: Path):
    return checkpoint_digest([(path.name, sha256_file(path))]), path.name


def _student_precision(artifact):
    """A tree model's parameters are its leaf values, held as float64 by both supported libraries."""
    text = artifact.model_text
    if artifact.backend == "lightgbm":
        n = sum(len(line.split("=", 1)[1].split()) for line in text.splitlines() if line.startswith("leaf_value="))
    else:
        raise IdentityError(f"leaf values of a {artifact.backend!r} model are not counted by this collector")
    if n == 0:
        raise IdentityError("the model text has no leaf values")
    fields = {"dtype.float64": str(n), "quantisation": "none", "device": "cpu"}
    return precision_digest(fields), f"{artifact.backend},leaf_values={n}"


def student_stamp(path: str | Path, artifact) -> ModelStamp:
    """A student artifact's model half, read by the process that loaded it from ``path``."""
    path = Path(path)
    rv = runtime_versions(STUDENT_RUNTIME)
    return ModelStamp({
        "checkpoint": _try(_student_checkpoint, path),
        "precision": _try(_student_precision, artifact),
        "runtime": Part(digest=lines_digest(rv, "=="), label=",".join(f"{k}=={v}" for k, v in sorted(rv.items()))),
        "answer_mode": Part(digest=_hex("classifier_output"), label="classifier_output"),
    })


def unreachable(detail: str) -> ModelStamp:
    """Every part absent because this collector could not reach it, with why: a record that says so, not none."""
    return ModelStamp({k: Part(absent="not_reachable", detail=detail) for k in MODEL_PARTS})
