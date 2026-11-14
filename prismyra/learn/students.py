"""Loading a `prismyra.learn.fit.StudentArtifact` for stage 4's router, and nothing past that.

`prismyra.learn.fit` is deliberately "not imported by anything else in the package" (its own docstring); this
module is the one, narrow exception, and it is still careful about it -- the artifact's own `backend`
(`prismyra.learn.backends.StudentBackend`, `"lightgbm"` or `"xgboost"`) is what `StudentHandle.__init__`
resolves to a `prismyra.learn.backends.get_backend(...).load(...)` call, and that is the one place either
library (`learn`/`learn-xgb` extra) is imported -- so that loading `prismyra.learn.router`/`prismyra.learn.
hook` for a server that has no student configured for any tag never requires either to be installed.

A student is only ever read here from the file an admitted run of `python -m prismyra.learn.fit` wrote --
nothing in this module trains, re-evaluates or mutates one. The one check this module adds past
`StudentArtifact.from_json` is the one `fit.py` itself warns a caller would need: `admitted` is `True`. A
`RouterConfig.student_path` naming an artifact that failed its own admission bar is refused the same way a
malformed spec entry is -- loudly, at startup, not by quietly answering every request with Prismyra forever
(which would look like "the router is doing nothing" with no error to explain why). The one exception is a tag
whose admission is ``"external"``: an `AdmissionRecord` written outside Prismyra then decides instead of `admitted`.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from .backends import get_backend
from .fit import StudentArtifact
from .spec import AdmissionConfig, RouterConfig, SpecError, parse_router


class StudentError(Exception):
    """A tag's `RouterConfig.student_path` does not resolve to an admitted `StudentArtifact`, or resolves to
    one that was admitted for a narrower policy than the router asking to load it actually runs (see
    `StudentHandle`'s own check of `StudentArtifact.admitted_for`, below)."""


def student_version(artifact: StudentArtifact) -> str:
    """A short, content-addressed name for one artifact -- what `served_by` (`f"student:{...}"`, both in the
    HTTP response and in a logged `Experience.answered_by`) and `prismyra.learn.router.decide`'s own
    `student_version` parameter use to tell two fits of the *same* tag apart. Hashes `model_text` (the trained
    model itself) and `backend` (switching a tag from one `prismyra.learn.backends.StudentBackend` to another
    is as real a change as retraining on new data, even on the rare chance two different backends produced
    byte-identical `model_text`) together with `spec_version` (DISTILL-RL-DESIGN-v2.md section 2's "backbone
    が変わると、生徒...は、すべて無効になる" travels through `spec_version` because a backbone change forces
    a new spec load in practice) rather than trusting `created_at`, which two artifacts fit back-to-back on
    identical data could plausibly share to the second.
    """
    payload = f"{artifact.spec_version}\x00{artifact.backend}\x00{artifact.model_text}".encode()
    return hashlib.sha256(payload).hexdigest()[:12]


#: The artifact fields that decide what a student answers. Everything else (when it was fit, Prismyra's own verdict
#: and its numbers) is a statement about the artifact, not part of it.
_ANSWER_FIELDS = (
    "tag",
    "spec_version",
    "options",
    "kind",
    "dims",
    "idf",
    "model_text",
    "features",
    "feature_names",
    "backend",
    "backend_version",
    "text_tokenizer",
)


def artifact_version(artifact: StudentArtifact) -> str:
    """The version an outside reader binds evidence to, opaque to every reader but Prismyra: a digest of every field
    that decides the artifact's answers, the Prismyra package that runs it (featurisation is package code), and the
    feature module's name. The module's own code is not covered; a builder changed in place must be renamed."""
    from prismyra import __version__

    raw = artifact.to_json()
    payload = json.dumps({k: raw[k] for k in _ANSWER_FIELDS}, sort_keys=True, separators=(",", ":"))
    v = f"student:{hashlib.sha256(payload.encode()).hexdigest()[:32]}+package:{__version__}"
    return f"{v}+features:{artifact.features}" if artifact.features else v


ADMISSION_RECORD_FORMAT = 1
_HEX64 = re.compile(r"[0-9a-f]{64}")
_RECORD_KEYS = {"format", "tag", "version", "router", "admitted_by", "evidence"}


@dataclass(frozen=True)
class AdmissionRecord:
    """Someone outside Prismyra admitted one student artifact for one tag (`prismyra.learn.spec.AdmissionConfig`
    mode ``"external"``). Prismyra's own format:

        {"format": 1, "tag": "...", "version": "<artifact_version>", "router": {...a router block...},
         "admitted_by": "...", "evidence": {"uri": "...", "sha256": "<64 hex>"}}

    ``version`` is `artifact_version` of the one artifact admitted. ``router`` is the block the writer declared the
    artifact may be served under; Prismyra cannot know whether the judgement covered that block, only that the live
    block is the declared one, so a changed tau or mode needs a new record. Prismyra checks the tag, the version and
    the block, and nothing else; it keeps ``evidence`` so a reader can find what the judgement rests on.
    A record whose version names a router's served traffic rather than an artifact never matches: admit through the
    student's own version (``observations score``, or ``export --per-component``).
    """

    tag: str
    version: str
    router: RouterConfig
    admitted_by: str
    evidence: dict

    @staticmethod
    def load(path: Path) -> AdmissionRecord:
        def bad(why: str):
            return StudentError(f"admission record {path}: {why}")

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as e:
            raise bad(e.strerror or str(e)) from None
        except json.JSONDecodeError as e:
            raise bad(e.msg) from None
        if not isinstance(raw, dict) or type(raw.get("format")) is not int or raw["format"] != ADMISSION_RECORD_FORMAT:
            raise bad(f"'format' must be the integer {ADMISSION_RECORD_FORMAT}")
        if set(raw) != _RECORD_KEYS:
            raise bad(f"keys must be exactly {sorted(_RECORD_KEYS)}, got {sorted(raw)}")
        for k in ("tag", "version", "admitted_by"):
            if not isinstance(raw[k], str) or not raw[k]:
                raise bad(f"{k!r} must be a non-empty string")
        ev = raw["evidence"]
        if not (
            isinstance(ev, dict)
            and set(ev) == {"uri", "sha256"}
            and isinstance(ev["uri"], str)
            and ev["uri"]
            and isinstance(ev["sha256"], str)
            and _HEX64.fullmatch(ev["sha256"])
        ):
            raise bad("'evidence' must be {uri, sha256 (64 lowercase hex)}")
        if not isinstance(raw["router"], dict):
            raise bad("'router' must be a router block")
        try:
            router = parse_router(raw["router"], str(path))
        except SpecError as e:
            raise bad(f"its router block: {e}") from None
        return AdmissionRecord(
            tag=raw["tag"], version=raw["version"], router=router, admitted_by=raw["admitted_by"], evidence=dict(ev)
        )

    def check(self, artifact: StudentArtifact, router: RouterConfig | None) -> None:
        """Refuse unless this record admits exactly ``artifact`` under exactly ``router``."""
        if artifact.tag != self.tag:
            raise StudentError(f"tag {artifact.tag!r}: the admission record is for tag {self.tag!r}")
        mine = artifact_version(artifact)
        if mine != self.version:
            raise StudentError(
                f"tag {artifact.tag!r}: the artifact to serve is {mine!r}, but the admission record "
                f"(by {self.admitted_by!r}) admits {self.version!r}. A refit needs a new judgement"
            )
        if router != self.router:
            raise StudentError(
                f"tag {artifact.tag!r}: the live router block {router!r} is not the one the admission "
                f"record declares ({self.router!r})"
            )


class StudentHandle:
    """One loaded, admitted artifact, ready to answer `prismyra.learn.router.decide`'s `student_predict`
    thunk. `predict` loads this artifact's own backend and builds the model once, in `__init__` (eager, not
    lazy past construction): a `StudentHandle` is only ever constructed for a tag whose `RouterConfig` is
    already live, so the cost of deserializing the model is paid once, at server startup, rather than on this
    tag's first request.

    **An artifact admitted under PREREG-R2's `"confidence"` admission mode is a narrower claim than a
    full-admission one** (`prismyra.learn.fit`'s own module docstring addendum has the full reasoning):
    "this student's hybrid answer, deferring to Prismyra below `tau`, is good enough" is not the same claim as
    "this student is good enough to answer everything", and serving it under a router mode that *would*
    answer everything (``"ab"``, ``"ramp"``, ``"bandit"``, or even ``"confidence"`` at a looser ``tau`` than
    it was actually measured at) would silently cash a check this artifact never wrote. So `router`, when
    given, is checked against `artifact.admitted_for`: a `"confidence"`-admitted artifact is accepted only for
    a router itself configured in `"confidence"` mode with its own ``tau`` no lower than the one this artifact
    was admitted at (a *stricter* live ``tau`` only ever serves a subset of what was measured, never more);
    every other combination -- no router given at all, or one in any other mode, or `"confidence"` at a
    looser ``tau`` -- is refused the same loud way a not-admitted artifact already is. A full-admission
    artifact (`admitted_for` is `None`) is unaffected by any of this, exactly stage 4's original behaviour.

    **Under an `AdmissionRecord`** (a tag in admission mode ``"external"``) none of the above applies: the artifact's
    own `admitted` and `admitted_for` are not read, and the record alone decides, by `AdmissionRecord.check`.
    """

    def __init__(
        self,
        artifact: StudentArtifact,
        router: RouterConfig | None = None,
        record: AdmissionRecord | None = None,
        raw: bytes | None = None,
    ):
        if record is not None:
            # An outside judgement replaces Prismyra's own bar (`admitted`, `admitted_for`) entirely.
            record.check(artifact, router)
        elif not artifact.admitted:
            reasons = "; ".join(artifact.reasons) or "no reason recorded"
            raise StudentError(
                f"tag {artifact.tag!r}: this artifact was not admitted ({reasons}) -- fit a new one, or point "
                "the tag's 'router.student_path' at one that passed"
            )
        if record is None and artifact.admitted_for is not None and artifact.admitted_for.get("mode") == "confidence":
            admitted_tau = artifact.admitted_for.get("tau")
            served_as_confidence = (
                router is not None
                and router.mode == "confidence"
                and router.tau is not None
                and admitted_tau is not None
                and router.tau >= admitted_tau
            )
            if not served_as_confidence:
                router_desc = f"mode={router.mode!r}, tau={router.tau!r}" if router is not None else "unset"
                raise StudentError(
                    f"tag {artifact.tag!r}: this artifact was admitted only for 'confidence' mode at "
                    f"tau>={admitted_tau!r} (admitted_for={artifact.admitted_for!r}), but the router serving "
                    f"it is {router_desc} -- fit a new artifact admitted for that policy, or route it under "
                    "'confidence' at a tau no lower than the one it was admitted at"
                )
        self.artifact = artifact
        self.record = record
        self.version = student_version(artifact)
        #: perigraph's model half for this student (`prismyra.identity.student_stamp`), over ``raw``, the bytes the
        #: artifact was parsed from. A handle built from an artifact in memory has no bytes to digest and says so.
        from prismyra.identity import stand_in, student_stamp

        self.model_stamp = (
            student_stamp(raw, artifact)
            if raw is not None
            else stand_in("the artifact was handed over in memory, not read from bytes")
        )
        # see module docstring: the one place either backend's own library is required, and only here
        self._model = get_backend(artifact.backend).load(artifact.model_text)
        self._vectorizer = artifact.vectorizer()

    def transform(self, context: str, question: str = "") -> tuple[float, ...]:
        """This artifact's own feature vector for ``(context, question)`` -- what
        `prismyra.learn.hook.LearnHook._retained_context` stores as `experience.Experience.precomputed_features`
        for a tag whose `prismyra.learn.spec.TagSpec.retain_context` is ``"features_only"``, without this
        module exposing the `Vectorizer` itself for a caller outside this package to depend on."""
        return tuple(self._vectorizer.transform(context, question))

    def predict(self, context: str, question: str = "") -> dict[str, float]:
        """`StudentArtifact.predict_proba`'s own logic, minus rebuilding the model on every call."""
        import numpy as np

        vec = self.transform(context, question)
        (probs,) = self._model.predict_proba(np.asarray([vec], dtype=np.float64))
        return dict(zip(self.artifact.options, (float(p) for p in probs), strict=True))


class StudentRegistry:
    """One tag -> one `StudentHandle`, loaded once and kept for the server's lifetime. `prismyra.learn.hook.
    LearnHook` owns exactly one of these, built only when at least one tag's spec entry has a `router` block
    (the "no router configured, nothing past `prismyra.learn.fit` is ever imported" half of stage 4's own
    off-by-default contract -- see that module's docstring).
    """

    def __init__(self) -> None:
        self._handles: dict[str, StudentHandle] = {}
        self._lock = threading.Lock()

    def load(
        self,
        task: str,
        spec_dir: Path,
        student_path: str,
        router: RouterConfig | None = None,
        *,
        admission: AdmissionConfig | None,
    ) -> StudentHandle:
        """``router`` is the tag's own `RouterConfig` (its ``mode``/``tau``, specifically) -- passed through to
        `StudentHandle.__init__` unchanged, so it can refuse to load a `"confidence"`-admitted artifact under a
        router that would not actually serve it the way it was measured (see that class's own docstring).
        `None` (the default) is the full-admission-only check stage 4 always had -- every existing caller that
        does not pass this keeps meaning exactly what it meant before this parameter existed. ``admission`` is the
        tag's own `AdmissionConfig`, required so no caller can skip a record by omission; in mode ``"external"`` its
        record, resolved against ``spec_dir``, replaces the artifact's own verdict.
        """
        path = Path(student_path)
        if not path.is_absolute():
            path = spec_dir / path
        if not path.is_file():
            raise StudentError(f"tag {task!r}: router.student_path {student_path!r} does not exist (resolved: {path})")
        raw = path.read_bytes()
        artifact = StudentArtifact.from_json(json.loads(raw.decode("utf-8")))
        if artifact.tag != task:
            raise StudentError(
                f"tag {task!r}: router.student_path {student_path!r} is an artifact for tag {artifact.tag!r}"
            )
        record = None
        if admission is not None and admission.mode == "external":
            if admission.record is None:
                raise StudentError(f"tag {task!r}: admission mode 'external' names no record")
            rp = Path(admission.record)
            record = AdmissionRecord.load(rp if rp.is_absolute() else spec_dir / rp)
        handle = StudentHandle(artifact, router=router, record=record, raw=raw)
        with self._lock:
            self._handles[task] = handle
        return handle

    def get(self, task: str) -> StudentHandle | None:
        with self._lock:
            return self._handles.get(task)
