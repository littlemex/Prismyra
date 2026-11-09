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
(which would look like "the router is doing nothing" with no error to explain why).
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

from .backends import get_backend
from .fit import StudentArtifact
from .spec import RouterConfig


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
    """

    def __init__(self, artifact: StudentArtifact, router: RouterConfig | None = None):
        if not artifact.admitted:
            reasons = "; ".join(artifact.reasons) or "no reason recorded"
            raise StudentError(
                f"tag {artifact.tag!r}: this artifact was not admitted ({reasons}) -- fit a new one, or point "
                "the tag's 'router.student_path' at one that passed"
            )
        if artifact.admitted_for is not None and artifact.admitted_for.get("mode") == "confidence":
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
        self.version = student_version(artifact)
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

    def load(self, task: str, spec_dir: Path, student_path: str, router: RouterConfig | None = None) -> StudentHandle:
        """``router`` is the tag's own `RouterConfig` (its ``mode``/``tau``, specifically) -- passed through to
        `StudentHandle.__init__` unchanged, so it can refuse to load a `"confidence"`-admitted artifact under a
        router that would not actually serve it the way it was measured (see that class's own docstring).
        `None` (the default) is the full-admission-only check stage 4 always had -- every existing caller that
        does not pass this keeps meaning exactly what it meant before this parameter existed.
        """
        path = Path(student_path)
        if not path.is_absolute():
            path = spec_dir / path
        if not path.is_file():
            raise StudentError(f"tag {task!r}: router.student_path {student_path!r} does not exist (resolved: {path})")
        import json

        artifact = StudentArtifact.from_json(json.loads(path.read_text(encoding="utf-8")))
        if artifact.tag != task:
            raise StudentError(
                f"tag {task!r}: router.student_path {student_path!r} is an artifact for tag {artifact.tag!r}"
            )
        handle = StudentHandle(artifact, router=router)
        with self._lock:
            self._handles[task] = handle
        return handle

    def get(self, task: str) -> StudentHandle | None:
        with self._lock:
            return self._handles.get(task)
