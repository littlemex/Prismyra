"""Loading a `prismyra.learn.fit.StudentArtifact` for stage 4's router, and nothing past that.

`prismyra.learn.fit` is deliberately "not imported by anything else in the package" (its own docstring); this
module is the one, narrow exception, and it is still careful about it -- `lightgbm` (the `learn` extra) is
imported lazily, inside `StudentHandle.predict`, exactly like `fit.py`'s own `StudentArtifact.predict_proba`
already does, so that loading `prismyra.learn.router`/`prismyra.learn.hook` for a server that has no student
configured for any tag never requires the extra to be installed.

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

from .fit import StudentArtifact


class StudentError(Exception):
    """A tag's `RouterConfig.student_path` does not resolve to an admitted `StudentArtifact`."""


def student_version(artifact: StudentArtifact) -> str:
    """A short, content-addressed name for one artifact -- what `served_by` (`f"student:{...}"`, both in the
    HTTP response and in a logged `Experience.answered_by`) and `prismyra.learn.router.decide`'s own
    `student_version` parameter use to tell two fits of the *same* tag apart. Hashes `model_text` (the trained
    booster itself) together with `spec_version` (DISTILL-RL-DESIGN-v2.md section 2's "backbone が変わると、
    生徒...は、すべて無効になる" travels through `spec_version` because a backbone change forces a new spec
    load in practice) rather than trusting `created_at`, which two artifacts fit back-to-back on identical
    data could plausibly share to the second.
    """
    payload = f"{artifact.spec_version}\x00{artifact.model_text}".encode()
    return hashlib.sha256(payload).hexdigest()[:12]


class StudentHandle:
    """One loaded, admitted artifact, ready to answer `prismyra.learn.router.decide`'s `student_predict`
    thunk. `predict` loads `lightgbm` and builds the `Booster` once, in `__init__` (eager, not lazy past
    construction): a `StudentHandle` is only ever constructed for a tag whose `RouterConfig` is already live,
    so the cost of building the booster is paid once, at server startup, rather than on this tag's first
    request.
    """

    def __init__(self, artifact: StudentArtifact):
        if not artifact.admitted:
            reasons = "; ".join(artifact.reasons) or "no reason recorded"
            raise StudentError(
                f"tag {artifact.tag!r}: this artifact was not admitted ({reasons}) -- fit a new one, or point "
                "the tag's 'router.student_path' at one that passed"
            )
        self.artifact = artifact
        self.version = student_version(artifact)
        import lightgbm as lgb  # see module docstring: the one place this extra is required, and only here

        self._booster = lgb.Booster(model_str=artifact.model_text)
        self._vectorizer = artifact.vectorizer()

    def transform(self, context: str, question: str = "") -> tuple[float, ...]:
        """This artifact's own feature vector for ``(context, question)`` -- what
        `prismyra.learn.hook.LearnHook._retained_context` stores as `experience.Experience.precomputed_features`
        for a tag whose `prismyra.learn.spec.TagSpec.retain_context` is ``"features_only"``, without this
        module exposing the `Vectorizer` itself for a caller outside this package to depend on."""
        return tuple(self._vectorizer.transform(context, question))

    def predict(self, context: str, question: str = "") -> dict[str, float]:
        """`StudentArtifact.predict_proba`'s own logic, minus rebuilding the booster on every call."""
        import numpy as np

        vec = self.transform(context, question)
        (probs,) = self._booster.predict(np.asarray([vec], dtype=np.float64))
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

    def load(self, task: str, spec_dir: Path, student_path: str) -> StudentHandle:
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
        handle = StudentHandle(artifact)
        with self._lock:
            self._handles[task] = handle
        return handle

    def get(self, task: str) -> StudentHandle | None:
        with self._lock:
            return self._handles.get(task)
