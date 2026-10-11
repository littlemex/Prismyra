"""The one object `prismyra.server` touches: load a spec, tag a request, and -- off the request path -- queue
what it learned. Everything stage 1/2 needs from a request handler's point of view is these three calls.

Constructing a `LearnHook` is the only place `prismyra.learn`'s heavier modules (`spec`, `tagging`, `experience`)
get imported from `prismyra.server`, and `prismyra.server` only constructs one inside the branch that already
has a ``--learn-spec`` path -- see that module for the no-op guarantee this exists to protect (DISTILL-RL-
DESIGN-v2.md section 8). Nothing in this module imports Ray; `experience.RayExperienceLogActor` does that
itself, and only when `backend="ray"` is asked for by name.
"""

from __future__ import annotations

import time
from pathlib import Path

from .experience import Experience, LocalExperienceLog, RayExperienceLogActor
from .spec import LearnSpec, TagSpec, load_spec
from .tagging import tag_for


class LearnHook:
    """Bound to one loaded spec and one running log. See `tag`, `submit` and `stats` -- the three methods
    `prismyra.server` calls."""

    def __init__(
        self,
        spec_path: str | Path,
        *,
        package_version: str,
        backbone: str,
        log_dir: str | Path | None = None,
        backend: str = "local",
        max_queue: int = 10_000,
    ):
        self.spec: LearnSpec = load_spec(spec_path)
        self.package_version = package_version
        self.backbone = backbone
        root = Path(log_dir) if log_dir is not None else self.spec.path.parent / "experience"
        if backend == "local":
            self.log: LocalExperienceLog | RayExperienceLogActor = LocalExperienceLog(
                root, self.spec, max_queue=max_queue
            ).start()
        elif backend == "ray":
            self.log = RayExperienceLogActor(root, self.spec, max_queue=max_queue)
        else:
            raise ValueError(f"backend must be 'local' or 'ray', got {backend!r}")

    def tag(self, *, task: str | None, question: str, options, kind: str) -> TagSpec | None:
        """The one comparison the request handler makes whether or not it turns out to match anything."""
        return tag_for(self.spec, task=task, question=question, options=options, kind=kind)

    def versions(self) -> dict[str, str]:
        return {"package": self.package_version, "backbone": self.backbone, "spec": self.spec.version}

    def submit(
        self,
        tag: TagSpec,
        *,
        context: str,
        question: str,
        options,
        kind: str,
        probabilities,
        answered_by: str,
        h: tuple[float, ...] | None = None,
    ) -> None:
        """Build and enqueue one `Experience`. Never blocks and never raises past a full queue -- see
        `experience.LocalExperienceLog.put`. Called from a FastAPI `BackgroundTasks` callback in
        `prismyra.server`, i.e. after the response has already been sent, which is what makes this safe to let
        take as long as a disk write takes without a caller ever waiting on it.
        """
        exp = Experience(
            tag=tag.task,
            spec_version=self.spec.version,
            ts=time.time(),
            context=context,
            question=question,
            options=tuple(options),
            kind=kind,
            probabilities=dict(probabilities),
            answered_by=answered_by,
            versions=self.versions(),
            h=h if tag.keep_hidden else None,
        )
        self.log.put(exp)

    def stats(self) -> dict:
        if isinstance(self.log, LocalExperienceLog):
            return self.log.stats.as_dict(self.log.queue_depth())
        return self.log.stats()  # RayExperienceLogActor

    def stop(self) -> None:
        self.log.stop()
