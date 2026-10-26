"""The one object `prismyra.server` touches: load a spec, tag a request, route it (stage 4, when a tag opted
in), and -- off the request path -- queue what it learned. Everything stage 1/2 needed from a request
handler's point of view was three calls (`tag`, `submit`, `stats`); stage 4 adds two more (`route`,
`record_route`) and a feedback side channel (`submit_feedback`), none of which change what the first three do
for a tag that never configured a `prismyra.learn.spec.RouterConfig`.

Constructing a `LearnHook` is the only place `prismyra.learn`'s heavier modules get imported from
`prismyra.server` -- see that module for the no-op guarantee this exists to protect (DISTILL-RL-DESIGN-v2.md
section 8). `prismyra.learn.router`, `.controller` and `.students` are themselves only imported, and a
`prismyra.learn.controller.Controller`/`prismyra.learn.students.StudentRegistry` only constructed, when at
least one tag's entry has a `"router"` block -- a spec with none keeps paying exactly stage 1/2's own cost.
Nothing in this module imports Ray; `experience.RayExperienceLogActor` does that itself, and only when
``backend="ray"`` is asked for by name.
"""

from __future__ import annotations

import time
from pathlib import Path

from .experience import (
    Experience,
    ExperienceSink,
    FeedbackLog,
    FeedbackRecord,
    LocalExperienceLog,
    MultiSink,
    RayExperienceLogActor,
)
from .spec import LearnSpec, TagSpec, load_spec
from .tagging import tag_for


def _argmax(probs) -> str:
    """`prismyra.learn.fit.argmax`'s own one-liner, kept local rather than importing that module here: `fit.py`
    is "deliberately not imported by anything else in the package" (its own docstring) -- this avoids being
    the one exception for a single comparison."""
    return max(probs.items(), key=lambda kv: kv[1])[0]


class LearnHook:
    """Bound to one loaded spec and one running log. See `tag`, `submit`, `route`, `record_route`,
    `submit_feedback` and `stats` -- the methods `prismyra.server` calls."""

    def __init__(
        self,
        spec_path: str | Path,
        *,
        package_version: str,
        backbone: str,
        log_dir: str | Path | None = None,
        backend: str = "local",
        max_queue: int = 10_000,
        otel_endpoint: str | None = None,
        otel_service_name: str = "prismyra",
    ):
        """``backend`` picks the primary `prismyra.learn.experience.ExperienceSink` -- ``"local"``
        (`LocalExperienceLog`, unchanged) or ``"ray"`` (`RayExperienceLogActor`), exactly as before
        ``otel_endpoint`` existed. ``otel_endpoint``, when given, adds a second sink
        (`prismyra.learn.otel_sink.OtelExperienceSink`) alongside the primary one -- "ローカルにも書き、OTel
        にも出す" at once, via `prismyra.learn.experience.MultiSink` -- rather than replacing it; `None` (the
        default) imports nothing under `prismyra.learn.otel_sink` and costs nothing past the one attribute
        check, the same off-by-default discipline every other optional piece of this package keeps.
        """
        self.spec: LearnSpec = load_spec(spec_path)
        self.package_version = package_version
        self.backbone = backbone
        root = Path(log_dir) if log_dir is not None else self.spec.path.parent / "experience"
        if backend == "local":
            primary: ExperienceSink = LocalExperienceLog(root, self.spec, max_queue=max_queue).start()
        elif backend == "ray":
            primary = RayExperienceLogActor(root, self.spec, max_queue=max_queue)
        else:
            raise ValueError(f"backend must be 'local' or 'ray', got {backend!r}")

        if otel_endpoint is None:
            self.log: ExperienceSink = primary
        else:
            from .otel_sink import OtelExperienceSink

            otel = OtelExperienceSink(endpoint=otel_endpoint, service_name=otel_service_name, max_queue=max_queue)
            self.log = MultiSink({"local" if backend == "local" else backend: primary, "otel": otel})
        self.feedback = FeedbackLog(root / "_feedback", max_queue=max_queue).start()

        # Stage 4: imported, and built, only for a spec that actually has a tag asking for it -- a spec with
        # none of this stays exactly stage 1/2's own footprint.
        self.controller = None
        self._students = None
        routed = [t for t in self.spec.tags if t.router is not None]
        if routed:
            from .controller import Controller
            from .students import StudentRegistry

            self.controller = Controller(root / "_controller" / "state.json")
            self._students = StudentRegistry()
            for t in routed:
                assert t.router is not None
                self._students.load(t.task, self.spec.path.parent, t.router.student_path)

    def tag(self, *, task: str | None, question: str, options, kind: str) -> TagSpec | None:
        """The one comparison the request handler makes whether or not it turns out to match anything."""
        return tag_for(self.spec, task=task, question=question, options=options, kind=kind)

    def versions(self) -> dict[str, str]:
        return {"package": self.package_version, "backbone": self.backbone, "spec": self.spec.version}

    def _retained_context(self, tag: TagSpec, context: str) -> tuple[str, tuple[float, ...] | None]:
        """`prismyra.learn.spec.TagSpec.retain_context`'s own enforcement point -- the only place either
        `submit` or `record_route` is allowed to write `context`/`precomputed_features` into an `Experience`,
        so the rule cannot be applied in one of the two call sites and forgotten in the other.
        """
        if tag.retain_context == "full":
            return context, None
        if tag.retain_context == "features_only" and self._students is not None:
            handle = self._students.get(tag.task)
            if handle is not None:
                return "", handle.transform(context, tag.question)
        return "", None

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

        This is stage 1/2's own entry point, unchanged: a tag with no `router` configured (every tag before
        stage 4, and every tag since that has not opted in) is logged exactly as it always was. `retain_context`
        is the one new thing every tag -- routed or not -- gets here, because the privacy question it answers
        (DISTILL-RL-DESIGN-v2.md section 8's note that this is "既存の retain_days と keep_hidden と同じ" a
        spec-level choice) has nothing to do with whether that tag also routes traffic.
        """
        from .features import feature_summary

        stored_context, precomputed = self._retained_context(tag, context)
        exp = Experience(
            tag=tag.task,
            spec_version=self.spec.version,
            ts=time.time(),
            context=stored_context,
            question=question,
            options=tuple(options),
            kind=kind,
            probabilities=dict(probabilities),
            answered_by=answered_by,
            versions=self.versions(),
            h=h if tag.keep_hidden else None,
            feature_summary=feature_summary(context),
            precomputed_features=precomputed,
        )
        self.log.put(exp)

    def route(self, tag: TagSpec, *, question: str, options, kind: str, context: str):
        """``None`` when ``tag.router`` is unset -- the request goes to Prismyra exactly as stage 1/2 always
        sent it, at no cost past the one attribute lookup. Otherwise a `prismyra.learn.router.RouteOutcome`,
        computed *before* the engine is ever asked, so `prismyra.server` can skip the device entirely for a
        request the outcome says the student alone will answer (``served_by`` starts with ``"student"`` and
        ``needs_engine`` is `False`).
        """
        if tag.router is None or self._students is None or self.controller is None:
            return None
        handle = self._students.get(tag.task)
        if handle is None:
            return None  # defensive: __init__ already loads every routed tag's student or refuses to start

        from . import router as router_mod

        normalized_input = context.strip()
        ratio, audit_fraction = self.controller.resolve(tag, tag.router)
        return router_mod.decide(
            tag,
            tag.router,
            normalized_input=normalized_input,
            ratio=ratio,
            audit_fraction=audit_fraction,
            student_version=handle.version,
            student_predict=lambda: handle.predict(context, tag.question),
        )

    def record_route(
        self,
        tag: TagSpec,
        outcome,
        *,
        context: str,
        question: str,
        options,
        kind: str,
        engine_probabilities: dict[str, float] | None,
    ) -> None:
        """`route`'s own `BackgroundTasks` counterpart: logs the `Experience` row stage 4 adds fields for, and
        folds the outcome into `prismyra.learn.controller.Controller`'s ramp/bandit/rollback bookkeeping.
        ``engine_probabilities`` is Prismyra's own answer, present exactly when `outcome.needs_engine` was
        `True` -- the served answer, if ``outcome.served_by == "prismyra"``; an audit-only comparison, if
        ``outcome.served_by`` started with ``"student"`` and `outcome.audited` was also `True`.
        """
        from .features import feature_summary

        assert tag.router is not None
        served_by_student = outcome.served_by.startswith("student")
        final_probabilities = outcome.answer_probabilities if served_by_student else engine_probabilities
        if final_probabilities is None:
            return  # nothing was actually answered (should not happen; logging must never raise past this)

        comparison_probabilities = None
        agreement: bool | None = None
        if served_by_student:
            if outcome.audited and engine_probabilities is not None:
                comparison_probabilities = engine_probabilities
                agreement = _argmax(outcome.answer_probabilities) == _argmax(engine_probabilities)
        elif outcome.student_probabilities is not None:
            comparison_probabilities = outcome.student_probabilities
            agreement = _argmax(engine_probabilities) == _argmax(outcome.student_probabilities)

        drift_vector = feature_summary(context)
        stored_context, precomputed = self._retained_context(tag, context)
        exp = Experience(
            tag=tag.task,
            spec_version=self.spec.version,
            ts=time.time(),
            context=stored_context,
            question=question,
            options=tuple(options),
            kind=kind,
            probabilities=dict(final_probabilities),
            answered_by=outcome.served_by,
            versions=self.versions(),
            h=None,
            request_id=outcome.request_id,
            route_mode=outcome.mode,
            route_bucket=outcome.bucket,
            route_params_version=outcome.params_version,
            comparison_probabilities=dict(comparison_probabilities) if comparison_probabilities is not None else None,
            feature_summary=drift_vector,
            precomputed_features=precomputed,
        )
        self.log.put(exp)
        assert self.controller is not None
        self.controller.record(tag, tag.router, outcome, agreement=agreement, drift_vector=drift_vector)

    def reload_student(self, task: str):
        """``POST /v1/learn/reload``'s whole implementation: re-read ``task``'s `RouterConfig.student_path`
        from disk and swap it into `prismyra.learn.students.StudentRegistry` -- the hot-swap half of idle-time
        retraining's own loop (`prismyra.learn.idle` fits and admits a new artifact out of band, writes it to
        exactly that path, then calls this). Raises `prismyra.learn.spec.SpecError` for an unregistered task,
        `ValueError` for one with no router, and `prismyra.learn.students.StudentError` for everything
        `StudentRegistry.load` itself refuses -- `prismyra.server` maps all three to a 4xx, not a 500: every
        one of them is "this reload was refused", never a bug in the server that is still running the student
        it had before this call.
        """
        tag = self.spec.by_task(task)
        if tag is None:
            from .spec import SpecError

            raise SpecError(f"no tag named {task!r} in this spec")
        if tag.router is None or self._students is None:
            raise ValueError(f"tag {task!r} has no router configured")
        return self._students.load(task, self.spec.path.parent, tag.router.student_path)

    def submit_feedback(self, request_id: str, *, label: str | None = None, reward: float | None = None) -> bool:
        """``POST /v1/feedback``'s whole implementation: append one `FeedbackRecord`, off the request path,
        joined back to its `Experience` row by ``request_id`` only later, at training time
        (`prismyra.learn.experience.load_feedback`). Exactly one of ``label``/``reward`` must be set -- this
        is the one place that is enforced, so a malformed call fails loudly at the API boundary rather than
        writing an ambiguous row a later reader has to guess about.
        """
        if (label is None) == (reward is None):
            raise ValueError("submit_feedback needs exactly one of 'label' or 'reward'")
        if not request_id:
            raise ValueError("submit_feedback needs a non-empty request_id")
        return self.feedback.put(FeedbackRecord(request_id=request_id, ts=time.time(), label=label, reward=reward))

    def stats(self) -> dict:
        out = self.log.stats()  # every ExperienceSink implementation shares this one method shape
        out["feedback"] = self.feedback.stats()
        if self.controller is not None:
            routing: dict[str, dict] = {}
            for t in self.spec.tags:
                if t.router is None:
                    continue
                ratio, audit_fraction = self.controller.peek(t, t.router)
                routing[t.task] = {"mode": t.router.mode, "ratio": ratio, "audit_fraction": audit_fraction}
            out["routing"] = routing
        return out

    def stop(self) -> None:
        self.log.stop()
        self.feedback.stop()
        if self.controller is not None:
            self.controller.flush()
