"""One experience, and where it goes. DISTILL-RL-DESIGN-v2.md section 2:

    1 件の経験 = (tag, spec の版, 入力, 特徴, Prismyra の確率, 誰が答えたか, 版の組, 報酬かラベル (あれば),
    h (spec が許すときだけ))

Two things that field list says are deliberately *not* separate columns here:

* **"特徴" (features) is not stored.** Section 3 is explicit that a feature is built from the input alone
  ("特徴は入力だけから作る"), at fit time, by whatever scheme ``prismyra.learn.fit`` is using that day. Storing a
  pre-computed feature vector in the log would freeze today's feature scheme into yesterday's data: a later
  change to hashing-dimension or to how a game's JSON gets flattened would leave old records speaking a format
  the new code does not read, for no benefit, since the raw input is already enough to recompute them. What is
  stored is the input itself -- enough to rebuild any feature scheme that is ever written against it.
* **"報酬かラベル (あれば)" is a field that is usually absent.** Nothing at serve time knows whether an answer was
  right; that only exists for a tag's ``eval_set`` or for a separate labeling pass run later. The field exists in
  the schema below so that a correction pipeline has somewhere to write it, not because stage 1/2 ever populates
  it on the request path.

This module has two halves. ``Experience`` is the record shape and its JSON encoding, used by every backend.
``LocalExperienceLog`` is the one backend this stage requires: a bounded queue plus one background thread, so a
burst of tagged traffic drops experience rather than ever blocking a response (section 8: "queue が満ちたら、ログ
を捨てて配信を優先する"). ``RayExperienceLogActor`` is the Ray-backed alternative the design allows ("Ray の
actor にするのは、Ray があるときの選択肢にする") but does not require; importing *this module* never imports
``ray`` -- only constructing a ``RayExperienceLogActor`` does, and nothing in ``prismyra.server`` constructs one
unless an operator asks for the ``ray`` backend by name.
"""

from __future__ import annotations

import calendar
import json
import queue
import re
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from .spec import LearnSpec

#: A tag name is also a directory name (one subdirectory per tag, under the log root) and must survive that
#: without surprises -- no path separators, no leading dot, nothing that `Path` would read as "go up a directory".
#: `spec.load_spec` validates that `task` is a non-empty string; this is the narrower check this module needs
#: before using one as a path component.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


#: This module's own schema version, bumped from the implicit "1" every row written before stage 4 existed
#: carried (none of them wrote the field at all). `Experience.from_json` reads a missing field as `1`, so
#: every stage 1/2 log on disk today stays readable without being rewritten -- see that method.
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Experience:
    """One tagged request-and-answer, exactly as described above. ``h`` is `None` whenever the tag's
    ``keep_hidden`` is false, or when the serving path that answered this request does not support capturing it
    yet (``prismyra.learn.hook``'s own docstring names which paths those are, for stage 1/2).

    Stage 4 (DISTILL-RL-DESIGN-v2.md section 4) adds everything below ``h``, all defaulted so a schema-1 row
    deserialises with every new field `None` rather than refusing to load:

    * ``request_id`` -- the id this same request's response carried back to the caller (`served_by` in the
      HTTP response), the join key `POST /v1/feedback` and `prismyra.learn.fit` use to attach a label or a
      reward collected after the fact. Empty string, not `None`, for a schema-1 row: that is what
      "no id was ever minted for this row" means for a field whose later readers index by it.
    * ``answered_by`` -- already existed, and already carries exactly what stage 4 needs: ``"prismyra"`` or,
      now, ``f"student:{artifact_version}"`` (`prismyra.learn.students.student_version`). Nothing new here.
    * ``route_mode``/``route_bucket``/``route_params_version`` -- which strategy decided this row's route, the
      hash bucket it fell into, and `prismyra.learn.router.params_version`'s name for the ratio/tau snapshot
      in effect -- together, enough to tell whether two rows were routed under comparable conditions.
    * ``comparison_probabilities`` -- the *other* side's probabilities, when this row was audited (section 4's
      bias-and-quality audit) or came from a confidence-mode fallback: the student's, if ``answered_by`` is
      ``"prismyra"``; Prismyra's, if ``answered_by`` starts with ``"student"``. `None` otherwise -- most rows,
      since auditing every request would cost what stage 4 exists to avoid costing.
    * ``feature_summary`` -- `prismyra.learn.features.feature_summary`'s small hashed-bucket histogram of the
      input, kept regardless of ``retain_context`` (it is lossy and cannot be un-hashed back into the text),
      because it is what `prismyra.learn.controller.Controller`'s drift check compares across time.
    * ``precomputed_features`` -- set only when `prismyra.learn.spec.TagSpec.retain_context` is
      ``"features_only"`` and a student (and so a fitted `Vectorizer`) already existed for this tag at submit
      time: the one thing worth keeping once the raw context itself is not.
    """

    tag: str
    spec_version: str
    ts: float
    context: str
    question: str
    options: tuple[str, ...]
    kind: str
    probabilities: Mapping[str, float]
    answered_by: str
    versions: Mapping[str, str]
    reward_or_label: object | None = None
    h: tuple[float, ...] | None = None
    request_id: str = ""
    route_mode: str | None = None
    route_bucket: int | None = None
    route_params_version: str | None = None
    comparison_probabilities: Mapping[str, float] | None = None
    feature_summary: tuple[int, ...] | None = None
    precomputed_features: tuple[float, ...] | None = None

    def to_json(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "tag": self.tag,
            "spec_version": self.spec_version,
            "ts": self.ts,
            "context": self.context,
            "question": self.question,
            "options": list(self.options),
            "kind": self.kind,
            "probabilities": dict(self.probabilities),
            "answered_by": self.answered_by,
            "versions": dict(self.versions),
            "reward_or_label": self.reward_or_label,
            "h": list(self.h) if self.h is not None else None,
            "request_id": self.request_id,
            "route_mode": self.route_mode,
            "route_bucket": self.route_bucket,
            "route_params_version": self.route_params_version,
            "comparison_probabilities": (
                dict(self.comparison_probabilities) if self.comparison_probabilities is not None else None
            ),
            "feature_summary": list(self.feature_summary) if self.feature_summary is not None else None,
            "precomputed_features": list(self.precomputed_features) if self.precomputed_features is not None else None,
        }

    @staticmethod
    def from_json(raw: dict) -> Experience:
        comparison = raw.get("comparison_probabilities")
        summary = raw.get("feature_summary")
        precomputed = raw.get("precomputed_features")
        return Experience(
            tag=raw["tag"],
            spec_version=raw["spec_version"],
            ts=raw["ts"],
            context=raw["context"],
            question=raw["question"],
            options=tuple(raw["options"]),
            kind=raw["kind"],
            probabilities=dict(raw["probabilities"]),
            answered_by=raw["answered_by"],
            versions=dict(raw["versions"]),
            reward_or_label=raw.get("reward_or_label"),
            h=tuple(raw["h"]) if raw.get("h") is not None else None,
            # `.get(..., "1"...)`: a row with no `schema_version` at all predates stage 4, and every field
            # from here down predates it too -- read as absent, not as an error.
            request_id=raw.get("request_id", ""),
            route_mode=raw.get("route_mode"),
            route_bucket=raw.get("route_bucket"),
            route_params_version=raw.get("route_params_version"),
            comparison_probabilities=dict(comparison) if comparison is not None else None,
            feature_summary=tuple(summary) if summary is not None else None,
            precomputed_features=tuple(precomputed) if precomputed is not None else None,
        )


def _date_str(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def _safe_tag_dir(root: Path, tag: str) -> Path:
    if not _SAFE_NAME.match(tag):
        raise ValueError(f"tag {tag!r} is not safe to use as a directory name")
    return root / tag


@dataclass
class LogStats:
    """What ``/stats`` reports (design section 3's "ログの件数、捨てた数、queue の深さ")."""

    logged: int = 0
    dropped: int = 0

    def as_dict(self, queue_depth: int) -> dict:
        return {"logged": self.logged, "dropped": self.dropped, "queue_depth": queue_depth}


class ExperienceSink(Protocol):
    """Where a tagged request's `Experience` goes once it has been built -- the interface
    `prismyra.learn.hook.LearnHook` writes to, so that *which* sink a server uses is a configuration choice,
    not a fact `LearnHook`'s own logic has to know. `LocalExperienceLog` (below) and `RayExperienceLogActor`
    already have exactly this shape, unchanged by this `Protocol` existing; `prismyra.learn.otel_sink.
    OtelExperienceSink` is the second implementation, and `MultiSink` (below) composes any number of them.

    Every implementation keeps three promises section 8 already made for the one sink stage 1/2 had: a
    bounded queue, a full queue drops rather than blocks, and writing happens off the request path (a
    background thread, a background task, or both).
    """

    def put(self, exp: Experience) -> bool: ...

    def queue_depth(self) -> int: ...

    def stats(self) -> dict: ...

    def stop(self) -> None: ...


class MultiSink:
    """Fans one `Experience` out to every sink in ``sinks`` -- "ローカルにも書き、OTel にも出す" at once,
    each still keeping its own bounded queue and its own drop accounting. The *first* sink's own `stats()`
    is reported under this object's own flat keys (`"logged"`/`"dropped"`/`"queue_depth"`), so a caller that
    only ever configured one sink (every tag before this existed) sees exactly the shape it always did;
    every sink's own numbers, by name, are additionally reported under `"sinks"`.
    """

    def __init__(self, sinks: Mapping[str, ExperienceSink]):
        if not sinks:
            raise ValueError("MultiSink needs at least one sink")
        self._sinks = dict(sinks)
        self._primary_name = next(iter(self._sinks))

    def put(self, exp: Experience) -> bool:
        # Every sink gets the attempt, independently -- one sink being full or erroring must not stop the
        # others from receiving the same experience (the whole point of configuring more than one). A list
        # comprehension, not `all(...)` over a generator: the latter short-circuits on the first `False` and
        # would skip calling `put` on every sink after it.
        results = [sink.put(exp) for sink in self._sinks.values()]
        return all(results)

    def queue_depth(self) -> int:
        return self._sinks[self._primary_name].queue_depth()

    def stats(self) -> dict:
        primary = self._sinks[self._primary_name].stats()
        return {**primary, "sinks": {name: sink.stats() for name, sink in self._sinks.items()}}

    def stop(self) -> None:
        for sink in self._sinks.values():
            sink.stop()


class LocalExperienceLog:
    """A bounded in-memory queue, drained by one background thread into one append-only JSONL file per
    ``(tag, UTC date)``, under ``root/<tag>/<YYYY-MM-DD>.jsonl``.

    ``put`` is the only method the request path calls, and it never blocks and never raises: a full queue
    increments ``dropped`` and returns `False`, exactly the choice `schedule.Batcher` already makes for a full
    admission queue (its own docstring: refuse fast rather than make the caller, or anyone behind them, wait).
    Writing to disk happens only on the background thread, so a slow or momentarily-unavailable filesystem
    delays nothing a caller is waiting on.
    """

    def __init__(self, root: str | Path, spec: LearnSpec, *, max_queue: int = 10_000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self._counts = LogStats()
        self._q: queue.Queue[Experience] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._open_files: dict[tuple[str, str], TextIO] = {}
        self._last_purge: dict[str, float] = {}
        self._lock = threading.Lock()  # guards _counts + _open_files, both touched only here and on the bg thread
        self._thread = threading.Thread(target=self._run, name="prismyra-learn-log", daemon=True)

    def start(self) -> LocalExperienceLog:
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)
        with self._lock:
            for fh in self._open_files.values():
                fh.close()
            self._open_files.clear()

    def put(self, exp: Experience) -> bool:
        try:
            self._q.put_nowait(exp)
            return True
        except queue.Full:
            with self._lock:
                self._counts.dropped += 1
            return False

    def queue_depth(self) -> int:
        return self._q.qsize()

    def stats(self) -> dict:
        with self._lock:
            return self._counts.as_dict(self.queue_depth())

    def _run(self) -> None:
        while not self._stop.is_set() or not self._q.empty():
            try:
                exp = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._write(exp)
                self._maybe_purge(exp.tag)
            except OSError:
                # A write failure must not kill the thread -- the next experience, and the next tag's purge,
                # still deserve a chance. Dropped-on-write is counted the same way dropped-on-full-queue is: both
                # mean "this experience did not make it to disk".
                with self._lock:
                    self._counts.dropped += 1

    def _write(self, exp: Experience) -> None:
        tag_dir = _safe_tag_dir(self.root, exp.tag)
        tag_dir.mkdir(parents=True, exist_ok=True)
        key = (exp.tag, _date_str(exp.ts))
        fh = self._open_files.get(key)
        if fh is None:
            fh = (tag_dir / f"{key[1]}.jsonl").open("a", encoding="utf-8")
            with self._lock:
                self._open_files[key] = fh
        fh.write(json.dumps(exp.to_json(), ensure_ascii=False) + "\n")
        fh.flush()
        with self._lock:
            self._counts.logged += 1

    def _maybe_purge(self, tag: str, *, min_interval_s: float = 60.0) -> None:
        """Delete date-partitioned files older than this tag's ``retain_days``. Rate-limited per tag so a hot tag
        does not pay for a directory listing on every single write.
        """
        now = time.time()
        last = self._last_purge.get(tag, 0.0)
        if now - last < min_interval_s:
            return
        self._last_purge[tag] = now
        entry = self.spec.by_task(tag)
        if entry is None:
            return
        cutoff = now - entry.retain_days * 86400
        tag_dir = _safe_tag_dir(self.root, tag)
        if not tag_dir.is_dir():
            return
        for p in tag_dir.glob("*.jsonl"):
            try:
                file_date = time.strptime(p.stem, "%Y-%m-%d")
            except ValueError:
                continue
            # `calendar.timegm`, not `time.mktime`: the filename is a UTC calendar date (`_date_str` builds it
            # from `time.gmtime`), and `mktime` would read it back as a *local* midnight, drifting the cutoff
            # by this host's UTC offset for no reason a filename format should ever depend on.
            file_ts = calendar.timegm(file_date)
            if file_ts < cutoff and (tag, p.stem) not in self._open_files:
                p.unlink(missing_ok=True)


@dataclass(frozen=True)
class FeedbackRecord:
    """One ``POST /v1/feedback`` call, joined back to its `Experience` row later by `request_id` -- the gold
    label or reward DISTILL-RL-DESIGN-v2.md section 2 describes as "報酬かラベル (あれば)" but, before stage 4,
    had nowhere to arrive from except by hand-editing a log. Exactly one of ``label``/``reward`` is set: a
    feedback item is either a classification gold answer (`fit.py`'s own `eval_set` vocabulary) or a scalar
    reward, never both, and never neither -- `prismyra.learn.hook.LearnHook.submit_feedback` is what enforces
    that before one of these is ever constructed.
    """

    request_id: str
    ts: float
    label: str | None = None
    reward: float | None = None

    def to_json(self) -> dict:
        return {"request_id": self.request_id, "ts": self.ts, "label": self.label, "reward": self.reward}

    @staticmethod
    def from_json(raw: dict) -> FeedbackRecord:
        return FeedbackRecord(
            request_id=raw["request_id"], ts=raw["ts"], label=raw.get("label"), reward=raw.get("reward")
        )


class FeedbackLog:
    """`LocalExperienceLog`'s own bounded-queue-plus-one-thread shape, reused for feedback instead of
    experience: a full queue drops rather than blocks (the same "never make `POST /v1/feedback`'s caller wait
    on a disk write" reasoning section 8 already applies to the request path), and writing happens only on
    the background thread. One file per UTC date under ``root`` directly -- feedback has no tag of its own
    (a caller supplies only ``request_id``; which tag it belongs to is only known by joining it back to an
    `Experience` row, which is `prismyra.learn.fit`'s job, not this log's), so there is no per-tag directory
    and no `retain_days`-driven purge: a feedback row this small, read only by an offline join, is left for an
    operator's own retention policy on the log directory as a whole rather than this module inventing a second
    one.
    """

    def __init__(self, root: str | Path, *, max_queue: int = 10_000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._counts = LogStats()
        self._q: queue.Queue[FeedbackRecord] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._open_files: dict[str, TextIO] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="prismyra-learn-feedback", daemon=True)

    def start(self) -> FeedbackLog:
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)
        with self._lock:
            for fh in self._open_files.values():
                fh.close()
            self._open_files.clear()

    def put(self, rec: FeedbackRecord) -> bool:
        try:
            self._q.put_nowait(rec)
            return True
        except queue.Full:
            with self._lock:
                self._counts.dropped += 1
            return False

    def queue_depth(self) -> int:
        return self._q.qsize()

    def stats(self) -> dict:
        with self._lock:
            return self._counts.as_dict(self.queue_depth())

    def _run(self) -> None:
        while not self._stop.is_set() or not self._q.empty():
            try:
                rec = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._write(rec)
            except OSError:
                with self._lock:
                    self._counts.dropped += 1

    def _write(self, rec: FeedbackRecord) -> None:
        date = _date_str(rec.ts)
        fh = self._open_files.get(date)
        if fh is None:
            fh = (self.root / f"{date}.jsonl").open("a", encoding="utf-8")
            with self._lock:
                self._open_files[date] = fh
        fh.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
        fh.flush()
        with self._lock:
            self._counts.logged += 1


def load_feedback(root: str | Path) -> dict[str, FeedbackRecord]:
    """Every `FeedbackRecord` under ``root`` (a `FeedbackLog`'s own directory), keyed by `request_id`, latest
    write wins -- `prismyra.learn.fit`'s join step, and `prismyra.learn.idle`'s idle-time retraining, both
    read this once per run rather than re-implementing the same directory scan.
    """
    root = Path(root)
    out: dict[str, FeedbackRecord] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                stripped = raw_line.strip()
                if stripped:
                    rec = FeedbackRecord.from_json(json.loads(stripped))
                    out[rec.request_id] = rec
    return out


class ExperienceSource(Protocol):
    """The read side of `ExperienceSink`: where `prismyra.learn.fit` and `prismyra.learn.idle` get the
    `Experience` rows they train on, decoupled from *how* those rows are stored. `LocalExperienceSource`
    (below) reads the same on-disk layout `LocalExperienceLog` writes; a second implementation can read
    from anywhere else that can produce the same objects -- `prismyra.learn.http_source.
    HttpPullExperienceSource` is the one this project ships, a generic paginated HTTP/JSON puller rather
    than one tied to a specific destination's own API, because the read side should not have to be rewritten
    for every place `ExperienceSink` could plausibly have written to.

    ``since`` filters to rows with ``Experience.ts >= since`` when given; every implementation applies the
    same filter, so a caller joining feedback by `request_id` (`load_feedback`, above) gets the same
    join regardless of which source produced the rows on the other side of it.
    """

    def iter_experiences(self, task: str, *, since: float | None = None) -> Iterator[Experience]: ...


class LocalExperienceSource:
    """Reads `LocalExperienceLog`'s own on-disk layout: ``root/<task>/<YYYY-MM-DD>.jsonl``, oldest file
    first. This is `prismyra.learn.fit`'s original, inline directory scan, promoted to a named class that
    implements `ExperienceSource` -- every existing caller that passed a bare `log_dir` path keeps working
    unchanged; `run()` builds one of these itself when no other `ExperienceSource` is given.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def iter_experiences(self, task: str, *, since: float | None = None) -> Iterator[Experience]:
        tag_dir = self.root / task
        if not tag_dir.is_dir():
            return
        for path in sorted(tag_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    exp = Experience.from_json(json.loads(stripped))
                    if since is None or exp.ts >= since:
                        yield exp


class _RayLogActorBody:
    """The actor's own methods, as a plain class. Not decorated with ``@ray.remote`` at class-definition time --
    that would run ``ray.remote`` (and so touch the ``ray`` import machinery) the moment this *module* is
    imported, for every caller, whether or not they ever ask for the Ray backend. The decoration happens inside
    ``RayExperienceLogActor.__init__`` instead, after ``ray`` has already been imported there on purpose.
    """

    def __init__(self, root: str | Path, spec: LearnSpec, max_queue: int):
        self._inner = LocalExperienceLog(root, spec, max_queue=max_queue).start()

    def put(self, exp: Experience) -> bool:
        return self._inner.put(exp)

    def queue_depth(self) -> int:
        return self._inner.queue_depth()

    def stats(self) -> dict:
        return self._inner.stats()

    def stop(self) -> None:
        self._inner.stop()


class RayExperienceLogActor:
    """The same write-and-purge logic as `LocalExperienceLog`, running inside a Ray actor, for an operator who
    already has a Ray cluster and would rather the log lived there than on this process's own disk.

    Importing this *class* costs nothing extra; constructing one does `import ray`, which is why this class
    exists in the same module as the import-free default rather than being imported unconditionally by
    ``prismyra.learn.hook`` -- see that module's own docstring for exactly when (never, for stage 1/2's own
    default) this gets constructed. ``put`` is fire-and-forget (``.remote()`` without a matching ``ray.get``):
    the request path already cannot block on this for the local backend, and waiting on an actor call over the
    Ray wire would be strictly worse, not equivalent, so the two backends share that property rather than only
    sharing an interface.
    """

    def __init__(self, root: str | Path, spec: LearnSpec, *, max_queue: int = 10_000):
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        actor_cls = ray.remote(_RayLogActorBody)
        self._handle = actor_cls.remote(root, spec, max_queue)

    def put(self, exp: Experience) -> bool:
        self._handle.put.remote(exp)  # fire-and-forget: see class docstring
        return True

    def queue_depth(self) -> int:
        import ray

        return ray.get(self._handle.queue_depth.remote())

    def stats(self) -> dict:
        import ray

        return ray.get(self._handle.stats.remote())

    def stop(self) -> None:
        import ray

        ray.get(self._handle.stop.remote())
