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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .spec import LearnSpec

#: A tag name is also a directory name (one subdirectory per tag, under the log root) and must survive that
#: without surprises -- no path separators, no leading dot, nothing that `Path` would read as "go up a directory".
#: `spec.load_spec` validates that `task` is a non-empty string; this is the narrower check this module needs
#: before using one as a path component.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class Experience:
    """One tagged request-and-answer, exactly as described above. ``h`` is `None` whenever the tag's
    ``keep_hidden`` is false, or when the serving path that answered this request does not support capturing it
    yet (``prismyra.learn.hook``'s own docstring names which paths those are, for stage 1/2)."""

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

    def to_json(self) -> dict:
        return {
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
        }

    @staticmethod
    def from_json(raw: dict) -> Experience:
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
        self.stats = LogStats()
        self._q: queue.Queue[Experience] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._open_files: dict[tuple[str, str], TextIO] = {}
        self._last_purge: dict[str, float] = {}
        self._lock = threading.Lock()  # guards stats + _open_files, both touched only here and on the bg thread
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
                self.stats.dropped += 1
            return False

    def queue_depth(self) -> int:
        return self._q.qsize()

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
                    self.stats.dropped += 1

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
            self.stats.logged += 1

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
        return self._inner.stats.as_dict(self._inner.queue_depth())

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
