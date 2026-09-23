"""Requests arrive one at a time; the device answers them several at a time.

`Batch` answers several documents in one forward pass, and a batch is not a scheduler. This is the scheduler: a queue,
one thread that owns the device, and a rule for which of the waiting requests go into the next pass.

**A pass waits a little for company, and the first version of this file argued that it should not.** That argument was
wrong, and how it was wrong is worth keeping. It said a batch fills from what has already arrived, so lingering only
changes the case where the queue is nearly empty -- and that is the case where the device is not the bottleneck anyway.
What it missed is that requests arriving *together* do not arrive at the same instant. The first one arrives some
microseconds ahead, finds nothing waiting, and starts a pass alone; the rest then travel in a second pass. Measured,
eight callers arriving together:

| | passes | total |
|---|---|---|
| no linger | one document, then seven | 640 ms |
| the same eight in one pass, in a loop | eight | **367 ms** |

The queue was not nearly empty. It was about to be full, and the scheduler could not tell the difference.

So there is a linger, and what bounds it is arithmetic rather than taste. Adding a document to a pass costs about 11 ms
per thousand of its tokens; a second pass costs a pass's fixed cost, about 110 ms, whatever it carries. **Waiting up to
that can pay for itself and waiting longer cannot**, because by then the pass could have run. That ceiling is
`Limits.worth_waiting_ms`, taken from the fastest read this engine has actually served.

`linger_ms` is not how long a pass waits. It is **how long a pass waits with nothing arriving**, and it refreshes every
time a request joins: a quiet stretch of that length ends the wait and a busy one extends it, up to the ceiling, which
does not move. So a lone request waits `linger_ms` and a burst is collected however long the burst takes, which is the
distinction a fixed wait cannot draw.

That shape was arrived at by breaking a fixed wait. Tokenising each document on the caller's thread rather than twice on
the device's -- an unrelated saving, worth 7 ms a pass -- spread the arrivals out, because eight callers tokenising take
their turns at it. Eight callers that had fitted inside a 2 ms wait no longer did, and the throughput fell from 20.37
requests a second back to 13.52 with the passes splitting into two again. **A wait that a 0.44 ms change breaks is tuned
rather than derived**, and a refreshing wait is not: it ends when arrivals do.

Three things bound a batch, and all three come from the engine rather than from a caller:

* **questions**, by the group: the batch answers in one pass and a pass has `group` rows;
* **documents**, by the group again, since every document needs at least one row;
* **tokens**, by what the page pool can hold at once.

A request that cannot fit any batch is refused when it is submitted, not when it reaches the front of the queue, because
a caller who will be refused should find out before waiting.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .queue import Job, QueueFull, Worker, WorkerStopped
from .schema import PrismyraError, Question, Result


@dataclass
class Request:
    """One caller's document and the questions they want answered about it.

    Not frozen, because `encoded` is filled in later. See there.
    """

    context: str
    questions: Sequence[Question]
    #: How the scheduler recognises two callers asking about the same document. A digest of the text rather than the
    #: text itself, because the key is compared on every request and a long document is a long comparison.
    digest: str = ""
    #: The document tokenised, filled the first time the scheduler needs to know how long it is and reused by the read.
    #: It used to be tokenised twice -- once to count it while forming a pass and once inside the read, 0.44 ms each, so
    #: 7 ms a pass on the one thread there is only one of.
    #:
    #: **Filled on the scheduler's thread rather than the caller's, and that was measured rather than assumed.** Moving
    #: it to the caller looks obviously better and is worse: encoding copies the ids to the device, and eight caller
    #: threads doing that take their turns at it, which spread the arrivals wider than any wait and split a pass of
    #: eight into two of four. Throughput went from 20.37 requests a second to 13.44. The work belongs where it is done
    #: once, not where there are more threads to do it on.
    encoded: Any = None


@dataclass(frozen=True)
class Limits:
    """What bounds a batch. Every figure is read off the engine; none of them is a knob.

    `tokens` is what the pool holds at once, which is the cache's own room. A batch of short documents is bounded by the
    group and a batch of long ones by this.
    """

    questions: int
    documents: int
    tokens: int
    #: How long a forming pass waits for another request before running. Small on purpose: what it covers is the jitter
    #: between callers that meant to arrive together, not the arrival of a request nobody has sent. Bounded by
    #: `worth_waiting_ms`, which is the fixed cost of a pass -- beyond that the pass could have run instead.
    linger_ms: float = 2.0

    @classmethod
    def of(cls, engine, linger_ms: float = 2.0) -> Limits:
        return cls(
            questions=engine.group,
            documents=engine.group,
            tokens=engine.longest_context,
            linger_ms=linger_ms,
        )

    def worth_waiting_ms(self, engine) -> float:
        """The longest a linger could pay for, from what this engine has measured. Zero until it has read anything.

        The fastest read this engine has served is the estimate of a pass's fixed cost: reading is that cost plus a
        slope in the tokens, so the shortest document seen is the closest thing to the intercept that exists without
        fitting a line. Zero when nothing has been read, which is the honest answer and also the safe one -- a scheduler
        with no evidence should not make anybody wait.
        """
        return engine.fastest_read_ms or 0.0


@dataclass
class Formed:
    """One pass's worth of requests, and why it stopped taking more."""

    jobs: list[Job] = field(default_factory=list)
    reason: str = "nothing waiting"

    @property
    def documents(self) -> int:
        return len(self.jobs)

    @property
    def questions(self) -> int:
        return sum(len(job.payload.questions) for job in self.jobs)


class Batcher:
    """A queue in front of one engine, answering several documents per pass.

    Requires the paged storage, because that is what lets rows of one pass belong to different documents.
    """

    def __init__(self, engine, max_queue: int = 512, linger_ms: float | None = None):
        if not getattr(engine, "paged", False):
            raise PrismyraError(
                "a batching scheduler needs the paged storage, because the joined storage keeps a document in one row "
                "that every row reads. Build the engine with Prismyra(..., paged=True)."
            )
        self.engine = engine
        self.limits = Limits.of(engine) if linger_ms is None else Limits.of(engine, linger_ms=linger_ms)
        self._worker = Worker(drive=self._drive, max_queue=max_queue)
        self._lock = threading.Lock()
        #: How wide each pass turned out to be, in order. The point of the whole exercise, so it is recorded rather than
        #: summarised: a mean of 1.0 and a mean of 8.0 are the difference between a scheduler and a queue.
        self.widths: list[int] = []
        #: Why each pass stopped taking requests. A scheduler whose passes are all narrow because of one limit is tuned
        #: differently from one that is narrow because nothing arrived, and the two are indistinguishable from the
        #: width.
        self.reasons: list[str] = []
        #: How many documents each pass had to read. Zero is the shelf working: every document in that pass was already
        #: on it, so the pass cost a branch and nothing else.
        self.reads: list[int] = []
        self._shelf = None
        #: Documents on the shelf, by digest, and the handle each one is known by.
        self._resident: dict[str, int] = {}
        self._digest_of: dict[int, str] = {}
        #: When each handle was last answered, for choosing what to drop. A counter rather than a clock: only the order
        #: matters and a counter cannot go backwards.
        self._used: dict[int, int] = {}
        self._clock = 0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> Batcher:
        self._worker.start()
        return self

    def stop(self, timeout: float = 30.0) -> None:
        self._worker.stop(timeout=timeout)
        if self._shelf is not None:
            self._shelf.close()
            self._shelf = None
        self._resident.clear()
        self._digest_of.clear()
        self._used.clear()

    def __enter__(self) -> Batcher:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ submitting
    def submit(self, context: str, questions: Sequence[Question]) -> Job:
        """Queue one document and its questions. Returns the job; wait on it for the answer.

        Refused here rather than at the front of the queue when it cannot fit any batch at all, because a caller who is
        going to be refused should not wait first.
        """
        if not questions:
            raise PrismyraError("a request needs at least one question")
        if len(questions) > self.limits.questions:
            raise PrismyraError(
                f"one request carries {len(questions)} questions and a pass has {self.limits.questions} rows. "
                f"Split it, or build the engine with a larger group."
            )
        self.engine.validate(list(questions))
        return self._worker.enqueue(Request(context=context, questions=tuple(questions), digest=_digest(context)))

    def ask(self, context: str, questions: Sequence[Question], timeout: float | None = None) -> Result:
        """Submit and wait. The convenience a caller wants when it has one thread per request."""
        job = self.submit(context, questions)
        if not job.done.wait(timeout):
            raise TimeoutError(f"no answer within {timeout} s")
        if job.error is not None:
            raise job.error
        return job.result

    # ------------------------------------------------------------------ the loop
    def form(self, first: Job) -> Formed:
        """The batch this job will travel in: itself, plus whatever else is already waiting and fits.

        Waits up to `Limits.linger_ms` for another request, and stops the moment the pass is full. The module
        docstring says why there is a wait at all and what bounds it: the first version of this method did not wait, and
        paid a whole pass's fixed cost to answer one request that seven others were microseconds behind.
        """
        formed = Formed(jobs=[first])
        tokens = self._tokens(first)
        # Bounded by what the engine has measured as well as by the setting. A linger longer than a pass's fixed cost
        # cannot pay, and before anything has been read there is no evidence for any wait at all.
        quiet = min(self.limits.linger_ms, self.limits.worth_waiting_ms(self.engine)) / 1e3
        ceiling = time.perf_counter() + self.limits.worth_waiting_ms(self.engine) / 1e3
        deadline = time.perf_counter() + quiet
        while True:
            if formed.documents >= self.limits.documents:
                formed.reason = f"the pass holds {self.limits.documents} documents"
                return formed
            nxt = self._worker.peek()
            if nxt is None:
                now = time.perf_counter()
                if now < deadline and now < ceiling:
                    # Nothing waiting yet, and there is still time for a straggler from the same burst.
                    time.sleep(0.0002)
                    continue
                formed.reason = "the ceiling was reached" if now >= ceiling else "nothing else was waiting"
                return formed
            questions = len(nxt.payload.questions)
            more = self._tokens(nxt)
            if formed.questions + questions > self.limits.questions:
                formed.reason = f"the next request's {questions} questions would pass {self.limits.questions} rows"
                return formed
            if tokens + more > self.limits.tokens:
                formed.reason = f"the next document's {more} tokens would pass {self.limits.tokens} in the pool"
                return formed
            taken = self._worker.take()
            if taken is None:  # pragma: no cover - another drain got there first
                formed.reason = "nothing else was waiting"
                return formed
            formed.jobs.append(taken)
            tokens += more
            # The wait refreshes when a request joins. A burst is defined by requests still arriving rather than by a
            # duration, so a quiet stretch ends the wait and a busy one extends it -- up to the ceiling, which does not
            # move.
            deadline = time.perf_counter() + quiet

    def _answer(self, formed: Formed) -> list:
        """Answer a formed pass, reading only the documents that are not already on the shelf.

        This is where the shelf earns its place. A document nobody has asked about before is read; a document already on
        the shelf costs nothing, so a second question about it is a branch pass and no more. The documents that do need
        reading are read **together**, because reading is mostly fixed cost.
        """
        shelf = self._on_shelf()
        # One entry per document, not per request. Three callers asking about the same document at the same moment are
        # three requests and one read -- without the de-duplication they were three reads of one document, each admitted
        # into its own pages, which is the waste this whole object exists to remove.
        fresh: dict[str, Job] = {}
        for job in formed.jobs:
            if job.payload.digest not in self._resident:
                fresh.setdefault(job.payload.digest, job)
        if fresh:
            jobs = list(fresh.values())
            self._make_room(jobs, keep={job.payload.digest for job in formed.jobs})
            handles = shelf.put_many([job.payload.context for job in jobs])
            for job, handle in zip(jobs, handles, strict=True):
                self._resident[job.payload.digest] = handle
                self._digest_of[handle] = job.payload.digest
        asked: dict[int, list[Question]] = {}
        for job in formed.jobs:
            handle = self._resident[job.payload.digest]
            self._used[handle] = self._clock
            self._clock += 1
            # Two callers asking about the same document in one pass would collide here, and one of them would get the
            # other's questions. They are merged instead -- one document, both callers' questions, one set of rows.
            asked.setdefault(handle, []).extend(job.payload.questions)
        answers = shelf.ask(asked)
        self.reads.append(len(fresh))
        return [answers[self._resident[job.payload.digest]] for job in formed.jobs]

    def _on_shelf(self):
        """The shelf, opened on the first pass rather than at construction, because opening it allocates."""
        if self._shelf is None:
            self._shelf = self.engine.open_shelf()
        return self._shelf

    def _make_room(self, fresh: list[Job], keep: set[str]) -> None:
        """Drop the least recently used documents until the new ones have somewhere to go.

        Least recently used, and never one in the pass being formed. What decides "enough room" is the pool, so this
        drops until the tokens fit the shelf's room rather than guessing at pages: the shelf refuses by name if the
        estimate is still wrong, and a refusal is recoverable where a wrong page is not.
        """
        shelf = self._on_shelf()
        wanted = sum(self._tokens(job) for job in fresh)
        while self._resident:
            held = sum(shelf.documents[handle].tokens for handle in shelf.documents)
            if held + wanted <= self.limits.tokens:
                return
            oldest = min(
                (handle for handle in shelf.documents if self._digest_of.get(handle) not in keep),
                key=lambda handle: self._used.get(handle, 0),
                default=None,
            )
            if oldest is None:
                # Everything on the shelf is in the pass being formed. Nothing can be dropped and the shelf will refuse
                # by name, which is the right outcome: the pass is too large for the pool.
                return
            shelf.drop(oldest)
            digest = self._digest_of.pop(oldest, None)
            if digest is not None:
                self._resident.pop(digest, None)
            self._used.pop(oldest, None)

    def _tokens(self, job: Job) -> int:
        """How long this document is, encoding it once and keeping the result for the read."""
        if job.payload.encoded is None:
            job.payload.encoded = self.engine.encode_context(job.payload.context)
        return job.payload.encoded.tokens

    def _drive(self, first: Job) -> None:
        """Form a batch around `first`, answer it, and give every job its own result."""
        formed = self.form(first)
        with self._lock:
            self.widths.append(formed.documents)
            self.reasons.append(formed.reason)
        try:
            results = self._answer(formed)
        except BaseException as e:  # noqa: BLE001 - one failure must not leave the rest of the batch waiting for ever
            for job in formed.jobs:
                job.started_at = job.started_at or first.started_at
                job.error = e
                job.finished_at = time.perf_counter()
                job.done.set()
            return
        depth = self._worker.depth
        for job, result in zip(formed.jobs, results, strict=True):
            # The others never went through the worker's own timing, so their start is the batch's start: they waited in
            # the queue until this pass began, which is what a queue time is supposed to say.
            job.started_at = job.started_at or first.started_at
            job.result = result
            job.finished_at = time.perf_counter()
            self._worker.note(job, depth)
            job.done.set()

    # ------------------------------------------------------------------ reporting
    def stats(self) -> dict:
        with self._lock:
            widths = list(self.widths)
            reasons = list(self.reasons)
            self.reads = list(self.reads)
        answered = sum(widths)
        return {
            "passes": len(widths),
            "documents": answered,
            "mean_documents_per_pass": round(answered / len(widths), 2) if widths else 0.0,
            "widths_seen": widths,
            "widest_pass": max(widths, default=0),
            "narrowest_pass": min(widths, default=0),
            # Counted, because a scheduler whose passes are narrow for lack of arrivals and one whose passes are narrow
            # because a limit binds look identical in the widths and need different work.
            "why_passes_stopped": {reason: reasons.count(reason) for reason in dict.fromkeys(reasons)},
            "limits": {
                "questions": self.limits.questions,
                "documents": self.limits.documents,
                "tokens": self.limits.tokens,
                "linger_ms": self.limits.linger_ms,
                "worth_waiting_ms": round(self.limits.worth_waiting_ms(self.engine), 1),
            },
            "documents_read": sum(self.reads),
            "documents_answered_without_reading": answered - sum(self.reads),
            "resident": len(self._resident),
            "queue": self._worker.stats(),
        }


__all__ = ["Batcher", "Formed", "Limits", "QueueFull", "Request", "WorkerStopped"]


def _digest(context: str) -> str:
    """How the scheduler recognises the same document twice.

    A digest of the text, not the text: the key is looked up on every request and compared against every resident
    document, and a long document is a long comparison. Collisions would answer about the wrong document, so this is a
    cryptographic digest rather than `hash`, which is randomised per process and truncated.
    """
    import hashlib

    return hashlib.sha256(context.encode("utf-8")).hexdigest()
