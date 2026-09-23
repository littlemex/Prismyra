"""Requests arrive one at a time; the device answers them several at a time.

`Batch` answers several documents in one forward pass, and a batch is not a scheduler. This is the scheduler: a queue,
one thread that owns the device, and a rule for which of the waiting requests go into the next pass.

**The rule is work-conserving, and that is a decision rather than an omission.** A batching scheduler usually has a
"linger" knob -- wait a few milliseconds in case more arrives -- and the reason there is none here is that the case it
helps is the case where batching does not matter. Lingering only changes anything when the queue is nearly empty,
because when requests are waiting there is nothing to wait for: the batch fills from what has already arrived. And when
the queue is nearly empty the device is not the bottleneck, so a batch of one is fine. The loop takes what is there.

What a linger would buy is bounded, and the bound is worth writing down. Adding a document to a pass costs about 11 ms
per thousand of its tokens; starting a second pass costs about 110 ms whatever it carries. So waiting up to the
fixed cost of a pass can pay in total work, and waiting longer than that cannot -- by then the pass could have run. If a
linger is ever added, `Limits.worth_waiting_ms` is that ceiling, and it is derived from what this engine has measured
rather than chosen.

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

from .queue import Job, QueueFull, Worker, WorkerStopped
from .schema import PrismyraError, Question, Result


@dataclass(frozen=True)
class Request:
    """One caller's document and the questions they want answered about it."""

    context: str
    questions: Sequence[Question]


@dataclass(frozen=True)
class Limits:
    """What bounds a batch. Every figure is read off the engine; none of them is a knob.

    `tokens` is what the pool holds at once, which is the cache's own room. A batch of short documents is bounded by the
    group and a batch of long ones by this.
    """

    questions: int
    documents: int
    tokens: int

    @classmethod
    def of(cls, engine) -> Limits:
        return cls(questions=engine.group, documents=engine.group, tokens=engine.longest_context)

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

    def __init__(self, engine, max_queue: int = 512):
        if not getattr(engine, "paged", False):
            raise PrismyraError(
                "a batching scheduler needs the paged storage, because the joined storage keeps a document in one row "
                "that every row reads. Build the engine with Prismyra(..., paged=True)."
            )
        self.engine = engine
        self.limits = Limits.of(engine)
        self._worker = Worker(drive=self._drive, max_queue=max_queue)
        self._lock = threading.Lock()
        #: How wide each pass turned out to be, in order. The point of the whole exercise, so it is recorded rather than
        #: summarised: a mean of 1.0 and a mean of 8.0 are the difference between a scheduler and a queue.
        self.widths: list[int] = []
        #: Why each pass stopped taking requests. A scheduler whose passes are all narrow because of one limit is tuned
        #: differently from one that is narrow because nothing arrived, and the two are indistinguishable from the
        #: width.
        self.reasons: list[str] = []

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> Batcher:
        self._worker.start()
        return self

    def stop(self, timeout: float = 30.0) -> None:
        self._worker.stop(timeout=timeout)

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
        return self._worker.enqueue(Request(context=context, questions=tuple(questions)))

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

        Nothing is waited for. See the module docstring: lingering only changes the case where the queue is nearly
        empty, and that is the case where a batch of one is fine.
        """
        formed = Formed(jobs=[first])
        tokens = self._tokens(first)
        while True:
            if formed.documents >= self.limits.documents:
                formed.reason = f"the pass holds {self.limits.documents} documents"
                return formed
            nxt = self._worker.peek()
            if nxt is None:
                formed.reason = "nothing else was waiting"
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

    def _tokens(self, job: Job) -> int:
        return len(self.engine.tokenizer(job.payload.context)["input_ids"])

    def _drive(self, first: Job) -> None:
        """Form a batch around `first`, answer it, and give every job its own result."""
        formed = self.form(first)
        with self._lock:
            self.widths.append(formed.documents)
            self.reasons.append(formed.reason)
        try:
            with self.engine.open_batch([job.payload.context for job in formed.jobs]) as batch:
                results = batch.ask([list(job.payload.questions) for job in formed.jobs])
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
        answered = sum(widths)
        return {
            "passes": len(widths),
            "documents": answered,
            "mean_documents_per_pass": round(answered / len(widths), 2) if widths else 0.0,
            "widest_pass": max(widths, default=0),
            "narrowest_pass": min(widths, default=0),
            # Counted, because a scheduler whose passes are narrow for lack of arrivals and one whose passes are narrow
            # because a limit binds look identical in the widths and need different work.
            "why_passes_stopped": {reason: reasons.count(reason) for reason in dict.fromkeys(reasons)},
            "limits": {
                "questions": self.limits.questions,
                "documents": self.limits.documents,
                "tokens": self.limits.tokens,
            },
            "queue": self._worker.stats(),
        }


__all__ = ["Batcher", "Formed", "Limits", "QueueFull", "Request", "WorkerStopped"]
