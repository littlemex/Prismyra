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

import torch

from .engine import _round_rows
from .queue import Job, QueueFull, Worker, WorkerStopped
from .schema import PrismyraError, Question, Result

#: Free device memory `_make_room` keeps clear of shelf residents, for the same reason `engine.GRAPH_MEMORY_MARGIN`
#: keeps it clear of a new recording: so the ordinary pass every call still runs is not the thing that starves.
#: `_make_room` turns this into a count by comparing it against `Batcher._slot_bytes` -- the margin alone only
#: answers "is there room for one more", and a shelf's non-page cost is not bounded by anything that asks "how many
#: are there already" without it. See `_make_room`.
SHELF_MEMORY_MARGIN = 2 * 1024**3

#: The same question `MAX_KEPT_RECORDINGS` asks for recordings, asked here for shelf residents: the margin above
#: only ever fires when *admitting a new document* finds the device short, which never happens while the token
#: budget and the memory margin both still have room to spare -- a shelf can and does sit on dozens of short
#: residents without ever being asked to drop one. Measured (THROUGHPUT.md 2026-10-05, the lane=2 go/no-go
#: harness's own rate=10/n=80 burst, `mem_breakdown_candidates.py`/`diag_shelf_resident_cap.py`): with no cap,
#: resident count came out at 33-37 across repeated runs of the identical scenario (seeded, but real wall-clock
#: Poisson arrivals make the exact eviction moment timing-sensitive) and *dominated* the free-memory variance --
#: capping residents at 24/16/8 measured free-after-`empty_cache()` of 3.132/3.966/5.079 GiB against the no-cap
#: run's 3.288 GiB, with no measured questions/second cost (34.7-35.3 q/s across all four, inside noise). The full
#: multi-rate open-loop benchmark's own natural steady state (THROUGHPUT.md "常駐文書数は到着率ごとに1〜8") never
#: approaches this cap, so it only ever fires on the kind of short, bursty accumulation the margin alone missed --
#: the same shape of gap `MAX_KEPT_RECORDINGS` closed for recordings. 16, a 2x safety factor over that observed
#: natural ceiling of 8, so it does not bind the ordinary case and only catches the transient this was added for.
SHELF_MAX_RESIDENTS = 16

#: Same constant and same reasoning as `engine.RECLAIM_COOLDOWN_S` -- see there. Kept separate (not imported) so
#: `schedule.py` does not need `engine.py` for a number that is really "how long a cheap, repeated CUDA sync is
#: worth skipping", not a fact about the engine.
RECLAIM_COOLDOWN_S = 1.0


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

    def __init__(
        self,
        engine,
        max_queue: int = 512,
        linger_ms: float | None = None,
        lanes: int = 1,
        lane_room: int | None = None,
        _lane_id: int = 0,
    ):
        """`lanes>1` (THROUGHPUT.md 2026-10-05, "本当に効く経路"): the single-worker-thread bottleneck a rate=10
        open-loop run exposed is not a tuning question, it is that one `Worker` thread runs one pass fully before
        starting the next. `lanes` builds that many fully independent (`Shelf`/`Pool`/`Worker`) stacks under one
        router, so two passes can be in flight on the device at once -- each lane is otherwise identical to the
        `lanes=1` object this already was, which is what keeps this safe to add: a lane's `Shelf` is never shared
        with another lane's, so the only new sharing between lanes is the model's weights (read-only) and, for the
        engine's own forking machinery, a lane-tagged `fork.OWNED` buffer (`Prismyra._owned`'s `lane` argument) --
        see `prismyra.engine.Prismyra.open_shelf`'s docstring for why a lane's `Shelf` is its own `Pool`/cache and
        not a bigger one shared out.

        `lane_room` is the token budget each lane's own shelf is opened with (default's reasoning: `open_shelf`'s
        own default, sized for being the *only* shelf, double-costs when there are two -- measured infeasible for
        the full-size default, see `diag_two_shelves_memory.py`). A document is routed to a lane by a hash of its
        digest, so repeat questions about the same document always land on the same lane and keep the shelf-hit
        benefit `Batcher` exists for.

        A document arriving at a lane that is momentarily busy still queues *for that lane only* -- `lanes` buys
        two lanes each as fast as one lane was, not one lane twice as fast, which is the whole reason to measure
        it against a single-lane baseline rather than assume it from the design.
        """
        self._lane_id = _lane_id
        if lanes > 1:
            if _lane_id != 0:
                raise PrismyraError("lanes>1 only builds the outer router; a lane does not build further lanes")
            self._children: list[Batcher] | None = [
                Batcher(engine, max_queue=max_queue, linger_ms=linger_ms, lanes=1, lane_room=lane_room, _lane_id=i)
                for i in range(lanes)
            ]
            self.engine = engine
            return
        self._children = None
        if not getattr(engine, "paged", False):
            raise PrismyraError(
                "a batching scheduler needs the paged storage, because the joined storage keeps a document in one row "
                "that every row reads. Build the engine with Prismyra(..., paged=True)."
            )
        #: Token budget this lane's own shelf is opened with when it first opens one (`_on_shelf`). `None` keeps
        #: `open_shelf`'s own default (the largest bucket admission accepts) -- unchanged from before `lanes`
        #: existed, for a lane-less (`lanes=1`, the default) `Batcher`.
        self._lane_room = lane_room
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
        #: The largest recurrent-state snapshot a document on this shelf has actually cost, in bytes. Zero until one
        #: has been read, which is honest: nothing has been measured yet, the same reason `fastest_read_ms` starts at
        #: `None` rather than a guess. Updated from what `Shelf.put_many` actually allocated for each new resident,
        #: not estimated from the document's length -- a recurrent layer's state does not depend on it. See
        #: `_make_room`.
        self._slot_bytes = 0
        #: Monotonic deadline before `_make_room` tries `empty_cache()` again, once an attempt has already left
        #: free memory at or below `SHELF_MEMORY_MARGIN`. See `RECLAIM_COOLDOWN_S`.
        self._reclaim_cooldown_until = 0.0
        #: fp8spd (S5 / SYNTHESIS.md): measurement only, see `engine.Prismyra._empty_cache_calls`.
        #: fp8spd3 (round 5): how many passes took the fused single-fresh-document path, out of how many passes
        #: total -- so a q/s measurement that shows little effect can be told apart from one where the path
        #: rarely engages at all. See `RUN-fp8spd.md` round 5.
        self._fused_single_passes = 0
        #: fp8spd6 (round 7): how many passes took the generalised fused path -- every document in the pass
        #: fresh, two or more of them -- out of the same `_total_passes` denominator above. See
        #: `RUN-fp8spd.md` round 7, "本題": this is what raises the single-document path's own 10.4% fusion
        #: rate under congestion, where a pass's companions are themselves almost always fresh documents too.
        self._fused_many_passes = 0
        self._total_passes = 0
        self._empty_cache_calls = 0
        self._empty_cache_ms = 0.0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> Batcher:
        if self._children is not None:
            for child in self._children:
                child.start()
            return self
        self._worker.start()
        return self

    def stop(self, timeout: float = 30.0) -> None:
        if self._children is not None:
            for child in self._children:
                child.stop(timeout=timeout)
            return
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

        `lanes>1`: routed to one lane by a hash of the document's digest, before any of the per-lane checks below
        run (a lane's own `submit` repeats them, so nothing here is skipped, just decided which lane does it) --
        stable so the same document always lands on the same lane and keeps its shelf-hit benefit across requests.
        """
        if self._children is not None:
            lane = int(_digest(context)[:8], 16) % len(self._children)
            return self._children[lane].submit(context, questions)
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

        That bound is a ceiling, not a bet, and the bet is what `backlogged` adds. `first` was already taken off the
        queue before this runs, so `self._worker.depth` here is the number of *other* jobs already waiting at this
        instant -- not a guess, a fact observed one line ago. Zero of them is not weak evidence that a companion is
        imminent; over an open-loop run against real documents (`THROUGHPUT.md`, "到着率10の負けの切り分け"), lingering
        on that non-evidence raised the median end-to-end latency at a sparse arrival rate (10 documents/s) from
        386.8ms to 436.9ms and dropped the deadline-hit rate from 0.875 to 0.713 -- waiting for a companion that
        usually was not coming, at everyone's expense once in a while when the wait ran long. One or more already
        waiting is real evidence, and at a busier rate (20 documents/s) skipping the wait entirely made the median
        latency worse (496.8ms to 625.7ms) and the hit rate worse (0.506 to 0.206): there, a companion usually was
        coming, and the amortised read paid for the wait. `backlogged` is what tells the two cases apart without being
        told the arrival rate -- it reads the queue the caller already has, rather than a configured guess at how busy
        things are.
        """
        formed = Formed(jobs=[first])
        tokens = self._tokens(first)
        backlogged = self._worker.depth > 0
        # Bounded by what the engine has measured as well as by the setting. A linger longer than a pass's fixed cost
        # cannot pay, and before anything has been read there is no evidence for any wait at all -- the same reason
        # `backlogged` being false means there is no evidence either, this time about whether anyone else is coming.
        quiet = min(self.limits.linger_ms if backlogged else 0.0, self.limits.worth_waiting_ms(self.engine)) / 1e3
        ceiling = time.perf_counter() + (self.limits.worth_waiting_ms(self.engine) if backlogged else 0.0) / 1e3
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
                if not backlogged:
                    formed.reason = "nothing was already waiting, so no wait was worth betting on"
                else:
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

        # fp8spd3 (round 5): the common low-concurrency case -- a formed pass naming exactly one document, and
        # that document fresh -- can skip `Shelf.put_many` and the later `Shelf.ask` entirely: one layer-
        # interleaved pass over `interleave.read_and_branch_shelf` does the read and this one request's own
        # (single, by `Limits.questions`) branch group together. A pass naming more than one document, or a
        # document already resident, still goes through the two-step path below unchanged. See
        # `Prismyra._shelf_ask_interleaved` and RUN-fp8spd.md round 5 for the gate this went through.
        self._total_passes += 1
        if self.engine.interleaved_fork and len(fresh) == 1 and len(formed.jobs) == 1:
            self._fused_single_passes += 1
            job = formed.jobs[0]
            self._make_room([job], keep={job.payload.digest})
            result, handle, shelved = self.engine._shelf_ask_interleaved(shelf, job.payload.context, list(job.payload.questions))
            self._resident[job.payload.digest] = handle
            self._digest_of[handle] = job.payload.digest
            shelf.documents[handle] = shelved
            self._slot_bytes = max(self._slot_bytes, shelved.snapshot_bytes)
            self._used[handle] = self._clock
            self._clock += 1
            self.reads.append(1)
            return [result]

        # fp8spd6 (round 7): the generalised case -- every document this pass names is fresh, two or more of
        # them, none already resident -- fuses the same way, through `interleave.read_and_branch_shelf_many`
        # instead of one document at a time. `len(fresh) == len(formed.jobs)` is what the single-document
        # branch above's own `len(fresh) == 1 and len(formed.jobs) == 1` generalises to: `fresh` already
        # de-duplicates by digest, so this many fresh digests for this many jobs means no two jobs name the
        # same document and none is resident -- a pass naming a resident document, or two jobs for the same
        # still-fresh document, falls through to the two-step path below unchanged, same as before. See
        # `RUN-fp8spd.md` round 7, "本題", for why this -- not loosening the single-document condition's own
        # exact-match -- is what the round-6 10.4%-fusion-rate finding under congestion needed.
        #
        # `padded_total <= self.engine.group`, checked **before** taking this path: `form()`'s own admission
        # (`Formed.questions`) bounds the *raw* question count a pass carries to `self.limits.questions`, not
        # the *padded* one -- and `interleave.read_and_branch_shelf_many` rounds each document's own branch
        # rows independently (this function's own docstring says why: rounding the combined total once, the
        # way the non-interleaved `_branch_across` does, is the exact design RUN-inv.md round 3 traced a
        # cross-document residual to). Independent rounding means the *sum* of several documents' own rounded
        # counts is no longer bounded by `cap` the way one document's own `_round_rows(count, cap) <= cap`
        # always is -- found on real hardware (RUN-fp8spd.md round 7): three fresh documents in one pass with
        # a combined raw count of 32 rounded, independently, to a combined 44 and crashed the pool
        # ("44 rows asked for and this pool holds 32") rather than disagreeing on an answer. A pass whose
        # padded total does not fit falls through to the two-step path below, unchanged -- not a smaller
        # version of this one, the same safe path every pass took before this round.
        padded_total = sum(_round_rows(len(job.payload.questions), self.engine.group) for job in formed.jobs)
        if (
            self.engine.interleaved_fork
            and len(fresh) == len(formed.jobs)
            and len(fresh) >= 2
            and padded_total <= self.engine.group
        ):
            self._fused_many_passes += 1
            jobs = formed.jobs
            self._make_room(jobs, keep={job.payload.digest for job in jobs})
            triples = self.engine._shelf_ask_interleaved_many(
                shelf, [job.payload.context for job in jobs], [list(job.payload.questions) for job in jobs]
            )
            results = []
            for job, (result, handle, shelved) in zip(jobs, triples, strict=True):
                self._resident[job.payload.digest] = handle
                self._digest_of[handle] = job.payload.digest
                shelf.documents[handle] = shelved
                self._slot_bytes = max(self._slot_bytes, shelved.snapshot_bytes)
                self._used[handle] = self._clock
                self._clock += 1
                results.append(result)
            self.reads.append(len(jobs))
            return results

        if fresh:
            jobs = list(fresh.values())
            self._make_room(jobs, keep={job.payload.digest for job in formed.jobs})
            handles = shelf.put_many([job.payload.context for job in jobs])
            for job, handle in zip(jobs, handles, strict=True):
                self._resident[job.payload.digest] = handle
                self._digest_of[handle] = job.payload.digest
                # Measured, not assumed: the largest snapshot actually seen so far, which `_make_room` uses to
                # decide how much free memory the *next* admission needs before it starts.
                self._slot_bytes = max(self._slot_bytes, shelf.documents[handle].snapshot_bytes)
        asked: dict[int, list[Question]] = {}
        for job in formed.jobs:
            handle = self._resident[job.payload.digest]
            self._used[handle] = self._clock
            self._clock += 1
            # Two callers asking about the same document in one pass would collide here, and one of them would get the
            # other's questions. They are merged instead -- one document, both callers' questions, one set of rows.
            asked.setdefault(handle, []).extend(job.payload.questions)
        answers = shelf.ask(asked, lane=self._lane_id)
        self.reads.append(len(fresh))
        return [answers[self._resident[job.payload.digest]] for job in formed.jobs]

    def _on_shelf(self):
        """The shelf, opened on the first pass rather than at construction, because opening it allocates."""
        if self._shelf is None:
            self._shelf = self.engine.open_shelf(room=self._lane_room, lane=self._lane_id)
        return self._shelf

    def _make_room(self, fresh: list[Job], keep: set[str]) -> None:
        """Drop the least recently used documents until the new ones have somewhere to go.

        Least recently used, and never one in the pass being formed. The pool decides whether the **pages** fit, so
        this drops until the tokens fit the shelf's room rather than guessing at pages: the shelf refuses by name if
        the estimate is still wrong, and a refusal is recoverable where a wrong page is not.

        That bound alone is not enough, and the gap is not pages. Every resident document also keeps a clone of the
        recurrent state it ended on (`Shelved.snapshot`), outside the pool, and that clone costs roughly the same
        whatever the document's length -- a shelf holding many *short* documents can exhaust device memory on that
        count alone while the token budget above still has room to spare. Measured: 97 documents averaging 251
        tokens each (24,394 total, far under a 65,536-token budget) exhausted a 44 GiB card one snapshot at a time,
        because nothing was ever evicted on their account. `self._slot_bytes` is what one of them actually cost,
        and this is `_make_room` asking, before admitting more, whether the device has that much to spare -- the
        same question `engine.GRAPH_MEMORY_MARGIN` asks before a recording, for the same reason: the margin alone
        only answers "is there room for one more", and pairing it with a per-document figure is what makes it also
        answer "how many are there already", which an unbounded resident count needs answered.

        That is still a margin, asking "is there room for one more" every time admission runs -- which only fires
        when the device is already short, not when residents have simply piled up while there was still room to
        spare. `SHELF_MAX_RESIDENTS` is the count version of the same gap `MAX_KEPT_RECORDINGS` closed for
        recordings (its own docstring explains why a margin alone cannot bound a count): evicting toward it is
        folded into this same loop rather than a separate pass, so a resident kept past the cap because the pass
        being formed needed it (`keep`) is still protected the same way `fits_tokens`/`fits_memory` already protect
        it.
        """
        shelf = self._on_shelf()
        wanted = sum(self._tokens(job) for job in fresh)
        incoming_snapshot = len(fresh) * self._slot_bytes
        tried_reclaim = False
        while self._resident:
            held = sum(shelf.documents[handle].tokens for handle in shelf.documents)
            fits_tokens = held + wanted <= self.limits.tokens
            fits_count = len(self._resident) + len(fresh) <= SHELF_MAX_RESIDENTS
            fits_memory = True
            if self.engine.torch_device.type == "cuda":
                free, _ = torch.cuda.mem_get_info(self.engine.torch_device)
                fits_memory = free - incoming_snapshot > SHELF_MEMORY_MARGIN
                # 2026-10-05 (THROUGHPUT.md item 2, same measurement and reasoning as
                # `engine._run_recorded`'s GRAPH_MEMORY_MARGIN check): a `mem_get_info` free number below the
                # margin is often PyTorch's caching allocator holding cached-but-unallocated blocks, not memory
                # any resident document actually needs -- `empty_cache()` reclaimed 7.6 of a measured 7.9 GiB gap
                # in one call. Tried once per `_make_room` call (not once per loop iteration: a resident document
                # evicted a moment ago may have made room on its own, which this re-checks first) before this
                # loop evicts a document that fits-memory alone would not have required dropping.
                # 2026-10-05 (coordinator's item 2 follow-up): across *calls*, not just within one -- the reclaim
                # itself costs 2-4% of questions/second under load (THROUGHPUT.md, isolated from the dispatcher-
                # narrowing change by comparing both independently). `RECLAIM_COOLDOWN_S` skips the call while a
                # previous attempt from a recent pass is still within its cooldown; `fits_memory` stays `False` on
                # the stale number either way, so skipping the call never admits past a margin that is still tight.
                now = time.perf_counter()
                if not fits_memory and not tried_reclaim and now >= self._reclaim_cooldown_until:
                    tried_reclaim = True
                    # fp8spd4 (correcting fp8spd3's S5 / SYNTHESIS.md): the previous version of this check
                    # *predicted* that `empty_cache()` would raise `mem_get_info`'s free number by
                    # `reserved - allocated` and set `fits_memory = True` on that prediction alone, without
                    # calling `empty_cache()` or re-reading `mem_get_info` to confirm it. integ's bisection
                    # (RUN-integ.md 2.5) found this broke `test_a_shelf_evicts_on_memory_pressure_even_with_
                    # tokens_to_spare`: that test monkeypatches `mem_get_info` to report a free number pinned
                    # below the margin regardless of what this process's allocator does, which is exactly the
                    # case the module docstring's own history (`SHELF_MAX_RESIDENTS`'s "97 documents... exhausted
                    # a 44 GiB card") this check exists to catch -- an externally (OS/driver) reported shortage
                    # that this process's own `reserved - allocated` slack does not explain and could not fix.
                    # The fix keeps the one case the prediction is *never* wrong about -- `cached_slack == 0`,
                    # nothing cached-but-unused to give back, so the call could not possibly help and skipping
                    # it is a true no-op -- and calls `empty_cache()` and re-reads the real `mem_get_info` for
                    # every other case, the same as before S5 existed. DOSSIER.md's own recorded decision
                    # ("`empty_cache()`の呼び出しはmargin checkの正しさに必要で、そのコストは意図して受け入れた")
                    # is what this restores; S5's round1/round5 own measurements never established the skip's
                    # real-world savings against a baseline (RUN-fp8spd.md), so there is nothing demonstrated to
                    # trade the correctness back for.
                    allocated = torch.cuda.memory_allocated(self.engine.torch_device)
                    reserved = torch.cuda.memory_reserved(self.engine.torch_device)
                    cached_slack = max(0, reserved - allocated)
                    if cached_slack == 0:
                        pass
                    else:
                        started = time.perf_counter()
                        torch.cuda.empty_cache()
                        self._empty_cache_calls += 1
                        self._empty_cache_ms += (time.perf_counter() - started) * 1e3
                        free, _ = torch.cuda.mem_get_info(self.engine.torch_device)
                        fits_memory = free - incoming_snapshot > SHELF_MEMORY_MARGIN
                        if not fits_memory:
                            self._reclaim_cooldown_until = now + RECLAIM_COOLDOWN_S
            if fits_tokens and fits_memory and fits_count:
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
        if self._children is not None:
            return {"lanes": [child.stats() for child in self._children]}
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
            "empty_cache_calls": self._empty_cache_calls,
            "empty_cache_ms": round(self._empty_cache_ms, 1),
            "fused_single_passes": self._fused_single_passes,
            "fused_many_passes": self._fused_many_passes,
            "total_passes": self._total_passes,
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
