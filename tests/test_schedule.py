"""The rule that decides which waiting requests share a pass.

Testable without a device, which is the point of keeping the rule in its own object: forming is where a scheduler is
right or wrong, and it is arithmetic over what has arrived. The engine here is a stub that records the batches it was
asked for, so what is asserted is the grouping rather than the answers.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pytest

from prismyra import Boolean
from prismyra.schedule import Batcher
from prismyra.schema import PrismyraError


class FakeResult(dict):
    """Enough of a `Result` for the scheduler: the jobs only carry it back to their callers."""


@dataclass
class FakeEncoded:
    """What `encode_context` returns: the tokens counted once, for the forming rule and the read to share."""

    text: str
    tokens: int


class FakeEngine:
    """A stand-in that records the batches it was handed, and answers instantly.

    `group` and `longest_context` are what the limits are read from. Nothing here runs a model, so the batches recorded
    are exactly the scheduler's decisions with nothing else mixed in.
    """

    paged = True

    def __init__(
        self,
        group: int = 8,
        longest_context: int = 64,
        per_call: float = 0.0,
        fastest_read_ms: float | None = None,
    ):
        self.group = group
        self.longest_context = longest_context
        # None means "nothing has been read yet", and the scheduler must not make anybody wait on no evidence. A figure
        # here is what a warmed engine looks like, and is what exercises the linger.
        self.fastest_read_ms = fastest_read_ms
        self.encoded = 0
        self.shelves = 0
        #: Which documents each read pass carried, and which handles each answering pass carried.
        self.reads: list[list[str]] = []
        self.passes: list[list[int]] = []
        self.dropped: list[int] = []
        self.batches: list[list[str]] = []
        self.per_call = per_call
        self.opened = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def validate(self, questions) -> None:
        for q in questions:
            if not q.prompt:
                raise PrismyraError("empty prompt")

    def tokenizer(self, text: str):
        # One token per word, which makes the token limit easy to write tests against.
        return {"input_ids": text.split()}

    def encode_context(self, text: str) -> FakeEncoded:
        self.encoded += 1
        return FakeEncoded(text=text, tokens=len(text.split()))

    def open_batch(self, contexts: list):
        self.batches.append([one.text if isinstance(one, FakeEncoded) else one for one in contexts])
        self.opened.set()
        self.release.wait(5)
        if self.per_call:
            time.sleep(self.per_call)
        return FakeBatch(list(contexts))

    def open_shelf(self, room: int | None = None) -> FakeShelf:
        self.shelves += 1
        return FakeShelf(self)


@dataclass
class FakeShelved:
    tokens: int


class FakeShelf:
    """Enough of a shelf to see the scheduler's decisions: which documents were read, and which were answered."""

    def __init__(self, engine: FakeEngine):
        self.engine = engine
        self.documents: dict[int, FakeShelved] = {}
        self._next = 0

    def put_many(self, contexts: list[str]) -> list[int]:
        self.engine.reads.append(list(contexts))
        handles = []
        for one in contexts:
            self.documents[self._next] = FakeShelved(tokens=len(one.split()))
            handles.append(self._next)
            self._next += 1
        self.engine.opened.set()
        self.engine.release.wait(5)
        if self.engine.per_call:
            time.sleep(self.engine.per_call)
        return handles

    def ask(self, asked: dict) -> dict:
        self.engine.passes.append(sorted(asked))
        return {handle: FakeResult({q.id: handle for q in questions}) for handle, questions in asked.items()}

    def drop(self, handle: int) -> None:
        self.engine.dropped.append(handle)
        del self.documents[handle]

    def close(self) -> None:
        self.documents.clear()


class FakeBatch:
    def __init__(self, contexts: list):
        self.contexts = [one.text if isinstance(one, FakeEncoded) else one for one in contexts]

    def ask(self, questions: list[list]) -> list[FakeResult]:
        return [
            FakeResult({q.id: context for q in asked}) for context, asked in zip(self.contexts, questions, strict=True)
        ]

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def asking(n: int) -> list[Boolean]:
    return [Boolean(id=f"q{i}", prompt=f"Is clause {i} about shipping?") for i in range(n)]


def words(n: int, of: str = "word") -> str:
    return " ".join([of] * n)


#: Distinct documents of the same length. The scheduler recognises the same document by a digest of its text, so a test
#: that means "four callers, four documents" has to say four different things, or it is testing the merge instead.
def distinct(count: int, length: int = 3) -> list[str]:
    return [words(length, of=f"doc{i}") for i in range(count)]


def answered(batcher: Batcher, jobs, timeout: float = 5.0) -> None:
    for job in jobs:
        assert job.done.wait(timeout), "a job never completed"
        assert job.error is None, job.error


def test_everything_waiting_that_fits_travels_together():
    """The whole point. Requests that arrived while the device was busy share the next pass."""
    engine = FakeEngine(group=8)
    batcher = Batcher(engine)
    # Queued before the loop starts, so they are all waiting when the first pass forms.
    contexts = distinct(4)
    jobs = [batcher.submit(one, asking(2)) for one in contexts]
    batcher.start()
    try:
        answered(batcher, jobs)
    finally:
        batcher.stop()
    assert engine.reads == [contexts], engine.reads
    assert engine.passes == [[0, 1, 2, 3]], engine.passes
    assert batcher.stats()["mean_documents_per_pass"] == 4.0


def test_a_pass_stops_at_the_group_and_says_so():
    """Four requests of three questions each against a group of eight: three fit, the fourth does not."""
    engine = FakeEngine(group=8)
    batcher = Batcher(engine)
    jobs = [batcher.submit(one, asking(3)) for one in distinct(4, length=2)]
    batcher.start()
    try:
        answered(batcher, jobs)
    finally:
        batcher.stop()
    assert [len(p) for p in engine.passes] == [2, 2], engine.passes
    why = batcher.stats()["why_passes_stopped"]
    assert any("would pass 8 rows" in reason for reason in why), why


def test_a_pass_stops_when_the_pool_would_overflow():
    """The other limit. Documents of forty words against a pool of a hundred: two fit and the third does not."""
    engine = FakeEngine(group=32, longest_context=100)
    batcher = Batcher(engine)
    jobs = [batcher.submit(one, asking(1)) for one in distinct(3, length=40)]
    batcher.start()
    try:
        answered(batcher, jobs)
    finally:
        batcher.stop()
    assert [len(p) for p in engine.passes] == [2, 1], engine.passes
    why = batcher.stats()["why_passes_stopped"]
    assert any("in the pool" in reason for reason in why), why


def test_a_pass_stops_at_one_document_per_row():
    """A batch cannot hold more documents than a pass has rows, because every document needs at least one."""
    engine = FakeEngine(group=2, longest_context=1000)
    batcher = Batcher(engine)
    jobs = [batcher.submit(one, asking(1)) for one in distinct(5, length=1)]
    batcher.start()
    try:
        answered(batcher, jobs)
    finally:
        batcher.stop()
    assert max(len(p) for p in engine.passes) == 2, engine.passes
    why = batcher.stats()["why_passes_stopped"]
    assert any("holds 2 documents" in reason for reason in why), why


def test_a_lone_request_does_not_wait_for_company():
    """Work-conserving. The scheduler that helps under load must not add latency when there is none."""
    engine = FakeEngine(group=8)
    with Batcher(engine) as batcher:
        started = time.perf_counter()
        result = batcher.ask(words(3), asking(2), timeout=5)
        took = (time.perf_counter() - started) * 1e3
    assert result
    assert engine.reads == [[words(3)]]
    # Nothing about this figure is a performance claim; it is the absence of a linger, which would be tens of
    # milliseconds even on a stub.
    assert took < 200, took


def test_a_request_too_wide_for_any_pass_is_refused_before_it_waits():
    engine = FakeEngine(group=4)
    batcher = Batcher(engine)
    with pytest.raises(PrismyraError, match="rows"):
        batcher.submit(words(3), asking(5))


def test_a_request_with_no_questions_is_refused():
    batcher = Batcher(FakeEngine())
    with pytest.raises(PrismyraError, match="at least one question"):
        batcher.submit(words(3), [])


def test_the_joined_storage_is_refused_at_construction():
    """Refused when the scheduler is built rather than when a batch forms: a caller cannot fix it later."""
    engine = FakeEngine()
    engine.paged = False
    with pytest.raises(PrismyraError, match="paged"):
        Batcher(engine)


def test_a_failing_pass_fails_every_request_in_it():
    """One document's failure must not leave the others waiting for ever, which is what would happen if only the job the
    worker handed over were completed."""

    class Broken(FakeEngine):
        def open_shelf(self, room: int | None = None):
            raise RuntimeError("the pass failed")

    engine = Broken(group=8)
    batcher = Batcher(engine)
    jobs = [batcher.submit(one, asking(2)) for one in distinct(3, length=2)]
    batcher.start()
    try:
        for job in jobs:
            assert job.done.wait(5), "a job never completed"
            assert isinstance(job.error, RuntimeError)
    finally:
        batcher.stop()


def test_every_request_in_a_pass_is_counted_with_its_own_wait():
    """A request that waited for a pass to start waited in the queue, and the figure has to say so -- otherwise batching
    looks free because only the first request's wait is ever recorded."""
    engine = FakeEngine(group=8)
    batcher = Batcher(engine)
    jobs = [batcher.submit(one, asking(2)) for one in distinct(4, length=2)]
    batcher.start()
    try:
        answered(batcher, jobs)
    finally:
        batcher.stop()
    queue_stats = batcher.stats()["queue"]
    assert queue_stats["completed"] == 4, queue_stats
    assert "queue_ms" in queue_stats


def test_a_document_is_tokenised_once_per_request():
    """Twice was the measured cost on the thread that owns the device: once to count it while forming a pass and once
    inside the read. Counted here rather than timed, because the count is the claim."""
    engine = FakeEngine(group=8)
    batcher = Batcher(engine)
    jobs = [batcher.submit(one, asking(2)) for one in distinct(4)]
    batcher.start()
    try:
        answered(batcher, jobs)
    finally:
        batcher.stop()
    assert engine.encoded == 4, engine.encoded


def test_nothing_waits_before_the_engine_has_read_anything():
    """The linger is bounded by what the engine has measured, and an engine that has read nothing has measured nothing.
    A scheduler with no evidence must not make anybody wait."""
    engine = FakeEngine(group=8, fastest_read_ms=None)
    batcher = Batcher(engine, linger_ms=50.0)
    assert batcher.limits.worth_waiting_ms(engine) == 0.0
    with batcher:
        started = time.perf_counter()
        batcher.ask(words(3), asking(2), timeout=5)
        took = (time.perf_counter() - started) * 1e3
    assert took < 40, took


def test_the_linger_is_capped_by_what_a_pass_costs():
    """Waiting longer than a pass's fixed cost cannot pay: by then the pass could have run. So the setting is a request
    and the engine's own figure is the ceiling."""
    engine = FakeEngine(group=8, fastest_read_ms=3.0)
    batcher = Batcher(engine, linger_ms=1000.0)
    assert batcher.limits.worth_waiting_ms(engine) == 3.0
    with batcher:
        started = time.perf_counter()
        batcher.ask(words(3), asking(2), timeout=5)
        took = (time.perf_counter() - started) * 1e3
    # Waited the ceiling rather than the setting. A second would have been the setting.
    assert took < 200, took


def test_a_document_already_on_the_shelf_is_not_read_again():
    """What the shelf is for. The second request about a document costs a pass and no read."""
    engine = FakeEngine(group=8)
    batcher = Batcher(engine)
    with batcher:
        batcher.ask(words(3), asking(2), timeout=5)
        batcher.ask(words(3), asking(2), timeout=5)
        batcher.ask(words(3), asking(2), timeout=5)
        stats = batcher.stats()
    assert engine.reads == [[words(3)]], engine.reads
    assert len(engine.passes) == 3
    assert stats["documents_read"] == 1
    assert stats["documents_answered_without_reading"] == 2


def test_two_callers_asking_about_one_document_in_a_pass_are_merged():
    """They cannot be two rows of two documents, because there is one document. Their questions share its rows, and the
    failure this prevents is one caller receiving the other's answers."""
    engine = FakeEngine(group=8)
    batcher = Batcher(engine)
    jobs = [batcher.submit(words(3), asking(2)) for _ in range(3)]
    batcher.start()
    try:
        answered(batcher, jobs)
    finally:
        batcher.stop()
    assert engine.reads == [[words(3)]], engine.reads
    assert engine.passes == [[0]], engine.passes
    # Every caller got an answer to each of its own questions.
    for job in jobs:
        assert set(job.result) == {"q0", "q1"}


def test_the_least_recently_used_document_is_dropped_to_make_room():
    """A shelf holds what fits. What goes is the document nobody has asked about for longest, and never one in the
    pass being formed."""
    engine = FakeEngine(group=8, longest_context=6)
    batcher = Batcher(engine)
    with batcher:
        batcher.ask(words(3, of="first"), asking(1), timeout=5)
        batcher.ask(words(3, of="second"), asking(1), timeout=5)
        # The shelf holds six tokens and each document is three, so this one has to displace something.
        batcher.ask(words(3, of="third"), asking(1), timeout=5)
    assert engine.dropped == [0], engine.dropped
