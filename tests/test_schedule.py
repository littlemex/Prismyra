"""The rule that decides which waiting requests share a pass.

Testable without a device, which is the point of keeping the rule in its own object: forming is where a scheduler is
right or wrong, and it is arithmetic over what has arrived. The engine here is a stub that records the batches it was
asked for, so what is asserted is the grouping rather than the answers.
"""

from __future__ import annotations

import threading
import time

import pytest

from prismyra import Boolean
from prismyra.schedule import Batcher
from prismyra.schema import PrismyraError


class FakeResult(dict):
    """Enough of a `Result` for the scheduler: the jobs only carry it back to their callers."""


class FakeEngine:
    """A stand-in that records the batches it was handed, and answers instantly.

    `group` and `longest_context` are what the limits are read from. Nothing here runs a model, so the batches recorded
    are exactly the scheduler's decisions with nothing else mixed in.
    """

    paged = True

    def __init__(self, group: int = 8, longest_context: int = 64, per_call: float = 0.0):
        self.group = group
        self.longest_context = longest_context
        self.fastest_read_ms: float | None = None
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

    def open_batch(self, contexts: list[str]):
        self.batches.append(list(contexts))
        self.opened.set()
        self.release.wait(5)
        if self.per_call:
            time.sleep(self.per_call)
        return FakeBatch(contexts)


class FakeBatch:
    def __init__(self, contexts: list[str]):
        self.contexts = contexts

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


def words(n: int) -> str:
    return " ".join(["word"] * n)


def answered(batcher: Batcher, jobs, timeout: float = 5.0) -> None:
    for job in jobs:
        assert job.done.wait(timeout), "a job never completed"
        assert job.error is None, job.error


def test_everything_waiting_that_fits_travels_together():
    """The whole point. Requests that arrived while the device was busy share the next pass."""
    engine = FakeEngine(group=8)
    batcher = Batcher(engine)
    # Queued before the loop starts, so they are all waiting when the first pass forms.
    jobs = [batcher.submit(words(3), asking(2)) for _ in range(4)]
    batcher.start()
    try:
        answered(batcher, jobs)
    finally:
        batcher.stop()
    assert engine.batches == [[words(3)] * 4], engine.batches
    assert batcher.stats()["mean_documents_per_pass"] == 4.0


def test_a_pass_stops_at_the_group_and_says_so():
    """Four requests of three questions each against a group of eight: three fit, the fourth does not."""
    engine = FakeEngine(group=8)
    batcher = Batcher(engine)
    jobs = [batcher.submit(words(2), asking(3)) for _ in range(4)]
    batcher.start()
    try:
        answered(batcher, jobs)
    finally:
        batcher.stop()
    assert [len(b) for b in engine.batches] == [2, 2], engine.batches
    why = batcher.stats()["why_passes_stopped"]
    assert any("would pass 8 rows" in reason for reason in why), why


def test_a_pass_stops_when_the_pool_would_overflow():
    """The other limit. Documents of forty words against a pool of a hundred: two fit and the third does not."""
    engine = FakeEngine(group=32, longest_context=100)
    batcher = Batcher(engine)
    jobs = [batcher.submit(words(40), asking(1)) for _ in range(3)]
    batcher.start()
    try:
        answered(batcher, jobs)
    finally:
        batcher.stop()
    assert [len(b) for b in engine.batches] == [2, 1], engine.batches
    why = batcher.stats()["why_passes_stopped"]
    assert any("in the pool" in reason for reason in why), why


def test_a_pass_stops_at_one_document_per_row():
    """A batch cannot hold more documents than a pass has rows, because every document needs at least one."""
    engine = FakeEngine(group=2, longest_context=1000)
    batcher = Batcher(engine)
    jobs = [batcher.submit(words(1), asking(1)) for _ in range(5)]
    batcher.start()
    try:
        answered(batcher, jobs)
    finally:
        batcher.stop()
    assert max(len(b) for b in engine.batches) == 2, engine.batches
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
    assert engine.batches == [[words(3)]]
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
        def open_batch(self, contexts):
            raise RuntimeError("the pass failed")

    engine = Broken(group=8)
    batcher = Batcher(engine)
    jobs = [batcher.submit(words(2), asking(2)) for _ in range(3)]
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
    jobs = [batcher.submit(words(2), asking(2)) for _ in range(4)]
    batcher.start()
    try:
        answered(batcher, jobs)
    finally:
        batcher.stop()
    queue_stats = batcher.stats()["queue"]
    assert queue_stats["completed"] == 4, queue_stats
    assert "queue_ms" in queue_stats
