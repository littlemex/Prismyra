"""The queue: one worker at a time, errors reaching the caller, waiting measured apart from working."""

from __future__ import annotations

import contextlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from prismyra.queue import QueueFull, Worker, WorkerStopped


def test_only_one_job_runs_at_a_time():
    """The property the whole module exists for: the handler may hold unlocked state."""
    inside = 0
    seen_together = 0
    lock = threading.Lock()

    def handler(payload):
        nonlocal inside, seen_together
        with lock:
            inside += 1
            if inside > 1:
                seen_together += 1
        time.sleep(0.01)
        with lock:
            inside -= 1
        return payload * 2

    worker = Worker(handler).start()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = [job.result for job in pool.map(worker.submit, range(16))]
    finally:
        worker.stop()
    assert results == [n * 2 for n in range(16)]
    assert seen_together == 0


def test_the_handlers_error_reaches_the_caller():
    def handler(_):
        raise ValueError("deliberate")

    worker = Worker(handler).start()
    try:
        with pytest.raises(ValueError, match="deliberate"):
            worker.submit(1)
    finally:
        worker.stop()


def test_stopping_with_a_full_queue_does_not_hang():
    """Regression: the first version enqueued a sentinel to stop, which cannot fit when the queue is full."""
    release = threading.Event()
    worker = Worker(lambda _: release.wait(timeout=5), max_queue=1).start()
    threading.Thread(target=lambda: _swallow(worker, 0), daemon=True).start()
    time.sleep(0.05)
    _swallow(worker, 1)  # fills the single slot and is abandoned
    release.set()
    worker.stop(timeout=5)
    assert not worker.alive


def _swallow(worker: Worker, payload: int) -> None:
    with contextlib.suppress(QueueFull, TimeoutError):
        worker.submit(payload, timeout=0.2)


def test_a_full_queue_is_refused_rather_than_made_to_wait():
    release = threading.Event()

    def handler(_):
        release.wait(timeout=5)

    worker = Worker(handler, max_queue=1).start()
    try:
        blocker = threading.Thread(target=lambda: worker.submit(0, timeout=5), daemon=True)
        blocker.start()
        time.sleep(0.05)
        errors = 0
        for _ in range(4):
            try:
                worker.submit(1, timeout=0.2)
            except QueueFull:
                errors += 1
            except TimeoutError:
                pass
        assert errors >= 1
        assert worker.rejected >= 1
    finally:
        release.set()
        worker.stop()


def test_stats_separate_waiting_from_working():
    worker = Worker(lambda n: time.sleep(0.005)).start()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(worker.submit, range(8)))
        stats = worker.stats()
    finally:
        worker.stop()
    assert stats["completed"] == 8
    assert "queue_ms" in stats and "service_ms" in stats
    # Under load the later jobs wait, so the queue's maximum must exceed one job's service time.
    assert stats["queue_ms"]["max"] > 0.0


def test_a_job_abandoned_while_queued_never_runs():
    """The point of the flag. A worker that honours abandoned work amplifies an overload instead of shedding it: the
    caller already has its 504, and running the job puts a request nobody waits for ahead of one somebody does."""
    ran: list[int] = []
    release = threading.Event()

    def handler(payload):
        if payload == 0:
            release.wait(timeout=5)
        ran.append(payload)

    worker = Worker(handler, max_queue=4).start()
    try:
        threading.Thread(target=lambda: _swallow(worker, 0), daemon=True).start()
        time.sleep(0.05)
        _swallow(worker, 1)  # queued behind the blocked job, then abandoned on timeout
        release.set()
        time.sleep(0.3)
        assert 1 not in ran
        assert worker.stats()["abandoned"] >= 1
    finally:
        worker.stop()


def test_stopping_wakes_a_caller_that_was_waiting_with_no_timeout():
    """Otherwise it waits for the life of the process: a stopped worker does not drain, so nothing sets the event."""
    release = threading.Event()
    outcome: list[BaseException | None] = []

    def handler(payload):
        if payload == 0:
            release.wait(timeout=5)

    def wait_forever():
        try:
            worker.submit(1)
            outcome.append(None)
        except BaseException as e:  # noqa: BLE001 - the test is about which exception arrives
            outcome.append(e)

    worker = Worker(handler, max_queue=4).start()
    threading.Thread(target=lambda: _swallow(worker, 0), daemon=True).start()
    time.sleep(0.05)
    waiter = threading.Thread(target=wait_forever, daemon=True)
    waiter.start()
    time.sleep(0.05)

    worker.stop(timeout=1)
    release.set()
    waiter.join(timeout=3)
    assert not waiter.is_alive()
    assert isinstance(outcome[0], WorkerStopped)


def test_submitting_to_a_stopped_worker_is_refused_rather_than_queued():
    """A job queued where nothing will run it is how a caller with no timeout waits for the life of the process."""
    worker = Worker(lambda n: n).start()
    worker.stop()
    with pytest.raises(WorkerStopped):
        worker.submit(1)
