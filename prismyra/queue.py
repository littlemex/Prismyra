"""One worker owns the device; callers queue.

This is in the core rather than in the server because it is a correctness and latency property of using one device from
several threads, not an HTTP concern. Requests entering the model together are correct but slow in a specific way: they
share one stream, so they all crawl and all finish late, and throughput drops. A queue lets the first one leave first.
Measured figures are in docs/PERFORMANCE.md.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class WorkerStopped(RuntimeError):
    """The worker shut down before this job ran. Raised in the caller's thread, because the alternative is a caller
    waiting on an event that nobody will ever set -- `submit` with no timeout would never return."""


class QueueFull(RuntimeError):
    """Admission was refused. Its own type because the handler can raise `RuntimeError` too, and a caller that cannot
    tell the two apart reports a failed request as an overloaded server."""


@dataclass
class Job:
    """One unit of work and where its answer goes."""

    payload: Any
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None
    queued_at: float = field(default_factory=time.perf_counter)
    started_at: float | None = None
    finished_at: float | None = None
    #: Set when the caller stopped waiting. A job still in the queue is then skipped rather than run: the answer has
    #: nowhere to go, and running it would put a request nobody is waiting for ahead of one somebody is.
    cancelled: bool = False

    @property
    def queue_ms(self) -> float:
        return ((self.started_at or time.perf_counter()) - self.queued_at) * 1e3

    @property
    def service_ms(self) -> float:
        if self.started_at is None:
            return 0.0
        return ((self.finished_at or time.perf_counter()) - self.started_at) * 1e3


class Worker:
    """A single thread that runs `handler` on one job at a time.

    `handler` is called only from the worker thread, so it may hold and mutate whatever state it likes without locking,
    which is the point, since the model's caches are exactly such state.
    """

    def __init__(self, handler: Callable[[Any], Any], max_queue: int = 512):
        self._handler = handler
        self._q: queue.Queue[Job] = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, name="prismyra", daemon=True)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._history: list[dict] = []
        self.completed = 0
        self.rejected = 0
        self.abandoned = 0

    def start(self) -> Worker:
        self._thread.start()
        return self

    def stop(self, timeout: float = 30.0) -> None:
        """Ask the worker to finish, wait up to `timeout` for it, then fail whatever is still queued.

        The wait is bounded, so this returns whether or not the worker actually stopped: a handler in the middle of a
        forward pass is not interrupted. `alive` says which happened.

        No sentinel job. Enqueuing one deadlocks when the queue is full: the stop flag is set first, so the worker
        leaves its loop without draining and the sentinel never fits. The worker polls instead, which costs a wake-up
        per tick and cannot hang the caller.
        """
        self._stop.set()
        self._thread.join(timeout=timeout)
        self._fail_the_rest()

    def _fail_the_rest(self) -> None:
        """Wake everyone still queued. A stopped worker will not drain, so without this a `submit` without a timeout
        blocks for the life of the process."""
        while True:
            try:
                job = self._q.get(block=False)
            except queue.Empty:
                return
            job.error = WorkerStopped("the worker stopped before this job ran")
            job.done.set()

    @property
    def depth(self) -> int:
        return self._q.qsize()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def submit(self, payload: Any, timeout: float | None = None) -> Job:
        """Queue a job and wait. Raises whatever the handler raised, in the caller's thread.

        A full queue is refused rather than made to wait: a caller that has already given up is not helped by being
        admitted, and its request would occupy the device for an answer that goes nowhere.

        A stopped worker is refused too. Queueing a job when nothing will run it is how a caller with no timeout waits
        for the life of the process; the flag is checked again after queueing, because `stop` can arrive in between.
        """
        if self._stop.is_set():
            raise WorkerStopped("the worker has stopped and is not accepting jobs")
        job = Job(payload=payload)
        try:
            self._q.put(job, block=False)
        except queue.Full:
            with self._lock:
                self.rejected += 1
            raise QueueFull(f"the queue is full at {self._q.maxsize} waiting jobs") from None
        if self._stop.is_set() and not job.done.is_set():
            # Stopped between the check and the put. Nobody will drain this, so fail it here rather than wait on it.
            job.error = WorkerStopped("the worker stopped while this job was being queued")
            job.done.set()
        if not job.done.wait(timeout=timeout):
            # Marked before raising. If it has not started, the worker drops it; if it has, it runs to the end, because
            # stopping a forward pass part-way would leave the caches in a state the next request would read.
            job.cancelled = True
            started = job.started_at is not None
            raise TimeoutError(
                f"waited {timeout}s with {self.depth} jobs still queued; this one "
                + ("had started and will finish" if started else "was dropped before it started")
            )
        if job.error is not None:
            raise job.error
        return job

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            if job.cancelled:
                # Abandoned while queued. Skipping it is the whole reason the flag exists: a worker that honours
                # abandoned work amplifies an overload instead of shedding it.
                with self._lock:
                    self.abandoned += 1
                job.done.set()
                continue
            job.started_at = time.perf_counter()
            depth = self._q.qsize()
            try:
                job.result = self._handler(job.payload)
            except BaseException as e:  # noqa: BLE001 - handed to the caller's thread rather than lost here
                job.error = e
            job.finished_at = time.perf_counter()
            with self._lock:
                self.completed += 1
                self._history.append({"queue_ms": job.queue_ms, "service_ms": job.service_ms, "depth": depth})
                if len(self._history) > 2_000:
                    del self._history[:1_000]
            job.done.set()

    def stats(self) -> dict:
        """Waiting and working reported separately, because their fixes are different.

        Time in the queue is answered by batching or another device; time in service is answered by kernels. One latency
        figure hides which one is binding.
        """
        with self._lock:
            history = list(self._history)
            completed, rejected, abandoned = self.completed, self.rejected, self.abandoned
        out: dict = {
            "completed": completed,
            "rejected": rejected,
            "abandoned": abandoned,
            "depth": self.depth,
        }
        if not history:
            return out
        for name in ("queue_ms", "service_ms"):
            values = sorted(h[name] for h in history)
            n = len(values)
            out[name] = {"median": values[n // 2], "p95": values[min(n - 1, int(n * 0.95))], "max": values[-1]}
        out["max_depth_seen"] = max(h["depth"] for h in history)
        return out
