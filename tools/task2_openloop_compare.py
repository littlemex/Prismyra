#!/usr/bin/env python3
"""fp8spd6 (round7, task2): RACE 368-document open-loop comparison. Measures questions/second and the
fraction of answered requests that beat a deadline, for the generalised multi-document fusion
(`Batcher._answer`'s new `len(fresh) == len(formed.jobs) >= 2` branch) against a baseline run with
`interleaved_fork` off (same branch) or against literal `origin/main` (set PRISMYRA_SRC accordingly).

Usage: PRISMYRA_SRC=... python3 task2_openloop_compare.py <rate> <mode> <deadline_ms>
  mode: "mainline" (plain Prismyra(paged=True), no interleaved_fork kwarg at all -- for a literal
        origin/main source tree that does not accept it) or "interleaved" (this branch,
        interleaved_fork=True).
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("PRISMYRA_SRC", "."))
sys.path.insert(0, os.path.join(os.environ.get("PRISMYRA_SRC", "."), "evals"))

import torch  # noqa: E402
import tasks  # noqa: E402
from prismyra import Prismyra  # noqa: E402
from prismyra.schedule import Batcher  # noqa: E402

MODEL = "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l"
RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
MODE = sys.argv[2] if len(sys.argv) > 2 else "interleaved"
DEADLINE_MS = float(sys.argv[3]) if len(sys.argv) > 3 else 500.0
PHASE_SECONDS = 8.0
DRAIN_TIMEOUT = 300.0
POOL_DOCUMENTS = 368


def main() -> None:
    items = tasks.load("race", POOL_DOCUMENTS, split="validation", seed=0)
    print(f"{len(items)} documents available, mode={MODE} rate={RATE} deadline={DEADLINE_MS}ms", flush=True)

    kwargs = dict(paged=True, graphs=True, require_kernels=True)
    if MODE == "interleaved":
        kwargs["interleaved_fork"] = True
    engine = Prismyra(MODEL, **kwargs)
    batcher = Batcher(engine).start()

    for item in items[:5]:
        batcher.ask(item.context, item.questions, timeout=30.0)
    print("warm-up done", flush=True)

    random.seed(0)
    n = max(1, round(RATE * PHASE_SECONDS))
    jobs = []
    dropped = 0
    t0 = time.perf_counter()
    next_at = t0
    for i in range(n):
        item = items[i % len(items)]
        now = time.perf_counter()
        if now < next_at:
            time.sleep(next_at - now)
        next_at += random.expovariate(RATE)
        try:
            job = batcher.submit(item.context, item.questions)
            jobs.append((job, item))
        except Exception:
            dropped += 1
            jobs.append((None, item))

    inject_done_at = time.perf_counter()
    deadline_wall = inject_done_at + DRAIN_TIMEOUT
    not_done = 0
    errored = 0
    for job, _ in jobs:
        if job is None:
            continue
        remaining = max(0.0, deadline_wall - time.perf_counter())
        job.done.wait(timeout=remaining)
        if not job.done.is_set():
            not_done += 1
        elif job.error is not None:
            errored += 1
    drain_elapsed = time.perf_counter() - inject_done_at
    print(f"drain_elapsed={drain_elapsed:.1f}s not_done={not_done} errored={errored}", flush=True)
    if errored:
        for job, _ in jobs:
            if job is not None and job.done.is_set() and job.error is not None:
                print("FIRST ERROR:", repr(job.error))
                break

    latencies_ms, answered_questions, hit = [], 0, 0
    first_queued = last_finished = None
    for job, item in jobs:
        if job is None or not job.done.is_set() or job.error is not None:
            continue
        latency = (job.finished_at - job.queued_at) * 1e3
        latencies_ms.append(latency)
        if latency <= DEADLINE_MS:
            hit += 1
        first_queued = job.queued_at if first_queued is None else min(first_queued, job.queued_at)
        last_finished = job.finished_at if last_finished is None else max(last_finished, job.finished_at)
        answered_questions += len(item.questions)

    wall = (last_finished - first_queued) if (first_queued and last_finished) else PHASE_SECONDS
    bstats = batcher.stats()

    report = {
        "mode": MODE,
        "rate_docs_per_s": RATE,
        "deadline_ms": DEADLINE_MS,
        "injected": n,
        "dropped": dropped,
        "answered_documents": len(latencies_ms),
        "answered_questions": answered_questions,
        "questions_per_second": round(answered_questions / wall, 2) if wall else 0.0,
        "deadline_hit_rate": round(hit / len(latencies_ms), 4) if latencies_ms else None,
        "p50_ms": round(statistics.median(latencies_ms), 1) if latencies_ms else None,
        "fused_single_passes": bstats.get("fused_single_passes"),
        "fused_many_passes": bstats.get("fused_many_passes"),
        "total_passes": bstats.get("total_passes"),
    }
    print(json.dumps(report, indent=2))
    Path(f"/work/fp8spd6_openloop_{MODE}_rate{RATE}.json").write_text(json.dumps(report, indent=2))
    batcher.stop()


if __name__ == "__main__":
    main()
