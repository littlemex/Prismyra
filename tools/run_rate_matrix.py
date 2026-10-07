#!/usr/bin/env python3
"""fp8spd6 (round8, task1): RACE 368-document open-loop matrix across arrival rates {10,20,40,60,80},
one process per mode (mainline / interleaved) so the model loads once instead of once per rate. Writes one
JSON report per rate to /work/matrix_<mode>_rate<rate>.json, same fields as task2_openloop_compare.py.
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
MODE = sys.argv[1] if len(sys.argv) > 1 else "interleaved"
RATES = [10.0, 20.0, 40.0, 60.0, 80.0]
DEADLINE_MS = 500.0
PHASE_SECONDS = 8.0
DRAIN_TIMEOUT = 300.0
POOL_DOCUMENTS = 368


def run_one(engine, batcher, items, rate: float) -> dict:
    random.seed(0)
    n = max(1, round(rate * PHASE_SECONDS))
    jobs = []
    dropped = 0
    t0 = time.perf_counter()
    next_at = t0
    for i in range(n):
        item = items[i % len(items)]
        now = time.perf_counter()
        if now < next_at:
            time.sleep(next_at - now)
        next_at += random.expovariate(rate)
        try:
            job = batcher.submit(item.context, item.questions)
            jobs.append((job, item))
        except Exception:
            dropped += 1
            jobs.append((None, item))

    deadline_wall = time.perf_counter() + DRAIN_TIMEOUT
    for job, _ in jobs:
        if job is None:
            continue
        remaining = max(0.0, deadline_wall - time.perf_counter())
        job.done.wait(timeout=remaining)

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
    return {
        "mode": MODE,
        "rate_docs_per_s": rate,
        "deadline_ms": DEADLINE_MS,
        "injected": n,
        "dropped": dropped,
        "answered_documents": len(latencies_ms),
        "answered_questions": answered_questions,
        "questions_per_second": round(answered_questions / wall, 2) if wall else 0.0,
        "deadline_hit_rate": round(hit / len(latencies_ms), 4) if latencies_ms else None,
        "p50_ms": round(statistics.median(latencies_ms), 1) if latencies_ms else None,
        "p99_ms": round(sorted(latencies_ms)[int(0.99 * (len(latencies_ms) - 1))], 1) if latencies_ms else None,
        "fused_single_passes": bstats.get("fused_single_passes"),
        "fused_many_passes": bstats.get("fused_many_passes"),
        "blocked_not_all_fresh": bstats.get("blocked_not_all_fresh"),
        "blocked_capacity": bstats.get("blocked_capacity"),
        "total_passes": bstats.get("total_passes"),
    }


def main() -> None:
    items = tasks.load("race", POOL_DOCUMENTS, split="validation", seed=0)
    print(f"{len(items)} documents available, mode={MODE}", flush=True)

    kwargs = dict(paged=True, graphs=True, require_kernels=True)
    if MODE == "interleaved":
        kwargs["interleaved_fork"] = True
    engine = Prismyra(MODEL, **kwargs)

    for rate in RATES:
        batcher = Batcher(engine).start()
        for item in items[:5]:
            batcher.ask(item.context, item.questions, timeout=30.0)
        report = run_one(engine, batcher, items, rate)
        print(json.dumps(report, indent=2), flush=True)
        Path(f"/work/matrix_{MODE}_rate{rate}.json").write_text(json.dumps(report, indent=2))
        batcher.stop()
        # A fresh Batcher (and the shelf it opens on first use) per rate, so a later rate does not inherit
        # an earlier rate's resident documents or pass-count stats -- each rate is its own measurement.


if __name__ == "__main__":
    main()
