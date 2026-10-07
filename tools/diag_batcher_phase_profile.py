#!/usr/bin/env python3
"""fp8spd4 (task 3): where does Batcher open-loop time actually go?

fp8spd3 round5 found the fused single-document path (`Batcher._answer`'s `interleave.read_and_branch_shelf`
branch) is -13..-46% against the old `Shelf.put_many`+`Shelf.ask` pair in an unwarmed, fresh-Shelf-per-call
microbenchmark, but close to 0% inside the real `Batcher` open-loop (RACE documents, ~400-800 tok, rate=5:
service_ms -1.2%; rate=40: questions/s +0.5~1%) even though the fused path fired 76.7%/10.4% of passes
respectively. This instruments the real phases a pass goes through -- `Batcher.form`'s own linger wait,
`Batcher._make_room`'s admission/eviction check, and the actual read+branch compute (fused or the two-step
pair) -- with wall-clock timers around the *unmodified* methods (monkeypatched from outside, nothing in
`schedule.py`/`engine.py` changes), run inside the same open-loop harness fp8spd3 used
(`s4c_openloop_after_lowrate.py`), to see which of "読みの側、待ち、メモリ、group の詰め方" the fixed
per-pass cost the module's own docstring describes ("about 110ms, whatever it carries") actually is.

Usage: python3 diag_batcher_phase_profile.py [rate]
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "/work/fp8spd/fp8spd4-src/evals")
sys.path.insert(0, "/work/fp8spd/fp8spd4-src")

import torch  # noqa: E402
import tasks  # noqa: E402
from prismyra import Prismyra  # noqa: E402
from prismyra.schedule import Batcher  # noqa: E402

MODEL = "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l"
RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
PHASE_SECONDS = 20.0
DRAIN_TIMEOUT = 60.0
POOL_DOCUMENTS = 368

PHASE_TOTALS: dict[str, float] = {"form_wait": 0.0, "make_room": 0.0, "fused_core": 0.0, "unfused_put_many": 0.0,
                                   "unfused_ask": 0.0, "answer_total": 0.0}
PHASE_COUNTS: dict[str, int] = {k: 0 for k in PHASE_TOTALS}


def _timed(obj, attr, phase_key):
    original = getattr(obj, attr)

    def wrapped(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = original(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        PHASE_TOTALS[phase_key] += (time.perf_counter() - t0) * 1e3
        PHASE_COUNTS[phase_key] += 1
        return out

    setattr(obj, attr, wrapped)


def main() -> None:
    items = tasks.load("race", POOL_DOCUMENTS, split="validation", seed=0)
    print(f"{len(items)} documents available", flush=True)

    engine = Prismyra(MODEL, paged=True, graphs=True, require_kernels=True, interleaved_fork=True)
    batcher = Batcher(engine).start()

    # Instrument the real, unmodified methods in place -- bound methods on the live instances, so this changes
    # nothing about what runs, only adds a wall-clock timer (CUDA-synced) around each call.
    _timed(batcher, "form", "form_wait")
    _timed(batcher, "_make_room", "make_room")
    _timed(batcher, "_answer", "answer_total")
    _timed(engine, "_shelf_ask_interleaved", "fused_core")
    shelf = batcher._on_shelf()
    _timed(shelf, "put_many", "unfused_put_many")
    _timed(shelf, "ask", "unfused_ask")

    truth = {}
    for item in items:
        truth[id(item)] = engine.ask(item.context, item.questions)
    print(f"ground truth computed for {len(truth)} documents", flush=True)

    for item in items[:5]:
        batcher.ask(item.context, item.questions, timeout=30.0)
    print("warm-up done", flush=True)
    for k in PHASE_TOTALS:
        PHASE_TOTALS[k] = 0.0
        PHASE_COUNTS[k] = 0

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
    injected_seconds = time.perf_counter() - t0

    deadline = time.perf_counter() + DRAIN_TIMEOUT
    for job, _ in jobs:
        if job is None:
            continue
        remaining = max(0.0, deadline - time.perf_counter())
        job.done.wait(timeout=remaining)

    latencies_ms, answered_questions = [], 0
    first_queued = last_finished = None
    for job, item in jobs:
        if job is None or not job.done.is_set() or job.error is not None:
            continue
        latency = (job.finished_at - job.queued_at) * 1e3
        latencies_ms.append(latency)
        first_queued = job.queued_at if first_queued is None else min(first_queued, job.queued_at)
        last_finished = job.finished_at if last_finished is None else max(last_finished, job.finished_at)
        answered_questions += len(item.questions)

    wall = (last_finished - first_queued) if (first_queued and last_finished) else injected_seconds
    bstats = batcher.stats()

    report = {
        "rate_docs_per_s": RATE,
        "injected": n,
        "dropped": dropped,
        "answered_documents": len(latencies_ms),
        "answered_questions": answered_questions,
        "questions_per_second": round(answered_questions / wall, 2) if wall else 0.0,
        "p50_ms": round(statistics.median(latencies_ms), 1) if latencies_ms else None,
        "fused_single_passes": bstats.get("fused_single_passes"),
        "total_passes": bstats.get("total_passes"),
        "phase_totals_ms": {k: round(v, 1) for k, v in PHASE_TOTALS.items()},
        "phase_counts": PHASE_COUNTS,
        "phase_avg_ms_per_call": {
            k: round(PHASE_TOTALS[k] / PHASE_COUNTS[k], 2) if PHASE_COUNTS[k] else None for k in PHASE_TOTALS
        },
        "answer_total_wall_ms": round(PHASE_TOTALS["answer_total"], 1),
        "sum_of_subphases_ms": round(
            PHASE_TOTALS["form_wait"] + PHASE_TOTALS["make_room"] + PHASE_TOTALS["fused_core"]
            + PHASE_TOTALS["unfused_put_many"] + PHASE_TOTALS["unfused_ask"], 1
        ),
    }
    print(json.dumps(report, indent=2))
    Path(f"/work/fp8spd/runs/fp8spd4_batcher_phase_profile_rate{RATE}.json").write_text(json.dumps(report, indent=2))
    batcher.stop()


if __name__ == "__main__":
    main()
