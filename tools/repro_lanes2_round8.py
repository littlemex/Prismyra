"""inv5, round 8 (new task from the chair): reproduce `test_lanes_two_decisions_under_a_burst_do_not_move`'s
CUDA illegal-memory-access / NaN failure without paying model-load cost on every attempt -- builds one paged
engine and runs the test's own burst (20 documents, lanes=2, lane_room=2048, linger_ms=0.0) in a loop, stopping
at the first failure (job.error set, a decision/probability move, or an exception) so a race condition that
only shows up some fraction of the time has more tries per GPU session.

Usage: PRISMYRA_TEST_MODEL=<repo> python3 repro_lanes2_round8.py [n_attempts]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from prismyra import Boolean, Prismyra  # noqa: E402
from prismyra.schedule import Batcher  # noqa: E402

MODEL = os.environ.get("PRISMYRA_TEST_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
CONTEXT = (
    "Returns are accepted within thirty days of delivery. Unopened items are refunded in full. Opened items are "
    "exchanged rather than refunded, unless a manufacturing fault is confirmed. Return shipping is paid by the "
    "seller when the item is faulty and by the buyer otherwise."
)
COMPANION_MOVEMENT = 0.3


def one_attempt(engine: Prismyra, attempt: int) -> str | None:
    """Returns None on success, or a string describing the failure."""
    question = Boolean(id="faulty", prompt="Does the seller pay return shipping on a faulty item?")
    docs = [f"{CONTEXT} (document {i} of this run, otherwise identical to its neighbours.)" for i in range(20)]
    truth = engine.ask(CONTEXT, [question])["faulty"]

    batcher = Batcher(engine, lanes=2, lane_room=2048, linger_ms=0.0).start()
    try:
        jobs = [batcher.submit(context, [question]) for context in docs]
        for i, job in enumerate(jobs):
            if not job.done.wait(timeout=30):
                return f"attempt {attempt}: document {i} never answered"
            if job.error is not None:
                return f"attempt {attempt}: document {i} job.error={job.error!r}"
            got = job.result["faulty"]
            if got.option != truth.option:
                return f"attempt {attempt}: document {i} decision changed ({got.option} vs {truth.option})"
            for option, p in truth.probabilities.items():
                gp = got.probabilities[option]
                if gp != gp or abs(gp - p) >= COMPANION_MOVEMENT:  # gp != gp catches NaN
                    return f"attempt {attempt}: document {i} option {option} moved too far ({gp} vs {p})"
    except Exception as e:  # noqa: BLE001
        return f"attempt {attempt}: raised {type(e).__name__}: {e}"
    finally:
        batcher.stop()
    return None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    engine = Prismyra(MODEL)
    engine.paged = True
    print(f"model={MODEL} device={torch.cuda.get_device_name(0)} attempts={n}", flush=True)
    for attempt in range(n):
        failure = one_attempt(engine, attempt)
        if failure is not None:
            print(f"FAILURE on {failure}", flush=True)
            return 1
        print(f"attempt {attempt}: OK", flush=True)
    print("all attempts passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
