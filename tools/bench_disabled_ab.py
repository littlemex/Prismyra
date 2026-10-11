"""learn1's own real-machine check for DISTILL-RL-DESIGN-v2.md section 8: without `--learn-spec`, the server
must answer bit-identically to the unmodified code and at the same speed (within measurement noise).

Not shipped as part of the package; a one-off tool for this change's own verification, kept the same way
`tools/audit_sm120.py` is -- a real measurement, not a unit test, because it needs the real model and the real
device.

Usage (run once per prismyra checkout under test, with that checkout's `prismyra` importable e.g. via
PYTHONPATH):

    PRISMYRA_TEST_MODEL=littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l \
      python3 tools/bench_disabled_ab.py --out results_main.json --label main

    PRISMYRA_TEST_MODEL=littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l \
      python3 tools/bench_disabled_ab.py --out results_mine.json --label mine

Then `tools/compare_disabled_ab.py results_main.json results_mine.json` reports the median/min/max per batch
size and whether every response byte (after the server's own 6-decimal rounding) is identical.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

CONTEXT = (
    "The committee reviewed the quarterly report before the board meeting. Revenue grew eight percent year "
    "over year, driven mostly by the enterprise segment, while the consumer segment declined slightly due to "
    "a pricing change introduced in the second quarter. Operating costs rose in line with headcount growth in "
    "engineering and support. The board asked for a deeper breakdown of churn in the consumer segment before "
    "approving next year's budget. Legal flagged one open contract dispute with a supplier over delivery "
    "delays, currently in arbitration and not expected to resolve before year end. The audit committee noted "
    "no material weaknesses in internal controls this quarter."
)

QUESTION_BANK = [
    ("revenue_grew", "Did revenue grow year over year?", "boolean"),
    ("consumer_declined", "Did the consumer segment decline?", "boolean"),
    ("who_flagged_dispute", "Which team flagged the contract dispute?", "choice", ["legal", "engineering", "support"]),
    ("segment_driving_growth", "Which segment mostly drove the growth?", "choice", ["enterprise", "consumer"]),
    ("urgency", "How urgent is the open contract dispute?", "scale", None, 1, 5),
]


def build_questions(n: int) -> list[dict]:
    out = []
    for i in range(n):
        bank_i = QUESTION_BANK[i % len(QUESTION_BANK)]
        qid, prompt, kind = bank_i[0], bank_i[1], bank_i[2]
        item = {"id": f"{qid}_{i}", "prompt": prompt, "kind": kind}
        if kind == "choice":
            item["choices"] = bank_i[3]
        elif kind == "scale":
            item["low"], item["high"] = bank_i[4], bank_i[5]
        out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sizes", default="1,16,64")
    parser.add_argument("--reps", type=int, default=9)
    args = parser.parse_args()

    model = os.environ.get("PRISMYRA_TEST_MODEL")
    if not model:
        raise SystemExit("set PRISMYRA_TEST_MODEL")

    from fastapi.testclient import TestClient

    from prismyra.server import create_app

    print(f"[{args.label}] loading {model} ...", flush=True)
    app = create_app(model, max_queue=64)
    client = TestClient(app)
    print(f"[{args.label}] loaded, warming up ...", flush=True)
    client.post("/ask", json={"context": CONTEXT, "questions": build_questions(1)})

    sizes = [int(s) for s in args.sizes.split(",")]
    results: dict[str, dict] = {}
    for size in sizes:
        questions = build_questions(size)
        # A batch size seen for the first time pays a one-off cost (new row/column shape through autotuned
        # kernels, a graph capture, …) that has nothing to do with which checkout is running. Warming up each
        # size once, not just size 1, is what keeps that cost out of the measured reps below.
        client.post("/ask", json={"context": CONTEXT, "questions": questions})
        latencies = []
        last_body = None
        for rep in range(args.reps):
            t0 = time.perf_counter()
            resp = client.post("/ask", json={"context": CONTEXT, "questions": questions})
            dt_ms = (time.perf_counter() - t0) * 1e3
            assert resp.status_code == 200, resp.text
            body = resp.json()
            latencies.append(dt_ms)
            last_body = body
            print(f"[{args.label}] size={size} rep={rep} total_ms={dt_ms:.2f}", flush=True)
        results[str(size)] = {
            "latencies_ms": latencies,
            "median_ms": statistics.median(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "answers": {k: v["probabilities"] for k, v in last_body["answers"].items()},
        }

    with open(args.out, "w") as fh:
        json.dump({"label": args.label, "model": model, "results": results}, fh, indent=2)
    print(f"[{args.label}] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
