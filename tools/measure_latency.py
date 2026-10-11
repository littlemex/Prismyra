"""learn2: Prismyra's own per-item serving latency for the two tags, to put next to the student's CPU
inference time in the report (task point 4: "生徒の推論の速さ...も、Prismyraと並べて書く"). Sequential
(not concurrent) single-item /ask calls -- the per-request latency a lone caller actually experiences,
not the throughput-maximising concurrent-batch number collect_pools.py measures. 60 requests per tag
after 5 warm-up requests (consistent with this project's shared-memory rule of >=5 repeated
measurements and reporting the median plus min/max), reading the real `timing.total_ms` the server
itself reports.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from snake_port import build_request, sample_states_with_up_legal

TWEET_LABELS = ["negative", "neutral", "positive"]
TWEET_QUESTION = "What sentiment does this tweet express?"


def main() -> int:
    model = os.environ.get("PRISMYRA_TEST_MODEL")
    if not model:
        raise SystemExit("set PRISMYRA_TEST_MODEL")

    from fastapi.testclient import TestClient

    from prismyra.server import create_app

    app = create_app(model, max_queue=64, batcher=False, learn_spec=None)
    client = TestClient(app)

    out = {}

    # snake: single "up" question, same shape as the registered tag.
    rng = random.Random(999001)
    states = sample_states_with_up_legal(rng, 65)
    times = []
    for i, s in enumerate(states):
        context, questions, cands = build_request(s)
        up_idx = next(j for j, c in enumerate(cands) if c.dir == "up")
        q = [questions[up_idx]]
        resp = client.post("/ask", json={"context": context, "questions": q})
        resp.raise_for_status()
        ms = resp.json()["timing"]["total_ms"]
        if i >= 5:  # discard 5 warm-up calls
            times.append(ms)
    out["snake"] = {"n": len(times), "median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times)}

    # tweet: single classification question.
    texts = [f"warmup or real tweet number {i} with some words to tokenize properly here" for i in range(65)]
    times = []
    for i, t in enumerate(texts):
        resp = client.post(
            "/ask",
            json={
                "context": t,
                "questions": [{"id": "sentiment", "prompt": TWEET_QUESTION, "kind": "choice", "choices": TWEET_LABELS}],
            },
        )
        resp.raise_for_status()
        ms = resp.json()["timing"]["total_ms"]
        if i >= 5:
            times.append(ms)
    out["tweet"] = {"n": len(times), "median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times)}

    print(json.dumps(out, indent=2))
    Path("/work/learn2/prismyra_latency.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
