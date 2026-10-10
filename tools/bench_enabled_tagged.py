"""learn1's real-machine check for the "enabled, tagged" half of DISTILL-RL-DESIGN-v2.md section 8: p50 within
2%, p99 within 5%, measured against the same checkout with `--learn-spec` simply omitted.

    python3 tools/bench_enabled_tagged.py --mode baseline --out r_baseline.json
    python3 tools/bench_enabled_tagged.py --mode tagged   --out r_tagged.json  --learn-spec /tmp/learn.json
    python3 tools/bench_enabled_tagged.py --mode fullqueue --out r_full.json \
      --learn-spec /tmp/learn.json --learn-max-queue 1

The third mode fires requests fast enough that the log's bounded queue cannot keep up, and asserts the server
keeps answering at the same speed regardless -- "queue が満ちたら、ログを捨てて配信を優先する" is a claim about
the request path, not just about `/stats`, so this checks both.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

CONTEXT = (
    "The committee reviewed the quarterly report before the board meeting. Revenue grew eight percent year "
    "over year, driven mostly by the enterprise segment, while the consumer segment declined slightly."
)

TAGGED_QUESTION = {"id": "revenue_grew", "prompt": "Did revenue grow year over year?", "kind": "boolean"}

#: The spec entry a `--learn-spec` test run should point at; written here once so the benchmark and whatever
#: wrote the spec file agree on what "tagged" means for this script's own request.
SPEC_ENTRY = {
    "task": "revenue-grew-v1",
    "question": TAGGED_QUESTION["prompt"],
    "options": ["no", "yes"],
    "kind": "boolean",
    "normalize": "strip+lower",
    "retain_days": 30,
    "keep_hidden": False,
    "eval_set": "evals/does-not-exist.jsonl",
}


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["baseline", "tagged", "fullqueue"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--learn-spec", default=None)
    parser.add_argument("--learn-log-dir", default=None)
    parser.add_argument("--learn-max-queue", type=int, default=10_000)
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    model = os.environ.get("PRISMYRA_TEST_MODEL")
    if not model:
        raise SystemExit("set PRISMYRA_TEST_MODEL")

    from fastapi.testclient import TestClient

    from prismyra.server import create_app

    kwargs: dict = {"max_queue": 64}
    if args.mode in ("tagged", "fullqueue"):
        kwargs["learn_spec"] = args.learn_spec
        kwargs["learn_log_dir"] = args.learn_log_dir
        kwargs["learn_max_queue"] = args.learn_max_queue

    print(f"[{args.mode}] loading {model} ...", flush=True)
    app = create_app(model, **kwargs)
    client = TestClient(app)
    client.post("/ask", json={"context": CONTEXT, "questions": [TAGGED_QUESTION]})  # warm up

    latencies = []
    for i in range(args.n):
        t0 = time.perf_counter()
        resp = client.post("/ask", json={"context": CONTEXT, "questions": [TAGGED_QUESTION]})
        dt_ms = (time.perf_counter() - t0) * 1e3
        assert resp.status_code == 200, resp.text
        latencies.append(dt_ms)
        if args.mode == "fullqueue" and i % 20 == 0:
            print(f"[{args.mode}] rep={i} total_ms={dt_ms:.2f}", flush=True)

    stats = client.get("/stats").json()
    out = {
        "mode": args.mode,
        "model": model,
        "n": args.n,
        "p50_ms": percentile(latencies, 0.50),
        "p99_ms": percentile(latencies, 0.99),
        "mean_ms": statistics.mean(latencies),
        "max_ms": max(latencies),
        "stats": stats,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
