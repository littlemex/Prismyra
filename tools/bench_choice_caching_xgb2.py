"""xgb-student2's real-machine measurement of the `prismyra.decide.choice_labels` caching fix: p50/p99 of
`POST /v1/decide` with a `kind="choice"` item, over a real HTTP connection to a real running `prismyra-serve`
process (not `fastapi.testclient.TestClient`'s in-process ASGI transport -- that would skip the actual socket
and uvicorn event loop, and the complaint this fix answers (RUN-xgb-student.md) was phrased as "HTTP 込みの
実測"). No `--learn-spec` at all: the cost this measures (`choice_labels` re-probing the tokenizer on every
request) exists independently of whether any tag or router is configured, so isolating it needs nothing but
a bare server and a repeated `choice` question -- see `prismyra.decide.decide_many`'s own call to
`choice_labels`, unconditional on any item actually needing the engine.

Not shipped as part of the package; a one-off script for this round's own verification.

Usage: start a real `prismyra-serve` process first (two of them, one per checkout, different ports -- see
`docs/LEARN.md`'s own account of this round for the exact commands used), then:

    python3.12 tools/bench_choice_caching_xgb2.py --port 8001 --label before --n 200 --warmup 10
    python3.12 tools/bench_choice_caching_xgb2.py --port 8002 --label after --n 200 --warmup 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request


def post_decide(base_url: str, options: list[str], text: str) -> float:
    payload = json.dumps(
        {"kind": "choice", "state": text, "question": "Which of these best fits?", "options": options}
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/decide", data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()
    t1 = time.perf_counter()
    if resp.status != 200:
        raise RuntimeError(f"unexpected status {resp.status}")
    return (t1 - t0) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--label", required=True, help="a name for this run (e.g. 'before'/'after'), echoed in the output")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--n-options", type=int, default=50, help="how many choice options per request")
    parser.add_argument("--out", default=None, help="write the result JSON here too")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    options = [f"cat{i:03d}" for i in range(args.n_options)]

    # Wait for the server to answer health, rather than assuming it is already up.
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200:
                    break
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1)
    else:
        raise SystemExit(f"server at {base_url} never became healthy")

    for i in range(args.warmup):
        post_decide(base_url, options, f"warmup request number {i} with some representative words in it")

    times = []
    for i in range(args.n):
        ms = post_decide(base_url, options, f"benchmark request number {i} with some representative words in it")
        times.append(ms)

    times.sort()
    p50 = statistics.median(times)
    p99 = times[int(len(times) * 0.99)] if len(times) > 1 else times[0]
    result = {
        "label": args.label,
        "n": args.n,
        "n_options": args.n_options,
        "p50_ms": p50,
        "p99_ms": p99,
        "min_ms": min(times),
        "max_ms": max(times),
        "mean_ms": statistics.mean(times),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
