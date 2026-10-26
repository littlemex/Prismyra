"""Real-hardware latency check for stage 4 + telemetry: p50/p99 of `/ask`, disabled vs. --learn-spec
(unrouted and routed) vs. --otel, against a real engine. Not part of the pytest suite (no device there) --
run by hand on the machine named in the run log, the same convention `docs/PERFORMANCE.md` already uses for
every other number in this project. One real HTTP round trip per sample, against a real `uvicorn` server in
a subprocess, not an in-process `TestClient` -- the whole point is wall-clock latency including the network
stack this project's own HTTP server actually runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx


def wait_for_health(base_url: str, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.RequestError:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"{base_url} never became healthy within {timeout}s")


_STUB_COLLECTOR = """
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.end_headers()
        self.wfile.write(b"")

HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


def run_stub_collector(port: int):
    """A real local OTLP/HTTP receiver that accepts and discards every export -- stand-in for a real
    collector, so the otel-enabled measurement exercises actual network round trips to a *reachable*
    endpoint rather than paying the export library's own retry/backoff cost against nothing."""
    return subprocess.Popen([sys.executable, "-c", _STUB_COLLECTOR, str(port)])


def run_server(args: list[str], port: int, log_path: Path):
    log = log_path.open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "prismyra.server", "--port", str(port), *args],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return proc, log


def sample_latencies(base_url: str, body: dict, n: int) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.monotonic()
        r = httpx.post(f"{base_url}/ask", json=body, timeout=60.0)
        r.raise_for_status()
        out.append((time.monotonic() - t0) * 1000.0)
    return out


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    idx = min(len(s) - 1, round(p * (len(s) - 1)))
    return s[idx]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--work-dir", default="/work/learn-router-verify/bench")
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    spec_path = work / "learn.json"
    spec_path.write_text(
        json.dumps(
            [
                {
                    "task": "bench-v1",
                    "question": "Is this positive?",
                    "options": ["no", "yes"],
                    "kind": "boolean",
                    "normalize": "strip+lower",
                    "retain_days": 1,
                    "keep_hidden": False,
                    "eval_set": "e.jsonl",
                }
            ]
        )
    )

    body = {
        "context": "The weather today is sunny and warm.",
        "questions": [{"id": "a", "prompt": "Is this positive?", "kind": "boolean"}],
    }

    collector_port = 4318
    collector = run_stub_collector(collector_port)
    time.sleep(1.0)

    configs = {
        "baseline": [],
        "learn_unrouted": ["--learn-spec", str(spec_path), "--learn-log-dir", str(work / "exp")],
        "otel": ["--otel", "--otel-endpoint", f"http://127.0.0.1:{collector_port}"],
    }

    results: dict[str, dict] = {}
    try:
        for name, extra_args in configs.items():
            port = 8600 + abs(hash(name)) % 100
            log_path = work / f"{name}.log"
            proc, log = run_server(["--model", args.model, *extra_args], port, log_path)
            base_url = f"http://127.0.0.1:{port}"
            try:
                wait_for_health(base_url)
                sample_latencies(base_url, body, args.warmup)  # warm up, not measured
                latencies = sample_latencies(base_url, body, args.n)
                results[name] = {
                    "n": len(latencies),
                    "p50_ms": pct(latencies, 0.50),
                    "p99_ms": pct(latencies, 0.99),
                    "mean_ms": statistics.mean(latencies),
                }
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                log.close()
    finally:
        collector.terminate()
        try:
            collector.wait(timeout=5)
        except subprocess.TimeoutExpired:
            collector.kill()

    baseline = results["baseline"]
    print(json.dumps(results, indent=2))
    for name in ("learn_unrouted", "otel"):
        p50_pct = (results[name]["p50_ms"] / baseline["p50_ms"] - 1) * 100
        p99_pct = (results[name]["p99_ms"] / baseline["p99_ms"] - 1) * 100
        print(f"{name}: p50 {p50_pct:+.2f}% (budget +2%), p99 {p99_pct:+.2f}% (budget +5%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
