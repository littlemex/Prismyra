"""Real-hardware check that idle-time retraining's own CPU work does not degrade `/ask` latency by more
than DISTILL-RL-DESIGN-v2.md section 5's own "空き時間の仕事が...乱さないこと" budget (p50 +2%/p99 +5%, the
same figures section 8 already uses for the learning layer's own request-path cost). Not part of the
pytest suite -- run by hand on the machine named in the run log, same convention every other number in this
project uses.

Part A: a continuous closed-loop `/ask` load (three question counts, one client thread each) runs the
whole time; a background process doing nothing but call `prismyra.learn.fit.run()` back-to-back, at the
same `nice`/`num_threads` `prismyra-learn-idle` defaults to, is switched on and off in alternating windows.
Part B: the real `prismyra-learn-idle` loop, against the same loaded server, to confirm it stays idle
(never actually fits) while the queue is non-empty, and to sample CPU usage while it is correctly doing
nothing but poll.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import psutil


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


def load_worker(base_url: str, qcount: int, stop_event: threading.Event, samples: list):
    body = {
        "context": "The weather today is sunny and pleasant for a walk in the park.",
        "questions": [{"id": f"q{i}", "prompt": "Is this positive?", "kind": "boolean"} for i in range(qcount)],
    }
    while not stop_event.is_set():
        t0 = time.monotonic()
        try:
            r = httpx.post(f"{base_url}/ask", json=body, timeout=60.0)
            r.raise_for_status()
            ok = True
        except httpx.HTTPError:
            ok = False
        samples.append((time.monotonic(), (time.monotonic() - t0) * 1000.0, ok))


_IDLE_FIT_LOOP = """
import os, sys, time
_blas_vars = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
for _var in _blas_vars:
    os.environ.setdefault(_var, str({num_threads}))
os.nice({nice})
sys.path.insert(0, {src!r})
from prismyra.learn.fit import run
spec_path = {spec_path!r}
log_dir = {log_dir!r}
n = 0
while True:
    run(spec_path=spec_path, task="idle-bench-v1", log_dir=log_dir, package_version="0", backbone="stub",
        dims=256, held_out_fraction=0.2, num_threads={num_threads}, seed=n, num_boost_round=100, min_data_in_leaf=5)
    n += 1
"""

# `prismyra-learn-idle`'s own real cadence is one fit per idle window *reached*, then it waits for the next
# scheduled poll (`--poll-interval-s`, default 30s) before even considering another -- never back-to-back.
# `_IDLE_FIT_LOOP` above is the deliberately pessimistic stress case (every window is spent fitting,
# continuously); `_IDLE_FIT_ONCE` matches the real cadence and is what the window-level numbers below are
# reported against, with the loop version kept as the worst-case upper bound.
_IDLE_FIT_ONCE = """
import os, sys, time
_blas_vars = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
for _var in _blas_vars:
    os.environ.setdefault(_var, str({num_threads}))
os.nice({nice})
sys.path.insert(0, {src!r})
from prismyra.learn.fit import run
run(spec_path={spec_path!r}, task="idle-bench-v1", log_dir={log_dir!r}, package_version="0", backbone="stub",
    dims=256, held_out_fraction=0.2, num_threads={num_threads}, seed={seed}, num_boost_round=100, min_data_in_leaf=5)
"""


def start_idle_fit_loop(src: str, spec_path: str, log_dir: str, nice: int, num_threads: int) -> subprocess.Popen:
    code = _IDLE_FIT_LOOP.format(nice=nice, src=src, spec_path=spec_path, log_dir=log_dir, num_threads=num_threads)
    return subprocess.Popen([sys.executable, "-c", code])


def run_idle_fit_once(
    src: str, spec_path: str, log_dir: str, nice: int, num_threads: int, seed: int
) -> tuple[float, float]:
    """Runs exactly one `fit.run()` call, in its own process, and returns its own ``(start, end)``
    (`time.monotonic()`) -- the precise sub-interval `summarize()` isolates to measure the instantaneous
    impact, as opposed to an "on" window's own, far more diluted average."""
    code = _IDLE_FIT_ONCE.format(
        nice=nice, src=src, spec_path=spec_path, log_dir=log_dir, num_threads=num_threads, seed=seed
    )
    t0 = time.monotonic()
    subprocess.run([sys.executable, "-c", code], check=True)
    return t0, time.monotonic()


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(p * (len(s) - 1)))]


def summarize(
    samples: list[tuple[float, float, bool]], window_bounds: list[tuple[float, float, str]], state: str
) -> dict:
    windows = [(s, e) for s, e, st in window_bounds if st == state]
    return summarize_intervals(samples, windows)


def summarize_intervals(samples: list[tuple[float, float, bool]], intervals: list[tuple[float, float]]) -> dict:
    picked = [lat for ts, lat, ok in samples if ok and any(s <= ts <= e for s, e in intervals)]
    duration = sum(e - s for s, e in intervals)
    if not picked:
        return {"n": 0, "p50_ms": None, "p99_ms": None, "throughput_rps": 0.0}
    return {
        "n": len(picked),
        "p50_ms": pct(picked, 0.50),
        "p99_ms": pct(picked, 0.99),
        "throughput_rps": len(picked) / duration if duration > 0 else 0.0,
    }


def part_a(args) -> dict:
    base_url = f"http://127.0.0.1:{args.port}"
    log = open(Path(args.work_dir) / "server_a.log", "w")  # noqa: SIM115 - held open across the whole function, closed in its own finally
    server = subprocess.Popen(
        [sys.executable, "-m", "prismyra.server", "--port", str(args.port), "--model", args.model],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    wait_for_health(base_url)

    samples_by_q = {q: [] for q in args.qcounts}
    stop_event = threading.Event()
    threads = [
        threading.Thread(target=load_worker, args=(base_url, q, stop_event, samples_by_q[q]), daemon=True)
        for q in args.qcounts
    ]
    for t in threads:
        t.start()
    time.sleep(10.0)  # let the load settle (cold-start compile, first-call warmup) before the first measured window

    window_bounds: list[tuple[float, float, str]] = []  # whole windows, "off" or "on" (on = window contains a fit)
    fit_intervals: list[tuple[float, float]] = []  # precise [start, end] of each actual fit() call
    fit_proc = None
    data_dir = str(Path(args.work_dir) / "data")
    try:
        for i in range(args.windows):
            state = "on" if i % 2 == 1 else "off"
            t0 = time.monotonic()
            if state == "on":
                if args.stress:
                    # Worst case: the window is spent fitting continuously, back-to-back -- see
                    # _IDLE_FIT_LOOP's own docstring for why this is more pessimistic than
                    # prismyra-learn-idle's real cadence, kept as an upper bound, not the headline number.
                    fit_proc = start_idle_fit_loop(
                        str(Path(args.src).resolve()),
                        f"{data_dir}/learn.json",
                        f"{data_dir}/experience",
                        args.nice,
                        args.num_threads,
                    )
                    time.sleep(args.window_s)
                    fit_proc.terminate()
                    fit_proc.wait(timeout=10)
                    fit_proc = None
                else:
                    # The realistic cadence: one fit, timed precisely, then idle for the rest of the window --
                    # exactly what prismyra-learn-idle itself does once per idle window it reaches.
                    f0, f1 = run_idle_fit_once(
                        str(Path(args.src).resolve()),
                        f"{data_dir}/learn.json",
                        f"{data_dir}/experience",
                        args.nice,
                        args.num_threads,
                        i,
                    )
                    fit_intervals.append((f0, f1))
                    remaining = args.window_s - (f1 - f0)
                    if remaining > 0:
                        time.sleep(remaining)
            else:
                time.sleep(args.window_s)
            t1 = time.monotonic()
            window_bounds.append((t0, t1, state))
            print(f"window {i} ({state}) done", flush=True)
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=5)
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()

    result = {}
    for q in args.qcounts:
        off = summarize(samples_by_q[q], window_bounds, "off")
        on_window = summarize(samples_by_q[q], window_bounds, "on")
        on_during_fit = summarize_intervals(samples_by_q[q], fit_intervals)

        def diff(a, b, key):
            return (a[key] / b[key] - 1) * 100 if b.get(key) and a.get(key) else None

        result[str(q)] = {
            "off": off,
            "on_window": on_window,
            "on_during_fit": on_during_fit,
            "on_window_p50_diff_pct": diff(on_window, off, "p50_ms"),
            "on_window_p99_diff_pct": diff(on_window, off, "p99_ms"),
            "on_window_throughput_diff_pct": diff(on_window, off, "throughput_rps"),
            "on_during_fit_p50_diff_pct": diff(on_during_fit, off, "p50_ms"),
            "on_during_fit_p99_diff_pct": diff(on_during_fit, off, "p99_ms"),
            "on_during_fit_throughput_diff_pct": diff(on_during_fit, off, "throughput_rps"),
        }
    return result


def part_b(args) -> dict:
    base_url = f"http://127.0.0.1:{args.port + 1}"
    log = open(Path(args.work_dir) / "server_b.log", "w")  # noqa: SIM115 - held open across the whole function, closed in its own finally
    server = subprocess.Popen(
        [sys.executable, "-m", "prismyra.server", "--port", str(args.port + 1), "--model", args.model],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    wait_for_health(base_url)

    stop_event = threading.Event()
    samples: list = []
    # Two concurrent closed-loop clients so the worker's own queue depth stays above 0 continuously --
    # `prismyra.learn.idle.is_busy`'s own floor (`max_queue_depth=0`, the default) should then never see
    # an idle poll.
    # qcount=1: see RUN-learn-router.md for why counts above 1 could not be used in this environment (a
    # pre-existing, unrelated engine/transformers incompatibility in the multi-question branch path, not
    # touched by this change). Two concurrent closed-loop clients still keep the worker's queue non-empty.
    threads = [
        threading.Thread(target=load_worker, args=(base_url, 1, stop_event, samples), daemon=True) for _ in range(2)
    ]
    for t in threads:
        t.start()
    time.sleep(10.0)

    cpu_samples = []
    cpu_stop = threading.Event()

    def sample_cpu():
        psutil.cpu_percent(interval=None)
        while not cpu_stop.is_set():
            cpu_samples.append(psutil.cpu_percent(interval=1.0))

    cpu_thread = threading.Thread(target=sample_cpu, daemon=True)
    cpu_thread.start()

    idle_log = open(Path(args.work_dir) / "idle_part_b.log", "w")  # noqa: SIM115 - same reason
    idle_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "prismyra.learn.idle",
            "--server-url",
            base_url,
            "--learn-spec",
            str(Path(args.work_dir) / "data" / "learn.json"),
            "--log-dir",
            str(Path(args.work_dir) / "data" / "experience"),
            "--backbone",
            args.model,
            "--state-path",
            str(Path(args.work_dir) / "data" / "idle_state_b.json"),
            "--poll-interval-s",
            "3",
            "--idle-polls-required",
            "3",
            "--stop-after",
            str(args.idle_b_polls),
            "--nice",
            str(args.nice),
            "--num-threads",
            str(args.num_threads),
        ],
        stdout=idle_log,
        stderr=subprocess.STDOUT,
    )
    try:
        idle_proc.wait(timeout=args.idle_b_polls * 4 + 30)
    except subprocess.TimeoutExpired:
        idle_proc.kill()
        idle_proc.wait(timeout=10)
    finally:
        cpu_stop.set()
        cpu_thread.join(timeout=5)
        stop_event.set()
        for t in threads:
            t.join(timeout=5)
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
        idle_log.close()

    state_path = Path(args.work_dir) / "data" / "idle_state_b.json"
    state_after = json.loads(state_path.read_text()) if state_path.is_file() else {}
    return {
        "idle_exit_code": idle_proc.returncode,
        "idle_log_tail": idle_log_tail(Path(args.work_dir) / "idle_part_b.log"),
        "state_after": state_after,
        "cpu_percent_samples": cpu_samples,
        "cpu_percent_mean": statistics.mean(cpu_samples) if cpu_samples else None,
        "cpu_percent_max": max(cpu_samples) if cpu_samples else None,
    }


def idle_log_tail(path: Path, n: int = 20) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text().splitlines()[-n:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--src", default="/work/idle-verify/src")
    parser.add_argument("--work-dir", default="/work/idle-verify")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--window-s", type=float, default=20.0)
    parser.add_argument("--nice", type=int, default=19)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--idle-b-polls", type=int, default=15)
    parser.add_argument("--part", choices=["a", "b", "both"], default="both")
    parser.add_argument("--qcounts", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument(
        "--stress",
        action="store_true",
        help="worst case: fit continuously through the whole 'on' window, instead of prismyra-learn-idle's "
        "real one-fit-per-window-reached cadence",
    )
    args = parser.parse_args()

    out = {}
    if args.part in ("a", "both"):
        out["part_a"] = part_a(args)
    if args.part in ("b", "both"):
        out["part_b"] = part_b(args)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
