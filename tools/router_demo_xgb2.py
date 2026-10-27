"""xgb-student2's real-server router demo: `ab` (both student backends), then `shadow` -> `confidence` ->
`ramp` deliberately sharing one `--learn-log-dir` across the mode switches -- the exact real-hardware
sequence RUN-xgb-student.md found broken (confidence silently inheriting shadow's own `ratio=0`, read
literally instead of through `confidence_ratio_gate`'s "0 means no extra gate" rule) -- to show the fix in
`prismyra.learn.controller.Controller._state_for` holds under a real running server, not only under the
deterministic `FakeClock` unit test (`tests/test_learn_controller.py::
test_switching_mode_reinitialises_ratio_instead_of_carrying_the_old_mode_over`) that pins the same scenario.

Not shipped as part of the package; a one-off script for this round's own real-machine verification,
resident for the whole sequence so it can manage each phase's `prismyra-serve` subprocess lifecycle itself
(start, wait for /health, drive requests, stop, wait for the GPU to actually free) rather than depending on
several separate shell invocations to agree on process state across calls.

Usage (on a box with the real model and a GPU, after `tools/collect_and_fit_xgb2.py` has written a
`StudentArtifact` JSON for the task this will route):

    python3.12 tools/router_demo_xgb2.py \\
      --student-artifact /work/xgb-student2-verify/run/snake-move-up-v1__lightgbm.json \\
      --xgboost-student-artifact /work/xgb-student2-verify/run/snake-move-up-v1__xgboost.json \\
      --task snake-move-up-v1 --kind boolean \\
      --options no,yes --question "Is moving up the best of the directions listed above?" \\
      --eval-jsonl /work/xgb-student2-verify/run/snake_eval.jsonl \\
      --out-dir /work/xgb-student2-verify/router-demo --port 8011
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def wait_healthy(base_url: str, timeout_s: float = 240.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    raise TimeoutError(f"{base_url} never became healthy within {timeout_s}s")


#: `snake-move-up-v1`'s own `StudentArtifact.features` names `"snake_board_v1"` -- a name `prismyra.learn.
#: features.get_feature_builder` only finds once `tools/snake_features.py` has actually been imported
#: (`register_features` runs as that import's own side effect, same as the collection script relies on). A
#: `prismyra-serve` subprocess started the normal way (`python -m prismyra.server`) never imports it, so the
#: student's own feature transform fails with `FeatureError: no feature builder named 'snake_board_v1' is
#: registered` the moment a request actually reaches it -- real-server-only, not something an in-process
#: `TestClient` run (which already had `tools/snake_features.py` imported by its own caller) would ever show.
#: Fixed by importing it ourselves before `prismyra.server.main()` runs, in the same process.
_SERVER_BOOTSTRAP = (
    "import sys; sys.path.insert(0, {tools_dir!r}); import snake_features; "
    "from prismyra.server import main; sys.exit(main())"
)


def start_server(model: str, spec_path: Path, log_dir: Path, port: int, log_file: Path) -> subprocess.Popen:
    tools_dir = str(Path(__file__).parent)
    cmd = [
        sys.executable,
        "-c",
        _SERVER_BOOTSTRAP.format(tools_dir=tools_dir),
        "--model",
        model,
        "--port",
        str(port),
        "--learn-spec",
        str(spec_path),
        "--learn-log-dir",
        str(log_dir),
    ]
    env = dict(os.environ)
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    return proc


def stop_server(proc: subprocess.Popen, port: int) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)
    # Wait for the port to actually free and the GPU memory this process held to actually drop, so the next
    # phase's `prismyra-serve` does not race it for either.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                time.sleep(2)
                continue
        except (urllib.error.URLError, OSError):
            break
    time.sleep(5)


def post_question(base_url: str, kind: str, question: str, options: list[str], context: str, task: str) -> dict:
    """One decision, over real HTTP, through whichever endpoint actually matches this tag.

    `/v1/decide` reads JEV's own kind vocabulary (`noul`/`score`/`choice`, `prismyra.decide.parse_item`), not
    the Python-question-class names (`boolean`/`choice`/`scale`) `prismyra.server.build_questions` reads for
    plain `/ask` -- a tag whose own spec entry says ``"kind": "boolean"`` (this round's `snake-move-up-v1`,
    carried over unmodified from `tools/collect_and_fit.py`'s own convention) only ever matches a request
    through `/ask`, because `prismyra.learn.tagging.tag_for` compares the *request's* kind string against the
    tag's, and nothing sends `/v1/decide` a `kind: "boolean"` (it would be refused as not one of `noul`/
    `score`/`choice`). So: `"boolean"`/`"scale"` go through `/ask` (`prismyra.server.ask`'s own response
    shape -- `answers[id].read_by`, `timing.total_ms`), and `"choice"` goes through `/v1/decide` as before.
    """
    t0 = time.perf_counter()
    if kind in ("boolean", "scale"):
        payload = json.dumps(
            {"context": context, "questions": [{"id": "q", "prompt": question, "kind": kind}]}
        ).encode()
        req = urllib.request.Request(
            f"{base_url}/ask", data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read())
        t1 = time.perf_counter()
        answer = raw["answers"]["q"]
        body = {
            "served_by": answer.get("read_by") or "prismyra",
            "server_total_ms": raw["timing"]["total_ms"],
            "_client_ms": (t1 - t0) * 1000.0,
            "_raw": raw,
        }
        return body
    payload = json.dumps(
        {"kind": kind, "task": task, "state": context, "question": question, "options": options}
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/decide", data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    t1 = time.perf_counter()
    body["_client_ms"] = (t1 - t0) * 1000.0
    return body


def get_stats(base_url: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/stats", timeout=10) as resp:
        return json.loads(resp.read())


def build_spec(task: str, question: str, options: list[str], kind: str, eval_set_name: str, router: dict) -> list[dict]:
    return [
        {
            "task": task,
            "question": question,
            "options": options,
            "kind": kind,
            "normalize": "strip+lower" if kind == "choice" else "none",
            "retain_days": 30,
            "keep_hidden": False,
            "eval_set": eval_set_name,
            "router": router,
        }
    ]


def run_phase(
    *,
    model: str,
    task: str,
    question: str,
    options: list[str],
    kind: str,
    out_dir: Path,
    port: int,
    phase_name: str,
    router_cfg: dict,
    log_dir: Path,
    contexts: list[str],
    n_requests: int,
) -> dict:
    print(f"\n=== phase: {phase_name} (router={router_cfg}, log_dir={log_dir}) ===", flush=True)
    spec_path = out_dir / f"spec_{phase_name}.json"
    spec_path.write_text(
        json.dumps(build_spec(task, question, options, kind, "placeholder_eval.jsonl", router_cfg), indent=2)
    )
    # `eval_set` must resolve relative to the spec's own directory for `prismyra.learn.spec.load_spec` to
    # accept it (`TagSpec.eval_set` is only ever opened by `prismyra.learn.fit`, never by the server, so a
    # missing file here is harmless for serving -- only the admission bar at fit time needs it to exist).
    (out_dir / "placeholder_eval.jsonl").touch(exist_ok=True)

    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = out_dir / f"server_{phase_name}.log"
    base_url = f"http://127.0.0.1:{port}"
    proc = start_server(model, spec_path, log_dir, port, server_log)
    try:
        wait_healthy(base_url)
        results = []
        for i in range(n_requests):
            ctx = contexts[i % len(contexts)]
            resp = post_question(base_url, kind, question, options, ctx, task)
            results.append(
                {
                    "served_by": resp.get("served_by", "prismyra"),
                    "client_ms": resp["_client_ms"],
                }
            )
        stats = get_stats(base_url)
        served_counts: dict[str, int] = {}
        for r in results:
            key = "student" if str(r["served_by"]).startswith("student") else "prismyra"
            served_counts[key] = served_counts.get(key, 0) + 1
        times = sorted(r["client_ms"] for r in results)
        p50 = times[len(times) // 2]
        p99 = times[int(len(times) * 0.99)] if len(times) > 1 else times[0]
        phase_result = {
            "phase": phase_name,
            "router_cfg": router_cfg,
            "n": n_requests,
            "served_counts": served_counts,
            "p50_ms": p50,
            "p99_ms": p99,
            "routing_stats": stats.get("learn", {}).get("routing", {}),
        }
        print(json.dumps(phase_result, indent=2), flush=True)
        return phase_result
    finally:
        stop_server(proc, port)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-artifact", required=True, help="lightgbm StudentArtifact JSON path")
    parser.add_argument("--xgboost-student-artifact", required=True, help="xgboost StudentArtifact JSON path")
    parser.add_argument("--task", required=True)
    parser.add_argument("--kind", required=True, choices=("choice", "boolean", "scale"))
    parser.add_argument("--question", required=True)
    parser.add_argument("--options", required=True, help="comma-separated")
    parser.add_argument("--eval-jsonl", required=True, help="JSONL with a 'context' field, used to drive requests")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--n-requests", type=int, default=60)
    args = parser.parse_args()

    model = os.environ.get("PRISMYRA_TEST_MODEL")
    if not model:
        raise SystemExit("set PRISMYRA_TEST_MODEL")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    options = args.options.split(",")

    contexts = []
    with open(args.eval_jsonl) as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                contexts.append(json.loads(stripped)["context"])
    if not contexts:
        raise SystemExit(f"no contexts found in {args.eval_jsonl}")
    print(f"driving requests from {len(contexts)} contexts in {args.eval_jsonl}", flush=True)

    lgbm_path = str(Path(args.student_artifact).resolve())
    xgb_path = str(Path(args.xgboost_student_artifact).resolve())

    all_results = []

    # Phase 1-2: "ab" with ratio=10000 (always student), one run per backend, fresh log dirs each -- the
    # LightGBM vs XGBoost serving comparison this round's own task asks for, independent of the mode-switch
    # bug fix below.
    for backend_name, student_path in (("lightgbm", lgbm_path), ("xgboost", xgb_path)):
        phase = f"ab_{backend_name}"
        router_cfg = {"mode": "ab", "student_path": student_path, "ratio": 10000}
        all_results.append(
            run_phase(
                model=model,
                task=args.task,
                question=args.question,
                options=options,
                kind=args.kind,
                out_dir=out_dir,
                port=args.port,
                phase_name=phase,
                router_cfg=router_cfg,
                log_dir=out_dir / f"log_{phase}",
                contexts=contexts,
                n_requests=args.n_requests,
            )
        )

    # Phase 3-5: the mode-switch bug's own real sequence -- shadow, then confidence, then ramp, deliberately
    # sharing ONE log dir across all three so each mode's `Controller` state is read back from what the
    # *previous* mode wrote, exactly RUN-xgb-student.md's own real-hardware repro.
    shared_log_dir = out_dir / "log_mode_switch_sequence"
    router_cfg = {"mode": "shadow", "student_path": lgbm_path}
    all_results.append(
        run_phase(
            model=model,
            task=args.task,
            question=args.question,
            options=options,
            kind=args.kind,
            out_dir=out_dir,
            port=args.port,
            phase_name="shadow",
            router_cfg=router_cfg,
            log_dir=shared_log_dir,
            contexts=contexts,
            n_requests=args.n_requests,
        )
    )
    router_cfg = {"mode": "confidence", "student_path": lgbm_path, "tau": 0.5}
    all_results.append(
        run_phase(
            model=model,
            task=args.task,
            question=args.question,
            options=options,
            kind=args.kind,
            out_dir=out_dir,
            port=args.port,
            phase_name="confidence",
            router_cfg=router_cfg,
            log_dir=shared_log_dir,
            contexts=contexts,
            n_requests=args.n_requests,
        )
    )
    router_cfg = {
        "mode": "ramp",
        "student_path": lgbm_path,
        "ramp_step": 2000,
        "ramp_interval_s": 8,
        "ramp_max_ratio": 10000,
    }
    all_results.append(
        run_phase(
            model=model,
            task=args.task,
            question=args.question,
            options=options,
            kind=args.kind,
            out_dir=out_dir,
            port=args.port,
            phase_name="ramp",
            router_cfg=router_cfg,
            log_dir=shared_log_dir,
            contexts=contexts,
            n_requests=args.n_requests,
        )
    )

    (out_dir / "router_demo_summary.json").write_text(json.dumps(all_results, indent=2))
    print("\nALL DONE. Wrote router_demo_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
