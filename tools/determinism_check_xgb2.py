"""xgb-student2's real-machine determinism check: the same input, 5 times within one server process, then
once more after a full process restart (fresh `prismyra-serve`, same artifact, same spec) -- the exact check
`docs/LEARN.md`'s existing determinism contract describes, re-run this round against the real `ab` (always
student) phase so this round's own report can cite a real-machine number rather than only the unit tests'.

Not shipped as part of the package; a one-off script for this round's own verification.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from router_demo_xgb2 import post_question, start_server, stop_server, wait_healthy


def main() -> int:
    model = os.environ.get("PRISMYRA_TEST_MODEL")
    if not model:
        raise SystemExit("set PRISMYRA_TEST_MODEL")

    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/work/xgb-student2-verify/determinism")
    out_dir.mkdir(parents=True, exist_ok=True)
    student_artifact = (
        sys.argv[2] if len(sys.argv) > 2 else "/work/xgb-student2-verify/run/snake-move-up-v1__lightgbm.json"
    )
    eval_jsonl = sys.argv[3] if len(sys.argv) > 3 else "/work/xgb-student2-verify/run/snake_eval.jsonl"
    task = "snake-move-up-v1"
    question = "Is moving up the best of the directions listed above?"
    options = ["no", "yes"]
    kind = "boolean"
    port = 8012

    with open(eval_jsonl) as f:
        context = json.loads(f.readline())["context"]

    spec_path = out_dir / "spec.json"
    spec_path.write_text(
        json.dumps(
            [
                {
                    "task": task,
                    "question": question,
                    "options": options,
                    "kind": kind,
                    "normalize": "none",
                    "retain_days": 30,
                    "keep_hidden": False,
                    "eval_set": "placeholder.jsonl",
                    "router": {"mode": "ab", "student_path": str(Path(student_artifact).resolve()), "ratio": 10000},
                }
            ],
            indent=2,
        )
    )
    (out_dir / "placeholder.jsonl").touch(exist_ok=True)

    def one_session(log_dir: Path, n: int) -> list[dict]:
        proc = start_server(model, spec_path, log_dir, port, out_dir / f"server_{log_dir.name}.log")
        try:
            wait_healthy(f"http://127.0.0.1:{port}")
            answers = []
            for _ in range(n):
                resp = post_question(f"http://127.0.0.1:{port}", kind, question, options, context, task)
                raw = resp.get("_raw", resp)
                answers.append(raw)
            return answers
        finally:
            stop_server(proc, port)

    same_session = one_session(out_dir / "log_session1", 5)
    restarted_session = one_session(out_dir / "log_session2", 1)

    def strip_volatile(d: dict) -> dict:
        d = json.loads(json.dumps(d))
        d.pop("request_id", None)
        if "timing" in d:
            d.pop("timing", None)
        return d

    stripped = [strip_volatile(a) for a in same_session]
    all_same_session_equal = all(s == stripped[0] for s in stripped)
    restart_equal = strip_volatile(restarted_session[0]) == stripped[0]

    result = {
        "same_session_5x_bitwise_equal_excluding_timing": all_same_session_equal,
        "after_restart_equal_excluding_timing": restart_equal,
        "same_session_answers": stripped,
        "after_restart_answer": strip_volatile(restarted_session[0]),
    }
    print(json.dumps(result, indent=2))
    (out_dir / "determinism_result.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
