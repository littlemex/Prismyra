#!/usr/bin/env python3
"""Measure the engine on the machine you are on, and compare against a checked-in result file.

Two commands, because measuring and judging are different jobs:

    prismyra-bench sweep                      # measure, print a table, optionally write a result file
    prismyra-bench compare --against FILE     # measure, then say whether anything moved

`sweep` measures only what a machine can measure about itself: the whole-job curve, the fit of the cost model, and the
per-kernel swap counts. The per-kernel profiles and the baseline engine's numbers in `results/*.json` come from
separate runs against that other engine and are recorded by hand; `--update` merges the measured keys into such a file
and leaves the hand-recorded ones alone, so a refresh cannot silently invent a baseline.

`compare` is the regression gate. CI cannot run it -- there is no GPU there -- so it is run by hand on the machine the
file names, and it fails loudly rather than printing a table nobody reads.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "benchmarks" / "results"

#: Question counts to measure. Chosen around the group boundary, since that is where the staircase steps and where a
#: regression in the branch pass would show up as a changed step rather than a changed level.
COUNTS = (1, 8, 16, 32, 64, 128)

#: Repeats per point. The first few are discarded: the first traversal of a shape pays for allocation and kernel
#: selection, which a served request does not.
REPEATS = 7
DISCARD = 3

#: A context long enough that the model's own work dominates, in the language the numbers were taken in. Japanese runs
#: about 0.516 tokens per character, so the token count is what to compare across languages and it is reported.
PARAGRAPH = (
    "返品は商品到着後三十日以内に限り受け付けます。未開封の商品は全額返金の対象となりますが、"
    "開封済みの商品については、初期不良が確認された場合を除き、返金ではなく交換のみの対応となります。"
    "返送にかかる送料は、初期不良の場合は当社が負担し、それ以外の理由による返品ではお客様のご負担となります。"
)


def build_questions(n: int):
    """`n` questions that differ in text but not in cost: one option pair, one branch each.

    Deliberately not `n` copies of one question. Identical prompts would tokenise to identical rows, and a kernel that
    quietly deduplicated them would look faster than it is.
    """
    from . import Boolean

    return [Boolean(id=f"q{i}", prompt=f"条項 {i} は返金を認めているか。") for i in range(n)]


def measure(engine, context: str, counts=COUNTS, repeats=REPEATS, discard=DISCARD) -> dict:
    """The whole-job curve: read the context and answer N questions, end to end, N times over.

    Wall time around the whole call, which includes tokenising and assembling the answers as well as the device. That
    is deliberate -- it is what a caller waits for -- but it is not the same as device time, and the result file says
    so.
    """
    out: dict[str, float] = {}
    for n in counts:
        questions = build_questions(n)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            engine.ask(context, questions)
            samples.append((time.perf_counter() - start) * 1e3)
        kept = samples[discard:]
        out[str(n)] = round(statistics.median(kept), 1)
    return out


def fit_cost_model(whole_job: dict[str, float], group: int) -> dict:
    """Recover the two constants from the curve. Two points decide them, and the rest are the test of the shape.

    The model is `context + per_group x ceil(n / group)`. The smallest count gives one group and the largest gives the
    most, so the slope comes from that pair and the intercept follows. Reporting the worst residual keeps the fit
    honest: a shape that has stopped being a staircase shows up there, not in the constants.
    """
    counts = sorted(int(k) for k in whole_job)
    lo, hi = counts[0], counts[-1]
    groups_lo, groups_hi = -(-lo // group), -(-hi // group)
    per_group = (whole_job[str(hi)] - whole_job[str(lo)]) / max(1, groups_hi - groups_lo)
    context_ms = whole_job[str(lo)] - per_group * groups_lo
    worst = max(abs(context_ms + per_group * -(-n // group) - whole_job[str(n)]) for n in counts)
    return {
        "note": "Fitted from the measured curve by prismyra-bench, not measured directly.",
        "context_ms": round(context_ms, 1),
        "per_group_ms": round(per_group, 1),
        "questions_per_group": group,
        "per_question_ms_at_full_group": round(per_group / group, 2),
        "worst_residual_ms": round(worst, 1),
    }


def environment() -> dict:
    import torch

    machine: dict = {"python": platform.python_version()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        machine["gpu"] = props.name
        machine["memory_gib"] = round(props.total_memory / 1024**3)
        machine["compute_capability"] = f"{props.major}.{props.minor}"
    software: dict[str, str | None] = {"torch": str(torch.__version__)}
    try:
        import vllm

        software["vllm"] = vllm.__version__
    except ImportError:
        software["vllm"] = None
    return {"machine": machine, "software": software}


def run_sweep(args) -> dict:
    from . import Prismyra

    engine = Prismyra(args.model, require_kernels=args.require_kernels)
    context = PARAGRAPH * args.repeat_paragraph
    tokens = len(engine.tokenizer(context)["input_ids"])

    whole_job = measure(engine, context, counts=tuple(args.counts), repeats=args.repeats)
    env = environment()
    return {
        "schema": 1,
        "measured_on": time.strftime("%Y-%m-%d"),
        **env,
        "model": args.model,
        "kernels": engine.applied.summary(),
        "context": {"characters": len(context), "tokens": tokens},
        "method": (
            f"Median of {args.repeats} in one process, the first {DISCARD} discarded. In-process end-to-end wall time "
            f"around `ask`: tokenising, the device, and building the answers, with no transport. Not device time alone."
        ),
        "whole_job": {"prismyra_ms": whole_job},
        "cost_model": fit_cost_model(whole_job, engine.group),
    }


def print_table(measured: dict) -> None:
    job = measured["whole_job"]["prismyra_ms"]
    model = measured["cost_model"]
    print(f"\n{measured['model']}  on  {measured['machine'].get('gpu', 'cpu')}")
    print(f"kernels: {measured['kernels']}")
    print(f"context: {measured['context']['tokens']} tokens\n")
    print(f"{'questions':>10}  {'total ms':>9}  {'per question':>13}")
    for n in sorted(job, key=int):
        print(f"{n:>10}  {job[n]:>9.1f}  {job[n] / int(n):>13.2f}")
    print(
        f"\ncost model: {model['context_ms']} ms + {model['per_group_ms']} ms x "
        f"ceil(questions / {model['questions_per_group']})   worst residual "
        f"{model['worst_residual_ms']} ms"
    )


def merge_into(path: Path, measured: dict) -> None:
    """Overwrite only the keys this harness measured, leaving the hand-recorded ones in place."""
    existing = json.loads(path.read_text()) if path.exists() else {}
    for key in ("measured_on", "machine", "software", "kernels", "context", "method", "cost_model"):
        if key in measured:
            existing[key] = measured[key]
    job = existing.setdefault("whole_job", {})
    job["prismyra_ms"] = measured["whole_job"]["prismyra_ms"]
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n")
    print(f"merged the measured keys into {path}")


def compare(measured: dict, reference: Path, tolerance: float) -> int:
    """Fail on any point that got slower by more than `tolerance`.

    One-sided on purpose. Getting faster is not a regression, but it is reported, because an unexplained improvement
    is usually a measurement that stopped measuring the same thing.
    """
    want = json.loads(reference.read_text())["whole_job"]["prismyra_ms"]
    got = measured["whole_job"]["prismyra_ms"]
    shared = sorted(set(want) & set(got), key=int)
    if not shared:
        print(f"no question count in common with {reference.name}", file=sys.stderr)
        return 1
    regressions = 0
    print(f"\n{'questions':>10}  {'reference':>10}  {'measured':>9}  {'change':>8}")
    for n in shared:
        change = (got[n] - want[n]) / want[n]
        flag = ""
        if change > tolerance:
            flag, regressions = "  SLOWER", regressions + 1
        elif change < -tolerance:
            flag = "  faster, and unexplained"
        print(f"{n:>10}  {want[n]:>10.1f}  {got[n]:>9.1f}  {change:>+7.1%}{flag}")
    if regressions:
        print(f"\n{regressions} point(s) slower than {reference.name} by more than {tolerance:.0%}", file=sys.stderr)
    return 1 if regressions else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prismyra-bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["sweep", "compare"])
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    parser.add_argument("--counts", type=int, nargs="+", default=list(COUNTS))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument(
        "--repeat-paragraph",
        type=int,
        default=40,
        help="how many times to repeat the sample paragraph; 40 is about 5,000 tokens",
    )
    parser.add_argument(
        "--require-kernels",
        action="store_true",
        help="refuse to measure without the faster kernels, so a slow run cannot be filed as a result",
    )
    parser.add_argument("--update", type=Path, help="merge the measured keys into this result file")
    parser.add_argument("--against", type=Path, help="the result file to compare with")
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--json", action="store_true", help="print the measurement as JSON instead of a table")
    args = parser.parse_args(argv)

    if args.command == "compare" and args.against is None:
        parser.error("compare needs --against FILE")

    measured = run_sweep(args)
    if args.json:
        print(json.dumps(measured, indent=2, ensure_ascii=False))
    else:
        print_table(measured)
    if args.update:
        merge_into(args.update, measured)
    if args.command == "compare":
        return compare(measured, args.against, args.tolerance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
