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


def measure_concurrency(engine, context: str, callers: tuple[int, ...], questions_each: int) -> list[dict]:
    """What arriving together costs, now that one worker owns the device.

    Worth measuring rather than assuming, because the answer decides whether multi-user work is worth doing. One
    device serves one request at a time whatever the arrival pattern, so concurrency cannot raise throughput here --
    what it can do is decide who waits. This reports the queueing delay separately from the service time, because
    those have different fixes: waiting is answered by another device, working only by kernels.
    """
    import statistics
    import threading
    import time

    from .queue import Worker

    out = []
    for count in callers:
        worker = Worker(lambda payload: engine.ask(payload[0], payload[1])).start()
        try:
            questions = build_questions(questions_each)
            jobs: list = []
            errors: list = []

            def call(worker=worker, questions=questions, jobs=jobs, errors=errors) -> None:
                # Bound as defaults rather than closed over: a closure here would read whichever loop iteration's
                # values happened to be current when the thread ran, which is a different measurement each time.
                try:
                    jobs.append(worker.submit((context, questions)))
                except Exception as e:  # noqa: BLE001 - a refused caller is a result, not a crash
                    errors.append(e)

            started = time.perf_counter()
            threads = [threading.Thread(target=call) for _ in range(count)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            elapsed = time.perf_counter() - started

            latencies = sorted((job.queue_ms + job.service_ms) for job in jobs)
            out.append(
                {
                    "callers": count,
                    "answered": len(jobs),
                    "refused": len(errors),
                    "wall_s": round(elapsed, 2),
                    "requests_per_second": round(len(jobs) / elapsed, 2),
                    "first_home_ms": round(latencies[0], 1) if latencies else None,
                    "median_ms": round(statistics.median(latencies), 1) if latencies else None,
                    "last_home_ms": round(latencies[-1], 1) if latencies else None,
                    "median_queue_ms": round(statistics.median(job.queue_ms for job in jobs), 1) if jobs else None,
                    "median_service_ms": round(statistics.median(job.service_ms for job in jobs), 1) if jobs else None,
                }
            )
        finally:
            worker.stop()
    return out


def measure_open_contexts(engine, context: str, limit: int = 64) -> dict:
    """How many contexts can be open at once, which is the memory ceiling as it stands.

    Each open context holds `group` physical copies of its keys and values, so this is the number paged single-copy
    storage would change. Measured by opening them until the engine refuses, which it does by name rather than by
    letting the allocator fail.
    """
    import torch

    from .schema import PrismyraError

    opened, refused = [], None
    try:
        for _ in range(limit):
            opened.append(engine.open_context(context))
    except PrismyraError as e:
        refused = str(e)
    except torch.OutOfMemoryError as e:  # pragma: no cover - the case the refusal exists to prevent
        refused = f"the allocator refused first, which the engine should have: {e}"
    finally:
        held = len(opened)
        for handle in opened:
            handle.close()

    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "open_at_once": held,
        "each_holds_gib": round(engine.cache_bytes(len(engine.tokenizer(context)["input_ids"])) / 1024**3, 2),
        "refused_with": refused,
    }


def measure_widths(model: str, context: str, widths: tuple[int, ...], repeats: int = 5) -> list[dict]:
    """What a branch pass costs at each batch width, which decides whether a narrow one is worth arranging.

    The premise of the whole design is that a traversal costs about the same whatever it carries, because the cost is
    reading the experts' weights rather than the rows' tokens. If that is true, a one-question request pays for
    thirty-two rows and there is nothing to reclaim. If it is not true, the engine is doing thirty-two rows of work
    for one answer and the smallest useful fix in the package is to stop.

    One engine at a time, rebuilt per width, because the cache is preallocated for the group and two copies of these
    weights do not fit on one card. That makes this slow and unavoidable.
    """
    import gc
    import statistics
    import time

    import torch

    from . import Prismyra

    out = []
    for width in widths:
        engine = Prismyra(model, group=width)
        try:
            questions = build_questions(width)
            samples = []
            for _ in range(repeats + 2):
                started = time.perf_counter()
                result = engine.ask(context, questions)
                samples.append((time.perf_counter() - started) * 1e3)
            kept = samples[2:]
            out.append(
                {
                    "width": width,
                    "total_ms": round(statistics.median(kept), 1),
                    "context_ms": round(result.timing.context_ms, 1),
                    "branch_ms": round(result.timing.readout_ms, 1),
                    "branch_ms_per_question": round(result.timing.readout_ms / width, 2),
                }
            )
        finally:
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out


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


def _print_concurrency(measured: dict) -> None:
    contexts = measured["open_contexts"]
    print(
        f"\n{measured['model']} on {measured['machine'].get('gpu', 'cpu')}, "
        f"{measured['context_tokens']} context tokens, group {measured['group']}\n"
    )
    print(f"contexts open at once: {contexts['open_at_once']}, each holding about {contexts['each_holds_gib']} GiB")
    if contexts["refused_with"]:
        print(f"  refused with: {contexts['refused_with'][:150]}")

    print(
        f"\n{'callers':>8} {'answered':>9} {'req/s':>7} {'first home':>11} {'median':>9} {'last home':>10} "
        f"{'median queue':>13} {'median service':>15}"
    )
    for row in measured["concurrency"]:
        print(
            f"{row['callers']:>8} {row['answered']:>9} {row['requests_per_second']:>7.2f} "
            f"{row['first_home_ms']:>11.0f} {row['median_ms']:>9.0f} {row['last_home_ms']:>10.0f} "
            f"{row['median_queue_ms']:>13.0f} {row['median_service_ms']:>15.0f}"
        )
    print(
        "\nOne device serves one request at a time, so concurrency cannot raise throughput here. What it decides is\n"
        "who waits: queueing delay is answered by another device, service time only by kernels."
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
    parser.add_argument("command", choices=["sweep", "compare", "concurrency", "widths"])
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
    parser.add_argument(
        "--callers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="how many arrive together, for the concurrency command",
    )
    parser.add_argument("--questions", type=int, default=8, help="questions per caller, for the concurrency command")
    parser.add_argument("--json", action="store_true", help="print the measurement as JSON instead of a table")
    args = parser.parse_args(argv)

    if args.command == "compare" and args.against is None:
        parser.error("compare needs --against FILE")

    if args.command == "widths":
        context = PARAGRAPH * args.repeat_paragraph
        rows = measure_widths(args.model, context, tuple(args.callers))
        print(f"\n{'width':>6} {'total ms':>9} {'context ms':>11} {'branch ms':>10} {'branch ms/question':>19}")
        for row in rows:
            print(
                f"{row['width']:>6} {row['total_ms']:>9.1f} {row['context_ms']:>11.1f} "
                f"{row['branch_ms']:>10.1f} {row['branch_ms_per_question']:>19.2f}"
            )
        narrow, wide = rows[0]["branch_ms"], rows[-1]["branch_ms"]
        print(
            f"\nA branch pass at width {rows[0]['width']} costs {narrow:.0f} ms and at width {rows[-1]['width']} "
            f"{wide:.0f} ms, a factor of {wide / narrow:.2f}."
        )
        print(
            "Near 1.00 means a traversal costs the same whatever it carries, so a narrow batch reclaims nothing and\n"
            "the design's premise holds. Much above 1.00 means a one-question request is paying for rows it does not\n"
            "have, and the engine should use the narrowest batch that fits."
        )
        if args.json:
            print(json.dumps(rows, indent=2))
        return 0

    if args.command == "concurrency":
        from . import Prismyra

        engine = Prismyra(args.model, require_kernels=args.require_kernels)
        context = PARAGRAPH * args.repeat_paragraph
        measured = {
            **environment(),
            "model": args.model,
            "context_tokens": len(engine.tokenizer(context)["input_ids"]),
            "group": engine.group,
            "open_contexts": measure_open_contexts(engine, context),
            "concurrency": measure_concurrency(engine, context, tuple(args.callers), args.questions),
        }
        _print_concurrency(measured)
        if args.json:
            args.json.write_text(json.dumps(measured, indent=2, ensure_ascii=False))
            print(f"\nwritten to {args.json}")
        return 0

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
