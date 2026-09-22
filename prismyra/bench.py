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


def measure_widths(model: str, context: str, widths: tuple[int, ...], repeats: int = 5, pad_to: int = 0) -> list[dict]:
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
        cuda = torch.cuda.is_available() and engine.torch_device.type == "cuda"
        try:
            asked = pad_context(engine, context, pad_to)
            questions = build_questions(width)
            samples = []
            for _ in range(repeats + 2):
                started = time.perf_counter()
                result = engine.ask(asked, questions)
                samples.append((time.perf_counter() - started) * 1e3)
            # The peaks come from one extra pass rather than from the timed ones, because resetting the allocator's
            # statistics inside a timed region synchronises the device and would put a stall in the number.
            #
            # Inside a function because a `Context` holds its engine. A name still bound to a closed context after the
            # `with` block keeps the whole 35 GiB of weights alive through it, and the next width then cannot load --
            # which is how this was found, as an out-of-memory error at the second width rather than the first.
            context_peak, held, branch_peak = _measure_peaks(engine, cuda, asked, questions)
            kept = samples[2:]
            out.append(
                {
                    "width": width,
                    "context_tokens": len(engine.tokenizer(asked)["input_ids"]),
                    "total_ms": round(statistics.median(kept), 1),
                    "context_ms": round(result.timing.context_ms, 1),
                    "branch_ms": round(result.timing.readout_ms, 1),
                    "branch_ms_per_question": round(result.timing.readout_ms / width, 2),
                    "held_gib": round(held / 1024**3, 3),
                    "context_peak_gib": round(context_peak / 1024**3, 3),
                    "branch_peak_gib": round(branch_peak / 1024**3, 3),
                }
            )
        finally:
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out


#: One context and the questions both storage layouts are asked, fixed so two runs ask exactly the same thing.
STORAGE_QUESTIONS = (
    ("thirty", "boolean", "Is there a thirty day limit?"),
    ("unopened", "boolean", "Are unopened items refunded in full?"),
    ("who_pays", "choice", "Who pays return shipping on a faulty item?"),
)

STORAGE_CONTEXT = (
    "Returns are accepted within thirty days of delivery. Unopened items are refunded in full. Opened items are "
    "exchanged rather than refunded, unless a manufacturing fault is confirmed. Return shipping is paid by the seller "
    "when the item is faulty and by the buyer otherwise."
)


def measure_admission(engine, context: str, narrow: int = 1, wide: int = 32) -> dict:
    """Whether the budget an engine forms from a narrow pass survives a wide one at the same context.

    The sequence a reviewer asked for, because it is the one that falsifies the claim: a fresh engine answers `narrow`
    questions, the estimate it then makes for `wide` is recorded, and `wide` is answered and measured. Per-row cost is
    not flat in width -- 0.111 GiB per row at width 1 against 0.155 at width 8 and above, at 24,327 context tokens --
    so an estimate formed at width 1 is expected to fall short, and the engine says so rather than claiming a budget.
    """
    import torch

    from . import Boolean

    asked = [Boolean(id=f"q{i}", prompt=f"Is clause {i} about returns?") for i in range(wide)]
    tokens = len(engine.tokenizer(context)["input_ids"])
    cuda = torch.cuda.is_available() and engine.torch_device.type == "cuda"

    _, _, narrow_peak = _measure_peaks(engine, cuda, context, asked[:narrow])
    predicted = engine.answering_bytes(tokens, questions=wide)
    evidenced = engine.budget_is_evidenced(questions=wide)
    _, _, wide_peak = _measure_peaks(engine, cuda, context, asked)

    return {
        "context_tokens": tokens,
        "narrow_rows": narrow,
        "wide_rows": wide,
        "narrow_peak_gib": round(narrow_peak / 1024**3, 3),
        "predicted_wide_gib": round(predicted / 1024**3, 3),
        "actual_wide_gib": round(wide_peak / 1024**3, 3),
        "claimed_as_evidenced": evidenced,
        "shortfall": round(wide_peak / predicted, 3) if predicted else None,
    }


def measure_ceiling(engine, lengths: tuple[int, ...]) -> list[dict]:
    """The longest context this card can read, and whether the refusal arrives as a diagnosis or as an allocator error.

    Reading is the larger of the two transients and no caller can shrink it, so there is a length beyond which a context
    cannot be opened at all -- measured at somewhere between 24,327 and 48,655 tokens on a 44 GiB card with these
    weights. Worth finding by name: the first context on a fresh engine is unbudgeted, because the figure admission uses
    is observed rather than derived, so what this shows is the budget taking effect after one read has been seen.
    """
    from .schema import PrismyraError

    out = []
    for length in lengths:
        context = pad_context(engine, STORAGE_CONTEXT, length)
        tokens = len(engine.tokenizer(context)["input_ids"])
        predicted = engine.reading_bytes(tokens)
        try:
            with engine.open_context(context):
                outcome, detail = "read", ""
        except PrismyraError as e:
            # Which of the two refused it is the point of the measurement, so the message is classified rather than
            # printed: "ran out of memory" is the allocator having the last word, anything else is admission.
            outcome = "allocator" if "ran out of memory" in str(e) else "refused"
            detail = str(e)
        out.append(
            {
                "context_tokens": tokens,
                "predicted_reading_gib": round(predicted / 1024**3, 3),
                "observed_per_token": engine.stats()["reading_bytes_per_context_token"],
                "outcome": outcome,
                "detail": detail,
            }
        )
        if outcome != "read":
            break
    return out


def _measure_peaks(engine, cuda: bool, context: str, questions) -> tuple[int, int, int]:
    """The largest transient of each phase, and what the open context holds between them."""
    baseline = _settle(engine, cuda)
    with engine.open_context(context) as opened:
        context_peak = _peak_since(engine, cuda, baseline)
        held = _allocated(engine, cuda) - baseline
        _reset_peak(engine, cuda)
        opened.ask(questions)
        return context_peak, held, _peak_since(engine, cuda, baseline + held)


def pad_context(engine, context: str, pad_to: int) -> str:
    """Repeat a context until it reaches about `pad_to` tokens. Repetition rather than prose because what is being
    varied is the length, and generating different text would vary the difficulty with it."""
    if not pad_to:
        return context
    while len(engine.tokenizer(context)["input_ids"]) < pad_to:
        context = context + " " + context
    return context


def storage_answers(engine, pad_to: int = 0) -> dict:
    """The answers one storage layout gives, recorded so another process can be compared against them.

    Two processes rather than two engines: two copies of these weights do not fit on one card, and releasing the first
    does not reliably return its memory. So one run records and the next compares, which is also why this is a
    benchmark rather than a test.

    Asked twice -- once within one group, once with enough filler to span two -- because a single group cannot see a
    cursor that was not reset between them.
    """
    import torch

    from . import Boolean, Choice, Question

    asked: list[Question] = []
    for name, kind, prompt in STORAGE_QUESTIONS:
        if kind == "boolean":
            asked.append(Boolean(id=name, prompt=prompt))
        else:
            asked.append(Choice(id=name, prompt=prompt, choices=["seller", "buyer"]))
    padding = [Boolean(id=f"pad{i}", prompt=f"Is clause {i} about opening hours?") for i in range(31)]

    # The join this path removes copies the context once per branch, so its cost grows with the context. A short
    # context is therefore the case where paging has least to win, and `pad_to` is how the long case gets measured.
    context = pad_context(engine, STORAGE_CONTEXT, pad_to)

    out = {
        "storage": engine.stats()["storage"],
        "context_tokens": len(engine.tokenizer(context)["input_ids"]),
        "groups": {},
    }
    cuda = torch.cuda.is_available() and engine.torch_device.type == "cuda"
    if cuda:
        torch.cuda.empty_cache()
    for label, questions in (("one_group", asked), ("two_groups", [*padding, *asked])):
        # The join this path removes is transient: allocated inside a layer and freed before the next. So it does not
        # show up in what an idle open context holds, and the only place it can be seen is the peak while answering.
        # That peak is what decides how many contexts can be answered at once, which is the claim paging is making.
        # Measured per phase, because a whole-`ask` peak cannot answer this question: reading the context is by far the
        # larger transient and it is identical under both layouts, so it hides whatever the join costs underneath it.
        # The branch phase on its own is where a join appears or does not.
        started = time.perf_counter()
        result = engine.ask(context, questions)
        wall = (time.perf_counter() - started) * 1e3
        context_peak, held, branch_peak = _measure_peaks(engine, cuda, context, questions)
        out["groups"][label] = {
            "wall_ms": round(wall, 1),
            "context_peak_gib": round(context_peak / 1024**3, 3),
            "branch_peak_gib": round(branch_peak / 1024**3, 3),
            "held_gib": round(held / 1024**3, 3),
            "context_ms": round(result.timing.context_ms, 1),
            "readout_ms": round(result.timing.readout_ms, 1),
            "answers": {
                a.id: {"option": a.option, "probabilities": {k: round(v, 6) for k, v in a.probabilities.items()}}
                for a in result.values()
                if not a.id.startswith("pad")
            },
        }
    return out


def _settle(engine, cuda: bool) -> int:
    """Bytes allocated once outstanding device work has finished. Without the synchronise the number read here belongs
    to whatever point the queue had reached, which is not a point in this program."""
    if not cuda:
        return 0
    import torch

    torch.cuda.synchronize(engine.torch_device)
    torch.cuda.reset_peak_memory_stats(engine.torch_device)
    return int(torch.cuda.memory_allocated(engine.torch_device))


def _allocated(engine, cuda: bool) -> int:
    if not cuda:
        return 0
    import torch

    torch.cuda.synchronize(engine.torch_device)
    return int(torch.cuda.memory_allocated(engine.torch_device))


def _reset_peak(engine, cuda: bool) -> None:
    if not cuda:
        return
    import torch

    torch.cuda.synchronize(engine.torch_device)
    torch.cuda.reset_peak_memory_stats(engine.torch_device)


def _peak_since(engine, cuda: bool, baseline: int) -> int:
    """The largest transient above a baseline. Reported rather than the absolute peak because the absolute number is
    mostly weights, and weights are the same under every layout."""
    if not cuda:
        return 0
    import torch

    torch.cuda.synchronize(engine.torch_device)
    return max(0, int(torch.cuda.max_memory_allocated(engine.torch_device)) - baseline)


def compare_storage(recorded: dict, measured: dict, tolerance: float = 2e-2) -> int:
    """Whether two layouts agree. On decisions and on probabilities within a tolerance, not on bits.

    A paged read reduces in a different order, and in a forty-layer mixture of experts a rounding step in a routing
    logit picks a different expert. Bit-identity is not available, so demanding it would reject a correct
    implementation -- and a decision that changes is the thing that actually matters to a caller.
    """
    print(f"\n{recorded['storage']} against {measured['storage']}, {measured['context_tokens']} context tokens")
    if recorded.get("context_tokens") != measured.get("context_tokens"):
        print(
            f"  [warning] the recording used {recorded.get('context_tokens')} tokens and this run used "
            f"{measured.get('context_tokens')}; the timings are not comparable"
        )
    problems = 0
    for label in recorded["groups"]:
        want, got = recorded["groups"][label], measured["groups"][label]
        print(
            f"\n{label}: {want['wall_ms']:.0f} ms -> {got['wall_ms']:.0f} ms "
            f"({want['readout_ms']:.0f} ms answering -> {got['readout_ms']:.0f} ms)"
        )
        print(
            f"  held {want.get('held_gib', 0):.3f} -> {got.get('held_gib', 0):.3f} GiB, "
            f"branch peak {want.get('branch_peak_gib', 0):.3f} -> {got.get('branch_peak_gib', 0):.3f} GiB, "
            f"context peak {want.get('context_peak_gib', 0):.2f} -> {got.get('context_peak_gib', 0):.2f} GiB"
        )
        for name, answer in want["answers"].items():
            mine = got["answers"][name]
            same = answer["option"] == mine["option"]
            moved = max(
                abs(probability - mine["probabilities"][key]) for key, probability in answer["probabilities"].items()
            )
            mark = "OK " if same and moved < tolerance else "NO "
            problems += int(not (same and moved < tolerance))
            print(f"  {mark}{name:<12} {answer['option']:>8} -> {mine['option']:<8} largest move {moved:.4f}")
    if problems:
        print(f"\n{problems} answer(s) disagreed beyond {tolerance}, so the layouts are not interchangeable")
    else:
        print(f"\nevery decision matched and no probability moved by {tolerance} or more")
    return 1 if problems else 0


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


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911 - one return per command reads better than a dispatch
    parser = argparse.ArgumentParser(
        prog="prismyra-bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=["sweep", "compare", "concurrency", "widths", "storage", "admission", "ceiling"],
    )
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
    parser.add_argument(
        "--against",
        type=Path,
        help="the file to compare with: a result file for compare, a recorded answer file for storage",
    )
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument(
        "--callers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="how many arrive together, for the concurrency command",
    )
    parser.add_argument("--questions", type=int, default=8, help="questions per caller, for the concurrency command")
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=0,
        help="lengthen the storage context to about this many tokens; what paging removes grows with the context",
    )
    parser.add_argument("--record", type=Path, help="write this layout's answers here, for the storage command")
    parser.add_argument("--json", action="store_true", help="print the measurement as JSON instead of a table")
    args = parser.parse_args(argv)

    if args.command == "compare" and args.against is None:
        parser.error("compare needs --against FILE")

    if args.command == "storage":
        from . import Prismyra

        engine = Prismyra(args.model, require_kernels=args.require_kernels)
        measured = storage_answers(engine, pad_to=args.context_tokens)
        if args.record:
            args.record.write_text(json.dumps(measured, indent=2))
            print(f"recorded the {measured['storage']} layout to {args.record}")
            return 0
        if args.against:
            return compare_storage(json.loads(args.against.read_text()), measured)
        print(json.dumps(measured, indent=2))
        return 0

    if args.command == "ceiling":
        from . import Prismyra

        engine = Prismyra(args.model, require_kernels=args.require_kernels)
        rows = measure_ceiling(engine, tuple(args.callers))
        print(f"\n{'tokens':>8} {'predicted read':>15} {'bytes/token':>12}  outcome")
        for row in rows:
            print(
                f"{row['context_tokens']:>8} {row['predicted_reading_gib']:>15.3f} "
                f"{row['observed_per_token']:>12}  {row['outcome']}"
            )
        last = rows[-1]
        if last["outcome"] == "refused":
            print(f"\nRefused by name before reading anything:\n  {last['detail']}")
        elif last["outcome"] == "allocator":
            print(
                f"\nThe allocator had the last word, which is what happens when nothing has been read yet and\n"
                f"there is no figure to budget from:\n  {last['detail']}"
            )
        else:
            print("\nEvery length read. The ceiling is above the longest asked for.")
        if args.json:
            print(json.dumps(rows, indent=2))
        return 0

    if args.command == "admission":
        from . import Prismyra

        engine = Prismyra(args.model, require_kernels=args.require_kernels)
        context = pad_context(engine, PARAGRAPH * args.repeat_paragraph, args.context_tokens)
        found = measure_admission(engine, context, wide=engine.group)
        print(f"\n{found['context_tokens']} context tokens, a fresh engine, {found['wide_rows']} rows to budget for")
        print(f"  a pass of {found['narrow_rows']} row(s) peaked at {found['narrow_peak_gib']:.3f} GiB")
        print(f"  the estimate it then made for {found['wide_rows']} rows was {found['predicted_wide_gib']:.3f} GiB")
        print(f"  that pass actually peaked at {found['actual_wide_gib']:.3f} GiB")
        print(f"  the engine claimed this estimate as evidenced: {found['claimed_as_evidenced']}")
        if found["shortfall"]:
            print(
                f"\nThe estimate was short by a factor of {found['shortfall']:.2f}. That is expected and is why the "
                f"engine\nreports it as a lower bound for a pass wider than any it has measured; refusing on it would "
                f"be\nrefusing on a figure nothing supports. The allocator is the gate for that pass."
            )
        if args.json:
            print(json.dumps(found, indent=2))
        return 0

    if args.command == "widths":
        context = PARAGRAPH * args.repeat_paragraph
        rows = measure_widths(args.model, context, tuple(args.callers), pad_to=args.context_tokens)
        print(
            f"\n{'width':>6} {'tokens':>7} {'total ms':>9} {'branch ms':>10} {'ms/question':>12} "
            f"{'held GiB':>9} {'ctx peak':>9} {'branch peak':>12}"
        )
        for row in rows:
            print(
                f"{row['width']:>6} {row['context_tokens']:>7} {row['total_ms']:>9.1f} "
                f"{row['branch_ms']:>10.1f} {row['branch_ms_per_question']:>12.2f} "
                f"{row['held_gib']:>9.3f} {row['context_peak_gib']:>9.3f} {row['branch_peak_gib']:>12.3f}"
            )
        # The peak is the number that decides how many callers fit, and it is not the held cache. Printed beside the
        # clock so the two are read together: the clock says a narrow batch reclaims nothing, and the peak says
        # whether it reclaims memory.
        first, last = rows[0], rows[-1]
        if last["branch_peak_gib"]:
            print(
                f"\nThe branch pass peaked at {first['branch_peak_gib']:.3f} GiB at width {first['width']} and "
                f"{last['branch_peak_gib']:.3f} GiB at width {last['width']}, against {last['held_gib']:.3f} GiB held."
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
