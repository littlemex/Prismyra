#!/usr/bin/env python3
"""Accuracy and latency in one table, because either alone is misleading.

    python evals/run.py --task race --limit 40
    python evals/run.py --task boolq --limit 200 --methods readout,generate,majority
    python evals/run.py --task unfair_tos --limit 100 --json results.json

Three ways of answering the same questions:

* **readout** -- what this package does. One traversal of the context, then one row per question, scoring the declared
  options at the branch's final position.
* **generate** -- the same weights and the same prompt, generating the answer and reading the text, once per question.
  This is the thing the speed claims are claims against.
* **majority** -- the most common label in the slice being scored. Not a strawman: on a sparse task it is hard to beat,
  and a method that does not beat it has not been shown to work.

A method's accuracy is reported beside its latency always, and never without it. A wrong answer can be made
arbitrarily fast, so a speed-up without an accuracy column is not a result.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tasks
from generate import answer_by_generating, parse


def run_readout(engine, items) -> tuple[list[dict], dict]:
    """Prismyra's own read-out. One context pass per item, whatever the question count."""
    rows, timings = [], {"context_ms": [], "readout_ms": []}
    for item in items:
        result = engine.ask(item.context, item.questions)
        timings["context_ms"].append(result.timing.context_ms)
        timings["readout_ms"].append(result.timing.readout_ms)
        for question in item.questions:
            answer = result[question.id]
            rows.append(
                {
                    "id": question.id,
                    "got": answer.value,
                    "want": item.gold[question.id],
                    "confidence": max(answer.probabilities.values()),
                    "questions_in_item": len(item.questions),
                }
            )
    return rows, timings


def run_generate(engine, items, reasoning: bool = False) -> tuple[list[dict], dict]:
    rows, timings = [], {"total_ms": [], "how": collections.Counter(), "hit_budget": 0}
    for item in items:
        texts, elapsed, hit = answer_by_generating(engine, item.context, item.questions, reasoning=reasoning)
        timings["total_ms"].append(elapsed)
        timings["hit_budget"] += hit
        for question in item.questions:
            value, how = parse(texts[question.id], question)
            timings["how"][how] += 1
            rows.append(
                {
                    "id": question.id,
                    "got": value,
                    "want": item.gold[question.id],
                    "how": how,
                    "text": texts[question.id][:60],
                    "questions_in_item": len(item.questions),
                }
            )
    return rows, timings


def run_thresholded(engine, items, fitted_on) -> tuple[list[dict], dict]:
    """The read-out with the decision point chosen per question, through the package's own mechanism.

    Calls `prismyra.thresholds` rather than reimplementing it. An earlier version of this function fitted its own cuts
    with different guards and a different idea of which option means yes, which meant the number it produced was not
    the number the package would produce -- the most serious thing a measurement can get wrong.

    It targets a measured failure rather than a supposed one. On the unfair terms-of-service task the labels are so
    sparse that answering no to everything scores 98.5%, and the read-out's argmax reached 60% recall at 6% precision:
    it says yes to nearly everything, because taking the larger of two probabilities stands the threshold at 0.5.
    """
    from prismyra.thresholds import Thresholds

    history = [(engine.ask(item.context, item.questions), item.gold) for item in fitted_on]
    cuts, report = Thresholds.fit(history)
    print(report.explain())
    print()

    rows = []
    for item in items:
        decided = cuts.decide(engine.ask(item.context, item.questions))
        for question in item.questions:
            rows.append(
                {
                    "id": question.id,
                    "got": decided[question.id].value,
                    "want": item.gold[question.id],
                    "questions_in_item": len(item.questions),
                }
            )
    return rows, {"cuts": cuts.cuts, "support": {k: list(v) for k, v in cuts.support.items()}}


def run_majority(items, fitted_on) -> tuple[list[dict], dict]:
    """Always answer the commonest label. Which label is read off a **different** slice.

    Fitting the constant on the slice being scored would make this an oracle rather than a baseline: it would be told
    the answer distribution it is about to be graded on. The label comes from the training split instead, which is
    what a method that had to be deployed could actually know.
    """
    counts = collections.Counter(str(item.gold[q.id]) for item in fitted_on for q in item.questions)
    top = counts.most_common(1)[0][0]
    rows = [
        {"id": q.id, "got": top, "want": str(item.gold[q.id]), "questions_in_item": len(item.questions)}
        for item in items
        for q in item.questions
    ]
    return rows, {"label": top, "fitted_on_contexts": len(fitted_on)}


def paired(a: list[dict], b: list[dict], items, draws: int = 2000, seed: int = 0) -> dict:
    """Whether one method beats another, resampling whole contexts rather than questions.

    Questions from one article share a passage, a topic and whatever the model does or does not understand about it,
    so they are not independent draws. Resampling questions would give an interval several times too narrow and turn a
    one-question difference into a result. Contexts are the unit that was sampled, so contexts are what is resampled.
    """
    import random

    per_context_a, per_context_b = _by_context(a, items), _by_context(b, items)
    observed = _accuracy(a) - _accuracy(b)

    rng = random.Random(seed)
    indices = range(len(items))
    wins = 0
    differences = []
    for _ in range(draws):
        picked = [rng.choice(indices) for _ in indices]
        got_a = sum(per_context_a[i][0] for i in picked) / max(1, sum(per_context_a[i][1] for i in picked))
        got_b = sum(per_context_b[i][0] for i in picked) / max(1, sum(per_context_b[i][1] for i in picked))
        differences.append(got_a - got_b)
        wins += int(got_a > got_b)
    differences.sort()
    return {
        "difference": observed,
        "low": differences[int(0.025 * draws)],
        "high": differences[int(0.975 * draws)],
        "wins": wins / draws,
    }


def _by_context(rows: list[dict], items) -> list[tuple[int, int]]:
    """(correct, total) per context, in the order the contexts were scored."""
    out, cursor = [], 0
    for item in items:
        take = rows[cursor : cursor + len(item.questions)]
        cursor += len(item.questions)
        out.append((sum(1 for r in take if str(r["got"]).lower() == str(r["want"]).lower()), len(take)))
    return out


def _accuracy(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if str(r["got"]).lower() == str(r["want"]).lower()) / len(rows)


def score(rows) -> dict:
    """Accuracy, and the numbers that matter when accuracy does not.

    On a sparse task accuracy is not a measure of anything. The unfair terms-of-service labels are mostly absent, so
    answering "no" to everything scores 99.2% and beats every method here -- measured, not hypothesised. What
    separates methods there is whether the rare positive is found at all, so the positives are scored separately: how
    many were found, how many claimed positives were right, and the balance of the two.
    """
    correct = sum(1 for row in rows if str(row["got"]).lower() == str(row["want"]).lower())
    answered = sum(1 for row in rows if row["got"] is not None)

    positive = [row for row in rows if _is_positive(row["want"])]
    claimed = [row for row in rows if _is_positive(row["got"])]
    hits = [row for row in claimed if _is_positive(row["want"])]
    recall = len(hits) / len(positive) if positive else 0.0
    precision = len(hits) / len(claimed) if claimed else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "questions": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "answered": answered,
        "accuracy_when_answered": correct / answered if answered else 0.0,
        "positives": len(positive),
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def _is_positive(value) -> bool:
    """Whether an answer is the rare, interesting class. Only meaningful on a two-option task; harmless elsewhere."""
    return str(value).lower() in ("true", "yes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prismyra-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", default="race", choices=["boolq", "race", "unfair_tos"])
    parser.add_argument("--limit", type=int, default=40, help="contexts, not questions")
    parser.add_argument(
        "--methods",
        default="readout,calibrated_options,generate,majority",
        help="readout, thresholded, calibrated_options, calibrated_context, generate, generate_thinking, majority",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    parser.add_argument("--seed", type=int, default=0, help="which random sample of contexts to score")
    parser.add_argument(
        "--split",
        default="validation",
        help="the split to score on. Use test once, at the end: a validation slice that has shaped a mechanism has "
        "become development data, and scoring on it again measures the shaping as well as the mechanism",
    )
    parser.add_argument("--fit-split", default="train", help="the split anything fitted is fitted on")
    parser.add_argument("--fit-limit", type=int, default=400, help="contexts to fit on, which is the label budget")
    parser.add_argument("--repeat", type=int, default=1, help="whole runs, so the latency spread is visible")
    parser.add_argument("--json", type=Path, help="write the whole measurement here")
    args = parser.parse_args(argv)

    wanted = [name.strip() for name in args.methods.split(",") if name.strip()]
    items = tasks.load(args.task, args.limit, split=args.split, seed=args.seed)
    # A separate slice for anything that has to be fitted, so nothing is fitted on what it is graded on. Its size is a
    # separate dial because it is the interesting one: at a 1.5% base rate, a hundred contexts hold one or two
    # positives per question, and a threshold fitted on two positives is a threshold fitted on noise. How many labels
    # a mechanism needs before it pays is the operational question, so it is a parameter rather than half the scored
    # slice.
    fit_slice = tasks.load(args.task, args.fit_limit, split=args.fit_split, seed=args.seed + 1)
    questions = sum(len(item.questions) for item in items)
    print(
        f"{args.task} [{args.split}, seed {args.seed}]: {len(items)} contexts, {questions} questions "
        f"({questions / len(items):.1f} per context), anything fitted uses {args.fit_split}\n"
    )

    engine = None
    needs_model = {
        "readout",
        "thresholded",
        "calibrated_options",
        "calibrated_context",
        "generate",
        "generate_thinking",
    }
    if needs_model & set(wanted):
        from prismyra import Prismyra

        started = time.perf_counter()
        engine = Prismyra(args.model)
        print(f"[load] {time.perf_counter() - started:.0f}s  kernels: {engine.applied.as_dict()['applied']}\n")

    results = {}
    for name in wanted:
        began = time.perf_counter()
        if name == "readout":
            rows, timings = run_readout(engine, items)
        elif name in ("calibrated_options", "calibrated_context"):
            # The same engine with the correction switched on, so nothing but the read-out differs. A second engine
            # would change the weights' placement as well and leave the difference unattributable.
            from prismyra.calibration import Calibration

            mode = "options_only" if name.endswith("options") else "null_context"
            engine.calibration = Calibration(mode=mode)
            try:
                rows, timings = run_readout(engine, items)
            finally:
                engine.calibration = None
        elif name == "generate":
            rows, timings = run_generate(engine, items, reasoning=False)
        elif name == "generate_thinking":
            rows, timings = run_generate(engine, items, reasoning=True)
        elif name == "thresholded":
            rows, timings = run_thresholded(engine, items, fitted_on=fit_slice)
        elif name == "majority":
            rows, timings = run_majority(items, fitted_on=fit_slice)
        else:
            raise ValueError(f"unknown method {name!r}")
        results[name] = {
            **score(rows),
            "timings": timings,
            "wall_s": time.perf_counter() - began,
            "rows": rows,
        }

    _print_table(args.task, items, results)
    if args.json:
        args.json.write_text(
            json.dumps({"task": args.task, "contexts": len(items), "results": results}, indent=2, default=str)
        )
        print(f"\nwritten to {args.json}")
    return 0


def _print_per_question(items, results: dict) -> None:
    """Per question id, and macro F1 beside the pooled figure.

    Eight thresholds were fitted, so eight results were produced, and a pooled score can be carried by two of them
    while six do nothing. Macro F1 is also what the literature on this task reports, so it is the number a reader will
    want to compare against.
    """
    ids = list(dict.fromkeys(q.id for item in items for q in item.questions))
    if len(ids) < 2 or len(ids) > 12:
        return

    print("\nper question, F1 (positives in the scored slice):")
    header = f"  {'question':<26}" + "".join(f"{name:>22}" for name in results)
    print(header)
    macro = {name: [] for name in results}
    for question_id in ids:
        positives = sum(
            1
            for name, result in results.items()
            for row in result["rows"]
            if row["id"] == question_id and _is_positive(row["want"]) and name == next(iter(results))
        )
        cells = ""
        for name, result in results.items():
            rows = [row for row in result["rows"] if row["id"] == question_id]
            f1 = score(rows)["f1"]
            macro[name].append(f1)
            cells += f"{f1:>21.1%} "
        print(f"  {question_id:<26}{cells} ({positives} positive)")
    print(f"  {'macro average':<26}" + "".join(f"{sum(v) / len(v):>21.1%} " for v in macro.values()))

    first = next(iter(results.values()))
    print(
        f"\n  {first['positives']} positive labels in {first['questions']} answers. At that count one label moves "
        f"recall by about {1 / max(1, first['positives']):.1%}, so read these as preliminary."
    )


def _sparse(results: dict, threshold: float = 0.2) -> float | None:
    """The share of positive labels, when it is small enough that accuracy is the wrong summary."""
    any_result = next(iter(results.values()), None)
    if not any_result or not any_result["questions"]:
        return None
    share = any_result["positives"] / any_result["questions"]
    return share if 0 < share < threshold else None


def _print_table(task: str, items, results: dict) -> None:
    questions = sum(len(item.questions) for item in items)
    sparse = _sparse(results)
    # The positive-class columns only exist where there is a positive class. On a lettered multiple choice there is
    # none, and printing three zeroes per row invites the reader to compare them.
    two_class = any(result["positives"] for result in results.values())
    header = f"{'method':<20} {'accuracy':>9} {'answered':>10}"
    if two_class:
        header += f" {'recall':>8} {'precision':>10} {'F1':>7}"
    print(header + f" {'ms/question':>12}")
    print("-" * (len(header) + 13))
    for name, result in results.items():
        per_question = result["wall_s"] * 1e3 / questions
        answered = f"{result['answered']}/{result['questions']}"
        latency = "       <0.05" if per_question < 0.05 else f"{per_question:>12.1f}"
        row = f"{name:<20} {result['accuracy']:>8.1%} {answered:>10}"
        if two_class:
            row += f" {result['recall']:>7.1%} {result['precision']:>9.1%} {result['f1']:>6.1%}"
        print(row + latency)

    if sparse:
        _print_per_question(items, results)
        share = sparse
        print(
            f"\n[read this table by F1, not accuracy] {share:.1%} of the answers are positive, so answering no to "
            f"everything scores {1 - share:.1%}. Accuracy does not separate the methods on this task."
        )

    if "readout" in results:
        timings = results["readout"]["timings"]
        print(
            f"\nread-out, per context: {statistics.median(timings['context_ms']):.0f} ms reading + "
            f"{statistics.median(timings['readout_ms']):.0f} ms answering (medians)"
        )
    for name in ("generate", "generate_thinking"):
        if name not in results:
            continue
        how = results[name]["timings"]["how"]
        parts = ", ".join(f"{label} {count}" for label, count in sorted(how.items(), key=lambda kv: -kv[1]))
        print(f"{name}, how each answer was read: {parts}")
        print(
            "  exact and prefix are the answer as given; mentioned needed a whole-word search; "
            "empty, absent and ambiguous scored as unanswered"
        )
    for name in ("generate", "generate_thinking"):
        if name in results and results[name]["timings"].get("hit_budget"):
            hit = results[name]["timings"]["hit_budget"]
            print(
                f"\n[warning] {hit} of {questions} {name} answers ran to the end of their token budget, so this "
                f"accuracy is confounded by the budget rather than measured by it"
            )

    for other in ("thresholded", "calibrated_options", "calibrated_context", "generate", "generate_thinking"):
        if "readout" not in results or other not in results:
            continue
        test = paired(results["readout"]["rows"], results[other]["rows"], items)
        print(
            f"\nread-out minus {other}: {test['difference']:+.1%} accuracy, "
            f"95% interval [{test['low']:+.1%}, {test['high']:+.1%}] over resampled contexts"
        )
        if test["low"] <= 0 <= test["high"]:
            print("  the interval spans zero, so this slice does not separate them")
        if other.startswith("generate"):
            ratio = results[other]["wall_s"] / results["readout"]["wall_s"]
            print(
                f"  {ratio:.1f}x the speed, but against an unoptimised decode loop with no batching and no prefix "
                f"sharing, so that is an upper bound on the advantage over generation rather than a floor"
            )


if __name__ == "__main__":
    sys.exit(main())
