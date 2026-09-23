"""The comparison `generate.py` said somebody would have to run: a real serving engine, batching and sharing the prefix.

`generate.py` compares the read-out against a plain decode loop and says in its own docstring that the speed advantage
it measures is an upper bound, because "a serving engine can batch the questions and share the article's prefix without
using this read-out at all". This is that engine. vLLM, the same weights, the same questions, the same card, with prefix
caching on -- which is how a caller gets one context serving many questions **without** a forked cache.

Three arms, and the middle one is the one that matters:

* **fork** -- this package. The context is written once into device memory and every question is one row of one batch.
* **separate** -- vLLM, one request per question, each carrying the whole context. The prefix cache means the context's
  keys and values are computed once and reused, so this is the same saving arrived at from the other end, and it is what
  `ikermoel/open-alternative-jev` measured as faster *and* more accurate than packing questions into one sequence.
* **packed** -- vLLM, all of an item's questions concatenated into one request. Included because it is the obvious thing
  to try and because its accuracy cost is the reason this package forks instead: in that repository's measurement,
  packing moved 8.0% of individual answers and cost 2.8 points of accuracy.

Two processes, because two copies of these weights do not fit on one card:

    python evals/against_vllm.py --record fork.json --limit 60
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python evals/against_vllm.py --against fork.json --mode separate

**What is and is not being compared.** The arms do not share a read-out. This package scores the option text each
question declared; the vLLM arms score the letters A, B, C, D, because that is what a request returning one token with
logprobs can see. So accuracy differences belong to the read-out as much as to the engine, and are reported without a
claim attached. Questions per second and tokens processed are what this file exists for, and those are the same
work in both arms.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import tasks

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

INSTRUCTION = "Choose the correct option. Reply with only its letter."


def lettered(item, question) -> str:
    """One request's prompt: the instruction, the context, the question, the options as letters.

    Written once and used by both vLLM arms so that the only difference between them is how many questions share a
    request. A different prompt per arm would make their comparison meaningless as well as this file's.
    """
    lines = [INSTRUCTION, "", "Context:", item.context, "", f"Question: {question.prompt}"]
    lines += [f"{LETTERS[i]}. {option}" for i, option in enumerate(question.options)]
    lines += ["", "Answer:"]
    return "\n".join(lines)


def follow_up(question) -> str:
    """A later question in a packed request. The context is not repeated -- that is the whole idea of packing."""
    lines = ["", "Answer the following question about the same context. Reply with only the letter.", ""]
    lines += [f"Question: {question.prompt}"]
    lines += [f"{LETTERS[i]}. {option}" for i, option in enumerate(question.options)]
    lines += ["", "Answer:"]
    return "\n".join(lines)


def letter_tokens(tokenizer, options: int) -> tuple[list[int], dict[int, str]]:
    """The token ids that spell a bare letter, with and without a leading space, and how to read one back.

    Both renderings, because which one the model reaches for depends on what precedes it, and a prompt ending in
    "Answer:" invites the spaced form. Leaving one out silently removes half the candidates.
    """
    allowed: list[int] = []
    back: dict[int, str] = {}
    for letter in LETTERS[:options]:
        for rendering in (letter, f" {letter}"):
            ids = tokenizer(rendering, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                allowed.append(ids[0])
                back[ids[0]] = letter
    if not allowed:
        raise SystemExit("no single token spells a bare letter in this tokenizer; this comparison needs one")
    return allowed, back


def read_letters(output, options: int, back: dict[int, str]) -> tuple[int, list[float]]:
    """Which option a one-token completion chose, from the logprobs over the letters it was restricted to.

    Normalised over those letters, which is what this package's own read-out does over the options a question declared,
    so the two measure the same kind of quantity even though they score different tokens. A letter that still does not
    appear is scored as very unlikely rather than as impossible, and the count of such cases is worth watching: it means
    the restriction did not take.
    """
    import math

    chosen = output.outputs[0]
    top = chosen.logprobs[0] if chosen.logprobs else {}
    by_letter: dict[str, float] = {}
    for token_id, entry in top.items():
        letter = back.get(token_id)
        if letter is None:
            continue
        by_letter[letter] = max(by_letter.get(letter, -math.inf), entry.logprob)
    if not by_letter:
        return 0, [1.0 / options] * options
    scores = [by_letter.get(LETTERS[i], -30.0) for i in range(options)]
    top_score = max(scores)
    weights = [math.exp(s - top_score) for s in scores]
    total = sum(weights)
    probabilities = [w / total for w in weights]
    return max(range(options), key=lambda i: probabilities[i]), probabilities


def run_fork(items, model: str) -> dict:
    """This package. One context pass per item, every question a row."""
    from prismyra import Prismyra

    engine = Prismyra(model)
    tokenizer = engine.tokenizer
    rows, seconds, tokens = [], [], 0
    for n, item in enumerate(items):
        started = time.perf_counter()
        result = engine.ask(item.context, item.questions)
        elapsed = time.perf_counter() - started
        # The first item pays for warm-up in both arms, so it is measured and then dropped by `summarise`.
        seconds.append(elapsed)
        # What the model computed, not what a caller typed: the context once, plus one padded suffix per question.
        context_tokens = len(tokenizer(item.context)["input_ids"])
        suffix = max(len(tokenizer("\n" + q.prompt)["input_ids"]) for q in item.questions)
        tokens += context_tokens + suffix * len(item.questions)
        for question in item.questions:
            answer = result[question.id]
            rows.append(
                {
                    "item": n,
                    "id": question.id,
                    "got": answer.value,
                    "want": item.gold[question.id],
                }
            )
    return {"arm": "fork", "rows": rows, "seconds": seconds, "tokens": tokens, "model": model}


def run_vllm(items, model: str, mode: str, max_len: int, utilisation: float) -> dict:
    """vLLM, with prefix caching on. `separate` sends one request per question, `packed` one per item."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        max_model_len=max_len,
        gpu_memory_utilization=utilisation,
        enforce_eager=True,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    tokenizer = llm.get_tokenizer()
    # The output is restricted to the letters, which is what makes this a fair comparison rather than a workable one.
    # This package's read-out softmaxes over the options a question declared and nothing else; asking vLLM for the top
    # twenty logprobs and hoping the letters are among them measures something weaker -- on this model they often are
    # not, because it opens with a newline and a `<think>`. Restricting the candidates puts both arms on the same
    # footing: score the declared answers, normalise over them.
    allowed, back = letter_tokens(tokenizer, max(len(q.options) for item in items for q in item.questions))
    sampling = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20, allowed_token_ids=allowed)

    rows, seconds, tokens = [], [], 0
    for n, item in enumerate(items):
        if mode == "separate":
            prompts = [lettered(item, q) for q in item.questions]
        else:
            first = lettered(item, item.questions[0])
            prompts = [first + "".join(follow_up(q) for q in item.questions[1:])]

        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        seconds.append(time.perf_counter() - started)
        tokens += sum(len(tokenizer(p)["input_ids"]) for p in prompts)

        if mode == "separate":
            for question, output in zip(item.questions, outputs, strict=True):
                pick, _ = read_letters(output, len(question.options), back)
                rows.append(
                    {
                        "item": n,
                        "id": question.id,
                        "got": question.value_of(question.options[pick]),
                        "want": item.gold[question.id],
                    }
                )
        else:
            # A packed request answers only at its final position, so one request gives one answer however many
            # questions it carries. Answering all of them needs the per-position read-out that request APIs do not
            # expose, which is itself a finding: packing is not available to a caller through a serving engine's API.
            pick, _ = read_letters(outputs[0], len(item.questions[-1].options), back)
            last = item.questions[-1]
            rows.append(
                {
                    "item": n,
                    "id": last.id,
                    "got": last.value_of(last.options[pick]),
                    "want": item.gold[last.id],
                    "note": "packed: only the final position is readable through the request API",
                }
            )
    return {"arm": mode, "rows": rows, "seconds": seconds, "tokens": tokens, "model": model}


def summarise(run: dict, questions: int) -> dict:
    """Questions per second and accuracy, with the first item dropped as warm-up in every arm."""
    seconds = run["seconds"][1:] or run["seconds"]
    answered = [r for r in run["rows"] if r["item"] > 0] or run["rows"]
    right = sum(r["got"] == r["want"] for r in answered)
    return {
        "arm": run["arm"],
        "items": len(seconds),
        "answers": len(answered),
        "accuracy": round(right / len(answered), 4) if answered else 0.0,
        "seconds": round(sum(seconds), 1),
        "questions_per_second": round(len(answered) / sum(seconds), 2) if sum(seconds) else 0.0,
        "median_ms_per_item": round(statistics.median(seconds) * 1e3, 1),
        "tokens": run["tokens"],
        "tokens_per_question": round(run["tokens"] / max(1, questions)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="against-vllm", description=__doc__)
    parser.add_argument("--task", default="race", choices=["boolq", "race", "unfair_tos"])
    parser.add_argument("--limit", type=int, default=60, help="contexts, not questions")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    parser.add_argument("--record", type=Path, help="run this package and write the result here")
    parser.add_argument("--against", type=Path, help="run vLLM and compare against a recorded run")
    parser.add_argument("--mode", default="separate", choices=["separate", "packed"])
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--utilisation", type=float, default=0.86)
    args = parser.parse_args(argv)

    # Identical items in both processes, because both load the same task with the same seed. Passing the items through
    # the recording instead would let a prompt difference hide in the file.
    items = tasks.load(args.task, args.limit, split=args.split, seed=args.seed)
    questions = sum(len(item.questions) for item in items)
    print(f"{args.task} [{args.split}, seed {args.seed}]: {len(items)} contexts, {questions} questions\n")

    if args.record:
        run = run_fork(items, args.model)
        args.record.write_text(json.dumps(run, indent=2, default=str))
        print(json.dumps(summarise(run, questions), indent=2))
        print(f"\nwritten to {args.record}")
        return 0

    if not args.against:
        raise SystemExit("one of --record or --against is required")

    recorded = json.loads(args.against.read_text())
    if recorded["model"] != args.model:
        raise SystemExit(f"the recording used {recorded['model']} and this run would use {args.model}")
    mine = run_vllm(items, args.model, args.mode, args.max_len, args.utilisation)

    left, right = summarise(recorded, questions), summarise(mine, questions)
    print(f"\n{'arm':>10} {'answers':>8} {'accuracy':>9} {'questions/s':>12} {'ms/context':>11} {'tokens':>10}")
    for side in (left, right):
        print(
            f"{side['arm']:>10} {side['answers']:>8} {side['accuracy']:>9.3f} "
            f"{side['questions_per_second']:>12.2f} {side['median_ms_per_item']:>11.1f} {side['tokens']:>10}"
        )
    if right["questions_per_second"]:
        ratio = left["questions_per_second"] / right["questions_per_second"]
        print(f"\nThe fork answers {ratio:.2f}x as many questions per second as vLLM {right['arm']}.")
    print(
        "\nTokens is what each arm sent through the model. The fork sends the context once per item; `separate`\n"
        "sends it once per question and relies on the prefix cache, so a large ratio there with a small\n"
        "ratio in time is the cache working. Accuracy is not a like-for-like comparison: the arms score different\n"
        "tokens -- this package scores the option text, and a request returning one token can only score the letters."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
