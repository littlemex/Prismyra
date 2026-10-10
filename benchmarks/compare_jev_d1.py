#!/usr/bin/env python3
"""Reproduce this project's own 1/16/64-question latency comparison against two other decision models.

Measures Prismyra's own checkpoint directly; the two competitors are measured through whatever client their own
weights ship with, not through this project's code (see "Usage" below for why). Both competitors are
third-party checkpoints with their own licences -- download them yourself:

    autotrust/JEV-27B-VL        bf16, needs a single GPU with 80 GB or more. For a 48 GB-class card, a
                                 community FP8 quantisation is `Atlas3D/JEV-27B-VL-FP8` (not an official release;
                                 verify its provenance yourself before trusting a comparison against it).
    LiquidAI/d1-3B               fits comfortably on any card this project targets.

Methodology, unchanged from the comparisons this project has already published (see `recipes/decision-lora/
README.md`'s own "Speed against other decision models" section): one synthetic document is built by repeating a
filler paragraph until a tokenizer reports at least `--target-tokens` (3,401 by default, this project's own prior
choice), then 1/16/64 questions are asked about it, 2 warm-up calls then 7 timed ones, and the median/min/max are
reported. The filler text is a public placeholder paragraph here, not this project's internal evaluation data
(RACE articles) -- length drives the measurement, not content, and a bundled fixture would not be reproducible
by a reader who does not already have this project's own non-public data files.

Usage:

    python3 benchmarks/compare_jev_d1.py --model littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l

Prismyra's own side is measured by this script directly. The competitors' own numbers are not reproduced by this
script -- there is no shared, stable client library for either one checked into this repository to call them
through, and vendoring one would drift from whatever the upstream repos ship next. Run them with their own
documented serving instructions, against the same document this script writes to `--doc-out`, and report the
same statistic (median of 7, after 2 warm-up calls) for a comparable number.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

FILLER = (
    "The committee reviewed the proposal in detail before reaching a decision. Every member had a chance to "
    "raise concerns, and the final vote was unanimous. The report that followed summarised the discussion and "
    "listed the next steps for the project. Several open questions were deferred to a later meeting, once more "
    "data had been collected from the field. "
)


def build_doc(tokenizer, target_tokens: int) -> tuple[str, int]:
    text = ""
    while True:
        text += FILLER
        n = len(tokenizer(text)["input_ids"])
        if n >= target_tokens:
            return text, n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="the Prismyra checkpoint to measure, repo id or local path")
    parser.add_argument("--target-tokens", type=int, default=3401)
    parser.add_argument("--questions", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--require-kernels", action="store_true")
    parser.add_argument("--doc-out", default=None, help="write the generated document's text here, so a competing "
                        "model's own client can be pointed at the identical bytes")
    args = parser.parse_args()

    from prismyra import Boolean, Prismyra

    engine = Prismyra(args.model, require_kernels=args.require_kernels)
    text, n_tokens = build_doc(engine.tokenizer, args.target_tokens)
    if args.doc_out:
        with open(args.doc_out, "w") as f:
            f.write(text)

    results = {"doc_tokens": n_tokens, "model": args.model}
    for n in args.questions:
        questions = [Boolean(id=f"q{i}", prompt=f"Is clause {i} about shipping?") for i in range(n)]
        for _ in range(args.warmup):
            engine.ask(text, questions)
        times_ms = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            engine.ask(text, questions)
            times_ms.append((time.perf_counter() - started) * 1000)
        results[str(n)] = {
            "median_ms": statistics.median(times_ms),
            "min_ms": min(times_ms),
            "max_ms": max(times_ms),
            "all_ms": times_ms,
        }
        print(f"{n:>3} question(s): median {results[str(n)]['median_ms']:.1f} ms "
              f"(min {results[str(n)]['min_ms']:.1f}, max {results[str(n)]['max_ms']:.1f})")

    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
