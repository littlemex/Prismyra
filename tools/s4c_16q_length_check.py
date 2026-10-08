#!/usr/bin/env python3
"""Does the long-context regression found at 64Q (group=64 WIDE_GROUP widening) also happen at 16Q
(the ordinary, non-widened interleaved_fork path, group stays at the engine's default 32)? If yes, the
context-length threshold belongs on `ask()`'s general `self.interleaved_fork` branch, not just the
`len(questions) == WIDE_GROUP` special case. If no, it is specific to widening the GDN state buffers to 64
rows and the threshold only needs to guard that one branch.

Usage: python3 s4c_16q_length_check.py
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.environ.get("PRISMYRA_SRC", "."))
import torch
from prismyra import Prismyra, Boolean

MODEL = os.environ.get("PRISMYRA_MODEL", "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l")
CONFIGS = ("two_pass", "interleaved")
LENGTHS = [5016, 9044, 20064]
ROUNDS = 9
WARMUP_ROUNDS = 3

PARAGRAPH = (
    "Returns are accepted only within thirty days of delivery. Unopened items qualify for a full refund, "
    "but opened items are exchanged rather than refunded unless a manufacturing fault is confirmed. "
    "Return shipping is paid by the seller when the item is faulty and by the buyer for any other reason."
)


def build_context(pad_to_tokens, tokenizer):
    text = PARAGRAPH
    while len(tokenizer(text)["input_ids"]) < pad_to_tokens:
        text += PARAGRAPH
    return text


def build_questions(n):
    return [Boolean(id=f"q{i}", prompt=f"Does clause {i} permit a refund?") for i in range(n)]


def run(engine, cfg, context, qs):
    engine.interleaved_fork = (cfg == "interleaved")
    return engine.ask(context, qs)


def main():
    engine = Prismyra(MODEL, require_kernels=True)
    tokenizer = engine.tokenizer
    qs = build_questions(16)

    all_reports = {}
    for target_tok in LENGTHS:
        context = build_context(target_tok, tokenizer)
        ntok = len(tokenizer(context)["input_ids"])

        samples = {cfg: [] for cfg in CONFIGS}
        for r in range(ROUNDS):
            for cfg in CONFIGS:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                run(engine, cfg, context, qs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                ms = (t1 - t0) * 1000
                if r >= WARMUP_ROUNDS:
                    samples[cfg].append(ms)

        report = {cfg: {"median_ms": statistics.median(vals), "min_ms": min(vals), "max_ms": max(vals), "n": len(vals)}
                   for cfg, vals in samples.items()}
        tp_med, il_med = report["two_pass"]["median_ms"], report["interleaved"]["median_ms"]
        report["pct_vs_two_pass"] = (il_med - tp_med) / tp_med * 100
        print(f"[tok={ntok}] two_pass={tp_med:.1f}ms interleaved={il_med:.1f}ms delta={report['pct_vs_two_pass']:+.2f}%",
              file=sys.stderr)
        all_reports[str(ntok)] = report

    print(json.dumps(all_reports, indent=2))
    with open(os.environ.get("S4C_16Q_OUT", "/tmp/s4c_16q_length_check.json"), "w") as f:
        json.dump(all_reports, f, indent=2)


if __name__ == "__main__":
    main()
