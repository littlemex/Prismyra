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
    "返品は商品到着後三十日以内に限り受け付けます。未開封の商品は全額返金の対象となりますが、"
    "開封済みの商品については、初期不良が確認された場合を除き、返金ではなく交換のみの対応となります。"
    "返送にかかる送料は、初期不良の場合は当社が負担し、それ以外の理由による返品ではお客様のご負担となります。"
)


def build_context(pad_to_tokens, tokenizer):
    text = PARAGRAPH
    while len(tokenizer(text)["input_ids"]) < pad_to_tokens:
        text += PARAGRAPH
    return text


def build_questions(n):
    return [Boolean(id=f"q{i}", prompt=f"条項 {i} は返金を認めているか。") for i in range(n)]


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
