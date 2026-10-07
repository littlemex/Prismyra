#!/usr/bin/env python3
"""recon: reconcile fp8spd's speed measurement (short Japanese paragraph ~5,016/~20,064 tokens,
16Q -19.8%/64Q short -9.6%, RUN-fp8spd.md round5/6) against integ's (RUN-integ.md 12节, "race150
context, 約22,000文字, group=32", 1/16/64Q, L40S +23.8%/Spain +11.8% worse at Q64, no bit-match).

Builds BOTH document types in one process (one engine, one weight load), measures Q1/Q16/Q64 for
both `interleaved_fork=False` (two_pass) and `True` (fused), round-robin, group=32 throughout
(matching what RUN-integ.md 12节 says it used), and reports: the race150 document's real token
count (the open question -- is it above or below fp8spd's own 8192-token crossover?), and whether
this engine's current speed numbers land closer to fp8spd's or integ's reported figures.
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.environ.get("PRISMYRA_SRC", "."))
import torch

from prismyra import Boolean, Prismyra

MODEL = os.environ.get("PRISMYRA_MODEL", "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l")
RACE150 = os.environ.get("RACE150_PATH", "/work/recon/race150.json")
PARAGRAPH = (
    "返品は商品到着後三十日以内に限り受け付けます。未開封の商品は全額返金の対象となりますが、"
    "開封済みの商品については、初期不良が確認された場合を除き、返金ではなく交換のみの対応となります。"
)


def build_ja_context(pad_to_tokens, tokenizer):
    text = PARAGRAPH
    while len(tokenizer(text)["input_ids"]) < pad_to_tokens:
        text += PARAGRAPH
    return text


def build_race150_context(target_chars=22000):
    data = json.load(open(RACE150))
    items = data["items"]
    text = ""
    i = 0
    while len(text) < target_chars:
        text += items[i % len(items)]["context"] + "\n\n"
        i += 1
    return text


def make_questions(n):
    return [Boolean(id=f"q{i}", prompt=f"この文書は{i}番目の論点を支持しているか。") for i in range(n)]


def time_ask(engine, context, n, warmups=2, rounds=15, drop=4):
    qs = make_questions(n)
    samples = []
    for r in range(rounds):
        t0 = time.perf_counter()
        engine.ask(context, qs)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        if r >= drop:
            samples.append(dt)
    return samples


def roundrobin(engine, context, qcounts, rounds=15, drop=4):
    """two_pass vs fused, interleaved every round, like fp8spd's own speed_session scripts."""
    results = {n: {"two_pass": [], "fused": []} for n in qcounts}
    for r in range(rounds):
        for n in qcounts:
            qs = make_questions(n)
            engine.interleaved_fork = False
            t0 = time.perf_counter()
            engine.ask(context, qs)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            if r >= drop:
                results[n]["two_pass"].append(dt)

            engine.interleaved_fork = True
            t0 = time.perf_counter()
            engine.ask(context, qs)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            if r >= drop:
                results[n]["fused"].append(dt)
    return results


def report(label, results):
    print(f"\n--- {label} ---")
    for n, d in results.items():
        tp = sorted(d["two_pass"])
        fu = sorted(d["fused"])
        tp_med = statistics.median(tp)
        fu_med = statistics.median(fu)
        pct = (fu_med - tp_med) / tp_med * 100
        print(
            f"Q={n}: two_pass median={tp_med:.1f}ms (min={tp[0]:.1f} max={tp[-1]:.1f}) "
            f"fused median={fu_med:.1f}ms (min={fu[0]:.1f} max={fu[-1]:.1f}) diff={pct:+.2f}%"
        )


def main():
    engine = Prismyra(MODEL, require_kernels=True, interleaved_fork=True, wide_group=False, group=32)
    tokenizer = engine.tokenizer

    ja_short = build_ja_context(5016, tokenizer)
    ja_short_tok = len(tokenizer(ja_short)["input_ids"])
    print(f"JA short doc tokens={ja_short_tok}")

    race_ctx = build_race150_context(22000)
    race_tok = len(tokenizer(race_ctx)["input_ids"])
    print(f"race150-concat doc chars={len(race_ctx)} tokens={race_tok}  "
          f"(fp8spd's own crossover threshold: 8192 tokens)")

    qcounts = [1, 16, 64]

    results_ja = roundrobin(engine, ja_short, qcounts)
    report(f"JA short doc ({ja_short_tok} tokens), group=32", results_ja)

    results_race = roundrobin(engine, race_ctx, qcounts)
    report(f"race150-concat doc ({race_tok} tokens), group=32", results_race)


if __name__ == "__main__":
    main()
