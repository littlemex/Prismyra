#!/usr/bin/env python3
"""Alternating speed measurement for wide_group (two_pass vs widened-to-WIDE_GROUP), on the v0.4.0
branch where 33-64 questions are confirmed torch.equal. Interleaves wide_group on/off every round,
round-robin across the given question counts, same engine/same weight load, so the comparison is
never confounded by a second construction -- two separately-constructed engines give measurements
that disagree with each other for reasons unrelated to wide_group itself.
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.environ.get("PRISMYRA_SRC", "."))
import torch

from prismyra import Boolean, Prismyra

MODEL = os.environ.get("PRISMYRA_MODEL", "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l")
PARAGRAPH = (
    "返品は商品到着後三十日以内に限り受け付けます。未開封の商品は全額返金の対象となりますが、"
    "開封済みの商品については、初期不良が確認された場合を除き、返金ではなく交換のみの対応となります。"
    "配送は注文確定から二営業日以内に発送します。離島・一部地域では追加で二日ほどかかる場合があります。"
)
QCOUNTS = [int(x) for x in os.environ.get("WG_SPEED_QCOUNTS", "33,48,63,64").split(",")]
PAD = int(os.environ.get("WG_SPEED_PAD", "5016"))
ROUNDS = int(os.environ.get("WG_SPEED_ROUNDS", "15"))
DROP = int(os.environ.get("WG_SPEED_DROP", "4"))


def build_context(pad_to_tokens, tokenizer):
    text = PARAGRAPH
    while len(tokenizer(text)["input_ids"]) < pad_to_tokens:
        text += PARAGRAPH
    return text


def make_questions(n):
    return [Boolean(id=f"q{i}", prompt=f"条項 {i} はこの文書の主題について述べているか。") for i in range(n)]


def roundrobin(engine, context, qcounts, rounds, drop):
    results = {n: {"two_pass": [], "widened": []} for n in qcounts}
    for r in range(rounds):
        for n in qcounts:
            qs = make_questions(n)
            engine.wide_group = False
            t0 = time.perf_counter()
            engine.ask(context, qs)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            if r >= drop:
                results[n]["two_pass"].append(dt)

            engine.wide_group = True
            t0 = time.perf_counter()
            engine.ask(context, qs)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            if r >= drop:
                results[n]["widened"].append(dt)
    return results


def main():
    engine = Prismyra(MODEL, require_kernels=True, wide_group=False, paged=False, interleaved_fork=False)
    tokenizer = engine.tokenizer
    context = build_context(PAD, tokenizer)
    ntok = len(tokenizer(context)["input_ids"])
    print(f"model={MODEL} tokens={ntok} qcounts={QCOUNTS} rounds={ROUNDS} drop={DROP}", flush=True)

    # warm both paths once outside the timed loop
    for n in QCOUNTS:
        qs = make_questions(n)
        engine.wide_group = False
        engine.ask(context, qs)
        engine.wide_group = True
        engine.ask(context, qs)
    torch.cuda.synchronize()

    results = roundrobin(engine, context, QCOUNTS, ROUNDS, DROP)
    out = {}
    for n in QCOUNTS:
        tp = results[n]["two_pass"]
        wd = results[n]["widened"]
        tp_med, wd_med = statistics.median(tp), statistics.median(wd)
        pct = (wd_med - tp_med) / tp_med * 100
        out[n] = {
            "two_pass_median_ms": tp_med, "two_pass_min_ms": min(tp), "two_pass_max_ms": max(tp),
            "widened_median_ms": wd_med, "widened_min_ms": min(wd), "widened_max_ms": max(wd),
            "pct_change": pct, "n_samples": len(tp),
        }
        print(
            f"n={n} tokens={ntok} two_pass_median={tp_med:.1f}ms [{min(tp):.1f},{max(tp):.1f}] "
            f"widened_median={wd_med:.1f}ms [{min(wd):.1f},{max(wd):.1f}] change={pct:+.2f}% "
            f"samples={len(tp)}",
            flush=True,
        )
    import json
    with open(os.environ.get("WG_SPEED_OUT", "/tmp/wg_speed.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
