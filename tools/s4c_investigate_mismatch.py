#!/usr/bin/env python3
"""fp8spd4: isolate why s4c_64q_threshold_verify.py's own (2-sentence) paragraph produced a non-bit-exact
32+32 (non-widened) interleaved result at ~20094 tokens while round5's (3-sentence) paragraph was exact at
~20064. Varies pad target and paragraph text independently to find which one the mismatch tracks.
"""
import sys
sys.path.insert(0, "/work/fp8spd/fp8spd4-src")
import torch
from prismyra import Prismyra, Boolean

MODEL = "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l"
P3 = ("返品は商品到着後三十日以内に限り受け付けます。未開封の商品は全額返金の対象となりますが、"
      "開封済みの商品については、初期不良が確認された場合を除き、返金ではなく交換のみの対応となります。"
      "返送にかかる送料は、初期不良の場合は当社が負担し、それ以外の理由による返品ではお客様のご負担となります。")
P2 = ("返品は商品到着後三十日以内に限り受け付けます。未開封の商品は全額返金の対象となりますが、"
      "開封済みの商品については、初期不良が確認された場合を除き、返金ではなく交換のみの対応となります。")


def build_context(pad_to_tokens, tokenizer, paragraph):
    text = paragraph
    while len(tokenizer(text)["input_ids"]) < pad_to_tokens:
        text += paragraph
    return text


def probs_of(result):
    return torch.tensor(
        [v for qid in sorted(result.answers) for v in result.answers[qid].probabilities.values()],
        dtype=torch.float64,
    )


def check(engine, tokenizer, pad, paragraph, label):
    context = build_context(pad, tokenizer, paragraph)
    ntok = len(tokenizer(context)["input_ids"])
    qs = [Boolean(id=f"q{i}", prompt=f"条項 {i} は返金を認めているか。") for i in range(64)]
    engine.interleaved_fork = False
    baseline = engine.ask(context, qs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    old = engine._ask_interleaved(context, qs, None)  # force non-widened, bypass ask()'s own decision
    eq = torch.equal(probs_of(baseline), probs_of(old))
    maxdiff = (probs_of(baseline) - probs_of(old)).abs().max().item()
    print(f"[{label}] tokens={ntok} torch.equal={eq} maxdiff={maxdiff:.4e}", flush=True)


def main():
    engine = Prismyra(MODEL, require_kernels=True, interleaved_fork=True)
    tokenizer = engine.tokenizer
    check(engine, tokenizer, 20000, P3, "P3(3-sentence) pad=20000")
    check(engine, tokenizer, 20094, P3, "P3(3-sentence) pad=20094")
    check(engine, tokenizer, 20000, P2, "P2(2-sentence) pad=20000")
    check(engine, tokenizer, 20094, P2, "P2(2-sentence) pad=20094")


if __name__ == "__main__":
    main()
