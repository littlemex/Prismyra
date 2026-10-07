#!/usr/bin/env python3
"""fp8spd4 (continued): does the WIDENED (group=64) interleaved path have the same token-count-specific
torch.equal gap as the non-widened (group=32) path just found at ctx_tokens in {20094, 20140}? If yes, this
is a shared, pre-existing root cause in the fused path generally (not introduced by the new length
threshold); if no, it is specific to the non-widened path and the threshold's fallback choice needs
reconsidering.
"""
import sys
sys.path.insert(0, "/work/fp8spd/fp8spd4-src")
import torch
from prismyra import Prismyra, Boolean
from prismyra.engine import WIDE_GROUP

MODEL = "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l"
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
    widened = engine._ask_interleaved(context, qs, WIDE_GROUP)  # force group=64 regardless of length
    eq = torch.equal(probs_of(baseline), probs_of(widened))
    maxdiff = (probs_of(baseline) - probs_of(widened)).abs().max().item()
    print(f"[{label}] tokens={ntok} torch.equal={eq} maxdiff={maxdiff:.4e}", flush=True)


def main():
    engine = Prismyra(MODEL, require_kernels=True, interleaved_fork=True)
    tokenizer = engine.tokenizer
    for pad in (20000, 20094):
        check(engine, tokenizer, pad, P2, f"P2 pad={pad} WIDENED(group=64)")


if __name__ == "__main__":
    main()
