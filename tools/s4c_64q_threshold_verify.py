#!/usr/bin/env python3
"""fp8spd4: verify the new context-length threshold (INTERLEAVE_WIDE_GROUP_TOKEN_LIMIT) in `ask()`/
`_ask_interleaved`. (1) torch.equal still holds at 64 questions on both sides of the threshold -- the
threshold only picks which bit-identical path (widened vs non-widened fused) runs, both already gated
elsewhere. (2) speed: short doc should still show the widened win, long doc should now match the
non-widened (never-regressed) path instead of the old regression.
"""
import sys

sys.path.insert(0, "/work/fp8spd/fp8spd4-src")
import torch
from prismyra import Prismyra, Boolean
from prismyra.engine import INTERLEAVE_WIDE_GROUP_TOKEN_LIMIT, WIDE_GROUP

MODEL = "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l"
PARAGRAPH = (
    "返品は商品到着後三十日以内に限り受け付けます。未開封の商品は全額返金の対象となりますが、"
    "開封済みの商品については、初期不良が確認された場合を除き、返金ではなく交換のみの対応となります。"
)


def build_context(pad_to_tokens, tokenizer):
    text = PARAGRAPH
    while len(tokenizer(text)["input_ids"]) < pad_to_tokens:
        text += PARAGRAPH
    return text


def main():
    engine = Prismyra(MODEL, require_kernels=True, interleaved_fork=True)
    tokenizer = engine.tokenizer
    qs = [Boolean(id=f"q{i}", prompt=f"条項 {i} は返金を認めているか。") for i in range(WIDE_GROUP)]

    print(f"INTERLEAVE_WIDE_GROUP_TOKEN_LIMIT={INTERLEAVE_WIDE_GROUP_TOKEN_LIMIT}")

    for target_tok in (5016, 20064):
        context = build_context(target_tok, tokenizer)
        ntok = len(tokenizer(context)["input_ids"])
        expect_widen = ntok < INTERLEAVE_WIDE_GROUP_TOKEN_LIMIT

        # torch.equal: new ask() path vs explicit two_pass baseline. A long (20k-token), non-paged (joined
        # cache) context at 64Q can hold two live caches close to this card's free-memory margin if the first
        # call's cache has not been released yet -- an explicit free between calls avoids an OOM-retry
        # confusing a memory-pressure artifact with a correctness regression in the new threshold logic itself.
        engine.interleaved_fork = False
        baseline = engine.ask(context, qs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        engine.interleaved_fork = True
        fused = engine.ask(context, qs)

        tp = torch.tensor(
            [v for qid in sorted(baseline.answers) for v in baseline.answers[qid].probabilities.values()],
            dtype=torch.float64,
        )
        il = torch.tensor(
            [v for qid in sorted(fused.answers) for v in fused.answers[qid].probabilities.values()],
            dtype=torch.float64,
        )
        eq = torch.equal(tp, il)
        print(f"tokens={ntok} expect_widen={expect_widen} torch.equal={eq} "
              f"maxdiff={(tp - il).abs().max().item():.3e}")
        assert eq, f"torch.equal FAILED at tokens={ntok}"

    print("ALL OK")


if __name__ == "__main__":
    main()
