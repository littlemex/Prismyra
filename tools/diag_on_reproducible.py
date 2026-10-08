#!/usr/bin/env python3
"""Does re-recording the one-pass graphs TWICE under the SAME "on" (batch-invariant) state give
the identical answer, in contrast to the "off" (default cuBLAS) state's own re-record noise measured in
`diag_onepass_capture.py` (off vs off-re-recorded maxdiff 0.0081)? If "on" is perfectly reproducible
across re-records while "off" is not, that confirms the noise source is cuBLAS/cuBLASLt's own
heuristic, timing-based algorithm selection for the un-pinned default matmul path -- not autotuner
racing (already ruled out: `diag_autotune_gap.py` showed every Triton Autotuner pinned to
configs=1, cache_entries=0 after full construction)."""
import os
import sys

sys.path.insert(0, os.environ.get("PRISMYRA_SRC", "."))
import torch

from prismyra import Boolean, Prismyra
import prismyra.engine as eng_mod
from prismyra import onepass

MODEL = os.environ.get("PRISMYRA_MODEL", "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l")
PARAGRAPH = (
    "返品は商品到着後三十日以内に限り受け付けます。未開封の商品は全額返金の対象となりますが、"
    "開封済みの商品については、初期不良が確認された場合を除き、返金ではなく交換のみの対応となります。"
)


def build_context(pad_to_tokens, tokenizer):
    text = PARAGRAPH
    while len(tokenizer(text)["input_ids"]) < pad_to_tokens:
        text += PARAGRAPH
    return text


def ask_q1(engine, context, qid="q0"):
    qs = [Boolean(id=qid, prompt="Does the clause permit a refund?")]
    result = engine.ask(context, qs)
    return result.answers[qid].probabilities


def reduce_tensor(d):
    return torch.tensor([d[k] for k in sorted(d)], dtype=torch.float64)


def main():
    engine = Prismyra(MODEL, require_kernels=True, interleaved_fork=False, wide_group=False)
    tokenizer = engine.tokenizer
    context = build_context(5016, tokenizer)

    eng_mod._enable_batch_invariance()
    engine._invariance_claimed = True
    engine._one_pass = onepass.record_all(engine, engine._pad_id(), engine._read_one_pass)
    probs_on1 = ask_q1(engine, context)
    print(f"[on, recording #1] probs={probs_on1}")

    engine._one_pass = onepass.record_all(engine, engine._pad_id(), engine._read_one_pass)
    probs_on2 = ask_q1(engine, context)
    print(f"[on, recording #2] probs={probs_on2}")

    engine._one_pass = onepass.record_all(engine, engine._pad_id(), engine._read_one_pass)
    probs_on3 = ask_q1(engine, context)
    print(f"[on, recording #3] probs={probs_on3}")

    t1, t2, t3 = reduce_tensor(probs_on1), reduce_tensor(probs_on2), reduce_tensor(probs_on3)
    print(f"on#1 vs on#2: torch.equal={torch.equal(t1, t2)} maxdiff={(t1-t2).abs().max().item():.6e}")
    print(f"on#1 vs on#3: torch.equal={torch.equal(t1, t3)} maxdiff={(t1-t3).abs().max().item():.6e}")


if __name__ == "__main__":
    main()
