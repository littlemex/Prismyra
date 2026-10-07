#!/usr/bin/env python3
"""recon: verify the fix (unconditional, permanent batch-invariance claim for every CUDA engine,
engine.py's new `self._invariance_base_claimed`). Three checks on one engine, one weight load:
  1. A plain default-flags engine now claims invariance (refcount==1, base_claimed=True).
  2. Its n=1 answer for the fixed document/question matches the value previously measured under
     interleaved_fork=True ('no': 0.040483687072992325, 'yes': 0.9595163464546204 at 5016 tokens) --
     i.e. the construction-flag dependence integ found is gone.
  3. Re-recording the one-pass graphs twice on this engine (same as recon_diag_onepass_capture.py's
     "off vs off-re-recorded" probe, which found 0.0081 noise before this fix) now round-trips exactly.
  4. interleaved_fork=True and wide_group=True (built fresh, same process after freeing the first
     engine) give the IDENTICAL n=1 answer too -- the whole point: config no longer matters.
"""
import gc
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
    qs = [Boolean(id=qid, prompt="条項は返金を認めているか。")]
    result = engine.ask(context, qs)
    return result.answers[qid].probabilities


def reduce_tensor(d):
    return torch.tensor([d[k] for k in sorted(d)], dtype=torch.float64)


EXPECTED_ON = {"no": 0.040483687072992325, "yes": 0.9595163464546204}

print("=== check 1+2: default-flags engine ===")
engine = Prismyra(MODEL, require_kernels=True, interleaved_fork=False, wide_group=False)
print(
    f"refcount={eng_mod._BATCH_INVARIANT_REFCOUNT} base_claimed={engine._invariance_base_claimed} "
    f"paged_claimed={engine._invariance_claimed}"
)
tokenizer = engine.tokenizer
context = build_context(5016, tokenizer)
probs_default = ask_q1(engine, context)
print(f"default engine n=1 probs={probs_default}")
t_default = reduce_tensor(probs_default)
t_expected = reduce_tensor(EXPECTED_ON)
print(
    f"default vs previously-measured 'on': torch.equal={torch.equal(t_default, t_expected)} "
    f"maxdiff={(t_default - t_expected).abs().max().item():.6e}"
)

print("\n=== check 3: re-recording round-trips exactly now ===")
probs_before = ask_q1(engine, context)
engine._one_pass = onepass.record_all(engine, engine._pad_id(), engine._read_one_pass)
probs_after = ask_q1(engine, context)
t_before, t_after = reduce_tensor(probs_before), reduce_tensor(probs_after)
print(
    f"before vs after re-record: torch.equal={torch.equal(t_before, t_after)} "
    f"maxdiff={(t_before - t_after).abs().max().item():.6e}"
)

del engine
gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()
print(f"after teardown: refcount={eng_mod._BATCH_INVARIANT_REFCOUNT} (expected to stay >=0, permanent claim never released by this engine's own del)")

print("\n=== check 4: interleaved_fork=True, wide_group=True gives the SAME n=1 answer ===")
engine2 = Prismyra(MODEL, require_kernels=True, interleaved_fork=True, wide_group=True)
print(
    f"refcount={eng_mod._BATCH_INVARIANT_REFCOUNT} base_claimed={engine2._invariance_base_claimed} "
    f"paged_claimed={engine2._invariance_claimed}"
)
probs_on = ask_q1(engine2, context)
print(f"on engine n=1 probs={probs_on}")
t_on = reduce_tensor(probs_on)
print(
    f"default vs on(interleaved_fork+wide_group): torch.equal={torch.equal(t_default, t_on)} "
    f"maxdiff={(t_default - t_on).abs().max().item():.6e}"
)
