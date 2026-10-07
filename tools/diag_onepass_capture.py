#!/usr/bin/env python3
"""recon: pin down WHERE in construction the interleaved_fork=True / wide_group=True config changes the
n=1 (_ask_in_one_pass) answer, given that _ask_in_one_pass itself never branches on either flag
(RUN-integ.md 12.3's puzzle). Hypothesis, from reading prismyra/engine.py:673-679 and
prismyra/onepass.py:350 (`record_all`): `Prismyra.__init__` calls `onepass.record_all(...)` -- which
does real `torch.cuda.graph` capture, baking in whichever matmul kernels are dispatched *at that exact
moment* -- strictly *after* the `interleaved_fork`-triggered `_enable_batch_invariance()` call
(engine.py:647-650). So an engine built with interleaved_fork=True captures its one-pass graphs under
vLLM's batch-invariant Triton kernel; the default engine captures them under the plain/default kernel.
Later `ask()` calls for n=1 just *replay* whichever graph was captured, which is why the result depends
on construction flags despite `_ask_in_one_pass` having no branch on them.

This script verifies that *re-recording* the one-pass graphs under a different batch-invariance state,
on the SAME engine / SAME weights (no second model load -- avoids OOM on a single 46GB card), changes
the n=1 replay by the same order of magnitude integ measured (RUN-integ.md 12.3: up to 0.208 on L40S),
and that toggling back round-trips to the original value exactly.
"""
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


def record_under(engine, state_label):
    """Re-run onepass.record_all under whatever batch-invariance state is currently active, and install
    the resulting graphs on the engine, mirroring exactly what `__init__` does at line 679."""
    print(f"[{state_label}] refcount before re-record = {eng_mod._BATCH_INVARIANT_REFCOUNT}", flush=True)
    held = onepass.record_all(engine, engine._pad_id(), engine._read_one_pass)
    engine._one_pass = held
    print(
        f"[{state_label}] re-recorded: buckets={sorted(held.buckets)} declined={held.declined} "
        f"record_ms={held.record_ms:.1f}",
        flush=True,
    )
    return held


def main():
    # Build the engine with everything off, so construction's own record_all captured the one-pass graphs
    # under NO registration -- this is "off"'s baseline, exactly like a default engine.
    engine = Prismyra(MODEL, require_kernels=True, interleaved_fork=False, wide_group=False)
    print(f"built: refcount={eng_mod._BATCH_INVARIANT_REFCOUNT} claimed={engine._invariance_claimed}")
    tokenizer = engine.tokenizer
    context = build_context(5016, tokenizer)
    ntok = len(tokenizer(context)["input_ids"])
    print(f"context tokens={ntok}")

    # Step 1: ask n=1 with the construction-time ("off") recording already in place.
    probs_off1 = ask_q1(engine, context)
    print(f"[off, construction recording] probs={probs_off1}")

    # Step 2: manually claim the registration (same effect interleaved_fork=True would have had, isolated
    # from any other side effect of that flag) and re-record the one-pass graphs under it.
    eng_mod._enable_batch_invariance()
    engine._invariance_claimed = True
    record_under(engine, "on (re-recorded)")
    probs_on = ask_q1(engine, context)
    print(f"[on, re-recorded]            probs={probs_on}")

    # Step 3: release the registration and re-record again -- should round-trip back to probs_off1 exactly,
    # since nothing else about the engine or the input changed.
    eng_mod._disable_batch_invariance()
    engine._invariance_claimed = False
    record_under(engine, "off (re-recorded, round-trip)")
    probs_off2 = ask_q1(engine, context)
    print(f"[off, re-recorded round-trip] probs={probs_off2}")

    t_off1 = reduce_tensor(probs_off1)
    t_on = reduce_tensor(probs_on)
    t_off2 = reduce_tensor(probs_off2)

    print("\n=== comparison ===")
    print(f"off(construction) vs on(re-recorded):      torch.equal={torch.equal(t_off1, t_on)} "
          f"maxdiff={(t_off1 - t_on).abs().max().item():.6e}")
    print(f"off(construction) vs off(re-recorded rt):   torch.equal={torch.equal(t_off1, t_off2)} "
          f"maxdiff={(t_off1 - t_off2).abs().max().item():.6e}")


if __name__ == "__main__":
    main()
