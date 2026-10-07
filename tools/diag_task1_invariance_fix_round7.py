#!/usr/bin/env python3
"""fp8spd6 (round7): verify the task-1 fix (`if (paged or interleaved_fork) and on_cuda:` in
`Prismyra.__init__`, inherited uncommitted from fp8spd5) actually closes the sm_120 2/3/16-question
mismatch fp8spd4 found (RUN-fp8spd.md round6) and fp8spd4's "L00.mlp context-side call diverges"
diagnosis (round6, section 13 in RUN-inv.md). inv5 independently found (RUN-inv.md 13.5) that
`FusedExpertsFp4` is innocent *when the engine's batch-invariance dispatcher registration is active* --
the fix tested here is exactly "make `interleaved_fork=True` turn that registration on, the same way
`paged=True` already does".

Two runs: (A) with the fix as-is (positive), (B) with the fix reverted in-process (monkeypatch
`engine._BATCH_INVARIANT_REFCOUNT`/`_enable_batch_invariance` is global and sticky across engines in
one process, so B is run in a **separate process invocation** with env var FP8SPD6_REVERT_FIX=1 that
monkeypatches `Prismyra.__init__`'s own module-level guard before constructing the engine) to show the
mismatch reappears without it -- a causal demonstration, not just a pass/fail snapshot.
"""
import os
import sys

sys.path.insert(0, os.environ.get("PRISMYRA_SRC", "."))
import torch

from prismyra import Boolean, Prismyra

MODEL = os.environ.get("PRISMYRA_MODEL", "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l")
PARAGRAPH = (
    "返品は商品到着後三十日以内に限り受け付けます。未開封の商品は全額返金の対象となりますが、"
    "開封済みの商品については、初期不良が確認された場合を除き、返金ではなく交換のみの対応となります。"
)
REVERT_FIX = os.environ.get("FP8SPD6_REVERT_FIX") == "1"


def build_context(pad_to_tokens, tokenizer):
    text = PARAGRAPH
    while len(tokenizer(text)["input_ids"]) < pad_to_tokens:
        text += PARAGRAPH
    return text


def main():
    if REVERT_FIX:
        # Negative control: patch the module-level `_enable_batch_invariance` to a no-op *before*
        # constructing the engine, so `interleaved_fork=True` no longer gets the dispatcher registration
        # (reproducing fp8spd4's round6 world) while `paged=True` would still work normally (not used here).
        import prismyra.engine as eng_mod

        _orig_init = eng_mod.Prismyra.__init__

        def _patched_init(self, *a, **kw):
            real_enable = eng_mod._enable_batch_invariance
            # Only suppress the call that `interleaved_fork` alone would trigger; still allow `paged` (not
            # exercised here) to keep working, by checking the kwarg this constructor call was given.
            interleaved_fork = kw.get("interleaved_fork", False)
            paged = kw.get("paged", False)
            if interleaved_fork and not paged:
                eng_mod._enable_batch_invariance = lambda: None
            try:
                _orig_init(self, *a, **kw)
            finally:
                eng_mod._enable_batch_invariance = real_enable

        eng_mod.Prismyra.__init__ = _patched_init

    engine = Prismyra(MODEL, require_kernels=True, interleaved_fork=True)
    tokenizer = engine.tokenizer

    qcounts = [1, 2, 3, 16, 33, 64]
    pads = [5016, 20064]

    all_eq = True
    for pad in pads:
        context = build_context(pad, tokenizer)
        ntok = len(tokenizer(context)["input_ids"])
        for n in qcounts:
            qs = [Boolean(id=f"q{i}", prompt=f"条項 {i} は返金を認めているか。") for i in range(n)]
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
            maxdiff = (tp - il).abs().max().item()
            all_eq = all_eq and eq
            print(
                f"REVERT_FIX={REVERT_FIX} model={MODEL} tokens={ntok} n={n} torch.equal={eq} "
                f"maxdiff={maxdiff:.6e}",
                flush=True,
            )

    print(f"ALL_EQUAL={all_eq}")


if __name__ == "__main__":
    main()
