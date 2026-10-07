#!/usr/bin/env python3
"""wg: does the wide_group (S4a, 33-64 questions widened to one group=64 pass) mismatch found by
fp8spd3 (RUN-fp8spd.md "33問だけ不一致") still reproduce on today's integrated branch (integ/v0.4.0,
a6b0087+)? Since that finding, three things landed that could have changed the picture:
  - per-document (not combined) `_round_rows` rounding (inv, round 2)
  - the branch-pass convolution batch fix (inv5, dadb996/32b93c3)
  - every CUDA engine unconditionally claims batch-invariance at construction (recon, ad7bafe)
This gate re-measures from scratch rather than trusting the old report.

Ground truth: `wide_group=False` (today's default), two passes (e.g. 33 -> 32+1). Compared against
`wide_group=True`, one pass widened to `WIDE_GROUP` (64) via `_group_for`. Both through plain `ask()`
(`paged=False`, `interleaved_fork=False`) -- the joined-cache path S4a's own gate used, independent of
`interleaved_fork`'s separate 64-exact mechanism in `_ask_interleaved`.

Asserts `engine._invariance_base_claimed` is True before measuring anything -- a diagnostic that ran
without this would not be testing the branch this task cares about.

Set WG_REVERT_FIX=1 to run a negative control: monkeypatch `_enable_batch_invariance` to a no-op before
construction (reproducing the pre-recon-fix world) to check whether the fix now unconditionally claimed
is actually *why* the old mismatch is gone, or whether something else changed independently.
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
    "配送は注文確定から二営業日以内に発送します。離島・一部地域では追加で二日ほどかかる場合があります。"
)
QCOUNTS = [int(x) for x in os.environ.get("WG_QCOUNTS", "33,40,48,63,64").split(",")]
PADS = [int(x) for x in os.environ.get("WG_PADS", "5016").split(",")]
REVERT_FIX = os.environ.get("WG_REVERT_FIX") == "1"


def build_context(pad_to_tokens, tokenizer):
    text = PARAGRAPH
    while len(tokenizer(text)["input_ids"]) < pad_to_tokens:
        text += PARAGRAPH
    return text


def questions_for(n):
    return [Boolean(id=f"q{i}", prompt=f"条項 {i} はこの文書の主題について述べているか。") for i in range(n)]


def probs_of(result):
    return torch.tensor(
        [v for qid in sorted(result.answers) for v in result.answers[qid].probabilities.values()],
        dtype=torch.float64,
    )


def main():
    if REVERT_FIX:
        import prismyra.engine as eng_mod

        eng_mod._enable_batch_invariance = lambda: None

    engine = Prismyra(MODEL, require_kernels=True, wide_group=False, paged=False, interleaved_fork=False)
    base_claimed = getattr(engine, "_invariance_base_claimed", None)
    print(f"REVERT_FIX={REVERT_FIX} invariance_base_claimed={base_claimed}", flush=True)
    if not REVERT_FIX:
        assert base_claimed is True, (
            "engine was built without the unconditional batch-invariance claim (recon's ad7bafe) -- "
            "this gate must not run against a world where that fix is absent."
        )
    tokenizer = engine.tokenizer

    all_eq = True
    for pad in PADS:
        context = build_context(pad, tokenizer)
        ntok = len(tokenizer(context)["input_ids"])
        for n in QCOUNTS:
            qs = questions_for(n)
            engine.wide_group = False
            two_pass = engine.ask(context, qs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            engine.wide_group = True
            widened = engine.ask(context, qs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            tp = probs_of(two_pass)
            wg = probs_of(widened)
            eq = torch.equal(tp, wg)
            maxdiff = (tp - wg).abs().max().item()
            first_bad = None
            for qid in sorted(two_pass.answers):
                for opt, p in two_pass.answers[qid].probabilities.items():
                    g = widened.answers[qid].probabilities[opt]
                    if g != p:
                        first_bad = (qid, opt, p, g)
                        break
                if first_bad:
                    break
            all_eq = all_eq and eq
            print(
                f"REVERT_FIX={REVERT_FIX} tokens={ntok} n={n} torch.equal={eq} maxdiff={maxdiff:.6e} "
                f"first_bad={first_bad}",
                flush=True,
            )

    print(f"ALL_EQUAL={all_eq}")


if __name__ == "__main__":
    main()
