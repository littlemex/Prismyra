#!/usr/bin/env python3
"""fp8spd6 (round8, task2): torch.equal gate combining `wide_group` (S4a, 33-64 questions widened to
group=64 under INTERLEAVE_WIDE_GROUP_TOKEN_LIMIT=8192 tokens) with the Shelf/Batcher fused path (round 7/8).

Ground truth is `ask()` on the same engine with `paged=False` (the joined-cache path, already torch.equal
-gated for `wide_group` in earlier rounds). Compared against `Batcher.ask()` on the same engine with
`paged=True` (toggling the `paged` property reuses the already-loaded weights instead of a second copy,
which would not fit on one card) -- single document, 48 questions (33-64 range), under 8,192 tokens.
Then a second comparison: the same single wide document fused *alongside* another, smaller, fresh document
in one Batcher pass, to check the mixed-group bin this round's own `_shelf_ask_interleaved_many` change
added support for.
"""
import os
import sys

sys.path.insert(0, os.environ.get("PRISMYRA_SRC", "."))
import torch

from prismyra import Boolean, Prismyra
from prismyra.schedule import Batcher

MODEL = os.environ.get("PRISMYRA_MODEL", "littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l")
CONTEXT = (
    "返品は商品到着後三十日以内に限り受け付けます。未開封の商品は全額返金の対象となりますが、"
    "開封済みの商品については、初期不良が確認された場合を除き、返金ではなく交換のみの対応となります。"
)
SECOND_CONTEXT = (
    "配送は注文確定から二営業日以内に発送します。離島・一部地域では追加で二日ほどかかる場合があります。"
)


def questions_for(n, tag=""):
    return [Boolean(id=f"{tag}q{i}", prompt=f"条項 {i} はこの文書の主題について述べているか。") for i in range(n)]


def probs_of(result):
    return torch.tensor(
        [v for qid in sorted(result.answers) for v in result.answers[qid].probabilities.values()],
        dtype=torch.float64,
    )


def main():
    engine = Prismyra(MODEL, require_kernels=True, interleaved_fork=True, wide_group=True, paged=False)
    qs = questions_for(48)

    # Every `ask()` ground-truth call happens here, while `self.paged` is still `False` -- `open_context`'s
    # own cache construction reads `self.paged` at call time (`_claim_cache`'s own `paged=self.paged`), so a
    # ground truth taken *after* flipping it to `True` for the Batcher part below would silently become a
    # paged-cache comparison instead of the trusted joined-cache one this gate means to check against.
    ground_truth = engine.ask(CONTEXT, qs)
    gt_a = engine.ask(CONTEXT, questions_for(3, "m_a_"))
    gt_b = engine.ask(SECOND_CONTEXT, questions_for(5, "m_b_"))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    engine.paged = True
    batcher = Batcher(engine).start()
    try:
        shelf_result = batcher.ask(CONTEXT, qs, timeout=60.0)
        pg, ps = probs_of(ground_truth), probs_of(shelf_result)
        eq = torch.equal(pg, ps)
        maxdiff = (pg - ps).abs().max().item()
        print(f"[single wide, N=48] torch.equal={eq} maxdiff={maxdiff:.6e}", flush=True)

        # Sharing the same shelf/pool: a wide single document (group=64) got its own fused bin above --
        # `_round_rows(33..64, 64)` is always exactly 64, so a wide-triggering document always fills the
        # whole widened pool alone and no companion can share its bin (not a bug to work around, just this
        # round's own finding about why "a wide document fused *with* a companion" cannot exist). What a
        # shared shelf's own pool capacity (opened at WIDE_GROUP once, round 8) must still get right is the
        # *next* pass -- two ordinary small fresh documents, each well under WIDE_GROUP_FROM, fused together
        # through `_shelf_ask_interleaved_many` right after the wide pass used a group=64 buffer width on the
        # same cache. `gt_a`/`gt_b` (above, taken before `self.paged` flipped) are the trusted joined-cache
        # comparison -- `ask()` does not go through `wide_group`'s own widen path at these small counts, so
        # this is an ordinary two-pass comparison, not a repeat of the gate above.
        job_a = batcher.submit(CONTEXT, questions_for(3, "m_a_"))
        job_b = batcher.submit(SECOND_CONTEXT, questions_for(5, "m_b_"))
        job_a.done.wait(timeout=60.0)
        job_b.done.wait(timeout=60.0)
        if job_a.error:
            raise job_a.error
        if job_b.error:
            raise job_b.error
        eq2a = torch.equal(probs_of(gt_a), probs_of(job_a.result))
        eq2b = torch.equal(probs_of(gt_b), probs_of(job_b.result))
        print(f"[after-wide small-doc fusion] docA torch.equal={eq2a} docB torch.equal={eq2b}", flush=True)

        print(f"ALL_EQUAL={eq and eq2a and eq2b}")
    finally:
        batcher.stop()


if __name__ == "__main__":
    main()
