"""inv4, round 6: does the qn=1 residual depend on *where* the target document sits in the flat concatenated
batch, not just on how many companions it has? `diag_round6_matrix.py` found an identical failure set (26/364,
same group_size/question-count breakdown) whether the paged branch-read uses FA2's `flash_attn_varlen_func` or
vLLM's `unified_attention` -- ruling attention out as the sole or dominant cause. The next candidate, per round 4's
own note that GDN/conv1d pass synthetic per-op tests but the real model still shows a residual, is a chunk-boundary
or global-offset sensitivity in the GatedDeltaNet chunked recurrence or the depthwise conv1d, both of which read
`varlen.current()` boundaries that encode each document's start offset in the flat row array -- a document's own
offset changes depending on which other documents are concatenated before it, even if its own row count does not.

This holds the target document and its question fixed and only moves its *position* in an 8-document open_batch
(first vs last), to see whether the move's size depends on position as well as on companion count.

Usage: PRISMYRA_MODEL=<repo> python3 diag_concat_position.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from prismyra import Boolean, Prismyra  # noqa: E402

MODEL = os.environ.get("PRISMYRA_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")

TARGET_CONTEXT = (
    "Returns are accepted within thirty days of delivery. Unopened items are refunded in full. Opened items are "
    "exchanged rather than refunded, unless a manufacturing fault is confirmed. Return shipping is paid by the "
    "seller when the item is faulty and by the buyer otherwise."
)
TARGET_QUESTION = Boolean(id="target", prompt="Is a refund limited to unopened items?")
COMPANION_QUESTION = Boolean(id="companion", prompt="Does this clause mention a time limit?")
COMPANION_CONTEXTS = [
    "Gift cards never expire and are not redeemable for cash. A lost gift card is replaced only with the original "
    "purchase receipt and a matching photo ID.",
    "Membership renews automatically each year unless cancelled thirty days before the renewal date. A refund is "
    "given only for the unused portion of the current year.",
    "Warranty claims require the original receipt and are void if the device was opened by anyone other than an "
    "authorised technician.",
    "Delivery windows are estimates, not guarantees, and a delay of under five business days does not qualify for "
    "a shipping refund.",
    "Price matching applies only to identical items sold directly by a competitor, not to marketplace sellers or "
    "clearance stock.",
    "A subscription paused for more than ninety days is treated as cancelled and restarting it requires a new "
    "sign-up at the current price.",
    "Store credit never expires but is forfeited if the account it was issued to is closed for inactivity.",
]


def main():
    engine = Prismyra(MODEL, paged=True, graphs=False, group=64)
    print(f"model={MODEL} device={torch.cuda.get_device_name(0)}", flush=True)

    truth = engine.ask(TARGET_CONTEXT, [TARGET_QUESTION])["target"]
    print(f"alone: {dict(truth.probabilities)}", flush=True)

    for position, label in ((0, "first"), (7, "last")):
        docs = COMPANION_CONTEXTS[:7]
        docs.insert(position, TARGET_CONTEXT)
        per_doc = [[COMPANION_QUESTION] for _ in docs]
        per_doc[position] = [TARGET_QUESTION]
        with engine.open_batch(docs) as batch:
            results = batch.ask(per_doc)
        got = results[position]["target"]
        diffs = {opt: abs(got.probabilities[opt] - truth.probabilities[opt]) for opt in truth.probabilities}
        worst = max(diffs.values())
        print(f"position={label} ({position}/8): got={dict(got.probabilities)} diffs={diffs} worst={worst:.6e}",
              flush=True)


if __name__ == "__main__":
    main()
