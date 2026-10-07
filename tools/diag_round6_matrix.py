"""inv4, round 6 (chair: "fix it, don't accept the residual"). A `datasets`-free stand-in for `audit_sm120.py`'s
full companion matrix -- `datasets` cannot be installed alongside this image's pinned `transformers`/`tokenizers`
(huggingface-hub<2.0) without breaking model loading entirely (confirmed on the fresh `prismyra-inv4` pod), so this
reuses the 8 inline documents from `diag_qn1_residual_round5.py`/`diag_num_splits_sweep.py` instead of real RACE
documents. Smaller than `audit_sm120.py`'s 24-document sweep but covers the same companion-count x question-count
shape (group sizes {1,2,3,8}, question counts {1,2,3,7} -- 7 stands in for `audit_sm120.py`'s 31/32/33 group=64
boundary probe, since these 8 short documents cannot carry 31+ distinct questions each).

Usage: PRISMYRA_MODEL=<repo> python3 diag_round6_matrix.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from prismyra import Boolean, Prismyra  # noqa: E402

MODEL = os.environ.get("PRISMYRA_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")

CONTEXTS = [
    "Returns are accepted within thirty days of delivery. Unopened items are refunded in full. Opened items are "
    "exchanged rather than refunded, unless a manufacturing fault is confirmed. Return shipping is paid by the "
    "seller when the item is faulty and by the buyer otherwise.",
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
BASE_QUESTIONS = [
    Boolean(id="q0", prompt="Is a refund limited to unopened items?"),
    Boolean(id="q1", prompt="Does this clause mention a time limit?"),
    Boolean(id="q2", prompt="Does this clause mention a receipt?"),
    Boolean(id="q3", prompt="Is a photo ID ever required?"),
    Boolean(id="q4", prompt="Does this clause mention shipping?"),
    Boolean(id="q5", prompt="Does this clause mention a technician?"),
    Boolean(id="q6", prompt="Does this clause mention a competitor?"),
]
GROUP_SIZES = (1, 2, 3, 8)
QUESTION_COUNTS = (1, 2, 3, 7)


def questions_for(doc_idx: int, n: int):
    return [dataclasses.replace(BASE_QUESTIONS[i], id=f"doc{doc_idx}_q{i}") for i in range(n)]


def main():
    engine = Prismyra(MODEL, paged=True, graphs=False, group=64)
    print(f"model={MODEL} device={torch.cuda.get_device_name(0)}", flush=True)

    total = 0
    non_exact = 0
    decision_flips = 0
    worst = 0.0
    by_group: dict[int, int] = {}
    by_qn: dict[int, int] = {}

    for qn in QUESTION_COUNTS:
        truth = {}
        for doc_idx in range(8):
            want = engine.ask(CONTEXTS[doc_idx], questions_for(doc_idx, qn))
            truth[doc_idx] = want

        for group_size in GROUP_SIZES:
            docs = CONTEXTS[:group_size]
            try:
                with engine.open_batch(docs) as batch:
                    per_doc = [questions_for(i, qn) for i in range(group_size)]
                    results = batch.ask(per_doc)
            except Exception as e:  # noqa: BLE001
                print(f"  group={group_size} qn={qn}: RAISED {type(e).__name__}: {e}", flush=True)
                continue
            for doc_idx in range(group_size):
                for q in questions_for(doc_idx, qn):
                    alone = truth[doc_idx][q.id]
                    got = results[doc_idx][q.id]
                    for option, p in alone.probabilities.items():
                        total += 1
                        gp = got.probabilities[option]
                        if gp != p:
                            non_exact += 1
                            by_group[group_size] = by_group.get(group_size, 0) + 1
                            by_qn[qn] = by_qn.get(qn, 0) + 1
                            worst = max(worst, abs(gp - p))
                    if got.option != alone.option:
                        decision_flips += 1

    print(f"\ntotal probability checks: {total}, bit-exact: {total - non_exact}, non-exact: {non_exact}", flush=True)
    print(f"decision flips: {decision_flips}, worst probability move: {worst:.6f}", flush=True)
    print(f"non-exact count by companion group_size: {by_group}", flush=True)
    print(f"non-exact count by question-count: {by_qn}", flush=True)


if __name__ == "__main__":
    main()
