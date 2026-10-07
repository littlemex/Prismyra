"""inv4, round 5 follow-up: `diag_qn1_residual_round5.py` traced the 72/4,392 question-count=1-only residual
`audit_sm120.py` still finds (commit 51ab8ff) to `flash_attn_varlen_func`'s paged branch-read call
(`block_table`/`seqused_k`, `prismyra/kernels/qwen3_moe.py`'s `FlashAttention.forward`). inv3 tried pinning
`num_splits=1` there and found it corrupts the output outright with this argument combination (not merely a
different reduction order) -- rejected without needing a sweep. This script asks the next question before
accepting "unfixable without an upstream flash-attention change": is *any* fixed, non-zero `num_splits` both
(a) correct (matches the kernel's own `num_splits=0` "let it choose" ground truth, which is assumed trustworthy
since nothing upstream disputes it) and (b) invariant (the target's output stops moving when seven companions,
each asking one question too, join the pass)? Monkeypatches `vllm.vllm_flash_attn.flash_attn_varlen_func` for the
duration of this process only -- nothing in `prismyra/` is edited by this script.

Usage: PRISMYRA_MODEL=<repo> python3 diag_num_splits_sweep.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from prismyra import Boolean, Prismyra  # noqa: E402

MODEL = os.environ.get("PRISMYRA_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
GROUP_SIZE = 8

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
ONE_QUESTION = Boolean(id="target", prompt="Is a refund limited to unopened items?")
COMPANION_QUESTION = Boolean(id="companion", prompt="Does this clause mention a time limit?")

FORCED_NUM_SPLITS = {"value": None}  # None = untouched (kernel default, 0)


def install_patch():
    import vllm.vllm_flash_attn as _fa
    original = _fa.flash_attn_varlen_func

    def wrapped(*args, **kwargs):
        if FORCED_NUM_SPLITS["value"] is not None and "block_table" in kwargs and kwargs["block_table"] is not None:
            kwargs = dict(kwargs)
            kwargs["num_splits"] = FORCED_NUM_SPLITS["value"]
        return original(*args, **kwargs)

    _fa.flash_attn_varlen_func = wrapped
    # `FlashAttention.forward` (prismyra/kernels/qwen3_moe.py) does `from vllm.vllm_flash_attn import
    # flash_attn_varlen_func` fresh inside the function on every call, so patching the module attribute above is
    # enough -- no separate patch needed on the `prismyra.kernels.qwen3_moe` module itself.


def run_alone(engine):
    return engine.ask(CONTEXTS[0], [ONE_QUESTION])


def run_group8(engine):
    with engine.open_batch(CONTEXTS[:GROUP_SIZE]) as batch:
        per_doc = [[ONE_QUESTION]] + [[COMPANION_QUESTION] for _ in CONTEXTS[1:GROUP_SIZE]]
        return batch.ask(per_doc)[0]


def main():
    engine = Prismyra(MODEL, paged=True, graphs=False, group=64)
    install_patch()

    print("=== ground truth (num_splits=0, kernel default) ===", flush=True)
    FORCED_NUM_SPLITS["value"] = None
    truth_alone = run_alone(engine)
    truth_group = run_group8(engine)
    truth = {opt: p for opt, p in truth_alone["target"].probabilities.items()}
    print(f"alone={truth} group8={dict(truth_group['target'].probabilities)}", flush=True)

    for n in (2, 3, 4, 8, 16, 32):
        print(f"\n=== num_splits={n} ===", flush=True)
        FORCED_NUM_SPLITS["value"] = n
        try:
            alone = run_alone(engine)
            group = run_group8(engine)
        except Exception as e:  # noqa: BLE001
            print(f"  RAISED: {type(e).__name__}: {e}", flush=True)
            continue
        a = dict(alone["target"].probabilities)
        g = dict(group["target"].probabilities)
        correct_alone = all(abs(a[o] - truth[o]) < 1e-3 for o in truth)
        correct_group = all(abs(g[o] - truth[o]) < 1e-3 for o in truth)
        invariant = all(a[o] == g[o] for o in a)
        print(f"  alone={a}", flush=True)
        print(f"  group8={g}", flush=True)
        print(f"  correct_vs_truth(alone)={correct_alone} correct_vs_truth(group8)={correct_group} "
              f"invariant(alone==group8)={invariant}", flush=True)


if __name__ == "__main__":
    main()
