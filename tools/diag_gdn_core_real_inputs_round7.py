"""inv5, round 7 (continued): the isolated `chunk_gated_delta_rule` repro (`diag_gdn_core_varlen_round7.py`,
synthetic q/k/v/g/beta, real shapes) came back IDENTICAL for every companion count tried, yet the live model
shows `FusedGatedRMSNorm`'s input (the GDN core's real output) differing for the same document, alone vs
group8. That means either the isolated repro is missing something about how the real call is built, or the
REAL q/k/v/g/beta/cu_seqlens/initial_state the live model passes to `chunk_gated_delta_rule` already differ
between scenarios (upstream of the core itself, in `in_proj_a`/`in_proj_b` or `_delta_wrapper`'s own argument
construction) -- `diag_gated_rmsnorm_io_round7.py` only checked the *gate* (z) projection was clean, not q/k/v.

This hooks the real `chunk_gated_delta_rule` call (as installed by `_delta_wrapper`, so this sees exactly what
the live model passes) for layer 0's first branch-phase call, capturing every one of its own arguments sliced
to document 0's own tokens, and compares them bit-for-bit between alone and group8 -- before looking at the
output at all.

Usage: PRISMYRA_MODEL=<repo> python3 diag_gdn_core_real_inputs_round7.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from prismyra import Boolean, Prismyra  # noqa: E402
from prismyra import varlen  # noqa: E402

MODEL = os.environ.get("PRISMYRA_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
GROUP_SIZE = int(os.environ.get("DIAG_GROUP_SIZE", "8"))

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

SCENARIO = {"name": None, "phase": "prefill", "branch_calls": 0}
RECORDS: dict[str, dict] = {}


def install_hook() -> None:
    """Wraps the FINAL installed callable (`modeling.torch_chunk_gated_delta_rule`, what `_install_gated_delta_
    rule` actually leaves on the framework module after its own install-time verification already ran), not the
    raw `chunk_gated_delta_rule` import. Patching the raw import before engine construction broke the engine's
    own install-time self-check (`_delta_wrapper`'s `inspect.signature(kernel)`-based argument filtering saw a
    wrapper's `**kwargs` instead of the real parameter names and silently dropped `use_qk_l2norm_in_kernel`,
    producing a 9e+34 disagreement and a refused install) -- this round's second false start of the same general
    kind as 13.5's dispatcher-registration lesson: a diagnostic that changes what it is trying to observe.
    Call this AFTER constructing the `Prismyra` engine instead.
    """
    import importlib

    modeling = importlib.import_module("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    original = modeling.torch_chunk_gated_delta_rule  # `_delta_wrapper`'s own `call(query, key, value, g=None,
                                                        # beta=None, **kwargs)` -- fixed signature, safe to wrap.

    def wrapped(query, key, value, g=None, beta=None, **kwargs):
        out = original(query, key, value, g=g, beta=beta, **kwargs)
        scenario = SCENARIO["name"]
        if scenario is not None and SCENARIO["phase"] == "branch":
            SCENARIO["branch_calls"] += 1
            if SCENARIO["branch_calls"] == 1:
                boundaries = varlen.current()
                n = boundaries.lengths[0] if boundaries is not None else 1
                cu_seqlens = kwargs.get("cu_seqlens")
                initial_state = kwargs.get("initial_state")
                o = out[0] if isinstance(out, (tuple, list)) else out
                RECORDS[scenario] = {
                    "q": query[:1].detach().to("cpu", torch.float32).clone(),
                    "k": key[:1].detach().to("cpu", torch.float32).clone(),
                    "v": value[:1].detach().to("cpu", torch.float32).clone(),
                    "g": g[:1].detach().to("cpu", torch.float32).clone() if g is not None else None,
                    "beta": beta[:1].detach().to("cpu", torch.float32).clone() if beta is not None else None,
                    "cu_seqlens": cu_seqlens.detach().to("cpu").clone() if cu_seqlens is not None else None,
                    "initial_state_doc0": (
                        initial_state[:1].detach().to("cpu", torch.float32).clone()
                        if initial_state is not None else None
                    ),
                    "out_doc0": o[:1].detach().to("cpu", torch.float32).clone(),
                }
        return out

    wrapped.replaced = getattr(original, "replaced", original)
    modeling.torch_chunk_gated_delta_rule = wrapped


def run_scenario(engine: Prismyra, name: str, group_size: int) -> None:
    SCENARIO["name"] = name
    SCENARIO["phase"] = "prefill"
    SCENARIO["branch_calls"] = 0
    docs = CONTEXTS[:group_size]
    with engine.open_batch(docs) as batch:
        SCENARIO["phase"] = "branch"
        per_doc = [[ONE_QUESTION]] + [[COMPANION_QUESTION] for _ in docs[1:]]
        results = batch.ask(per_doc)
        want = results[0]
    SCENARIO["name"] = None
    print(f"{name}: group_size={group_size}, final p={want['target'].probabilities}", flush=True)


def main():
    engine = Prismyra(MODEL, paged=True, graphs=False, group=64)
    install_hook()  # after construction: patch the final installed callable, not the raw import (see docstring)
    import prismyra.engine as _engine_mod
    assert _engine_mod._BATCH_INVARIANT_REFCOUNT > 0, "dispatcher not registered -- see round-7 13.5 note"
    print(f"model={MODEL} device={torch.cuda.get_device_name(0)}", flush=True)

    run_scenario(engine, "alone", 1)
    run_scenario(engine, f"group{GROUP_SIZE}", GROUP_SIZE)

    a, b = RECORDS["alone"], RECORDS[f"group{GROUP_SIZE}"]
    print(f"\ncu_seqlens: alone={a['cu_seqlens'].tolist() if a['cu_seqlens'] is not None else None}  "
          f"group{GROUP_SIZE}={b['cu_seqlens'].tolist() if b['cu_seqlens'] is not None else None}")
    for key in ("q", "k", "v", "g", "beta", "initial_state_doc0", "out_doc0"):
        ta, tb = a[key], b[key]
        if ta is None or tb is None:
            print(f"{key}: one side is None (ta={ta is not None}, tb={tb is not None})")
            continue
        if ta.shape != tb.shape:
            print(f"{key}: shape differs {tuple(ta.shape)} vs {tuple(tb.shape)}")
            continue
        eq = torch.equal(ta, tb)
        print(f"{key}: {'IDENTICAL' if eq else f'DIFFERS max|diff|={(ta - tb).abs().max().item():.3e}'}")


if __name__ == "__main__":
    main()
